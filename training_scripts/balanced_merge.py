"""
Balanced Epoch-Wise Rotation Merge Script
==========================================
The problem:
  NightOwls alone has ~280k images while all daytime datasets combined are ~82-142k.
  Naively merging would bias the model toward nighttime.

The solution — Epoch-Wise Rotation:
  1. Combine ALL daytime datasets into one pool  (call it "day_pool")
  2. Divide NightOwls into N equal chunks, where
        chunk_size ≈ len(day_pool)
        N          = ceil(len(nightowls) / chunk_size)
  3. Create N rotation manifests (plain .txt files listing image paths).
     Each rotation = day_pool  +  one NightOwls chunk  →  balanced 50/50
  4. Generate one dataset_{rotation}.yaml per rotation.

Training workflow:
  Round 1:  yolo train data=dataset_rot0.yaml  epochs=50
  Round 2:  yolo train data=dataset_rot1.yaml  resume ...
  ...continues until all NightOwls chunks are consumed...

This way, over all rounds:
  - Every NightOwls image is used exactly once
  - Every daytime image is used N times (but that's fine — they are fewer)
  - Balance at every epoch ≈ 50/50

Other "large" datasets (besides NightOwls) are also automatically rotated 
if they exceed the daytime pool size.

Required layout (each dataset must already be in YOLO format):
    {dataset_root}/
    ├── images/
    │   ├── train/  ← all image files
    │   └── val/
    └── labels/
        ├── train/
        └── val/

Usage Examples:
    # Basic — NightOwls + 3 daytime datasets
    python balanced_merge.py \\
        --night   path/to/nightowls_yolo \\
        --day     path/to/coco_yolo \\
                  path/to/crowdhuman_yolo \\
                  path/to/citypersons_yolo \\
        --output  path/to/merged

    # With EuroCity and WiderPerson as extra daytime
    python balanced_merge.py \\
        --night   path/to/nightowls_yolo \\
        --day     path/to/coco_yolo path/to/crowdhuman_yolo \\
                  path/to/citypersons_yolo path/to/eurocity_yolo \\
                  path/to/widerperson_yolo \\
        --output  path/to/merged \\
        --val-from-each 300          # take 300 val images from each dataset

Output:
    merged/
    ├── manifests/
    │   ├── train_rot0.txt   ← image paths for rotation 0 training
    │   ├── train_rot1.txt
    │   └── ...
    ├── val.txt              ← combined validation image paths
    ├── dataset_rot0.yaml
    ├── dataset_rot1.yaml
    └── merge_report.txt     ← human-readable summary
"""

import os
import math
import random
import argparse
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(dataset_root: str, split: str = "train") -> list[Path]:
    """Return sorted list of image paths in {dataset_root}/images/{split}/."""
    img_dir = Path(dataset_root) / "images" / split
    if not img_dir.exists():
        print(f"  ⚠  images/{split}/ not found in {dataset_root}")
        return []
    paths = sorted(
        p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS
    )
    return paths


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """Split lst into chunks of at most chunk_size elements."""
    random.shuffle(lst)          # shuffle so each chunk is diverse
    return [lst[i: i + chunk_size] for i in range(0, len(lst), chunk_size)]


def write_manifest(paths: list[Path], out_file: Path) -> None:
    """Write absolute image paths to a text file, one per line."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        for p in paths:
            f.write(str(p.resolve()) + "\n")


def write_yaml(
    manifest_train: Path,
    manifest_val:   Path,
    yaml_path:      Path,
    rotation_idx:   int,
    total_rotations: int,
) -> None:
    """Write a dataset YAML for Ultralytics YOLO that points at manifest files."""
    yaml_path.write_text(
        f"# Auto-generated — Rotation {rotation_idx + 1}/{total_rotations}\n"
        f"# {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        f"\n"
        f"train: {manifest_train.resolve()}\n"
        f"val:   {manifest_val.resolve()}\n"
        f"\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: person\n"
    )


# ─────────────────────────────────────────────────────────────
#  Main logic
# ─────────────────────────────────────────────────────────────

def build_rotations(
    night_roots:    list[str],
    day_roots:      list[str],
    output_dir:     str,
    val_from_each:  int  = 200,
    seed:           int  = 42,
    cap_day:        int  = 0,   # 0 = no cap; >0 = max daytime images to use
    cap_night:      int  = 0,   # 0 = no cap; >0 = max nighttime images to use
) -> None:
    random.seed(seed)
    out = Path(output_dir)
    manifest_dir = out / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Collect all images ──────────────────────────────────────────
    print("\n=== Scanning datasets ===")

    day_train:  list[Path] = []
    day_val:    list[Path] = []

    for root in day_roots:
        name = Path(root).name
        tr   = collect_images(root, "train")
        va   = collect_images(root, "val")
        print(f"  [DAY ] {name:30s}  train={len(tr):>6,}   val={len(va):>5,}")
        day_train.extend(tr)
        day_val.extend(va)

    # Large (night) datasets are collected separately for chunking
    large_train:  dict[str, list[Path]] = {}   # name → image paths
    large_val:    list[Path]            = []

    for root in night_roots:
        name = Path(root).name
        tr   = collect_images(root, "train")
        va   = collect_images(root, "val")
        print(f"  [NIGHT] {name:30s}  train={len(tr):>6,}   val={len(va):>5,}")
        large_train[name] = tr
        large_val.extend(va)

    if not day_train:
        raise RuntimeError("No daytime training images found. Check --day paths.")

    # ── Apply caps ────────────────────────────────────────────────────
    if cap_day and cap_day < len(day_train):
        random.shuffle(day_train)
        day_train = day_train[:cap_day]
        print(f"  [CAP ] Daytime capped to {cap_day:,} images (--cap-day)")

    if cap_night:
        for name in list(large_train.keys()):
            if cap_night < len(large_train[name]):
                random.shuffle(large_train[name])
                large_train[name] = large_train[name][:cap_night]
                print(f"  [CAP ] Night '{name}' capped to {cap_night:,} images (--cap-night)")

    day_pool_size = len(day_train)
    print(f"\n  Total daytime train images : {day_pool_size:,}")

    # ── 2. Chunk large (night) datasets ───────────────────────────────
    all_chunks: list[list[Path]] = []   # flat list of night chunks

    for name, paths in large_train.items():
        if len(paths) == 0:
            continue
        n_chunks = math.ceil(len(paths) / day_pool_size)
        chunks   = chunk_list(paths, day_pool_size)
        print(f"  NightOwls '{name}': {len(paths):,} images → "
              f"{n_chunks} chunk(s) of ≤{day_pool_size:,}")
        all_chunks.extend(chunks)

    if not all_chunks:
        # Edge case: night dataset is smaller than day pool → single rotation
        print("  Night dataset smaller than day pool — single rotation.")
        all_chunks = [[p for paths in large_train.values() for p in paths]]

    total_rotations = len(all_chunks)
    print(f"\n  Total rotations: {total_rotations}")

    # ── 3. Build validation manifest (shared across all rotations) ────
    # Take up to val_from_each images from each val split
    combined_val = day_val + large_val
    if val_from_each > 0:
        random.shuffle(combined_val)
        combined_val = combined_val[: val_from_each * (len(day_roots) + len(night_roots))]
    val_manifest = out / "val.txt"
    write_manifest(combined_val, val_manifest)
    print(f"\n  Val manifest   : {len(combined_val):,} images → {val_manifest}")

    # ── 4. Build per-rotation train manifests & YAMLs ─────────────────
    print("\n=== Building rotation manifests ===")
    report_lines = [
        f"Balanced Merge Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Daytime pool size : {day_pool_size:,}",
        f"Total rotations   : {total_rotations}",
        f"Validation images : {len(combined_val):,}",
        "",
    ]

    yaml_files = []
    for idx, night_chunk in enumerate(all_chunks):
        # this rotation = all daytime + one night chunk
        rotation_images = day_train + night_chunk
        random.shuffle(rotation_images)

        night_count = len(night_chunk)
        day_count   = len(day_train)
        balance_pct = 100.0 * night_count / max(1, night_count + day_count)

        manifest_path = manifest_dir / f"train_rot{idx}.txt"
        write_manifest(rotation_images, manifest_path)

        yaml_path = out / f"dataset_rot{idx}.yaml"
        write_yaml(manifest_path, val_manifest, yaml_path, idx, total_rotations)
        yaml_files.append(yaml_path)

        line = (f"  Rotation {idx:>2}: "
                f"day={day_count:>6,}  night_chunk={night_count:>6,}  "
                f"total={len(rotation_images):>7,}  night%={balance_pct:.1f}%")
        print(line)
        report_lines.append(line)

    # ── 5. Write training command reference ───────────────────────────
    report_lines += [
        "",
        "=== Suggested Training Commands ===",
        "# Run each rotation sequentially (or on separate machines):",
        "",
    ]
    for idx, yaml_path in enumerate(yaml_files):
        if idx == 0:
            cmd = (f"yolo detect train "
                   f"model=yolov8s.pt "
                   f"data={yaml_path.resolve()} "
                   f"epochs=50 batch=32 imgsz=640 "
                   f"lr0=0.001 lrf=0.01 freeze=10 "
                   f"device=cuda classes=[0] patience=20 save_period=5 "
                   f"project=runs/detect name=yolov8s_rot{idx} exist_ok=True")
        else:
            prev_weights = (f"runs/detect/yolov8s_rot{idx - 1}/weights/best.pt")
            cmd = (f"yolo detect train "
                   f"model={prev_weights} "      # continue from previous rotation
                   f"data={yaml_path.resolve()} "
                   f"epochs=50 batch=32 imgsz=640 "
                   f"lr0=0.0005 lrf=0.01 freeze=0 "  # lower lr for later rotations
                   f"device=cuda classes=[0] patience=20 save_period=5 "
                   f"project=runs/detect name=yolov8s_rot{idx} exist_ok=True")
        report_lines.append(f"# Rotation {idx}")
        report_lines.append(cmd)
        report_lines.append("")

    report_path = out / "merge_report.txt"
    report_path.write_text("\n".join(report_lines))
    print(f"\n✔ Report written → {report_path}")
    print(f"✔ {total_rotations} dataset YAML(s) written to {out}/")
    print("\nAll done. Run each rotation YAML sequentially as shown in merge_report.txt")


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Create balanced epoch-wise rotation manifests for YOLO training.\n"
            "Large (night) datasets are chunked to match daytime pool size."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--night", nargs="+", required=True, metavar="DIR",
        help="Path(s) to large / nighttime YOLO dataset root(s) (e.g. NightOwls).",
    )
    parser.add_argument(
        "--day", nargs="+", required=True, metavar="DIR",
        help="Path(s) to daytime YOLO dataset root(s) "
             "(COCO, CrowdHuman, CityPersons, EuroCity, WiderPerson …)",
    )
    parser.add_argument(
        "--output", required=True, metavar="DIR",
        help="Output directory for manifests and YAML files.",
    )
    parser.add_argument(
        "--val-from-each", type=int, default=200,
        help="Max validation images to take from each dataset (default: 200). "
             "Set 0 to use all available.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible shuffling (default: 42).",
    )
    parser.add_argument(
        "--cap-day", type=int, default=0, metavar="N",
        help=("Cap total daytime images to N. Useful when NightOwls count is "
              "smaller than your daytime pool and you want a 1:1 balance. "
              "Example: --cap-day 20000 caps daytime to 20k to match 20k NightOwls."),
    )
    parser.add_argument(
        "--cap-night", type=int, default=0, metavar="N",
        help=("Cap total nighttime images to N. Mirrors --cap-day for the night side."),
    )
    args = parser.parse_args()

    build_rotations(
        night_roots   = args.night,
        day_roots     = args.day,
        output_dir    = args.output,
        val_from_each = args.val_from_each,
        seed          = args.seed,
        cap_day       = args.cap_day,
        cap_night     = args.cap_night,
    )


if __name__ == "__main__":
    main()
