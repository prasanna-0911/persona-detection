"""
WiderPerson → YOLO Format Converter
=====================================
WiderPerson is a large-scale pedestrian detection dataset.
Download from: http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/
Also available on Kaggle (search "WiderPerson").

Expected input structure after extraction:
    WiderPerson/
    ├── Images/
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   └── ...
    ├── Annotations/
    │   ├── 000001.jpg.txt
    │   ├── 000002.jpg.txt
    │   └── ...
    ├── train.txt          ← list of image filenames for train split
    └── val.txt            ← list of image filenames for val split

Each annotation file format:
    <num_annotations>
    <label> <x1> <y1> <x2> <y2>
    ...

Label meanings:
    1 = pedestrians              (full-body, clearly visible)
    2 = riders                   (cyclists, motorcyclists)
    3 = partially-visible persons
    4 = ignore regions           ← we SKIP these
    5 = crowd                    ← we SKIP these (no precise bbox)

By default this script converts labels 1 (pedestrians) only.
Use --include-riders and --include-partial to widen coverage.

Output (YOLO format):
    widerperson_yolo/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── dataset.yaml

Usage:
    # Pedestrians only (recommended):
    python convert_widerperson.py \\
        --wider-root  path/to/WiderPerson \\
        --output-dir  path/to/widerperson_yolo

    # Include riders and partially visible:
    python convert_widerperson.py \\
        --wider-root  path/to/WiderPerson \\
        --output-dir  path/to/widerperson_yolo \\
        --include-riders --include-partial
"""

import os
import shutil
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# WiderPerson label IDs we accept (mapped to class 0 = person in YOLO)
LABEL_PEDESTRIAN = 1
LABEL_RIDER      = 2
LABEL_PARTIAL    = 3
LABEL_IGNORE     = 4    # always excluded
LABEL_CROWD      = 5    # always excluded


IMAGE_EXTS = [".jpg", ".jpeg", ".png"]


# ─────────────────────────────────────────────
#  Core converter
# ─────────────────────────────────────────────

def get_image_size(img_path: Path):
    """Return (width, height) of image."""
    with Image.open(img_path) as im:
        return im.size     # (W, H)


def convert_split(
    wider_root:      str,
    split:           str,         # "train" or "val"
    out_img_dir:     str,
    out_lbl_dir:     str,
    include_riders:  bool = False,
    include_partial: bool = False,
    min_height_px:   int  = 10,
) -> int:
    """Convert one split of WiderPerson to YOLO format."""
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    wider       = Path(wider_root)
    split_file  = wider / f"{split}.txt"
    img_dir     = wider / "Images"
    ann_dir     = wider / "Annotations"

    if not split_file.exists():
        print(f"  ⚠  Split file not found: {split_file}")
        return 0

    # Build accepted label set
    accepted_labels = {LABEL_PEDESTRIAN}
    if include_riders:
        accepted_labels.add(LABEL_RIDER)
    if include_partial:
        accepted_labels.add(LABEL_PARTIAL)

    # Read split file
    with open(split_file, "r") as f:
        image_names = [l.strip() for l in f if l.strip()]

    converted = 0
    skipped   = 0

    for img_name in tqdm(image_names, desc=f"  WiderPerson {split}"):
        # ── find source image ────────────────────────────────────────────
        img_path = img_dir / img_name
        if not img_path.exists():
            # try without extension, then try common exts
            stem = Path(img_name).stem
            img_path = None
            for ext in IMAGE_EXTS:
                candidate = img_dir / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    img_name = stem + ext
                    break
            if img_path is None:
                skipped += 1
                continue

        # ── find annotation file ─────────────────────────────────────────
        ann_path = ann_dir / (img_name + ".txt")
        if not ann_path.exists():
            # try stem-only
            ann_path = ann_dir / (Path(img_name).stem + ".txt")
        if not ann_path.exists():
            skipped += 1
            continue

        # ── image dimensions ─────────────────────────────────────────────
        try:
            img_w, img_h = get_image_size(img_path)
        except Exception:
            skipped += 1
            continue

        # ── parse annotation ─────────────────────────────────────────────
        with open(ann_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        if not lines:
            skipped += 1
            continue

        try:
            n_ann = int(lines[0])
        except ValueError:
            skipped += 1
            continue

        yolo_lines = []
        for line in lines[1: n_ann + 1]:
            parts = line.split()
            if len(parts) < 5:
                continue

            label = int(parts[0])
            if label not in accepted_labels:
                continue            # skip ignore, crowd, unwanted classes

            x1, y1, x2, y2 = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0 or h < min_height_px:
                continue

            # clip to image boundary
            x1 = max(0.0, x1)
            y1 = max(0.0, y1)
            w  = min(w, img_w - x1)
            h  = min(h, img_h - y1)
            if w <= 0 or h <= 0:
                continue

            xc = (x1 + w / 2) / img_w
            yc = (y1 + h / 2) / img_h
            wn = w / img_w
            hn = h / img_h

            xc, yc, wn, hn = (min(max(v, 0.0), 1.0) for v in (xc, yc, wn, hn))
            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        if not yolo_lines:
            skipped += 1
            continue

        # ── write outputs ────────────────────────────────────────────────
        out_img = Path(out_img_dir) / Path(img_name).name
        out_lbl = Path(out_lbl_dir) / (Path(img_name).stem + ".txt")

        if not out_img.exists():
            shutil.copy2(img_path, out_img)
        out_lbl.write_text("\n".join(yolo_lines))
        converted += 1

    print(f"    ✔ Converted: {converted}  |  Skipped: {skipped}")
    return converted


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert WiderPerson dataset to YOLO format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--wider-root",  required=True,
                        help="Root of WiderPerson (contains Images/, Annotations/, train.txt, val.txt)")
    parser.add_argument("--output-dir",  required=True,
                        help="Root output directory for the YOLO dataset")
    parser.add_argument("--include-riders",  action="store_true",
                        help="Also include label 2 (riders) as class 0")
    parser.add_argument("--include-partial", action="store_true",
                        help="Also include label 3 (partially-visible persons) as class 0")
    parser.add_argument("--min-height", type=int, default=10,
                        help="Minimum bounding-box height in pixels (default: 10)")
    args = parser.parse_args()

    out = Path(args.output_dir)

    for split in ["train", "val"]:
        print(f"\n[{split.upper()}]")
        convert_split(
            wider_root      = args.wider_root,
            split           = split,
            out_img_dir     = str(out / "images" / split),
            out_lbl_dir     = str(out / "labels" / split),
            include_riders  = args.include_riders,
            include_partial = args.include_partial,
            min_height_px   = args.min_height,
        )

    # Write dataset.yaml
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: person\n"
    )
    print(f"\n✔ dataset.yaml written → {yaml_path}")
    print("Done.")


if __name__ == "__main__":
    main()
