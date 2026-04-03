"""
EuroCity Persons (ECP) → YOLO Format Converter
================================================
EuroCity Persons is a pedestrian detection dataset collected across
European cities. It requires registration at:
    https://eurocity-dataset.bitbucket.io

Expected input structure after extraction:
    ECP/
    ├── day/
    │   ├── img/
    │   │   ├── train/  {city}/  *.png
    │   │   └── val/    {city}/  *.png
    │   └── labels/
    │       ├── train/  {city}/  *.json
    │       └── val/    {city}/  *.json
    └── night/                          ← same structure
        ├── img/   ...
        └── labels/ ...

Each annotation JSON file is structured as:
    {
        "children": [
            {
                "identity": "pedestrian",        ← or "person-group-far-away", "rider" ...
                "pos":      [x0, y0, x1, y1],    ← top-left & bottom-right in PIXELS
                "occluded": 0.0,                 ← 0.0–1.0 occlusion ratio
                "tags":     [...],
                ...
            }
        ],
        "identity": "frame",
        "imageheight": H,
        "imagewidth":  W
    }

Output (YOLO format):
    eurocity_yolo/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── dataset.yaml

Usage:
    # Convert only daytime images:
    python convert_eurocity.py \\
        --ecp-root  path/to/ECP \\
        --output-dir path/to/eurocity_yolo \\
        --modes day

    # Convert both day and night:
    python convert_eurocity.py \\
        --ecp-root  path/to/ECP \\
        --output-dir path/to/eurocity_yolo \\
        --modes day night

    # Skip occluded persons (>50% occluded):
    python convert_eurocity.py \\
        --ecp-root path/to/ECP --output-dir path/to/eurocity_yolo \\
        --max-occluded 0.5
"""

import json
import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


# ECP identity labels that map to YOLO class 0 (person)
PERSON_IDENTITIES = {"pedestrian", "person"}

# These identities exist in ECP but we exclude by default
# (riders, groups, animals, vehicles …)
# Override with --include-riders if you want to keep riders too.
RIDER_IDENTITIES  = {"rider", "bicyclist", "motorcyclist"}


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def find_image(img_dir: Path, stem: str) -> Path | None:
    """Find an image file matching stem in img_dir, any extension."""
    for ext in IMAGE_EXTS:
        p = img_dir / (stem + ext)
        if p.exists():
            return p
    return None


def parse_annotation(ann_path: Path, include_riders: bool, max_occluded: float):
    """Parse one ECP annotation JSON and return (img_w, img_h, yolo_lines)."""
    with open(ann_path, "r") as f:
        data = json.load(f)

    img_w = data.get("imagewidth",  None)
    img_h = data.get("imageheight", None)

    yolo_lines = []
    for child in data.get("children", []):
        identity = child.get("identity", "").lower()

        is_person = identity in PERSON_IDENTITIES
        is_rider  = identity in RIDER_IDENTITIES
        if not is_person and not (include_riders and is_rider):
            continue

        occ = float(child.get("occluded", 0.0))
        if occ > max_occluded:
            continue

        pos = child.get("pos")
        if pos is None or len(pos) < 4:
            continue

        x0, y0, x1, y1 = pos
        w = x1 - x0
        h = y1 - y0

        if w <= 0 or h <= 0:
            continue

        # if JSON doesn't have image dimensions, skip normalisation
        if img_w is None or img_h is None:
            continue

        x0 = max(0.0, float(x0))
        y0 = max(0.0, float(y0))
        w  = min(float(w), img_w - x0)
        h  = min(float(h), img_h - y0)
        if w <= 0 or h <= 0:
            continue

        xc = (x0 + w / 2) / img_w
        yc = (y0 + h / 2) / img_h
        wn = w / img_w
        hn = h / img_h

        xc, yc, wn, hn = (min(max(v, 0.0), 1.0) for v in (xc, yc, wn, hn))
        yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

    return img_w, img_h, yolo_lines


# ─────────────────────────────────────────────
#  Core converter
# ─────────────────────────────────────────────

def convert_split(
    ecp_root:        str,
    mode:            str,   # "day" or "night"
    split:           str,   # "train" or "val"
    out_img_dir:     str,
    out_lbl_dir:     str,
    include_riders:  bool  = False,
    max_occluded:    float = 1.0,
    min_height_px:   int   = 10,
) -> int:
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    lbl_root = Path(ecp_root) / mode / "labels" / split
    img_root = Path(ecp_root) / mode / "img"    / split

    if not lbl_root.exists():
        print(f"  ⚠  Not found: {lbl_root}")
        return 0

    ann_files = sorted(lbl_root.rglob("*.json"))
    if not ann_files:
        print(f"  ⚠  No JSON files found under {lbl_root}")
        return 0

    converted = 0
    skipped   = 0

    for ann_file in tqdm(ann_files, desc=f"  ECP {mode}/{split}"):
        city = ann_file.parent.name
        stem = ann_file.stem                # e.g. "berlin_000001_000019_leftImg8bit"

        img_w, img_h, yolo_lines = parse_annotation(ann_file, include_riders, max_occluded)

        if not yolo_lines:
            skipped += 1
            continue

        # Find source image (city sub-directory)
        src_img = find_image(img_root / city, stem)
        if src_img is None:
            src_img = find_image(img_root, stem)   # fallback: flat layout
        if src_img is None:
            skipped += 1
            continue

        # Filter by min height in pixels (approximate: use img_h * hn)
        if min_height_px > 0 and img_h is not None:
            yolo_lines = [
                l for l in yolo_lines
                if float(l.split()[4]) * img_h >= min_height_px
            ]
        if not yolo_lines:
            skipped += 1
            continue

        # Unique output name: {mode}_{city}_{stem}
        out_name = f"{mode}_{city}_{stem}"
        out_img  = Path(out_img_dir) / (out_name + src_img.suffix)
        out_lbl  = Path(out_lbl_dir) / (out_name + ".txt")

        if not out_img.exists():
            shutil.copy2(src_img, out_img)
        out_lbl.write_text("\n".join(yolo_lines))
        converted += 1

    print(f"    ✔ Converted: {converted}  |  Skipped: {skipped}")
    return converted


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert EuroCity Persons (ECP) to YOLO format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--ecp-root",   required=True,
                        help="Root of ECP dataset (contains day/ and/or night/)")
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory for the YOLO dataset")
    parser.add_argument("--modes", nargs="+", default=["day"],
                        choices=["day", "night"],
                        help="Which lighting modes to convert (default: day)")
    parser.add_argument("--include-riders", action="store_true",
                        help="Also include rider/bicyclist/motorcyclist as class 0")
    parser.add_argument("--max-occluded", type=float, default=1.0,
                        help="Skip persons with occlusion ratio > this value (0–1). "
                             "E.g. 0.5 skips persons more than 50%% occluded (default: 1.0 = keep all)")
    parser.add_argument("--min-height", type=int, default=10,
                        help="Minimum bounding-box height in pixels (default: 10)")
    args = parser.parse_args()

    out = Path(args.output_dir)

    for mode in args.modes:
        for split in ["train", "val"]:
            print(f"\n[{mode.upper()} / {split.upper()}]")
            convert_split(
                ecp_root       = args.ecp_root,
                mode           = mode,
                split          = split,
                out_img_dir    = str(out / "images" / split),
                out_lbl_dir    = str(out / "labels" / split),
                include_riders = args.include_riders,
                max_occluded   = args.max_occluded,
                min_height_px  = args.min_height,
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
