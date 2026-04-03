"""
CityPersons (Cityscapes) → YOLO Format Converter
==================================================
CityPersons annotations are distributed as part of Cityscapes.
You need two downloads from https://www.cityscapes-dataset.com :
  1. leftImg8bit_trainvaltest.zip   →  the actual images
  2. gtBbox_cityPersons_trainval.zip →  CityPersons bounding-box annotations

Expected input structure after extraction:
    cityscapes/
    ├── leftImg8bit/
    │   ├── train/  {city}/{city}_{seq}_{frame}_leftImg8bit.png
    │   └── val/
    └── gtBboxCityPersons/
        ├── train/  {city}/{city}_{seq}_{frame}_gtBboxCityPersons.json
        └── val/

Each annotation JSON looks like:
    {
        "imgHeight": 1024,
        "imgWidth":  2048,
        "annotation": {
            "list": [
                {
                    "lbl":      "pedestrian",
                    "bbleft":   x,     # left edge (pixels)
                    "bbtop":    y,     # top  edge (pixels)
                    "bbwidth":  w,
                    "bbheight": h,
                    "occl":     0,     # occlusion flag (0 = visible)
                    ...
                }
            ]
        }
    }

Output structure (YOLO format):
    citypersons_yolo/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── dataset.yaml

Usage:
    python convert_citypersons.py \\
        --cityscapes-root  path/to/cityscapes \\
        --output-dir       path/to/citypersons_yolo
"""

import json
import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


# Labels we treat as "person" (class 0)
PERSON_LABELS = {"pedestrian", "person"}

# Cityscapes images are 2048×1024 by default, but we read from JSON to be safe


# ─────────────────────────────────────────────
#  Core converter
# ─────────────────────────────────────────────

def convert_split(
    cityscapes_root: str,
    split: str,                    # "train" or "val"
    output_images_dir: str,
    output_labels_dir: str,
    skip_occluded: bool = False,   # if True, skip heavily occluded persons
    min_height_px: int = 10,
) -> int:
    """Convert one split of CityPersons to YOLO format.

    Returns:
        Number of successfully converted images.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    ann_root = Path(cityscapes_root) / "gtBboxCityPersons" / split
    img_root = Path(cityscapes_root) / "leftImg8bit"       / split

    if not ann_root.exists():
        print(f"  ⚠ Annotation directory not found: {ann_root}")
        return 0

    ann_files = sorted(ann_root.rglob("*_gtBboxCityPersons.json"))
    if not ann_files:
        print(f"  ⚠ No annotation files found under {ann_root}")
        return 0

    converted         = 0
    skipped_no_image  = 0
    skipped_no_person = 0

    for ann_file in tqdm(ann_files, desc=f"  {split}"):
        with open(ann_file, "r") as f:
            data = json.load(f)

        img_h = data.get("imgHeight", 1024)
        img_w = data.get("imgWidth",  2048)

        persons = data.get("annotation", {}).get("list", [])

        # ── build YOLO label lines ───────────────────────────────────────
        yolo_lines = []
        for p in persons:
            lbl = str(p.get("lbl", "")).lower()
            if lbl not in PERSON_LABELS:
                continue
            if skip_occluded and p.get("occl", 0) == 1:
                continue

            x = float(p.get("bbleft",   0))
            y = float(p.get("bbtop",    0))
            w = float(p.get("bbwidth",  0))
            h = float(p.get("bbheight", 0))

            if w <= 0 or h <= 0 or h < min_height_px:
                continue

            # clip
            x = max(0.0, x)
            y = max(0.0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)
            if w <= 0 or h <= 0:
                continue

            xc = (x + w / 2) / img_w
            yc = (y + h / 2) / img_h
            wn = w / img_w
            hn = h / img_h

            xc, yc, wn, hn = (min(max(v, 0.0), 1.0) for v in (xc, yc, wn, hn))
            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        if not yolo_lines:
            skipped_no_person += 1
            continue

        # ── locate source image ──────────────────────────────────────────
        # ann stem:  "aachen_000000_000000_gtBboxCityPersons"
        # img stem:  "aachen_000000_000000_leftImg8bit"
        city      = ann_file.parent.name
        img_stem  = ann_file.stem.replace("_gtBboxCityPersons", "_leftImg8bit")
        img_path  = img_root / city / (img_stem + ".png")

        if not img_path.exists():
            skipped_no_image += 1
            continue

        # ── write outputs ────────────────────────────────────────────────
        out_img_name = f"{city}_{img_stem}.png"   # prefix city to avoid name clashes
        out_img_path = Path(output_images_dir) / out_img_name
        out_lbl_path = Path(output_labels_dir) / (out_img_name.replace(".png", ".txt"))

        if not out_img_path.exists():
            shutil.copy2(img_path, out_img_path)

        out_lbl_path.write_text("\n".join(yolo_lines))
        converted += 1

    print(f"    ✔ Converted: {converted} | "
          f"Skipped (no image): {skipped_no_image} | "
          f"Skipped (no person): {skipped_no_person}")
    return converted


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert CityPersons (Cityscapes) to YOLO format."
    )
    parser.add_argument("--cityscapes-root", required=True,
                        help="Root of Cityscapes (contains leftImg8bit/ and gtBboxCityPersons/)")
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory for the YOLO dataset")
    parser.add_argument("--skip-occluded", action="store_true",
                        help="Exclude occluded person annotations")
    parser.add_argument("--min-height", type=int, default=10,
                        help="Minimum bounding-box height in pixels (default: 10)")
    args = parser.parse_args()

    out = Path(args.output_dir)

    for split in ["train", "val"]:
        print(f"\n[{split.upper()}]")
        convert_split(
            cityscapes_root   = args.cityscapes_root,
            split             = split,
            output_images_dir = str(out / "images" / split),
            output_labels_dir = str(out / "labels" / split),
            skip_occluded     = args.skip_occluded,
            min_height_px     = args.min_height,
        )

    # ── write dataset.yaml ───────────────────────────────────────────────
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
