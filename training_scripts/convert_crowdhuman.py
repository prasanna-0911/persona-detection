"""
CrowdHuman → YOLO Format Converter
====================================
CrowdHuman uses .odgt annotation files (one JSON object per line).

Expected input structure:
    crowdhuman/
    ├── Images/                   ← all images (.jpg)
    ├── annotation_train.odgt
    └── annotation_val.odgt

Output structure (YOLO format):
    crowdhuman_yolo/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    └── dataset.yaml

YOLO label format per line: 0 x_center y_center width height  (all normalized 0-1)
Class 0 = person  (using full-body box 'fbox')

Usage:
    python convert_crowdhuman.py \\
        --images-dir  path/to/crowdhuman/Images \\
        --train-ann   path/to/annotation_train.odgt \\
        --val-ann     path/to/annotation_val.odgt \\
        --output-dir  path/to/crowdhuman_yolo
"""

import json
import os
import shutil
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


# ─────────────────────────────────────────────
#  Core converter
# ─────────────────────────────────────────────

def convert_split(
    images_dir: str,
    annotation_file: str,
    output_images_dir: str,
    output_labels_dir: str,
    use_visible_box: bool = False,   # if True, use vbox instead of fbox
    min_height_px: int = 10,         # skip boxes shorter than this (pixels)
) -> int:
    """Convert one split (train or val) from CrowdHuman .odgt to YOLO format.

    Returns:
        Number of successfully converted images.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    converted = 0
    skipped_no_image = 0
    skipped_no_person = 0

    with open(annotation_file, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in tqdm(lines, desc=f"  {Path(annotation_file).name}"):
        data = json.loads(line)
        img_id   = data["ID"]             # e.g. "273275,1a1d3900..."
        gtboxes  = data.get("gtboxes", [])

        # ── find image file ──────────────────────────────────────────────
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = Path(images_dir) / (img_id + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            skipped_no_image += 1
            continue

        # ── read dimensions ──────────────────────────────────────────────
        try:
            with Image.open(img_path) as im:
                img_w, img_h = im.size
        except Exception:
            skipped_no_image += 1
            continue

        # ── parse bounding boxes ─────────────────────────────────────────
        yolo_lines = []
        box_key = "vbox" if use_visible_box else "fbox"

        for box in gtboxes:
            if box.get("tag") != "person":
                continue
            raw = box.get(box_key)
            if raw is None:
                raw = box.get("fbox")          # fallback
            if raw is None:
                continue

            x, y, w, h = raw

            # skip degenerate / too-small boxes
            if w <= 0 or h <= 0 or h < min_height_px:
                continue

            # clip to image boundary
            x = max(0.0, float(x))
            y = max(0.0, float(y))
            w = min(float(w), img_w - x)
            h = min(float(h), img_h - y)
            if w <= 0 or h <= 0:
                continue

            # convert to YOLO (normalised)
            xc = (x + w / 2) / img_w
            yc = (y + h / 2) / img_h
            wn = w / img_w
            hn = h / img_h

            # hard clamp to [0, 1] for safety
            xc, yc, wn, hn = (min(max(v, 0.0), 1.0) for v in (xc, yc, wn, hn))
            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        if not yolo_lines:
            skipped_no_person += 1
            continue

        # ── write outputs ────────────────────────────────────────────────
        out_img_name  = img_id + img_path.suffix
        out_img_path  = Path(output_images_dir) / out_img_name
        out_lbl_path  = Path(output_labels_dir) / (img_id + ".txt")

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
        description="Convert CrowdHuman dataset to YOLO format (person class only)."
    )
    parser.add_argument("--images-dir", required=True,
                        help="Path to CrowdHuman Images/ directory")
    parser.add_argument("--train-ann",  required=True,
                        help="Path to annotation_train.odgt")
    parser.add_argument("--val-ann",    required=True,
                        help="Path to annotation_val.odgt")
    parser.add_argument("--output-dir", required=True,
                        help="Root output directory for the YOLO dataset")
    parser.add_argument("--use-visible-box", action="store_true",
                        help="Use visible-body box (vbox) instead of full-body box (fbox)")
    parser.add_argument("--min-height", type=int, default=10,
                        help="Minimum person height in pixels to keep (default: 10)")
    args = parser.parse_args()

    out = Path(args.output_dir)

    splits = [
        ("train", args.train_ann),
        ("val",   args.val_ann),
    ]

    for split_name, ann_file in splits:
        print(f"\n[{split_name.upper()}]")
        convert_split(
            images_dir          = args.images_dir,
            annotation_file     = ann_file,
            output_images_dir   = str(out / "images" / split_name),
            output_labels_dir   = str(out / "labels" / split_name),
            use_visible_box     = args.use_visible_box,
            min_height_px       = args.min_height,
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
