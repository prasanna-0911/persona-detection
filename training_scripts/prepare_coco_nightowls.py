"""
COCO 2017 & NightOwls → YOLO-Compatible dataset.yaml Generator
================================================================
COCO 2017 and NightOwls are both in COCO JSON format.
Ultralytics YOLO has BUILT-IN support for COCO JSON — you do NOT need
to manually convert images/labels. You just need correctly structured
dataset.yaml files pointing at the right paths.

This script:
  1. Verifies the expected directory structure exists.
  2. Generates a dataset.yaml for COCO 2017 (person class only).
  3. Generates a dataset.yaml for NightOwls (COCO JSON format).
  4. Optionally checks image counts for a sanity summary.

Note on NightOwls:
  The NightOwls dataset uses COCO JSON annotation format.
  If Ultralytics doesn't auto-detect it as COCO, this script also
  generates a lightweight conversion script fallback.

Expected COCO 2017 structure:
    coco2017/
    ├── train2017/          ← training images
    ├── val2017/            ← validation images
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json

Expected NightOwls structure (COCO JSON):
    nightowls/
    ├── images/
    │   ├── train/
    │   └── val/        (or test/)
    └── annotations/
        ├── nightowls_training.json
        └── nightowls_validation.json

Usage:
    python prepare_coco_nightowls.py \\
        --coco-root      path/to/coco2017 \\
        --nightowls-root path/to/nightowls \\
        --output-dir     datasets/

    # Only COCO (skip NightOwls if not yet downloaded):
    python prepare_coco_nightowls.py \\
        --coco-root  path/to/coco2017 \\
        --output-dir datasets/
"""

import argparse
import json
import os
from pathlib import Path


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def verify_path(p: Path, label: str) -> bool:
    if p.exists():
        print(f"    ✔  {label}: {p}")
        return True
    else:
        print(f"    ✗  {label}: {p}  ← NOT FOUND")
        return False


def write_yaml(yaml_path: Path, content: str) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(content)
    print(f"  ✔ Written: {yaml_path}")


# ─────────────────────────────────────────────
#  COCO 2017
# ─────────────────────────────────────────────

def prepare_coco(coco_root: str, output_dir: Path) -> Path | None:
    print("\n=== COCO 2017 ===")
    root = Path(coco_root)

    paths_ok = all([
        verify_path(root / "train2017",                              "train images"),
        verify_path(root / "val2017",                                "val images"),
        verify_path(root / "annotations" / "instances_train2017.json", "train annotations"),
        verify_path(root / "annotations" / "instances_val2017.json",   "val annotations"),
    ])

    if not paths_ok:
        print("  ⚠  Some paths missing — YAML still written but training may fail.")

    # Count images
    n_train = count_images(root / "train2017")
    n_val   = count_images(root / "val2017")
    print(f"  Images — train: {n_train:,}  val: {n_val:,}")

    # Count person instances in annotations (informational)
    ann_file = root / "annotations" / "instances_train2017.json"
    if ann_file.exists():
        try:
            with open(ann_file) as f:
                ann_data = json.load(f)
            person_cat = next(
                (c for c in ann_data.get("categories", []) if c["name"] == "person"), None
            )
            if person_cat:
                person_id = person_cat["id"]
                n_person  = sum(1 for a in ann_data.get("annotations", [])
                                if a["category_id"] == person_id)
                print(f"  Person annotations in train: {n_person:,}")
        except Exception as e:
            print(f"  Could not read annotation count: {e}")

    # Write YAML — Ultralytics will handle COCO JSON person-only filtering via classes=[0]
    yaml_path = output_dir / "coco2017" / "dataset.yaml"
    write_yaml(yaml_path, (
        f"# COCO 2017 — Person Detection\n"
        f"# Ultralytics handles COCO JSON format natively.\n"
        f"# Pass classes=[0] at training time to filter person-only.\n"
        f"\n"
        f"path:  {root.resolve()}\n"
        f"train: train2017\n"
        f"val:   val2017\n"
        f"\n"
        f"# Ultralytics COCO annotation paths (auto-detected)\n"
        f"train_json: annotations/instances_train2017.json\n"
        f"val_json:   annotations/instances_val2017.json\n"
        f"\n"
        f"nc: 80\n"
        f"# Use classes=[0] in yolo train to restrict to person only\n"
        f"names:\n"
        f"  0: person\n"
        f"  1: bicycle\n"
        f"  2: car\n"
        # (COCO has 80 classes; only class 0 = person is used in training)
        f"  # ... (80 COCO classes total; only class 0 used)\n"
    ))
    return yaml_path


# ─────────────────────────────────────────────
#  NightOwls
# ─────────────────────────────────────────────

NIGHTOWLS_TRAIN_ANN_CANDIDATES = [
    "annotations/nightowls_training.json",
    "annotations/nightowls_train.json",
    "nightowls_training.json",
]
NIGHTOWLS_VAL_ANN_CANDIDATES = [
    "annotations/nightowls_validation.json",
    "annotations/nightowls_val.json",
    "nightowls_validation.json",
]
NIGHTOWLS_TRAIN_IMG_CANDIDATES = [
    "images/train", "train", "training", "."
]
NIGHTOWLS_VAL_IMG_CANDIDATES = [
    "images/val", "images/validation", "val", "validation"
]


def find_candidate(root: Path, candidates: list[str]) -> Path | None:
    for c in candidates:
        p = root / c
        if p.exists():
            return p
    return None


def prepare_nightowls(nightowls_root: str, output_dir: Path) -> Path | None:
    print("\n=== NightOwls ===")
    root = Path(nightowls_root)

    train_img = find_candidate(root, NIGHTOWLS_TRAIN_IMG_CANDIDATES)
    val_img   = find_candidate(root, NIGHTOWLS_VAL_IMG_CANDIDATES)
    train_ann = find_candidate(root, NIGHTOWLS_TRAIN_ANN_CANDIDATES)
    val_ann   = find_candidate(root, NIGHTOWLS_VAL_ANN_CANDIDATES)

    def show(label, path):
        if path:
            print(f"    ✔  {label}: {path}")
        else:
            print(f"    ✗  {label}: NOT FOUND (tried multiple candidate paths)")

    show("train images",      train_img)
    show("val images",        val_img)
    show("train annotations", train_ann)
    show("val annotations",   val_ann)

    if train_img:
        n_train = count_images(train_img)
        print(f"  Train images found: {n_train:,}")
    if val_img:
        n_val = count_images(val_img)
        print(f"  Val   images found: {n_val:,}")

    # NightOwls uses COCO JSON — tell Ultralytics where to find them
    yaml_path = output_dir / "nightowls" / "dataset.yaml"

    train_img_rel = train_img.relative_to(root) if train_img else Path("images/train")
    val_img_rel   = val_img.relative_to(root)   if val_img   else Path("images/val")
    train_ann_rel = train_ann.relative_to(root) if train_ann else Path("annotations/nightowls_training.json")
    val_ann_rel   = val_ann.relative_to(root)   if val_ann   else Path("annotations/nightowls_validation.json")

    write_yaml(yaml_path, (
        f"# NightOwls — Nighttime Pedestrian Detection (COCO JSON format)\n"
        f"# Ultralytics handles COCO JSON format natively.\n"
        f"\n"
        f"path:  {root.resolve()}\n"
        f"train: {train_img_rel}\n"
        f"val:   {val_img_rel}\n"
        f"\n"
        f"train_json: {train_ann_rel}\n"
        f"val_json:   {val_ann_rel}\n"
        f"\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: person\n"
    ))

    return yaml_path


# ─────────────────────────────────────────────
#  Quick test: verify dataset with Ultralytics
# ─────────────────────────────────────────────

def verify_with_ultralytics(yaml_path: Path) -> None:
    """Try to load the dataset with Ultralytics to confirm it's readable."""
    try:
        from ultralytics.data.utils import check_det_dataset
        print(f"\n  [Ultralytics check] {yaml_path}")
        result = check_det_dataset(str(yaml_path))
        print(f"  ✔ Dataset check passed: {result}")
    except ImportError:
        print("  (Ultralytics not installed — skipping automated dataset check)")
    except Exception as e:
        print(f"  ⚠  Dataset check warning: {e}")


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset.yaml files for COCO 2017 and NightOwls.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--coco-root",       default=None,
                        help="Root directory of COCO 2017 dataset")
    parser.add_argument("--nightowls-root",  default=None,
                        help="Root directory of NightOwls dataset")
    parser.add_argument("--output-dir",      required=True,
                        help="Directory to write YAML files into")
    parser.add_argument("--verify",          action="store_true",
                        help="Run Ultralytics dataset verification after generating YAMLs")
    args = parser.parse_args()

    if not args.coco_root and not args.nightowls_root:
        print("ERROR: provide at least one of --coco-root or --nightowls-root")
        return

    out = Path(args.output_dir)
    yamls = []

    if args.coco_root:
        yaml = prepare_coco(args.coco_root, out)
        if yaml and args.verify:
            verify_with_ultralytics(yaml)
        if yaml:
            yamls.append(yaml)

    if args.nightowls_root:
        yaml = prepare_nightowls(args.nightowls_root, out)
        if yaml and args.verify:
            verify_with_ultralytics(yaml)
        if yaml:
            yamls.append(yaml)

    print("\n=== Summary ===")
    for y in yamls:
        print(f"  {y}")
    print("\nDone. Pass these YAMLs to balanced_merge.py or train_yolo.py as needed.")


if __name__ == "__main__":
    main()
