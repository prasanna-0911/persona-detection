"""
Re-ID Dataset Verification Script
====================================
Run this on the TRAINING PC after extracting the Re-ID dataset zips
to confirm all three datasets are correctly structured and readable.

Usage:
    python phase2_reid/verify_reid_datasets.py \\
        --market  path/to/Market-1501-v15.09.15 \\
        --msmt17  path/to/MSMT17_V1 \\
        --last    path/to/LaST
"""

import argparse
import os
import sys
from pathlib import Path


def check(condition: bool, msg_ok: str, msg_fail: str) -> bool:
    if condition:
        print(f"    ✔ {msg_ok}")
    else:
        print(f"    ✗ {msg_fail}")
    return condition


def count_images(directory: str) -> int:
    exts = {'.jpg', '.jpeg', '.png', '.bmp'}
    total = 0
    for root, _, files in os.walk(directory):
        total += sum(1 for f in files
                     if os.path.splitext(f)[1].lower() in exts)
    return total


# ─────────────────────────────────────────────
#  Market-1501
# ─────────────────────────────────────────────

def verify_market1501(root: str) -> bool:
    print("\n=== Market-1501 ===")
    if not root:
        print("  (skipped — not provided)")
        return True

    ok = True
    ok &= check(os.path.isdir(root), f"Root exists: {root}",
                f"Root NOT found: {root}")

    for subdir in ['bounding_box_train', 'bounding_box_test', 'query']:
        path = os.path.join(root, subdir)
        exists = os.path.isdir(path)
        ok &= check(exists, f"{subdir}/ found", f"{subdir}/ MISSING")
        if exists:
            n = count_images(path)
            print(f"       → {n:,} images")

    return ok


# ─────────────────────────────────────────────
#  MSMT17
# ─────────────────────────────────────────────

def verify_msmt17(root: str) -> bool:
    print("\n=== MSMT17 ===")
    if not root:
        print("  (skipped — not provided)")
        return True

    ok = True
    ok &= check(os.path.isdir(root), f"Root exists: {root}",
                f"Root NOT found: {root}")

    for subdir in ['train', 'test']:
        path = os.path.join(root, subdir)
        exists = os.path.isdir(path)
        ok &= check(exists, f"{subdir}/ found", f"{subdir}/ MISSING")
        if exists:
            n = count_images(path)
            print(f"       → {n:,} images")

    for list_file in ['list_train.txt', 'list_val.txt',
                      'list_query.txt', 'list_gallery.txt']:
        path = os.path.join(root, list_file)
        ok &= check(os.path.exists(path), f"{list_file} found",
                    f"{list_file} MISSING")
        if os.path.exists(path):
            with open(path) as f:
                n = sum(1 for _ in f)
            print(f"       → {n:,} entries")

    return ok


# ─────────────────────────────────────────────
#  LaST
# ─────────────────────────────────────────────

def verify_last(root: str) -> bool:
    print("\n=== LaST ===")
    if not root:
        print("  (skipped — not provided)")
        return True

    ok = True
    ok &= check(os.path.isdir(root), f"Root exists: {root}",
                f"Root NOT found: {root}")

    for subdir in ['train', 'val', 'test']:
        path = os.path.join(root, subdir)
        exists = os.path.isdir(path)
        ok &= check(exists, f"{subdir}/ found", f"{subdir}/ MISSING")
        if exists:
            n = count_images(path)
            print(f"       → {n:,} images")

    # Check val has query and gallery
    for subdir in ['val/query', 'val/gallery']:
        path = os.path.join(root, *subdir.split('/'))
        ok &= check(os.path.isdir(path), f"{subdir}/ found",
                    f"{subdir}/ MISSING")

    # Count identities in train
    train_dir = os.path.join(root, 'train')
    if os.path.isdir(train_dir):
        identity_folders = [
            d for d in os.listdir(train_dir)
            if os.path.isdir(os.path.join(train_dir, d))
        ]
        print(f"    ✔ Train identities: {len(identity_folders):,} "
              f"(incl. '000000' interference if present)")

    return ok


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify Re-ID dataset structure on the training PC."
    )
    parser.add_argument('--market',  default=None,
                        help='Path to Market-1501-v15.09.15 root')
    parser.add_argument('--msmt17',  default=None,
                        help='Path to MSMT17_V1 root')
    parser.add_argument('--last',    default=None,
                        help='Path to LaST root')
    args = parser.parse_args()

    if not any([args.market, args.msmt17, args.last]):
        print("ERROR: Provide at least one dataset path "
              "(--market, --msmt17, --last).")
        sys.exit(1)

    results = {
        'Market-1501': verify_market1501(args.market),
        'MSMT17':      verify_msmt17(args.msmt17),
        'LaST':        verify_last(args.last),
    }

    print("\n" + "=" * 50)
    print("  VERIFICATION SUMMARY")
    print("=" * 50)
    all_ok = True
    for name, status in results.items():
        icon = "✔" if status else "✗"
        print(f"  {icon}  {name}: {'OK' if status else 'ISSUES FOUND'}")
        if not status:
            all_ok = False

    if all_ok:
        print("\n  All datasets verified successfully!")
        print("  You can now run: python phase2_reid/train_reid_multi.py ...")
    else:
        print("\n  ⚠  Fix the issues above before running training.")
    print("=" * 50)


if __name__ == '__main__':
    main()
