"""
YOLO Fine-Tuning — Rotation-Based Training Runner
===================================================
This script drives the full rotation-based training pipeline produced by
balanced_merge.py.  It:

  1. Discovers all  dataset_rot*.yaml  files in the merged output directory.
  2. Trains each rotation sequentially:
       - Rotation 0  →  starts from  yolov8s.pt  (pretrained on COCO)
       - Rotation N  →  starts from  runs/detect/yolov8s_rotN-1/weights/best.pt
  3. Uses a lower learning rate for later rotations (fine-tune on top of
     what we already learned).
  4. Writes a  training_log.json  with per-rotation metrics.

Usage:
    # Standard run (auto-discovers all rotation YAMLs):
    python train_yolo.py \\
        --merged-dir  datasets/merged \\
        --project     runs/detect \\
        --device      cuda          # or 0,1 for multi-GPU

    # Resume from a specific rotation (if training was interrupted):
    python train_yolo.py \\
        --merged-dir  datasets/merged \\
        --start-from  2             # skip rot0, rot1 — start at rot2
        --project     runs/detect

    # Single rotation (useful for quick testing):
    python train_yolo.py \\
        --merged-dir  datasets/merged \\
        --only-rotation 0

Requirements:
    pip install ultralytics tqdm
"""

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────
#  Hyper-parameter defaults
# ─────────────────────────────────────────────────────────────

DEFAULT_BASE_MODEL  = "yolov8s.pt"    # starting weights for rotation 0
DEFAULT_EPOCHS      = 50              # per rotation
DEFAULT_BATCH       = 32
DEFAULT_IMGSZ       = 640
DEFAULT_LR0_FIRST   = 0.001           # learning rate — first rotation
DEFAULT_LR0_LATER   = 0.0005          # learning rate — subsequent rotations
DEFAULT_LRF         = 0.01            # final lr = lr0 × lrf
DEFAULT_FREEZE      = 10              # freeze first N backbone layers in rot 0
DEFAULT_PATIENCE    = 20              # early stopping patience
DEFAULT_SAVE_PERIOD = 5               # save checkpoint every N epochs
DEFAULT_WORKERS     = 8


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def discover_rotation_yamls(merged_dir: str) -> list[Path]:
    """Return sorted list of dataset_rot*.yaml files."""
    pattern = os.path.join(merged_dir, "dataset_rot*.yaml")
    files   = sorted(glob.glob(pattern), key=lambda p: _rot_index(Path(p)))
    return [Path(f) for f in files]


def _rot_index(yp: Path) -> int:
    """Extract rotation index from filename: dataset_rot3.yaml → 3"""
    stem = yp.stem               # "dataset_rot3"
    try:
        return int(stem.replace("dataset_rot", ""))
    except ValueError:
        return 9999


def find_best_weights(project: str, run_name: str) -> Path | None:
    """Return best.pt path for a completed run, or None."""
    p = Path(project) / run_name / "weights" / "best.pt"
    return p if p.exists() else None


def read_results(project: str, run_name: str) -> dict:
    """Try to read the results.json produced by Ultralytics."""
    result_json = Path(project) / run_name / "results.json"
    result_csv  = Path(project) / run_name / "results.csv"

    if result_json.exists():
        with open(result_json) as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                pass

    # Fallback: read last line of CSV
    if result_csv.exists():
        with open(result_csv) as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) >= 2:
            headers = [h.strip() for h in lines[0].split(",")]
            values  = [v.strip() for v in lines[-1].split(",")]
            return dict(zip(headers, values))

    return {}


# ─────────────────────────────────────────────────────────────
#  Core trainer
# ─────────────────────────────────────────────────────────────

def train_rotation(
    yaml_path:    Path,
    weights:      str,
    run_name:     str,
    project:      str,
    rotation_idx: int,
    args,                # parsed CLI args
) -> dict:
    """Train one rotation. Returns a summary dict."""

    is_first = (rotation_idx == 0)
    # Use CLI --lr0 if provided, otherwise fall back to the built-in defaults
    lr0      = (args.lr0 if (is_first and args.lr0 is not None) else
                DEFAULT_LR0_FIRST if is_first else DEFAULT_LR0_LATER)
    freeze   = args.freeze if is_first else 0   # only freeze backbone on first rotation

    print(f"\n{'='*60}")
    print(f"  ROTATION {rotation_idx}  |  {run_name}")
    print(f"  Weights : {weights}")
    print(f"  Data    : {yaml_path}")
    print(f"  lr0={lr0}  freeze={freeze}  epochs={args.epochs}")
    print(f"{'='*60}\n")

    model = YOLO(weights)
    start = time.time()

    train_args = dict(
        data        = str(yaml_path),
        epochs      = args.epochs,
        batch       = args.batch,
        imgsz       = args.imgsz,
        lr0         = lr0,
        lrf         = args.lrf,
        freeze      = freeze,
        device      = args.device,
        classes     = [0],             # person only
        patience    = args.patience,
        save_period = args.save_period,
        workers     = args.workers,
        project     = project,
        name        = run_name,
        exist_ok    = True,
        verbose     = True,
    )

    results = model.train(**train_args)
    elapsed = time.time() - start

    # Collect metrics from the Results object
    try:
        map50    = float(results.results_dict.get("metrics/mAP50(B)",   0.0))
        map5095  = float(results.results_dict.get("metrics/mAP50-95(B)",0.0))
        precision= float(results.results_dict.get("metrics/precision(B)",0.0))
        recall   = float(results.results_dict.get("metrics/recall(B)",  0.0))
    except Exception:
        map50 = map5095 = precision = recall = None

    best_wts = find_best_weights(project, run_name)

    summary = {
        "rotation":    rotation_idx,
        "run_name":    run_name,
        "yaml":        str(yaml_path),
        "weights_in":  weights,
        "weights_out": str(best_wts) if best_wts else None,
        "elapsed_sec": round(elapsed, 1),
        "mAP50":       map50,
        "mAP50-95":    map5095,
        "precision":   precision,
        "recall":      recall,
        "timestamp":   datetime.now().isoformat(),
    }

    print(f"\n  ✔ Rotation {rotation_idx} done in {elapsed/3600:.1f}h")
    if map50 is not None:
        print(f"    mAP50={map50:.4f}  mAP50-95={map5095:.4f}  "
              f"P={precision:.4f}  R={recall:.4f}")
    if best_wts:
        print(f"    Best weights → {best_wts}")

    return summary


# ─────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Rotation-based YOLO fine-tuning on balanced day+night data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Required
    parser.add_argument("--merged-dir", required=True,
                        help="Output directory from balanced_merge.py (contains dataset_rot*.yaml)")

    # Training configuration
    parser.add_argument("--base-model",  default=DEFAULT_BASE_MODEL,
                        help=f"Starting weights for rotation 0 (default: {DEFAULT_BASE_MODEL})")
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch",       type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--imgsz",       type=int,   default=DEFAULT_IMGSZ)
    parser.add_argument("--lrf",         type=float, default=DEFAULT_LRF)
    parser.add_argument("--freeze",      type=int,   default=DEFAULT_FREEZE,
                        help="Backbone layers to freeze in rotation 0 (default: 10)")
    parser.add_argument("--lr0",         type=float, default=None,
                        help="Override lr for rotation 0 (default: 0.001). Later rotations always use 0.0005.")

    parser.add_argument("--save-period", type=int,   default=DEFAULT_SAVE_PERIOD)
    parser.add_argument("--workers",     type=int,   default=DEFAULT_WORKERS)
    parser.add_argument("--device",      default="cuda",
                        help="Device: 'cuda', '0', '0,1', 'cpu' (default: cuda)")

    # Run control
    parser.add_argument("--project",      default="runs/detect",
                        help="Ultralytics project directory (default: runs/detect)")
    parser.add_argument("--name-prefix",  default="yolov8s_rot",
                        help="Run name prefix; rotation index appended (default: yolov8s_rot)")
    parser.add_argument("--start-from",   type=int, default=0,
                        help="Skip rotations before this index (for resuming interrupted runs)")
    parser.add_argument("--only-rotation", type=int, default=None,
                        help="Train only this single rotation index")

    args = parser.parse_args()

    # ── discover rotation YAMLs ──────────────────────────────────────────
    yaml_files = discover_rotation_yamls(args.merged_dir)
    if not yaml_files:
        print(f"ERROR: No dataset_rot*.yaml files found in: {args.merged_dir}")
        sys.exit(1)

    print(f"Found {len(yaml_files)} rotation YAML(s) in {args.merged_dir}")

    # Apply filters
    if args.only_rotation is not None:
        yaml_files = [y for y in yaml_files if _rot_index(y) == args.only_rotation]
        if not yaml_files:
            print(f"ERROR: rotation {args.only_rotation} not found.")
            sys.exit(1)
    elif args.start_from > 0:
        yaml_files = [y for y in yaml_files if _rot_index(y) >= args.start_from]
        print(f"Skipping rotations before {args.start_from}. "
              f"Remaining: {len(yaml_files)}")

    # ── training loop ────────────────────────────────────────────────────
    log = []
    current_weights = args.base_model

    for yaml_path in yaml_files:
        rot_idx  = _rot_index(yaml_path)
        run_name = f"{args.name_prefix}{rot_idx}"

        # If resuming: check if this rotation already has weights
        if args.start_from > 0 and rot_idx == args.start_from:
            # Look for previous rotation's best.pt to chain from
            prev_name = f"{args.name_prefix}{rot_idx - 1}"
            prev_wts  = find_best_weights(args.project, prev_name)
            if prev_wts:
                current_weights = str(prev_wts)
                print(f"Resuming: chaining from previous rotation → {current_weights}")

        summary = train_rotation(
            yaml_path    = yaml_path,
            weights      = current_weights,
            run_name     = run_name,
            project      = args.project,
            rotation_idx = rot_idx,
            args         = args,
        )
        log.append(summary)

        # Chain: next rotation starts from this rotation's best.pt
        if summary["weights_out"]:
            current_weights = summary["weights_out"]
        else:
            print(f"WARNING: best.pt not found for rotation {rot_idx}. "
                  f"Using same weights for next rotation.")

    # ── save training log ────────────────────────────────────────────────
    log_path = Path(args.project) / "training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n✔ Training log saved → {log_path}")

    # ── final summary ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  TRAINING COMPLETE — Summary")
    print("="*60)
    for s in log:
        map50_str = f"{s['mAP50']:.4f}" if s['mAP50'] else "N/A"
        print(f"  Rotation {s['rotation']:>2}  |  mAP50={map50_str}  |  "
              f"time={s['elapsed_sec']/3600:.1f}h  |  {s['run_name']}")
    print(f"\n  Final weights: {current_weights}")
    print(f"  Log:           {log_path}")


if __name__ == "__main__":
    main()
