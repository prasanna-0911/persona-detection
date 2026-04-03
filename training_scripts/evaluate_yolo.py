"""
YOLO Model Evaluation Script
==============================
Evaluates a trained YOLO model on:
  1. A YOLO-format validation dataset  → mAP50, mAP50-95, Precision, Recall
  2. A raw video file                  → per-frame detections saved to output video
  3. A directory of test images        → annotated images + summary

Usage:
    # Val dataset evaluation:
    python evaluate_yolo.py \\
        --model   runs/detect/yolov8s_rot2/weights/best.pt \\
        --data    datasets/merged/dataset_rot0.yaml \\
        --output  eval_results/

    # Video inference (nighttime CCTV clip):
    python evaluate_yolo.py \\
        --model  runs/detect/yolov8s_rot2/weights/best.pt \\
        --video  live.mp4 \\
        --output eval_results/

    # Both at once:
    python evaluate_yolo.py \\
        --model  runs/detect/yolov8s_rot2/weights/best.pt \\
        --data   datasets/merged/dataset_rot0.yaml \\
        --video  live.mp4 \\
        --output eval_results/

Requirements:
    pip install ultralytics opencv-python tqdm
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import cv2
from ultralytics import YOLO


# ─────────────────────────────────────────────
#  Val-set evaluation
# ─────────────────────────────────────────────

def evaluate_on_dataset(model: YOLO, data_yaml: str, output_dir: Path,
                        device: str, batch: int, imgsz: int, conf: float) -> dict:
    """Run model.val() and return metrics dict."""
    print("\n[1/2] Evaluating on validation dataset ...")
    print(f"      data   = {data_yaml}")
    print(f"      device = {device}  |  imgsz = {imgsz}  |  conf = {conf}")

    results = model.val(
        data    = data_yaml,
        batch   = batch,
        imgsz   = imgsz,
        conf    = conf,
        device  = device,
        classes = [0],
        save_json = True,
        project = str(output_dir),
        name    = "val",
        exist_ok= True,
        verbose = True,
    )

    try:
        rd = results.results_dict
        metrics = {
            "mAP50":       float(rd.get("metrics/mAP50(B)",    0)),
            "mAP50-95":    float(rd.get("metrics/mAP50-95(B)", 0)),
            "precision":   float(rd.get("metrics/precision(B)",0)),
            "recall":      float(rd.get("metrics/recall(B)",   0)),
        }
    except Exception as e:
        print(f"  ⚠  Could not extract metrics: {e}")
        metrics = {}

    print("\n  === Validation Metrics ===")
    for k, v in metrics.items():
        print(f"    {k:12s}: {v:.4f}")

    return metrics


# ─────────────────────────────────────────────
#  Video inference
# ─────────────────────────────────────────────

def evaluate_on_video(model: YOLO, video_path: str, output_dir: Path,
                      device: str, imgsz: int, conf: float) -> dict:
    """Run inference on a video file; save annotated output video."""
    print(f"\n[2/2] Running inference on video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ⚠  Could not open video: {video_path}")
        return {}

    fps     = cap.get(cv2.CAP_PROP_FPS) or 25
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_dir = output_dir / "video"
    out_dir.mkdir(parents=True, exist_ok=True)
    vid_stem = Path(video_path).stem
    out_path = out_dir / f"{vid_stem}_detections.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    print(f"  Video: {width}x{height} @ {fps:.1f}fps  |  {total} frames")
    print(f"  Output: {out_path}")

    frame_idx     = 0
    total_dets    = 0
    frame_stats   = []
    start         = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(
            frame,
            conf    = conf,
            classes = [0],
            device  = device,
            imgsz   = imgsz,
            verbose = False,
        )

        n_dets = len(results[0].boxes) if results[0].boxes is not None else 0
        total_dets += n_dets

        # Draw detections
        annotated = results[0].plot()
        writer.write(annotated)

        frame_stats.append({"frame": frame_idx, "detections": n_dets})
        frame_idx += 1

        if frame_idx % 100 == 0:
            elapsed = time.time() - start
            fps_proc = frame_idx / max(elapsed, 0.001)
            print(f"  Frame {frame_idx:>5}/{total}  |  "
                  f"{fps_proc:.1f} fps  |  dets this batch: {n_dets}")

    cap.release()
    writer.release()

    elapsed = time.time() - start
    avg_dets = total_dets / max(frame_idx, 1)

    print(f"\n  ✔ Processed {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx/elapsed:.1f} fps)")
    print(f"  Average detections/frame: {avg_dets:.2f}")
    print(f"  Saved annotated video   : {out_path}")

    # Save frame stats
    stats_path = out_dir / f"{vid_stem}_frame_stats.json"
    with open(stats_path, "w") as f:
        json.dump(frame_stats, f, indent=2)

    return {
        "video_path":    str(out_path),
        "total_frames":  frame_idx,
        "total_dets":    total_dets,
        "avg_dets":      round(avg_dets, 3),
        "elapsed_sec":   round(elapsed, 1),
    }


# ─────────────────────────────────────────────
#  Image directory inference
# ─────────────────────────────────────────────

def evaluate_on_images(model: YOLO, img_dir: str, output_dir: Path,
                       device: str, imgsz: int, conf: float) -> dict:
    """Run inference on all images in a directory."""
    print(f"\n[IMG] Running inference on directory: {img_dir}")
    out_img_dir = output_dir / "images"
    out_img_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = sorted(p for p in Path(img_dir).rglob("*") if p.suffix.lower() in exts)
    if not imgs:
        print("  ⚠  No images found.")
        return {}

    results = model.predict(
        source  = img_dir,
        conf    = conf,
        classes = [0],
        device  = device,
        imgsz   = imgsz,
        save    = True,
        project = str(out_img_dir),
        name    = "predict",
        exist_ok= True,
        verbose = False,
    )

    total_dets = sum(len(r.boxes) for r in results if r.boxes is not None)
    print(f"  ✔ Processed {len(imgs)} images  |  Total detections: {total_dets}")

    return {
        "num_images": len(imgs),
        "total_dets": total_dets,
        "avg_dets":   round(total_dets / max(len(imgs), 1), 2),
    }


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned YOLO model on val data, video, or images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model",   required=True,
                        help="Path to trained model weights (e.g. best.pt)")
    parser.add_argument("--output",  required=True,
                        help="Output directory for evaluation results")
    parser.add_argument("--data",    default=None,
                        help="YOLO dataset YAML for validation set evaluation")
    parser.add_argument("--video",   default=None,
                        help="Path to a video file for inference")
    parser.add_argument("--img-dir", default=None,
                        help="Path to image directory for inference")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Detection confidence threshold (default: 0.25)")
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16,
                        help="Batch size for val evaluation (default: 16)")
    parser.add_argument("--device",  default="cuda")
    args = parser.parse_args()

    if not args.data and not args.video and not args.img_dir:
        print("ERROR: provide at least one of --data, --video, or --img-dir")
        sys.exit(1)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  YOLO Evaluation — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Model : {args.model}")
    print(f"{'='*60}")

    model = YOLO(args.model)
    report = {
        "model":     args.model,
        "timestamp": datetime.now().isoformat(),
        "conf":      args.conf,
        "imgsz":     args.imgsz,
    }

    # 1. Val dataset
    if args.data:
        val_metrics = evaluate_on_dataset(
            model, args.data, out, args.device, args.batch, args.imgsz, args.conf
        )
        report["val_metrics"] = val_metrics

    # 2. Video
    if args.video:
        vid_metrics = evaluate_on_video(
            model, args.video, out, args.device, args.imgsz, args.conf
        )
        report["video_metrics"] = vid_metrics

    # 3. Image directory
    if args.img_dir:
        img_metrics = evaluate_on_images(
            model, args.img_dir, out, args.device, args.imgsz, args.conf
        )
        report["image_metrics"] = img_metrics

    # Save full report
    rpt_path = out / "eval_report.json"
    with open(rpt_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n✔ Evaluation report saved → {rpt_path}")


if __name__ == "__main__":
    main()
