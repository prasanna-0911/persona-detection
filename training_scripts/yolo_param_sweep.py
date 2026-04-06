import csv
import itertools
from pathlib import Path
from ultralytics import YOLO


def load_completed_tests(csv_path):
    """Read the CSV and return a set of already-completed (conf, iou, imgsz) tuples."""
    completed = set()
    if not csv_path.exists():
        return completed
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (float(row["Conf"]), float(row["IOU"]), int(row["Img_Size"]))
                completed.add(key)
            except (KeyError, ValueError):
                pass  # Skip malformed rows
    return completed


def main():
    # --- Configuration ---
    model_path = "/content/workspace/persona-detection-main/runs/detect/yolov8s_rot0/weights/best.pt"
    data_yaml  = "/content/workspace/persona-detection-main/nightowls_val.yaml"

    # --- Parameters to deviate (Sweep Grid) ---
    conf_values  = [0.15, 0.25, 0.50]
    iou_values   = [0.45, 0.60, 0.70]
    imgsz_values = [640, 960]

    # --- Output: Save DIRECTLY to Google Drive root ---
    # Results are written here permanently after every single test.
    # Even if Colab disconnects mid-sweep, all completed rows are safe in Drive!
    output_csv = Path("/content/drive/MyDrive/yolo_sweep_results.csv")

    # Generate all combinations
    all_combinations = list(itertools.product(conf_values, iou_values, imgsz_values))

    # --- RESUME LOGIC: Check Drive for already-completed tests ---
    completed = load_completed_tests(output_csv)
    remaining = [(c, i, s) for c, i, s in all_combinations if (c, i, s) not in completed]

    print(f"Total planned tests  : {len(all_combinations)}")
    print(f"Already completed    : {len(completed)}  (loaded from Google Drive)")
    print(f"Remaining to run     : {len(remaining)}")

    if not remaining:
        print("\n✅ All tests already completed! Open MyDrive/yolo_sweep_results.csv")
        return

    # Load the model once
    model = YOLO(model_path)

    # Open CSV in APPEND mode on Drive.
    # Every f.flush() call immediately syncs the row to Google Drive permanently.
    file_is_new = not output_csv.exists() or output_csv.stat().st_size == 0
    with open(output_csv, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Only write the header if this is a brand new file
        if file_is_new:
            writer.writerow(["Conf", "IOU", "Img_Size", "Precision", "Recall", "mAP50", "mAP50-95"])
            f.flush()

        for idx, (conf, iou, imgsz) in enumerate(remaining, 1):
            print(f"\n=======================================================")
            print(f" TEST {idx}/{len(remaining)}: Conf={conf} | IOU={iou} | ImgSz={imgsz}")
            print(f"=======================================================")

            try:
                metrics = model.val(
                    data=data_yaml,
                    conf=conf,
                    iou=iou,
                    imgsz=imgsz,
                    split="val",
                    device="cuda",
                    verbose=False
                )

                precision = metrics.results_dict.get('metrics/precision(B)', 0.0)
                recall    = metrics.results_dict.get('metrics/recall(B)', 0.0)
                map50     = metrics.results_dict.get('metrics/mAP50(B)', 0.0)
                map50_95  = metrics.results_dict.get('metrics/mAP50-95(B)', 0.0)

                writer.writerow([conf, iou, imgsz,
                                  round(precision, 4), round(recall, 4),
                                  round(map50, 4),     round(map50_95, 4)])
                f.flush()  # ← Immediately writes this row to Google Drive!
                print(f"   ✅ Saved to Drive → P={precision:.4f}  R={recall:.4f}  mAP50={map50:.4f}")

            except Exception as e:
                print(f"   ❌ FAILED: {e} — skipping this combination and continuing.")

    print(f"\n=======================================================")
    print(f"✅ Sweep complete! Open 'MyDrive/yolo_sweep_results.csv' in Google Drive!")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
