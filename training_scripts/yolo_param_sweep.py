import csv
import itertools
from pathlib import Path
from ultralytics import YOLO

def main():
    # --- Configuration ---
    # Change these paths if necessary
    model_path = r"F:\persona-detection-main\runs\detect\yolov8s_rot0\weights\best.pt"
    # Using the merged rotation YAML so it tests against the validation split
    data_yaml = r"D:\persona_detection\datasets\merged\dataset_rot0.yaml"
    
    # --- Parameters to deviate (Sweep Grid) ---
    # conf: Below 0.25 increases recall (misses fewer people) but lowers precision (more false positives)
    conf_values = [0.10, 0.15, 0.20, 0.25,0.3, 0.35, 0.4, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
    
    # iou: How strictly overlapping boxes are handled. 
    # 0.45 is strict (might delete a real person standing behind another person).
    # 0.70 is loose (might draw 2 boxes on the exact same person).
    iou_values = [0.45, 0.60, 0.70]
    
    # imgsz: Higher resolution makes small people in the dark visible, but runs slower
    imgsz_values = [640, 960]

    # --- Output ---
    output_csv = Path("results/yolo_sweep_results.csv")
    output_csv.parent.mkdir(exist_ok=True)

    # Generate all combinations
    combinations = list(itertools.product(conf_values, iou_values, imgsz_values))
    print(f"Starting extremely thorough YOLO validation sweep ({len(combinations)} total tests)...")

    # Load the model once
    model = YOLO(model_path)

    # Open CSV for writing results
    with open(output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Conf", "IOU", "Img_Size", "Precision", "Recall", "mAP50", "mAP50-95"])

        for conf, iou, imgsz in combinations:
            print(f"\n=======================================================")
            print(f" TESTING: Conf={conf} | IOU={iou} | ImgSz={imgsz}")
            print(f"=======================================================")
            
            # Run validation
            metrics = model.val(
                data=data_yaml,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                split="val",
                device="cuda",
                verbose=False # Keep console slightly cleaner
            )

            # Extract metrics ( Ultralytics stores them in results_dict )
            precision = metrics.results_dict.get('metrics/precision(B)', 0.0)
            recall = metrics.results_dict.get('metrics/recall(B)', 0.0)
            map50 = metrics.results_dict.get('metrics/mAP50(B)', 0.0)
            map50_95 = metrics.results_dict.get('metrics/mAP50-95(B)', 0.0)

            # Write to CSV
            writer.writerow([conf, iou, imgsz, round(precision, 4), round(recall, 4), round(map50, 4), round(map50_95, 4)])
            f.flush() # Save immediately in case it crashes

    print(f"\n=======================================================")
    print(f"✅ Sweep Complete! Open '{output_csv}' to compare all parameters!")
    print(f"=======================================================")

if __name__ == "__main__":
    main()
