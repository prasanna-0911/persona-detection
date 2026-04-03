import argparse
import os
import shutil
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Convert COCO JSON to proper YOLO format for balanced_merge.py")
    parser.add_argument("--json", required=True, help="Path to COCO JSON file")
    parser.add_argument("--images-dir", required=True, help="Path to folder containing original images")
    parser.add_argument("--output-dir", required=True, help="Where to save YOLO dataset (e.g., F:\\datasets\\coco_yolo)")
    args = parser.parse_args()

    json_path = Path(args.json)
    img_dir = Path(args.images_dir)
    out_dir = Path(args.output_dir)

    print(f"\n[1/3] Converting JSON: {json_path}")
    
    # 1. Ultralytics convert_coco requires the JSON to be isolated in a folder
    temp_json_dir = out_dir / "_temp_json"
    temp_json_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(json_path, temp_json_dir / json_path.name)

    # 2. Run Ultralytics native conversion
    from ultralytics.data.converter import convert_coco
    try:
        convert_coco(labels_dir=str(temp_json_dir), save_dir=str(out_dir), use_segments=False)
    except Exception as e:
        print(f"Error during conversion: {e}")
        return

    # 3. Rename output labels folder from 'json_name' to 'train'
    gen_labels = out_dir / "labels" / json_path.stem
    final_labels = out_dir / "labels" / "train"
    if gen_labels.exists():
        if final_labels.exists():
            shutil.rmtree(final_labels)
        gen_labels.rename(final_labels)

    # 4. Copy images into standard YOLO structure
    print(f"\n[2/3] Linking images to {out_dir}/images/train ...")
    final_images = out_dir / "images" / "train"
    final_images.mkdir(parents=True, exist_ok=True)
    
    copied = 0
    for img in img_dir.iterdir():
        if img.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            dst = final_images / img.name
            if not dst.exists():
                try:
                    os.link(img, dst) # Fast hardlink (0 bytes extra space)
                except OSError:
                    shutil.copy2(img, dst)
            copied += 1

    shutil.rmtree(temp_json_dir, ignore_errors=True)
    print(f"\n[3/3] DONE! Generated standard YOLO format dataset at: {out_dir}")
    print(f"      Labels generated. Images copied/linked: {copied}")

if __name__ == "__main__":
    main()
