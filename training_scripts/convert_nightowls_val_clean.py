import json
from pathlib import Path
import argparse
import shutil

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True, help="Path to nightowls_validation.json")
    parser.add_argument("--images-dir", required=True, help="Path to the downloaded images folder")
    parser.add_argument("--output-dir", required=True, help="Path to output the YOLO dataset")
    args = parser.parse_args()

    json_path = Path(args.json)
    images_in_dir = Path(args.images_dir)
    
    output_base = Path(args.output_dir)
    output_labels = output_base / "labels" / "train"
    output_images = output_base / "images" / "train"
    
    output_labels.mkdir(parents=True, exist_ok=True)
    output_images.mkdir(parents=True, exist_ok=True)

    print(f"Loading annotations from {json_path}...")
    with open(json_path) as f:
        data = json.load(f)

    # Automatically identify pedestrian class
    person_cat = next(
        (c for c in data['categories'] if c['name'].lower() in ['person', 'pedestrian', 'people', 'human']),
        data['categories'][0] 
    )
    person_id = person_cat['id']
    print(f"✅ Using category: ID={person_id} Name='{person_cat['name']}'")

    img_lookup = {img['id']: img for img in data['images']}

    ann_by_img = {}
    for ann in data['annotations']:
        if ann['category_id'] != person_id:
            continue
        ann_by_img.setdefault(ann['image_id'], []).append(ann)

    written = 0
    copied = 0
    print("Processing images and filtering labels...")
    for img_id, anns in ann_by_img.items():
        img_info = img_lookup[img_id]
        W, H = img_info['width'], img_info['height']
        file_name = img_info['file_name']
        stem = Path(file_name).stem
        
        # Check if the image actually exists in the downloaded subset
        src_img_path = images_in_dir / file_name
        if not src_img_path.exists():
            continue # Skip images that weren't downloaded in the 25GB subset

        lines = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            xc = (x + w/2) / W
            yc = (y + h/2) / H
            wn, hn = w / W, h / H
            # Clamp to prevent out-of-bounds errors
            xc = max(0.001, min(0.999, xc))
            yc = max(0.001, min(0.999, yc))
            wn = max(0.001, min(0.999, wn))
            hn = max(0.001, min(0.999, hn))
            lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            
        if lines:
            # Write label file
            with open(output_labels / f"{stem}.txt", 'w') as f:
                f.write("\n".join(lines))
            written += 1
            
            # Link/Copy the image so YOLO can find it easily alongside the label
            dst_img_path = output_images / file_name
            if not dst_img_path.exists():
                try:
                    import os
                    os.link(src_img_path, dst_img_path) # Fast hardlink
                except Exception:
                    shutil.copy2(src_img_path, dst_img_path) # Fallback to copy
            copied += 1

    print(f"\n✅ Created clean YOLO dataset at: {output_base}")
    print(f"   - Wrote {written} pedestrian-only label files")
    print(f"   - Linked {copied} existing images")
    print(f"   - Skipped any missing images (from your 25GB subset limit)")

if __name__ == "__main__":
    main()
