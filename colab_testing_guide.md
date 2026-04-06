# Google Colab: The Definitive Validation Guide (Drive Strategy)

You have outsmarted the storage limits! By keeping the compressed ZIP file safely inside your Google Drive, your Colab instance only has to hold the extracted images. This saves massive amounts of space (Colab's 80GB is perfectly safe), allowing us to do a pure mathematical parameter sweep.

*(Note: The entire NightOwls dataset is 100GB+, but the **Validation** subset specifically is much smaller!)*

---

### Phase 1: Environment & Dataset Preparation

Run these in separate Code Cells `[ + Code ]`:

**1. Mount Drive & Unzip Your Repo**
```python
from google.colab import drive
drive.mount('/content/drive')
```
```bash
# Extract your repository code into the fast Colab workspace
!unzip "/content/drive/MyDrive/persona-detection-main.zip" -d "/content/workspace"
```

**2. Install Dependencies**
```bash
%cd /content/workspace/persona-detection-main
!pip install ultralytics lapx filterpy
```

**3. The Google Drive Storage Trick**
We will download the validation data *directly* into your mapped Google Drive so it doesn't take up any of Colab's internal storage limits.
```bash
# Navigate into your Google Drive
%cd /content/drive/MyDrive

# Download the validation zip safely into your Drive
!wget http://www.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_validation.zip
!wget http://www.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_validation.json
```

**4. Extract the Images into Colab**
Now we extract those images from the Drive directly into Colab's high-speed workspace.
```bash
!mkdir -p /content/nightowls_val
# Extract the images FROM Drive TO Colab Workspace
!unzip -q /content/drive/MyDrive/nightowls_validation.zip -d /content/nightowls_val

# Copy the JSON file over so they are side-by-side
!cp /content/drive/MyDrive/nightowls_validation.json /content/nightowls_val/
```

**5. Convert the Validation Data to YOLO format**
```bash
%cd /content/workspace/persona-detection-main

# Run your custom converter script to make .txt labels
!python training_scripts/convert_coco_yolo.py \
    --json "/content/nightowls_val/nightowls_validation.json" \
    --images-dir "/content/nightowls_val/nightowls_validation" \
    --output-dir "/content/nightowls_val_yolo"
```

**6. Create a YOLO Configuration File**
We need to tell YOLO where to find this new dataset.
```python
# Run this inside a python code cell
import yaml

data = {
    'path': '/content/nightowls_val_yolo',
    'val': 'images/train', # The converter outputs to 'train' folder by default
    'nc': 1,
    'names': {0: 'person'}
}

with open('/content/workspace/persona-detection-main/nightowls_val.yaml', 'w') as f:
    yaml.dump(data, f)
```

---

### Phase 2: The Quantitative Sweep

**1. Update the Sweep Script**
Open the file `training_scripts/yolo_param_sweep.py` in your Colab side panel by double-clicking it. Change the paths on Lines 9 and 11:
```python
model_path = "/content/workspace/persona-detection-main/runs/detect/yolov8s_rot0/weights/best.pt"
data_yaml = "/content/workspace/persona-detection-main/nightowls_val.yaml"
```

**2. Run the Parameter Sweep**
```bash
!python training_scripts/yolo_param_sweep.py
```
*(When it finishes, hover over `yolo_sweep_results.csv` on the far-left Colab sidebar and download it! Look at the spreadsheet and identify which `Conf` and `IOU` gave you the best `mAP50` score).*

---

### Phase 3: The Visual CCTV Test

Now that you have the mathematically perfect parameters from the `.csv` file, use them to run your visual test on your CCTV video. *(Let's pretend the sweep found that `conf 0.25` is mathematically superior)*:

```bash
!python run_tracker.py \
    --source "/content/drive/MyDrive/cctv_clip.mp4" \
    --yolo-model "/content/workspace/persona-detection-main/runs/detect/yolov8s_rot0/weights/best.pt" \
    --conf 0.25 \
    --imgsz 640 \
    --output "/content/workspace/final_cctv_test.mp4"
```

Download `final_cctv_test.mp4` from the Colab sidebar to your PC and you are done!
