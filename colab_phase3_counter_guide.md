# Phase3 Counter - Google Colab Testing Guide

This guide explains how to test the Room Occupancy Counter on Google Colab with your own videos.

---

## Step 1: Upload Your Project to Google Drive

### Option A: Zip and Upload

1. **Zip the project folder** on your local PC:
   ```bash
   # In Windows PowerShell
   Compress-Archive -Path "C:\path\to\persona-detection-main" -DestinationPath "persona-detection.zip"
   ```

2. **Upload** `persona-detection.zip` to your Google Drive

3. **Or upload directly** using Colab file upload:
   ```python
   from google.colab import files
   uploaded = files.upload()
   ```

### Option B: Clone from GitHub

```python
!git clone https://github.com/your-username/persona-detection.git
%cd persona-detection
```

---

## Step 2: Setup Environment

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')
```

```bash
# Install dependencies
!pip install ultralytics opencv-python numpy tqdm torch torchvision

# Extract project (if zipped)
!unzip "/content/drive/MyDrive/persona-detection.zip" -d "/content/workspace"
%cd /content/workspace/persona-detection-main
```

---

## Step 3: Upload Your Test Videos

### Option A: Upload to Google Drive

1. Upload your `.mp4` videos to Google Drive
2. Note the path (e.g., `/content/drive/MyDrive/test_videos/video1.mp4`)

### Option B: Upload Directly to Colab

```python
from google.colab import files

print("Upload your test video:")
uploaded = files.upload()

# Get the filename
video_file = list(uploaded.keys())[0]
print(f"Uploaded: {video_file}")
```

### Option C: Copy from Drive to Colab

```python
import shutil
import os

# Create test footage folder
os.makedirs('phase3_counter/test_footage', exist_ok=True)

# Copy video from Drive to Colab workspace
shutil.copy('/content/drive/MyDrive/video1.mp4', 'phase3_counter/test_footage/')
print("Video copied to workspace")
```

---

## Step 4: Configure for Testing

Create or update `counter_config.json`:

```python
import json

config = {
    "room_name": "Test Room",
    "yolo_model": "yolov8m.pt",
    "conf": 0.45,
    "imgsz": 640,  # Smaller size for faster Colab processing
    "device": "cuda",  # Use GPU on Colab
    "display": False,  # No display on Colab
    "save_output": True,
    "output_dir": "phase3_counter/outputs",
    "logging": {
        "log_dir": "phase3_counter/logs",
        "flush_every": 10
    },
    "cameras": [
        {
            "name": "Test_Camera",
            "source": "phase3_counter/test_footage/video1.mp4",  # Your test video
            "crossing_mode": "line",   # line, region, or both (see below)
            "crossing_point": "foot",  # foot, center, top, or mid-foot
            "lines": [
                {
                    "door_id": "Door_A",
                    "door_name": "Test Door",
                    "start": [0, 0],  # Will calibrate
                    "end": [0, 0],    # Will calibrate
                    "inside_sign": 1,
                    "band_width": 25,
                    "direction": "both"
                }
            ]
        }
    ]
}

with open('phase3_counter/config/counter_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Config saved!")
```

### Crossing Mode Selector

| `crossing_mode` | Behavior |
|----------------|----------|
| `"line"` (default) | Only count people crossing the virtual door lines |
| `"region"` | Only count people crossing the polygon boundary (no lines needed) |
| `"both"` | Both line and region count independently. Duplicate events from same track in same frame are removed. |

### Crossing Point Selector

| `crossing_point` | Point Used | Best For |
|-----------------|-----------|----------|
| `"foot"` (default) | Bottom of bounding box `(cx, y2)` | Eye-level cameras |
| `"center"` | Center of bounding box `(cx, cy)` | High-angle corner cameras, occlusion |
| `"top"` | Top of bounding box `(cx, y1)` | Overhead cameras |
| `"mid-foot"` | Midpoint between center and foot `(cx, (cy+y2)//2)` | Balanced angles |

---

## Step 5: Run Basic Tracking Test (No Calibration)

Test with simple tracking first to verify setup:

```python
import subprocess

result = subprocess.run([
    'python', 'phase3_counter/counter_main.py',
    '--config', 'phase3_counter/config/counter_config.json',
    '--max-frames', '100',
    '--no-display',
    '--no-save'
], capture_output=True, text=True)

print(result.stdout)
print(result.stderr)
```

---

## Step 6: Manual Calibration (Without Interactive Window)

Since Colab doesn't support interactive windows, you need to manually set coordinates.

### Option A: Estimate Coordinates

1. Open your video in VLC or any player
2. Note the resolution (e.g., 1920x1080)
3. Estimate door line positions:
   - `start`: top position of door line (x1, y1)
   - `end`: bottom position of door line (x2, y2)
   - `inside_sign`: Which side is the room interior?

```python
import json

# Load current config
with open('phase3_counter/config/counter_config.json', 'r') as f:
    config = json.load(f)

# Update with your estimated coordinates
config['cameras'][0]['lines'][0]['start'] = [320, 200]   # Replace with your values
config['cameras'][0]['lines'][0]['end'] = [320, 700]     # Replace with your values
config['cameras'][0]['lines'][0]['inside_sign'] = 1      # 1 or -1

# Set crossing point (foot/center/top/mid-foot)
config['cameras'][0]['crossing_point'] = 'center'

# Set crossing mode (line/region/both)
config['cameras'][0]['crossing_mode'] = 'line'

# Save updated config
with open('phase3_counter/config/counter_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Calibration coordinates updated!")
```

### Option B: Use Frame Analysis to Find Coordinates

```python
import cv2
import numpy as np

# Load video to analyze frame
video_path = 'phase3_counter/test_footage/video1.mp4'
cap = cv2.VideoCapture(video_path)

# Read first frame
ret, frame = cap.read()
if ret:
    h, w = frame.shape[:2]
    print(f"Video resolution: {w}x{h}")

    # Save frame for manual inspection
    cv2.imwrite('phase3_counter/test_footage/frame_sample.jpg', frame)
    print("Sample frame saved: phase3_counter/test_footage/frame_sample.jpg")

cap.release()
```

Then manually check the image and estimate coordinates.

---

## Step 7: Run Full Counter Test

```python
import subprocess
import os

# Run the counter with limited frames for testing
result = subprocess.run([
    'python', 'phase3_counter/counter_main.py',
    '--config', 'phase3_counter/config/counter_config.json',
    '--max-frames', '500',  # Test with 500 frames
    '--no-display'
], capture_output=True, text=True)

print("=== STDOUT ===")
print(result.stdout)

if result.stderr:
    print("\n=== STDERR ===")
    print(result.stderr)
```

---

## Step 8: Download Results

```python
from google.colab import files
import shutil

# Download output video
if os.path.exists('phase3_counter/outputs/Camera_1.avi'):
    files.download('phase3_counter/outputs/Camera_1.avi')
    print("Output video downloaded!")

# Download logs
if os.path.exists('phase3_counter/logs'):
    import glob
    logs = glob.glob('phase3_counter/logs/*.csv')
    for log in logs:
        print(f"Log: {log}")
```

### Download to Google Drive

```python
# Copy results to Drive for persistence
import shutil
import os

os.makedirs('/content/drive/MyDrive/counter_results', exist_ok=True)
shutil.copy('phase3_counter/outputs/Camera_1.avi',
            '/content/drive/MyDrive/counter_results/Camera_1.avi')

# Copy logs
for f in os.listdir('phase3_counter/logs'):
    shutil.copy(f'phase3_counter/logs/{f}',
                f'/content/drive/MyDrive/counter_results/{f}')

print("Results saved to Google Drive!")
```

---

## Step 9: Add Room Region (Polygon)

Room region can be used in three ways depending on your `crossing_mode`:

### Mode A: Event Filter (with `crossing_mode: "line"`)
Only count door-line events where the person ends up inside the polygon.

```python
import json

with open('phase3_counter/config/counter_config.json', 'r') as f:
    config = json.load(f)

# Add room region polygon (4-point rectangle)
config['cameras'][0]['room_region'] = {
    "polygon": [
        [50, 100],      # Top-left
        [800, 100],     # Top-right
        [800, 600],     # Bottom-right
        [50, 600]       # Bottom-left
    ]
}
config['cameras'][0]['crossing_mode'] = 'line'

with open('phase3_counter/config/counter_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Room region added as event filter!")
```

### Mode B: Standalone Counter (with `crossing_mode: "region"`)
No door lines needed. The polygon boundary itself generates ENTRY/EXIT events when people cross it. Remove the `lines` array entirely:

```python
import json

with open('phase3_counter/config/counter_config.json', 'r') as f:
    config = json.load(f)

# Remove door lines (no longer needed)
del config['cameras'][0]['lines']

# Define polygon as the counting boundary
config['cameras'][0]['room_region'] = {
    "polygon": [
        [50, 100],      # Top-left
        [800, 100],     # Top-right
        [800, 600],     # Bottom-right
        [50, 600]       # Bottom-left
    ]
}
config['cameras'][0]['crossing_mode'] = 'region'
config['cameras'][0]['crossing_point'] = 'center'

with open('phase3_counter/config/counter_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Room region set as standalone counter!")
```

### Mode C: Both (with `crossing_mode: "both"`)
Both door lines AND polygon boundary generate events independently. Duplicate events from the same track on the same frame are removed automatically.

```python
import json

with open('phase3_counter/config/counter_config.json', 'r') as f:
    config = json.load(f)

config['cameras'][0]['room_region'] = {
    "polygon": [
        [50, 100],
        [800, 100],
        [800, 600],
        [50, 600]
    ]
}
config['cameras'][0]['crossing_mode'] = 'both'

with open('phase3_counter/config/counter_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("Room region added, crossing_mode set to both!")
```

---

## Troubleshooting

### Issue: No Display Available

```python
# Already handled with --no-display flag
# Make sure this is always used in Colab
```

### Issue: CUDA Out of Memory

```python
# Use smaller image size
config['imgsz'] = 640  # Instead of 960 or higher
```

### Issue: Video Not Found

```python
# Verify video path
import os
print(os.path.exists('phase3_counter/test_footage/video1.mp4'))
```

### Issue: Permission Denied

```python
# Make files executable
!chmod +x phase3_counter/counter_main.py
!chmod +x phase3_counter/utils/*.py
```

---

## Quick Test Templates

### Template A: Door Line Mode (default)

```python
# === QUICK TEST — DOOR LINE MODE ===

# 1. Setup
from google.colab import drive
drive.mount('/content/drive')
!pip install ultralytics opencv-python numpy tqdm

# 2. Navigate to project
%cd /content/workspace/persona-detection-main

# 3. Create config with door lines
import json
config = {
    "room_name": "Test",
    "yolo_model": "yolov8m.pt",
    "conf": 0.45,
    "imgsz": 640,
    "device": "cuda",
    "display": False,
    "save_output": True,
    "output_dir": "phase3_counter/outputs",
    "logging": {"log_dir": "phase3_counter/logs", "flush_every": 10},
    "cameras": [{
        "name": "Test",
        "source": "YOUR_VIDEO_PATH.mp4",
        "crossing_mode": "line",
        "crossing_point": "foot",
        "lines": [{
            "door_id": "D1",
            "door_name": "Door",
            "start": [X1, Y1],  # Replace with your values
            "end": [X2, Y2],    # Replace with your values
            "inside_sign": 1,
            "band_width": 25,
            "direction": "both"
        }]
    }]
}

with open('phase3_counter/config/counter_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# 4. Run test
!python phase3_counter/counter_main.py --max-frames 200 --no-display

# 5. Check logs
import glob
print("Logs:", glob.glob('phase3_counter/logs/*.csv'))
```

### Template B: Standalone Region Mode (no door lines)

```python
# === QUICK TEST — STANDALONE REGION MODE ===
# Use this when you don't need door lines.
# The polygon boundary itself counts ENTRY/EXIT.

# 1. Setup
from google.colab import drive
drive.mount('/content/drive')
!pip install ultralytics opencv-python numpy tqdm

# 2. Navigate to project
%cd /content/workspace/persona-detection-main

# 3. Create config with region only (no lines)
import json
config = {
    "room_name": "Test",
    "yolo_model": "yolov8m.pt",
    "conf": 0.45,
    "imgsz": 640,
    "device": "cuda",
    "display": False,
    "save_output": True,
    "output_dir": "phase3_counter/outputs",
    "logging": {"log_dir": "phase3_counter/logs", "flush_every": 10},
    "cameras": [{
        "name": "Test",
        "source": "YOUR_VIDEO_PATH.mp4",
        "crossing_mode": "region",
        "crossing_point": "center",
        "room_region": {
            "polygon": [[50, 100], [800, 100], [800, 600], [50, 600]]
        }
    }]
}

with open('phase3_counter/config/counter_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# 4. Run test
!python phase3_counter/counter_main.py --max-frames 200 --no-display

# 5. Check logs
import glob
print("Logs:", glob.glob('phase3_counter/logs/*.csv'))
```

### Template C: Both Mode (lines + region)

```python
# === QUICK TEST — BOTH MODE ===
# Door lines AND region boundary both count independently.
# Same track on same frame is counted once.

# 1. Setup
from google.colab import drive
drive.mount('/content/drive')
!pip install ultralytics opencv-python numpy tqdm

# 2. Navigate to project
%cd /content/workspace/persona-detection-main

# 3. Create config with both lines and region
import json
config = {
    "room_name": "Test",
    "yolo_model": "yolov8m.pt",
    "conf": 0.45,
    "imgsz": 640,
    "device": "cuda",
    "display": False,
    "save_output": True,
    "output_dir": "phase3_counter/outputs",
    "logging": {"log_dir": "phase3_counter/logs", "flush_every": 10},
    "cameras": [{
        "name": "Test",
        "source": "YOUR_VIDEO_PATH.mp4",
        "crossing_mode": "both",
        "crossing_point": "center",
        "lines": [{
            "door_id": "D1",
            "door_name": "Door",
            "start": [X1, Y1],
            "end": [X2, Y2],
            "inside_sign": 1,
            "band_width": 25,
            "direction": "both"
        }],
        "room_region": {
            "polygon": [[50, 100], [800, 100], [800, 600], [50, 600]]
        }
    }]
}

with open('phase3_counter/config/counter_config.json', 'w') as f:
    json.dump(config, f, indent=2)

# 4. Run test
!python phase3_counter/counter_main.py --max-frames 200 --no-display

# 5. Check logs
import glob
print("Logs:", glob.glob('phase3_counter/logs/*.csv'))
```