# 🎯 Persona Detection System
<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

> A complete person detection, re-identification, and tracking system built as an AI/ML internship project.
</div>



## 📋 Table of Contents

- [Features](#-features)
- [Performance](#-performance)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Training](#-training)
- [Technologies](#-technologies)
- [Key Learnings](#-key-learnings)
- [Future Improvements](#-future-improvements)

---

## 🚀 Features

| Feature | Description |
|---------|-------------|
| **Real-time Detection** | Detect persons in video frames using YOLO |
| **Re-Identification** | Recognize same person across different camera views |
| **Consistent Tracking** | Maintain strict unique IDs using BoT-SORT local mapping |
| **Occlusion Handling** | Track persons even when temporarily hidden |

---

## 📊 Performance

### Detection (Phase 1)
| Metric | Value |
|--------|-------|
| Speed | 30-50 FPS |
| Confidence | >75% |
| False Positive Rate | <5% |
| Crowded Scenes | ✅ Supported |
| Lighting Variations | ✅ Handled |

### Re-Identification (Phase 2)
| Metric | Value |
|--------|-------|
| Same Person Similarity | 0.912 |
| Different Person Similarity | 0.078 |
| Separation Gap | 0.834 |
| Training Loss | 0.0034 |
| Embedding Dimension | 128 |

### Tracking (Phase 3)
| Metric | Value |
|--------|-------|
| ID Consistency | ✅ Maintained |
| Occlusion Handling | ✅ Supported |
| Real-time Capable | ✅ Yes |

---

## 🏗 Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                        TRACKING PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Video        YOLO           Re-ID         BoT-SORT            │
│   Frame  ──►  Detection  ──► Features  ──►  Tracking  ──► Output│
│                                                                 │
│             "Where are      "Who is       "Track ID:42          │
│              people?"        this?"        maintained"          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
### Phase Breakdown

| Phase | Component | Purpose |
|-------|-----------|---------|
| Phase 1 | YOLO | Detect persons in each frame |
| Phase 2 | ResNet50 + Triplet Loss | Extract appearance features |
| Phase 3 | BoT-SORT (Ultralytics) | Track and maintain local IDs flawlessly |

---

## 📁 Project Structure
## 📁 Project Structure

```text
persona_detection_final/
│
├── 📂 phase1_detection/
│   └── person_detector.py
│
├── 📂 phase2_reid/
│   ├── 📂 datasets/
│   │   └── market1501.py       # Dataset loader
│   ├── 📂 models/
│   │   └── reid_net.py         # ReID network
│   ├── 📂 losses/
│   │   └── triplet_loss.py     # Loss function
│   ├── 📂 checkpoints/
│   │   └── best_reid_model.pth # Trained model (24.6M params)
│   ├── train_reid.py           # Training script
│   └── evaluation_results.json # Metrics
│
├── multi_camera_tracker.py     # BoT-SORT multi-camera core pipeline
├── run_tracker.py              # CLI Entrypoint for tracker
│
├── 📂 results/
│   └── tracking_result.mp4     # Demo video
│
├── 📂 reports/
│   ├── final_project_report.txt
│   ├── project_summary.json
│   └── presentation_outline.txt
│
└── README.md
```
---

## 🛠 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Install Dependencies

```bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install scipy
pip install numpy
pip install Pillow
pip install tqdm
```
OR From requirements.txt -
```bash
pip install -r requirements.txt
```
### Clone Repository
```bash
https://github.com/prasanna-0911/persona-detection
# Download from Google Drive or clone repository
cd persona-detection
```

# Create virtual environment
```python
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\\Scripts\\activate  # Windows
```


## 📥 Download Pre-trained Model

The trained Re-ID model is required to run the tracking system.

### GitHub Releases 

Download from [Releases](https://github.com/prasanna-0911/persona-detection/releases/latest):
- `best_reid_model.pth` (≈282 MB)

### Installation

After downloading, place the model file in:
```text
persona-detection/
└── phase2_reid/
└── checkpoints/
└── best_reid_model.pth ← Place here
```



### Model Information

| Property | Value |
|----------|-------|
| Architecture | ResNet50 + 128-dim embedding |
| Parameters | 24.6M |
| Training Dataset | Market-1501 |
| Training Epochs | 30 |
| Final Loss | 0.0034 |
| Same Person Similarity | 0.912 |
| Separation Gap | 0.834 |


## 💻 Usage

### Quick Start (Single Camera / Video)
```bash
python run_tracker.py --source your_video.mp4 --output result.mp4 --device cpu
```

### Multi-Camera Tracking
Process multiple video streams and map the same person ID across different cameras:
```bash
python run_tracker.py --multi-camera --config cameras.json --output-dir outputs/
```

### Live Webcam
```bash
python run_tracker.py --webcam --output live_webcam.mp4
```

## 🎓 Training
### Re-ID Model Training
# Dataset: Market-1501

Training images: 12,936
Identities: 751
Image size: 256 x 128
# Configuration:

```python

CONFIG = {
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 0.0001,
    'embedding_dim': 128,
    'margin': 0.3,  # Triplet loss margin
}
```

### Run Training:

```bash
python phase2_reid/train_reid.py
```
### Training Results:

Final Loss: 0.0034
Same Person Similarity: 0.912
Separation Gap: 0.834
## 🔧 Technologies

| Category | Technology |
| :--- | :--- |
| **Language** | Python 3.12 |
| **Deep Learning** | PyTorch 2.x |
| **Detection** | Ultralytics YOLOv5 |
| **Video Processing** | OpenCV |
| **Numerical** | NumPy, SciPy |

## 📚 Key Learnings
### 1. Transfer Learning is Essential
Challenge: Training from scratch failed completely.

```text

Custom Model Results:
- Training Loss: Decreased ✓
- Actual Detections: 0 ✗
- Max Confidence: 0.9% ✗
```
**Root Cause**: Class imbalance (95% background, 5% persons)

**Solution**: Use pretrained YOLOv5 → Immediate success!

### 2. Low Loss ≠ Good Performance
Always validate with task-specific metrics:

```text

Don't Trust:          Do Trust:
- Training loss       - Similarity gap
                      - Rank-1 accuracy
                      - Visual inspection
```
### 3. Save Progress Frequently
**Problem**: Colab disconnections lost training progress.

**Solution**: Auto-save checkpoints every epoch to Google Drive.
```Python

# Save every epoch
torch.save(checkpoint, 'drive/MyDrive/.../checkpoint.pth')
```

## 🔮 Future Improvements

| Improvement | Description |
| :--- | :--- |
| **Multi-Camera** | Cross-view tracking between cameras |
| **Pose Estimation** | Add skeleton detection for activity recognition |
| **Edge Deployment** | Optimize for Jetson Nano, Raspberry Pi |
| **Web Interface** | Real-time monitoring dashboard |
| **REST API** | Production deployment with Flask/FastAPI |
| **Attribute Recognition** | Detect clothing color, gender, age |

## 📈 Results Demo
The system successfully:

✅ Detects multiple persons in crowded scenes  
✅ Assigns unique IDs to each person  
✅ Maintains IDs across frames  
✅ Handles occlusions and re-appearances  
✅ Processes video in real-time
**Demo video**: results/tracking_result.mp4

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.


## 🙏 Acknowledgments
**YOLO & BoT-SORT** by Ultralytics  
**Market-1501** dataset by Zheng et al. 
<div align="center">
Status: ✅ PRODUCTION READY

</div>

<div align="center">
⭐ Star this repository if you find it helpful!<br>
Built with ❤️ as an AI/ML Internship Project

</div>
