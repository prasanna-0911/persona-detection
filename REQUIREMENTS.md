# Hardware & Software Requirements

## Minimum Recommended Hardware

| Component | Minimum | Recommended | Notes |
|---|---|---|---|
| **GPU** | NVIDIA GTX 1080 (8 GB VRAM) | NVIDIA RTX 4070 (12 GB VRAM) | CUDA-capable GPU required. CPU-only is extremely slow (unusable for real-time). |
| **GPU VRAM** | 6 GB | 12 GB | 6 GB supports `imgsz=640`. 12 GB required for `imgsz=960` (recommended for nighttime accuracy). |
| **RAM** | 16 GB | 32 GB | YOLO loads full model + frame buffers into RAM alongside GPU VRAM. |
| **CPU** | 4-core @ 2.5 GHz | 8-core @ 3.5 GHz+ | Workers for data loading during training. |
| **Storage (SSD)** | 50 GB free | 100 GB free | Training datasets (NightOwls + COCO/CrowdHuman) require ~40 GB. Model checkpoints need ~5 GB per training run. |
| **OS** | Windows 10 / Ubuntu 20.04 | Windows 11 / Ubuntu 22.04 | |

---

## Software Requirements

| Package | Version | Install Command |
|---|---|---|
| Python | 3.10 or 3.12 | [python.org](https://python.org) |
| PyTorch | 2.0+ with CUDA 12.x | `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` |
| Ultralytics | 8.0+ | `pip install ultralytics` |
| OpenCV | 4.8+ | `pip install opencv-python` |
| CUDA Toolkit | 12.1+ | [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) |
| cuDNN | 8.9+ | Bundled with PyTorch CUDA wheels |

### Full environment setup:
```bash
pip install ultralytics opencv-python torch torchvision lapx filterpy tqdm pillow
```

---

## Real-Time Performance Benchmarks

Tested on **RTX 4070 (12 GB VRAM)**:

| Configuration | FPS | Suitable For |
|---|---|---|
| `imgsz=640`, `conf=0.25` | ~45 FPS | Daytime, low-latency streams |
| `imgsz=960`, `conf=0.25` | ~22 FPS | Nighttime, recommended setting |
| `imgsz=960`, 4 cameras parallel | ~6 FPS/camera | Multi-camera deployment |

> [!IMPORTANT]
> For real-time RTSP stream processing, a minimum of **22 FPS** processing speed is required to avoid significant lag. If your GPU cannot sustain this at `imgsz=960`, the system will prompt you to downgrade to `imgsz=640` automatically at startup.

---

## GPU VRAM Auto-Degradation

The system automatically checks GPU VRAM at startup:
- **VRAM ≥ 6 GB**: Proceeds with requested `imgsz` (default: 960)
- **VRAM < 6 GB**: Prints a warning and asks the user to confirm downgrade to `imgsz=640`

Example warning:
```
⚠️  WARNING: Only 4.1 GB VRAM detected.
   Recommended minimum for imgsz=960: 6 GB
   Drop imgsz from 960 → 640 for safety? (y/n):
```

---

## Cloud / Colab Deployment

For running validation or inference on Google Colab:
- Select **T4 GPU** runtime (free tier) or **A100** (Colab Pro)
- T4 has 14.9 GB VRAM — sufficient for `imgsz=960`
- Install dependencies at the start of each session:
  ```bash
  !pip install ultralytics lapx filterpy
  ```

---

## Storage Breakdown (Training Machine)

| Dataset / File | Approx Size |
|---|---|
| NightOwls validation images (~27k) | 26 GB |
| COCO train2017 | 18 GB |
| CrowdHuman | 7 GB |
| EuroCity Persons | 7 GB |
| Training run checkpoints (per run) | 1–2 GB |
| Final model (`best.pt`) | ~45 MB |
