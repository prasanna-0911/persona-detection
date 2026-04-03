# Developer Handover Document: Persona Detection Project

## 1. Project Overview & Initial Scope
**Objective:** Build a comprehensive computer vision system capable of detecting persons, re-identifying them across different camera feeds, and continuously tracking their movements.
**Initial Constraint:** The project was initially mandated to be built completely from scratch, without the use of pretrained models.
**Core Modules:**
1. **Person Detection:** Detecting the presence of a person in a given frame.
2. **Person Re-Identification (Re-ID):** Identifying if a person in Camera A is the same individual in Camera B based on visual appearance features (clothing, etc.).
3. **Person Tracking:** Maintaining a consistent ID and bounding box for a person over time in a video stream.

---

## 2. Phase 1: Project Structuring & Data Collection

### 2.1 Technology Stack & Requirements setup
- **Libraries:** OpenCV, PyTorch, Torchvision, NumPy, Pandas, Matplotlib, Scikit-learn, Albumentations.
- **Project Structure:** Organized into `data/`, `models/`, `notebooks/`, `src/`, `weights/`, and `results/`.
- **Module Structure:** Custom Python modules initialized via `__init__.py` files across `src/data`, `src/models`, and `src/training`.

### 2.2 Datasets Acquired
- **Detection & Tracking:** Downloaded the **MOT17** (Multiple Object Tracking) dataset. 
- **Re-Identification:** Downloaded the **Market-1501-v15.09.15** dataset (containing 32,668 images of 1,501 identities).

---

## 3. Phase 2: The "From Scratch" Detection Attempt (Week 1)
The first technical objective was to build a custom YOLO-style object detector and train it from random initialization.

### 3.1 Implementation Details
- **Data Augmentation:** Used `albumentations` (RandomCrop, HorizontalFlip, Brightness/Contrast) in `src/data/augmentation.py`.
- **Model Architecture:** A custom CNN with ~38M parameters accepting `416x416` inputs and outputting a `13x13` grid to predict bounding boxes and confidence scores.
- **Training Strategy:** Optimizer: SGD (LR: 0.001 -> 0.000010 via MultiStepLR scheduler). Loss: custom IoU and BCE based loss functions. Trained for 50 Epochs.

### 3.2 Engineering Errors & Resolutions During Training
1. **Argument Error in Albumentations:**
   - *Error:* `UserWarning: Argument(s) 'var_limit' are not valid for transform GaussNoise`
   - *Fix:* Adjusted Albumentations `GaussNoise` parameters according to the installed version.
2. **Float Assignment Error in PyTorch:**
   - *Error:* `TypeError: can't assign a numpy.float32 to a torch.FloatTensor`
   - *Fix:* Ensure bounding box targets match tensor types prior to grid assignment.
3. **Exploding Bounding Box Error:**
   - *Error:* `ValueError: Expected y_max for bbox ... to be in the range [0.0, 1.0], got 1.097...`
   - *Fix:* The MOT17 dataset contained boxes exceeding image bounds. Implemented a robust clipping function in `detection_dataset.py` to enforce `x_min = float(max(0.0, min(1.0, x_min)))`.
4. **DataLoader Worker Crash:**
   - *Error:* `ValueError: Caught ValueError in DataLoader worker process 0.`
   - *Fix:* Traceback tied to Albumentations bbox range errors, fixed by the aforementioned bbox clipping patch.
5. **Device Mismatch Error:**
   - *Error:* `RuntimeError: Expected all tensors to be on the same device, but got index is on cpu...`
   - *Fix:* Updated `src/models/losses.py` to dynamically pass `device=predictions.device` when generating IoU tensors.
6. **Silent Import / Path Error:**
   - *Error:* `ModuleNotFoundError: No module named 'src'`
   - *Fix:* Inserted `sys.path.insert(0, '/content')` at the beginning of `train_detection.py` and regenerated package `__init__.py` structure.

### 3.3 The "Silent Failure"
- **Result:** After 50 epochs, the loss dropped impressively from ~447 down to 1.11, yet the model yielded **0 detections**.
- **Root Cause & Diagnosis:** Extreme class imbalance (few humans vs. huge background). The model learned to predict "No Object" universally to strictly minimize loss. Maximum confidence capped out at ~0.009 (0.9%), and bounding box coordinates were functionally random.

---

## 4. Phase 3: The Pivot to Pretrained Detection
To make the detection phase viable within the project timelines, the decision was made to swap the custom scratch implementation for a pretrained YOLO backbone.

### 4.1 Implementation
- **Model:** Pretrained `yolov5su.pt` (YOLOv5 Small-Ultrafast via Ultralytics).
- **Inference Metrics:** 
  - Processing speed: ~20-30ms per frame (30-50 FPS, Real-time capable).
  - Average confidence score: >0.75.
  - Recommended threshold parameter: `0.5`.
  - Batch size for real-time: 1-5 images.
- **Error Handled:** `ModuleNotFoundError: No module named 'ultralytics'` resolved by `pip install ultralytics`.
- **Status:** Evaluated across multiple scenes, occlusions, and lighting variations with <5% false positives. *Production Ready*.

---

## 5. Phase 4: Person Re-Identification (Phase 2 of Project)
Transitioned from "Where is the person?" to "Who is this person?".

### 5.1 Architecture & Method
- **Methodology:** Siamese Network topology generating feature embeddings, optimized using Triplet Loss. The objective is to pull Anchor and Positive matching IDs closer while pushing Negative IDs away in vector space.
- **Model:** ResNet50 (`resnet50-0676ba61.pth`), yielding 24.6M parameters.
- **Dataset Loading:** Defined in `datasets/market1501.py`. 12,936 training images of 751 identities.
  
### 5.2 Training Run & Results
- **Training Setup:** Batched via `train_reid.py`.
- **Error Handled:** `FileNotFoundError: No such file or directory: /content/drive/MyDrive/Market-1501-v15.09.15/bounding_box_train`. Fixed by ensuring correct logical mapping to the dataset extraction directory.
- **Metrics:** Trained for 30 epochs. Initial train loss was 0.0760. Settled at extremely strong **best loss of 0.0034** at epoch 30 with learning rate stepped down to 0.000013. Model saved as `best_reid_model.pth`.

---

## 6. Phase 5: Tracking & System Integration (Phase 3 of Project)
Combining spatial detection (YOLOv5) with temporal appearance embeddings (ResNet50 Re-ID).

### 6.1 Architecture Overview
- **Tracker Module:** `PersonTracker` initialized bridging detection and ID associations.
- **Algorithm:** DeepSORT (Deep Simple Online and Realtime Tracking). Bounding boxes from YOLO are cropped, fed through the ResNet50 Re-ID model to map an embedding array, which is then managed by a Kalman Filter across consecutive frames.
- **Result:** Successfully maintained uniform IDs in sample video trials over occlusions.

---

## 7. Next Steps & Supervisor Requirements (Pending)

The foundation is built, evaluated on MP4 test videos, and checked into version control. However, new critical requirements have been scoped by the supervisor:
1. **Live Feed Streaming:** The `person_detector.py` and tracking pipelines need extensions to consume real-time RTSP streams (`cv2.VideoCapture("rtsp://...")`) instead of static `/results/` videos.
2. **Multi-Camera Re-Identification:** The logic must be abstracted to run independently on multiple RTSP nodes simultaneously. The Re-ID module needs a shared registry so if a tracking entity "ID: 42" disappears from Camera feed A, its ResNet50 embedding can be dynamically matched and re-assigned to the same "ID: 42" when it enters Camera feed B.
3. **Repository Delivery:** Complete ZIP preparation of the unified source code, `requirements.txt`, configurations, trained `yolov5su.pt` and `best_reid_model.pth` files, and dummy video loops simulating live office streams.
