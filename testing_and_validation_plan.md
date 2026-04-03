# Comprehensive Pipeline Validation Plan

This document defines the strict testing parameters you must use to evaluate both the **YOLO Low-Light Detection Model** and the **Re-Identification (Re-ID) Model**. To ensure production readiness, you must test them in isolation first, then together.

---

## Phase 1: YOLO Detection (Isolated Target Testing)

We need to mathematically prove the model detects people in the dark better than the stock model, without generating massive amounts of false positives (hallucinating people in the shadows).

### 1. The Core Metrics
You will use Ultralytics validation logic to generate these metrics. 
- **mAP50 (Mean Average Precision at 50% overlap):** The overall grade of the model's ability to find people.
- **Recall (R):** The percentage of *actual* people in the dark that the model successfully found. (Crucial for low-light).
- **Precision (P):** The percentage of the model's predictions that were actually people, not trash cans or shadows.

### 2. Thresholds and Deviations
- **Recall Target:** > `0.75` (We want to find at least 75% of people hiding in the dark).
- **Precision Target:** > `0.80`
- **Acceptable Deviation:** In low-light models, it is normal to see Precision drop slightly (-0.05 deviation) while Recall spikes (+0.15). If the AI accidentally flags a mailbox as a person once, it is an acceptable trade-off to ensure a real person in the dark is never missed.

### 3. How to Test It
Do not test on the training data. Run validation on a completely unseen nighttime video or subset:
```powershell
yolo val model=F:\persona-detection-main\runs\detect\yolov8s_rot0\weights\best.pt data=datasets/nightowls_test.yaml split=test
```

> [!IMPORTANT]  
> If **Recall** sits below 0.60 on dark images, the YOLO model requires either a lower confidence threshold (`conf=0.25` during tracking) or another rotation of training.

---

## Phase 2: Re-Identification (Isolated Target Testing)

Re-ID is entirely responsible for ensuring Person A on Camera 1 is correctly matched to Person A on Camera 2.

### 1. The Core Metrics
Run `evaluate_reid.py` on the `LaST` or `Market1501` dataset. You are looking for two exact numbers:
- **Rank-1 Accuracy:** If I give the AI a cropped image of Person X, what is the probability that its #1 closest match in the database is actually Person X?
- **mAP (Mean Average Precision for ReID):** Overall clustering quality.

### 2. Thresholds and Deviations
- **Rank-1 Target:** > `85%` `(0.85)`
- **Deviation Metric (Cosine Distance Margin):** 
  When the model converts an image to a mathematical embedding (128 numbers), you measure the distance between two images.
  - Same Person: Distance should be `< 0.35`
  - Different People: Distance should be `> 0.65`
  - **The "Danger Zone":** Between `0.35` and `0.65`. This gap must be at least `0.3`. If the distance between two different people drops to `0.4`, your model is collapsing identities together.

### 3. How to Test It
Use your previously fixed evaluation script on the LaST dataset:
```powershell
python phase2_reid/evaluate_reid.py --model phase2_reid/checkpoints/best_reid_model.pth --dataset last --root "E:/reid_datasets/last"
```

---

## Phase 3: The End-to-End Stress Test (Tracker)

This is the ultimate test. You feed the pipeline a real-world, highly challenging video. YOLO detects the boxes, Re-ID extracts the embeddings, and the Tracker (`botsort` or `bytetrack`) connects them over time.

### 1. The Core Metrics
- **IDSW (Identity Switches):** This is the ultimate failure metric. An ID switch happens when a person is labeled "ID: 4", they walk behind a car for 2 seconds, emerge, and the AI aggressively re-labels them as "ID: 19".
- **MOTA (Multiple Object Tracking Accuracy):** A combined score that heavily penalizes ID switches and missed detections.

### 2. Thresholds
- **IDSW Target:** `< 2` switches per continuous human track. If a person walks across the screen for 30 seconds uninterrupted, their ID number should absolutely never change. 
- **Revert Time:** If an ID *does* switch because of an occlusion, Re-ID should recognize them and revert their ID back to the original within `30 frames` (1 second).

### 3. How to Test It
Run the tracker script on the hardest video you can find (a 5-minute video of people walking under streetlights, crossing paths, and going behind cars).
```powershell
python run_tracker.py `
    --source "test_footage_night.mp4" `
    --yolo-model "F:\persona-detection-main\runs\detect\yolov8s_rot0\weights\best.pt" `
    --reid-model "phase2_reid/checkpoints/best_reid_model.pth" `
    --output "results/final_stress_test.mp4"
```

### 4. The Final Output
The true final deliverable is the `final_stress_test.mp4` video. Every person should have a **Bounding Box** and a bright **ID Number** hovering over their heads. 

You must sit down, watch the 5-minute video, and physically count the ID Switches. No automated mathematical metric can beat the human eye evaluating if the system correctly re-identified someone emerging from the shadows.
