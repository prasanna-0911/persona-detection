"""
Complete Person Tracking Pipeline

Combines YOLOv5 detection, Re-ID feature extraction, and DeepSORT tracking
into a unified person tracking system.

Usage:
    tracker = PersonTracker('path/to/reid_model.pth')
    tracker.process_video('input.mp4', 'output.mp4')
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms

# Add project paths
PROJECT_ROOT = '/content/drive/MyDrive/persona_detection_final'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, f'{PROJECT_ROOT}/phase2_reid')
sys.path.insert(0, f'{PROJECT_ROOT}/phase3_tracking')

from models.reid_net import ReIDNetwork
from trackers import DeepSORTTracker


class PersonTracker:
    """
    Complete person tracking system.
    
    Pipeline:
    1. YOLOv5 detects persons in frame
    2. Re-ID model extracts appearance features
    3. DeepSORT tracks persons across frames
    
    Args:
        reid_model_path: Path to trained Re-ID model
        device: 'cuda' or 'cpu'
    """
    
    def __init__(self, reid_model_path, device='cuda'):
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu'
        )
        print(f"Using device: {self.device}")
        
        # Load YOLOv5 detector
        print("Loading YOLOv5...")
        from ultralytics import YOLO
        self.detector = YOLO('yolov5su.pt')
        
        # Load Re-ID model
        print("Loading Re-ID model...")
        self.reid_model = ReIDNetwork(embedding_dim=128, pretrained=False)
        checkpoint = torch.load(reid_model_path, map_location=self.device)
        self.reid_model.load_state_dict(checkpoint['model_state_dict'])
        self.reid_model = self.reid_model.to(self.device)
        self.reid_model.eval()
        
        # Re-ID preprocessing
        self.reid_transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Initialize tracker
        self.tracker = DeepSORTTracker(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7,
            max_cosine_distance=0.3
        )
        
        # Colors for visualization
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)
        
        print("✅ PersonTracker initialized!")
    
    def extract_reid_features(self, frame, detections):
        """
        Extract Re-ID features for detected persons.
        
        Args:
            frame: BGR image
            detections: Array of [x1, y1, x2, y2, ...]
            
        Returns:
            List of 128-dim feature vectors
        """
        features = []
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            
            # Ensure valid crop coordinates
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                features.append(np.zeros(128))
                continue
            
            # Crop and preprocess
            crop = frame[y1:y2, x1:x2]
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = Image.fromarray(crop)
            crop_tensor = self.reid_transform(crop).unsqueeze(0).to(self.device)
            
            # Extract features
            with torch.no_grad():
                feature = self.reid_model(crop_tensor)
            
            features.append(feature.cpu().numpy().flatten())
        
        return features
    
    def process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame: BGR image (numpy array)
            
        Returns:
            tracks: List of (track_id, bbox) tuples
            detections: Raw detections
        """
        # Run YOLOv5 detection
        results = self.detector(frame, classes=[0], conf=0.5, verbose=False)
        
        # Extract detections
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections.append([x1, y1, x2, y2, conf])
        
        detections = np.array(detections) if detections else np.array([])
        
        if len(detections) == 0:
            return [], []
        
        # Extract Re-ID features
        features = self.extract_reid_features(frame, detections)
        
        # Update tracker
        tracks = self.tracker.update(detections, features)
        
        return tracks, detections
    
    def draw_tracks(self, frame, tracks):
        """
        Draw tracking results on frame.
        
        Args:
            frame: BGR image
            tracks: List of (track_id, bbox) tuples
            
        Returns:
            Annotated frame
        """
        for track_id, bbox in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            
            # Get consistent color for this ID
            color = tuple(map(int, self.colors[track_id % len(self.colors)]))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID label
            label = f'ID: {track_id}'
            label_size, _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            # Label background
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color, -1
            )
            
            # Label text
            cv2.putText(
                frame, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        return frame
    
    def process_video(self, video_path, output_path, max_frames=None):
        """
        Process entire video.
        
        Args:
            video_path: Input video path
            output_path: Output video path
            max_frames: Maximum frames to process (None for all)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        pbar = tqdm(total=total_frames, desc="Processing")
        
        while cap.isOpened():
            if max_frames and frame_count >= max_frames:
                break
                
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            tracks, _ = self.process_frame(frame)
            
            # Draw results
            frame = self.draw_tracks(frame, tracks)
            
            # Add info text
            info = f'Frame: {frame_count} | Tracks: {len(tracks)}'
            cv2.putText(
                frame, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            
            # Write frame
            out.write(frame)
            
            frame_count += 1
            pbar.update(1)
        
        pbar.close()
        cap.release()
        out.release()
        
        print(f"\n✅ Output saved to: {output_path}")


def main():
    """Demo usage"""
    PROJECT_ROOT = '/content/drive/MyDrive/persona_detection_final'
    reid_model_path = f'{PROJECT_ROOT}/phase2_reid/checkpoints/best_reid_model.pth'
    
    tracker = PersonTracker(reid_model_path)
    
    print("\n🎬 PersonTracker ready!")
    print("Usage:")
    print("  tracker.process_video('input.mp4', 'output.mp4')")
    print("  tracks, dets = tracker.process_frame(frame)")
    
    return tracker


if __name__ == '__main__':
    main()
