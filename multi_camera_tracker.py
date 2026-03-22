"""
Multi-Camera Person Tracking System

This module enables tracking persons across multiple cameras while
maintaining the SAME ID for each person regardless of which camera
they appear in.

Key Features:
- Unified video file and RTSP stream support
- Global person gallery for cross-camera Re-ID
- Same person ID maintained across all cameras
- Automatic ID assignment based on appearance similarity

Architecture:
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  Camera 1   │     │  Camera 2   │     │  Camera 3   │
    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Global Gallery    │
                    │  (Person Features)  │
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Same ID Assigned  │
                    │   Across Cameras    │
                    └─────────────────────┘

Usage:
    # Single source (video or RTSP)
    tracker = MultiCameraTracker(reid_model_path)
    tracker.process_source("video.mp4", output_path="output.mp4")
    tracker.process_source("rtsp://user:pass@ip:554/stream", output_path="live.mp4")
    
    # Multiple cameras with global ID
    tracker.process_multi_camera([
        {"name": "Entrance", "source": "rtsp://..."},
        {"name": "Lobby", "source": "rtsp://..."},
        {"name": "Exit", "source": "rtsp://..."},
    ], output_dir="outputs/")
"""

import os

# Fix OpenMP error: "Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import cv2
import time
import threading
import queue
import numpy as np
from datetime import datetime
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Bug #1 fix: Use dynamic path resolution instead of hardcoded Colab path.
# The old code used '/content/drive/MyDrive/persona_detection_final' which
# only works in Google Colab and crashes on Windows/Linux local environments.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'phase2_reid'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'phase3_tracking'))

from models.reid_net import ReIDNetwork
# DeepSORTTracker removed; using Ultralytics BoT-SORT natively
from ultralytics import YOLO


@dataclass
class PersonRecord:
    """
    Record of a tracked person in the global gallery.
    
    Attributes:
        global_id: Unique ID across all cameras
        features: List of Re-ID feature vectors
        last_seen: Timestamp of last detection
        cameras_seen: Set of camera names where person was seen
        total_detections: Total number of detections
    """
    global_id: int
    features: List[np.ndarray]
    last_seen: float
    cameras_seen: set
    total_detections: int = 0
    
    def get_average_feature(self) -> np.ndarray:
        """Get average feature vector for matching."""
        if not self.features:
            return None
        return np.mean(self.features, axis=0)
    
    def add_feature(self, feature: np.ndarray, camera_name: str):
        """Add a new feature observation."""
        self.features.append(feature)
        # Keep only last 50 features
        if len(self.features) > 50:
            self.features = self.features[-50:]
        self.last_seen = time.time()
        self.cameras_seen.add(camera_name)
        self.total_detections += 1


class GlobalPersonGallery:
    """
    Global gallery for cross-camera person Re-ID.
    
    Maintains a database of all known persons and their features.
    When a new person is detected, checks if they match any existing
    person in the gallery.
    """
    
    def __init__(self, similarity_threshold: float = 0.80, max_gallery_size: int = 1000):
        """
        Initialize global gallery.
        
        Args:
            similarity_threshold: Minimum similarity to match (0-1).
                Stricter threshold prevents different people in crowded scenes
                from swapping their Re-ID assignments.
            max_gallery_size: Maximum number of persons to track
        """
        self.persons: Dict[int, PersonRecord] = {}
        self.next_global_id = 1
        self.similarity_threshold = similarity_threshold
        self.max_gallery_size = max_gallery_size
        
        # Local to global ID mapping per camera
        # camera_name -> {local_track_id -> global_id}
        self.local_to_global: Dict[str, Dict[int, int]] = defaultdict(dict)
        
        # Lock for thread safety
        self.lock = threading.Lock()
    
    def find_or_create_global_id(
        self, 
        local_track_id: int, 
        feature: np.ndarray, 
        camera_name: str,
        active_local_ids: set
    ) -> int:
        """
        Find existing global ID or create new one.
        
        Args:
            local_track_id: Track ID from local camera tracker
            feature: Re-ID feature vector
            camera_name: Name of the camera
            active_local_ids: Set of all local track IDs actively detected in this frame
            
        Returns:
            Global ID for this person
        """
        with self.lock:
            # Prevent 2 distinct local persons from mapping to the same Global ID
            # by identifying which Global IDs are already "taken" in this exact frame.
            active_global_ids_in_camera = {
                self.local_to_global[camera_name][tid]
                for tid in active_local_ids
                if tid in self.local_to_global[camera_name]
            }

            # Check if we already have a mapping for this local track
            if local_track_id in self.local_to_global[camera_name]:
                global_id = self.local_to_global[camera_name][local_track_id]
                # Update features
                if global_id in self.persons:
                    self.persons[global_id].add_feature(feature, camera_name)
                return global_id
            
            # Search for matching person in gallery
            best_match_id = None
            best_similarity = 0
            
            for global_id, person in self.persons.items():
                # CRITICAL: Do not merge with a Global ID that is already assigned
                # to another DIFFERENT person actively standing in this exact camera frame!
                if global_id in active_global_ids_in_camera:
                    continue

                avg_feature = person.get_average_feature()
                if avg_feature is None:
                    continue
                
                similarity = self._compute_similarity(feature, avg_feature)
                
                if similarity > best_similarity and similarity >= self.similarity_threshold:
                    best_similarity = similarity
                    best_match_id = global_id
            
            if best_match_id is not None:
                # Found a match - use existing global ID
                self.persons[best_match_id].add_feature(feature, camera_name)
                self.local_to_global[camera_name][local_track_id] = best_match_id
                return best_match_id
            else:
                # No match - create new global ID
                new_global_id = self._create_new_person(feature, camera_name)
                self.local_to_global[camera_name][local_track_id] = new_global_id
                return new_global_id
    
    def _compute_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """Compute cosine similarity between two features."""
        # Normalize
        f1 = feature1 / (np.linalg.norm(feature1) + 1e-8)
        f2 = feature2 / (np.linalg.norm(feature2) + 1e-8)
        
        # Cosine similarity
        similarity = np.dot(f1, f2)
        return float(similarity)
    
    def _create_new_person(self, feature: np.ndarray, camera_name: str) -> int:
        """Create a new person record."""
        # Clean up old entries if gallery is full
        if len(self.persons) >= self.max_gallery_size:
            self._cleanup_old_entries()
        
        global_id = self.next_global_id
        self.next_global_id += 1
        
        self.persons[global_id] = PersonRecord(
            global_id=global_id,
            features=[feature],
            last_seen=time.time(),
            cameras_seen={camera_name},
            total_detections=1
        )
        
        return global_id
    
    def _cleanup_old_entries(self, max_age_seconds: float = 3600):
        """Remove old entries from gallery."""
        current_time = time.time()
        to_remove = []
        
        for global_id, person in self.persons.items():
            if current_time - person.last_seen > max_age_seconds:
                to_remove.append(global_id)
        
        for global_id in to_remove[:len(self.persons) // 4]:  # Remove at most 25%
            del self.persons[global_id]
    
    def clear_camera_mappings(self, camera_name: str):
        """Clear local-to-global mappings for a camera (for new session)."""
        with self.lock:
            self.local_to_global[camera_name].clear()
    
    def get_statistics(self) -> dict:
        """Get gallery statistics."""
        return {
            'total_persons': len(self.persons),
            'next_global_id': self.next_global_id,
            'cameras': list(set(
                cam for p in self.persons.values() for cam in p.cameras_seen
            )),
            'cross_camera_persons': sum(
                1 for p in self.persons.values() if len(p.cameras_seen) > 1
            )
        }


class VideoSource:
    """
    Unified video source handler for both files and RTSP streams.
    """
    
    def __init__(self, source: Union[str, int], buffer_size: int = 2):
        """
        Initialize video source.
        
        Args:
            source: Video file path, RTSP URL, or camera index
            buffer_size: Frame buffer size for RTSP streams
        """
        self.source = source
        self.buffer_size = buffer_size
        self.cap = None
        self.is_stream = False
        self.frame_queue = None
        self.reader_thread = None
        self.running = False
        
        # Properties
        self.width = 0
        self.height = 0
        self.fps = 0
        self.total_frames = 0
    
    def open(self) -> bool:
        """Open the video source."""
        # Determine source type
        if isinstance(self.source, int):
            self.is_stream = True  # Webcam
        elif isinstance(self.source, str):
            self.is_stream = self.source.lower().startswith(('rtsp://', 'http://', 'https://'))
        
        # Open capture
        self.cap = cv2.VideoCapture(self.source)
        
        if not self.cap.isOpened():
            return False
        
        # Get properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # FPS fix: cv2.CAP_PROP_FPS is unreliable for RTSP streams — it often
        # returns 0, 90000, or other garbage values.  Clamp to a sane range
        # (1-60).  If still out of range, fall back to 25 FPS.
        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if raw_fps and 1 <= raw_fps <= 60:
            self.fps = raw_fps
        else:
            self.fps = 25  # safe default

        # For live streams the camera-reported FPS is used only as a hint;
        # we always write output at a safe, capped value (set in process_source).
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.total_frames <= 0:
            self.total_frames = float('inf')  # Stream
        
        # Start threaded reader for streams
        if self.is_stream:
            self.frame_queue = queue.Queue(maxsize=self.buffer_size)
            self.running = True
            self.reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
            self.reader_thread.start()
        
        return True
    
    def _reader_loop(self):
        """Background thread for reading stream frames."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue
            
            # Drop old frames if buffer full
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_queue.put(frame)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the source."""
        if self.is_stream:
            try:
                frame = self.frame_queue.get(timeout=2.0)
                return True, frame
            except queue.Empty:
                return False, None
        else:
            return self.cap.read()
    
    def release(self):
        """Release the video source."""
        self.running = False
        if self.reader_thread:
            self.reader_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
    
    def get_info(self) -> dict:
        """Get source information."""
        return {
            'source': str(self.source),
            'is_stream': self.is_stream,
            'width': self.width,
            'height': self.height,
            'fps': self.fps,
            'total_frames': self.total_frames if self.total_frames != float('inf') else 'Stream'
        }


class MultiCameraTracker:
    """
    Multi-Camera Person Tracking System.
    
    Tracks persons across multiple cameras while maintaining
    consistent global IDs.
    
    Features:
    - Unified video file and RTSP support
    - Global person gallery for cross-camera matching
    - Same ID maintained when person moves between cameras
    - Thread-safe for concurrent camera processing
    """
    
    def __init__(
        self, 
        reid_model_path: str,
        device: str = 'cuda',
        similarity_threshold: float = 0.7
    ):
        """
        Initialize multi-camera tracker.
        
        Args:
            reid_model_path: Path to trained Re-ID model
            device: 'cuda' or 'cpu'
            similarity_threshold: Threshold for cross-camera matching (0-1)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"🖥️  Device: {self.device}")
        
        # YOLO detector is now loaded per-camera in process_source
        # to isolate BoT-SORT tracking states between cameras.
        
        # Load Re-ID model
        print("🧠 Loading Re-ID model...")
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
        
        # Global person gallery (shared across all cameras)
        self.gallery = GlobalPersonGallery(similarity_threshold=similarity_threshold)
        
        # Lock for thread-safe Re-ID inference.
        # The Re-ID model is shared across camera threads; PyTorch eval-mode
        # forward() is generally thread-safe but we protect it with a lock
        # to prevent subtle race conditions on the same tensor buffers.
        self._reid_lock = threading.Lock()
        
        # Colors for visualization
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(10000, 3), dtype=np.uint8)
        
        print("✅ MultiCameraTracker initialized!")
    
    def extract_features(self, frame: np.ndarray, detections: np.ndarray) -> List[np.ndarray]:
        """Extract Re-ID features for detected persons."""
        features = []
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            
            # Validate coordinates
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
            
            # Lock around model inference so parallel camera threads don't race
            with self._reid_lock:
                with torch.no_grad():
                    feature = self.reid_model(crop_tensor)
            
            features.append(feature.cpu().numpy().flatten())
        
        return features

    def enhance_frame_lowlight(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance a dark / low-light frame before passing to YOLO.

        Two-stage pipeline:
        1. CLAHE on the L channel of LAB colour space:
           Boosts local contrast in dark regions (e.g. shadowed person against
           dark road) without over-brightening already-lit areas (street lamps).
        2. Gamma correction (gamma < 1.0 brightens shadows):
           Lifts overall dark-pixel intensity, making silhouettes more visible.

        The original frame is NOT modified; a new frame is returned.
        """
        # --- CLAHE on LAB L-channel ---
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        # clipLimit=3.0, tileGridSize=(8,8) are tuned for CCTV dark footage
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        enhanced = cv2.merge([l_ch, a_ch, b_ch])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

        # --- Gamma correction to lift dark shadows ---
        gamma = 0.6          # < 1.0 brightens; 0.6 is a strong lift for night scenes
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in range(256)
        ], dtype=np.uint8)
        enhanced = cv2.LUT(enhanced, table)

        return enhanced
    
    def process_frame(
        self, 
        frame: np.ndarray, 
        camera_name: str,
        local_tracker: YOLO
    ) -> Tuple[List[Tuple[int, np.ndarray]], np.ndarray]:
        """
        Process a single frame.
        
        Args:
            frame: BGR image
            camera_name: Name of the camera
            local_tracker: Camera-specific DeepSORT tracker
            
        Returns:
            tracks: List of (global_id, bbox) tuples
            annotated_frame: Frame with visualizations
        """
        # Low-light enhancement: apply CLAHE + gamma before YOLO so the model
        # can see people in shadows and dark street-light conditions.
        enhanced_frame = self.enhance_frame_lowlight(frame)

        # Track persons using built-in BoT-SORT
        # conf=0.25: lower than default 0.5 so dim/dark detections aren't discarded
        results = local_tracker.track(enhanced_frame, persist=True, tracker="botsort.yaml", classes=[0], conf=0.25, verbose=False)
        
        detections = []
        local_tracks = []
        
        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().tolist()
            
            for box, track_id in zip(boxes, track_ids):
                detections.append([box[0], box[1], box[2], box[3]])
                local_tracks.append({"track_id": track_id, "bbox": box})
                
        detections = np.array(detections) if detections else np.array([])
        
        if len(detections) == 0:
            return [], frame

        # Extract features from the ORIGINAL frame (not enhanced):
        # Re-ID model was trained on normal-exposure images; enhanced frame
        # may distort colour/texture features and hurt matching accuracy.
        features = self.extract_features(frame, detections)
        
        # Get all active local track IDs in this frame to prevent gallery collisions
        active_local_ids = {track_info["track_id"] for track_info in local_tracks}
        
        global_tracks = []
        for track_info, track_feature in zip(local_tracks, features):
            global_id = self.gallery.find_or_create_global_id(
                track_info["track_id"], track_feature, camera_name, active_local_ids
            )
            global_tracks.append((global_id, track_info["bbox"]))
        
        # Draw results on the enhanced frame so the visualisation shows the
        # brightness-corrected view that YOLO actually used for detection.
        annotated_frame = self.draw_tracks(enhanced_frame.copy(), global_tracks, camera_name)
        
        return global_tracks, annotated_frame
    
    def draw_tracks(
        self, 
        frame: np.ndarray, 
        tracks: List[Tuple[int, np.ndarray]], 
        camera_name: str
    ) -> np.ndarray:
        """Draw tracking results on frame."""
        for global_id, bbox in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            color = tuple(map(int, self.colors[global_id % len(self.colors)]))
            
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # ID label (show GLOBAL ID prominently)
            label = f'ID: {global_id}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            cv2.rectangle(
                frame, 
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0] + 10, y1), 
                color, -1
            )
            cv2.putText(
                frame, label, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )
        
        # Camera name label
        cv2.putText(
            frame, f'Camera: {camera_name}',
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        
        return frame
    
    def process_source(
        self,
        source: Union[str, int],
        camera_name: str = "Camera_1",
        output_path: Optional[str] = None,
        max_frames: Optional[int] = None,
        display: bool = False
    ) -> dict:
        """
        Process a single video source (file or RTSP).
        
        Args:
            source: Video file path, RTSP URL, or camera index
            camera_name: Name for this camera
            output_path: Path to save output video
            max_frames: Maximum frames to process
            display: Show live display (local only)
            
        Returns:
            Statistics dictionary
        """
        print(f"\n📹 Processing: {source}")
        print(f"   Camera name: {camera_name}")
        
        # Open source
        video_source = VideoSource(source)
        if not video_source.open():
            print(f"❌ Cannot open source: {source}")
            return {'error': 'Cannot open source'}
        
        info = video_source.get_info()
        print(f"   Resolution: {info['width']}x{info['height']}")
        print(f"   FPS: {info['fps']}")
        print(f"   Type: {'Stream' if info['is_stream'] else 'Video file'}")
        
        # Setup output writer
        out = None
        output_fps = 25  # default / stream fallback
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

            # For live streams, always use 25 FPS output.
            # For video files, honour the source FPS (already clamped 1-60).
            output_fps = 25 if info['is_stream'] else info['fps']

            # Use XVID + AVI: far more reliable on Windows than mp4v.
            # mp4v can produce unplayable files when the process ends early.
            # Automatically rewrite any .mp4 output path to .avi.
            avi_output_path = output_path
            if output_path.lower().endswith('.mp4'):
                avi_output_path = output_path[:-4] + '.avi'
                print(f"   ⚠️  Output changed to AVI for reliability: {avi_output_path}")

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(
                avi_output_path, fourcc, output_fps,
                (info['width'], info['height'])
            )
            if not out.isOpened():
                print("❌ VideoWriter failed to open. Output will not be saved.")
                out = None
            else:
                print(f"   Output: {avi_output_path}")
                print(f"   Output FPS: {output_fps}")
        
        # Initialize isolated YOLO BoT-SORT tracker for this camera stream.
        # yolov8s (small) is used instead of yolov8n (nano) — significantly
        # better detection in hard conditions (low light, small/occluded persons)
        # with only ~2× more compute; acceptable on a GPU.
        local_tracker = YOLO('yolov8s.pt')
        
        # Processing statistics
        stats = {
            'camera_name': camera_name,
            'source': str(source),
            'frames_processed': 0,
            'total_detections': 0,
            'unique_persons': set(),
            'start_time': time.time()
        }
        
        # Frame-duplication clock for live streams.
        #
        # WHY it's needed:
        #   VideoWriter has no concept of real time.  It encodes whatever frames
        #   you hand it at the declared output_fps.  If the CPU only processes
        #   2 frames/second but output_fps=25, each "real" second produces only
        #   2 file-frames → plays at 25/2 = 12.5× speed.
        #
        # FIX:
        #   After processing each frame, calculate how much real time has passed
        #   since the last written frame.  Write that frame N = round(elapsed × fps)
        #   times so the output file contains the correct number of frames per
        #   real second (duplicating the last frame during slow processing
        #   gives a slight freeze-frame artefact but correct playback speed).
        last_frame_real_time = time.time()   # wall-clock of last written frame
        MAX_DUP = int(output_fps * 5)        # safety cap: never > 5 seconds of dups

        # Determine total frames for progress bar
        total = max_frames if max_frames else (
            info['total_frames'] if info['total_frames'] != 'Stream' else None
        )
        
        pbar = tqdm(total=total, desc=f"Processing {camera_name}")
        
        print("\n🎬 Starting processing... (Ctrl+C to stop)")
        
        try:
            while True:
                # Check max frames
                if max_frames and stats['frames_processed'] >= max_frames:
                    break
                
                # Read frame
                ret, frame = video_source.read()
                if not ret:
                    if video_source.is_stream:
                        continue  # Keep trying for streams
                    else:
                        break  # End of video file
                
                # Process frame
                tracks, annotated_frame = self.process_frame(
                    frame, camera_name, local_tracker
                )
                
                # Update statistics
                stats['frames_processed'] += 1
                stats['total_detections'] += len(tracks)
                for global_id, _ in tracks:
                    stats['unique_persons'].add(global_id)
                
                # Add info overlay
                elapsed = time.time() - stats['start_time']
                fps = stats['frames_processed'] / elapsed if elapsed > 0 else 0
                
                info_lines = [
                    f"FPS: {fps:.1f}",
                    f"Frame: {stats['frames_processed']}",
                    f"Persons: {len(tracks)}",
                    f"Global IDs: {len(stats['unique_persons'])}"
                ]
                
                for i, line in enumerate(info_lines):
                    cv2.putText(
                        annotated_frame, line,
                        (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )
                
                # Timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    annotated_frame, timestamp,
                    (info['width'] - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                
                # Write output — frame-duplication for real-time playback
                if out:
                    now = time.time()
                    if info['is_stream']:
                        # How many output frames should this one real frame represent?
                        real_elapsed = now - last_frame_real_time
                        last_frame_real_time = now
                        dup_count = max(1, min(round(real_elapsed * output_fps), MAX_DUP))
                        for _ in range(dup_count):
                            out.write(annotated_frame)
                    else:
                        # Video file: write every processed frame (speed already correct)
                        out.write(annotated_frame)
                
                # Display (local only)
                if display:
                    cv2.imshow(camera_name, annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                pbar.update(1)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Stopped by user")
        
        finally:
            pbar.close()
            video_source.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        # Final statistics
        elapsed = time.time() - stats['start_time']
        stats['elapsed_time'] = elapsed
        stats['average_fps'] = stats['frames_processed'] / elapsed if elapsed > 0 else 0
        stats['unique_persons'] = len(stats['unique_persons'])
        
        print(f"\n📊 Statistics for {camera_name}:")
        print(f"   Frames: {stats['frames_processed']}")
        print(f"   Time: {elapsed:.1f}s")
        print(f"   Avg FPS: {stats['average_fps']:.1f}")
        print(f"   Unique persons: {stats['unique_persons']}")
        
        return stats
    
    def process_multi_camera(
        self,
        cameras: List[Dict],
        output_dir: str = "outputs",
        max_frames_per_camera: Optional[int] = None,
        sequential: bool = False   # Default: parallel (all cameras at once)
    ) -> List[dict]:
        """
        Process multiple cameras simultaneously using one thread per camera.

        Args:
            cameras: List of camera configurations
                     [{"name": "Entrance", "source": "rtsp://..."}, ...]
            output_dir: Directory for output videos
            max_frames_per_camera: Max frames per camera (None = run until stopped)
            sequential: False (default) = all cameras run in parallel.
                        True = one camera at a time (legacy mode).

        Returns:
            List of statistics for each camera
        """
        print(f"\n{'='*60}")
        print(f"📹 MULTI-CAMERA TRACKING")
        print(f"   Cameras: {len(cameras)}")
        print(f"   Mode: {'Sequential' if sequential else 'Parallel (all cameras simultaneously)'}")
        print(f"   Output: {output_dir}")
        print(f"{'='*60}")

        os.makedirs(output_dir, exist_ok=True)

        if sequential:
            # ── Legacy mode: one camera at a time ─────────────────────────
            all_stats = []
            for i, cam_config in enumerate(cameras):
                name = cam_config.get('name', f'Camera_{i+1}')
                source = cam_config.get('source')
                print(f"\n{'='*60}")
                print(f"📷 Camera {i+1}/{len(cameras)}: {name}")
                print(f"{'='*60}")
                output_path = os.path.join(output_dir, f"{name}.mp4")
                stats = self.process_source(
                    source=source,
                    camera_name=name,
                    output_path=output_path,
                    max_frames=max_frames_per_camera
                )
                all_stats.append(stats)
        else:
            # ── Parallel mode: all cameras run simultaneously ──────────────
            # Each camera gets its own thread that calls process_source().
            # The shared GlobalPersonGallery and reid_model are already
            # thread-safe (Gallery has a Lock; ReID uses self._reid_lock).

            all_stats = [None] * len(cameras)   # pre-allocate result slots
            stop_event = threading.Event()       # signal all threads to stop

            def _camera_worker(index: int, cam_config: dict):
                """Thread target: process one camera stream."""
                name = cam_config.get('name', f'Camera_{index+1}')
                source = cam_config.get('source')
                output_path = os.path.join(output_dir, f"{name}.mp4")
                print(f"\n🚀 Starting camera thread: {name}")
                try:
                    stats = self.process_source(
                        source=source,
                        camera_name=name,
                        output_path=output_path,
                        max_frames=max_frames_per_camera
                    )
                except Exception as exc:
                    print(f"\n❌ Camera {name} thread error: {exc}")
                    stats = {'error': str(exc), 'camera_name': name}
                all_stats[index] = stats
                print(f"\n✅ Camera thread finished: {name}")

            # Launch one thread per camera
            threads = []
            for i, cam_config in enumerate(cameras):
                t = threading.Thread(
                    target=_camera_worker,
                    args=(i, cam_config),
                    daemon=True,
                    name=f"cam-{cam_config.get('name', i)}"
                )
                threads.append(t)
                t.start()

            print(f"\n🎬 All {len(cameras)} camera threads launched. Press Ctrl+C to stop all.")

            try:
                # Wait for all threads to finish
                for t in threads:
                    while t.is_alive():
                        t.join(timeout=1.0)   # check every second so Ctrl+C works
            except KeyboardInterrupt:
                print("\n\n🛑 Ctrl+C — stopping all camera threads...")
                # process_source() catches KeyboardInterrupt internally;
                # threads will finish their current frame and exit cleanly.
                for t in threads:
                    t.join(timeout=10.0)
                print("✅ All threads stopped.")

        # ── Summary ────────────────────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("📊 MULTI-CAMERA SUMMARY")
        print(f"{'='*60}")
        
        gallery_stats = self.gallery.get_statistics()
        print(f"\n🌐 Global Gallery Statistics:")
        print(f"   Total unique persons: {gallery_stats['total_persons']}")
        print(f"   Cross-camera persons: {gallery_stats['cross_camera_persons']}")
        print(f"   Cameras processed: {gallery_stats['cameras']}")
        
        print(f"\n📹 Per-Camera Statistics:")
        for stats in all_stats:
            if 'error' not in stats:
                print(f"   {stats['camera_name']}:")
                print(f"      Frames: {stats['frames_processed']}")
                print(f"      Unique persons: {stats['unique_persons']}")
                print(f"      Avg FPS: {stats['average_fps']:.1f}")
        
        return all_stats
    
    def reset_gallery(self):
        """Reset the global person gallery (for new session)."""
        self.gallery = GlobalPersonGallery(
            similarity_threshold=self.gallery.similarity_threshold
        )
        print("✅ Global gallery reset")
    
    def get_gallery_info(self) -> dict:
        """Get global gallery information."""
        return self.gallery.get_statistics()


def main():
    """Demo usage"""
    print("=" * 60)
    print("🎯 MULTI-CAMERA PERSON TRACKING SYSTEM")
    print("=" * 60)
    
    PROJECT_ROOT = '/content/drive/MyDrive/persona_detection_final'
    reid_model_path = f'{PROJECT_ROOT}/phase2_reid/checkpoints/best_reid_model.pth'
    
    # Check model exists
    if not os.path.exists(reid_model_path):
        print(f"❌ Model not found: {reid_model_path}")
        print("   Run download_model.py first")
        return
    
    # Initialize tracker
    tracker = MultiCameraTracker(reid_model_path)
    
    print("\n✅ Tracker ready!")
    print("\n" + "=" * 60)
    print("📋 USAGE EXAMPLES")
    print("=" * 60)
    
    print("""
# Process single video file:
tracker.process_source(
    source="video.mp4",
    camera_name="Entrance",
    output_path="output.mp4"
)

# Process RTSP stream:
tracker.process_source(
    source="rtsp://user:pass@192.168.1.100:554/stream",
    camera_name="Lobby",
    output_path="lobby_output.mp4",
    max_frames=500
)

# Process multiple cameras (same person gets same ID!):
tracker.process_multi_camera([
    {"name": "Entrance", "source": "video1.mp4"},
    {"name": "Lobby", "source": "video2.mp4"},
    {"name": "Exit", "source": "video3.mp4"},
], output_dir="outputs/")

# Check global gallery:
print(tracker.get_gallery_info())
    """)
    
    return tracker


if __name__ == '__main__':
    main()
