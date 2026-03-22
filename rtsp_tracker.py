"""
RTSP Live Stream Person Tracker

Enables real-time person tracking from IP cameras and RTSP streams.

Features:
- Live RTSP stream processing
- Multi-camera support
- Automatic reconnection
- Frame buffering for stability
- Real-time display (local) or recording

Usage:
    # Single camera
    tracker = RTSPTracker(reid_model_path)
    tracker.run("rtsp://user:pass@192.168.1.100:554/stream")
    
    # Multiple cameras
    tracker.run_multi([
        "rtsp://user:pass@192.168.1.100:554/stream",
        "rtsp://user:pass@192.168.1.101:554/stream",
    ])
"""

import os
import sys
import cv2
import time
import threading
import queue
import numpy as np
from datetime import datetime
from collections import deque

import torch
from torchvision import transforms
from PIL import Image

# Bug #1 fix: Use dynamic path resolution instead of hardcoded Colab path.
# The old code used '/content/drive/MyDrive/persona_detection_final' which
# only works in Google Colab and crashes on Windows/Linux local environments.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'phase2_reid'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'phase3_tracking'))

from models.reid_net import ReIDNetwork
from trackers import DeepSORTTracker


class RTSPStreamReader:
    """
    Threaded RTSP stream reader for stable frame capture.
    
    Runs in separate thread to prevent blocking and frame drops.
    """
    
    def __init__(self, rtsp_url, buffer_size=2):
        """
        Initialize stream reader.
        
        Args:
            rtsp_url: RTSP URL or camera index
            buffer_size: Number of frames to buffer
        """
        self.rtsp_url = rtsp_url
        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        
        self.cap = None
        self.running = False
        self.thread = None
        
        self.fps = 0
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # Stream info
        self.width = 0
        self.height = 0
        self.stream_fps = 0
    
    def start(self):
        """Start the stream reader thread."""
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        if not self.cap.isOpened():
            raise ConnectionError(f"Cannot connect to: {self.rtsp_url}")
        
        # Get stream properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # FPS fix: clamp to sane range (1-60), default 25 if unreliable
        raw_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.stream_fps = raw_fps if raw_fps and 1 <= raw_fps <= 60 else 25
        
        # Set buffer size for RTSP
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.running = True
        self.thread = threading.Thread(target=self._reader_thread, daemon=True)
        self.thread.start()
        
        print(f"✅ Stream connected: {self.width}x{self.height} @ {self.stream_fps}fps")
        
        return self
    
    def _reader_thread(self):
        """Background thread to continuously read frames."""
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("⚠️ Stream read failed, attempting reconnect...")
                self._reconnect()
                continue
            
            # Update FPS calculation
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_frame_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_frame_time = current_time
            
            # Add frame to queue (drop old frames if full)
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_queue.put(frame)
    
    def _reconnect(self, max_attempts=5):
        """Attempt to reconnect to stream."""
        for attempt in range(max_attempts):
            print(f"   Reconnect attempt {attempt + 1}/{max_attempts}...")
            
            if self.cap is not None:
                self.cap.release()
            
            time.sleep(2)
            self.cap = cv2.VideoCapture(self.rtsp_url)
            
            if self.cap.isOpened():
                print("✅ Reconnected successfully!")
                return True
        
        print("❌ Failed to reconnect after max attempts")
        self.running = False
        return False
    
    def read(self):
        """
        Read the latest frame.
        
        Returns:
            success: True if frame available
            frame: The frame (or None if not available)
        """
        try:
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            return False, None
    
    def stop(self):
        """Stop the stream reader."""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
        print("🛑 Stream reader stopped")
    
    def get_info(self):
        """Get stream information."""
        return {
            'url': self.rtsp_url,
            'width': self.width,
            'height': self.height,
            'stream_fps': self.stream_fps,
            'current_fps': self.fps
        }


class RTSPTracker:
    """
    Real-time person tracking from RTSP streams.
    
    Combines YOLOv5 detection, Re-ID features, and DeepSORT tracking
    for live video streams.
    """
    
    def __init__(self, reid_model_path, device='cuda'):
        """
        Initialize RTSP tracker.
        
        Args:
            reid_model_path: Path to trained Re-ID model
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(
            device if torch.cuda.is_available() else 'cpu'
        )
        print(f"🖥️ Using device: {self.device}")
        
        # Load YOLOv5
        print("🔍 Loading YOLOv5...")
        from ultralytics import YOLO
        self.detector = YOLO('yolov5su.pt')
        
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
        
        # Tracking colors
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(1000, 3), dtype=np.uint8)
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'total_detections': 0,
            'start_time': None
        }
        
        print("✅ RTSPTracker initialized!")
    
    def extract_reid_features(self, frame, detections):
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
            
            with torch.no_grad():
                feature = self.reid_model(crop_tensor)
            
            features.append(feature.cpu().numpy().flatten())
        
        return features
    
    def process_frame(self, frame, tracker):
        """Process a single frame."""
        # Detect persons
        results = self.detector(frame, classes=[0], conf=0.5, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections.append([x1, y1, x2, y2, float(conf)])
        
        detections = np.array(detections) if detections else np.array([])
        
        if len(detections) == 0:
            return [], frame
        
        # Extract features
        features = self.extract_reid_features(frame, detections)
        
        # Update tracker — returns (track_id, bbox, ema_feature) triples (Bug #2 fix)
        raw_tracks = tracker.update(detections, features)
        tracks = [(tid, bbox) for tid, bbox, _ in raw_tracks]
        
        # Draw results
        frame = self.draw_tracks(frame, tracks)
        
        return tracks, frame
    
    def draw_tracks(self, frame, tracks):
        """Draw tracking results on frame."""
        for track_id, bbox in tracks:
            x1, y1, x2, y2 = map(int, bbox)
            color = tuple(map(int, self.colors[track_id % len(self.colors)]))
            
            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # ID label
            label = f'ID: {track_id}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10),
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def run(self, rtsp_url, output_path=None, display=False, max_frames=None):
        """
        Run tracking on RTSP stream.
        
        Args:
            rtsp_url: RTSP URL or camera index (0 for webcam)
            output_path: Path to save output video (optional)
            display: Show live display (only works locally, not in Colab)
            max_frames: Maximum frames to process (None for infinite)
        
        Returns:
            Statistics dictionary
        """
        print(f"\n📹 Connecting to: {rtsp_url}")
        
        # Initialize stream reader
        stream = RTSPStreamReader(rtsp_url)
        stream.start()
        
        # Initialize tracker (new tracker for each stream)
        tracker = DeepSORTTracker(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7,
            max_cosine_distance=0.3
        )
        
        # Setup output video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path, fourcc, 25,
                (stream.width, stream.height)
            )
            print(f"📁 Recording to: {output_path}")
        
        # Statistics
        self.stats['start_time'] = time.time()
        self.stats['frames_processed'] = 0
        self.stats['total_detections'] = 0
        
        fps_counter = deque(maxlen=30)
        last_time = time.time()
        
        print("\n🎬 Starting tracking... (Press Ctrl+C to stop)")
        print("-" * 50)
        
        try:
            while True:
                # Check max frames
                if max_frames and self.stats['frames_processed'] >= max_frames:
                    print(f"\n✅ Reached max frames: {max_frames}")
                    break
                
                # Read frame
                ret, frame = stream.read()
                if not ret:
                    continue
                
                # Process frame
                frame_start = time.time()
                tracks, annotated_frame = self.process_frame(frame, tracker)
                process_time = time.time() - frame_start
                
                # Calculate FPS
                current_time = time.time()
                fps_counter.append(1.0 / (current_time - last_time))
                last_time = current_time
                current_fps = sum(fps_counter) / len(fps_counter)
                
                # Update stats
                self.stats['frames_processed'] += 1
                self.stats['total_detections'] += len(tracks)
                
                # Add overlay info
                info_text = [
                    f"FPS: {current_fps:.1f}",
                    f"Tracks: {len(tracks)}",
                    f"Frame: {self.stats['frames_processed']}"
                ]
                
                for i, text in enumerate(info_text):
                    cv2.putText(
                        annotated_frame, text,
                        (10, 30 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                    )
                
                # Add timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(
                    annotated_frame, timestamp,
                    (stream.width - 200, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                )
                
                # Write to output
                if out:
                    out.write(annotated_frame)
                
                # Display (only works locally)
                if display:
                    cv2.imshow('RTSP Tracker', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Print status every 100 frames
                if self.stats['frames_processed'] % 100 == 0:
                    elapsed = time.time() - self.stats['start_time']
                    print(f"   Processed: {self.stats['frames_processed']} frames | "
                          f"FPS: {current_fps:.1f} | "
                          f"Tracks: {len(tracks)} | "
                          f"Time: {elapsed:.1f}s")
        
        except KeyboardInterrupt:
            print("\n\n🛑 Stopped by user")
        
        finally:
            # Cleanup
            stream.stop()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        # Final statistics
        elapsed = time.time() - self.stats['start_time']
        avg_fps = self.stats['frames_processed'] / elapsed if elapsed > 0 else 0
        
        self.stats['elapsed_time'] = elapsed
        self.stats['average_fps'] = avg_fps
        
        print("\n" + "=" * 50)
        print("📊 SESSION STATISTICS")
        print("=" * 50)
        print(f"   Frames Processed: {self.stats['frames_processed']}")
        print(f"   Total Time: {elapsed:.1f} seconds")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Total Detections: {self.stats['total_detections']}")
        if output_path:
            print(f"   Output saved to: {output_path}")
        
        return self.stats
    
    def run_multi(self, rtsp_urls, output_dir=None, grid_size=None):
        """
        Run tracking on multiple RTSP streams simultaneously.
        
        Args:
            rtsp_urls: List of RTSP URLs
            output_dir: Directory to save outputs
            grid_size: Tuple (rows, cols) for display grid
        
        Returns:
            Statistics for all streams
        """
        print(f"\n📹 Multi-camera tracking: {len(rtsp_urls)} cameras")
        
        # This would require threading for true simultaneous processing
        # For now, process sequentially
        all_stats = []
        
        for i, url in enumerate(rtsp_urls):
            print(f"\n{'='*50}")
            print(f"📷 Camera {i+1}/{len(rtsp_urls)}")
            
            output_path = None
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = f"{output_dir}/camera_{i+1}.mp4"
            
            stats = self.run(url, output_path=output_path, max_frames=500)
            all_stats.append(stats)
        
        return all_stats


def test_rtsp_connection(rtsp_url):
    """Test RTSP connection without processing."""
    print(f"\n🔍 Testing connection to: {rtsp_url}")
    
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("❌ Connection FAILED")
        return False
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        print("✅ Connection SUCCESSFUL")
        print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
        return True
    else:
        print("❌ Could not read frame")
        return False


# Main entry point
def main():
    """Demo usage"""
    print("=" * 60)
    print("📹 RTSP PERSON TRACKER")
    print("=" * 60)
    
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    reid_model_path = os.path.join(PROJECT_ROOT, 'phase2_reid', 'checkpoints', 'best_reid_model.pth')
    
    # Initialize tracker
    tracker = RTSPTracker(reid_model_path)
    
    print("\n✅ Tracker ready!")
    print("\nUsage examples:")
    print("  # Test connection")
    print("  test_rtsp_connection('rtsp://user:pass@192.168.1.100:554/stream')")
    print("\n  # Run tracking")
    print("  tracker.run('rtsp://user:pass@192.168.1.100:554/stream', output_path='output.mp4')")
    print("\n  # Webcam (if available)")
    print("  tracker.run(0, output_path='webcam_output.mp4')")
    
    return tracker


if __name__ == '__main__':
    main()
