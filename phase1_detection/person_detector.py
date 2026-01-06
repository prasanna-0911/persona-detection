"""
Person Detection Module using YOLOv5

This module provides a wrapper around YOLOv5 for detecting persons
in images and videos. It serves as Phase 1 of the Persona Detection System.

Features:
- Real-time person detection (30-50 FPS on GPU)
- Confidence filtering
- Batch processing support
- Easy integration with tracking pipeline

Usage:
    detector = PersonDetector()
    detections = detector.detect(image)
    detector.process_video('input.mp4', 'output.mp4')
"""

import cv2
import numpy as np
from tqdm import tqdm


class PersonDetector:
    """
    YOLOv5-based person detector.
    
    Detects persons in images/videos using pretrained YOLOv5 model.
    Only returns detections for class 0 (person) from COCO dataset.
    
    Args:
        model_name: YOLOv5 model variant ('yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
        confidence: Minimum confidence threshold (0-1)
        device: 'cuda' or 'cpu'
    
    Example:
        >>> detector = PersonDetector(confidence=0.5)
        >>> detections = detector.detect(image)
        >>> for det in detections:
        ...     x1, y1, x2, y2, conf = det
        ...     print(f"Person at ({x1}, {y1}) to ({x2}, {y2}) with {conf:.2f} confidence")
    """
    
    def __init__(self, model_name='yolov5su', confidence=0.5, device='cuda'):
        """Initialize the person detector."""
        import torch
        
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.confidence = confidence
        
        print(f"🔍 Initializing PersonDetector...")
        print(f"   Device: {self.device}")
        print(f"   Model: {model_name}")
        print(f"   Confidence threshold: {confidence}")
        
        # Load YOLOv5 model using ultralytics
        try:
            from ultralytics import YOLO
            self.model = YOLO(f'{model_name}.pt')
            self.use_ultralytics = True
            print("   Using: Ultralytics YOLO")
        except ImportError:
            # Fallback to torch hub
            import torch
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            self.model.classes = [0]  # Only detect persons
            self.model.conf = confidence
            self.use_ultralytics = False
            print("   Using: Torch Hub YOLO")
        
        print("✅ PersonDetector initialized!")
    
    def detect(self, image):
        """
        Detect persons in an image.
        
        Args:
            image: BGR image as numpy array (OpenCV format)
                   or path to image file
        
        Returns:
            numpy array of detections, each row: [x1, y1, x2, y2, confidence]
            Empty array if no persons detected
        
        Example:
            >>> image = cv2.imread('photo.jpg')
            >>> detections = detector.detect(image)
            >>> print(f"Found {len(detections)} persons")
        """
        # Load image if path provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Cannot load image: {image}")
        
        if self.use_ultralytics:
            # Ultralytics YOLO
            results = self.model(
                image, 
                classes=[0],  # Only persons
                conf=self.confidence,
                verbose=False
            )
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        detections.append([x1, y1, x2, y2, float(conf)])
            
            return np.array(detections) if detections else np.array([])
        
        else:
            # Torch Hub YOLO
            results = self.model(image)
            detections = results.xyxy[0].cpu().numpy()
            
            # Filter for persons only (class 0)
            person_detections = detections[detections[:, 5] == 0]
            
            # Return [x1, y1, x2, y2, confidence]
            return person_detections[:, :5] if len(person_detections) > 0 else np.array([])
    
    def detect_batch(self, images):
        """
        Detect persons in multiple images.
        
        Args:
            images: List of BGR images as numpy arrays
        
        Returns:
            List of detection arrays, one per image
        
        Example:
            >>> images = [cv2.imread(f'frame_{i}.jpg') for i in range(10)]
            >>> all_detections = detector.detect_batch(images)
        """
        all_detections = []
        
        for image in images:
            detections = self.detect(image)
            all_detections.append(detections)
        
        return all_detections
    
    def draw_detections(self, image, detections, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes on image.
        
        Args:
            image: BGR image as numpy array
            detections: Array of [x1, y1, x2, y2, confidence] detections
            color: BGR color tuple for bounding boxes
            thickness: Line thickness
        
        Returns:
            Image with drawn bounding boxes
        
        Example:
            >>> detections = detector.detect(image)
            >>> annotated = detector.draw_detections(image, detections)
            >>> cv2.imwrite('result.jpg', annotated)
        """
        image_copy = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            conf = det[4] if len(det) > 4 else 1.0
            
            # Draw bounding box
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw confidence label
            label = f'Person: {conf:.2f}'
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Label background
            cv2.rectangle(
                image_copy,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color, -1
            )
            
            # Label text
            cv2.putText(
                image_copy, label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        return image_copy
    
    def process_video(self, input_path, output_path, max_frames=None, show_progress=True):
        """
        Process a video file and save with detections drawn.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            max_frames: Maximum frames to process (None for all)
            show_progress: Show progress bar
        
        Returns:
            dict with processing statistics
        
        Example:
            >>> stats = detector.process_video('input.mp4', 'output.mp4')
            >>> print(f"Processed {stats['frames']} frames, found {stats['total_detections']} persons")
        """
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        print(f"📹 Processing video: {input_path}")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Frames: {total_frames}")
        
        # Setup output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        total_detections = 0
        
        if show_progress:
            pbar = tqdm(total=total_frames, desc="Processing")
        
        while cap.isOpened():
            if max_frames and frame_count >= max_frames:
                break
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect persons
            detections = self.detect(frame)
            total_detections += len(detections)
            
            # Draw detections
            frame = self.draw_detections(frame, detections)
            
            # Add frame info
            info = f'Frame: {frame_count} | Persons: {len(detections)}'
            cv2.putText(
                frame, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
            )
            
            # Write frame
            out.write(frame)
            
            frame_count += 1
            
            if show_progress:
                pbar.update(1)
        
        if show_progress:
            pbar.close()
        
        cap.release()
        out.release()
        
        stats = {
            'frames': frame_count,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / max(frame_count, 1),
            'output_path': output_path
        }
        
        print(f"\n✅ Video saved to: {output_path}")
        print(f"   Total persons detected: {total_detections}")
        print(f"   Average per frame: {stats['avg_detections_per_frame']:.1f}")
        
        return stats
    
    def get_person_crops(self, image, detections, min_size=50):
        """
        Extract cropped images of detected persons.
        
        Args:
            image: BGR image as numpy array
            detections: Array of detections
            min_size: Minimum crop size (skip smaller detections)
        
        Returns:
            List of (crop, bbox) tuples
        
        Example:
            >>> detections = detector.detect(image)
            >>> crops = detector.get_person_crops(image, detections)
            >>> for crop, bbox in crops:
            ...     cv2.imwrite(f'person_{bbox[0]}.jpg', crop)
        """
        crops = []
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            
            # Validate coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            # Check minimum size
            if (x2 - x1) < min_size or (y2 - y1) < min_size:
                continue
            
            crop = image[y1:y2, x1:x2].copy()
            crops.append((crop, (x1, y1, x2, y2)))
        
        return crops
    
    def set_confidence(self, confidence):
        """
        Update confidence threshold.
        
        Args:
            confidence: New threshold (0-1)
        """
        self.confidence = confidence
        
        if not self.use_ultralytics:
            self.model.conf = confidence
        
        print(f"✅ Confidence threshold updated to: {confidence}")


class PersonDetectorLite:
    """
    Lightweight person detector for edge devices.
    
    Uses YOLOv5 nano model for faster inference on CPU.
    """
    
    def __init__(self, confidence=0.5):
        from ultralytics import YOLO
        
        self.model = YOLO('yolov5nu.pt')  # Nano model
        self.confidence = confidence
        
        print("✅ PersonDetectorLite initialized (using YOLOv5 Nano)")
    
    def detect(self, image):
        """Detect persons in image."""
        results = self.model(
            image,
            classes=[0],
            conf=self.confidence,
            verbose=False
        )
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections.append([x1, y1, x2, y2, float(conf)])
        
        return np.array(detections) if detections else np.array([])


def benchmark_detector(detector, num_frames=100):
    """
    Benchmark detector performance.
    
    Args:
        detector: PersonDetector instance
        num_frames: Number of frames to test
    
    Returns:
        dict with benchmark results
    """
    import time
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print(f"🏃 Benchmarking detector on {num_frames} frames...")
    
    # Warm up
    for _ in range(5):
        detector.detect(dummy_image)
    
    # Benchmark
    start_time = time.time()
    
    for _ in tqdm(range(num_frames), desc="Benchmarking"):
        detector.detect(dummy_image)
    
    elapsed = time.time() - start_time
    fps = num_frames / elapsed
    
    results = {
        'frames': num_frames,
        'total_time': elapsed,
        'fps': fps,
        'ms_per_frame': (elapsed / num_frames) * 1000
    }
    
    print(f"\n📊 Benchmark Results:")
    print(f"   FPS: {fps:.1f}")
    print(f"   Time per frame: {results['ms_per_frame']:.1f} ms")
    
    return results


# Main entry point for testing
def main():
    """Test the person detector."""
    print("=" * 60)
    print("🔍 PERSON DETECTOR TEST")
    print("=" * 60)
    
    # Initialize detector
    detector = PersonDetector(confidence=0.5)
    
    # Create test image (random)
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Run detection
    print("\n📸 Running detection on test image...")
    detections = detector.detect(test_image)
    
    print(f"   Detected {len(detections)} persons")
    
    # Benchmark
    print("\n" + "=" * 60)
    benchmark_detector(detector, num_frames=50)
    
    print("\n✅ Person detector is working!")
    
    return detector


if __name__ == '__main__':
    main()
