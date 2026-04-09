#!/usr/bin/env python3
"""
Persona Detection System - Unified Tracker

Supports both video files and RTSP live streams with
consistent person IDs across multiple cameras.

Usage:
    # Single video file
    python run_tracker.py --source video.mp4 --output result.mp4

    # Single RTSP stream
    python run_tracker.py --source rtsp://user:pass@ip:554/stream --output live.mp4

    # Multiple cameras (same person ID across cameras)
    python run_tracker.py --multi-camera --config cameras.json --output-dir outputs/

    # Webcam
    python run_tracker.py --webcam --output webcam.mp4
"""

import argparse
import os
import sys
import json

# Fix OpenMP error: "Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
# This is a common conflict between PyTorch and other libraries (like YOLO/Intel MKL) on Windows.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description='Persona Detection - Multi-Camera Person Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Video file
    python run_tracker.py --source video.mp4 --output tracked.mp4
    
    # RTSP stream
    python run_tracker.py --source rtsp://user:pass@192.168.1.100:554/stream --output live.mp4
    
    # Webcam
    python run_tracker.py --webcam --output webcam.mp4
    
    # Multiple cameras with same person IDs
    python run_tracker.py --multi-camera --config cameras.json --output-dir outputs/
        """
    )
    
    # Source options
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--source', '-s', type=str, help='Video file or RTSP URL')
    source_group.add_argument('--webcam', action='store_true', help='Use webcam')
    source_group.add_argument('--multi-camera', action='store_true', help='Multi-camera mode')
    
    # Output options
    parser.add_argument('--output', '-o', type=str, default='output.mp4', help='Output video path')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory for multi-camera')
    
    # Multi-camera options
    parser.add_argument('--config', type=str, help='Camera config JSON file')
    
    # Processing options
    parser.add_argument('--max-frames', type=int, default=None, help='Max frames to process')
    parser.add_argument('--camera-name', type=str, default='Camera_1', help='Camera name')
    parser.add_argument('--similarity-threshold', type=float, default=0.70,
                       help='Cross-camera Re-ID matching threshold (0-1). '
                            'OSNet recommended range: 0.65-0.75. '
                            'Higher = stricter (fewer false matches). '
                            'Lower = more lenient (more cross-camera matches but risk of wrong ID). '
                            'Default 0.70 is the sweet spot for OSNet on CCTV footage.')
    parser.add_argument('--model', type=str,
                       default='osnet_x1_0',
                       help='Re-ID model: OSNet name (osnet_x1_0 / osnet_x0_75 / osnet_x0_5) '
                            'or path to custom .pth file. '
                            'OSNet requires: pip install git+https://github.com/KaiyangZhou/deep-person-reid.git')

    # Detection model options
    parser.add_argument('--yolo-model', type=str,
                       default='yolov8s.pt',
                       help='YOLO model to use (default: yolov8s.pt)')
    parser.add_argument('--day-model', type=str,
                       default='yolov8s.pt',
                       help='YOLO model for daytime/bright scenes (default: yolov8s.pt)')
    parser.add_argument('--night-model', type=str,
                       default='runs/detect/yolov8s_rot0/weights/best.pt',
                       help='YOLO model for nighttime/dark scenes (default: fine-tuned best.pt)')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    # Detection tuning options
    parser.add_argument('--conf', type=float, default=0.45,
                       help='YOLO confidence threshold (0.0-1.0). Lower = more detections.')
    parser.add_argument('--imgsz', type=int, default=960,
                       help='YOLO inference image size. 960 recommended for nighttime.')
    parser.add_argument('--botsort-cfg', type=str, default='botsort_custom.yaml',
                       help='BotSort tracker config YAML. Use botsort_custom.yaml (tuned) '
                            'or botsort.yaml (default Ultralytics). '
                            'botsort_custom.yaml reduces ID swaps at pedestrian crossings.')
    
    args = parser.parse_args()
    
    # Check model exists (skip check for known OSNet model names — they auto-download)
    _OSNET_NAMES = ('osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x1_4')
    if args.model not in _OSNET_NAMES and not os.path.exists(args.model):
        print(f"\u274c Re-ID model not found: {args.model}")
        print("\nOptions:")
        print("  1. Use OSNet (recommended):  --model osnet_x1_0")
        print("  2. Use custom model:         --model phase2_reid/checkpoints/best_reid_model.pth")
        sys.exit(1)
    
    # Import tracker
    from multi_camera_tracker import MultiCameraTracker
    
    # Initialize tracker
    print("🚀 Initializing Multi-Camera Tracker...")
    print(f"   YOLO model : {args.yolo_model}")
    print(f"   Day  model : {args.day_model}")
    print(f"   Night model: {args.night_model}")
    print(f"   Conf       : {args.conf}")
    print(f"   Img size   : {args.imgsz}")
    print(f"   BotSort cfg: {args.botsort_cfg}")
    tracker = MultiCameraTracker(
        reid_model_path=args.model,
        device=args.device,
        similarity_threshold=args.similarity_threshold,
        yolo_model_path=args.yolo_model,
        day_model_path=args.day_model,
        night_model_path=args.night_model,
        conf=args.conf,
        imgsz=args.imgsz,
        botsort_cfg=args.botsort_cfg
    )
    
    # Process based on mode
    if args.webcam:
        # Webcam mode
        tracker.process_source(
            source=0,
            camera_name='Webcam',
            output_path=args.output,
            max_frames=args.max_frames
        )
    
    elif args.multi_camera:
        # Multi-camera mode
        if not args.config:
            # Default example configuration
            cameras = [
                {"name": "Camera_1", "source": "video1.mp4"},
                {"name": "Camera_2", "source": "video2.mp4"},
            ]
            print("⚠️ No --config provided. Using example configuration.")
            print("   Create cameras.json with your camera sources.")
        else:
            # Load from config file
            with open(args.config, 'r') as f:
                cameras = json.load(f)
        
        tracker.process_multi_camera(
            cameras=cameras,
            output_dir=args.output_dir,
            max_frames_per_camera=args.max_frames
        )
    
    else:
        # Single source mode (video or RTSP)
        tracker.process_source(
            source=args.source,
            camera_name=args.camera_name,
            output_path=args.output,
            max_frames=args.max_frames
        )
    
    # Print gallery info
    print("\n📊 Global Person Gallery:")
    info = tracker.get_gallery_info()
    print(f"   Total unique persons: {info['total_persons']}")
    print(f"   Cross-camera persons: {info['cross_camera_persons']}")


if __name__ == '__main__':
    main()
