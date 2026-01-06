#!/usr/bin/env python3
"""
Persona Detection System - Demo Script

Run person tracking on a video file.

Usage:
    python demo.py --input video.mp4 --output result.mp4
    python demo.py -i video.mp4 -o result.mp4 --max-frames 100
"""

import argparse
import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description='Persona Detection - Person Tracking Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python demo.py --input pedestrians.mp4 --output tracked.mp4
    python demo.py -i video.mp4 -o out.mp4 --max-frames 200
    python demo.py -i video.mp4 -o out.mp4 --device cpu
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Input video path'
    )
    parser.add_argument(
        '--output', '-o',
        default='output.mp4',
        help='Output video path (default: output.mp4)'
    )
    parser.add_argument(
        '--model', '-m',
        default='phase2_reid/checkpoints/best_reid_model.pth',
        help='Path to Re-ID model'
    )
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Maximum frames to process (default: all)'
    )
    parser.add_argument(
        '--device',
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"❌ Error: Input video not found: {args.input}")
        sys.exit(1)
    
    # Validate model
    if not os.path.exists(args.model):
        print(f"❌ Error: Model not found: {args.model}")
        print("\nPlease download the model from GitHub releases:")
        print("  https://github.com/YOUR_USERNAME/persona-detection/releases")
        sys.exit(1)
    
    # Import and run tracker
    print("🚀 Initializing Persona Detection System...")
    
    from phase3_tracking.person_tracker import PersonTracker
    
    tracker = PersonTracker(args.model, device=args.device)
    
    print(f"\n🎬 Processing: {args.input}")
    tracker.process_video(args.input, args.output, max_frames=args.max_frames)
    
    print(f"\n✅ Done! Output saved to: {args.output}")


if __name__ == '__main__':
    main()
