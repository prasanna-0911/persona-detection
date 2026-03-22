#!/usr/bin/env python3
"""
Download Pre-trained Model

This script downloads the pre-trained Re-ID model required for tracking.

Usage:
    python download_model.py
"""

import os
import sys
import urllib.request
from pathlib import Path


def download_from_github_release():
    """Download model from GitHub releases."""
    
    # Update this URL after creating the release
    MODEL_URL = "https://github.com/prasanna-0911/persona-detection/releases/download/v1.0.0/best_reid_model.pth"
    
    output_dir = Path("phase2_reid/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "best_reid_model.pth"
    
    if output_path.exists():
        print(f"✅ Model already exists: {output_path}")
        return str(output_path)
    
    print("📥 Downloading pre-trained model...")
    print(f"   URL: {MODEL_URL}")
    print(f"   Destination: {output_path}")
    
    try:
        urllib.request.urlretrieve(MODEL_URL, output_path)
        print(f"✅ Download complete!")
        return str(output_path)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nPlease download manually from:")
        print("  https://github.com/prasanna-0911/persona-detection/releases")
        return None


def download_from_gdrive(file_id):
    """Download model from Google Drive."""
    try:
        import gdown
    except ImportError:
        print("Installing gdown...")
        os.system("pip install gdown -q")
        import gdown
    
    output_dir = Path("phase2_reid/checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "best_reid_model.pth"
    
    if output_path.exists():
        print(f"✅ Model already exists: {output_path}")
        return str(output_path)
    
    url = f"https://drive.google.com/uc?id={file_id}"
    
    print("📥 Downloading from Google Drive...")
    gdown.download(url, str(output_path), quiet=False)
    
    if output_path.exists():
        print(f"✅ Download complete!")
        return str(output_path)
    else:
        print("❌ Download failed!")
        return None


def main():
    print("=" * 60)
    print("📥 PERSONA DETECTION - MODEL DOWNLOADER")
    print("=" * 60)
    
    # Check if model exists
    model_path = Path("phase2_reid/checkpoints/best_reid_model.pth")
    
    if model_path.exists():
        size = model_path.stat().st_size / (1024 * 1024)
        print(f"\n✅ Model already downloaded!")
        print(f"   Path: {model_path}")
        print(f"   Size: {size:.1f} MB")
        return
    
    print("\nModel not found. Attempting download...")
    
    # Try GitHub releases first
    result = download_from_github_release()
    
    if result:
        print("\n✅ Model ready to use!")
        print("   Run: python demo.py --input video.mp4 --output result.mp4")
    else:
        print("\n" + "=" * 60)
        print("📋 MANUAL DOWNLOAD INSTRUCTIONS")
        print("=" * 60)
        print("""
1. Go to: https://github.com/prasanna-0911/persona-detection/releases

2. Download: best_reid_model.pth

3. Place in: phase2_reid/checkpoints/best_reid_model.pth

4. Run: python demo.py --input video.mp4 --output result.mp4
        """)


if __name__ == "__main__":
    main()
