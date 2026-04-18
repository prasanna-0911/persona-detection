# Test Footage

Place your door-specific test video files here.

Expected filenames (matching counter_config.json defaults):
  - door_test_1.mp4   → used by Camera_1
  - door_test_2.mp4   → used by Camera_2

Supported formats: MP4, AVI, MKV (any OpenCV-compatible format).

Once you place the files here, run:
  python phase3_counter/counter_main.py --max-frames 300

Note: This folder is intentionally excluded from the main codebase.
      Add it to .gitignore if you do not want test videos committed to git.
