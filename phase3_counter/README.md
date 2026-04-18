# Room Occupancy Counter

Person entry/exit counting system built on top of the persona-detection project.
Counts people entering and exiting through configured doors using:

- YOLOv8: Person detection
- BoT-SORT: Motion-only tracking (Re-ID not needed)
- Virtual line crossing: Band-zone algorithm to prevent jitter counts
- Multi-camera and multi-door: One camera can watch multiple doors
- Live display: Annotated OpenCV window per camera
- CSV and JSON logs: Real-time event log and session summary on exit

Note: Zero existing project files are modified. All new code lives in the phase3_counter directory.

---

## Folder Structure

```text
phase3_counter/
│
├── config/
│   ├── counter_config.json          <- Your camera, door, and line config
│   ├── counter_config_example.json  <- Example showing all options
│   └── botsort_counter.yaml         <- Tracker config (no Re-ID, pure motion)
│
├── core/
│   ├── door_line_crosser.py         <- Virtual line crossing detection
│   ├── occupancy_aggregator.py      <- Thread-safe room occupancy counter
│   └── event_logger.py              <- CSV and JSON logging
│
├── utils/
│   └── line_calibrator.py           <- Click-to-define door lines tool
│
├── logs/                            <- Auto-created: CSV and JSON logs
├── outputs/                         <- Auto-created: annotated output videos
├── test_footage/                    <- Place your test .mp4 files here
│
└── counter_main.py                  <- Main entry point
```

---

## Core Features and Tools

### Interactive Calibration
You do not need to manually guess pixel coordinates for the door lines. A built-in calibration script lets you click and draw the lines.
- Screen Scaling: Automatically fits high-resolution footage onto smaller screens while preserving coordinate accuracy.
- Video Playback Option: You can calibrate using a static image, or enable video playback to watch where people actually walk before placing the line. Press Spacebar during calibration to pause or resume the video.

### Execution Modes
- Live Mode: Displays real-time OpenCV windows showing detections and counts.
- Headless Mode: Runs in the background without windows to save computer resources. It outputs a video file that can be monitored live through VLC player while it is still processing.

---

## Quick Start

### Step 1: Install dependencies
```bash
pip install ultralytics opencv-python torch torchvision numpy tqdm
```

### Step 2: Edit configuration
Open `phase3_counter/config/counter_config.json` and set the RTSP URL or video file path for each camera.

### Step 3: Calibrate virtual door lines (run once per setup)
Run the calibration tool to draw the door boundaries.
```bash
python phase3_counter/counter_main.py --calibrate
```
To play the video while drawing lines, use the video calibration flag:
```bash
python phase3_counter/counter_main.py --calibrate --calib-vid
```

An OpenCV window opens for each camera.
- Click 1: One end of the door line
- Click 2: Other end of the door line
- Click 3: Anywhere on the INSIDE of the room

Press 's' to save each line, 'q' to quit early.

### Step 4: Run the counter
```bash
python phase3_counter/counter_main.py
```

---

## All Command-Line Options

```bash
# Standard run
python phase3_counter/counter_main.py

# Custom config file
python phase3_counter/counter_main.py --config phase3_counter/config/counter_config.json

# Start mid-day with people already in the room
python phase3_counter/counter_main.py --initial-count 5

# Test with first 500 frames only (for video files)
python phase3_counter/counter_main.py --max-frames 500

# No live display (headless mode)
python phase3_counter/counter_main.py --no-display

# Do not save output videos
python phase3_counter/counter_main.py --no-save

# Combine flags
python phase3_counter/counter_main.py \
    --config phase3_counter/config/counter_config.json \
    --initial-count 3 \
    --max-frames 1000
```

---

## Log Files

All logs are written to `phase3_counter/logs/`.

### entry_exit_log_YYYY-MM-DD.csv
Real-time event log, one row per detected entry or exit.

| Column | Description |
|--------|-------------|
| timestamp | Event time (YYYY-MM-DD HH:MM:SS.mmm) |
| camera_name | Example: Camera_1 |
| door_id | Example: Door_A |
| door_name | Example: Main Entrance |
| event | ENTRY or EXIT |
| track_id | BoT-SORT track ID |
| occupancy_after | Room occupancy after this event |

**Example:**
```csv
timestamp,camera_name,door_id,door_name,event,track_id,occupancy_after
2026-04-18 09:15:32.441,Camera_1,Door_A,Main Entrance,ENTRY,42,1
2026-04-18 09:17:10.112,Camera_2,Door_B,Side Entrance,ENTRY,7,2
2026-04-18 09:18:45.890,Camera_1,Door_A,Main Entrance,EXIT,51,1
```

### session_summary_YYYY-MM-DD_HHMMSS.json
Written once on clean shutdown. Contains:
- Total entered and exited
- Peak occupancy and peak time
- Per-door breakdown
- Session duration

---

## Config Reference

```json
{
  "room_name": "Conference Room A",
  "yolo_model": "yolov8m.pt",
  "conf": 0.45,
  "imgsz": 960,
  "device": "cuda",
  "display": true,
  "save_output": true,
  "output_dir": "phase3_counter/outputs",
  "logging": {
    "log_dir": "phase3_counter/logs",
    "flush_every": 10
  },
  "cameras": [
    {
      "name": "Camera_1",
      "source": "rtsp://admin:pass@192.168.1.100:554/...",
      "lines": [
        {
          "door_id": "Door_A",
          "door_name": "Main Entrance",
          "start": [320, 200],
          "end": [320, 700],
          "inside_sign": 1,
          "band_width": 25
        }
      ]
    }
  ]
}
```

**inside_sign**: 1 = left side of line (start to end direction) is room interior. -1 = right side. Set automatically by the calibration script.

**band_width**: Crossing zone in pixels. A person must fully cross this band to register a count. Higher values reduce jitter but slow detection near the line.

---

## Multi-Door Single Camera

One camera can watch multiple doors. Add multiple entries to the "lines" array:

```json
"lines": [
  {"door_id": "Door_A", "door_name": "Left Door",  "start": [...], "end": [...], ...},
  {"door_id": "Door_B", "door_name": "Right Door", "start": [...], "end": [...], ...}
]
```

---

## Occupancy Alert

The alert threshold is pre-wired in `core/occupancy_aggregator.py` as a comment. Uncomment and set `self._alert_threshold = N` once the room capacity limit is confirmed.

---

## Notes

- **Zero Re-ID required**: BoT-SORT runs in pure-motion mode (with_reid is False).
- **One YOLO instance per camera**: Ensures BoT-SORT state is isolated between cameras.
- **Thread-safe**: Camera threads share RoomOccupancyAggregator and EventLogger safely.
- **Startup occupancy**: Use --initial-count N if the system starts mid-day.
- **Output format**: XVID/AVI (reliable on Windows; avoids video corruption on early exit).
