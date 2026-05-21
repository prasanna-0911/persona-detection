# Room Occupancy Counter

Person entry/exit counting system built on top of the persona-detection project.
Counts people entering and exiting through configured doors using:

- YOLOv8: Person detection
- BoT-SORT: Motion-only tracking (Re-ID not needed)
- Virtual line crossing: Band-zone algorithm to prevent jitter counts
- Direction detection: Filters false counts from backward movement
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
│   ├── event_logger.py              <- CSV and JSON logging
│   ├── direction_tracker.py         <- Movement direction detection
│   └── room_region.py               <- Polygon-based room region filtering
│
├── utils/
│   ├── line_calibrator.py           <- Click-to-define door lines tool
│   └── room_region_calibrator.py    <- Click-to-draw room polygon tool
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
- Screen Scaling: Automatically fits high-resolution footage onto smaller screens while preserves coordinate accuracy.
- Video Playback Option: You can calibrate using a static image, or enable video playback to watch where people actually walk before placing the line. Press Spacebar during calibration to pause or resume the video.

### Direction Detection
The system can filter false counts caused by backward movement through doors.

**How it works:**
- Tracks foot positions between consecutive frames
- Computes movement velocity vector (dx, dy)
- Compares against door line direction to determine forward/backward
- Filters events based on the `direction` setting

**Direction settings:**
| Value | Behavior |
|-------|----------|
| `"both"` (default) | Count both entry and exit (backward compatible) |
| `"entry"` | Only count forward movement INTO the room |
| `"exit"` | Only count forward movement OUT OF the room |

### Room Region (Polygon)
When a camera can see multiple rooms, you can define a room interior polygon to only count people inside that specific area.

**How it works:**
- Draw a polygon defining the room interior
- Only people with foot position inside the polygon are counted
- People in other visible rooms are ignored

**Calibration:**
```bash
# Draw room region polygon
python phase3_counter/counter_main.py --calibrate-region

# With video playback during calibration
python phase3_counter/counter_main.py --calibrate-region --calib-region-vid
```

**Controls during calibration:**
- Left Click: Add polygon point
- 'u': Undo last point
- 'c': Clear all points
- 's': Save and move to next camera
- 'q': Quit without saving
- Spacebar: Play/pause video (video mode only)

**Configuration:**
```json
{
  "room_region": {
    "polygon": [
      [50, 100],
      [800, 100],
      [800, 600],
      [50, 600]
    ]
  }
}
```

### Execution Modes
- Live Mode: Displays real-time OpenCV windows showing detections and counts.
- Headless Mode: Runs in the background without windows to save computer resources. It outputs a video file that can be monitored live through VLC player while it is still processing.

---

## Quick Start

### Step 1: Install dependencies
```bash
pip install ultralytics opencv-python torch torchvision numpy tqdm
```

### Step 2: Prepare test footage
Place your `.mp4` or `.avi` video files inside `phase3_counter/test_footage/`.  
If using RTSP cameras, you can skip this step and configure the source URL later.

**Download sample test videos** (5–10 second clips of people walking through doorways work best):
- Free stock footage sites: [Pexels](https://www.pexels.com/search/video/door/), [Pixabay](https://pixabay.com/videos/search/door/)
- Or record your own short clip with a phone/webcam

### Step 3: Edit the config file
Open `phase3_counter/config/counter_config.json` and configure:

**A. Main settings:**
```json
{
  "room_name": "Conference Room A",
  "yolo_model": "yolov8m.pt",
  "conf": 0.45,
  "imgsz": 960,
  "device": "cuda"
}
```
- `conf`: Detection confidence threshold (lower = more detections, more false positives)
- `imgsz`: Inference image size (larger = slower but more accurate)
- `device`: `"cuda"` for GPU, `"cpu"` if no GPU

**B. Camera source(s):**
```json
"cameras": [
  {
    "name": "Camera_1",
    "source": "phase3_counter/test_footage/my_video.mp4",
    ...
  }
]
```
- `source`: Path to video file or RTSP URL (`rtsp://user:pass@ip:port/stream`)
- `name`: Any label for log files

### Step 4: Calibrate virtual door lines (run once per setup)
Run the calibration tool to draw the door boundaries interactively:
```bash
python phase3_counter/counter_main.py --calibrate
```
To play the video while drawing lines (helps see where people actually walk):
```bash
python phase3_counter/counter_main.py --calibrate --calib-vid
```

**What you do in the OpenCV window for each camera:**
- **Click 1:** One end of the door line
- **Click 2:** Other end of the door line
- **Click 3:** Anywhere on the INSIDE of the room (the side people enter into)

**Controls:**
| Key | Action |
|-----|--------|
| Left Click | Place a point |
| `s` | Save line for this camera |
| `q` | Quit calibration (skip remaining cameras) |
| Spacebar | Play/pause video (only with `--calib-vid`) |

After saving, the coordinates are written back to `counter_config.json` automatically.

### Step 5: (Optional) Calibrate room region polygon
If your camera sees multiple rooms and you want to only count people inside a specific room:
```bash
# Draw polygon defining the room interior
python phase3_counter/counter_main.py --calibrate-region

# With video playback during calibration
python phase3_counter/counter_main.py --calibrate-region --calib-region-vid
```

**What you do:**
- Left Click each vertex of the room polygon
- Press `s` to save when done
- The polygon is written to `counter_config.json` automatically

**Controls:**
| Key | Action |
|-----|--------|
| Left Click | Add polygon point |
| `u` | Undo last point |
| `c` | Clear all points |
| `s` | Save and move to next camera |
| `q` | Quit without saving |
| Spacebar | Play/pause video (video mode only) |

### Step 6: Run the counter
```bash
python phase3_counter/counter_main.py
```

**First run with limited frames (recommended):**
```bash
python phase3_counter/counter_main.py --max-frames 300
```
This processes only the first 300 frames — useful to verify everything works before a full run.

**Common flags:**
| Flag | Purpose |
|------|---------|
| `--max-frames N` | Process only N frames |
| `--no-display` | Run headless (no OpenCV windows) |
| `--no-save` | Don't save annotated output video |
| `--initial-count N` | Starting occupancy (e.g., 5 if people already inside) |
| `--config PATH` | Use a different config file |

### Step 7: Check the results
- **CSV log:** `phase3_counter/logs/entry_exit_log_YYYY-MM-DD.csv` — every entry/exit event
- **JSON summary:** `phase3_counter/logs/session_summary_*.json` — total counts, peak occupancy
- **Output video:** `phase3_counter/outputs/` — annotated video with counts drawn on frames

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
          "band_width": 25,
          "direction": "both"
        }
      ]
    }
  ]
}
```

**inside_sign**: 1 = left side of line (start to end direction) is room interior. -1 = right side. Set automatically by the calibration script.

**band_width**: Crossing zone in pixels. A person must fully cross this band to register a count. Higher values reduce jitter but slow detection near the line.

**direction**: Controls event filtering based on movement direction. See Configuration Templates below.

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

## Configuration Templates

### Template 1: Single Camera (Bidirectional)

Use when one camera watches a single door with both entry and exit traffic.

```json
{
  "cameras": [
    {
      "name": "Main_Door",
      "source": "main_door.mp4",
      "lines": [
        {
          "door_id": "Door_A",
          "door_name": "Main Entrance",
          "start": [320, 200],
          "end": [320, 700],
          "inside_sign": 1,
          "band_width": 25,
          "direction": "both"
        }
      ]
    }
  ]
}
```

### Template 2: Dual Cameras (Entry + Exit)

Use when you have two separate cameras — one for entry, one for exit.

```json
{
  "cameras": [
    {
      "name": "Entry_Camera",
      "source": "entry_door.mp4",
      "lines": [
        {
          "door_id": "Door_Entry",
          "door_name": "Main Entrance",
          "start": [320, 200],
          "end": [320, 700],
          "inside_sign": 1,
          "band_width": 25,
          "direction": "entry"
        }
      ]
    },
    {
      "name": "Exit_Camera",
      "source": "exit_door.mp4",
      "lines": [
        {
          "door_id": "Door_Exit",
          "door_name": "Emergency Exit",
          "start": [400, 150],
          "end": [400, 650],
          "inside_sign": 1,
          "band_width": 25,
          "direction": "exit"
        }
      ]
    }
  ]
}
```

**How it works:**
- Entry camera only counts forward movement INTO the room
- Exit camera only counts forward movement OUT OF the room
- Backward movement is filtered out (ignored)

### Template 3: Same Doorway (Double Door)

Use when one camera watches a double door where left is entry and right is exit.

```json
{
  "cameras": [
    {
      "name": "Double_Door_Camera",
      "source": "double_door.mp4",
      "lines": [
        {
          "door_id": "Door_Left",
          "door_name": "Entry Side",
          "start": [200, 200],
          "end": [200, 700],
          "inside_sign": 1,
          "band_width": 25,
          "direction": "entry"
        },
        {
          "door_id": "Door_Right",
          "door_name": "Exit Side",
          "start": [600, 200],
          "end": [600, 700],
          "inside_sign": -1,
          "band_width": 25,
          "direction": "exit"
        }
      ]
    }
  ]
}
```

---

## Direction Detection Behavior

| Direction Setting | Forward Movement Into Room | Backward Movement | Forward Movement Out of Room |
|-------------------|---------------------------|-------------------|---------------------------|
| `"both"` | ENTRY ✅ | EXIT ⚠️ | EXIT ✅ |
| `"entry"` | ENTRY ✅ | **IGNORED** ❌ | **IGNORED** ❌ |
| `"exit"` | **IGNORED** ❌ | **IGNORED** ❌ | EXIT ✅ |

**Note:** Even with direction filtering, the system correctly tracks room occupancy. A person who backs out and then enters again will result in net zero change (EXIT then ENTRY).

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
