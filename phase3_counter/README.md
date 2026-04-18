# 🚪 Room Occupancy Counter

Person entry/exit counting system built on top of the `persona-detection` project.
Counts people entering and exiting through configured doors using:

- **YOLOv8** — person detection
- **BoT-SORT** — motion-only tracking (Re-ID not needed)
- **Virtual line crossing** — band-zone algorithm (no jitter counts)
- **Multi-camera + multi-door** — one camera can watch N doors
- **Live display** — annotated OpenCV window per camera
- **CSV + JSON logs** — real-time event log + session summary on exit

> ✅ Zero existing project files are modified. All new code lives in `phase3_counter/`.

---

## Folder Structure

```
phase3_counter/
│
├── config/
│   ├── counter_config.json          ← Your camera + door + line config
│   ├── counter_config_example.json  ← Example showing all options
│   └── botsort_counter.yaml         ← Tracker config (no Re-ID, pure motion)
│
├── core/
│   ├── door_line_crosser.py         ← Virtual line crossing detection
│   ├── occupancy_aggregator.py      ← Thread-safe room occupancy counter
│   └── event_logger.py              ← CSV + JSON logging
│
├── utils/
│   └── line_calibrator.py           ← Click-to-define door lines tool
│
├── logs/                            ← Auto-created: CSV + JSON logs
├── outputs/                         ← Auto-created: annotated output videos
├── test_footage/                    ← Place your test .mp4 files here
│
└── counter_main.py                  ← Main entry point
```

---

## Quick Start

### Step 1 — Install dependencies (same as parent project)
```bash
pip install ultralytics opencv-python torch torchvision numpy tqdm
```

### Step 2 — Edit config (add your camera sources)
Open `phase3_counter/config/counter_config.json` and set the RTSP URL
or video file path for each camera.

### Step 3 — Calibrate virtual door lines (run once per setup)
```bash
python phase3_counter/counter_main.py --calibrate
```
An OpenCV window opens for each camera.
- **Click 1**: One end of the door line
- **Click 2**: Other end of the door line
- **Click 3**: Anywhere on the **INSIDE** of the room

Press `s` to save each line, `q` to quit early.

### Step 4 — Run the counter
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

# No live display (headless / server mode)
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

### `entry_exit_log_YYYY-MM-DD.csv`

Real-time event log — one row per detected entry or exit.

| Column | Description |
|--------|-------------|
| `timestamp` | Event time (YYYY-MM-DD HH:MM:SS.mmm) |
| `camera_name` | e.g. Camera_1 |
| `door_id` | e.g. Door_A |
| `door_name` | e.g. Main Entrance |
| `event` | ENTRY or EXIT |
| `track_id` | BoT-SORT track ID |
| `occupancy_after` | Room occupancy after this event |

**Example:**
```csv
timestamp,camera_name,door_id,door_name,event,track_id,occupancy_after
2026-04-18 09:15:32.441,Camera_1,Door_A,Main Entrance,ENTRY,42,1
2026-04-18 09:17:10.112,Camera_2,Door_B,Side Entrance,ENTRY,7,2
2026-04-18 09:18:45.890,Camera_1,Door_A,Main Entrance,EXIT,51,1
```

### `session_summary_YYYY-MM-DD_HHMMSS.json`

Written once on clean shutdown. Contains:
- Total entered / exited
- Peak occupancy + peak time
- Per-door breakdown
- Session duration

---

## Config Reference

```json
{
  "room_name"   : "Conference Room A",
  "yolo_model"  : "yolov8m.pt",
  "conf"        : 0.45,
  "imgsz"       : 960,
  "device"      : "cuda",
  "display"     : true,
  "save_output" : true,
  "output_dir"  : "phase3_counter/outputs",
  "logging": {
    "log_dir"    : "phase3_counter/logs",
    "flush_every": 10
  },
  "cameras": [
    {
      "name"  : "Camera_1",
      "source": "rtsp://admin:pass@192.168.1.100:554/...",
      "lines" : [
        {
          "door_id"    : "Door_A",
          "door_name"  : "Main Entrance",
          "start"      : [320, 200],
          "end"        : [320, 700],
          "inside_sign": 1,
          "band_width" : 25
        }
      ]
    }
  ]
}
```

**`inside_sign`**: `+1` = left side of line (start→end direction) is room interior.
`-1` = right side. Set automatically by `--calibrate`.

**`band_width`**: Crossing zone in pixels. A person must fully cross this band to
register a count. Higher = less jitter, but slower detection near the line.

---

## Multi-Door Single Camera

One camera can watch multiple doors. Add multiple entries to `"lines"`:

```json
"lines": [
  {"door_id": "Door_A", "door_name": "Left Door",  "start": [...], "end": [...], ...},
  {"door_id": "Door_B", "door_name": "Right Door", "start": [...], "end": [...], ...}
]
```

---

## Occupancy Alert (Threshold TBD)

The alert threshold is pre-wired in `core/occupancy_aggregator.py` as a comment.
Uncomment and set `self._alert_threshold = N` once the client confirms their
room capacity limit.

---

## Notes

- **Zero Re-ID required** — BoT-SORT runs in pure-motion mode (`with_reid: False`).
- **One YOLO instance per camera** — ensures BoT-SORT state is isolated between cameras.
- **Thread-safe** — camera threads share `RoomOccupancyAggregator` and `EventLogger` safely.
- **Startup occupancy** — use `--initial-count N` if the system starts mid-day.
- **Output format** — XVID/AVI (reliable on Windows; avoids mp4v corruption on early exit).
