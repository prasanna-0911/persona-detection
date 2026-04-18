"""
Room Occupancy Counter — Main Entry Point

Tracks people entering and exiting through configured doors using:
  • YOLOv8  — person detection
  • BoT-SORT — motion-only tracking (no Re-ID needed)
  • DoorLineCrosser — virtual line crossing detection
  • RoomOccupancyAggregator — live room occupancy
  • EventLogger — CSV + JSON log files

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Usage (run from repo root — persona-detection-main/)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  # Step 1 — calibrate virtual lines for each camera (run once per setup)
  python phase3_counter/counter_main.py --calibrate

  # Step 2 — run the counter
  python phase3_counter/counter_main.py

  # With a custom config
  python phase3_counter/counter_main.py --config phase3_counter/config/counter_config.json

  # Set initial occupancy (useful when starting mid-day)
  python phase3_counter/counter_main.py --initial-count 5

  # Limit to first N frames of video file (useful for testing)
  python phase3_counter/counter_main.py --max-frames 500

  # All options
  python phase3_counter/counter_main.py \\
      --config phase3_counter/config/counter_config.json \\
      --initial-count 0 \\
      --max-frames 1000 \\
      --no-display \\
      --no-save

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Live display controls
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Press  q  in any camera window — stop all cameras and exit cleanly
  Press  Ctrl+C  in terminal     — same effect
"""

import os
import sys
import json
import time
import argparse
import threading
import datetime
import numpy as np
import cv2
from collections import deque
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

# ── Fix OpenMP conflict on Windows (PyTorch + Intel MKL) ──────────────────────
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ── Path setup ────────────────────────────────────────────────────────────────
# Allow importing from the parent repo (multi_camera_tracker.py, etc.)
# WITHOUT modifying any existing file.
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# ── Imports from parent repo (read-only) ──────────────────────────────────────
from multi_camera_tracker import VideoSource   # reused; file not modified

# ── Imports from phase3_counter ───────────────────────────────────────────────
from core.door_line_crosser    import DoorLineCrosser
from core.occupancy_aggregator import RoomOccupancyAggregator
from core.event_logger         import EventLogger

# ── Ultralytics YOLO ──────────────────────────────────────────────────────────
from ultralytics import YOLO

# ── Shared stop signal (set by any thread to cleanly shut down all) ───────────
_STOP_EVENT = threading.Event()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Frame annotation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _annotate_frame(frame: np.ndarray,
                    tracks: List[Tuple[int, np.ndarray]],
                    crosser: DoorLineCrosser,
                    snapshot: Dict,
                    cam_name: str,
                    recent_events: deque) -> np.ndarray:
    """
    Draw all annotations on a copy of the frame:
      • Virtual door lines + band zone (yellow)
      • Person bounding boxes + track IDs (orange)
      • Foot-center dots (red)
      • ENTRY/EXIT event flash text (green/red — shows for 3 seconds)
      • Occupancy overlay panel (top-left, semi-transparent)
      • Camera name + timestamp

    Args:
        frame:         Raw BGR frame from the camera.
        tracks:        List of (track_id, bbox) for this frame.
        crosser:       DoorLineCrosser for this camera (holds line configs).
        snapshot:      Occupancy snapshot dict from aggregator.
        cam_name:      Camera label for display.
        recent_events: deque of (event_dict, timestamp) pairs.

    Returns:
        Annotated BGR frame (frame is NOT modified in-place).
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # ── 1. Draw virtual lines + band zones ────────────────────────────────────
    for line_cfg in crosser.get_line_configs():
        pt1  = line_cfg['start']
        pt2  = line_cfg['end']
        band = line_cfg['band_width']

        if pt1 == (0, 0) and pt2 == (0, 0):
            continue  # not yet calibrated

        # Compute perpendicular unit vector for band visualization
        dx   = pt2[0] - pt1[0]
        dy   = pt2[1] - pt1[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length > 0:
            nx, ny = -dy / length, dx / length  # perpendicular

            # Semi-transparent band overlay
            band_pts = np.array([
                [int(pt1[0] + nx * band), int(pt1[1] + ny * band)],
                [int(pt1[0] - nx * band), int(pt1[1] - ny * band)],
                [int(pt2[0] - nx * band), int(pt2[1] - ny * band)],
                [int(pt2[0] + nx * band), int(pt2[1] + ny * band)],
            ], np.int32)
            overlay = out.copy()
            cv2.fillPoly(overlay, [band_pts], (0, 200, 200))
            cv2.addWeighted(overlay, 0.15, out, 0.85, 0, out)

        # Main door line (bright yellow)
        cv2.line(out, pt1, pt2, (0, 255, 255), 2)
        cv2.circle(out, pt1, 5, (0, 255, 255), -1)
        cv2.circle(out, pt2, 5, (0, 255, 255), -1)

        # Door label at midpoint
        mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.putText(out, line_cfg['door_name'],
                    (mid[0] + 6, mid[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # ── 2. Draw person bounding boxes + foot dots ──────────────────────────────
    for track_id, bbox in tracks:
        x1, y1, x2, y2 = map(int, bbox)
        foot = ((x1 + x2) // 2, y2)

        # Box (orange)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)

        # Track ID label
        label = f"#{track_id}"
        lw, lh = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        cv2.rectangle(out, (x1, y1 - lh - 8), (x1 + lw + 6, y1), (0, 165, 255), -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Foot-center dot (red) — this is what the crossing algorithm uses
        cv2.circle(out, foot, 4, (0, 0, 255), -1)

    # ── 3. Event flash (last 3 seconds) ───────────────────────────────────────
    now = time.time()
    flash = [(ev, ts) for ev, ts in recent_events if now - ts < 3.0]
    for i, (ev, _) in enumerate(reversed(flash)):
        color = (0, 220, 0) if ev['event'] == 'ENTRY' else (0, 0, 220)
        symbol = '→ ENTRY' if ev['event'] == 'ENTRY' else '← EXIT'
        text   = f"{symbol}  {ev['door_name']}"
        y_pos  = h - 20 - i * 32
        # Shadow
        cv2.putText(out, text, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4)
        cv2.putText(out, text, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    # ── 4. Occupancy panel (top-left, semi-transparent) ───────────────────────
    panel_w, panel_h = 260, 115
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 32), (panel_w, 32 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

    occupancy = snapshot['current_occupancy']
    # Colour-code occupancy number: green < 80%, yellow 80–99%, red ≥ 100%
    occ_color = (0, 220, 0)   # default green

    cv2.putText(out, f"Occupancy: {occupancy}",
                (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.85, occ_color, 2)
    cv2.putText(out, f"Entered : {snapshot['total_entered']}",
                (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)
    cv2.putText(out, f"Exited  : {snapshot['total_exited']}",
                (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)
    cv2.putText(out, f"Peak    : {snapshot['peak_occupancy']}",
                (10, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1)

    # ── 5. Camera name (bottom-left) + timestamp (top-right) ──────────────────
    cv2.putText(out, f"Cam: {cam_name}",
                (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1)

    ts_str = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    ts_w   = cv2.getTextSize(ts_str, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)[0][0]
    cv2.putText(out, ts_str, (w - ts_w - 8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)

    # ── 6. Header bar with camera name ────────────────────────────────────────
    cv2.rectangle(out, (0, 0), (w, 30), (30, 30, 30), -1)
    cv2.putText(out, f"Room Occupancy Counter  |  {cam_name}",
                (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1)

    return out


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Camera worker thread
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _camera_worker(cam_cfg: Dict,
                   yolo_model: str,
                   conf: float,
                   imgsz: int,
                   botsort_cfg: str,
                   aggregator: RoomOccupancyAggregator,
                   logger: EventLogger,
                   display: bool,
                   save_output: bool,
                   output_dir: str,
                   max_frames: Optional[int]) -> None:
    """
    Thread target: process one camera stream end-to-end.

    Each camera gets its own YOLO instance to ensure independent BoT-SORT
    tracking state (track IDs don't bleed between cameras).

    Sets _STOP_EVENT if the user presses 'q' in this camera's window.
    """
    cam_name  = cam_cfg.get('name', 'Camera')
    source    = cam_cfg.get('source', '')
    lines_cfg = cam_cfg.get('lines', [])

    # ── Validate calibration ───────────────────────────────────────────────────
    crosser = DoorLineCrosser(lines_cfg)
    if not crosser.is_calibrated():
        print(f"\n⚠️  [{cam_name}] Virtual lines not calibrated (coordinates are [0,0]).")
        print(f"    Run:  python phase3_counter/counter_main.py --calibrate")
        print(f"    Skipping {cam_name}.\n")
        return

    # ── Open video source ──────────────────────────────────────────────────────
    vs = VideoSource(source)
    if not vs.open():
        print(f"❌ [{cam_name}] Cannot open source: {source}")
        return

    info = vs.get_info()
    print(f"✅ [{cam_name}] {info['width']}x{info['height']} "
          f"@ {info['fps']:.1f}fps  |  "
          f"{'RTSP stream' if info['is_stream'] else 'Video file'}")

    # ── Load YOLO (separate per camera for independent BoT-SORT state) ─────────
    print(f"   [{cam_name}] Loading YOLO: {yolo_model}")
    model = YOLO(yolo_model)

    # ── Total frame count (for tqdm in headless mode) ─────────────────────────
    # Only meaningful for video files; streams are infinite.
    total_frames = None
    if not info['is_stream']:
        _probe = cv2.VideoCapture(str(source))
        _n = int(_probe.get(cv2.CAP_PROP_FRAME_COUNT))
        _probe.release()
        total_frames = _n if _n > 0 else None

    # ── Output video writer ────────────────────────────────────────────────────
    out_writer = None
    out_path   = None
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        out_path  = os.path.abspath(os.path.join(output_dir, f"{cam_name}.avi"))
        out_fps   = 25 if info['is_stream'] else info['fps']
        fourcc    = cv2.VideoWriter_fourcc(*'XVID')
        out_writer = cv2.VideoWriter(out_path, fourcc, out_fps,
                                     (info['width'], info['height']))
        if out_writer.isOpened():
            print(f"   [{cam_name}] 💾 Saving output  →  {out_path}")
        else:
            print(f"   [{cam_name}] ⚠️  VideoWriter failed. Output will not be saved.")
            out_writer = None

    # ── Headless mode: print VLC instructions ──────────────────────────────────
    if not display and out_path:
        print(f"")
        print(f"   ┌─ HEADLESS MODE ({cam_name}) ────────────────────────────────────")
        print(f"   │  Processing without a display window.")
        print(f"   │  Output is being written to:")
        print(f"   │    {out_path}")
        print(f"   │")
        print(f"   │  Watch while processing (like persona detection project):")
        print(f"   │    VLC → Media → Open File → select the .avi above")
        print(f"   │    VLC reads the already-written frames as processing continues.")
        print(f"   │")
        print(f"   │  Or watch after processing: just open the same .avi in VLC.")
        print(f"   └─────────────────────────────────────────────────────────────────")
        print(f"")

    # ── Per-camera state ───────────────────────────────────────────────────────
    recent_events: deque = deque(maxlen=8)   # (event_dict, timestamp) — for flash overlay
    frame_count = 0

    mode_label = 'headless → VLC' if not display else 'live window'
    print(f"🎬 [{cam_name}] Starting ({mode_label}) ... "
          f"{'Press Ctrl+C to stop' if not display else 'press q or Ctrl+C'}")

    # ── tqdm progress bar (only in headless mode; hidden when window is shown) ──
    pbar = tqdm(
        total       = total_frames,
        desc        = f"  {cam_name}",
        unit        = " fr",
        disable     = display,          # bar only shows when no display window
        dynamic_ncols = True,
        colour      = 'cyan',
    )
    pbar.set_postfix(occ=0, IN=0, OUT=0)

    try:
        while not _STOP_EVENT.is_set():

            # ── Frame limit (for testing with video files) ─────────────────
            if max_frames and frame_count >= max_frames:
                print(f"   [{cam_name}] Reached --max-frames limit ({max_frames}).")
                break

            # ── Read frame ─────────────────────────────────────────────────
            ret, frame = vs.read()
            if not ret:
                if vs.is_stream:
                    continue  # RTSP: wait for next frame
                else:
                    print(f"   [{cam_name}] End of video file.")
                    break

            # ── Run BoT-SORT tracking ──────────────────────────────────────
            # with_reid=False in botsort_counter.yaml → pure motion tracking
            results = model.track(
                frame,
                persist    = True,
                tracker    = botsort_cfg,
                classes    = [0],           # class 0 = person in COCO
                conf       = conf,
                imgsz      = imgsz,
                verbose    = False,
            )

            # ── Extract tracks ─────────────────────────────────────────────
            tracks: List[Tuple[int, np.ndarray]] = []
            if (results and
                    results[0].boxes is not None and
                    results[0].boxes.id is not None):
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids   = results[0].boxes.id.int().cpu().tolist()
                tracks = list(zip(ids, boxes))

            # ── Check line crossings ───────────────────────────────────────
            events = crosser.update(tracks)

            # ── Process events ─────────────────────────────────────────────
            now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            for ev in events:
                aggregator.add_event(ev['event'], ev['door_id'])
                snap = aggregator.get_snapshot()
                logger.log_event({
                    'timestamp'      : now_str,
                    'camera_name'    : cam_name,
                    'door_id'        : ev['door_id'],
                    'door_name'      : ev['door_name'],
                    'event'          : ev['event'],
                    'track_id'       : ev['track_id'],
                    'occupancy_after': snap['current_occupancy'],
                })
                recent_events.append((ev, time.time()))
                icon = '🟢' if ev['event'] == 'ENTRY' else '🔴'
                print(f"  {icon} [{cam_name}] {ev['event']:5s}  "
                      f"{ev['door_name']}  "
                      f"| Occupancy: {snap['current_occupancy']}")

            # ── Annotate + display ─────────────────────────────────────────
            snap      = aggregator.get_snapshot()
            annotated = _annotate_frame(
                frame, tracks, crosser, snap, cam_name, recent_events
            )

            if display:
                cv2.imshow(cam_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print(f"\n   [{cam_name}] 'q' pressed — stopping all cameras.")
                    _STOP_EVENT.set()
                    break

            if out_writer:
                out_writer.write(annotated)

            frame_count += 1

            # ── Update progress bar (headless mode) ────────────────────────
            if not display:
                pbar.update(1)
                # Refresh postfix every 30 frames to avoid lock contention
                if frame_count % 30 == 0:
                    _s = aggregator.get_snapshot()
                    pbar.set_postfix(
                        occ = _s['current_occupancy'],
                        IN  = _s['total_entered'],
                        OUT = _s['total_exited'],
                    )

    except KeyboardInterrupt:
        pass

    finally:
        pbar.close()
        vs.release()
        if out_writer:
            out_writer.release()
        if display:
            cv2.destroyWindow(cam_name)
        print(f"✅ [{cam_name}] Stopped after {frame_count} frames.")
        if not display and out_path:
            print(f"   📂 Output saved: {out_path}  ← open this in VLC")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CLI argument parser
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='counter_main.py',
        description='Room Occupancy Counter — Entry/Exit tracking per door',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    p.add_argument('--config', '-c',
                   default='phase3_counter/config/counter_config.json',
                   help='Path to counter_config.json  (default: %(default)s)')
    p.add_argument('--calibrate', action='store_true',
                   help='Run the interactive line calibration tool and exit.')
    p.add_argument('--calib-vid', action='store_true',
                   help='Play the video feed during calibration instead of a static frame.')
    p.add_argument('--initial-count', type=int, default=0,
                   help='Seed occupancy with N people already in the room '
                        '(useful when starting mid-day).  Default: 0')
    p.add_argument('--max-frames', type=int, default=None,
                   help='Stop each camera after N frames (for testing). '
                        'Default: run until end of file / Ctrl+C.')
    p.add_argument('--no-display', action='store_true',
                   help='Disable live OpenCV windows (headless mode).')
    p.add_argument('--no-save', action='store_true',
                   help='Do not save output video files.')
    return p


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    args = _build_parser().parse_args()

    # ── Load config ────────────────────────────────────────────────────────────
    cfg_path = os.path.abspath(args.config)
    if not os.path.isfile(cfg_path):
        print(f"❌ Config file not found: {cfg_path}")
        print("   Default: phase3_counter/config/counter_config.json")
        sys.exit(1)

    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    # ── Calibration mode ───────────────────────────────────────────────────────
    if args.calibrate:
        print("\n🔧 Starting line calibration tool...")
        from utils.line_calibrator import run_calibration
        run_calibration(cfg_path, play_video=args.calib_vid)
        return

    # ── Banner ─────────────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("🚪  ROOM OCCUPANCY COUNTER")
    print(f"    Room    : {cfg.get('room_name', 'N/A')}")
    print(f"    Config  : {cfg_path}")
    print(f"    Cameras : {len(cfg.get('cameras', []))}")
    print(f"    Initial : {args.initial_count} people")
    print("=" * 65)

    # ── Resolve paths ──────────────────────────────────────────────────────────
    botsort_cfg = os.path.join(_THIS_DIR, 'config', 'botsort_counter.yaml')
    yolo_model  = cfg.get('yolo_model', 'yolov8m.pt')
    conf        = cfg.get('conf',  0.45)
    imgsz       = cfg.get('imgsz', 960)
    display     = cfg.get('display', True) and not args.no_display
    save_output = cfg.get('save_output', True) and not args.no_save
    output_dir  = cfg.get('output_dir', 'phase3_counter/outputs')
    log_cfg     = cfg.get('logging', {})
    log_dir     = log_cfg.get('log_dir', 'phase3_counter/logs')
    flush_every = log_cfg.get('flush_every', 10)
    cameras     = cfg.get('cameras', [])

    if not cameras:
        print("❌ No cameras defined in config. Nothing to do.")
        sys.exit(1)

    # ── Shared components ──────────────────────────────────────────────────────
    aggregator = RoomOccupancyAggregator(initial_count=args.initial_count)
    logger     = EventLogger(log_dir=log_dir, flush_every=flush_every)

    if args.initial_count > 0:
        print(f"\nℹ️  Initial occupancy seeded to {args.initial_count} "
              f"(mid-day startup mode).")

    # ── Launch one thread per camera ───────────────────────────────────────────
    threads = []
    for cam_cfg in cameras:
        t = threading.Thread(
            target    = _camera_worker,
            kwargs    = dict(
                cam_cfg     = cam_cfg,
                yolo_model  = yolo_model,
                conf        = conf,
                imgsz       = imgsz,
                botsort_cfg = botsort_cfg,
                aggregator  = aggregator,
                logger      = logger,
                display     = display,
                save_output = save_output,
                output_dir  = output_dir,
                max_frames  = args.max_frames,
            ),
            daemon    = True,
            name      = f"cam-{cam_cfg.get('name', 'unknown')}",
        )
        threads.append(t)
        t.start()

    print(f"\n🎬 {len(threads)} camera thread(s) running.  "
          f"Press Ctrl+C or 'q' in any window to stop.\n")

    # ── Wait for all threads to finish ─────────────────────────────────────────
    try:
        for t in threads:
            while t.is_alive():
                t.join(timeout=1.0)   # 1s timeout so Ctrl+C is responsive
    except KeyboardInterrupt:
        print("\n\n🛑 Ctrl+C — stopping all cameras...")
        _STOP_EVENT.set()
        for t in threads:
            t.join(timeout=10.0)
        print("✅ All camera threads stopped.")

    # ── Shutdown: flush logger + write session summary ─────────────────────────
    print("\n" + "=" * 65)
    print("📊 FINAL SESSION SUMMARY")
    print("=" * 65)

    summary = aggregator.get_session_summary()

    print(f"   Room              : {cfg.get('room_name', 'N/A')}")
    print(f"   Session start     : {summary['session_start']}")
    print(f"   Duration          : {summary['session_elapsed_s']}s")
    print(f"   Final occupancy   : {summary['final_occupancy']}")
    print(f"   Total entered     : {summary['total_entered']}")
    print(f"   Total exited      : {summary['total_exited']}")
    print(f"   Peak occupancy    : {summary['peak_occupancy']}  @ {summary['peak_time']}")
    print(f"\n   Per-door breakdown:")
    for door_id, stats in summary['per_door'].items():
        print(f"      {door_id:12s}  entered={stats['entered']}  exited={stats['exited']}")

    logger.write_session_summary({**summary, 'room_name': cfg.get('room_name', 'N/A')})
    logger.close()

    print("\n✅ Done.\n")


if __name__ == '__main__':
    main()
