"""
Line Calibration Tool — Interactive click-to-define virtual door lines.

Run this ONCE per new camera setup. For each camera in the config:
  1. Opens a live frame from the camera source in an OpenCV window.
  2. You LEFT-CLICK twice to define the two endpoints of the door line.
  3. After the line is drawn, you LEFT-CLICK once on the INSIDE of the room.
     This tells the system which side of the line is the "room interior",
     so it knows which direction = ENTRY and which = EXIT.
  4. Coordinates are saved back to your counter_config.json instantly.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Keyboard controls (while the calibration window is open):
  LEFT-CLICK  — set a calibration point
  r           — reset the current line (start over for this door)
  s           — save and move to the next camera / door
  q           — quit calibration without saving remaining cameras
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage (from repo root):
    python phase3_counter/counter_main.py --calibrate
    python phase3_counter/counter_main.py --calibrate --config phase3_counter/config/counter_config.json
"""

import os
import sys
import json
import copy
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

# ── Allow importing VideoSource from the parent repo without modifying it ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from multi_camera_tracker import VideoSource   # read-only import; file unchanged


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Geometry helper
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _cross_sign(point: Tuple[int, int],
                line_start: Tuple[int, int],
                line_end:   Tuple[int, int]) -> int:
    """Sign of cross product (line_end-line_start) × (point-line_start)."""
    dx = line_end[0]   - line_start[0]
    dy = line_end[1]   - line_start[1]
    px = point[0]      - line_start[0]
    py = point[1]      - line_start[1]
    cross = dx * py - dy * px
    if   cross > 0: return  1
    elif cross < 0: return -1
    return 0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Drawing helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _draw_instructions(canvas: np.ndarray, lines: List[str],
                       y_start: int = 50, color=(255, 255, 0)) -> None:
    """Overlay instruction text on the canvas.

    NOTE: y_start=50 (not 20) so text appears BELOW the 30px header banner.
    """
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (10, y_start + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4)   # shadow
        cv2.putText(canvas, line, (10, y_start + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def _draw_state(canvas: np.ndarray, state: Dict,
                door_name: str, cam_name: str) -> np.ndarray:
    """Redraw the calibration canvas based on current interaction state."""
    display = canvas.copy()
    h, w = display.shape[:2]

    # Header banner
    cv2.rectangle(display, (0, 0), (w, 30), (30, 30, 30), -1)
    cv2.putText(display, f"CALIBRATION  |  Camera: {cam_name}  |  Door: {door_name}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1)

    step = state['step']
    pt1  = state['pt1']
    pt2  = state['pt2']

    if step == 'pt1':
        _draw_instructions(display, [
            "Step 1 / 3 — LEFT-CLICK on one end of the door line.",
            "              (place the point at the door frame edge)",
            "  [r] reset   [q] quit",
        ])

    elif step == 'pt2':
        # Draw first point
        cv2.circle(display, pt1, 7, (0, 255, 255), -1)
        cv2.putText(display, "P1", (pt1[0] + 10, pt1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        _draw_instructions(display, [
            "Step 2 / 3 — LEFT-CLICK on the OTHER end of the door line.",
            "  [r] reset   [q] quit",
        ])

    elif step == 'inside':
        # Draw completed line + both endpoints
        cv2.line(display, pt1, pt2, (0, 255, 255), 2)
        cv2.circle(display, pt1, 7, (0, 255, 255), -1)
        cv2.circle(display, pt2, 7, (0, 255, 255), -1)
        # Midpoint label
        mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.putText(display, door_name, (mid[0] + 5, mid[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        _draw_instructions(display, [
            "Step 3 / 3 — LEFT-CLICK anywhere on the INSIDE of the room.",
            "              (the side where people will be AFTER entering)",
            "  [r] reset   [q] quit",
        ], color=(0, 255, 100))

    elif step == 'done':
        inside_pt = state['inside_pt']
        cv2.line(display, pt1, pt2, (0, 200, 0), 3)
        cv2.circle(display, pt1, 7, (0, 200, 0), -1)
        cv2.circle(display, pt2, 7, (0, 200, 0), -1)
        cv2.circle(display, inside_pt, 10, (255, 100, 0), -1)

        # Green arrow hinting the ENTRY direction at the midpoint
        mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.putText(display, door_name, (mid[0] + 5, mid[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        cv2.circle(display, inside_pt, 10, (0, 100, 255), 2)
        cv2.putText(display, "INSIDE", (inside_pt[0] + 12, inside_pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)

        _draw_instructions(display, [
            "✅  Line saved!  Press [s] to continue to next door / camera.",
            "                Press [r] to redo this line.",
            "                Press [q] to quit.",
        ], color=(0, 255, 100))

    return display


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main calibration function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_calibration(config_path: str) -> bool:
    """
    Interactive calibration loop. Iterates over every camera × line in
    the config and lets the user define each virtual door line.

    Args:
        config_path: Path to counter_config.json (will be overwritten).

    Returns:
        True if calibration completed (at least one line saved),
        False if user quit early.
    """
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    cameras   = cfg.get('cameras', [])
    any_saved = False

    for cam_idx, cam_cfg in enumerate(cameras):
        cam_name  = cam_cfg.get('name', f'Camera_{cam_idx + 1}')
        source    = cam_cfg.get('source', '')
        lines_cfg = cam_cfg.get('lines', [])

        print(f"\n📷  Calibrating {cam_name}  ({source})")

        # ── Grab a reference frame ──────────────────────────────────────────
        vs = VideoSource(source)
        if not vs.open():
            print(f"   ❌ Cannot open source. Skipping {cam_name}.")
            continue

        # Try up to 30 frames to get a non-empty frame
        canvas = None
        for _ in range(30):
            ret, frame = vs.read()
            if ret and frame is not None:
                canvas = frame.copy()
                break
        vs.release()

        if canvas is None:
            print(f"   ❌ Could not read a frame from {cam_name}. Skipping.")
            continue

        print(f"   Frame grabbed: {canvas.shape[1]}x{canvas.shape[0]}")

        # ── Calibrate each line on this camera ──────────────────────────────
        for line_idx, line_cfg in enumerate(lines_cfg):
            door_id   = line_cfg.get('door_id',   f'Door_{line_idx + 1}')
            door_name = line_cfg.get('door_name', door_id)

            print(f"\n   🚪  Defining line for: {door_name} ({door_id})")

            window_name = f"Calibrate — {cam_name} — {door_name}"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, min(canvas.shape[1], 1280),
                             min(canvas.shape[0], 720))

            # ── Interaction state ────────────────────────────────────────────
            state = {'step': 'pt1', 'pt1': None, 'pt2': None, 'inside_pt': None}

            def mouse_callback(event, x, y, flags, param):
                # Only process LBUTTONDOWN — NOT LBUTTONUP.
                # A single click fires both events at the same (x, y).
                # If we also handle LBUTTONUP, the second event would run
                # the next step's handler with the previous click's coords,
                # causing the 'inside' handler to receive pt2's coordinates
                # (which are exactly on the line → cross product = 0 always).
                if event != cv2.EVENT_LBUTTONDOWN:
                    return
                s = param

                if s['step'] == 'pt1':
                    s['pt1'] = (x, y)
                    s['step'] = 'pt2'
                    print(f"   ✔  Point 1 set at {(x, y)} — now click Point 2")

                elif s['step'] == 'pt2':
                    s['pt2'] = (x, y)
                    if s['pt1'] == s['pt2']:
                        print("   ⚠️  Both points are the same. Click a different location.")
                        s['step'] = 'pt2'
                    else:
                        s['step'] = 'inside'
                        print(f"   ✔  Point 2 set at {(x, y)} — now click INSIDE the room")

                elif s['step'] == 'inside':
                    sign = _cross_sign((x, y), s['pt1'], s['pt2'])
                    if sign == 0:
                        print("   ⚠️  Click is exactly on the line. Click clearly inside the room.")
                        return
                    s['inside_pt']   = (x, y)
                    s['inside_sign'] = sign
                    s['step']        = 'done'
                    print(f"   ✔  Inside point set at {(x, y)} — press [s] to save")

            # ── Interaction loop ─────────────────────────────────────────────
            # IMPORTANT (Windows): show the window FIRST, then register the
            # mouse callback. On Windows, callbacks registered before the first
            # imshow() call are silently ignored by the Win32 event loop.
            quit_all = False
            first_draw = _draw_state(canvas, state, door_name, cam_name)
            cv2.imshow(window_name, first_draw)
            cv2.waitKey(200)          # let the window fully appear + gain focus
            cv2.setMouseCallback(window_name, mouse_callback, state)
            print("   👆  Window ready. Click in the window to begin.")

            while True:
                display = _draw_state(canvas, state, door_name, cam_name)
                cv2.imshow(window_name, display)
                key = cv2.waitKey(30) & 0xFF

                if key == ord('r'):
                    # Reset this line
                    state.update({'step': 'pt1', 'pt1': None, 'pt2': None,
                                  'inside_pt': None})
                    print("   🔄  Reset. Start over for this line.")

                elif key == ord('s') and state['step'] == 'done':
                    # Save to config
                    line_cfg['start']       = list(state['pt1'])
                    line_cfg['end']         = list(state['pt2'])
                    line_cfg['inside_sign'] = state['inside_sign']
                    print(f"   ✅  Saved: start={state['pt1']}  "
                          f"end={state['pt2']}  inside_sign={state['inside_sign']}")
                    any_saved = True
                    break

                elif key == ord('s') and state['step'] != 'done':
                    print("   ⚠️  Complete all 3 steps before pressing [s].")

                elif key == ord('q'):
                    print("\n   ⛔  Calibration quit by user.")
                    quit_all = True
                    break

            cv2.destroyWindow(window_name)

            if quit_all:
                # Write whatever was saved so far
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                print(f"\n📁 Config saved (partial): {config_path}")
                return False

    # ── Write updated config ──────────────────────────────────────────────────
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"\n✅  Calibration complete! Config saved: {config_path}")
    print("    You can now run: python phase3_counter/counter_main.py")
    return any_saved
