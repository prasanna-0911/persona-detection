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
import msvcrt
import cv2
import ctypes
import numpy as np
from typing import Dict, List, Optional, Tuple

# ── Allow importing VideoSource from the parent repo without modifying it ──
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from multi_camera_tracker import VideoSource   # read-only import; file unchanged


def _win_prompt(prompt_text: str) -> str:
    sys.stdout.write(prompt_text)
    sys.stdout.flush()
    chars = []
    while True:
        ch = msvcrt.getwch()
        if ch == '\r':
            sys.stdout.write('\n')
            sys.stdout.flush()
            return ''.join(chars)
        if ch == '\b':
            if chars:
                chars.pop()
                sys.stdout.write('\b \b')
                sys.stdout.flush()
        else:
            chars.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()


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
    """Overlay instruction text on the canvas."""
    h, w = canvas.shape[:2]
    text_h = len(lines) * 30 + 20
    x2 = min(420, w - 10)
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (x2, text_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)
    for i, line in enumerate(lines):
        cv2.putText(canvas, line, (10, y_start + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def _draw_state(canvas: np.ndarray, state: Dict,
                door_name: str, cam_name: str) -> np.ndarray:
    """Redraw the calibration canvas based on current interaction state."""
    display = canvas.copy()
    h, w = display.shape[:2]

    # Camera/door label at top-left (small, no background bar)
    cv2.putText(display, f"{cam_name} | {door_name}", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 1)

    step = state['step']
    pt1  = state['pt1']
    pt2  = state['pt2']

    if step == 'pt1':
        _draw_instructions(display, [
            "Step 1 / 3 — LEFT-CLICK on one end of the door line",
            "  [SPACE] pause/play   [r] reset   [q] quit",
        ])

    elif step == 'pt2':
        cv2.circle(display, pt1, 7, (0, 255, 255), -1)
        cv2.putText(display, "P1", (pt1[0] + 10, pt1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        _draw_instructions(display, [
            "Step 2 / 3 — LEFT-CLICK on the OTHER end of the door line",
            "  [r] reset   [q] quit",
        ])

    elif step == 'inside':
        cv2.line(display, pt1, pt2, (0, 255, 255), 2)
        cv2.circle(display, pt1, 7, (0, 255, 255), -1)
        cv2.circle(display, pt2, 7, (0, 255, 255), -1)
        mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.putText(display, door_name, (mid[0] + 5, mid[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        _draw_instructions(display, [
            "Step 3 / 3 — LEFT-CLICK on the INSIDE of the room",
            "  [r] reset   [q] quit",
        ], color=(0, 255, 100))

    elif step == 'done':
        inside_pt = state['inside_pt']
        cv2.line(display, pt1, pt2, (0, 200, 0), 3)
        cv2.circle(display, pt1, 7, (0, 200, 0), -1)
        cv2.circle(display, pt2, 7, (0, 200, 0), -1)
        cv2.circle(display, inside_pt, 10, (255, 100, 0), -1)
        mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.putText(display, door_name, (mid[0] + 5, mid[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        cv2.circle(display, inside_pt, 10, (0, 100, 255), 2)
        cv2.putText(display, "INSIDE", (inside_pt[0] + 12, inside_pt[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 2)
        _draw_instructions(display, [
            "Line saved!  Press [s] to continue  [r] redo  [q] quit",
        ], color=(0, 255, 100))

    return display

    return display


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main calibration function
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_calibration(config_path: str, play_video: bool = False) -> bool:
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

    # Close any leftover windows from previous runs
    cv2.destroyAllWindows()

    for cam_idx, cam_cfg in enumerate(cameras):
        cam_name  = cam_cfg.get('name', f'Camera_{cam_idx + 1}')
        source    = cam_cfg.get('source', '')
        lines_cfg = cam_cfg.get('lines', [])

        print(f"\n[CAM]  Calibrating {cam_name}  ({source})")

        # ── Grab a reference frame ──────────────────────────────────────────
        vs = VideoSource(source)
        if not vs.open():
            print(f"   [ERR] Cannot open source. Skipping {cam_name}.")
            continue

        # Try up to 30 frames to get a non-empty frame
        canvas = None
        for _ in range(30):
            ret, frame = vs.read()
            if ret and frame is not None:
                canvas = frame.copy()
                break
        
        if not play_video:
            vs.release()

        if canvas is None:
            print(f"   [ERR] Could not read a frame from {cam_name}. Skipping.")
            if play_video: vs.release()
            continue

        print(f"   Frame grabbed: {canvas.shape[1]}x{canvas.shape[0]}")

        # ── Calibrate each line on this camera ──────────────────────────────
        for line_idx, line_cfg in enumerate(lines_cfg):
            door_id   = line_cfg.get('door_id',   f'Door_{line_idx + 1}')
            door_name = line_cfg.get('door_name', door_id)

            print(f"\n   [DOOR]  Defining line for: {door_name} ({door_id})")

            # Dynamically get the Windows screen size to fit-to-screen perfectly
            try:
                user32 = ctypes.windll.user32
                screen_w = user32.GetSystemMetrics(0)
                screen_h = user32.GetSystemMetrics(1)
            except:
                screen_w, screen_h = 1920, 1080

            # Scale to 90% of screen to leave room for taskbar/window borders
            max_w = screen_w
            max_h = screen_h

            h, w = canvas.shape[:2]
            scale = min(max_w / w, max_h / h)
            if scale > 1.0: 
                scale = 1.0  # dont upscale small images
            new_w, new_h = int(w * scale), int(h * scale)

            # ── Interaction state ────────────────────────────────────────────
            state = {'step': 'pt1', 'pt1': None, 'pt2': None, 'inside_pt': None, 'scale': scale, 'pause_vid': False}
            window_name = f"Calibrate - {cam_name} - {door_name}"

            def mouse_callback(event, x, y, flags, param):
                # Only process LBUTTONDOWN — NOT LBUTTONUP.
                if event != cv2.EVENT_LBUTTONDOWN:
                    return
                s = param
                sc = s.get('scale', 1.0)
                orig_x, orig_y = int(x / sc), int(y / sc)

                if s['step'] == 'pt1':
                    s['pt1'] = (orig_x, orig_y)
                    s['step'] = 'pt2'
                    s['pause_vid'] = True  # Auto-pause video after 1st click for stability
                    print(f"   [OK]  Point 1 set at {(orig_x, orig_y)} - now click Point 2")

                elif s['step'] == 'pt2':
                    s['pt2'] = (orig_x, orig_y)
                    if s['pt1'] == s['pt2']:
                        print("   [WARN]  Both points are the same. Click a different location.")
                        s['step'] = 'pt2'
                    else:
                        s['step'] = 'inside'
                        print(f"   [OK]  Point 2 set at {(orig_x, orig_y)} - now click INSIDE the room")

                elif s['step'] == 'inside':
                    sign = _cross_sign((orig_x, orig_y), s['pt1'], s['pt2'])
                    if sign == 0:
                        print("   [WARN]  Click is exactly on the line. Click clearly inside the room.")
                        return
                    s['inside_pt']   = (orig_x, orig_y)
                    s['inside_sign'] = sign
                    s['step']        = 'done'
                    print(f"   [OK]  Inside point set at {(orig_x, orig_y)} - press [s] to save")

            # ── Interaction loop ─────────────────────────────────────────────
            quit_all = False
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            first_draw = _draw_state(canvas, state, door_name, cam_name)
            cv2.imshow(window_name, cv2.resize(first_draw, (new_w, new_h)))
            cv2.waitKey(200)          # let the window fully appear + gain focus
            cv2.setMouseCallback(window_name, mouse_callback, state)
            
            help_msg = "   [CLICK]  Window ready. Click in the window to begin."
            if play_video:
                help_msg += "  (Press [SPACE] to pause/play video anytime)"
            print(help_msg)

            while True:
                if play_video and not state.get('pause_vid', False):
                    ret, new_frame = vs.read()
                    if ret and new_frame is not None:
                        canvas = new_frame.copy()

                display = _draw_state(canvas, state, door_name, cam_name)
                display_resized = cv2.resize(display, (new_w, new_h))
                cv2.imshow(window_name, display_resized)
                key = cv2.waitKey(30) & 0xFF

                if key == ord('r'):
                    # Reset this line
                    state.update({'step': 'pt1', 'pt1': None, 'pt2': None,
                                  'inside_pt': None, 'pause_vid': False})
                    print("   [RESET]  Reset. Start over for this line.")

                elif key == ord(' '):
                    state['pause_vid'] = not state.get('pause_vid', False)
                    print(f"   [VIDEO]  Video {'paused' if state.get('pause_vid') else 'playing'}.")

                elif key == ord('s') and state['step'] == 'done':
                    # Save to config
                    line_cfg['start']       = list(state['pt1'])
                    line_cfg['end']         = list(state['pt2'])
                    line_cfg['inside_sign'] = state['inside_sign']
                    print(f"   [OK]  Saved: start={state['pt1']}  "
                          f"end={state['pt2']}  inside_sign={state['inside_sign']}")
                    any_saved = True
                    break

                elif key == ord('s') and state['step'] != 'done':
                    print("   [WARN]  Complete all 3 steps before pressing [s].")

                elif key == ord('q'):
                    print("\n   [QUIT]  Calibration quit by user.")
                    quit_all = True
                    break

            cv2.destroyAllWindows()

            if quit_all:
                cv2.destroyAllWindows()
                # Write whatever was saved so far
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(cfg, f, indent=2, ensure_ascii=False)
                print(f"\n[SAVE] Config saved (partial): {config_path}")
                if play_video: vs.release()
                return False

        # ── Prompt for crossing point (once per camera) ─────────────────
        valid_options = {'foot', 'center', 'top', 'mid-foot'}
        while True:
            try:
                cp_input = _win_prompt(
                    f"\n   [INPUT]  Crossing point for {cam_name}"
                    f" (foot/center/top/mid-foot) [foot]: "
                ).strip().lower()
                if not cp_input:
                    cp_input = 'foot'
                if cp_input in valid_options:
                    cam_cfg['crossing_point'] = cp_input
                    print(f"   [OK]  Crossing point set to: {cp_input}")
                    break
                else:
                    print(f"   [WARN]  Invalid option '{cp_input}'. "
                          f"Choose from: foot, center, top, mid-foot")
            except (EOFError, KeyboardInterrupt):
                cam_cfg['crossing_point'] = 'foot'
                print("\n   [WARN]  Defaulting to: foot")
                break

        # ── Prompt for crossing mode (once per camera) ──────────────────
        valid_modes = {'line', 'region', 'both'}
        current_lines = cam_cfg.get('lines', [])
        has_active_lines = any(
            l.get('start') != [0, 0] or l.get('end') != [0, 0]
            for l in current_lines
        )
        default_mode = 'line' if has_active_lines else 'region'
        while True:
            try:
                cm_input = _win_prompt(
                    f"\n   [INPUT]  Crossing mode for {cam_name}"
                    f" (line/region/both) [{default_mode}]: "
                ).strip().lower()
                if not cm_input:
                    cm_input = default_mode
                if cm_input in valid_modes:
                    cam_cfg['crossing_mode'] = cm_input
                    print(f"   [OK]  Crossing mode set to: {cm_input}")
                    break
                else:
                    print(f"   [WARN]  Invalid option '{cm_input}'. "
                          f"Choose from: line, region, both")
            except (EOFError, KeyboardInterrupt):
                cam_cfg['crossing_mode'] = default_mode
                print(f"\n   [WARN]  Defaulting to: {default_mode}")
                break

        if play_video:
            vs.release()

    # ── Write updated config ──────────────────────────────────────────────────
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"\n[OK]  Calibration complete! Config saved: {config_path}")
    cv2.destroyAllWindows()
    print("    You can now run: python phase3_counter/counter_main.py")
    return any_saved
