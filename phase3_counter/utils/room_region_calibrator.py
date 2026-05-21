"""
Room Region Calibrator — Interactive polygon drawing tool for room interior.

This tool allows users to click and draw a polygon defining the room interior.
The drawn polygon is saved to the counter_config.json for use during counting.

Usage:
    python phase3_counter/utils/room_region_calibrator.py --config phase3_counter/config/counter_config.json

Controls:
    Left Click     : Add polygon point
    'u'            : Undo last point
    'c'            : Clear all points
    's'            : Save and exit
    'q'            : Quit without saving
    Spacebar       : Play/pause video (if video mode)
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict

# Add parent directories to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Import VideoSource from parent
from multi_camera_tracker import VideoSource


class RoomRegionCalibrator:
    """
    Interactive tool to draw room interior polygon.
    """

    def __init__(self, config_path: str, play_video: bool = False):
        """
        Initialize calibrator.

        Args:
            config_path: Path to counter_config.json
            play_video: If True, play video while calibrating
        """
        self.config_path = config_path
        self.play_video = play_video

        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        self.cameras = self.config.get('cameras', [])

        if not self.cameras:
            print("❌ No cameras defined in config.")
            sys.exit(1)

        # State
        self.current_cam_idx = 0
        self.current_cam_name = ''
        self.current_scale = 1.0
        self.polygons: Dict[str, List[Tuple[int, int]]] = {}
        self._crossing_points: Dict[str, str] = {}
        self._crossing_modes: Dict[str, str] = {}
        self.drawing = True

        # Load existing polygons from config
        for cam in self.cameras:
            cam_name = cam.get('name', 'Camera')
            region = cam.get('room_region', {})
            polygon = region.get('polygon', [])
            if polygon:
                self.polygons[cam_name] = [tuple(p) for p in polygon]

    def run(self):
        """Run the calibration for all cameras."""
        print("\n" + "=" * 60)
        print("🖊️  ROOM REGION CALIBRATOR")
        print("=" * 60)
        print("\nInstructions:")
        print("  • Left Click  : Add polygon point")
        print("  • 'u'         : Undo last point")
        print("  • 'c'         : Clear all points")
        print("  • 's'         : Save and move to next camera")
        print("  • 'q'         : Quit without saving")
        print("  • Spacebar    : Play/pause video (video mode only)")
        print("=" * 60 + "\n")

        for cam_idx, cam in enumerate(self.cameras):
            self.current_cam_idx = cam_idx
            self._calibrate_camera(cam, cam_idx)

        self._save_config()

    def _on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            scale = self.current_scale
            orig_x = int(x / scale) if scale != 1.0 else x
            orig_y = int(y / scale) if scale != 1.0 else y
            self._add_point(self.current_cam_name, (orig_x, orig_y))

    def _calibrate_camera(self, cam: Dict, cam_idx: int = 0):
        """Calibrate room region for one camera."""
        cam_name = cam.get('name', f'Camera_{cam_idx + 1}')
        source = cam.get('source', '')

        print(f"\n📷 Camera: {cam_name}")
        print(f"   Source: {source}")

        # Initialize polygon for this camera
        if cam_name not in self.polygons:
            self.polygons[cam_name] = []

        points = self.polygons[cam_name]

        # Open video source
        vs = VideoSource(source)
        if not vs.open():
            print(f"   ❌ Cannot open source: {source}")
            return

        info = vs.get_info()
        print(f"   Resolution: {info['width']}x{info['height']}")
        print(f"   FPS: {info['fps']:.1f}")

        # Create window
        window_name = f"Room Region - {cam_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_w, display_h)

        # Try to fit window to screen
        screen_w = 1280
        screen_h = 720
        self.current_cam_name = cam_name
        self.current_scale = min(screen_w / info['width'], screen_h / info['height'], 1.0)
        scale = self.current_scale
        display_w = int(info['width'] * scale)
        display_h = int(info['height'] * scale)
        cv2.resizeWindow(window_name, display_w, display_h)

        print(f"\n   📝 Draw polygon for room interior.")
        print(f"      Click points to define the room boundary.")
        print(f"      Press 's' to save when done.")

        frame = None
        playing = self.play_video
        mouse_callback_attached = False

        while True:
            # Get frame
            if playing or frame is None:
                ret, frame = vs.read()
                if not ret:
                    if info['is_stream']:
                        continue
                    else:
                        vs.release()
                        vs = VideoSource(source)
                        vs.open()
                        ret, frame = vs.read()
                        if not ret:
                            break

            if frame is None:
                break

            # Attach mouse callback once
            if not mouse_callback_attached:
                cv2.setMouseCallback(window_name, self._on_mouse_click)
                mouse_callback_attached = True

            # Draw
            display = frame.copy()

            if points:
                pts = np.array(points, dtype=np.int32)
                pts = pts.reshape((-1, 1, 2))

                overlay = display.copy()
                cv2.fillPoly(overlay, [pts], (0, 200, 0))
                cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)

                cv2.polylines(display, [pts], True, (0, 255, 0), 2)

                for i, pt in enumerate(points):
                    color = (0, 255, 0) if i == len(points) - 1 else (0, 255, 255)
                    cv2.circle(display, pt, 8, color, -1)

                for i in range(len(points) - 1):
                    cv2.line(display, points[i], points[i+1], (0, 255, 255), 2)

            overlay_bg = display.copy()
            cv2.rectangle(overlay_bg, (5, 5), (350, 110), (0, 0, 0), -1)
            cv2.addWeighted(overlay_bg, 0.5, display, 0.5, 0, display)
            cv2.putText(display, f"Room: {cam_name}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"Points: {len(points)}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(display, "Click=Add Point", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
            cv2.putText(display, "u=Undo  c=Clear  s=Save  q=Quit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            if scale != 1.0:
                display_small = cv2.resize(display, (display_w, display_h))
                cv2.imshow(window_name, display_small)
            else:
                cv2.imshow(window_name, display)

            # Wait for key
            key = cv2.waitKey(1 if playing else 30) & 0xFF

            # Handle keys
            if key == ord('q'):
                print(f"\n   ❌ Calibration cancelled for {cam_name}")
                vs.release()
                cv2.destroyWindow(window_name)
                return

            elif key == ord('s'):
                print(f"   ✅ Saved {len(points)} points for {cam_name}")
                break

            elif key == ord('u'):
                if points:
                    points.pop()
                    print(f"   ↩️  Undo - {len(points)} points remaining")

            elif key == ord('c'):
                points.clear()
                print(f"   🗑️  Cleared all points")

            elif key == ord(' ') and info['is_stream']:
                playing = not playing
                print(f"   {'▶️' if playing else '⏸️'} {'Playing' if playing else 'Paused'}")

        vs.release()
        cv2.destroyWindow(window_name)

        # Store points
        self.polygons[cam_name] = points

        # ── Prompt for crossing point ────────────────────────────────────
        if len(points) >= 3:
            valid_options = {'foot', 'center', 'top', 'mid-foot'}
            prompt_text = (f"\n   🎯  Crossing point for {cam_name}"
                          f" (foot/center/top/mid-foot) [foot]: ")
            while True:
                try:
                    cp_input = input(prompt_text).strip().lower()
                    if not cp_input:
                        cp_input = 'foot'
                    if cp_input in valid_options:
                        self._crossing_points[cam_name] = cp_input
                        print(f"   ✅  Crossing point set to: {cp_input}")
                        break
                    else:
                        print(f"   ⚠️  Invalid option '{cp_input}'. "
                             f"Choose from: foot, center, top, mid-foot")
                except (EOFError, KeyboardInterrupt):
                    self._crossing_points[cam_name] = 'foot'
                    print("\n   ⚠️  Defaulting to: foot")
                    break

        # ── Prompt for crossing mode ────────────────────────────────────
        if len(points) >= 3:
            valid_modes = {'line', 'region', 'both'}
            lines_cfg = self.config.get('cameras', [])[self.current_cam_idx].get('lines', [])
            has_active_lines = any(
                l.get('start') != [0, 0] or l.get('end') != [0, 0]
                for l in lines_cfg
            ) if lines_cfg else False
            default_mode = 'line' if has_active_lines else 'region'
            mode_prompt = (f"\n   🎯  Crossing mode for {cam_name}"
                           f" (line/region/both) [{default_mode}]: ")
            while True:
                try:
                    cm_input = input(mode_prompt).strip().lower()
                    if not cm_input:
                        cm_input = default_mode
                    if cm_input in valid_modes:
                        self._crossing_modes[cam_name] = cm_input
                        print(f"   ✅  Crossing mode set to: {cm_input}")
                        break
                    else:
                        print(f"   ⚠️  Invalid option '{cm_input}'. "
                              f"Choose from: line, region, both")
                except (EOFError, KeyboardInterrupt):
                    self._crossing_modes[cam_name] = default_mode
                    print(f"\n   ⚠️  Defaulting to: {default_mode}")
                    break

    def _add_point(self, cam_name: str, point: Tuple[int, int]):
        """Add a point to the polygon."""
        points = self.polygons[cam_name]
        points.append(point)
        print(f"   + Point ({point[0]}, {point[1]}) - Total: {len(points)}")

    def _save_config(self):
        """Save polygons and crossing points to config file."""
        # Update config
        for cam in self.cameras:
            cam_name = cam.get('name', 'Camera')
            points = self.polygons.get(cam_name, [])

            if len(points) >= 3:
                cam['room_region'] = {
                    'polygon': [list(p) for p in points]
                }
            elif 'room_region' in cam:
                del cam['room_region']

            # Save crossing point if set
            if cam_name in self._crossing_points:
                cam['crossing_point'] = self._crossing_points[cam_name]

            # Save crossing mode if set
            if cam_name in self._crossing_modes:
                cam['crossing_mode'] = self._crossing_modes[cam_name]

        # Save to file
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)

        print("\n" + "=" * 60)
        print("💾 Configuration saved!")
        print("=" * 60)

        for cam_name, points in self.polygons.items():
            if len(points) >= 3:
                print(f"   ✅ {cam_name}: {len(points)} points")
            else:
                print(f"   ⚠️ {cam_name}: Not configured (need 3+ points)")


def run_calibration(config_path: str = None, play_video: bool = False):
    """
    Run room region calibration.

    Args:
        config_path: Path to counter_config.json
        play_video: If True, play video while calibrating
    """
    if config_path is None:
        _THIS_DIR = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(_THIS_DIR, '..', 'config', 'counter_config.json')

    calibrator = RoomRegionCalibrator(config_path, play_video)
    calibrator.run()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Room Region Calibrator')
    parser.add_argument('--config', type=str,
                       default='phase3_counter/config/counter_config.json',
                       help='Path to counter_config.json')
    parser.add_argument('--calib-vid', action='store_true',
                       help='Play video while calibrating')
    args = parser.parse_args()

    run_calibration(args.config, args.calib_vid)