"""
DoorLineCrosser — Virtual line crossing detection for person counting.

Each DoorLineCrosser instance manages ALL virtual lines for ONE camera feed.
A camera can have one OR more lines (e.g., if it faces multiple doors or
has a wide entrance split into two counting zones).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALGORITHM: Band-Zone Line Crossing
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Rather than a 1-pixel line, each virtual line has a BAND ZONE (configurable
width in pixels, default 25). This prevents jitter-counts when a person
hesitates in the doorway.

    OUTSIDE ZONE          BAND ZONE           INSIDE ZONE
    (sign = -1)     ░░░░░░░░░░░░░░░░░░░░░     (sign = +1)
                    ░  ← band_width px  ░
                    ░░░░░░░░░░░░░░░░░░░░░

State machine (per line, per tracked person):
  • prev_side: last known side of the line (-1 or +1), only set when
               person is OUTSIDE the band (distance > band_width).
  • In the band:   prev_side is FROZEN — no events, no update.
  • Outside band:  if sign changed vs prev_side → fire event, update.

This means a person must FULLY cross through the band to register a count.
If they stop in the doorway and come back, no count is fired.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GEOMETRY: Cross-Product Side Detection
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

For line defined by start=(x1,y1) → end=(x2,y2):
  sign = sign( (end-start) × (foot-start) )
       = sign( dx*(foot_y - y1) - dy*(foot_x - x1) )
  +1 = foot is on the LEFT  of the line direction
  -1 = foot is on the RIGHT of the line direction

inside_sign (+1 or -1) is set by the calibration tool when the user
clicks on the "inside of the room". The calibration tool computes which
sign corresponds to the room interior and saves it to config.

ENTRY = foot moves TO the inside_sign side   (came from outside, now inside)
EXIT  = foot moves AWAY from inside_sign side (came from inside, now outside)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DIRECTION FILTERING (Optional)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If direction parameter is set, events are filtered based on movement direction:
  • "both" (default): Count both entry and exit (backward compatible)
  • "entry": Only count forward movement INTO the room (ignore backward)
  • "exit": Only count forward movement OUT OF the room (ignore backward)

This requires velocity_dict to be passed to update() method, containing
track_id -> (dx, dy) normalized velocity vectors.
"""

import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Optional


class DoorLineCrosser:
    """
    Detects virtual line crossings for one camera feed.

    Supports multiple lines per camera (1 camera → N doors).

    Args:
        lines_config: List of line configuration dicts, each containing:
            door_id     (str) : Unique door identifier, e.g. "Door_A"
            door_name   (str) : Human-readable name, e.g. "Main Entrance"
            start       (list): [x, y] — pixel coords of one end of the line
            end         (list): [x, y] — pixel coords of the other end
            inside_sign (int) : +1 or -1 — which cross-product sign = inside room.
                                Set automatically by the calibration tool.
            band_width  (int) : Crossing zone width in pixels (default 25).
            direction   (str) : "both" (default), "entry", or "exit".
                               If "entry", only count forward entries.
                               If "exit", only count forward exits.
    """

    def __init__(self, lines_config: List[Dict]):
        self._lines: List[Dict] = []
        for cfg in lines_config:
            direction = cfg.get('direction', 'both')
            if direction not in ('both', 'entry', 'exit'):
                direction = 'both'

            crossing_point = cfg.get('crossing_point', 'foot')
            if crossing_point not in ('foot', 'center', 'top', 'mid-foot'):
                crossing_point = 'foot'

            self._lines.append({
                'door_id'       : cfg['door_id'],
                'door_name'     : cfg.get('door_name', cfg['door_id']),
                'start'         : tuple(cfg['start']),
                'end'           : tuple(cfg['end']),
                'inside_sign'   : int(cfg.get('inside_sign', 1)),
                'band_width'    : int(cfg.get('band_width', 25)),
                'direction'     : direction,
                'crossing_point': crossing_point,
            })

        # State per line: line_index → {track_id: last_known_side}
        # Side is stored only when person is CLEARLY outside the band.
        self._prev_sides: Dict[int, Dict[int, int]] = defaultdict(dict)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Public API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def update(self, tracks: List[Tuple[int, np.ndarray]],
               velocity_dict: Optional[Dict[int, Tuple[float, float]]] = None) -> List[Dict]:
        """
        Process one frame of tracking results and detect crossings.

        Args:
            tracks: List of (track_id, bbox) tuples.
                    bbox = numpy array [x1, y1, x2, y2] in pixel coords.
            velocity_dict: Optional dict mapping track_id to (dx, dy) normalized
                          velocity vectors. If provided, direction filtering
                          will be applied based on each line's 'direction' setting.

        Returns:
            List of crossing event dicts. Each dict contains:
                door_id   (str): e.g. "Door_A"
                door_name (str): e.g. "Main Entrance"
                event     (str): "ENTRY" or "EXIT"
                track_id  (int): BoT-SORT track ID that crossed
                foot      (tuple): (x, y) foot position for debugging
                filtered  (bool): True if event was filtered out due to direction
        """
        events: List[Dict] = []

        if not self._lines:
            return events

        active_ids = {t[0] for t in tracks}
        velocity_dict = velocity_dict or {}

        for line_idx, line in enumerate(self._lines):
            prev_sides = self._prev_sides[line_idx]

            # ── Cleanup: remove state for tracks that have disappeared ──
            vanished = [tid for tid in prev_sides if tid not in active_ids]
            for tid in vanished:
                del prev_sides[tid]

            # ── Validate line coordinates ──
            if line['start'] == (0, 0) and line['end'] == (0, 0):
                # Line not yet calibrated — skip silently
                continue

            start       = line['start']
            end         = line['end']
            inside_sign = line['inside_sign']
            band_width  = line['band_width']
            direction   = line.get('direction', 'both')

            # Compute normalized line direction for movement comparison
            line_dx = end[0] - start[0]
            line_dy = end[1] - start[1]
            line_mag = (line_dx * line_dx + line_dy * line_dy) ** 0.5
            if line_mag > 0:
                line_dir = (line_dx / line_mag, line_dy / line_mag)
            else:
                line_dir = (0.0, 0.0)

            for track_id, bbox in tracks:
                x1, y1, x2, y2 = map(int, bbox)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # Use crossing point based on config
                cp_type = line.get('crossing_point', 'foot')
                if cp_type == 'foot':
                    foot = (cx, y2)
                elif cp_type == 'center':
                    foot = (cx, cy)
                elif cp_type == 'top':
                    foot = (cx, y1)
                elif cp_type == 'mid-foot':
                    foot = (cx, (cy + y2) // 2)
                else:
                    foot = (cx, y2)

                # Distance from foot to the virtual line
                dist = self._dist_to_line(foot, start, end)

                # Which side of the line is the foot on?
                sign = self._cross_sign(foot, start, end)

                # Degenerate: exactly on the line (rare edge case)
                if sign == 0:
                    continue

                # ── Inside the band: freeze state, no action ──
                if dist <= band_width:
                    continue

                # ── Clearly outside the band ──
                if track_id in prev_sides:
                    prev_sign = prev_sides[track_id]

                    if prev_sign != sign:
                        # ══ CROSSING DETECTED ══
                        # Person was on one side, is now clearly on the other.
                        event_type = 'ENTRY' if sign == inside_sign else 'EXIT'

                        # ── Direction Filtering ──
                        # If direction is "entry" or "exit", check movement direction
                        filtered = False
                        if direction != 'both' and track_id in velocity_dict:
                            dx, dy = velocity_dict[track_id]

                            # Determine if movement is forward or backward
                            if dx != 0 or dy != 0:
                                dot = dx * line_dir[0] + dy * line_dir[1]

                                # Determine actual movement direction
                                movement_is_forward = dot > 0.1
                                movement_is_backward = dot < -0.1

                                if direction == 'entry':
                                    # Only count forward movement INTO room
                                    if event_type == 'EXIT':
                                        filtered = True
                                    elif event_type == 'ENTRY' and not movement_is_forward:
                                        filtered = True

                                elif direction == 'exit':
                                    # Only count forward movement OUT OF room
                                    if event_type == 'ENTRY':
                                        filtered = True
                                    elif event_type == 'EXIT' and not movement_is_forward:
                                        filtered = True

                        # Only add event if not filtered
                        if not filtered:
                            events.append({
                                'door_id'  : line['door_id'],
                                'door_name': line['door_name'],
                                'event'    : event_type,
                                'track_id' : track_id,
                                'foot'     : foot,
                            })

                # Update side record (only when clearly outside band)
                prev_sides[track_id] = sign

        return events

    def reset(self):
        """Clear all tracking state. Call between sessions or video files."""
        self._prev_sides.clear()

    def get_line_configs(self) -> List[Dict]:
        """Return line configurations (used by annotation layer)."""
        return list(self._lines)

    def is_calibrated(self) -> bool:
        """Return True if ALL lines have non-zero coordinates."""
        for line in self._lines:
            if line['start'] == (0, 0) and line['end'] == (0, 0):
                return False
        return True

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Geometry helpers
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    @staticmethod
    def _cross_sign(point: Tuple[int, int],
                    line_start: Tuple[int, int],
                    line_end:   Tuple[int, int]) -> int:
        """
        Sign of the 2D cross product: (line_end - line_start) × (point - line_start).

        Returns:
            +1: point is on the LEFT  of the direction line_start → line_end
            -1: point is on the RIGHT of the direction line_start → line_end
             0: point is exactly on the line (collinear)
        """
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        px = point[0]    - line_start[0]
        py = point[1]    - line_start[1]
        cross = dx * py - dy * px
        if   cross > 0: return  1
        elif cross < 0: return -1
        return 0

    @staticmethod
    def _dist_to_line(point:      Tuple[int, int],
                      line_start: Tuple[int, int],
                      line_end:   Tuple[int, int]) -> float:
        """
        Perpendicular distance from point to the infinite line through
        line_start and line_end.
        """
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-9:
            # Degenerate: start and end are the same point
            return ((point[0] - line_start[0]) ** 2 +
                    (point[1] - line_start[1]) ** 2) ** 0.5
        cross_abs = abs(dx * (line_start[1] - point[1]) -
                        (line_start[0] - point[0]) * dy)
        return cross_abs / length
