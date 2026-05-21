"""
MovementDirectionTracker — Tracks foot positions to compute movement direction.

This module tracks the foot positions of tracked persons over time and computes
velocity vectors (dx, dy) between consecutive frames. This information is used
to determine if a person is moving forward (into room) or backward (out of room)
relative to the door line direction.

Usage:
    tracker = MovementDirectionTracker()

    # In each frame:
    velocity_dict = tracker.update(tracks)

    # tracks = List of (track_id, bbox) tuples
    # velocity_dict = {track_id: (dx, dy)} velocity vectors
"""

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


class MovementDirectionTracker:
    """
    Tracks foot positions over time to compute movement direction.

    Maintains a history of foot positions per track_id and computes
    velocity vector (dx, dy) between consecutive frames.

    The velocity is normalized and represents the direction of movement,
    which can be compared against the door line direction to determine
    if the person is moving forward or backward.
    """

    def __init__(self, max_history: int = 5):
        """
        Initialize the movement direction tracker.

        Args:
            max_history: Maximum number of previous positions to keep per track.
                        Default 5 provides enough history for smoothing.
        """
        self._foot_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self._max_history = max_history

    def update(self, tracks: List[Tuple[int, np.ndarray]]) -> Dict[int, Tuple[float, float]]:
        """
        Update foot positions and compute velocity vectors.

        Args:
            tracks: List of (track_id, bbox) tuples where bbox is
                   numpy array [x1, y1, x2, y2] in pixel coordinates.

        Returns:
            Dict mapping track_id -> (dx, dy) normalized velocity vector.
            Returns empty dict if no valid velocities (first frame or no tracks).
        """
        velocity_dict: Dict[int, Tuple[float, float]] = {}

        for track_id, bbox in tracks:
            # Compute foot position (bottom-center of bbox)
            x1, y1, x2, y2 = map(int, bbox)
            foot_x = (x1 + x2) // 2
            foot_y = y2

            # Get previous foot positions for this track
            history = self._foot_history[track_id]

            if history:
                # Compute velocity based on most recent position
                prev_x, prev_y = history[-1]

                dx = foot_x - prev_x
                dy = foot_y - prev_y

                # Normalize velocity
                magnitude = np.sqrt(dx * dx + dy * dy)
                if magnitude > 0:
                    dx_norm = dx / magnitude
                    dy_norm = dy / magnitude
                    velocity_dict[track_id] = (dx_norm, dy_norm)
                else:
                    # No movement (standing still)
                    velocity_dict[track_id] = (0.0, 0.0)

            # Update history
            history.append((foot_x, foot_y))

            # Keep only last max_history positions
            if len(history) > self._max_history:
                history.pop(0)

        # Clean up history for tracks that no longer exist
        active_ids = {track_id for track_id, _ in tracks}
        removed_ids = set(self._foot_history.keys()) - active_ids
        for track_id in removed_ids:
            del self._foot_history[track_id]

        return velocity_dict

    def get_direction(self, track_id: int, line_direction: Tuple[float, float]) -> str:
        """
        Get movement direction relative to a line direction.

        Args:
            track_id: The track ID to check.
            line_direction: Normalized direction vector of the door line
                          (from start to end).

        Returns:
            "forward" if moving in same direction as line,
            "backward" if moving in opposite direction,
            "none" if no movement or track not found.
        """
        if track_id not in self._foot_history or len(self._foot_history[track_id]) < 2:
            return "none"

        history = self._foot_history[track_id]
        prev_x, prev_y = history[-2]
        curr_x, curr_y = history[-1]

        dx = curr_x - prev_x
        dy = curr_y - prev_y

        magnitude = np.sqrt(dx * dx + dy * dy)
        if magnitude < 1.0:  # Threshold for "standing still"
            return "none"

        # Normalize
        dx_norm = dx / magnitude
        dy_norm = dy / magnitude

        # Dot product to determine direction
        dot = dx_norm * line_direction[0] + dy_norm * line_direction[1]

        if dot > 0.1:  # Threshold for "forward"
            return "forward"
        elif dot < -0.1:  # Threshold for "backward"
            return "backward"
        else:
            return "none"  # Perpendicular movement

    def reset(self):
        """Clear all tracking history. Call between sessions or video files."""
        self._foot_history.clear()

    def get_foot_position(self, bbox: np.ndarray) -> Tuple[int, int]:
        """
        Extract foot position from bounding box.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            (foot_x, foot_y) tuple
        """
        x1, y1, x2, y2 = map(int, bbox)
        return ((x1 + x2) // 2, y2)