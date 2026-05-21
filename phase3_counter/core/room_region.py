"""
RoomRegion — Polygon-based room interior region detection.

This module provides polygon-based room region detection to handle cases
where a camera can see multiple rooms. Only people inside the defined
room polygon are counted.

Usage:
    region = RoomRegion(polygon_points)  # List of (x, y) tuples
    is_inside = region.is_point_inside(x, y)
    inside_tracks = region.filter_tracks(tracks)
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


class RoomRegion:
    """
    Defines a room interior as a polygon for filtering people counts.

    When a camera can see multiple rooms, this allows filtering to only
    count people inside the specified room region.
    """

    def __init__(self, polygon: List[Tuple[int, int]]):
        """
        Initialize room region with polygon vertices.

        Args:
            polygon: List of (x, y) tuples defining the polygon vertices
                    in clockwise or counter-clockwise order.
                    Must have at least 3 points.
        """
        if len(polygon) < 3:
            raise ValueError("Room region polygon must have at least 3 points")

        self.polygon = polygon
        self._cached_edges = self._precompute_edges()

    def _precompute_edges(self):
        """Precompute normalized edge vectors and constants for point-in-polygon test."""
        n = len(self.polygon)
        edges = []

        for i in range(n):
            x1, y1 = self.polygon[i]
            x2, y2 = self.polygon[(i + 1) % n]

            # Edge equation: a*x + b*y = c
            # where edge goes from (x1,y1) to (x2,y2)
            a = y2 - y1
            b = x1 - x2
            c = a * x1 + b * y1

            edges.append((a, b, c))

        return edges

    def is_point_inside(self, x: int, y: int) -> bool:
        """
        Check if a point is inside the polygon using ray casting algorithm.

        Args:
            x, y: Point coordinates

        Returns:
            True if point is inside the polygon, False otherwise
        """
        n = len(self.polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = self.polygon[i]
            xj, yj = self.polygon[j]

            # Check if ray from point crosses edge
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside

            j = i

        return inside

    def is_point_inside_smooth(self, x: float, y: float) -> bool:
        """
        Check if a point is inside (float version for precise checks).

        Args:
            x, y: Point coordinates (can be float)

        Returns:
            True if point is inside the polygon, False otherwise
        """
        return self.is_point_inside(int(x), int(y))

    def get_foot_position(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Extract foot position (bottom-center) from bounding box.

        Args:
            bbox: (x1, y1, x2, y2) bounding box

        Returns:
            (foot_x, foot_y) foot position
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) // 2, y2)

    def is_track_inside(self, track_id: int, bbox: np.ndarray) -> bool:
        """
        Check if a tracked person's foot is inside the room region.

        Args:
            track_id: Track ID (for compatibility)
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            True if person's foot is inside the room
        """
        foot_x, foot_y = self.get_foot_position(bbox)
        return self.is_point_inside(foot_x, foot_y)

    def filter_tracks(self, tracks: List[Tuple[int, np.ndarray]]) -> List[Tuple[int, np.ndarray]]:
        """
        Filter tracks to only include those inside the room region.

        Args:
            tracks: List of (track_id, bbox) tuples

        Returns:
            Filtered list of tracks inside the room
        """
        inside_tracks = []

        for track_id, bbox in tracks:
            if self.is_track_inside(track_id, bbox):
                inside_tracks.append((track_id, bbox))

        return inside_tracks

    def count_inside(self, tracks: List[Tuple[int, np.ndarray]]) -> int:
        """
        Count number of tracked people currently inside the room.

        Args:
            tracks: List of (track_id, bbox) tuples

        Returns:
            Number of people inside the room
        """
        return len(self.filter_tracks(tracks))

    def get_inside_info(self, tracks: List[Tuple[int, np.ndarray]]) -> Dict:
        """
        Get detailed info about tracks inside the room.

        Args:
            tracks: List of (track_id, bbox) tuples

        Returns:
            Dict with inside count, outside count, and track IDs
        """
        inside_tracks = self.filter_tracks(tracks)
        outside_tracks = [(tid, bbox) for tid, bbox in tracks
                         if not self.is_track_inside(tid, bbox)]

        return {
            'inside_count': len(inside_tracks),
            'outside_count': len(outside_tracks),
            'inside_track_ids': [tid for tid, _ in inside_tracks],
            'inside_tracks': inside_tracks,
            'outside_tracks': outside_tracks
        }

    def is_valid(self) -> bool:
        """Check if region is valid (has polygon defined)."""
        return len(self.polygon) >= 3 and all(
            p[0] > 0 or p[1] > 0 for p in self.polygon
        )

    @staticmethod
    def from_dict(config: Dict) -> Optional['RoomRegion']:
        """
        Create RoomRegion from configuration dict.

        Args:
            config: Dict containing 'polygon' key with list of [x, y] points

        Returns:
            RoomRegion instance or None if not configured
        """
        polygon_list = config.get('polygon', [])

        if not polygon_list or len(polygon_list) < 3:
            return None

        polygon = [tuple(point) for point in polygon_list]

        try:
            return RoomRegion(polygon)
        except ValueError:
            return None


class RoomRegionManager:
    """
    Manages room regions for multiple cameras.

    Each camera can have its own room region definition.
    """

    def __init__(self, regions_config: Optional[List[Dict]] = None):
        """
        Initialize room region manager.

        Args:
            regions_config: List of camera region configs,
                           each containing 'camera_name' and 'polygon'
        """
        self._regions: Dict[str, RoomRegion] = {}

        if regions_config:
            for cfg in regions_config:
                camera_name = cfg.get('camera_name', cfg.get('name', 'default'))
                polygon = cfg.get('polygon', [])

                if len(polygon) >= 3:
                    region = RoomRegion([tuple(p) for p in polygon])
                    self._regions[camera_name] = region

    def get_region(self, camera_name: str) -> Optional[RoomRegion]:
        """Get room region for a specific camera."""
        return self._regions.get(camera_name)

    def has_region(self, camera_name: str) -> bool:
        """Check if camera has a region defined."""
        return camera_name in self._regions

    def add_region(self, camera_name: str, polygon: List[Tuple[int, int]]):
        """Add or update a room region for a camera."""
        if len(polygon) >= 3:
            self._regions[camera_name] = RoomRegion(polygon)

    def filter_tracks_for_camera(self, camera_name: str,
                                  tracks: List[Tuple[int, np.ndarray]]) -> List[Tuple[int, np.ndarray]]:
        """
        Filter tracks for a specific camera based on its room region.

        If no region is defined for the camera, returns all tracks.
        """
        region = self.get_region(camera_name)

        if region is None:
            return tracks  # No region defined, return all

        return region.filter_tracks(tracks)

    def get_inside_count_for_camera(self, camera_name: str,
                                     tracks: List[Tuple[int, np.ndarray]]) -> int:
        """Get count of tracks inside room for a specific camera."""
        region = self.get_region(camera_name)

        if region is None:
            return len(tracks)  # No region, count all

        return region.count_inside(tracks)