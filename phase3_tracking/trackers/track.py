"""
Track Class for Object Tracking

Represents a single tracked object (person) with:
- Kalman filter state
- Track history
- Re-ID feature history
"""

import numpy as np


class TrackState:
    """
    Track state enumeration.
    
    Tentative: New track, waiting for confirmation
    Confirmed: Track is confirmed and active
    Deleted: Track is marked for deletion
    """
    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    Represents a single tracked object.
    
    Attributes:
        track_id: Unique identifier for this track
        hits: Number of successful detections
        age: Total frames since track creation
        time_since_update: Frames since last detection
        state: Current track state (Tentative/Confirmed/Deleted)
        features: List of Re-ID feature vectors
    """
    
    _next_id = 1
    
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        """
        Initialize a new track.
        
        Args:
            mean: Initial Kalman filter state mean [8]
            covariance: Initial Kalman filter state covariance [8, 8]
            track_id: Unique track identifier
            n_init: Number of consecutive detections to confirm track
            max_age: Maximum frames to keep track without detection
            feature: Optional initial Re-ID feature vector
        """
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        
        self.state = TrackState.Tentative
        
        self._n_init = n_init
        self._max_age = max_age
        
        # Store Re-ID features for appearance matching
        self.features = []
        if feature is not None:
            self.features.append(feature)
    
    def to_tlwh(self):
        """
        Get bounding box in top-left-width-height format.
        
        Returns:
            [x, y, w, h] where (x, y) is top-left corner
        """
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2  # Convert center to top-left
        return ret
    
    def to_tlbr(self):
        """
        Get bounding box in top-left-bottom-right format.
        
        Returns:
            [x1, y1, x2, y2] where (x1, y1) is top-left, (x2, y2) is bottom-right
        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret
    
    def to_xyah(self):
        """
        Get bounding box in center-aspect-height format.
        
        Returns:
            [cx, cy, aspect_ratio, height]
        """
        ret = self.mean[:4].copy()
        ret[2] /= ret[3]  # width / height = aspect ratio
        return ret
    
    def predict(self, kf):
        """
        Predict next state using Kalman filter.
        
        Args:
            kf: KalmanFilter instance
        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
    
    def update(self, kf, detection, feature=None):
        """
        Update track with new detection.
        
        Args:
            kf: KalmanFilter instance
            detection: New detection [x, y, w, h] in center format
            feature: Optional Re-ID feature vector
        """
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection
        )
        
        # Store feature for appearance matching
        if feature is not None:
            self.features.append(feature)
            # Keep only recent features (memory efficiency)
            if len(self.features) > 100:
                self.features = self.features[-100:]
        
        self.hits += 1
        self.time_since_update = 0
        
        # Confirm track if enough hits
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
    
    def mark_missed(self):
        """Mark track as missed (no detection this frame)."""
        if self.state == TrackState.Tentative:
            # Delete tentative tracks immediately
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            # Delete confirmed tracks after max_age frames
            self.state = TrackState.Deleted
    
    def is_tentative(self):
        """Check if track is tentative."""
        return self.state == TrackState.Tentative
    
    def is_confirmed(self):
        """Check if track is confirmed."""
        return self.state == TrackState.Confirmed
    
    def is_deleted(self):
        """Check if track is deleted."""
        return self.state == TrackState.Deleted
