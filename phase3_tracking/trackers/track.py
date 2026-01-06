
"""
Track class - represents a single tracked person
"""

import numpy as np


class TrackState:
    """Enumeration of track states"""
    Tentative = 1   # New track, not yet confirmed
    Confirmed = 2   # Track is confirmed
    Deleted = 3     # Track is deleted


class Track:
    """
    Represents a single tracked person.
    
    Attributes:
        track_id: Unique ID for this track
        hits: Number of times detected
        age: Number of frames since creation
        time_since_update: Frames since last detection
        state: Current track state
        features: List of Re-ID feature vectors
    """
    
    _next_id = 1
    
    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        """
        Args:
            mean: Initial state mean (from Kalman filter)
            covariance: Initial state covariance
            track_id: Unique track ID
            n_init: Frames before track is confirmed
            max_age: Max frames to keep track without detection
            feature: Initial Re-ID feature vector
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
        """Get bounding box in format (top-left x, top-left y, width, height)"""
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2  # Convert center to top-left
        return ret
    
    def to_tlbr(self):
        """Get bounding box in format (top-left x, top-left y, bottom-right x, bottom-right y)"""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret
    
    def predict(self, kf):
        """Predict next state using Kalman filter"""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
    
    def update(self, kf, detection, feature=None):
        """
        Update track with new detection.
        
        Args:
            kf: Kalman filter instance
            detection: New detection [x, y, w, h]
            feature: Re-ID feature vector (optional)
        """
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection)
        
        if feature is not None:
            self.features.append(feature)
            # Keep only last 100 features
            if len(self.features) > 100:
                self.features = self.features[-100:]
        
        self.hits += 1
        self.time_since_update = 0
        
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed
    
    def mark_missed(self):
        """Mark track as missed (no detection this frame)"""
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
    
    def is_tentative(self):
        return self.state == TrackState.Tentative
    
    def is_confirmed(self):
        return self.state == TrackState.Confirmed
    
    def is_deleted(self):
        return self.state == TrackState.Deleted
