
"""
DeepSORT Tracker - Main tracking algorithm
Combines Kalman filter motion prediction with Re-ID appearance features
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman_filter import KalmanFilter
from .track import Track, TrackState


def iou(bbox1, bbox2):
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]"""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0


def cosine_distance(features1, features2):
    """Compute cosine distance between feature sets"""
    if len(features1) == 0 or len(features2) == 0:
        return 1.0
    
    features1 = np.array(features1)
    features2 = np.array(features2)
    
    # Normalize
    features1 = features1 / (np.linalg.norm(features1, axis=1, keepdims=True) + 1e-8)
    features2 = features2 / (np.linalg.norm(features2, axis=1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity
    similarity = np.dot(features1, features2.T)
    
    # Return minimum distance (maximum similarity)
    return 1 - np.max(similarity)


class DeepSORTTracker:
    """
    DeepSORT multi-object tracker.
    
    Combines:
    1. Kalman filter for motion prediction
    2. Re-ID features for appearance matching
    3. Hungarian algorithm for optimal assignment
    """
    
    def __init__(self, max_age=30, n_init=3, max_iou_distance=0.7, 
                 max_cosine_distance=0.3):
        """
        Args:
            max_age: Max frames to keep track without detection
            n_init: Frames before track is confirmed
            max_iou_distance: Max IoU distance for matching
            max_cosine_distance: Max cosine distance for Re-ID matching
        """
        self.max_age = max_age
        self.n_init = n_init
        self.max_iou_distance = max_iou_distance
        self.max_cosine_distance = max_cosine_distance
        
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1
    
    def update(self, detections, features):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of [x1, y1, x2, y2, confidence] detections
            features: List of Re-ID feature vectors (same length as detections)
            
        Returns:
            List of (track_id, bbox) for confirmed tracks
        """
        # Predict new locations of existing tracks
        for track in self.tracks:
            track.predict(self.kf)
        
        # Convert detections to center format [cx, cy, w, h]
        det_centers = []
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            det_centers.append([cx, cy, w, h])
        
        # Split tracks into confirmed and unconfirmed
        confirmed_tracks = [t for t in self.tracks if t.is_confirmed()]
        unconfirmed_tracks = [t for t in self.tracks if t.is_tentative()]
        
        # Match confirmed tracks using appearance + motion
        matches_a, unmatched_tracks_a, unmatched_detections = \
            self._match_confirmed(confirmed_tracks, det_centers, features, detections)
        
        # Match remaining tracks using IoU only
        matches_b, unmatched_tracks_b, unmatched_detections = \
            self._match_unconfirmed(unconfirmed_tracks, det_centers, features, 
                                   detections, unmatched_detections)
        
        # Update matched tracks
        for track_idx, det_idx in matches_a + matches_b:
            if track_idx < len(confirmed_tracks):
                track = confirmed_tracks[track_idx]
            else:
                track = unconfirmed_tracks[track_idx - len(confirmed_tracks)]
            
            feature = features[det_idx] if features is not None else None
            track.update(self.kf, det_centers[det_idx], feature)
        
        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks_a:
            confirmed_tracks[track_idx].mark_missed()
        for track_idx in unmatched_tracks_b:
            unconfirmed_tracks[track_idx].mark_missed()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            self._initiate_track(det_centers[det_idx], 
                               features[det_idx] if features is not None else None)
        
        # Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # Return confirmed tracks
        results = []
        for track in self.tracks:
            if track.is_confirmed():
                bbox = track.to_tlbr()
                results.append((track.track_id, bbox))
        
        return results
    
    def _match_confirmed(self, tracks, detections, features, raw_detections):
        """Match confirmed tracks using appearance and motion"""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))
        
        # Compute cost matrix
        cost_matrix = np.zeros((len(tracks), len(detections)))
        
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # Appearance cost (cosine distance)
                if features is not None and len(track.features) > 0:
                    appearance_cost = cosine_distance(track.features, [features[j]])
                else:
                    appearance_cost = 0
                
                # Motion cost (IoU)
                track_bbox = track.to_tlbr()
                det_bbox = raw_detections[j][:4]
                iou_cost = 1 - iou(track_bbox, det_bbox)
                
                # Combined cost
                cost_matrix[i, j] = 0.5 * appearance_cost + 0.5 * iou_cost
        
        # Apply gating
        cost_matrix[cost_matrix > self.max_cosine_distance] = 1e5
        
        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(range(len(detections)))
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 1e5:
                matches.append((row, col))
                unmatched_tracks.remove(row)
                unmatched_detections.remove(col)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _match_unconfirmed(self, tracks, detections, features, raw_detections, 
                          detection_indices):
        """Match unconfirmed tracks using IoU only"""
        if len(tracks) == 0 or len(detection_indices) == 0:
            return [], list(range(len(tracks))), detection_indices
        
        # Compute IoU matrix
        iou_matrix = np.zeros((len(tracks), len(detection_indices)))
        
        for i, track in enumerate(tracks):
            track_bbox = track.to_tlbr()
            for j, det_idx in enumerate(detection_indices):
                det_bbox = raw_detections[det_idx][:4]
                iou_matrix[i, j] = iou(track_bbox, det_bbox)
        
        # Convert to cost (1 - IoU)
        cost_matrix = 1 - iou_matrix
        cost_matrix[cost_matrix > self.max_iou_distance] = 1e5
        
        # Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        matches = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_detections = list(detection_indices)
        
        for row, col in zip(row_indices, col_indices):
            if cost_matrix[row, col] < 1e5:
                det_idx = detection_indices[col]
                matches.append((row + len([t for t in self.tracks if t.is_confirmed()]), det_idx))
                unmatched_tracks.remove(row)
                unmatched_detections.remove(det_idx)
        
        return matches, unmatched_tracks, unmatched_detections
    
    def _initiate_track(self, detection, feature):
        """Create new track"""
        mean, covariance = self.kf.initiate(detection)
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age, feature
        ))
        self._next_id += 1
