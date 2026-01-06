"""
Kalman Filter for Object Tracking

Predicts object positions in the next frame based on motion model.
Uses constant velocity assumption.

State vector: [x, y, w, h, vx, vy, vw, vh]
- (x, y): Bounding box center
- (w, h): Bounding box width and height
- (vx, vy, vw, vh): Velocities
"""

import numpy as np
import scipy.linalg


class KalmanFilter:
    """
    Kalman Filter for tracking bounding boxes.
    
    Uses a constant velocity motion model:
    - Position at time t+1 = Position at time t + Velocity
    """
    
    def __init__(self):
        # State dimension: 8 (x, y, w, h, vx, vy, vw, vh)
        # Measurement dimension: 4 (x, y, w, h)
        
        self.ndim = 4
        self.dt = 1.0  # Time step
        
        # State transition matrix (constant velocity model)
        # x' = x + vx, y' = y + vy, etc.
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        
        # Measurement matrix (we only observe position, not velocity)
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)
        
        # Motion uncertainty weights
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
    
    def initiate(self, measurement):
        """
        Create a new track from initial detection.
        
        Args:
            measurement: [x, y, w, h] center position and size
            
        Returns:
            mean: Initial state mean [8]
            covariance: Initial state covariance [8, 8]
        """
        # Initial position from measurement, velocity = 0
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])
        
        # Initial uncertainty (higher for velocity since unknown)
        std = [
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[2],
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[2],
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        
        return mean, covariance
    
    def predict(self, mean, covariance):
        """
        Predict next state.
        
        Args:
            mean: Current state mean
            covariance: Current state covariance
            
        Returns:
            mean: Predicted state mean
            covariance: Predicted state covariance
        """
        # Motion noise
        std_pos = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[2],
            self._std_weight_velocity * mean[3]
        ]
        
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel])))
        
        # Predict: x' = F * x, P' = F * P * F^T + Q
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot([
            self._motion_mat, covariance, self._motion_mat.T
        ]) + motion_cov
        
        return mean, covariance
    
    def project(self, mean, covariance):
        """
        Project state to measurement space.
        
        Args:
            mean: State mean
            covariance: State covariance
            
        Returns:
            mean: Projected mean [4]
            covariance: Projected covariance [4, 4]
        """
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot([
            self._update_mat, covariance, self._update_mat.T
        ])
        
        return mean, covariance + innovation_cov
    
    def update(self, mean, covariance, measurement):
        """
        Update state with new measurement.
        
        Args:
            mean: Predicted state mean
            covariance: Predicted state covariance
            measurement: New observation [x, y, w, h]
            
        Returns:
            mean: Updated state mean
            covariance: Updated state covariance
        """
        # Project to measurement space
        projected_mean, projected_cov = self.project(mean, covariance)
        
        # Kalman gain
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T
        
        # Innovation (measurement residual)
        innovation = measurement - projected_mean
        
        # Update
        mean = mean + np.dot(innovation, kalman_gain.T)
        covariance = covariance - np.linalg.multi_dot([
            kalman_gain, projected_cov, kalman_gain.T
        ])
        
        return mean, covariance
    
    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """
        Compute Mahalanobis distance for gating.
        
        Used to filter out unlikely matches before Hungarian algorithm.
        
        Args:
            mean: State mean
            covariance: State covariance
            measurements: Array of measurements [N, 4]
            only_position: If True, only use (x, y) for distance
            
        Returns:
            distances: Mahalanobis distances [N]
        """
        mean, covariance = self.project(mean, covariance)
        
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        
        return np.sum(z * z, axis=0)
