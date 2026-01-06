
"""
Kalman Filter for tracking bounding box positions
Predicts where a person will be in the next frame
"""

import numpy as np
from scipy.linalg import block_diag


class KalmanFilter:
    """
    A simple Kalman filter for tracking bounding boxes in image space.
    
    State vector: [x, y, w, h, vx, vy, vw, vh]
    - (x, y): center position
    - (w, h): width and height
    - (vx, vy, vw, vh): velocities
    """
    
    def __init__(self):
        # State dimension: 8 (position + velocity for x, y, w, h)
        # Measurement dimension: 4 (x, y, w, h)
        
        self.ndim = 4
        self.dt = 1.0
        
        # State transition matrix (constant velocity model)
        self._motion_mat = np.eye(2 * self.ndim, 2 * self.ndim)
        for i in range(self.ndim):
            self._motion_mat[i, self.ndim + i] = self.dt
        
        # Measurement matrix
        self._update_mat = np.eye(self.ndim, 2 * self.ndim)
        
        # Process noise (motion uncertainty)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
    
    def initiate(self, measurement):
        """
        Create track from initial bounding box.
        
        Args:
            measurement: [x, y, w, h] - center position and size
            
        Returns:
            mean: Initial state mean
            covariance: Initial state covariance
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.concatenate([mean_pos, mean_vel])
        
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
        
        mean = np.dot(self._motion_mat, mean)
        covariance = np.dot(np.dot(self._motion_mat, covariance), 
                          self._motion_mat.T) + motion_cov
        
        return mean, covariance
    
    def update(self, mean, covariance, measurement):
        """
        Update state with new measurement.
        
        Args:
            mean: Predicted state mean
            covariance: Predicted state covariance
            measurement: New measurement [x, y, w, h]
            
        Returns:
            mean: Updated state mean
            covariance: Updated state covariance
        """
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.dot(np.dot(self._update_mat, covariance), 
                              self._update_mat.T)
        
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]
        ]
        innovation_cov = projected_cov + np.diag(np.square(std))
        
        # Kalman gain
        kalman_gain = np.linalg.solve(
            innovation_cov.T,
            np.dot(covariance, self._update_mat.T).T
        ).T
        
        innovation = measurement - projected_mean
        
        mean = mean + np.dot(kalman_gain, innovation)
        covariance = covariance - np.dot(np.dot(kalman_gain, innovation_cov), 
                                         kalman_gain.T)
        
        return mean, covariance
    
    def gating_distance(self, mean, covariance, measurements):
        """
        Compute gating distance between state and measurements.
        Used for matching detections to tracks.
        """
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = np.dot(np.dot(self._update_mat, covariance), 
                              self._update_mat.T)
        
        std = [
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[2],
            self._std_weight_position * mean[3]
        ]
        projected_cov += np.diag(np.square(std))
        
        d = measurements - projected_mean
        
        cholesky = np.linalg.cholesky(projected_cov)
        z = np.linalg.solve(cholesky, d.T).T
        
        return np.sum(z * z, axis=1)
