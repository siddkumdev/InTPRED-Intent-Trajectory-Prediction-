import torch
import numpy as np
import math

class ComposeTransforms:
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, past_coords, future_coords):
        for t in self.transforms:
            past_coords, future_coords = t(past_coords, future_coords)
        return past_coords, future_coords

class RandomRotate:
    """Rotates the past and future trajectories by a random angle around the origin."""
    def __init__(self, max_angle_degrees=360.0):
        self.max_angle = max_angle_degrees

    def __call__(self, past_coords, future_coords):
        # Pick a random angle in radians
        angle_deg = np.random.uniform(-self.max_angle, self.max_angle)
        angle_rad = math.radians(angle_deg)
        
        # Create a 2D rotation matrix
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ], dtype=np.float32)
        
        # Apply the rotation matrix to both sequence chunks
        # Shape: (Time, 2) @ (2, 2) -> (Time, 2)
        rotated_past = np.dot(past_coords, rotation_matrix)
        rotated_future = np.dot(future_coords, rotation_matrix)
        
        return rotated_past, rotated_future

class AddGaussianNoise:
    """
    Adds slight random noise to the past coordinates. 
    This simulates the real-world inaccuracy of LiDAR/Camera bounding box tracking 
    and prevents the model from overfitting to perfect, smooth trajectories.
    """
    def __init__(self, std=0.05):
        self.std = std

    def __call__(self, past_coords, future_coords):
        # Generate noise with the same shape as the past coordinates
        noise = np.random.normal(loc=0.0, scale=self.std, size=past_coords.shape)
        noisy_past = past_coords + noise
        
        # We DO NOT add noise to the future_coords (the ground truth target)
        return noisy_past, future_coords

class ToTensor:
    """Converts numpy arrays to PyTorch float tensors."""
    def __call__(self, past_coords, future_coords):
        past_tensor = torch.tensor(past_coords, dtype=torch.float32)
        future_tensor = torch.tensor(future_coords, dtype=torch.float32)
        return past_tensor, future_tensor