import torch
import numpy as np

def gazeto2d(gaze, rad=False):
    """
    Convert 3D gaze vector to 2D gaze (pitch, yaw) angles.
    
    Args:
        gaze: torch tensor of shape (batch_size, 3) representing the 3D gaze vector
        rad: boolean to convert angles to radians
    Returns:
        gaze_2d: numpy array of shape (batch_size, 2) representing the 2D gaze angles
    """
    # Convert gaze vector to pitch and yaw angles
    yaw = torch.atan2(-gaze[:, 0], -gaze[:, 2])
    pitch = torch.asin(-gaze[:, 1])
    gaze_2d = torch.stack([yaw, pitch], dim=1)
    gaze_2d = gaze_2d.cpu().detach().numpy()
    if not rad:
        gaze_2d = (gaze_2d) * (180/np.pi)
    return gaze_2d