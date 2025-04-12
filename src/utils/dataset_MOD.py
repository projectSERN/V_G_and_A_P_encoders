import os
import cv2

import torch
import numpy as np
from torchvision import transforms

def select_frames(flags):
    """
    Select >=290 continuous frames with detected eyes.
    If length of continuous frames with undetected eyes >1/fps seconds, seperate into two continuous frames
    
    Args:
        flags: list of boolean values representing acceptable frames
        fps: frames per second (int)
    Returns:
        selected_frames: list of indices of continuous frames with detected eyes
    """
    counter = 0
    valid_frames = {}
    invalid_frames = []
    
    valid_frames[counter] = []
    for frame_idx, flag in enumerate(flags):
        if flag:
            if len(invalid_frames) > 0:
                if len(invalid_frames) > 1:
                    counter += 1
                    valid_frames[counter] = [frame_idx]
                else:
                    valid_frames[counter].append(frame_idx)
                invalid_frames = []
            else:
                valid_frames[counter].append(frame_idx)
        else:
            invalid_frames.append(frame_idx)
                
    if counter == 0:
        if len(valid_frames[0]) != 0 and len(valid_frames[0]) >= 290:
            return valid_frames[0]
        else:
            return None
    else:
        max_len_key = 0
        for k in valid_frames.keys():
            if len(valid_frames[k]) > len(valid_frames[max_len_key]):
                max_len_key = k
        
        if len(valid_frames[max_len_key]) >= 290:
            return valid_frames[max_len_key]
        else:
            return None


def eye_image_transform(eyes_left, eyes_right, device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Transform the eye images for each subject to correct shape and format.
    
    Args:
        eyes_left: list of left eye images extracted from the subject for each frame
        eyes_right: list of right eye images extracted from the subject for each frame
        device: 'cuda' or 'cpu
    Returns:
        transformed left and right images
    """
    # Perform processing of eye images
    eyes_left = np.array(eyes_left)
    eyes_right = np.array(eyes_right)
        
    eyes_left = torch.from_numpy(eyes_left).float().to(device)
    eyes_right = torch.from_numpy(eyes_right).float().to(device)
    
    if eyes_left.dim() == 1 or eyes_right.dim() == 1:
        # Need to investigate why error occurs
        print("ERROR: Eye images are not correct shape")
        return None, None
    
    eyes_left = eyes_left.permute(0, 3, 1, 2)
    eyes_right = eyes_right.permute(0, 3, 1, 2)
    
    transformations = transforms.Compose([
            transforms.Normalize(mean=[0.2323, 0.2088, 0.1767],
                                std=[0.0621, 0.0577, 0.0613])
        ])
    
    eyes_left = transformations(eyes_left)
    eyes_right = transformations(eyes_right)

    return eyes_left, eyes_right


def extract_frames(video_path):
    """
    Extract the frames from a video by loading the video path.
    
    Arg:
        video_path: path to the video file
    Returns:
        frames: the frames from the video with shape (frames, height, width, channels)
    """
    
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps


def normalized_cross_correlation(signal1, signal2):
    """
    Perform cross correlation of two signals and normalized the output to [-1, 1].
    
    Args:
        signal1, signal2: numpy array of signals to cross correlate
    Returns:
        numpy array of normalized cross correlated signal, same length as input signals
    """
    mean1 = signal1 - np.mean(signal1)
    mean2 = signal2 - np.mean(signal2)
    corr = np.correlate(mean1, mean2, mode='same')
    norm_factor = np.std(signal1) * np.std(signal2) * len(signal1)
    
    # Return correlation (avoiding division by zero)
    return corr / norm_factor if norm_factor != 0 else corr


def gaze_bins(gaze, width):
    """
    Section gaze into bins.
    
    Args:
        gaze: numpy array of shape (batch_size, 2) representing the [yaw, pitch] angle
        width: int representing bin width
    Returns:
        yaw_binned: numpy array of shape (batch_size,) representing the yaw bin
        pitch_binned: numpy array of shape (batch_size,) representing the pitch bin
    """
    bins = np.arange(-90, 90, width)
    gaze_binned = np.digitize(gaze, bins) - 1
    
    return gaze_binned

