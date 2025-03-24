import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from models.ARNet import *
from utils import *

class ARNetPipeline:
    def __init__(self, model_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the ARNet (eye gaze) pipeline.
        'extract_face_features' class found in utils.py
        
        Args:
            model_path: Path to the saved ARNet model
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.model  = self.__load_model(model_path)
        self.detector = extract_face_features()
        
    def __load_model(self, model_path):
        """
        Load the pretrained ARNet model.
        
        Args:
            model_path: Path to the saved ARNet model
        Returns:
            model: ARNet model with saved state dictionary
        """
        model = ARNet()
        saved_state_dict = torch.load(model_path, weights_only=True)
        model_dict = model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in saved_state_dict.keys():
            model_dict[k.replace('module.', '')] = saved_state_dict[k]
        model.load_state_dict(model_dict)
        cudnn.benchmark = True
        model = model.to(self.device)
        model.eval()
        return model
    
    def detect_gaze(self, left, right, degrees=False, frames=None):
        """
        Detect eye gaze when left and right eyes are detected using RT-Gene's landmark extractor.
        'gazeto2d' func found in utils.py
        
        Args:
            left: list of left eye images
            right: list of right eye images
            degrees: True if output is in degrees, False if in radians
            frames: list of video frames
        Returns:
            numpy array of [yaw, pitch] gaze for each frame where eyes are detected
        """
        # Extract left, right eye images from video frames if provided
        if frames is not None:
            left, right = self.detector.extract_eye_images(frames)

            if left is None or right is None:
                # Disregard video when fails to extract eye images
                return None
        
        # Detect gaze
        with torch.no_grad():
            gaze = self.model(left, right)
            # Converts model output (3D vector) to [yaw, pitch] (2D angles)
            if degrees:
                gaze = gazeto2d(gaze)
            else:
                gaze = gaze.cpu().detach().numpy()
        
        return np.array(gaze)