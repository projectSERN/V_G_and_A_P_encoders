import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from model import *
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
        saved_state_dict = torch.load(model_path)
        model_dict = model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in saved_state_dict.keys():
            model_dict[k.replace('module.', '')] = saved_state_dict[k]
        model.load_state_dict(model_dict)
        cudnn.benchmark = True
        model = model.to(self.device)
        model.eval()
        return model
    
    def detect_gaze(self, video_path):
        """
        Detect eye gaze when left and right eyes are detected using RT-Gene's landmark extractor.
        'gazeto2d' func found in utils.py
        
        Args:
            video_path: path to the video file
        Returns:
            numpy array of [yaw, pitch] gaze for each frame where eyes are detected
        """
        # Load video
        frames, _ = extract_frames(video_path)
        
        # Debug statements
        print(f"Total number of frames: {len(frames)}")
        # print(f"FPS: {fps}")
        fps = 30 # Hard code to avoid mathematical issues
        
        # Extract left, right eye images from video frames
        left, right = self.detector.extract_eye_images(frames)

        # Detect gaze
        with torch.no_grad():
            gaze = self.model(left, right)
            # Converts model output (3D vector) to [yaw, pitch] (2D angles)
            gaze = gazeto2d(gaze)
        
        return np.array(gaze), fps