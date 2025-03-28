import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset
import pandas as pd

from src.gaze_detector import ARNetPipeline
from src.pause_detector import PauseDetector
from src.feature_detector import extract_face_features
from src.utils.dataset_MOD import select_frames, eye_image_transform, extract_frames, normalized_cross_correlation

class DFDC_subset(Dataset):
    def __init__(self, path, train=True, val=False, test=False,
                 gaze_model='/scratch/zceenaa/ARE-Gaze_Model/edited/AREGaze-rtgene_best_model/best_model.pth',
                 device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Load a specific subset of the created DFDC subset.
        Store list of [subject, video_name] in 'self.video'.
        
        Args:
            path: path to the specific subset of the created DFDC subset folder (str)
            train: toggle for train split (bool)
            val: toggle for val split (bool)
            test: toggle for test split (bool)
            gaze_model: path to trained gaze model (str)
            device: 'cuda' or 'cpu' (str)
        """
        self.device = device
        self.path = path
        self.lines = []
        self.face_features_detector = extract_face_features(device=self.device)
        self.gaze_detector = ARNetPipeline(gaze_model, device=self.device)
        self.pause_detector = PauseDetector()
        
        self.split = os.listdir(self.path)
        
        label_path = os.path.join(self.path, 'subset_metadata.csv')
        
        # subset_metadata.json
        # (video_file, label, part, split)
        with open(label_path, 'r') as file:
            lines = file.readlines()
            for index in range(1, len(lines)):
                line = lines[index]
                line = line.split(',')
                video = line[0]
                label = line[1]
                part = line[2]
                split = line[3].strip()
                self.lines.append([video, label, part, split])
              
    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        """
        Get information of item in the loaded dataset.
        
        Args:
            idx: Index of selected item
        Returns:
            data: dictionary of {'yaw_corr', 'pitch_corr'}
            label: 'REAL' or 'FAKE' label attached to selected item
            split: 'train', 'val' or 'test' split of selected item
        """
        data = self.lines[idx]
        
        video = data[0]
        label = data[1]
        part = data[2]
        split = data[3]
        
        video_path = os.path.join(self.path, split, video)
        
        # Get normalised cross correlation of yaw and pitch signals with pauses
        data = self._preprocess(video_path)
        
        
        return data, label, split
    
    
    # Additional methods
    def _preprocess(self, video_path):
        """
        Convert detected gaze and pauses from a video to correlated signals.
        
        Args:
            video_path: index of the video in the dataset
        Returns:
            data: dictionary of {
                    'yaw_corr': normalised cross correlation of yaw signal with pauses,
                    'pitch_corr': normalised cross correlation of pitch signal with pauses,
                    }
        """
        # Get frames from video
        frames, fps = extract_frames(video_path)
        # fps = 30 # Hard code to avoid mathematical issues
        
        # Extract left and right eye images
        eyes_left, eyes_right, flags = self.face_features_detector.extract_eye_images(frames)

        # Select frames with detected eyes
        selected_frames = select_frames(flags, fps)
        
        selected_left = []
        selected_right = []
        for index in selected_frames:
            selected_left.append(eyes_left[index])
            selected_right.append(eyes_right[index])
        
        # Transform eyes correct shape and format
        selected_left, selected_right = eye_image_transform(selected_left, selected_right)
        
        # Detect gaze
        gaze = self.gaze_detector.detect_gaze(selected_left, selected_right)
        if gaze is None:
            # Disregard video when gaze not detected
            print("ERROR: Gaze not detected")
            return "Gaze", video_path
        
        # Detect pauses
        _, pauses, _ = self.pause_detector.detect_pauses(video_path)

        # Majority voting for pauses to relate to gaze
        items = len(pauses)//len(gaze)
        pauses_matched = [] # Pause frequency matched to gaze (i.e. per frame)
        for index in range(0, len(pauses), items):
            sample = pauses[index:index+items]
            num_neg = np.sum(np.array(sample) == -1)
            num_pos = np.sum(np.array(sample) == 1)
            value = 1 if (num_pos > num_neg) else -1
            pauses_matched.append(value)
            
        yaw = gaze[:,0]
        pitch = gaze[:,1]
        
        # Normalised cross correlation
        yaw_corr = normalized_cross_correlation(yaw, pauses_matched)
        pitch_corr = normalized_cross_correlation(pitch, pauses_matched)

        data = {'yaw_corr': yaw_corr, 'pitch_corr': pitch_corr}
                
        return data
        

if __name__ == "__main__":
    path = '/scratch/zceenaa/DFDC_subsets/subset_01/'
    save_path = '/scratch/zceenaa/DFDC_subsets_preprocessed/subset_09/'
    save_path_2 = '/scratch/zceenaa/DFDC_test/'
    preprocessed_path = '/scratch/zceenaa/DFDC_subsets_preprocessed/'       
    
    dataset = DFDC_subset(path)
    data, label, split = dataset.__getitem__(0)
    print(data)
    print(len(data))