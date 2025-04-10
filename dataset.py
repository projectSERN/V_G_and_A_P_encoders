import os
import glob
import ast
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset
from scipy.signal import resample


from src.utils.dataset_MOD import *

class DFDC_subset(Dataset):
    def __init__(self, subset, gaze_model='/scratch/zceenaa/ARE-Gaze_Model/edited/AREGaze-rtgene_best_model/best_model.pth',
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
        from src.gaze_detector import ARNetPipeline
        from src.pause_detector import PauseDetector
        from src.feature_detector import extract_face_features
        
        self.device = device
        self.subset = subset
        self.path = f'/scratch/zceenaa/DFDC_subsets/subset_{self.subset}/'
        self.data = []
        self.face_features_detector = extract_face_features(device=self.device)
        self.gaze_detector = ARNetPipeline(gaze_model, device=self.device)
        self.pause_detector = PauseDetector()
        
        self.split = os.listdir(self.path)
        
        # Get all video file paths
        video_files = []
        for split in self.split:
            video_files += glob.glob(os.path.join(self.path, split, '*.mp4'))
        video_names = [os.path.basename(file) for file in video_files]
        
        # Get correct split
        label_path = os.path.join(self.path, f'subset_metadata.csv')
        
        # subset_metadata_{subset}.csv
        # (video_file, split, label)
        with open(label_path, 'r') as file:
            lines = file.readlines()
            for line in range(1, len(lines)):
                line = lines[line].strip().split(',')
                video = line[0]
                split = line[1]
                label = line[2]
                
                index = video_names.index(video)
                self.data.append([video_files[index], split, label])
                
              
    def __len__(self):
        return len(self.data)

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
        data = self.data[idx]
        
        video_path = data[0]
        video = os.path.basename(video_path)
        split = data[1]
        label = data[2]
        
        video_info = {'split': split, 'video': video}

        # Get normalised cross correlation of yaw and pitch signals with pauses
        data = self._preprocess(video_path)
        return data, label, video_info
    
    
    # Additional methods
    def _preprocess(self, video_path):
        """
        Convert detected gaze and pauses from a video to correlated signals.
        
        Args:
            video_path: index of the video in the dataset
        Returns:
            data: dictionary of {
                    'gaze': gaze angles (yaw,pitch),
                    'pauses': tagged downsampled pauses,
                    'gaze-pause': [yaw_corr, pitch_corr]
                    }
        """
        # Get frames from video
        frames, fps = extract_frames(video_path)
        
        # Extract left and right eye images
        eyes_left, eyes_right, flags = self.face_features_detector.extract_eye_images(frames)
        
        if eyes_left is None:
            return None

        # Select frames with detected eyes (truncate to 290 frames)
        selected_frames = select_frames(flags, fps)
        if selected_frames is None:
            print("ERROR: No consistent detection of eyes")
            return None
        selected_frames = selected_frames[:290]
        
        selected_left = []
        selected_right = []
        for index in selected_frames:
            selected_left.append(eyes_left[index])
            selected_right.append(eyes_right[index])
        
        # Transform eyes correct shape and format
        selected_left, selected_right = eye_image_transform(selected_left, selected_right, device=self.device)
        
        # Detect gaze
        gaze = self.gaze_detector.detect_gaze(selected_left, selected_right)
        # Detect pauses (truncate to 220500 samples)
        _, pauses, sr = self.pause_detector.detect_pauses(video_path)
        pauses = pauses[:220500]

        # Downsample gaze to match fps of video (i.e. 29 fps)
        pauses_matched = np.ravel(resample(pauses, 290))
            
        yaw = np.ravel(gaze[:,0])
        pitch = np.ravel(gaze[:,1])
        
        # Normalised cross correlation
        yaw_corr = normalized_cross_correlation(yaw, pauses_matched)
        pitch_corr = normalized_cross_correlation(pitch, pauses_matched)

        data = {'yaw': yaw, 'pitch': pitch, 'pauses': pauses_matched, 'gaze-pause': [yaw_corr, pitch_corr]}
                
        return data
    
    def save_preprocessed(self):
        """
        Save preprocessed data of DFDC dataset.
        Information to save - gaze=[yaw,pitch], pauses=[tagged pauses], gaze_pause_corr=[yaw_corr,pitch_corr]
        """
        video_paths = []
        data_split = []
        yaw = []
        pitch = []
        pauses = []
        yaw_corr = []
        pitch_corr = []
        labels = []
        errors = 0
        
        print(f"Saving gaze-pause data of subset_{self.subset} -------device: {self.device}")
        for idx in tqdm(range(self.__len__())):
            data, label, video_info = self.__getitem__(idx)
            if data is None:
                errors += 1
                print(f"Errors so far: {errors}")
                continue
            
            split = video_info['split']
            print(f'Video: {video_info["video"]} Label: {label} Split: {split}')
            
            video_paths.append(video_info['video'])
            data_split.append(video_info['split'])
            yaw.append(data['yaw'])
            pitch.append(data['pitch'])
            pauses.append(data['pauses'])
            yaw_corr.append(data['gaze-pause'][0])
            pitch_corr.append(data['gaze-pause'][1])
            num_label = 0 if 'REAL' in label else 1
            labels.append(num_label)
        
        print(f"Total errors: {errors}")
        
        # Save numpy arrays to npy files
        save_path = f'/scratch/zceenaa/DFDC_subsets_preprocessed/subset_{self.subset}'
        os.makedirs(save_path, exist_ok=True)        
        np.save(os.path.join(save_path, 'video_paths.npy'), video_paths)
        np.save(os.path.join(save_path, 'split.npy'), data_split)
        np.save(os.path.join(save_path, 'yaw.npy'), yaw)
        np.save(os.path.join(save_path, 'pitch.npy'), pitch)
        np.save(os.path.join(save_path, 'pauses.npy'), pauses)
        np.save(os.path.join(save_path, 'yaw_corr.npy'), yaw_corr)
        np.save(os.path.join(save_path, 'pitch_corr.npy'), pitch_corr)
        np.save(os.path.join(save_path, 'labels.npy'), labels)


class DFDC_preprocessed(Dataset):
    def __init__(self, subset, split='train', feature='gaze', width=10):
        """
        Load a specific subset of the preprocessed DFDC subset.
        Store data into dictionary (self.data) = {
            'feature': [gaze, pause or gaze-pause],
            'label': [REAL (0) or FAKE (1)]
        }
        
        Args:
            path: path to the specific subset of the preprocessed DFDC subset folder (str)
            subset: subset number (str)
            split: select which split to load (str)
            feature: select which feature to load (i.e. gaze, pause or gaze-pause) (str)
            width: width of gaze bins (int)
        """
        self.subset = subset
        self.path = f'/scratch/zceenaa/DFDC_subsets_preprocessed/subset_{subset}/'
        self.data = {'feature': [], 'label': []}
        
        # Reading npy files
        splits = np.load(os.path.join(self.path, 'split.npy'))
        yaw = np.load(os.path.join(self.path, 'yaw.npy'))
        pitch = np.load(os.path.join(self.path, 'pitch.npy'))
        pauses = np.load(os.path.join(self.path, 'pauses.npy'))
        yaw_corr = np.load(os.path.join(self.path, 'yaw_corr.npy'))
        pitch_corr = np.load(os.path.join(self.path, 'pitch_corr.npy'))
        labels = np.load(os.path.join(self.path, 'labels.npy'))
        
        # Get correct split data
        for i in range(len(splits)):
            if splits[i] == split:
                if feature == 'gaze':
                    yaw_binned = gaze_bins(yaw[i], width)
                    pitch_binned = gaze_bins(pitch[i], width)
                    # gaze = np.ravel([yaw_binned, pitch_binned])
                    gaze = np.ravel([[yaw_binned[i], pitch_binned[i]] for i in range(len(yaw_binned))])
                    self.data['feature'].append(gaze)
                elif feature == 'pause':
                    self.data['feature'].append(pauses[i])
                elif feature == 'gaze-pause':
                    corr = np.ravel([[yaw_corr[i], pitch_corr[i]] for i in range(len(yaw_corr))])
                    self.data['feature'].append(corr)
                
                self.data['label'].append(labels[i])



    def __len__(self):
        return len(self.data['feature'])
    
    def __getitem__(self, idx):
        """
        Get information of item in the loaded dataset.
        
        Args:
            idx: index of selected item
        Returns:
            data: required features for model training (i.e. [yaw,pitch] or [pauses] or [yaw_corr,pitch_corr])
            label: 'REAL' or 'FAKE' label attached to selected item
        """
        
        data = self.data['feature'][idx]
        label = self.data['label'][idx]
        
        # Convert data to tensor
        data = torch.tensor(data).float()
        label = torch.tensor(label).float()
        
        return data, label

if __name__ == "__main__":
    # dataset = DFDC_subset(subset='06', device='cuda:1')
    # dataset.save_preprocessed()
    dataset2 = DFDC_preprocessed(subset='06', split='train', feature='gaze')