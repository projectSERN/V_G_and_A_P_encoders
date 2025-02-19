import os
import numpy as np
import cv2
import json


import torch
from torch.utils.data.dataset import Dataset

from torch.utils.data import DataLoader

class DFDC(Dataset):
    def __init__(self, path, test=False, valid=False):
        """
        Load the DFDC dataset for model training, validation and testing.
        Store list of [subject, video_name] in 'self.video'.
        
        Args:
            path = Path to the DFDC dataset folder
            test = Trigger to load test set
            valid = Trigger to load validation set
        """
        self.path = path
        self.videos = []

        self.test = test
        self.valid = valid
        
        self.subjects = os.listdir(self.path)
        # print(self.subjects)
        # print(len(os.listdir(os.path.join(self.path, self.subjects[0]))))
        # exit()
        
        if self.test == True:
            # Testing set
            self.setting = 'test'
        elif self.valid == True:
            # Validation set
            self.setting = 'valid'
        else:
            # Training set
            self.setting = 'train'
        
        
        for subject in self.subjects:
            label_path = os.path.join(self.path, subject, 'metadata.json')
            
            # metadata.json
            # dictionary: {'video': {'label', 'split', 'original (if FAKE)'}}
            with open(label_path, 'r') as file:
                data = json.load(file)
                for video in data:
                    # print(subject, video)
                    self.videos.append([subject, video])
      
    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        """
        Get information of item in the loaded dataset.
        
        Args:
            idx: Index of selected item
        Returns:
            video_path: Path to video of selected item
            label: 'REAL' or 'FAKE' label attached to selected item
        """
        video = self.videos[idx]

        subject = video[0]
        name = video[1]
        video_path = os.path.join(self.path, subject, name)
        label_path = os.path.join(self.path, subject, 'metadata.json')
        with open(label_path, 'r') as file:
            metadata = json.load(file)
            labels = metadata[name]
            label = labels['label']
        
        return video_path, label
  
if __name__ == "__main__":
    path = '/scratch/zceenaa/DFDC_dataset/'
    dataset = DFDC(path)
    index = []
    for i in range(len(dataset)):
        video_path, label = dataset.__getitem__(i)
        if label == "REAL":
        # if i == 138:
            print(i, video_path, label)
            index.append(i)
            
    print(index)
    print(len(index))