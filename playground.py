import os
import argparse

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
from distutils.util import strtobool
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import *
from src.gaze_detector import *
from src.pause_detector import *
from dataset import DFDC

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Deepfake detection from Gaze and Pauses.')
    # Model args ---------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--model', dest='model', help='Gaze model to use for training.',
        default='/scratch/zceenaa/ARE-Gaze_Model/edited/AREGaze-rtgene_best_model/best_model.pth', type=str
    )
    # Dataset args ---------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    # DFDC
    parser.add_argument(
        '--dfdc_dir', dest='dfdc_dir', help='Directory path for DFDC dataset.',
        default='/scratch/zceenaa/DFDC_dataset/', type=str)
    args = parser.parse_args()
    return args


#===========MAIN===========
if __name__ == '__main__':
    args = parse_args()
    gpu_num = 2
    cuda = True
    gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load detectors
    gaze_detector = ARNetPipeline(args.model)
    pause_detector = PauseDetector()
    
    # Load data
    print('Loading data...')
    path_dataset = args.dfdc_dir
    dataset = DFDC(path_dataset)
    # print(len(dataset))
    video_path, label = dataset.__getitem__(20674)
    # 664, 138
    
    # Detect gaze and pauses
    print('Detecting gaze and pauses...')
    gaze, _ = gaze_detector.detect_gaze(video_path)
    _, pauses, sr = pause_detector.detect_pauses(video_path)
    fps = 30
    # print(gaze)
    # print(pauses)
    print(gaze.shape)
    print(len(pauses))
    
    # Majority voting for pauses to relate to gaze
    print(sr//fps)
    # for sample in range(0, len(pauses), len(pauses)//fps):