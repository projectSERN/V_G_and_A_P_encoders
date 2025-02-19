#!/usr/bin/env python

# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

from __future__ import print_function, division, absolute_import

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from rtgene.extract_landmarks_method_base import LandmarkMethodBase
from rtgene.gaze_tools import get_phi_theta_from_euler, limit_yaw
from rtgene.gaze_tools_standalone import euler_from_matrix
from utils import *

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, '../..')))
from dataset import *

def extract_eye_image_patches(subjects):
    for subject in subjects:
        le_c, re_c, _, _ = subject.get_eye_image_from_landmarks(subject, landmark_estimator.eye_image_size)
        subject.left_eye_color = le_c
        subject.right_eye_color = re_c


def estimate_gaze(color_img, index):
    faceboxes = landmark_estimator.get_face_bb(color_img)
    if len(faceboxes) == 0:
        tqdm.write('Could not find faces in the image')
        return

    subjects = landmark_estimator.get_subjects_from_faceboxes(color_img, faceboxes)
    extract_eye_image_patches(subjects)

    for idx, subject in enumerate(subjects):
        # print(subject.face_color.shape)
        cv2.imwrite('output_face.png', subject.face_color)
        if subject.left_eye_color is None or subject.right_eye_color is None:
            tqdm.write('Failed to extract eye image patches')
            continue
        else:
            # print(subject.left_eye_color.shape)
            cv2.imwrite('output_eye_left.png', subject.left_eye_color)
            # print(subject.right_eye_color.shape)
            cv2.imwrite('output_eye_right.png', subject.right_eye_color)
            
    return index+1


if __name__ == '__main__':
    # Load data
    print('Loading data...')
    path_dataset = '/scratch/zceenaa/DFDC_dataset/'
    dataset = DFDC(path_dataset)
    print(len(dataset))
    video_path, label = dataset.__getitem__(138)
    # 664, 138
    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    tqdm.write('Loading networks...')
    landmark_estimator = LandmarkMethodBase(device_id_facedetection=device,
                                            checkpoint_path_face=os.path.abspath(os.path.join(script_path, "../src/rtgene/model_nets/SFD/s3fd_facedetector.pth")),
                                            checkpoint_path_landmark=os.path.abspath(
                                                os.path.join(script_path, "../src/rtgene/model_nets/phase1_wpdc_vdc.pth.tar")),
                                            model_points_file=os.path.abspath(os.path.join(script_path, "../src/rtgene/model_nets/face_model_68.txt")))
    
    # Detect gaze and pauses
    print('Detecting gaze and pauses...')
    frames, _ = extract_frames(video_path)
    print(len(frames))
    # img = frames[0]
    # cv2.imwrite('output_org_img.png', img)
    index = 0
    for frame in frames:
        index = estimate_gaze(frame, index)
    print(index)