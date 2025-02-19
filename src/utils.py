import os
import cv2
import sys

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms

from src.rtgene.extract_landmarks_method_base import LandmarkMethodBase

def gazeto2d(gaze, rad=False):
    # Convert gaze vector to pitch and yaw angles
    yaw = torch.atan2(-gaze[:, 0], -gaze[:, 2])
    pitch = torch.asin(-gaze[:, 1])
    gaze_2d = torch.stack([yaw, pitch], dim=1)
    gaze_2d = gaze_2d.cpu().detach().numpy()
    if not rad:
        gaze_2d = (gaze_2d) * (180/np.pi)
    return gaze_2d

# EDIT FOR NEW NN MODEL
def get_arnet_vars(model):
    r_var = [model.arnetbase, model.arnetdense, model.arnetsplit]
    for i in range(len(r_var)):
        for module_name, module in r_var[i].named_modules():
            for name, param in module.named_parameters():
                yield param

# EDIT FOR DFDC DATASET
def load_data(path, batch_size, transformations, train=False, valid=False, test=False):
    # Load data for train, validation and test
    if train:
        dataset = RTGene(path, transformations, train=True)
    elif valid:
        dataset = RTGene(path, transformations, valid=True)
    else:
        dataset = RTGene(path, transformations, test=True)

    data = DataLoader(
        dataset=dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=True)
    
    return data

def early_stopping(avg_val_loss, best_val_loss, epochs_no_improve, epoch, model, output):
    # Early stopping algorithm
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        best_epoch = epoch+1
        torch.save(model.state_dict(), os.path.join(output, 'best_model.pth')) 
    else:
        epochs_no_improve += 1
        
    return best_val_loss, epochs_no_improve

def loger(epoch, train_loss, val_loss, outfile, model, output):
    # Log the average loss for each epoch
    loger = f"[epoch---{epoch+1}] Train Loss:{train_loss}, Val Loss:{val_loss}\n"
    outfile.write(loger)
    print(loger)
    
    print('Taking snapshot...')
    torch.save(model.state_dict(), output +'/'+ '_epoch_' + str(epoch+1) + '.pth')


def plot_loss(path, epochs, train_loss, val_loss, angle, log=False):
    # Plot training and validation loss
    fig = plt.figure(figsize=(14, 8))        
    plt.xlabel('epoch')
    plt.ylabel(f'avg error ({angle})')
    plt.title('Gaze angular error (MAE)')
    plt.plot(epochs, train_loss, color='tab:red', label='train')
    plt.plot(epochs, val_loss, color='tab:blue', label='val')
    plt.legend(['Train_Loss', 'Val_Loss'])
    plt.show()
    fig.savefig(os.path.join(path, "Train_Loss.png"), format='png')
    
    if log:
        plt.yscale('log')
        fig.savefig(os.path.join(path, "TrainLoss_log.png"), format='png')
    plt.clf()
   
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

# Haarcascade (NO LONGER USED)
class extract_face_features_haar():
    def __init__(self, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        # Load Haarcascade
        path_face = '../haarcascade_frontalface_alt2.xml'
        path_left = '../haarcascade_lefteye_2splits.xml'
        path_right = '../haarcascade_righteye_2splits.xml'
        self.detector_face = cv2.CascadeClassifier(path_face)
        self.detector_left = cv2.CascadeClassifier(path_left)
        self.detector_right = cv2.CascadeClassifier(path_right)
        self.device = device
         
    def extract_face_images(self, frames):
        # Extract face images from frame
        faces = []
        index = 0
        for frame in frames:
            # Detect face
            faceRect = self.detector_face.detectMultiScale(frame, 1.05, 5)
            if len(faceRect) == 0:
                print('No face detected...')
                cv2.imwrite(f'undetected_face_{index}.png', frame)
                index += 1
                continue
            
            # Extract face
            (x, y, w, d) = faceRect[0]
            frame_cropped = frame[y:(y + d), x:(x + w)]
            faces.append(frame_cropped)

            # Save cropped face
            # frame_cropped = frame_cropped.astype(np.uint8)
            # cv2.imwrite('output_face_cv2.png', frame_cropped)
            
        return faces
        
    def extract_eye_images(self, frames):
        # Extract left and right eye images from frames
        # Extract face from frames
        faces = self.extract_face_images(frames)
        
        eyes_left = []
        eyes_right = []
        index = 0
        for face in faces:
            eye_left_rect = self.detector_left.detectMultiScale(
                face, scaleFactor=1.1, minNeighbors=10,
                minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
            eye_right_rect = self.detector_right.detectMultiScale(
                face, scaleFactor=1.1, minNeighbors=10,
                minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)
            
            if (len(eye_left_rect) == 0) or (len(eye_right_rect) == 0):
                print('No eyes detected...')
                cv2.imwrite(f'undetected_eyes_{index}.png', face)
                index += 1
                continue
                    
            (x, y, w, d) = eye_left_rect[0]
            (x, y, w, d) = eye_right_rect[0]
            
            eye_left = face[y:(y + d), x:(x + w)]
            eye_right = face[y:(y + d), x:(x + w)]
            
            eye_left = cv2.resize(eye_left, (60, 36))
            eye_right = cv2.resize(eye_right, (60, 36))
            
            eyes_left.append(eye_left)
            eyes_right.append(eye_right)
            
            # print(eye_left.shape)
            # print(eye_right.shape)
            
            # eye_left = eye_left.astype(np.uint8)
            # eye_right = eye_right.astype(np.uint8)
            # cv2.imwrite('output_eye_left_cv2.png', eye_left)
            # cv2.imwrite('output_eye_right_cv2.png', eye_right)
        
        # Convert eyes_left and eyes_right to correct shape and size    
        eyes_left = np.array(eyes_left)
        eyes_right = np.array(eyes_right)
            
        eyes_left = torch.from_numpy(eyes_left).float().cuda(self.device)
        eyes_right = torch.from_numpy(eyes_right).float().cuda(self.device)
        
        eyes_left = eyes_left.permute(0, 3, 1, 2)
        eyes_right = eyes_right.permute(0, 3, 1, 2)
        
        transformations = transforms.Compose([
                transforms.Normalize(mean=[0.2323, 0.2088, 0.1767],
                                     std=[0.0621, 0.0577, 0.0613])
            ])
        
        eyes_left = transformations(eyes_left)
        eyes_right = transformations(eyes_right)
        
        return eyes_left, eyes_right
  
# Using RT-Gene (CURRENTLY USED)  
class extract_face_features():
    """
    Class to extract relevant face features.
    
    Licensing:
    Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        script_path = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(os.path.abspath(os.path.join(script_path, '../..')))
        self.landmark_estimator = LandmarkMethodBase(device_id_facedetection=self.device,
                                            checkpoint_path_face=os.path.abspath(os.path.join(script_path, "../src/rtgene/model_nets/SFD/s3fd_facedetector.pth")),
                                            checkpoint_path_landmark=os.path.abspath(
                                                os.path.join(script_path, "../src/rtgene/model_nets/phase1_wpdc_vdc.pth.tar")),
                                            model_points_file=os.path.abspath(os.path.join(script_path, "../src/rtgene/model_nets/face_model_68.txt")))
    
    def __extract_eye_image_patches(self, subject):
        le_c, re_c, _, _ = subject.get_eye_image_from_landmarks(subject, self.landmark_estimator.eye_image_size)
        subject.left_eye_color = le_c
        subject.right_eye_color = re_c


    def extract_face_images(self, frames):
        """
        Extract face from video frames.
        
        Args:
            frames: list of the video frames
        Returns:
            faces: list of face images extracted from each frame
            subjects: extracted faces stored in list as objects for each frame
        """
        faces = []
        subjects = []
        for frame in frames:
            faceboxes = self.landmark_estimator.get_face_bb(frame)
            if len(faceboxes) == 0:
                print('Could not find faces in the image')
                return

            img_subjects = self.landmark_estimator.get_subjects_from_faceboxes(frame, faceboxes)
            for subject in img_subjects:
                subjects.append(subject) # subject = object
                # subject.face_color contains face image of shape (227, 227, 3)
                faces.append(subject.face_color)
                # cv2.imwrite('output_face.png', subject.face_color)
            
        return faces, subjects
            
    def extract_eye_images(self, frames):
        """
        Extract left and right eyes from video frames.
        
        Args:
            frames: list of the video frames
        Returns:
            eyes_left: list of left eye images extracted from each frame
            eyes_right: list of right eye images extracted from each frame

        """
        # Extract face images
        faces, subjects = self.extract_face_images(frames)

        eyes_left = []
        eyes_right = []
        # For each frame where face is extracted, extract left and right eyes
        for subject in subjects:
            # Extract eye patches
            self.__extract_eye_image_patches(subject)
            if subject.left_eye_color is None or subject.right_eye_color is None:
                print('Failed to extract eye image patches')
                continue
            else:
                eyes_left.append(torch.from_numpy(subject.left_eye_color).float())
                eyes_right.append(torch.from_numpy(subject.right_eye_color).float())
                # print(subject.left_eye_color.shape)
                # cv2.imwrite('output_eye_left.png', subject.left_eye_color)
                # print(subject.right_eye_color.shape)
                # cv2.imwrite('output_eye_right.png', subject.right_eye_color)
                
        # Processing of eye images
        eyes_left = np.array(eyes_left)
        eyes_right = np.array(eyes_right)
            
        eyes_left = torch.from_numpy(eyes_left).float().cuda(self.device)
        eyes_right = torch.from_numpy(eyes_right).float().cuda(self.device)
        
        eyes_left = eyes_left.permute(0, 3, 1, 2)
        eyes_right = eyes_right.permute(0, 3, 1, 2)
        
        transformations = transforms.Compose([
                transforms.Normalize(mean=[0.2323, 0.2088, 0.1767],
                                     std=[0.0621, 0.0577, 0.0613])
            ])
        
        eyes_left = transformations(eyes_left)
        eyes_right = transformations(eyes_right)
        
        return eyes_left, eyes_right