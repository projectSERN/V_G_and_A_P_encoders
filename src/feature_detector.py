import os
import cv2
import sys

import torch
from torchvision import transforms
import numpy as np

from src.rtgene.extract_landmarks_method_base import LandmarkMethodBase
from src.rtgene.gaze_tools import visualize_landmarks

# Using RT-Gene (SFD [faceboxes] + 3DDFA [facial landmarks]) to extract face and eye images 
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
        """
        Extract eye image patches from the subject
        
        Args:
            subject: subject object containing facial landmarks
        Returns:
            None
        """
        le_c, re_c, _, _ = subject.get_eye_image_from_landmarks(subject, self.landmark_estimator.eye_image_size)
        subject.left_eye_color = le_c
        subject.right_eye_color = re_c
        
    def __calculate_iou(self, boxA, boxB):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes
        
        Args:
            boxA: first bounding box
            boxB: second bounding box
        Returns:
            IoU: Intersection over Union of the two bounding boxes
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
        
    def __match_faces(self, previous_subjects, current_subjects):
        """
        Match current faceboxes to previous faceboxes
        
        Args:
            previous_subjects: dictionary of faceboxes from previous frame (key = subject index)
            current_subjects: list of subjects from current frame
        Returns:
            matched_subjects: dictionary of matched faces to subject index => {'subject index': subject object}
        """
        matched_subjects = {}
        for current_subject in current_subjects:
            threshold = 0.6
            matched = None
            # for previous_box in previous_subjects:
            for previous_subject in previous_subjects:
                # Calculate IoU between current facebox and previous facebox
                if previous_subjects[previous_subject] is None:
                    continue
                previous_box = previous_subjects[previous_subject].box
                current_box = current_subject.box
                iou = self.__calculate_iou(current_box, previous_box)
                if iou > threshold:
                    matched = previous_subject
            matched_subjects[matched] = current_subject
            
        return matched_subjects
    
    def __visualize_face_tracking(self, frames, subjects):
        """
        Visualize face tracking in a video and save the video
        
        Args:
            frames: list of video frames
            subjects: list of subjects
        Returns:
            None
        """
        colours = {}
        for subject in subjects:
            new_colour = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
            colours[subject] = new_colour
            
        tracked_frames = []
        for index in range(len(frames)):
            img = frames[index].copy()
            for subject in subjects:
                if subjects[subject][index] is not None:
                    box = subjects[subject][index].box.astype(int)
                    img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), colours[subject], 2)
            tracked_frames.append(img)
        
        out = cv2.VideoWriter('face_tracking.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (frames[0].shape[1], frames[0].shape[0]))
        for frame in tracked_frames:
            out.write(frame)
        out.release()
    
    # Ignore for gaze estimation
    def extract_subjects(self, frames):
        """
        Extract subjects from video frames.
        
        Args:
            frames: list of the video frames
        Returns:
            subjects: dictionary of extracted faces, {'subject index': [subject object for each frame]}
            error: the error associated with the video (default = None)
        """
        error = None
        subjects = {}
        previous_faceboxes = None
        
        current_frame = 0
        for frame in frames:
            faceboxes = self.landmark_estimator.get_face_bb(frame)
            if len(faceboxes) == 0:
                print('ERROR: Could not find faces in the image')
                return None, "Face"
            
            current_subjects = self.landmark_estimator.get_subjects_from_faceboxes(frame, faceboxes)
            
            if previous_faceboxes is not None:
                # Match faceboxes to previous faceboxes
                matched_subjects = self.__match_faces(previous_faceboxes, current_subjects)
                
                # If previous subject not detected in current frame, append None to its current frame
                for subject in subjects:
                    if subject not in matched_subjects:
                        matched_subjects[subject] = None       
                        
                for matched_subject in matched_subjects:
                    if matched_subject not in subjects:
                        # Add new dictionary key if new face detected
                        new_subject_index = len(subjects)
                        subjects[new_subject_index] = [None] * current_frame
                        subjects[new_subject_index].append(matched_subjects[matched_subject])
                    else:   
                        # Append subject object depending on subject index (i.e. value of matched_subject)
                        subjects[matched_subject].append(matched_subjects[matched_subject])
            else:
                for index in range(len(current_subjects)):
                    subjects[index] = [current_subjects[index]]
            
            # Link previous faceboxes based on subject index
            previous_faceboxes = {}
            for subject in subjects:
                previous_faceboxes[subject] = subjects[subject][current_frame]
            current_frame += 1
        
        # Remove any subject tracked for less than one second
        removed_subjects = []
        for subject in subjects:
            frames_tracked = len(subjects[subject]) - subjects[subject].count(None)
            if frames_tracked != len(frames):
                removed_subjects.append(subject)        
        for subject in removed_subjects:
            del subjects[subject]
        
        if len(subjects) > 1:
            print(f"ERROR: More than one subject detected - {len(subjects)}")
            error = "Subjects"
        
        # self.__visualize_face_tracking(frames, subjects)
        
        return subjects, error
            
            
    def determine_errors(self, frames):
        """
        Determine the errors in the video frames.
        
        Args:
            frames: list of video frames
        Returns:
            subjects: dictionary of extracted faces, {'subject index': [subject object for each frame]}
            error: error message associated with the video (default = None)
        """
        error = None
        subjects = {}
        eyes_left = {}
        eyes_right = {}
        previous_faceboxes = None
        
        current_frame = 0
        count = 0 # Count number of frames subject not detected
        for frame in frames:
            faceboxes = self.landmark_estimator.get_face_bb(frame)
            if len(faceboxes) == 0:
                count += 1
                if count > 1:
                    # As human eyes move ~20-40ms, if face not detected for more than 1 continuous frame, ignore video
                    print('ERROR: Could not find faces in the image')
                    return [], "Face"
                continue
            else:
                count = 0
            
            current_subjects = self.landmark_estimator.get_subjects_from_faceboxes(frame, faceboxes)
            
            if previous_faceboxes is not None:
                # Match faceboxes to previous faceboxes
                matched_subjects = self.__match_faces(previous_faceboxes, current_subjects)
                
                # If previous subject not detected in current frame, append None to its current frame
                for subject in subjects:
                    if subject not in matched_subjects:
                        matched_subjects[subject] = None       
                        
                for matched_subject in matched_subjects:
                    if matched_subject not in subjects:
                        # Add new dictionary key if new face detected
                        new_subject = matched_subjects[matched_subject]
                        new_subject_index = len(subjects)
                        subjects[new_subject_index] = [None] * current_frame
                        subjects[new_subject_index].append(new_subject)
                        
                        # Extract eye patches
                        self.__extract_eye_image_patches(new_subject)
                        
                        eyes_left[new_subject_index] = [None] * current_frame
                        eyes_right[new_subject_index] = [None] * current_frame
                        if new_subject.left_eye_color is None or new_subject.right_eye_color is None:
                            eyes_left[new_subject_index].append(None)
                            eyes_right[new_subject_index].append(None)
                        else:
                            eyes_left[new_subject_index].append(torch.from_numpy(new_subject.left_eye_color).float())
                            eyes_right[new_subject_index].append(torch.from_numpy(new_subject.right_eye_color).float())
                    else:   
                        # Append subject object depending on subject index (i.e. value of matched_subject)
                        subject = matched_subjects[matched_subject]
                        subjects[matched_subject].append(subject)
                        
                        if subject is not None:
                            # Extract eye patches
                            self.__extract_eye_image_patches(subject)
                            if subject.left_eye_color is None or subject.right_eye_color is None:
                                eyes_left[matched_subject].append(None)
                                eyes_right[matched_subject].append(None)
                            else:
                                eyes_left[matched_subject].append(torch.from_numpy(subject.left_eye_color).float())
                                eyes_right[matched_subject].append(torch.from_numpy(subject.right_eye_color).float())
            else:
                for index in range(len(current_subjects)):
                    subject = current_subjects[index]
                    subjects[index] = [subject]
                    
                    # Extract eye patches
                    self.__extract_eye_image_patches(subject)
                    if subject.left_eye_color is None or subject.right_eye_color is None:
                        eyes_left[index] = [None]
                        eyes_right[index] = [None]
                    else:
                        eyes_left[index] = [torch.from_numpy(subject.left_eye_color).float()]
                        eyes_right[index] = [torch.from_numpy(subject.right_eye_color).float()]
            
            # Link previous faceboxes based on subject index
            previous_faceboxes = {}
            for subject in subjects:
                previous_faceboxes[subject] = subjects[subject][current_frame]
            current_frame += 1
        
        # Remove any subject tracked for less than one second
        removed_subjects = []
        for subject in subjects:
            # Check if faces are semi-tracked
            count = 0 # Count number of frames subject is not detected
            for frame in range(len(subjects[subject])):
                if subjects[subject][frame] is None:
                    count += 1
                    if count > 1:
                        removed_subjects.append(subject)
                        break
                else:
                    count = 0
        
        # Delete subjects that were not fully tracked
        for subject in removed_subjects:
            del subjects[subject]
        
        if len(subjects) > 1:
            print(f"ERROR: More than one subject detected - {len(subjects)}")
            return subjects, "Subjects"
        
        # Processing of eye images for fully tracked face (single subject only)
        for subject in subjects:
            # Check if eye images are semi-tracked
            count = 0 # Count number of frames eye is not detected
            for frame in range(len(eyes_left[subject])):
                if eyes_left[subject][frame] is None or eyes_right[subject][frame] is None:
                    count += 1
                    if count > 1:
                        # As human eyes move ~20-40ms, if eye not detected for more than 1 continuous frame, ignore video
                        print("ERROR: Failed to extract eye images")
                        return [], "Eyes"
                else:
                    count = 0
            
            # Remove None values from eye images
            eyes_left[subject] = [eye for eye in eyes_left[subject] if eye is not None]
            eyes_right[subject] = [eye for eye in eyes_right[subject] if eye is not None]
            
            # Perform processing of eye images
            eyes_left[subject] = np.array(eyes_left[subject])
            eyes_right[subject] = np.array(eyes_right[subject])
                
            eyes_left[subject] = torch.from_numpy(eyes_left[subject]).float().to(self.device)
            eyes_right[subject] = torch.from_numpy(eyes_right[subject]).float().to(self.device)
            
            if eyes_left[subject].dim() == 1 or eyes_right[subject].dim() == 1:
                # Need to investigate why error occurs
                print("ERROR: Eye images are not correct shape")
                return [], "Dimensions"
            
            eyes_left[subject] = eyes_left[subject].permute(0, 3, 1, 2)
            eyes_right[subject] = eyes_right[subject].permute(0, 3, 1, 2)
            
            transformations = transforms.Compose([
                    transforms.Normalize(mean=[0.2323, 0.2088, 0.1767],
                                        std=[0.0621, 0.0577, 0.0613])
                ])
            
            eyes_left[subject] = transformations(eyes_left[subject])
            eyes_right[subject] = transformations(eyes_right[subject])
        
        return subjects, error
    
    
    def extract_eye_images(self, frames):
        """
        Extract left and right eyes from video frames. Based on extract_subjects method.
        Assumes errorneous videos have been removed.
        Video segmentation performed after this method.
        
        Args:
            frames: list of video frames
        Returns:
            eyes_left: list of left eye images extracted from the subject for each frame
            eyes_right: list of right eye images extracted from the subject for each frame
            flag: list of flags indicating if eye images are extracted correctly
        """
        subjects = {}
        eyes_left = {}
        eyes_right = {}
        previous_faceboxes = None
        
        current_frame = 0
        count = 0 # Count number of frames subject not detected
        for frame in frames:
            faceboxes = self.landmark_estimator.get_face_bb(frame)
            if len(faceboxes) == 0:
                count += 1
                if count > 1:
                    # As human eyes move ~20-40ms, if face not detected for more than 1 continuous frame, ignore video
                    print('ERROR: Could not find faces in the image')
                    return None, None, None
                current_frame += 1
                for subject in subjects:
                    subjects[subject].append(None)
                continue
            else:
                count = 0
            
            # Get subjects from faceboxes
            current_subjects = self.landmark_estimator.get_subjects_from_faceboxes(frame, faceboxes)
            
            # Match faceboxes to subjects
            if previous_faceboxes is not None:
                # Match faceboxes to previous faceboxes
                matched_subjects = self.__match_faces(previous_faceboxes, current_subjects)
                
                # If previous subject not detected in current frame, append None to its current frame
                for subject in subjects:
                    if subject not in matched_subjects:
                        matched_subjects[subject] = None       
                        
                for matched_subject in matched_subjects:
                    if matched_subject not in subjects:
                        # Add new dictionary key if new face detected
                        new_subject = matched_subjects[matched_subject]
                        new_subject_index = len(subjects)
                        subjects[new_subject_index] = [None] * current_frame
                        subjects[new_subject_index].append(new_subject)
                        
                        # Extract eye patches
                        self.__extract_eye_image_patches(new_subject)
                        
                        eyes_left[new_subject_index] = [None] * current_frame
                        eyes_right[new_subject_index] = [None] * current_frame
                        if new_subject.left_eye_color is None or new_subject.right_eye_color is None:
                            eyes_left[new_subject_index].append(None)
                            eyes_right[new_subject_index].append(None)
                        else:
                            eyes_left[new_subject_index].append(torch.from_numpy(new_subject.left_eye_color).float())
                            eyes_right[new_subject_index].append(torch.from_numpy(new_subject.right_eye_color).float())
                    else:   
                        # Append subject object depending on subject index (i.e. value of matched_subject)
                        subject = matched_subjects[matched_subject]
                        subjects[matched_subject].append(subject)
                        
                        if subject is not None:
                            # Extract eye patches
                            self.__extract_eye_image_patches(subject)
                            if subject.left_eye_color is None or subject.right_eye_color is None:
                                eyes_left[matched_subject].append(None)
                                eyes_right[matched_subject].append(None)
                            else:
                                eyes_left[matched_subject].append(torch.from_numpy(subject.left_eye_color).float())
                                eyes_right[matched_subject].append(torch.from_numpy(subject.right_eye_color).float())
            else:
                for index in range(len(current_subjects)):
                    subject = current_subjects[index]
                    subjects[index] = [subject]
                    
                    # Extract eye patches
                    self.__extract_eye_image_patches(subject)
                    if subject.left_eye_color is None or subject.right_eye_color is None:
                        eyes_left[index] = [None]
                        eyes_right[index] = [None]
                    else:
                        eyes_left[index] = [torch.from_numpy(subject.left_eye_color).float()]
                        eyes_right[index] = [torch.from_numpy(subject.right_eye_color).float()]
            
            # Link previous faceboxes based on subject index
            previous_faceboxes = {}
            for subject in subjects:
                previous_faceboxes[subject] = subjects[subject][current_frame]
            current_frame += 1
        
        # Remove any subject tracked for less than one second
        removed_subjects = []
        for subject in subjects:
            # Check if faces are semi-tracked
            count = 0 # Count number of frames subject is not detected
            for frame in range(len(subjects[subject])):
                if subjects[subject][frame] is None:
                    count += 1
                    if count > 1:
                        removed_subjects.append(subject)
                        break
                else:
                    count = 0
        
        # Delete subjects that were not fully tracked
        for subject in removed_subjects:
            del subjects[subject]
            
        if len(subjects) > 1:
            print(f"ERROR: More than one subject detected - {len(subjects)}")
            return None, None, None
        elif len(subjects) == 0:
            print("ERROR: No subjects detected")
            return None, None, None
        else:
            # Convert dictionary with single key to list
            subject_idx = list(subjects.keys())[0]
            eyes_left = eyes_left[subject_idx]
            eyes_right = eyes_right[subject_idx]
            
            # Create flag list
            flags = []
            for index in range(len(eyes_left)):
                if eyes_left[index] is None or eyes_right[index] is None:
                    flags.append(False)
                else:
                    flags.append(True)
         
        return eyes_left, eyes_right, flags
    
    
    def extract_lip_landmarks(self, subjects):
        """
        Extract lip landmarks from video frames.
        
        Args:
            subjects: dictionary of subjects extracted from video frames
        Returns:
            lip_landmarks: list of lip landmarks extracted from each frame - attached to subject index
        """
        lip_landmarks = {}
        for subject in subjects:
            frames = subjects[subject]
            lip_landmark = {'upperlip': [], 'lowerlip': []}
            for frame in frames:
                landmarks = frame.landmarks.astype(int)
                # Standard index for upper inner lip landmarks (61-63)
                upperlip_landmarks = landmarks[61:64]
                lip_landmark['upperlip'].append(upperlip_landmarks)
                
                # Standard index for lower inner lip landmarks (65-67)
                lowerlip_landmarks = landmarks[67:64:-1]
                lip_landmark['lowerlip'].append(lowerlip_landmarks)
                
            lip_landmarks[subject] = lip_landmark
 
        return lip_landmarks
