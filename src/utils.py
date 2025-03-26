import os
import cv2

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from dataset import *

def gaze_bins(gaze, width, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Section gaze into bins.
    
    Args:
        gaze: numpy array of shape (batch_size, 2) representing the [yaw, pitch] angle
        width: int representing bin width
    Returns:
        yaw_binned: numpy array of shape (batch_size,) representing the yaw bin
        pitch_binned: numpy array of shape (batch_size,) representing the pitch bin
    """
    if isinstance(gaze, np.ndarray):
        bins = np.arange(-90, 90, width)
        gaze_binned = np.digitize(gaze, bins) - 1
    else:
        bins = torch.arange(-90, 90, width).to(device)
        gaze_binned = torch.bucketize(gaze, bins) - 1
        gaze_binned = gaze_binned.to(torch.float32)
    
    return gaze_binned


def gazeto2d(gaze, rad=False):
    """
    Convert 3D gaze vector to 2D gaze (pitch, yaw) angles.
    
    Args:
        gaze: torch tensor of shape (batch_size, 3) representing the 3D gaze vector
        rad: boolean to convert angles to radians
    Returns:
        gaze_2d: numpy array of shape (batch_size, 2) representing the 2D gaze angles
    """
    # Convert gaze vector to pitch and yaw angles
    yaw = torch.atan2(-gaze[:, 0], -gaze[:, 2])
    pitch = torch.asin(-gaze[:, 1])
    gaze_2d = torch.stack([yaw, pitch], dim=1)
    gaze_2d = gaze_2d.cpu().detach().numpy()
    if not rad:
        gaze_2d = (gaze_2d) * (180/np.pi)
    return gaze_2d


# EDIT FOR NEW NN MODEL
def get_vars(model):
    for name, param in model.named_parameters():
        yield param


def load_data(path, batch_size):
    # Load data for train, validation and test
    dataset = DFDC_preprocessed(path)
    
    # Split dataset into train, validation, and test sets
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(120)
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=int(batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


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


def plot_loss(path, epochs, train_loss, val_loss, dataset, log=False):
    # Plot training and validation loss
    fig = plt.figure(figsize=(14, 8))        
    plt.xlabel('Epoch')
    plt.ylabel('Avg error (BCE Loss)')
    plt.title(f'Binary Cross Entropy Loss - {dataset}')
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
