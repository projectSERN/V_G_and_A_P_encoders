import os
import cv2

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

from dataset import DFDC_preprocessed

def get_vars(model):
    """
    Get model parameters.
    
    Args:
        model: DFD model
    Returns:
        param: model parameters
    """
    for name, param in model.named_parameters():
        yield param


def load_data(batch_size, subset='06', feature='gaze', width=10,
              device="cuda" if torch.cuda.is_available() else "cpu"):
    # Load the train, validation and test datasets
    train_dataset = DFDC_preprocessed(subset=subset, split='train', feature=feature, width=width)
    val_dataset = DFDC_preprocessed(subset=subset, split='val', feature=feature, width=width)
    test_dataset = DFDC_preprocessed(subset=subset, split='test', feature=feature, width=width)

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
    """
    Early stopping algorithm.
    
    Args:
        avg_val_loss: average validation loss (float)
        best_val_loss: best validation loss (float)
        epochs_no_improve: number of epochs without improvement to validation loss (int)
        epoch: current epoch (int)
        model: Gaze_Pause DFD model
        output: output directory (str)
    """
    # Early stopping algorithm
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(output, 'best_model.pth')) 
    else:
        epochs_no_improve += 1
        
    return best_val_loss, epochs_no_improve


def loger(epoch, train_loss, val_loss, outfile, model, output):
    """
    Log the average loss for each epoch into a .txt file.
    
    Args:
        epoch: current epoch (int)
        train_loss: epoch's train loss (float)
        val_loss: epoch's validation loss (float)
        outfile: loss loger file (str)
        model: Gaze_Pause DFD model
        output: output directory (str)
    """
    # Log the average loss for each epoch
    loger = f"[epoch---{epoch+1}] Train Loss:{train_loss}, Val Loss:{val_loss}\n"
    outfile.write(loger)
    print(loger)
    
    print('Taking snapshot...')
    torch.save(model.state_dict(), output +'/'+ '_epoch_' + str(epoch+1) + '.pth')


def plot_loss(path, epochs, train_loss, val_loss, dataset, log=False):
    """
    Plot training and validation losses for all epochs.
    
    Args:
        path: directory to save plot (str)
        epochs: list of epochs the model was trained over (list)
        train_loss: all train losses (list)
        val_loss: all validation losses (list)
        dataset: name of dataset model was trained on (str)
        log: trigger plotting on log scale (bool)
    """
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

