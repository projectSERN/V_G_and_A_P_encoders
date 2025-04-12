import optuna
import os
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from model import *
from src.utils.main_utils import *


def objective(trial, feature, criterion, num_epochs,
              device="cuda" if torch.cuda.is_available() else "cpu"):
    """
    Objective function for Optuna hyperparameter optimization
    """
    lr = trial.suggest_loguniform('lr', 1e-5, 5e-1)
    dropout = trial.suggest_uniform('dropout', 0.01, 0.6)
    lambda2 = trial.suggest_loguniform('lambda2', 1e-6, 0.9)
    width = 10
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    train_loader, val_loader, _ = load_data(batch_size, subset=subset, feature=feature, width=width, device=device)
    
    if feature == 'gaze-pause':
        model = GazePause(dropout=dropout)
    elif feature == 'gaze':
        width = trial.suggest_categorical('width', [2, 6, 10, 15, 20, 30, 45])
        model = GazeDFD(dropout=dropout)
    elif feature == 'pause':
        model = PauseDFD(dropout=dropout)
    model.to(device)
    model.init_weights()
    torch.backends.cudnn.benchmark = True
    
    optimizer = torch.optim.Adam(get_vars(model), lr=lr, weight_decay=lambda2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    trainer = Trainer(model, train_loader, criterion, optimizer, device, scheduler)
    validator = Validator(model, val_loader, criterion, device)
    
    # Settings for early stopping
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Other prerequisites
    best_val_acc = 0
    
    # Training and validation
    for epoch in tqdm(range(num_epochs)):
        # Train DFD model
        train_loss = trainer.train()
        val_loss, accuracy = validator.validate()
        
        best_val_acc = max(best_val_acc, accuracy)
        
        trial.report(accuracy, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        # Early stopping
        best_val_loss, epochs_no_improve = early_stopping(val_loss, best_val_loss, epochs_no_improve, epoch, model)
            
        # Triggering early stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered...")
            break
    
    return accuracy

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Deepfake detection from Gaze and Pauses.')
    # Dataset args ---------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--dataset', dest='dataset', help='DFDC, LAV-DF',
        default= "DFDC", type=str)
    # DFDC
    parser.add_argument(
        '--dfdc_dir', dest='dfdc_dir', help='Directory path for DFDC dataset.',
        default='/scratch/zceenaa/DFDC_preprocessed/', type=str)
    parser.add_argument(
        '--subset', dest='subset', help='Index of DFDC subset.',
        default=['05', '06', '07'], type=list)
    # Detector args --------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--feature', dest='feature', help='Example: gaze-pause, gaze, pause.',
        default='gaze', type=str)
    # Training args --------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--num_epochs', dest='num_epochs', help='Maximum number of training epochs.',
        default=50, type=int)
    parser.add_argument(
        '--patience', dest='patience', help='Patience for early stopping.',
        default=5, type=int)
    # Other args --------------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------
    parser.add_argument(
        '--output', dest='output', help='Path of output models.',
        default='/scratch/zceenaa/DFD/', type=str)
    parser.add_argument(
        '--gpu', dest='gpu', help='GPU device id to use [0] or multiple 0,1,2,3',
        default='0', type=str)
    
    args = parser.parse_args()
    return args


#===========MAIN===========
if __name__ == '__main__':
    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    feature = args.feature
    data_set = args.dataset
    subset = args.subset
    
    # Define loss functions
    print('Defining loss and optimizer functions...')
    criterion = nn.BCELoss().to(device)
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, feature, criterion, num_epochs, device), n_trials=200)
    print("Best hyperparameters:", study.best_params)