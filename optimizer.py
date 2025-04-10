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
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    dropout = trial.suggest_uniform('dropout', 0.1, 0.5)
    lambda2 = trial.suggest_loguniform('lambda2', 1e-6, 1e-2)
    width = 10
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    
    train_loader, val_loader, _ = load_data(batch_size, subset=subset, feature=feature, width=width, device=device)
    
    if feature == 'gaze-pause':
        model = GazePause()
    elif feature == 'gaze':
        model = GazeDFD(dropout=dropout)
        width = trial.suggest_categorical('width', [2, 6, 10, 15, 20, 30, 45])
    elif feature == 'pause':
        model = PauseDFD(dropout=dropout)
    model.to(device)
    torch.backends.cudnn.benchmark = True
    
    optimizer = torch.optim.Adam(get_vars(model), lr=lr, weight_decay=lambda2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    trainer = Trainer(model, train_loader, criterion, optimizer, device, scheduler)
    validator = Validator(model, val_loader, criterion, device)
    
    # Other prerequisites
    best_val_loss = float('inf')
    
    # Training and validation
    for epoch in tqdm(range(num_epochs)):
        # Train DFD model
        train_loss = trainer.train()
        val_loss = validator.validate()
        
        best_val_loss = min(best_val_loss, val_loss)
        
        trial.report(val_loss, epoch)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return val_loss

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
        default='06', type=str)
    parser.add_argument(
        '--width', dest='width', help='Width of gaze bins.',
        default=10, type=int)
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
        '--batch_size', dest='batch_size', help='Batch size.',
        default=64, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Learning rate.',
        default=0.01, type=float)
    parser.add_argument(
        '--dropout', dest='dropout', help='Dropout rate.',
        default=0.5, type=float)
    parser.add_argument(
        '--lambda2', dest='lambda2', help='L2 Regularization lambda.',
        default=1e-4, type=float)
    parser.add_argument(
        '--patience', dest='patience', help='Patience for early stopping.',
        default=10, type=int)
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
    batch_size = args.batch_size
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    feature = args.feature
    data_set = args.dataset
    output = os.path.join(args.output, feature)
    width = args.width
    if feature == 'gaze':
        output = os.path.join(output, f'width={width}')
    lr = args.lr
    lambda2 = args.lambda2
    dropout = args.dropout
    subset = args.subset
    
    # Load data
    print('Loading data...')
    train_loader, val_loader, test_loader = load_data(batch_size, subset=subset, feature=feature, width=width, device=device)
    
    # Define loss functions
    print('Defining loss and optimizer functions...')
    criterion = nn.BCELoss().to(device)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, feature, criterion, num_epochs, device), n_trials=200)
    print("Best hyperparameters:", study.best_params)