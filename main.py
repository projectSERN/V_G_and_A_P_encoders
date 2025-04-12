import os
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from tqdm import tqdm

from model import *
from src.utils.main_utils import *

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
    
    # Set up summary folder
    summary_name = '{}{}_{}_{}_{}_{}'.format('DFD-', data_set, batch_size, lr, dropout, lambda2)
    output=os.path.join(output, summary_name)
    if not os.path.exists(output):
        os.makedirs(output)
        
    # Set up txt file to write avg epoch loss
    outfile = open(output + '/loss_loger.txt', "w")
    outfile.write(f"Dataset={data_set} | batch_size={batch_size}, lr={lr}, dropout={dropout}, lambda2={lambda2}\n")
    
    configuration = f"\nTrain Configuration, gpu_id={device}, batch_size={batch_size}, lr={lr}, dropout={dropout}, lambda2={lambda2}\nStart testing dataset={data_set}------------------------- \n"
    print(configuration)
    
    # Load data
    print('Loading data...')
    train_loader, val_loader, test_loader = load_data(batch_size, subset=subset, feature=feature, width=width, device=device)
            
    # Define loss functions
    print('Defining loss and optimizer functions...')
    criterion = nn.BCELoss().to(device)
    
    # Load detection model
    print(f'Loading {feature} DFD model...')
    if feature == 'gaze-pause':
        model = GazePause()
    elif feature == 'gaze':
        model = GazeDFD(dropout=dropout)
    elif feature == 'pause':
        model = PauseDFD(dropout=dropout)
    model.to(device)
    torch.backends.cudnn.benchmark = True
    
    # Define optimizer
    optimizer = torch.optim.Adam(get_vars(model), lr=lr, weight_decay=lambda2)
    # optimizer = torch.optim.Adam(get_vars(model), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    trainer = Trainer(model, train_loader, criterion, optimizer, device, scheduler)
    validator = Validator(model, val_loader, criterion, device)
    
    # Settings for early stopping
    patience = args.patience
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Other prerequisites
    start = 0
    min_error = 0
    all_train_loss = []
    all_val_loss = []
    
    # Training and validation
    for epoch in tqdm(range(start,num_epochs)):
        # Train DFD model
        train_loss = trainer.train()
        val_loss, accuracy = validator.validate()
        
        all_train_loss.append(train_loss)
        all_val_loss.append(val_loss)
        
        # Logging
        loger(epoch, train_loss, val_loss, outfile, model, output)
        
        # Early stopping
        best_val_loss, epochs_no_improve = early_stopping(val_loss, best_val_loss, epochs_no_improve, epoch, model, output)
            
        # Triggering early stopping
        if epochs_no_improve >= patience:
            print("Early stopping triggered...")
            break

    outfile.close()

    # Plot training and validation loss
    epoch_list = [x for x in range(1, len(all_train_loss)+1)]
    if feature =='gaze':
        plot_path = os.path.join("evaluation", feature, f'width={width}', summary_name)
    else:
        plot_path = os.path.join("evaluation", feature, summary_name)
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)
    plot_loss(plot_path, epoch_list, all_train_loss, all_val_loss, data_set, log=True)


    # Test the best model
    configuration = f"\nStart testing dataset={data_set}----------------------------------------\n"
    print(configuration)

    # Base network structure
    print("Loading best model...")
    if feature == 'gaze-pause':
        model_test = GazePause()
    elif feature == 'gaze':
        model_test = GazeDFD()
    elif feature == 'pause':
        model_test = PauseDFD()
    best_model_path = os.path.join(output, 'best_model.pth')
    saved_state_dict = torch.load(best_model_path, weights_only=True)
    model_test.load_state_dict(saved_state_dict)
    model_test.to(device)
    tester = Tester(model_test, test_loader, criterion, device)

    metrics = tester.test()
    print(f"BCE: {metrics[0]} | Accuracy: {metrics[1]} | ROC AUC: {metrics[2]}")