import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
    
# DFD Models 
class GazePause(nn.Module):
    """
    CNN model for DFD from gaze-pause correlations.
    """
    def __init__(self,dropout=0.5):
        super(GazePause, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16,32, kernel_size=4, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32,64, kernel_size=3, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        # self.conv4 = nn.Sequential(
        #     nn.Conv1d(64,128, kernel_size=3, stride=2),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU()
        # )
        
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 23, 512),
            nn.Dropout(p=dropout),
            nn.ReLU()
        )
        
        # Embedding output
        # self.linear = nn.Linear(128 * 11, 256)
        self.linear = nn.Linear(512, 256)
        
        # Classifier head
        self.sigmoid = nn.Sequential(
            nn.Linear(256,1),
            nn.Sigmoid()
        )
  
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                if m.out_features == 1:
                    init.kaiming_normal_(m.weight, nonlinearity='sigmoid')
                else:
                    init.kaiming_normal_(m.weight, nonlinearity='relu')
  
    def forward(self, corr):
        x = self.conv1(corr)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        embedding = self.linear(x) # Output shape = 256
        
        pred = self.sigmoid(embedding)
        
        return pred, embedding


class GazeDFD(nn.Module):
    """
    Linear model for DFD from [yaw, pitch] gaze
    """
    def __init__(self, dropout=0.5):
        super(GazeDFD, self).__init__()
        # Convolutional Layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(16,32, kernel_size=4, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(32,64, kernel_size=3, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Embedding output   
        self.linear = nn.Linear(64 * 23, 255)
        
        # Classifier head
        self.sigmoid = nn.Sequential(
            nn.Linear(255,1),
            nn.Sigmoid()
        )
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == 1:
                    init.kaiming_normal_(m.weight, nonlinearity='sigmoid')
                else:
                    init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self,gaze):
        x = self.conv1(gaze)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = torch.flatten(x, 1)
        
        x = self.linear(x)
        
        pred = self.sigmoid(x)
        
        embedding = torch.cat((x, pred), dim=1)
        
        return pred, embedding


class PauseDFD(nn.Module):
    """
    Linear model for DFD from tagged pauses.
    """
    def __init__(self, dropout=0.5):
        super(PauseDFD, self).__init__()
        # Embedding output   
        self.linear = nn.Sequential(
            nn.Linear(290, 255),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )
        
        # Classifier head
        self.sigmoid = nn.Sequential(
            nn.Linear(255,1),
            nn.Sigmoid()
        )
    
    def init_weights(self):
        for m in self.modules():
            if m.out_features == 1:
                init.kaiming_normal_(m.weight, nonlinearity='sigmoid')
            else:
                init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, pauses):
        x = self.linear(pauses)
        
        pred = self.sigmoid(x)
        
        embedding = torch.cat((x, pred), dim=1)
        
        return pred, embedding
    

# Trainers
class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, device, scheduler=None):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
    def train(self):
        self.model.train()
        train_loss = 0
        iter_train = 0
        
        for i, (data, label) in enumerate(self.train_loader):
            data = Variable(data).to(self.device)
            
            label = Variable(label).to(self.device).float().unsqueeze(1)
                        
            self.optimizer.zero_grad()
            
            # Predictions
            pred, _ = self.model(data)
            pred = pred.float()
            
            loss = self.criterion(pred, label)
            train_loss += loss.item()
            
            loss.backward()
            self.optimizer.step()
            
            iter_train += 1
        
        avg_train_loss = train_loss / len(self.train_loader)
        
        if self.scheduler:
            self.scheduler.step()
        
        return avg_train_loss
    
class Validator:
    def __init__(self, model, val_loader, criterion, device):
        self.model = model
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        
    def validate(self):
        self.model.eval()
        val_loss = 0
        iter_val = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for i, (data, label) in enumerate(self.val_loader):
                data = Variable(data).to(self.device)
                
                label = Variable(label).to(self.device).float().unsqueeze(1)
                
                # Predictions
                pred, _ = self.model(data)
                pred = pred.float()
                
                loss = self.criterion(pred, label)
                
                val_loss += loss.item()
                
                iter_val += 1
                
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
                
        avg_val_loss = val_loss / len(self.val_loader)
        
        # Calculate accuracy
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_preds_binary = (all_preds > 0.5).astype(int)  # Convert probabilities to binary predictions
        accuracy = accuracy_score(all_labels, all_preds_binary)
        
        return avg_val_loss, accuracy
    
class Tester:
    def __init__(self, model, test_loader, criterion, device):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        
    def test(self):
        self.model.eval()
        iter_test = 0
        test_loss = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for i, (data, label) in enumerate(self.test_loader):
                data = Variable(data).to(self.device)
                
                label = Variable(label).to(self.device).float().unsqueeze(1)
                
                # Predictions
                pred, _ = self.model(data)
                pred = pred.float()
                
                loss = self.criterion(pred, label)
                
                test_loss += loss.item()
                iter_test += 1
                
                all_labels.extend(label.cpu().numpy())
                all_preds.extend(pred.cpu().numpy())
                
        avg_test_loss = test_loss / len(self.test_loader)
        # Calculate accuracy and ROC AUC
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_preds_binary = (all_preds > 0.5).astype(int)  # Convert probabilities to binary predictions
        accuracy = accuracy_score(all_labels, all_preds_binary)
        roc_auc = roc_auc_score(all_labels, all_preds)
        
        return avg_test_loss, accuracy, roc_auc



if __name__ == "__main__":
    model = GazePause()
    feature = torch.zeros((1, 2, 290))
    model(feature)