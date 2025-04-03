import os

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
    
# DFD Models 
class GazePause(nn.Module):
    """
    CNN model for DFD from gaze-pause correlations.
    """
    def __init__(self):
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
        
        # Embedding output
        self.linear = nn.Linear(64 * 23, 256)
        
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
  
    def forward(self, yaw, pitch):
        gaze = torch.stack((yaw, pitch), 1)
        gaze = self.conv1(gaze)
        gaze = self.conv2(gaze)
        gaze = self.conv3(gaze)
        
        gaze = torch.flatten(gaze)
        
        embedding = self.linear(gaze) # Output shape = 256
        
        pred = self.sigmoid(embedding)
        
        return pred, embedding


class GazeDFD(nn.Module):
    """
    Linear model for DFD from [yaw, pitch] gaze
    """
    def __init__(self):
        super(GazeDFD, self).__init__()
        
        # Embedding output   
        self.linear = nn.Sequential(
            nn.Linear(580, 255),
            nn.ReLU()
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

    def forward(self, yaw, pitch):
        gaze = torch.concat((yaw, pitch), 1)
        x = self.linear(gaze)
        x = torch.flatten(x)
        
        pred = self.sigmoid(x)
        
        embedding = torch.concat((x, pred))
        
        return pred, embedding


class PauseDFD(nn.Module):
    """
    Linear model for DFD from tagged pauses.
    """
    def __init__(self):
        super(PauseDFD, self).__init__()
        # Embedding output   
        self.linear = nn.Sequential(
            nn.Linear(290, 255),
            nn.ReLU()
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
        x = torch.flatten(x)
        
        pred = self.sigmoid(x)
        
        embedding = torch.concat((x, pred))
        
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
        
    def train(self, epoch, num_epochs, batch_size):
        self.model.train()
        train_loss = 0
        iter_train = 0
        
        for i, (yaw_corr, pitch_corr, label) in enumerate(self.train_loader):
            yaw_corr = Variable(yaw_corr).cuda(self.device)
            pitch_corr = Variable(pitch_corr).cuda(self.device)
            
            label = Variable(label).cuda(self.device).float().unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            # Predictions
            pred = self.model(yaw_corr, pitch_corr).float()
            
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
        
        with torch.no_grad():
            for i, (yaw_corr, pitch_corr, label) in enumerate(self.val_loader):
                yaw_corr = Variable(yaw_corr).cuda(self.device)
                pitch_corr = Variable(pitch_corr).cuda(self.device)
                
                label = Variable(label).cuda(self.device).float().unsqueeze(1)
                
                # Predictions
                pred = self.model(yaw_corr, pitch_corr).float()
                
                loss = self.criterion(pred, label)
                
                val_loss += loss.item()
                
                iter_val += 1
                
        avg_val_loss = val_loss / len(self.val_loader)
        
        return avg_val_loss
    
class Tester:
    def __init__(self, model, test_loader, criterion, device):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        
    def test(self, plot_path):
        self.model.eval()
        iter_test = 0
        test_loss = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for i, (yaw_corr, pitch_corr, label) in enumerate(self.test_loader):
                yaw_corr = Variable(yaw_corr).cuda(self.device)
                pitch_corr = Variable(pitch_corr).cuda(self.device)
                
                label = Variable(label).cuda(self.device).float().unsqueeze(1)
                
                # Predictions
                pred = self.model(yaw_corr, pitch_corr).float()
                
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
        f1 = f1_score(all_labels, all_preds_binary)
        real = 0
        for i in range(len(all_labels)):
            if all_labels[i] == 0:
                real = i
                break
        
        print(all_labels[real], all_labels[-1])
        print(all_preds[real], all_preds[-1])
        print(all_preds_binary[real], all_preds_binary[-1])
        
        # Calculate and display confusion matrix
        cm = confusion_matrix(all_labels, all_preds_binary)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(plot_path, "ConfusionMatrix.png"), format='png')
        plt.show()
        
        return avg_test_loss, accuracy, roc_auc, f1
   
if __name__ == '__main__':
    model = PauseDFD()
    features = {
        "yaw_corr": torch.rand(1, 290),
        "pitch_corr": torch.rand(1, 290),
        "pauses": torch.rand(1, 290)
    }
    
    pred, embedding = model(features['pauses'])
    print(pred.shape, embedding.shape)
    