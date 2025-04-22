import torch
import torch.nn as nn
import torch.nn.init as init


class ARNet(nn.Module):
    def __init__(self, dropout=0.5):
        super(ARNet, self).__init__()
        self.arnetbase = self.basecnn(500)
        self.arnetdense = self.dense(1000, 500)
        self.arnetsplit = self.arnet_split(dropout)
        self.linear = nn.Linear(1500, 3)

    def arnet_split(self, dropout=0.5):
        layers = []
        layers.append(self.basecnn(1000))
        layers.append(nn.Dropout(p=dropout))
        layers.append(self.dense(1000, 500))
        
        return nn.Sequential(*layers)
    
    def basecnn(self, units=500):
        layers = []
        
        layers.append(self.Conv2d(3, 64, padding1=[1], padding2=[1]))
        layers.append(self.Conv2d(64, 128, padding1=[1], padding2=[1]))
        layers.append(self.Conv2d(128, 256, padding1=[0], padding2=[0]))
        layers.append(nn.Flatten())
        layers.append(self.dense(256 * 4, units))
        
        return nn.Sequential(*layers)
    
    def Conv2d(self, input_ch, output_ch, kernel=3, padding1=[2,2], padding2=[2,2]):
        layers = []
        layers.append(nn.Conv2d(input_ch, output_ch, kernel_size=kernel, padding=padding1, stride=1))
        layers.append(nn.BatchNorm2d(output_ch))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(output_ch, output_ch, kernel_size=kernel, padding=padding2, stride=2))
        layers.append(nn.BatchNorm2d(output_ch))
        layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.MaxPool2d(kernel_size=2, stride=1))
        
        return nn.Sequential(*layers)
    
    def dense(self, input_ch, output_ch, bias=0.1):
        layers = []
        fc = nn.Linear(input_ch, output_ch)
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(fc.weight)
        # Initialize biases with a constant value
        nn.init.constant_(fc.bias, bias)
        
        layers.append(fc)
        
        layers.append(nn.ReLU(inplace=True))
        
        return nn.Sequential(*layers)
  
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')
  
    def forward(self, left, right):
        # Frame = single input
        # extract left and right eye images (create function)
        # transformations [36 * 60 * 3] image required
         
        # AR-NET (both)
        arnet_lboth = self.arnetbase(left)
        arnet_rboth = self.arnetbase(right)
        
        arnet_both = torch.concat((arnet_lboth, arnet_rboth), 1)
        arnet_both = self.arnetdense(arnet_both)
        
        # AR-NET (split)
        arnet_lsplit = self.arnetsplit(left)
        arnet_rsplit = self.arnetsplit(right)
        
        finalfeature = torch.concat([arnet_both, arnet_lsplit, arnet_rsplit],1)

        eyegaze = self.linear(finalfeature)
        
        # Gaze vector is normalized
        eyegaze = eyegaze.to(torch.float64)
        eyegaze = eyegaze / torch.sqrt(torch.sum(eyegaze ** 2, dim=1, keepdim=True))
        eyegaze = eyegaze.to(torch.float32)
        
        return eyegaze
    
    def __call__(self, left, right):
        return self.forward(left, right)