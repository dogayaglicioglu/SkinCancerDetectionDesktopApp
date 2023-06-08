import torch.nn as nn
from torch.nn.functional import relu, sigmoid, leaky_relu
import numpy as np
from torch import Tensor
import torch

class SkinCancer(nn.Module):
    def __init__(self,num_classes=2):
        super(SkinCancer,self).__init__()
        
        self.conv1=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(num_features=32)
        self.relu1=nn.ReLU()
        self.pool=nn.MaxPool2d(kernel_size=2)
        
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(num_features=64)
        self.relu2=nn.ReLU()
        
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.bn3=nn.BatchNorm2d(num_features=128)
        self.relu3=nn.ReLU()
        
        self.conv4=nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1)
        self.bn4=nn.BatchNorm2d(num_features=256)
        self.relu4=nn.ReLU()
        
        self.gap=nn.AdaptiveAvgPool2d((1,1))
        self.fc1=nn.Linear(in_features=256,out_features=128)
        self.relu5=nn.ReLU()
        self.fc2=nn.Linear(in_features=128,out_features=num_classes)
        
        self.sigmoid=nn.Sigmoid()
        
    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)

        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.pool(output)

        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)
        output = self.pool(output)

        output = self.conv4(output)
        output = self.bn4(output)
        output = self.relu4(output)
        output = self.gap(output)
        output = output.view(-1, 256)
        output = self.fc1(output)
        output = self.relu5(output)
        output = self.fc2(output)
        output = self.sigmoid(output)
        return output
model2 = SkinCancer(num_classes=2)
classes = ['benign','malignant']