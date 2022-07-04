import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from dataclasses import dataclass
from collections import OrderedDict



class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out,dilation):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True,dilation=dilation),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True,dilation=dilation),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x
         
class CNN_model_dilated_conv(nn.Module):
    def __init__(self,n_channels, n_st ,n_classes,return_embs=False):
        super().__init__()
        self.relu = nn.ReLU()
        
        self.conv1 = conv_block(n_channels, n_st*2,dilation=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = conv_block(n_st*2, n_st*4,dilation=2)
        #self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = conv_block(n_st*4, n_st*8,dilation=3)
        #self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.conv4 = conv_block(n_st*8, n_st*16,dilation=1)
        
        #self.adapool=nn.AdaptiveAvgPool2d((12,8))
        self.dropout = nn.Dropout(0.6)
        self.dropout2d = nn.Dropout2d(0.1)
        self.fc1 = nn.Linear( n_st*16, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, n_classes)
        self.return_embs = return_embs
    
    def forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = self.dropout2d(self.maxpool(self.conv2(x)))
        x = self.dropout2d(self.maxpool(self.conv3(x)))
        x = self.dropout2d(self.conv4(x))
       
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
        
        out_emb = F.normalize(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        if self.return_embs:
            return out_emb, x
        else:
            return x

'''
x=torch.rand([4,3,2736,1824])
model=small_model(n_channels=3, n_st=4,n_classes=4)

model(x)
'''
