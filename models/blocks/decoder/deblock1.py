import torch.nn as nn
from collections import OrderedDict

def init(in_channels, out_channels):
  return nn.Sequential(
    OrderedDict([
      ('conv1', nn.Conv3d( in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1 ) ),
      ('bn1',   nn.BatchNorm3d( num_features=out_channels ) ),
      ('relu1', nn.ReLU(inplace=True) ),
      ('conv2', nn.Conv3d( in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1 ) ),
      ('bn2',   nn.BatchNorm3d( num_features=out_channels ) ),
      ('relu2', nn.ReLU(inplace=True) ),
    ])
  )

def forward(self, x):
  return self(x)
