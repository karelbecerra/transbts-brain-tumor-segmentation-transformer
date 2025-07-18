import torch.nn as nn
from collections import OrderedDict

def init(in_channels, out_channels):
  return nn.Sequential(
    OrderedDict([
      ('conv', nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1) ),
    ])
  )

def forward(self, x):
    return self(x)