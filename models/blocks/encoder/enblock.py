import torch.nn as nn
from collections import OrderedDict

def normalization(in_channels, norm='bn', groups=8):
    if norm == 'bn':
        m = nn.BatchNorm3d(num_features=in_channels)
    elif norm == 'gn':
        m = nn.GroupNorm(num_groups=groups, num_channels=in_channels)
    elif norm == 'in':
        m = nn.InstanceNorm3d(num_features=in_channels)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

def init(in_channels):
  return nn.Sequential(
    OrderedDict([
      ('bn1',   normalization(in_channels) ),
      ('relu1', nn.ReLU(inplace=True) ),
      ('conv1', nn.Conv3d(in_channels, out_channels=in_channels, kernel_size=3, padding=1) ),
      ('bn2',   normalization(in_channels) ),
      ('relu2', nn.ReLU(inplace=True) ),
      ('conv2', nn.Conv3d(in_channels, out_channels=in_channels, kernel_size=3, padding=1) ),
    ])
  )

def forward(self, x):
  return self(x) + x