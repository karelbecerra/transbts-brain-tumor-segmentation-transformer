import torch.nn as nn
import torch.nn.functional as F

def init(in_channels=4, out_channels=16, dropout=0.2):
  self = nn.Module()
  self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
  self.dropout = dropout
  return self

def forward(self, x):
  return F.dropout3d( self.conv(x), self.dropout)

