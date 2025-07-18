import torch
import torch.nn as nn

def init(in_channels, out_channels):
  self = nn.Module()
  self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
  self.conv2 = nn.ConvTranspose3d(out_channels, out_channels, kernel_size=2, stride=2)
  self.conv3 = nn.Conv3d(out_channels*2, out_channels, kernel_size=1)
  return self

def forward(self, x, prev):
  x1 = self.conv1(x)
  y = self.conv2(x1)
  # y = y + prev
  y = torch.cat((prev, y), dim=1)
  y = self.conv3(y)
  return y