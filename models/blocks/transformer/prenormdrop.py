import torch.nn as nn

class PreNormDrop(nn.Module):
  def __init__(self, dim, dropout_rate, fn):
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.dropout = nn.Dropout(p=dropout_rate)
    self.fn = fn

  def forward(self, x):
    return self.dropout(self.fn(self.norm(x)))