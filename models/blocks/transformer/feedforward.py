import torch.nn as nn

class FeedForward(nn.Module):
  def __init__(self, dim, hidden_dim, dropout_rate):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(p=dropout_rate), 
      nn.Linear(hidden_dim, dim), nn.Dropout(p=dropout_rate),
    )

  def forward(self, x):
    return self.net(x)