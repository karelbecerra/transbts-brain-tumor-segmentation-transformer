import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
  def __init__(self, parameters):
    super(LearnedPositionalEncoding, self).__init__()
    self.position_embeddings = nn.Parameter(torch.zeros(1, parameters.encoder.position_embeddings_x, parameters.encoder.position_embeddings_y)) #8x

  def forward(self, x):
    return x + self.position_embeddings