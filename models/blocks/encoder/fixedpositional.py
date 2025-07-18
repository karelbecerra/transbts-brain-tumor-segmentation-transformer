import torch
import torch.nn as nn
import torch.nn.functional as F

class FixedPositionalEncoding(nn.Module):
  def __init__(self, parameters):
    super(FixedPositionalEncoding, self).__init__()
    # TODO
    max_length = 512
    pe = torch.zeros(parameters.encoder.max_length, parameters.embedding_dim)
    position = torch.arange(0, parameters.encoder.max_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, parameters.embedding_dim, 2).float()
        * (-torch.log(torch.tensor(10000.0)) / parameters.embedding_dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(0, 1)
    self.register_buffer('pe', pe)

  def forward(self, x):
    x = x + self.pe[: x.size(0), :]
    return x

