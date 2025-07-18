import torch.nn as nn

from models.blocks.transformer import residual, prenormdrop, selfattention, prenorm, feedforward, intermediatesequential

def init( parameters ):
  assert parameters.transformer.img_dim % parameters.transformer.patch_dim == 0
  self = nn.Module()
  layers = []
  self.img_dim = parameters.transformer.img_dim
  self.patch_dim = parameters.transformer.patch_dim
  self.embedding_dim = parameters.embedding_dim

  dim = parameters.transformer.dim
  dropout_rate = parameters.transformer.dropout_rate
  attn_dropout_rate = parameters.transformer.attn_dropout_rate
  heads = parameters.transformer.heads
  hidden_dim = parameters.transformer.hidden_dim
  for _ in range(parameters.transformer.num_layers):
    layers.extend([
        residual.Residual(
          prenormdrop.PreNormDrop( dim, dropout_rate, selfattention.SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate)) 
          ),
        residual.Residual( prenorm.PreNorm(dim, feedforward.FeedForward( dim=dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate)) ),
      ]
    )
    # dim = dim / 2
  self.seq = intermediatesequential.IntermediateSequential(*layers)
  return self

def _reshape_output(self, x):
  dim = int(self.img_dim / self.patch_dim)
  x = x.view( x.size(0), dim, dim, dim, self.embedding_dim)
  x = x.permute(0, 4, 1, 2, 3).contiguous()
  return x

def transformer_to_decoder(self, x, intmd_layers=[1, 2, 3, 4]):
  assert intmd_layers is not None, "pass the intermediate layers for MLA"
  encoder_outputs = {}
  all_keys = []
  for i in intmd_layers:
    val = str(2 * i - 1)
    _key = 'Z' + str(i)
    all_keys.append(_key)
    encoder_outputs[_key] = x[val]
  all_keys.reverse()

  x = encoder_outputs[all_keys[0]]
  x = _reshape_output(self, x)
  return x

def forward(self, x):
  x = self.seq(x)
  return transformer_to_decoder(self, x)
