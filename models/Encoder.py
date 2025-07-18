import torch.nn as nn

from models.blocks.encoder import learnedpositional, fixedpositional
from models.blocks.encoder import endown, enblock, initconv

def get_positional_encoding(parameters):
  return learnedpositional.LearnedPositionalEncoding( parameters ) if parameters.encoder.positional_encoding_type == "learned" else fixedpositional.FixedPositionalEncoding( parameters )

def init( parameters ):
  assert parameters.embedding_dim % parameters.transformer.heads == 0
  self = nn.Module()

  self.embedding_dim = parameters.embedding_dim
  self.conv_patch_representation = parameters.encoder.conv_patch_representation
  self.flatten_dim = parameters.encoder.flatten_dim

  channels = parameters.encoder.channels
  self.InitConv = initconv.init( in_channels=channels.b0, out_channels=channels.b1, dropout=0.2 )

  self.EnBlock1   = enblock.init( in_channels=channels.b1 )
  self.EnDown1     = endown.init( in_channels=channels.b1, out_channels=channels.b2 )
  self.EnBlock2_1 = enblock.init( in_channels=channels.b2 )
  self.EnBlock2_2 = enblock.init( in_channels=channels.b2 )
  self.EnDown2     = endown.init( in_channels=channels.b2, out_channels=channels.b3 )
  self.EnBlock3_1 = enblock.init( in_channels=channels.b3 )
  self.EnBlock3_2 = enblock.init( in_channels=channels.b3 )
  self.EnDown3     = endown.init( in_channels=channels.b3, out_channels=channels.b4 )
  self.EnBlock4_1 = enblock.init( in_channels=channels.b4 )
  self.EnBlock4_2 = enblock.init( in_channels=channels.b4 )
  self.EnBlock4_3 = enblock.init( in_channels=channels.b4 )
  self.EnBlock4_4 = enblock.init( in_channels=channels.b4 )

  self.bn = nn.BatchNorm3d( channels.b4 )
  self.relu = nn.ReLU( inplace=True )
  if self.conv_patch_representation:
    self.conv_x = nn.Conv3d( in_channels=channels.b4, out_channels=self.embedding_dim, kernel_size=3, stride=1, padding=1 )
    
  self.linear_encoding = nn.Linear(self.flatten_dim, self.embedding_dim)
  self.position_encoding = get_positional_encoding(parameters=parameters)
  self.pe_dropout = nn.Dropout(p=parameters.dropout_rate)
  return self


def forward(self, x):
  x = initconv.forward( self.InitConv, x )       # (1, 16, 128, 128, 128)

  skip1 = enblock.forward( self.EnBlock1, x )
  x = endown.forward( self.EnDown1, skip1 )  # (1, 32, 64, 64, 64)

  x = enblock.forward( self.EnBlock2_1, x )
  skip2 = enblock.forward( self.EnBlock2_2, x )
  x = endown.forward( self.EnDown2, skip2 )  # (1, 64, 32, 32, 32)

  x = enblock.forward( self.EnBlock3_1, x )
  skip3 = enblock.forward( self.EnBlock3_2, x )
  x = endown.forward( self.EnDown3, skip3 )  # (1, 128, 16, 16, 16)

  x = enblock.forward( self.EnBlock4_1, x )
  x = enblock.forward( self.EnBlock4_2, x )
  x = enblock.forward( self.EnBlock4_3, x )
  x = enblock.forward( self.EnBlock4_4, x )  # (1, 128, 16, 16, 16)
    
  # TODO conv_patch_representation meaning?
  if self.conv_patch_representation:
    # combine embedding with conv patch distribution
    #x1_1, x2_1, x3_1, x = self.Unet(x)
    x = self.bn(x)
    x = self.relu(x)
    x = self.conv_x(x)
    x = x.permute(0, 2, 3, 4, 1).contiguous()
    x = x.view(x.size(0), -1, self.embedding_dim)
  else:
    #x = self.Unet(x)
    x = self.bn(x)
    x = self.relu(x)
    x = ( x.unfold(2, 2, 2).unfold(3, 2, 2).unfold(4, 2, 2).contiguous() )
    x = x.view(x.size(0), x.size(1), -1, 8)
    x = x.permute(0, 2, 3, 1).contiguous()
    x = x.view(x.size(0), -1, self.flatten_dim)
    x = self.linear_encoding(x)

  #print('shape x ', x.shape)
  x = self.position_encoding(x)
  x = self.pe_dropout(x)

  return skip1, skip2, skip3, x
