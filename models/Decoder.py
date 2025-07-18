import torch.nn as nn

from models.blocks.decoder import deblock, deblock1, deupcat

def init(parameters):
  self = nn.Module()
  
  channels = parameters.decoder.channels

  self.DeBlock8_1 = deblock1.init( in_channels=channels.b0, out_channels=channels.b1 )
  self.DeBlock8_2 = deblock.init( in_channels=channels.b1, out_channels=channels.b1 )
  self.DeUp4      = deupcat.init( in_channels=channels.b1, out_channels=channels.b2 )
  self.DeBlock4   = deblock.init( in_channels=channels.b2, out_channels=channels.b2 )
  self.DeUp3      = deupcat.init( in_channels=channels.b2, out_channels=channels.b3 )
  self.DeBlock3   = deblock.init( in_channels=channels.b3, out_channels=channels.b3 )
  self.DeUp2      = deupcat.init( in_channels=channels.b3, out_channels=channels.b4 )
  self.DeBlock2   = deblock.init( in_channels=channels.b4, out_channels=channels.b4 )

  self.endconv = nn.Conv3d(channels.b4, 4, kernel_size=1)
  self.Softmax = nn.Softmax(dim=1)
  return self
  
def forward(self, skip1, skip2, skip3, x):

  x = deblock1.forward(self.DeBlock8_1, x)
  x = deblock.forward(self.DeBlock8_2, x)

  x = deupcat.forward( self.DeUp4, x, skip3)  # (1, 64, 32, 32, 32)
  x = deblock.forward(self.DeBlock4, x)

  x = deupcat.forward( self.DeUp3, x, skip2)  # (1, 32, 64, 64, 64)
  x = deblock.forward(self.DeBlock3, x)

  x = deupcat.forward( self.DeUp2, x, skip1)  # (1, 16, 128, 128, 128)
  x = deblock.forward( self.DeBlock2, x)

  x = self.endconv(x)      # (1, 4, 128, 128, 128)
  x = self.Softmax(x)
  return x
