import torch.nn as nn

from models import Encoder, Decoder, Transformer

class UnetTransformer(nn.Module):
  def __init__( self, parameters ):
    super(UnetTransformer, self).__init__()

    self.Encoder = Encoder.init( parameters=parameters )
    self.Transformer = Transformer.init( parameters=parameters )
    #self.pre_head_ln = nn.LayerNorm( parameters.embedding_dim )
    self.Decoder = Decoder.init( parameters=parameters )

  def forward(self, x):
    skip1, skip2, skip3, x = Encoder.forward( self.Encoder, x=x )

    # apply transformer
    x = Transformer.forward( self.Transformer, x=x)
    #x = self.pre_head_ln(x)

    decoder_output = Decoder.forward( self.Decoder, skip1=skip1, skip2=skip2, skip3=skip3, x=x )

    return decoder_output
