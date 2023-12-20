from machineTranslation.components.common import MultiHeadAttention, LayerNormalization, FeedForwardNetwork, MultiHeadCrossAttention

import torch
import torch.nn as nn

class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 ffn_hidden: int,
                 n_head: int,
                 drop_prob: float):
        super().__init__()
        self.masked_att = MultiHeadAttention(input_dim=d_model,
                                        d_model=d_model,
                                        n_head=n_head)
        self.ffn = FeedForwardNetwork(d_model=d_model,
                                      ffn_hidden=ffn_hidden,
                                      drop_prob=drop_prob)
        self.l_norm1 = LayerNormalization(parameters_shape=[d_model])
        self.l_norm2 = LayerNormalization(parameters_shape=[d_model])
        self.l_norm3 = LayerNormalization(parameters_shape=[d_model])


        self.mcross_att = MultiHeadCrossAttention(d_model=d_model,
                                                  n_head=n_head)



    def forward(self,
                x: torch.tensor,
                mask: torch.tensor,
                enc_out: torch.tensor):
        
        _, att = self.masked_att(x, mask)
        att = self.l_norm1(att + x)

        _, out = self.mcross_att(enc_out, att)
        out = self.l_norm2(out + att)
        
        f_out = self.ffn(out)
        out = self.l_norm2(f_out + out)

        return out


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, mask, y = inputs
        for module in self._modules.values():
            y = module(x, mask, y) #30 x 200 x 512
        return y

class Decoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 ffn_hidden: int,
                 n_head: int,
                 drop_prob: float,
                 n_layers: int):
        super().__init__()
        self.l = SequentialDecoder(*[DecoderLayer(d_model=d_model,
                                              ffn_hidden=ffn_hidden,
                                              n_head=n_head,
                                              drop_prob=drop_prob) for _ in range(n_layers)])

    def forward(self,
                x: torch.tensor,
                mask: torch.tensor,
                enc_out: torch.tensor) -> torch.tensor:
        out = self.l(x, mask, enc_out)
        return out