from machineTranslation.components.common import MultiHeadAttention, LayerNormalization, FeedForwardNetwork

import torch
import torch.nn as nn

class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 ffn_hidden: int,
                 n_head: int,
                 drop_prob: float):
        super().__init__()
        self.m_att = MultiHeadAttention(input_dim=d_model,
                                        d_model=d_model,
                                        n_head=n_head)
        self.l_norm1 = LayerNormalization(parameters_shape=[d_model])
        self.ffn = FeedForwardNetwork(d_model=d_model,
                                      ffn_hidden=ffn_hidden,
                                      drop_prob=drop_prob)
        self.l_norm2 = LayerNormalization(parameters_shape=[d_model])

    def forward(self,
                x):
        _, att = self.m_att(x)
        att = self.l_norm1(att + x)

        out = self.ffn(att)
        out = self.l_norm2(att + out)

        return out


class Encoder(nn.Module):
    def __init__(self,
                 d_model: int,
                 ffn_hidden: int,
                 n_head: int,
                 drop_prob: float,
                 n_layers: int):
        super().__init__()
        self.l = nn.Sequential(*[EncoderLayer(d_model=d_model,
                                              ffn_hidden=ffn_hidden,
                                              n_head=n_head,
                                              drop_prob=drop_prob) for _ in range(n_layers)])

    def forward(self,
                x: torch.tensor) -> torch.tensor:
        out = self.l(x)
        return out