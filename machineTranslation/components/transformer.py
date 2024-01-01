from machineTranslation.components.encoder import Encoder
from machineTranslation.components.decoder import Decoder

import torch

d_model = 512
n_heads = 8
drop_prob = 0.1
batch_size = 32
max_seq_len = 200
ffn_hidden = 2048
n_layers = 5
x = torch.randn( (batch_size, max_seq_len, d_model)) 

dec = Decoder(d_model,
             ffn_hidden,
             n_heads, 
             drop_prob,
             1)
# mask = torch.full([max_seq_len, max_seq_len] , float('-inf'))
# mask = torch.triu(mask, diagonal=1)
print(dec(x=x, enc_out=x, mask=None).shape)