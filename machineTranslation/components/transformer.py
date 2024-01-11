from machineTranslation.components.encoder import Encoder
from machineTranslation.components.decoder import Decoder
from machineTranslation.components.common import PositionalEncoding
import torch.nn as nn


MAX_SEQ_LEN = 1024 # 

class Transformer(nn.Module):
    def __init__(self,
               d_model: int,
               ffn_hidden: int,
               n_head: int,
               drop_prob: int,
               n_layers: int,
               hi_vocab_size: int):
            super().__init__()

            self.encoder = Encoder(d_model=d_model,
                                ffn_hidden=ffn_hidden,
                                n_head=n_head,
                                drop_prob=drop_prob,
                                n_layers=n_layers)
            self.decoder = Decoder(d_model=d_model,
                                ffn_hidden=ffn_hidden,
                                n_head=n_head,
                                drop_prob=drop_prob,
                                n_layers=n_layers)
            self.l = nn.Linear(d_model, hi_vocab_size)

    def forward(self,
                enc_b,
                dec_b,
                enc_mask,
                dec_mask):
          enc_out = self.encoder(enc_b, enc_mask)
          dec_out = self.decoder(dec_b, dec_mask, enc_out)
          out = self.l(dec_out)
          return out
    

class MachineTranslation(nn.Module):
    def __init__(self,
                 d_model,
                 ffn_hidden,
                 n_head,
                 drop_prob,
                 n_layers,
                 eng_vocab_size,
                 hi_vocab_size,
                 device="cpu"):

        super().__init__()
        self.transformer = Transformer(d_model=d_model,
                                       ffn_hidden=ffn_hidden,
                                       n_head=n_head,
                                       drop_prob=drop_prob,
                                       n_layers=n_layers,
                                       hi_vocab_size=hi_vocab_size)

        self.pos_enc = PositionalEncoding(d_model, MAX_SEQ_LEN)
        self.hi_emb = nn.Embedding(hi_vocab_size, d_model)
        self.eng_emb = nn.Embedding(eng_vocab_size, d_model)
        self.device = device

    def forward(self,
                enc_b,
                dec_b,
                enc_mask,
                dec_mask):

        # self.pos_enc = self.pos_enc.to(self.device)
        # print(f"pos: {self.pos_enc().device}")
        # print(f"pos: {self.hi_emb.device}")
        # print(f"pos: {self.eng_emb.device}")
        enc_b = self.eng_emb(enc_b) + self.pos_enc().to(self.device)
        dec_b = self.hi_emb(dec_b) + self.pos_enc().to(self.device)
        out = self.transformer(enc_b,
                          dec_b,
                          enc_mask,
                          dec_mask)
        return out