import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

    def forward(self) -> torch.tensor:
        pos = torch.arange(0, self.max_seq_len)
        denominator = torch.arange(0, self.d_model, 2)
        denominator = torch.pow(10_000, denominator/self.d_model)

        pos = pos.reshape(-1, 1)
        denominator = denominator.reshape(1, -1)
        even_pos = torch.sin(pos / denominator)
        odd_pos = torch.cos(pos / denominator)

        PE = torch.stack([even_pos, odd_pos], dim=2)
        PE = torch.flatten(PE, start_dim=1, end_dim=2)
        return PE


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 input_dim: int,
                 d_model: int,
                 n_head: int):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.n_head = n_head
        self.h_dim = d_model // n_head
        self.qkv_layer = nn.Linear(input_dim, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self,
                x: torch.tensor,
                mask: torch.tensor = None):
        B, sen_len, input_dim = x.size()
        qkv = self.qkv_layer(x)  # B, sen_len, 3 * d_model
        qkv = qkv.reshape(B, sen_len, self.n_head, self.h_dim * 3)
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        d_k = q.size()[-1]
        att = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(att)
            att += mask
        att = F.softmax(att, dim=-1)
        new_emb = att @ v
        new_emb = new_emb.reshape(B, sen_len, self.n_head * self.h_dim)
        new_emb = self.linear_layer(new_emb)
        return att, new_emb


class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, x):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = x.mean(dim=dims, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (x - mean) / std
        out = self.gamma * y + self.beta
        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self,
                 d_model: int,
                 ffn_hidden: int,
                 drop_prob: float):
        super().__init__()
        self.l = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(ffn_hidden, d_model),
            nn.ReLU(),
            nn.Dropout(drop_prob),
        )

    def forward(self,
                x):
        out = self.l(x)
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_head: int):

        super().__init__()
        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.n_head = n_head

    def forward(self,
                enc_out: torch.tensor,
                dec_out: torch.tensor):

        B, max_sen_len, d_model = enc_out.size()

        q = self.q_layer(dec_out)
        k = self.k_layer(enc_out)
        v = self.v_layer(enc_out)

        q = q.reshape(B, max_sen_len, self.n_head, d_model // self.n_head)
        k = k.reshape(B, max_sen_len, self.n_head, d_model // self.n_head)
        v = v.reshape(B, max_sen_len, self.n_head, d_model // self.n_head)

        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        att = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_model))
        att = F.softmax(att, dim=-1)

        new_emb = att @ v
        new_emb = new_emb.reshape(B, max_sen_len, self.n_head * (d_model // self.n_head))

        return att, new_emb


