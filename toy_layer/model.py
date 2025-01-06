import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
    

class CausalSelfAttention(nn.Module):
    def __init__(self, config, mesh=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.mesh = mesh
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        if self.mesh:
            tp_size = self.mesh.mesh.size(0)
            assert self.n_head % tp_size == 0
        else:
            tp_size = 1
        q, k, v = self.c_attn(x).split(self.n_embd // tp_size, dim=2)
        k = k.view(B, T, self.n_head // tp_size, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head // tp_size, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head // tp_size, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C // tp_size) # re-assemble all head outputs side by side
        y = self.resid_dropout(self.c_proj(y))
        return y
    

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, config.n_embd * 4, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd * 4, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config, mesh=None):
        super().__init__()
        self.ln1 = LayerNorm(config.n_embd, bias=config.bias)
        self.ln2 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config, mesh)
        self.mlp = MLP(config)

    def forward(self, x, y=None):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))

        if y is not None:
            assert y.shape == x.shape
            loss = F.mse_loss(x, y)
        else:
            loss = None
        return x, loss