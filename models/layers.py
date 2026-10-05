import math
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F

from models.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 dropout: float = 0.0):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))
        
        self.dropout = dropout
        self.mask = nn.Buffer(torch.ones_like(self.weight))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight * self.mask if self.training else self.weight
        return F.linear(input, weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)
    
    def reset_mask(self):
        if not self.training or self.dropout == 0.0:
            self.mask.fill_(1.0)
        else:
            self.mask.bernoulli_(1 - self.dropout).div_(1 - self.dropout)

class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    # Promote bf16/fp16 to fp32 but NEVER demote fp64 (same idiom as
    # QuotientIntegrator._hi / gauge_layers.to_c), so an fp64 analysis stays fp64
    # through the trunk.  Bitwise no-op for fp32/bf16.
    if hidden_states.dtype not in (torch.float32, torch.float64):
        hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)
