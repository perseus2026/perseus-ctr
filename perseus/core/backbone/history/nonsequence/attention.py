"""Token-level attention blocks for the non-sequence stack."""

import typing as t

import torch
from torch import nn

from perseus.core.backbone.history.nonsequence.stack import (
    residual_wrap,
)


class LowRankAttention(nn.Module):
    """Low-rank QKV self-attention over the flattened token set (HiFormer)."""

    def __init__(self, dim: int, num_tokens: int, num_heads: int, rank: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_tokens = num_tokens
        self.dim = dim
        input_dim = num_tokens * dim
        self.query = nn.Sequential(nn.Linear(input_dim, rank, bias=False), nn.Linear(rank, input_dim, bias=False))
        self.key = nn.Sequential(nn.Linear(input_dim, rank, bias=False), nn.Linear(rank, input_dim, bias=False))
        self.value = nn.Sequential(nn.Linear(input_dim, rank, bias=False), nn.Linear(rank, input_dim, bias=False))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        flattened = tokens.flatten(start_dim=1)
        query, key, value = [
            projection(flattened).view(-1, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
            for projection in (self.query, self.key, self.value)
        ]
        output = nn.functional.scaled_dot_product_attention(query, key, value)
        return output.transpose(1, 2).contiguous().view(-1, self.num_tokens, self.dim)




def low_rank_attention(dim: int, num_tokens: int, /, **params: t.Any) -> nn.Module:
    num_heads = params.get("num_heads", 2)
    qkv_rank = params.get("qkv_rank", 256)
    if type(num_heads) is not int or num_heads < 1:
        raise ValueError("num_heads must be a positive integer")
    if type(qkv_rank) is not int or qkv_rank < 1:
        raise ValueError("qkv_rank must be a positive integer")
    return residual_wrap(dim, LowRankAttention(dim, num_tokens, num_heads, qkv_rank), params)
