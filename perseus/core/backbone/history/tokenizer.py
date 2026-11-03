"""Whole-input tokenization shared by history query and interaction branches."""

import torch
from torch import nn


class AutoSplitTokenizer(nn.Module):
    """Per-token MLP over the whole flattened input."""

    def __init__(self, input_dim: int, token_dim: int, num_tokens: int, rank: int | None = None) -> None:
        super().__init__()

        def mlp() -> nn.Sequential:
            if rank is None:
                return nn.Sequential(nn.Linear(input_dim, token_dim), nn.LayerNorm(token_dim), nn.SiLU())
            return nn.Sequential(
                nn.Linear(input_dim, rank),
                nn.Linear(rank, token_dim),
                nn.LayerNorm(token_dim),
                nn.SiLU(),
            )

        self.token_mlps = nn.ModuleList(mlp() for _ in range(num_tokens))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack([mlp(features) for mlp in self.token_mlps], dim=1)
