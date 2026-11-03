"""Project concatenated raw event fields with one bias, as in UniRank."""

import torch
from torch import nn

from perseus.core.backbone.event import base


class Aggregator(base.Aggregator):
    raw_fields = True

    def __init__(self, dim: int, num_features: int, /, *, embedding_dim: int) -> None:
        super().__init__(dim, num_features)
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        input_dim = num_features * embedding_dim
        self.projection = nn.Linear(input_dim, dim)

    def forward(self, features: torch.Tensor, /) -> torch.Tensor:
        if features.shape[-2:] != (self.num_features, self.embedding_dim):
            raise ValueError("raw event feature dimensions do not match the configured projection")
        return self.projection(features.flatten(start_dim=-2))
