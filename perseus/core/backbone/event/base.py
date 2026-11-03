from abc import ABC, abstractmethod

import torch
from torch import nn


class Aggregator(nn.Module, ABC):
    # Raw-field aggregators receive feature embeddings before per-field projection.
    raw_fields = False

    @abstractmethod
    def __init__(self, dim: int, num_features: int, /) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, features: torch.Tensor, /) -> torch.Tensor: ...
