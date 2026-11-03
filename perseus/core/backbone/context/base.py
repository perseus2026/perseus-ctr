import typing as t
from abc import ABC, abstractmethod

import torch
from torch import nn


class Layout(t.NamedTuple):
    """Token layout of the context group produced by the context aggregator.

    num_fields/embedding_dim describe the raw field embeddings, num_tokens the
    emitted tokens, field_aligned marks that token i corresponds to field i and
    raw_fields marks that the aggregator consumes the unprojected embeddings.
    """

    num_fields: int
    embedding_dim: int
    num_tokens: int
    field_aligned: bool
    raw_fields: bool = False


class Aggregator(nn.Module, ABC):
    # Raw-field aggregators receive feature embeddings before per-field projection.
    raw_fields = False

    @abstractmethod
    def __init__(self, dim: int, num_features: int, /) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, features: torch.Tensor, /) -> torch.Tensor: ...
