"""Shared pointwise feed-forward sequence encoder."""

import typing as t

import torch
from torch import nn

from perseus.core.backbone.history.nonsequence.ffn import ACTIVATIONS
from perseus.core.backbone.history.nonsequence.stack import (
    SHARED_ACTIVATIONS,
    GroupTransform,
    _activation,
    residual_wrap,
)
from perseus.core.backbone.history.sequence import base
from perseus.core.backbone.history.sequence.position import SinPositionSignal


class SwiGLU(nn.Module):
    """Pointwise FFN with GELU gating."""

    def __init__(self, dim: int, /, *, expand: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, expand * dim)
        self.w2 = nn.Linear(dim, expand * dim)
        self.w3 = nn.Linear(expand * dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.w3(self.dropout(self.act(self.w1(tokens)) * self.w2(tokens)))


class SharedFeedForward(nn.Module):
    """Token-wise FFN with weights shared across positions."""

    def __init__(self, dim: int, expansion: int, activation: str, dropout: float, /) -> None:
        super().__init__()
        if activation == "swiglu":
            self.net: nn.Module = SwiGLU(dim, expand=expansion, dropout=dropout)
            return
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown shared_ffn activation `{activation}`; available: {sorted(ACTIVATIONS)}")
        self.net = nn.Sequential(
            nn.Linear(dim, expansion * dim),
            ACTIVATIONS[activation](),
            nn.Dropout(dropout),
            nn.Linear(expansion * dim, dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


def shared_ffn(dim: int, _num_tokens: int = 0, /, **params: t.Any) -> nn.Module:
    activation = _activation(params.get("activation", "relu"), SHARED_ACTIVATIONS)
    expansion = params.get("expansion_factor", 4)
    if type(expansion) is not int or expansion < 1:
        raise ValueError("expansion_factor must be a positive integer")
    return residual_wrap(dim, SharedFeedForward(dim, expansion, activation, 0.0), params)




class Aggregator(base.Aggregator):
    """Stack of token-wise feed-forward layers over a sequence."""

    requires_attention_mask = False

    @property
    def normalization_layers(self) -> nn.ModuleList:
        return self.layers

    def __init__(
        self,
        dim: int,
        max_events_per_sequence: int,
        /,
        *,
        num_layers: int = 1,
        positional: bool = False,
        shared_layers: bool = False,
        zero_padding: bool = False,
        **params: t.Any,
    ) -> None:
        super().__init__(dim, max_events_per_sequence)
        if type(positional) is not bool:
            raise ValueError("positional must be boolean")
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        self.num_layers = num_layers
        self.position = SinPositionSignal(dim, max_events_per_sequence) if positional else None
        layer_count = 1 if shared_layers else self.num_layers
        self.layers = nn.ModuleList([
            GroupTransform(["sequence"], nn.ModuleList([shared_ffn(dim, **params)]), zero_padding=zero_padding)
            for _ in range(layer_count)
        ])

    def prepare(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        _timestamps: torch.Tensor,
        _mask: torch.Tensor | None,
        /,
    ) -> base.SequenceState:
        padding = positions == 0
        if self.position is not None:
            embeddings = self.position(embeddings, padding)
        return base.SequenceState(embeddings, padding)

    def step(self, state: base.SequenceState, index: int, /) -> base.SequenceState:
        layer = self.layers[min(index, len(self.layers) - 1)]
        output = layer({"sequence": state.x}, state.padding_mask)["sequence"]
        return state._replace(x=output)

    def finalize(self, state: base.SequenceState, /) -> torch.Tensor:
        return state.x

    def forward(
        self, embeddings: torch.Tensor, positions: torch.Tensor, timestamps: torch.Tensor, mask: torch.Tensor | None, /,
        *, padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if padding_mask is None:
            return super().forward(embeddings, positions, timestamps, mask)
        if self.position is not None:
            embeddings = self.position(embeddings, padding_mask)
        state = base.SequenceState(embeddings, padding_mask)
        for index in range(self.num_layers):
            state = self.step(state, index)
        return self.finalize(state)
