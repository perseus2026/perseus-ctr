"""Pre-norm self-attention sequence encoder."""

import typing as t

import torch
from torch import nn

from perseus.core.backbone.history.nonsequence.stack import (
    GroupTransform,
    PaddedPreNormResidual,
)
from perseus.core.backbone.history.sequence import base
from perseus.core.backbone.history.sequence.ffn import shared_ffn
from perseus.core.backbone.history.sequence.position import SinPositionSignal


class SequenceAttention(nn.Module):
    """Self-attention over the sequence group with q/k/v/output projections."""

    def __init__(self, dim: int, num_heads: int, *, mask_padding: bool = True) -> None:
        super().__init__()
        if type(num_heads) is not int or num_heads < 1:
            raise ValueError("num_heads must be a positive integer")
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mask_padding = mask_padding
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        batch, length, dim = tokens.shape
        attn_mask = None
        if padding_mask is not None and padding_mask.ndim == 3:
            attn_mask = (~padding_mask).unsqueeze(1)
        elif padding_mask is not None and padding_mask.shape[1] == length and self.mask_padding:
            valid = ~padding_mask
            has_tokens = valid.any(dim=-1)
            attn_mask = valid.clone()
            attn_mask[~has_tokens, 0] = True
        query, key, value = [
            projection(tokens).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
            for projection in (self.query, self.key, self.value)
        ]
        if attn_mask is not None and attn_mask.ndim == 2:
            attn_mask = attn_mask[:, None, None, :]
        attended = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask)
        attended = attended.transpose(1, 2).reshape(batch, length, dim)
        return self.output(attended)


def self_attention(dim: int, _num_tokens: int = 0, /, **params: t.Any) -> nn.Module:
    num_heads = params.get("num_heads", 2)
    return PaddedPreNormResidual(dim, SequenceAttention(dim, num_heads, mask_padding=params.get("mask_padding", True)))




class Aggregator(base.Aggregator):
    """Stack of pre-norm self-attention layers, optionally followed by FFNs."""

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
        num_heads: int = 2,
        ffn: dict[str, t.Any] | None = None,
        positional: bool = False,
        shared_layers: bool = False,
        zero_padding: bool = False,
        mask_padding: bool = True,
    ) -> None:
        super().__init__(dim, max_events_per_sequence)
        if type(positional) is not bool:
            raise ValueError("positional must be boolean")
        if type(num_layers) is not int or num_layers < 1:
            raise ValueError("num_layers must be a positive integer")
        self.num_layers = num_layers
        self.mask_padding = mask_padding
        self.position = SinPositionSignal(dim, max_events_per_sequence) if positional else None
        layer_count = 1 if shared_layers else self.num_layers
        self.layers = nn.ModuleList()
        for _ in range(layer_count):
            blocks = [self_attention(dim, num_heads=num_heads, mask_padding=mask_padding)]
            if ffn is not None:
                blocks.append(shared_ffn(dim, **dict(ffn)))
            self.layers.append(GroupTransform(["sequence"], nn.ModuleList(blocks), zero_padding=zero_padding))

    def prepare(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        _timestamps: torch.Tensor,
        _mask: torch.Tensor,
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
        self, embeddings: torch.Tensor, positions: torch.Tensor, timestamps: torch.Tensor, mask: torch.Tensor, /,
        *, padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if padding_mask is None:
            return super().forward(embeddings, positions, timestamps, mask)
        if not self.mask_padding:
            raise ValueError("all_as_sequence self_attention requires mask_padding=true")
        tokens = embeddings
        if self.position is not None:
            tokens = self.position(tokens, padding_mask)
        for index in range(self.num_layers):
            layer = self.layers[min(index, len(self.layers) - 1)]
            if layer.zero_padding:
                tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0)
            for block in layer.blocks:
                tokens = block(tokens, mask)
                if layer.zero_padding:
                    tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0)
        return tokens
