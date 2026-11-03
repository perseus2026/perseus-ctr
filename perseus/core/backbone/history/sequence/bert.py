import typing as t

import torch
from torch import nn

from perseus.core.backbone.history.sequence import base


class State(t.NamedTuple):
    x: torch.Tensor
    mask: torch.Tensor


class Aggregator(base.Aggregator):
    def __init__(
        self,
        dim: int,
        max_events_per_sequence: int,
        /,
        *,
        num_heads: int = 1,
        dropout: float = 0.1,
        num_layers: int = 1,
    ) -> None:
        super().__init__(dim, max_events_per_sequence)
        self.num_layers = num_layers

        self.num_heads = num_heads
        self.positions_embeddings = nn.Embedding(max_events_per_sequence + 1, dim, padding_idx=0)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            norm=nn.LayerNorm(dim),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

    def prepare(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        _timestamps: torch.Tensor,
        mask: torch.Tensor,
        /,
    ) -> State:
        batch_size, seq_len, _ = embeddings.shape
        mask = (
            mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1).reshape(batch_size * self.num_heads, seq_len, seq_len)
        )
        return State(embeddings + self.positions_embeddings(positions), mask)

    def step(self, state: State, index: int, /) -> State:
        return state._replace(x=self.encoder.layers[index](state.x, state.mask))

    def finalize(self, state: State, /) -> torch.Tensor:
        return self.encoder.norm(state.x)
