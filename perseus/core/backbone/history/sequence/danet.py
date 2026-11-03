import typing as t

import torch
from torch import nn

from perseus.core.backbone.history.sequence import base


class State(t.NamedTuple):
    x: torch.Tensor
    cos: torch.Tensor
    scaler: torch.Tensor


class Aggregator(base.Aggregator):
    def __init__(
        self,
        dim: int,
        max_events_per_sequence: int,
        /,
        *,
        dropout: float = 0.1,
        num_layers: int = 1,
    ) -> None:
        super().__init__(dim, max_events_per_sequence)
        self.num_layers = num_layers

        inv_freq = 1.0 / (10_000.0 ** (torch.arange(0, dim, dtype=torch.float32) / dim))
        positions = torch.arange(max_events_per_sequence + 1, dtype=torch.float32)
        self.register_buffer("cos", torch.outer(positions, inv_freq).cos())

        self.layers = nn.ModuleList(EncoderLayer(dim, dropout) for _ in range(num_layers))

    def prepare(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        _timestamps: torch.Tensor,
        mask: torch.Tensor,
        /,
    ) -> State:
        mask = mask.unsqueeze(dim=1)
        cos = self.cos[positions]
        keep = (~mask[:, 0, -1, :]).to(embeddings.dtype)
        scaler = (keep / keep.sum(dim=-1, keepdim=True).pow(1.0 / 3)).unsqueeze(dim=-1)
        return State(embeddings, cos, scaler)

    def step(self, state: State, index: int, /) -> State:
        return state._replace(x=self.layers[index](state.x, state.cos, state.scaler))

    def finalize(self, state: State, /) -> torch.Tensor:
        return state.x


class EncoderLayer(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()

        self.attention = DenseAttention(dim, dropout)
        self.ffn = SwiGLUFFN(dim, dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, scaler: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self._max_norm(x) * scaler
        x = self.attention(x, cos)
        x = self.ffn(x)
        return residual + self._max_norm(x)

    @staticmethod
    def _max_norm(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        return x / (x.abs().max(dim=-1, keepdim=True).values + eps)


class DenseAttention(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        q = self.q_proj(x) * cos
        k = x * cos
        v = x
        kv_state = k.transpose(-2, -1) @ v
        return q @ kv_state


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()

        hidden_dim = int(dim * 8 / 3)
        hidden_dim = (hidden_dim + 127) // 128 * 128

        self.gate_up_proj = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.out_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.expansion_dropout = nn.Dropout(dropout)
        self.contraction_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x = self.expansion_dropout(nn.functional.silu(gate) * up)
        return self.contraction_dropout(self.out_proj(x))
