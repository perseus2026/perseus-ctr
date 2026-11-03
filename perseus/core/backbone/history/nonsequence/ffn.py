import typing as t

import torch
from torch import nn

from perseus.core.backbone.history.nonsequence.base import Encoder
from perseus.core.backbone.history.nonsequence.stack import (
    PER_TOKEN_ACTIVATIONS,
    _activation,
    residual_wrap,
)

ACTIVATIONS: dict[str, type[nn.Module]] = {"relu": nn.ReLU, "gelu": nn.GELU}


class PerTokenFeedForward(nn.Module):
    def __init__(self, dim: int, num_tokens: int, expansion: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.first_weight = nn.Parameter(torch.empty(num_tokens, dim, dim * expansion))
        self.first_bias = nn.Parameter(torch.zeros(num_tokens, dim * expansion))
        self.second_weight = nn.Parameter(torch.empty(num_tokens, dim * expansion, dim))
        self.second_bias = nn.Parameter(torch.zeros(num_tokens, dim))
        self.act = ACTIVATIONS[activation]()
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.first_weight)
        nn.init.xavier_uniform_(self.second_weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = torch.einsum("btd,tdk->btk", tokens, self.first_weight) + self.first_bias
        hidden = self.dropout(self.act(hidden))
        return torch.einsum("btk,tkd->btd", hidden, self.second_weight) + self.second_bias


class PerTokenSwiGLU(nn.Module):
    """GELU-gated feed-forward layer with independent weights per token."""

    def __init__(self, dim: int, num_tokens: int, expand: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = expand * dim
        self.w1 = nn.Parameter(torch.empty(num_tokens, dim, hidden_dim))
        self.b1 = nn.Parameter(torch.zeros(num_tokens, hidden_dim))
        self.w2 = nn.Parameter(torch.empty(num_tokens, dim, hidden_dim))
        self.b2 = nn.Parameter(torch.zeros(num_tokens, hidden_dim))
        self.w3 = nn.Parameter(torch.empty(num_tokens, hidden_dim, dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.w3)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        gate = self.dropout(self._activation(tokens))
        up = torch.einsum("btd,tdk->btk", tokens, self.w2) + self.b2
        return torch.einsum("btk,tkd->btd", gate * up, self.w3)

    def _activation(self, tokens: torch.Tensor) -> torch.Tensor:
        values = torch.einsum("btd,tdk->btk", tokens, self.w1) + self.b1
        return nn.functional.gelu(values)


def per_token_ffn(dim: int, num_tokens: int, /, **params: t.Any) -> nn.Module:
    activation = _activation(params.get("activation", "gelu"), PER_TOKEN_ACTIVATIONS)
    expansion = params.get("expansion_factor", 2)
    if type(expansion) is not int or expansion < 1:
        raise ValueError("expansion_factor must be a positive integer")
    ffn: nn.Module = (
        PerTokenSwiGLU(dim, num_tokens, expansion, 0.0)
        if activation == "swiglu"
        else PerTokenFeedForward(dim, num_tokens, expansion, 0.0, activation)
    )
    return residual_wrap(dim, ffn, params)


class FeedForward(Encoder):
    def build_blocks(self, dim: int, num_tokens: int, params: dict[str, t.Any]) -> list[nn.Module]:
        return [per_token_ffn(dim, num_tokens, **{key: value for key, value in params.items() if key != "num_layers"})]
