import typing as t

import torch
from torch import nn

from perseus.core.backbone.history.nonsequence.base import Encoder
from perseus.core.backbone.history.nonsequence.ffn import PerTokenSwiGLU, per_token_ffn
from perseus.core.backbone.history.nonsequence.stack import (
    _SCOPE,
    TOKEN_MIXING_KERNELS,
    residual_wrap,
    width_after,
)


class MultiHeadTokenMixing(nn.Module):
    """Parameterless permutation of channel chunks between tokens."""

    def __init__(self, dim: int, num_tokens: int) -> None:
        super().__init__()
        self.input_dim = dim
        self.num_tokens = num_tokens
        self.padded_dim = -(-dim // num_tokens) * num_tokens
        self.head_dim = self.padded_dim // num_tokens
        valid = torch.arange(self.padded_dim) < dim
        mask = valid.expand(num_tokens, -1).reshape(num_tokens, num_tokens, self.head_dim).transpose(0, 1)
        self.register_buffer("valid_index", mask.reshape(-1).nonzero(as_tuple=False).flatten(), persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.padded_dim != self.input_dim:
            tokens = nn.functional.pad(tokens, (0, self.padded_dim - self.input_dim))
        heads = torch.tensor_split(tokens, self.num_tokens, dim=-1)
        mixed = torch.stack(heads, dim=1).flatten(start_dim=2)
        batch_size = tokens.size(0)
        return (
            mixed.flatten(start_dim=1)
            .index_select(1, self.valid_index)
            .reshape(batch_size, self.num_tokens, self.input_dim)
        )


class ReversibleTokenMixing(nn.Module):
    """Learned-reversible token mixing: mix, per-token FFN in the mixed basis, revert, residual."""

    def __init__(self, dim: int, num_tokens: int, expansion: int, dropout: float, /) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.block_dim = -(-dim // num_tokens) * num_tokens
        self.output_dim = self.block_dim
        self.mixer_norm = nn.LayerNorm(self.block_dim)
        self.mixer = MultiHeadTokenMixing(self.block_dim, num_tokens)
        self.revert = MultiHeadTokenMixing(self.block_dim, num_tokens)
        self.pffn = PerTokenSwiGLU(self.block_dim, num_tokens, expand=expansion, dropout=dropout)

    def forward(self, tokens: torch.Tensor, _padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.size(-1) < self.block_dim:
            tokens = nn.functional.pad(tokens, (0, self.block_dim - tokens.size(-1)))
        mixed = self.pffn(self.mixer(self.mixer_norm(tokens)))
        return self.revert(mixed) + tokens


class LowRankTokenMixing(nn.Module):
    """Head-wise low-rank mixing over flattened token channels (HeMix HeteroMixing)."""

    def __init__(self, dim: int, num_tokens: int, num_heads: int, rank: int) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.mix_dim = num_tokens * self.head_dim
        self.left_weight = nn.Parameter(torch.empty(num_heads, self.mix_dim, rank))
        self.left_bias = nn.Parameter(torch.zeros(num_heads, rank))
        self.right_weight = nn.Parameter(torch.empty(num_heads, rank, self.mix_dim))
        self.right_bias = nn.Parameter(torch.zeros(num_heads, self.mix_dim))
        nn.init.xavier_uniform_(self.left_weight)
        nn.init.xavier_uniform_(self.right_weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, length, dim = tokens.shape
        mixed = tokens.view(batch, length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        mixed = mixed.reshape(batch, self.num_heads, self.mix_dim)
        hidden = torch.einsum("bhf,hfr->bhr", mixed, self.left_weight) + self.left_bias.unsqueeze(0)
        output = torch.einsum("bhr,hrf->bhf", hidden.relu(), self.right_weight) + self.right_bias.unsqueeze(0)
        return (
            output.view(batch, self.num_heads, length, self.head_dim)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(batch, length, dim)
        )


def token_mixing(dim: int, num_tokens: int, /, **params: t.Any) -> nn.Module:
    kernel = params["kernel"]
    if kernel not in TOKEN_MIXING_KERNELS:
        raise ValueError(f"unknown {_SCOPE} token_mixing kernel `{kernel}`; available: {sorted(TOKEN_MIXING_KERNELS)}")
    if kernel == "low_rank":
        if "reversible" in params:
            raise ValueError(f"{_SCOPE} token_mixing kernel `low_rank` is not reversible")
        num_mix_heads = params.get("num_mix_heads", 2)
        low_rank_dim = params.get("low_rank_dim", 16)
        if type(num_mix_heads) is not int or num_mix_heads < 1:
            raise ValueError("num_mix_heads must be a positive integer")
        if type(low_rank_dim) is not int or low_rank_dim < 1:
            raise ValueError("low_rank_dim must be a positive integer")
        return residual_wrap(dim, LowRankTokenMixing(dim, num_tokens, num_mix_heads, low_rank_dim), params)
    reversible = params.get("reversible", False)
    if not isinstance(reversible, bool):
        raise TypeError(f"{_SCOPE} token_mixing `reversible` must be a bool, got {reversible!r}")
    expansion = params.get("expansion_factor", 2)
    if type(expansion) is not int or expansion < 1:
        raise ValueError("expansion_factor must be a positive integer")
    if reversible:
        if params.get("norm", "layer_norm") != "layer_norm" or params.get("norm_group") is not None:
            raise ValueError("reversible token_mixing requires its own layer_norm and does not support norm_group")
        return ReversibleTokenMixing(dim, num_tokens, expansion, 0.0)
    return residual_wrap(dim, MultiHeadTokenMixing(dim, num_tokens), params)


class TokenMixer(Encoder):
    def build_blocks(self, dim: int, num_tokens: int, params: dict[str, t.Any]) -> list[nn.Module]:
        mixing_params = {key: value for key, value in params.items() if key not in {"num_layers", "ffn"}}
        mixing = token_mixing(dim, num_tokens, **mixing_params)
        output_dim = width_after(dim, mixing)
        return [mixing, per_token_ffn(output_dim, num_tokens, **dict(params.get("ffn") or {}))]
