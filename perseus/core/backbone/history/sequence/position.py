"""Sinusoidal positional signal used by sequence encoders that request it."""

import torch
from torch import nn


class SinPositionSignal(nn.Module):
    """Fixed sinusoidal signal of the valid-rank position, added to the sequence tokens."""

    def __init__(self, dim: int, max_positions: int, /) -> None:
        super().__init__()
        half = dim // 2
        position = torch.arange(1, max_positions + 1, dtype=torch.float32).unsqueeze(-1)
        freqs = 1.0 / (10_000.0 ** (torch.arange(half, dtype=torch.float32) / max(half, 1)))
        signal = torch.cat([torch.sin(position * freqs), torch.cos(position * freqs)], dim=-1)
        if dim % 2:
            signal = torch.cat([signal, torch.zeros(max_positions, 1)], dim=-1)
        self.register_buffer("signal", signal, persistent=False)

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor | None, /) -> torch.Tensor:
        if padding_mask is None:
            return tokens
        valid = ~padding_mask
        ranks = valid.long().flip(-1).cumsum(-1).flip(-1)
        steps = (ranks - 1).clamp_(min=0).long().clamp_(max=self.signal.size(0) - 1)
        signal = self.signal.to(tokens.dtype)
        return tokens + signal[steps] * valid.unsqueeze(-1).to(tokens.dtype)
