"""Shared interface for interactions between history and nonsequence tokens."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


class Interaction(nn.Module, ABC):
    is_noop = False
    step_is_noop = False
    requires_sequence_step = False
    requires_fixed_width = True

    def __init__(
        self, carrier_group: str | None, carrier_tokens: int | None, /, *,
        nonsequence_groups: list[str], nonsequence_tokens: int, query_tokens: int | None = None,
    ) -> None:
        super().__init__()
        self.carrier_group = carrier_group
        self.token_counts = {carrier_group: carrier_tokens} if carrier_group is not None else {}
        self.nonsequence_groups = nonsequence_groups
        self.nonsequence_tokens = nonsequence_tokens
        self.query_tokens = query_tokens

    def update_sequence(
        self, sequence_state: Any, _groups: dict[str, torch.Tensor], _padding: torch.Tensor,
        _index: int = 0, /,
    ) -> Any:
        return sequence_state

    @abstractmethod
    def forward(
        self, groups: dict[str, torch.Tensor], padding_mask: torch.Tensor, layer_index: int = 0, /,
    ) -> dict[str, torch.Tensor]: ...


class Identity(Interaction):
    is_noop = True
    step_is_noop = True
    requires_fixed_width = False

    def __init__(
        self, _dim: int, num_context_tokens: int, _params: dict, _num_layers: int = 1, /, **kwargs: Any,
    ) -> None:
        super().__init__(
            None, None, nonsequence_groups=["context"],
            nonsequence_tokens=num_context_tokens + int(kwargs.get("readout", True)),
        )

    def forward(
        self, groups: dict[str, torch.Tensor], _padding: torch.Tensor, _index: int = 0, /,
    ) -> dict[str, torch.Tensor]:
        return groups
