import typing as t
from abc import ABC, abstractmethod

import torch
from torch import nn


class State(t.NamedTuple):
    x: torch.Tensor


class SequenceState(t.NamedTuple):
    x: torch.Tensor
    padding_mask: torch.Tensor


class Aggregator(nn.Module, ABC):
    """A history encoder driven through prepare/step/finalize.

    Layered history calls these methods directly, so hooks on this module do not
    fire in that schedule. The once and all-as-sequence schedules call forward
    normally. Hooks on the owning history aggregator or backbone run in every
    schedule.
    """

    # Only components that explicitly declare no effect may be omitted from a schedule.
    is_noop = False
    step_is_noop = False
    requires_attention_mask = True
    num_layers: int

    @property
    def normalization_layers(self) -> t.Sequence[nn.Module]:
        return ()

    @abstractmethod
    def __init__(self, dim: int, max_events_per_sequence: int, /) -> None:
        super().__init__()

    @abstractmethod
    def prepare(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor | None,
        /,
    ) -> State: ...

    @abstractmethod
    def step(self, state: State, index: int, /) -> State: ...

    @abstractmethod
    def finalize(self, state: State, /) -> torch.Tensor: ...

    def forward(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor | None,
        /,
        *,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del padding_mask
        state = self.prepare(embeddings, positions, timestamps, mask)
        for index in range(self.num_layers):
            state = self.step(state, index)
        return self.finalize(state)
