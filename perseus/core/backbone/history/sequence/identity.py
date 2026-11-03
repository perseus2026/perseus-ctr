import torch

from perseus.core.backbone.history.sequence import base
from perseus.core.backbone.history.sequence.position import SinPositionSignal


class Aggregator(base.Aggregator):
    requires_attention_mask = False
    num_layers = 0
    step_is_noop = True

    def __init__(
        self, dim: int, max_events_per_sequence: int, /, *, positional: bool = False, num_layers: int | None = None,
    ) -> None:
        del num_layers
        super().__init__(dim, max_events_per_sequence)
        if type(positional) is not bool:
            raise ValueError("positional must be boolean")
        self.is_noop = not positional
        self.position = SinPositionSignal(dim, max_events_per_sequence) if positional else None

    def prepare(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        _timestamps: torch.Tensor,
        _mask: torch.Tensor | None,
        /,
    ) -> base.State:
        if self.position is not None:
            embeddings = self.position(embeddings, positions == 0)
        return base.State(embeddings)

    def step(self, state: base.State, _index: int, /) -> base.State:
        return state

    def finalize(self, state: base.State, /) -> torch.Tensor:
        return state.x

    def forward(
        self, embeddings: torch.Tensor, positions: torch.Tensor, timestamps: torch.Tensor, mask: torch.Tensor | None, /,
        *, padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if padding_mask is None:
            return super().forward(embeddings, positions, timestamps, mask)
        return self.position(embeddings, padding_mask) if self.position is not None else embeddings
