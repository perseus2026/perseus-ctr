"""Query generators used by non-sequential encoders."""

import typing as t

import torch

from perseus.core.backbone.history.pooling import masked_mean
from perseus.core.backbone.history.tokenizer import AutoSplitTokenizer

if t.TYPE_CHECKING:
    from perseus.core.backbone.context.base import Layout


class PersistentQueries(AutoSplitTokenizer):
    """Generate persistent tokens from context and the pooled sequence.

    Inherit the tokenizer directly to preserve checkpoint parameter paths.
    """

    def __init__(
        self, dim: int, num_context_tokens: int, *, params: dict[str, t.Any],
        context_layout: "Layout | None" = None, readout: bool = True,
    ) -> None:
        if "query_rank" in params:
            raise ValueError(
                "queries.params.query_rank has been replaced by queries.params.rank; migrate the config",
            )
        num_tokens = params["num_tokens"]
        rank = params.get("rank")
        if type(num_tokens) is not int or num_tokens < 1:
            raise ValueError("query num_tokens must be a positive integer")
        if rank is not None and (type(rank) is not int or rank < 1):
            raise ValueError("query rank must be a positive integer")
        raw_fields = context_layout is not None and context_layout.raw_fields
        width = (context_layout.num_fields * context_layout.embedding_dim if raw_fields
                 else num_context_tokens * dim) + (dim if readout else 0) + dim
        super().__init__(width, dim, num_tokens, rank)
        self.num_tokens = num_tokens
        self.raw_fields = raw_fields
        self.readout = readout

    def forward(
        self, sequence: torch.Tensor, context: torch.Tensor, padding_mask: torch.Tensor,
        raw_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.raw_fields:
            inputs = [raw_context.flatten(start_dim=1)]
            if self.readout:
                inputs.append(context[:, -1])
        else:
            inputs = [context.flatten(start_dim=1)]
        inputs.append(masked_mean(sequence, padding_mask))
        return super().forward(torch.cat(inputs, dim=-1))


registry = {"persistent": PersistentQueries}
