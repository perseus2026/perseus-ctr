import typing as t

import torch
from torch import nn

from perseus.core.backbone.context.base import Layout
from perseus.core.backbone.history.interaction.base import Interaction
from perseus.core.backbone.history.interaction.tokenize import (
    FieldTaskTokenize,
    GlobalGroupTokenize,
    MixedQueriesTokenizer,
)


class TargetAttention(nn.Module):
    """Pool the history sequence with the candidate as the query."""

    def __init__(self, input_dim: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, dim, bias=False)
        self.key = nn.Linear(input_dim, dim, bias=False)
        self.value = nn.Linear(input_dim, dim, bias=False)
        self.output = nn.Linear(dim, input_dim, bias=False)
        self.dropout = dropout

    def forward(
        self, candidate: torch.Tensor, history: torch.Tensor, padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if history.size(1) == 0:
            return torch.zeros_like(candidate)
        safe_mask = ~padding_mask
        has_history = safe_mask.any(dim=-1)
        safe_mask[~has_history, 0] = True
        query = self.query(candidate)[:, None, None, :]
        key = self.key(history).unsqueeze(1)
        value = self.value(history).unsqueeze(1)
        pooled = nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=safe_mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
        )
        return torch.where(
            has_history[:, None],
            self.output(pooled[:, 0, 0]),
            torch.zeros_like(candidate),
        )


class TargetAttentionPooling(nn.Module):
    """Query history with the readout, or the sum of explicitly selected context tokens."""

    def __init__(
        self, dim: int, dropout: float, /, *, query_context_indices: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.pooling = TargetAttention(dim, dim, dropout)
        self.query_context_indices = query_context_indices

    def forward(
        self, history: torch.Tensor, group: torch.Tensor, padding_mask: torch.Tensor, /,
    ) -> torch.Tensor:
        query = (
            group[:, -1] if self.query_context_indices is None
            else group[:, self.query_context_indices, :].sum(dim=1)
        )
        return self.pooling(query, history, padding_mask)


class QueryPooling(nn.Module):
    """Conditioned and fixed query sets cross-attend their halves of the history."""

    def __init__(
        self, dim: int, num_context_tokens: int, /, *,
        recent_length: int, num_dynamic_queries: int, num_learned_queries: int, attention_heads: int,
        readout: bool = True,
        raw_context_dim: int | None = None,
        projection: str = "per_query_q",
    ) -> None:
        super().__init__()
        self.recent_length = recent_length
        input_dim = (num_context_tokens + (1 if readout else 0)) * dim if raw_context_dim is None else raw_context_dim
        self.tokenizer = MixedQueriesTokenizer(
            input_dim, dim, num_dynamic_queries, num_learned_queries, attention_heads, projection,
        )

    def forward(
        self, history: torch.Tensor, group: torch.Tensor, padding_mask: torch.Tensor, /,
    ) -> torch.Tensor:
        valid = (~padding_mask).unsqueeze(-1).to(history.dtype)
        recent = history[:, -self.recent_length :] * valid[:, -self.recent_length :]
        earlier = history[:, : -self.recent_length] * valid[:, : -self.recent_length :]
        return self.tokenizer(group.flatten(start_dim=1), recent, earlier)


class GlobalGroupTokenizer(GlobalGroupTokenize):
    def __init__(self, dim: int, num_fields: int, num_group_tokens: int = 4) -> None:
        super().__init__(dim, num_fields * dim, num_group_tokens)
        self.num_tokens = num_group_tokens + 1


class FieldTaskTokenizer(FieldTaskTokenize):
    def __init__(self, dim: int, num_fields: int, num_tasks: int) -> None:
        super().__init__(dim, num_fields, num_tasks)
        self.num_tokens = num_fields + num_tasks


tokenizer_registry = {"global_group": GlobalGroupTokenizer, "field_task": FieldTaskTokenizer}


class TargetAttentionInteraction(Interaction):
    def __init__(self, dim: int, num_context_tokens: int, params: dict[str, t.Any], /,
                 num_layers: int = 1, *, context_fields: list[str] | None = None,
                 context_layout: Layout | None = None, readout: bool = True,
                 schedule: str = "layered", independent_layers: bool = False) -> None:
        del schedule, independent_layers
        if WINDOW_DEFAULTS.keys() & params.keys():
            raise ValueError("window parameters require window_attention interaction")
        fields = params.get("query_context_fields")
        indices = params.get("query_context_indices")
        if fields is not None and indices is not None:
            raise ValueError("specify query context fields or indices, not both")
        for name, values, item_type in (("query_context_fields", fields, str),
                                        ("query_context_indices", indices, int)):
            if name in params and (not isinstance(values, list) or not values
                                   or any(type(value) is not item_type for value in values)
                                   or len(set(values)) != len(values)):
                raise ValueError(f"{name} must be a non-empty list of distinct {item_type.__name__} values")
        indices = params.get("query_context_indices")
        if "query_context_fields" in params:
            if context_fields is None:
                raise ValueError("query context fields require an ordered context field schema")
            if context_layout is not None and not context_layout.field_aligned:
                raise ValueError("query context fields require field-aligned context tokens")
            unknown_fields = set(params["query_context_fields"]) - set(context_fields)
            if unknown_fields:
                raise ValueError(f"unknown context fields: {sorted(unknown_fields)}")
            indices = [context_fields.index(name) for name in params["query_context_fields"]]
        indices = tuple(indices) if indices is not None else None
        if indices is not None and any(index < 0 or index >= num_context_tokens for index in indices):
            raise ValueError("query context index is outside the context tokens")
        if indices is None and not readout:
            raise ValueError("target attention requires a readout token or selected context query")
        pools = [TargetAttentionPooling(dim, 0.0, query_context_indices=indices) for _ in range(num_layers)]
        tokenizers = []
        carrier, count = "pooled", 1
        tokenize = params.get("tokenize")
        if tokenize is not None:
            tokenizer = tokenizer_registry[tokenize["type"]]
            fields = num_context_tokens + int(readout) + 1
            tokenizers = [tokenizer(dim, fields, **tokenize.get("params", {})) for _ in range(num_layers)]
            count, carrier = tokenizers[0].num_tokens, "tokens"
        super().__init__(carrier, count, nonsequence_groups=[carrier], nonsequence_tokens=count)
        self.tokenize = bool(tokenizers)
        self.pools = nn.ModuleList(pools)
        self.tokenizers = nn.ModuleList(tokenizers)

    def forward(self, groups: dict[str, torch.Tensor], padding_mask: torch.Tensor,
                layer_index: int = 0, /) -> dict[str, torch.Tensor]:
        pool = self.pools[min(layer_index, len(self.pools) - 1)]
        context = groups["context"]
        pooled = pool(groups["sequence"], context, padding_mask)
        output = {**groups, "pooled": pooled.unsqueeze(1)}
        if self.tokenize:
            output["tokens"] = self.tokenizers[min(layer_index, len(self.tokenizers) - 1)](context, pooled)
        return output


WINDOW_DEFAULTS = {"recent_length": 20, "num_dynamic_queries": 2, "num_learned_queries": 2, "attention_heads": 2}


class WindowAttentionInteraction(Interaction):
    def __init__(self, dim: int, num_context_tokens: int, params: dict[str, t.Any], /,
                 num_layers: int = 1, *, context_fields: list[str] | None = None,
                 context_layout: Layout | None = None, readout: bool = True,
                 schedule: str = "layered", independent_layers: bool = False) -> None:
        del schedule, independent_layers, context_fields
        for key in WINDOW_DEFAULTS.keys() & params.keys():
            if type(params[key]) is not int or params[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        context_source = params.get("context_source", "tokens")
        raw_dim = None
        if context_source == "raw":
            if readout or num_context_tokens or context_layout is None or not context_layout.raw_fields:
                raise ValueError("raw windows require raw fields and no projected context/readout tokens")
            raw_dim = context_layout.num_fields * context_layout.embedding_dim
            context_source = "raw_context"
        elif context_source != "tokens":
            raise ValueError(f"unknown context_source {context_source!r}")
        else:
            context_source = "context"
        values = {key: params.get(key, default) for key, default in WINDOW_DEFAULTS.items()}
        pools = [QueryPooling(dim, num_context_tokens, **values, readout=readout, raw_context_dim=raw_dim,
                              projection=params.get("projection", "per_query_q"))
                 for _ in range(num_layers)]
        carrier, count = "tokens", 4 * values["num_dynamic_queries"] + 2 * values["num_learned_queries"]
        super().__init__(carrier, count, nonsequence_groups=[carrier], nonsequence_tokens=count)
        self.context_source = context_source
        self.pools = nn.ModuleList(pools)

    def forward(self, groups: dict[str, torch.Tensor], padding_mask: torch.Tensor,
                layer_index: int = 0, /) -> dict[str, torch.Tensor]:
        pool = self.pools[min(layer_index, len(self.pools) - 1)]
        tokens = pool(groups["sequence"], groups[self.context_source], padding_mask)
        return {**groups, "tokens": tokens}
