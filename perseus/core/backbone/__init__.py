import typing as t

import torch
from torch import nn

from perseus import utils
from perseus.core import encoders
from perseus.core.backbone import context, event, history


class FieldLinear(nn.Module):
    """Per-field projection with independent weights."""

    def __init__(self, embedding_dim: int, dim: int, num_fields: int, /) -> None:
        super().__init__()

        self.fields = nn.ModuleList(nn.Linear(embedding_dim, dim) for _ in range(num_fields))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.stack([field(tokens[..., i, :]) for i, field in enumerate(self.fields)], dim=-2)

__all__ = [
    "Backbone",
    "context",
    "event",
    "history",
]


class HistoryAggregatorConfig(t.TypedDict, total=False):
    max_events_per_sequence: int
    sequence_encoder: dict[str, t.Any]
    nonsequence_encoder: dict[str, t.Any]
    interaction: dict[str, t.Any]
    sequence_state_fp32: bool


class EventAggregatorConfig(t.TypedDict):
    num_features: int
    type: str
    params: dict[str, t.Any]


class ContextAggregatorConfig(t.TypedDict):
    num_features: int
    type: str
    params: dict[str, t.Any]


class Backbone(utils.torch.Module):
    def __init__(
        self,
        dim: int,
        /,
        *,
        event_aggregator_config: EventAggregatorConfig,
        history_aggregator_config: HistoryAggregatorConfig | None = None,
        context_aggregator_config: ContextAggregatorConfig | None = None,
        embedding_dim: int | None = None,
        mask_context_from_history: bool = False,
        context_fields: list[str] | None = None,
        readout_token: bool = True,
        zero_context_features: list[str] | None = None,
    ) -> None:
        super().__init__()
        embedding_dim = dim if embedding_dim is None else embedding_dim
        self.output_shape = (dim,)
        self.event_projection = None
        self.context_projection = None
        self.history_token_output = False
        self.mask_context_from_history = mask_context_from_history
        self.context_raw_fields = False
        interaction_config = (history_aggregator_config or {}).get("interaction") or {}
        interaction_params = interaction_config.get("params") or {}
        self.raw_context_only = interaction_params.get("context_source") == "raw"
        if self.raw_context_only:
            if interaction_config.get("type") != "window_attention":
                raise ValueError("history context_source `raw` requires window_attention")
            if readout_token:
                raise ValueError("history context_source `raw` requires readout_token=false")
            if not context_aggregator_config or context_aggregator_config["type"] != "identity":
                raise ValueError("history context_source `raw` requires an identity context aggregator with fields")
            if context_aggregator_config["params"]:
                raise ValueError("history context_source `raw` identity context aggregator takes no params")
            if context_fields is not None and len(context_fields) != context_aggregator_config["num_features"]:
                raise ValueError("history context_source `raw` field schema does not match the context feature count")
        self.context_fields = list(context_fields) if context_fields is not None else None
        self.zero_context_features = zero_context_features or []
        if self.context_fields is not None and set(self.zero_context_features) - set(self.context_fields):
            raise ValueError("zero_context_features must name existing context fields")

        event_cls = event.registry[event_aggregator_config["type"]]
        raw_event_fields = event_cls.raw_fields
        event_params = dict(event_aggregator_config["params"])
        if raw_event_fields:
            event_params["embedding_dim"] = embedding_dim
        self.event_aggregator = event_cls(dim, event_aggregator_config["num_features"], **event_params)
        if embedding_dim != dim and not raw_event_fields:
            self.event_projection = FieldLinear(
                embedding_dim, dim, event_aggregator_config["num_features"],
            )

        context_layout = None
        num_context_tokens = 0
        if context_aggregator_config:
            aggregator_cls = context.registry[context_aggregator_config["type"]]
            num_features = context_aggregator_config["num_features"]
            raw_fields = aggregator_cls.raw_fields
            num_tokens = num_features
            if raw_fields:
                num_tokens = context_aggregator_config["params"].get("num_tokens", num_features)
                if isinstance(num_tokens, bool) or not isinstance(num_tokens, int) or num_tokens < 1:
                    raise ValueError(
                        f"context aggregator tokenizer `num_tokens` must be an integer >= 1, got {num_tokens!r}",
                    )
                if context_fields is not None and len(context_fields) != num_features:
                    raise ValueError(
                        f"context field schema has {len(context_fields)} names for "
                        f"{num_features} context features",
                    )
            num_context_tokens = num_tokens
            if self.raw_context_only:
                num_context_tokens = num_tokens = 0
                raw_fields = True
            context_layout = context.Layout(
                num_features, embedding_dim, num_tokens, num_tokens == num_features, raw_fields,
            )

        self.history_aggregator = history.HistoryAggregator(
            history_aggregator_config,
            dim,
            num_context_tokens,
            context_fields=self.context_fields,
            context_layout=context_layout,
            readout=readout_token,
        )
        emitted_shape = getattr(self.history_aggregator, "output_shape", None)
        if emitted_shape is not None:
            self.output_shape = emitted_shape
        self.history_token_output = emitted_shape is not None

        if context_aggregator_config:
            aggregator_cls = context.registry[context_aggregator_config["type"]]
            num_features = context_aggregator_config["num_features"]
            if self.raw_context_only:
                self.context_raw_fields = True
                self.context_aggregator = None
            elif aggregator_cls.raw_fields:
                self.context_raw_fields = True
                self.context_aggregator = aggregator_cls(
                    dim,
                    num_features,
                    embedding_dim=embedding_dim,
                    **context_aggregator_config["params"],
                )
            else:
                self.context_aggregator = aggregator_cls(
                    dim, num_features, **context_aggregator_config["params"],
                )
                if embedding_dim != dim:
                    self.context_projection = FieldLinear(
                        embedding_dim, dim, num_features,
                    )

        if readout_token:
            self.readout_token = nn.Parameter(torch.empty(dim))
            nn.init.trunc_normal_(self.readout_token, std=utils.torch.INIT_WEIGHTS_STD)
        else:
            self.readout_token = None

    def forward(
        self,
        feature_to_embedder: dict[str, encoders.Embedder],
        events_features: dict[str, torch.Tensor],
        events_positions: torch.Tensor,
        events_timestamps: torch.Tensor,
        sample_timestamp: torch.Tensor,
        context_features: dict[str, torch.Tensor] | None = None,
        /,
    ) -> torch.Tensor:
        embeddings = self.event_aggregator(self._event_embeddings(feature_to_embedder, events_features))
        raw_fields = None
        num_context_tokens = 0
        if context_features:
            context_embeddings = torch.stack(
                [feature_to_embedder[feature](values) for feature, values in context_features.items()],
                dim=1,
            )
            if self.zero_context_features:
                fields = list(context_features)
                for feature in self.zero_context_features:
                    context_embeddings[:, fields.index(feature)] = 0
            raw_fields = context_embeddings if self.context_raw_fields else None
            if not self.raw_context_only:
                if self.context_projection is not None:
                    context_embeddings = self.context_projection(context_embeddings)
                aggregated_context = self.context_aggregator(context_embeddings)
                num_context_tokens = aggregated_context.size(1)
                embeddings = torch.cat([embeddings, aggregated_context], dim=1)

        if self.readout_token is not None:
            embeddings = torch.cat(
                [embeddings, self.readout_token.expand(embeddings.size(0), 1, embeddings.size(2))],
                dim=1,
            )
        positions = self._build_positions(events_positions, num_context_tokens)
        timestamps = self._build_timestamps(events_timestamps, sample_timestamp, num_context_tokens)
        mask = None
        if self.history_aggregator.requires_attention_mask:
            mask = self._build_mask(
                events_positions, num_context_tokens, mask_context_from_history=self.mask_context_from_history,
            )
        history_tokens = self.history_aggregator(embeddings, positions, timestamps, mask, raw_fields)
        if self.history_token_output:
            return history_tokens
        return history_tokens[:, -1]

    def _event_embeddings(
        self,
        feature_to_embedder: dict[str, encoders.Embedder],
        events_features: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        events_embeddings = torch.stack(
            [feature_to_embedder[feature](values) for feature, values in events_features.items()],
            dim=2,
        )
        if self.event_projection is not None:
            events_embeddings = self.event_projection(events_embeddings)
        return events_embeddings

    def _build_positions(self, events_positions: torch.Tensor, num_context_tokens: int) -> torch.Tensor:
        block = num_context_tokens + (1 if self.readout_token is not None else 0)
        if block == 0:
            return events_positions
        context_positions = torch.zeros_like(events_positions[:, :1]).expand(-1, block)
        return torch.cat([events_positions, context_positions], dim=1)

    def _build_timestamps(
        self,
        events_timestamps: torch.Tensor,
        sample_timestamp: torch.Tensor,
        num_context_tokens: int,
    ) -> torch.Tensor:
        block = num_context_tokens + (1 if self.readout_token is not None else 0)
        if block == 0:
            return events_timestamps
        context_timestamps = sample_timestamp.unsqueeze(dim=1).expand(-1, block)
        return torch.cat([events_timestamps, context_timestamps], dim=-1)

    def _build_mask(
        self,
        events_positions: torch.Tensor,
        num_context_tokens: int,
        /,
        *,
        mask_context_from_history: bool = False,
    ) -> torch.Tensor:
        num_readout = 1 if self.readout_token is not None else 0
        events_padding_mask = events_positions == 0
        block = num_context_tokens + num_readout
        padding_mask = events_padding_mask
        if block:
            context_padding = torch.zeros_like(events_padding_mask[:, :1]).expand(-1, block)
            padding_mask = torch.cat([events_padding_mask, context_padding], dim=-1)

        total_len = events_positions.size(1) + block
        attention_mask = torch.zeros(total_len, total_len, dtype=torch.bool, device=events_positions.device)
        if num_readout:
            attention_mask[:-1, -1] = True
        if mask_context_from_history and num_context_tokens:
            events = events_positions.size(1)
            attention_mask[:events, events : events + num_context_tokens] = True

        mask = padding_mask.unsqueeze(1) | attention_mask.unsqueeze(0)
        diag = torch.arange(mask.size(-1), device=mask.device)
        mask[:, diag, diag] = False
        return mask
