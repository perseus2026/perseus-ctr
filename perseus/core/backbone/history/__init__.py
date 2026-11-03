"""History encoder assembled from sequence, interaction and nonsequence modules."""

import typing as t

import torch
from torch import nn

from perseus.core.backbone.context.base import Layout
from perseus.core.backbone.history import interaction, nonsequence, pooling, sequence
from perseus.core.backbone.history.nonsequence.stack import bind_norm_groups, check_norm_groups
from perseus.core.backbone.history.sequence.base import Aggregator, State

registry = sequence.registry
__all__ = ["Aggregator", "HistoryAggregator", "State", "registry"]


class HistoryAggregator(nn.Module):
    """A history model assembled from sequence, interaction and non-sequence encoders."""

    @property
    def requires_attention_mask(self) -> bool:
        return self.sequence.requires_attention_mask

    def __init__(
        self, config: dict[str, t.Any], dim: int, num_context_tokens: int, /, *,
        context_fields: list[str] | None = None, context_layout: Layout | None = None,
        readout: bool = True,
    ) -> None:
        super().__init__()
        sequence_config = config.get("sequence_encoder") or {"type": "identity"}
        nonsequence_config = config.get("nonsequence_encoder") or {"type": "identity"}
        interaction_config = config.get("interaction") or {"type": "identity"}
        sequence_params = dict(sequence_config.get("params") or {})
        nonsequence_params = dict(nonsequence_config.get("params") or {})
        interaction_params = interaction_config.get("params") or {}
        self.schedule = interaction_config.get("schedule", "all_as_sequence")
        self.sequence_state_fp32 = config.get("sequence_state_fp32", False)
        if self.sequence_state_fp32 and self.schedule == "all_as_sequence":
            raise ValueError("sequence_state_fp32 requires a separate sequence stream (once or layered schedule)")
        self.layer_order = config.get("layer_order") or ["sequence_encoder", "interaction", "nonsequence_encoder"]
        self.num_context_tokens = num_context_tokens
        self.readout = readout
        self.finalize_in_stream = sequence_config.get("finalize_in_stream", False)
        independent = interaction_config.get("independent_layers", False)
        common_layers = config.get("num_layers")
        if common_layers is None:
            common_layers = 1
        if self.schedule == "layered":
            sequence_params["num_layers"] = common_layers
            nonsequence_params["num_layers"] = common_layers
        self.num_layers = common_layers if self.schedule == "layered" else 1
        num_interaction_layers = self.num_layers if independent and self.schedule == "layered" else 1

        # Initialize the interaction before the non-sequence encoder so its
        # token layout determines the number of non-sequence inputs.
        self.interaction = interaction.registry[interaction_config["type"]](
            dim, num_context_tokens, interaction_params, num_interaction_layers,
            context_fields=context_fields, context_layout=context_layout, readout=readout,
            schedule=self.schedule, independent_layers=independent,
        )
        self.sequence = sequence.registry[sequence_config["type"]](
            dim, config["max_events_per_sequence"], **sequence_params,
        )

        interaction_group = self.interaction.carrier_group
        context_count = num_context_tokens + int(readout)
        groups = self.interaction.nonsequence_groups
        num_tokens = self.interaction.nonsequence_tokens
        block_slices = [operation["nonsequence_encoder"] for operation in self.layer_order
                        if isinstance(operation, dict)]
        self.nonsequence = nonsequence.registry[nonsequence_config["type"]](
            dim, num_context_tokens, groups, num_tokens, params=nonsequence_params,
            num_layers=self.num_layers, independent_layers=independent, sliced=bool(block_slices),
            block_slices=block_slices,
            once=self.schedule == "once", context_layout=context_layout, readout=readout,
        )
        components = {
            "sequence_encoder": self.sequence,
            "interaction": self.interaction,
            "nonsequence_encoder": self.nonsequence,
        }
        if self.schedule == "all_as_sequence":
            if any(not component.is_noop for name, component in components.items() if name != "sequence_encoder"):
                raise ValueError("all_as_sequence does not use interaction or nonsequence encoders")
            check_norm_groups([self.sequence])
            self.output_shape = None
            return
        if self.schedule == "layered":
            names = {operation if isinstance(operation, str) else next(iter(operation))
                     for operation in self.layer_order}
            if any(not component.step_is_noop and name not in names for name, component in components.items()):
                raise ValueError("layer_order must include each non-noop component")
            if self.interaction.requires_sequence_step and "sequence_encoder" not in names:
                raise ValueError("interaction requires sequence_encoder in layer_order")

        if self.interaction.query_tokens is not None:
            if self.nonsequence.num_queries is None:
                raise ValueError("interaction requires a nonsequence query generator")
            if self.nonsequence.num_queries != self.interaction.query_tokens:
                raise ValueError("query generator and interaction must use the same number of queries")
        if (self.schedule == "layered" and self.interaction.requires_fixed_width
                and self.nonsequence.output_dim != dim):
            raise ValueError("layered nonsequence encoder must preserve token width for interaction")

        self.shared_norms = nn.ModuleList()
        if self.schedule == "layered":
            sequence_norms = self.sequence.normalization_layers
            nonsequence_norms = self.nonsequence.normalization_layers
            for index in range(self.num_layers if independent else 1):
                if independent:
                    modules = [nonsequence_norms[min(index, len(nonsequence_norms) - 1)]]
                    if sequence_norms:
                        modules.append(sequence_norms[min(index, len(sequence_norms) - 1)])
                else:
                    modules = [*nonsequence_norms, *sequence_norms]
                self.shared_norms.extend(bind_norm_groups(modules))
        check_norm_groups([self.sequence, self.nonsequence])

        self.pools = [(interaction_config.get("pool"), interaction_group), (nonsequence_config.get("pool"), groups[-1]),
                      (sequence_config.get("pool"), "sequence")]
        lengths = {
            "sequence": config["max_events_per_sequence"],
            "context": context_count,
            **self.interaction.token_counts,
        }
        widths = {key: (self.nonsequence.output_dim if key in groups else dim) for key in lengths}
        self.output_shape = pooling.output_shape(self.pools, lengths, widths, readout=readout)

    def forward(
        self,
        embeddings: torch.Tensor,
        positions: torch.Tensor,
        timestamps: torch.Tensor,
        mask: torch.Tensor | None,
        raw_fields: torch.Tensor | None = None,
        /,
    ) -> torch.Tensor:
        match self.schedule:
            case "all_as_sequence":
                padding_mask = positions == 0
                context_length = self.num_context_tokens + int(self.readout)
                if context_length:
                    padding_mask = padding_mask.clone()
                    padding_mask[:, -context_length:] = False
                return self.sequence(embeddings, positions, timestamps, mask, padding_mask=padding_mask)
            case "once" | "layered":
                pass

        history_length = embeddings.size(1) - self.num_context_tokens - int(self.readout)
        history = embeddings[:, :history_length]
        if self.sequence_state_fp32:
            history = history.float()
        history_positions = positions[:, :history_length]
        history_timestamps = timestamps[:, :history_length]
        history_mask = None if mask is None else mask[:, :history_length, :history_length]
        padding_mask = positions[:, :history_length] == 0
        groups = {"context": embeddings[:, history_length:], "raw_context": raw_fields}

        match self.schedule:
            case "once":
                groups["sequence"] = self.sequence(history, history_positions, history_timestamps, history_mask)
                if self.sequence_state_fp32:
                    groups["sequence"] = groups["sequence"].float()
                queries = self.nonsequence.create_queries(
                    groups["sequence"], groups["context"], padding_mask, raw_fields,
                )
                if queries is not None:
                    groups["queries"] = queries
                groups = self.interaction(groups, padding_mask)
                groups = self.nonsequence(groups, padding_mask)
            case "layered":
                state = self.sequence.prepare(history, history_positions, history_timestamps, history_mask)
                groups["sequence"] = state.x
                queries = self.nonsequence.create_queries(
                    groups["sequence"], groups["context"], padding_mask, raw_fields,
                )
                if queries is not None:
                    groups["queries"] = queries
                for index in range(self.num_layers):
                    for operation in self.layer_order:
                        if operation == "sequence_encoder":
                            state = self.sequence.step(state, index)
                            if self.finalize_in_stream and index == self.num_layers - 1:
                                state = state._replace(x=self.sequence.finalize(state))
                            state = state._replace(
                                x=self.interaction.update_sequence(state.x, groups, padding_mask, index),
                            )
                            if self.sequence_state_fp32:
                                state = state._replace(x=state.x.float())
                            groups["sequence"] = state.x
                        elif operation == "interaction":
                            groups = self.interaction(groups, padding_mask, index)
                        elif operation == "nonsequence_encoder":
                            groups = self.nonsequence(groups, padding_mask, index)
                        else:
                            groups = self.nonsequence(groups, padding_mask, index, operation["nonsequence_encoder"])
        return pooling.pool(groups, self.pools, padding_mask)
