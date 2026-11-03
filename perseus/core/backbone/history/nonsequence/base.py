"""Non-sequential token encoders and optional query generation."""

import typing as t

import torch
from torch import nn

from perseus.core.backbone.history.nonsequence import queries, stack

if t.TYPE_CHECKING:
    from perseus.core.backbone.context.base import Layout


class Encoder(nn.Module):
    """Apply non-sequence blocks to context, queries or interaction tokens."""

    step_is_noop = False
    supports_slices = False

    @property
    def normalization_layers(self) -> nn.ModuleList:
        return self.layers

    def __init__(
        self,
        dim: int,
        num_context_tokens: int,
        groups: list[str],
        num_tokens: int,
        /,
        *,
        params: dict[str, t.Any],
        num_layers: int,
        independent_layers: bool,
        sliced: bool = False,
        block_slices: list[list[int]] | None = None,
        once: bool = False,
        context_layout: "Layout | None" = None,
        readout: bool = True,
    ) -> None:
        super().__init__()
        params = dict(params)
        if "num_layers" in params and (type(params["num_layers"]) is not int or params["num_layers"] < 1):
            raise ValueError("num_layers must be a positive integer")
        self.configure_queries(params, dim, num_context_tokens, context_layout, readout=readout)
        self.is_noop = self.step_is_noop and self.query_generator is None
        self.sliced = sliced
        self.layers = nn.ModuleList()
        self.output_dim = dim
        count = 1 if once or not independent_layers else num_layers
        depth = params.get("num_layers", 1) if once else 1
        for _ in range(count):
            self.output_dim = dim
            if self.step_is_noop:
                self.layers.append(nn.Identity())
            else:
                blocks = nn.ModuleList()
                for _ in range(depth):
                    for block in self.build_blocks(self.output_dim, num_tokens, params):
                        self.output_dim = stack.width_after(self.output_dim, block)
                        blocks.append(block)
                if sliced:
                    self.layers.append(nn.ModuleList(stack.GroupTransform(groups, nn.ModuleList([block]))
                                                     for block in blocks))
                else:
                    self.layers.append(stack.GroupTransform(groups, blocks))
        if block_slices:
            if not self.supports_slices:
                raise ValueError("nonsequence slices require a stack")
            indices = [index for group in block_slices for index in group]
            if any(not group for group in block_slices) or sorted(indices) != list(range(len(self.layers[0]))):
                raise ValueError("nonsequence slices must cover every block exactly once")

    def forward(
        self, groups: dict[str, torch.Tensor], padding_mask: torch.Tensor,
        layer_index: int | None = None, blocks: list[int] | None = None,
    ) -> dict[str, torch.Tensor]:
        if self.step_is_noop:
            return groups
        index = 0 if layer_index is None or len(self.layers) == 1 else layer_index
        selected = self.layers[index]
        if self.sliced:
            for block_index in blocks if blocks is not None else range(len(selected)):
                groups = selected[block_index](groups, padding_mask)
            return groups
        return selected(groups, padding_mask)

    def build_blocks(self, dim: int, num_tokens: int, params: dict[str, t.Any]) -> list[nn.Module]:
        raise NotImplementedError

    def configure_queries(
        self, params: dict[str, t.Any], dim: int, num_context_tokens: int,
        context_layout: "Layout | None", *, readout: bool,
    ) -> None:
        query_config = params.pop("queries", None)
        self.num_queries = None
        self.query_generator = None
        if query_config is not None:
            kind = query_config["type"]
            if kind not in queries.registry:
                raise ValueError(f"unknown query generator: {kind}")
            self.query_generator = queries.registry[kind](
                dim, num_context_tokens, context_layout=context_layout, readout=readout,
                params=query_config.get("params") or {},
            )
            self.num_queries = self.query_generator.num_tokens

    def create_queries(
        self, sequence: torch.Tensor, context: torch.Tensor, padding_mask: torch.Tensor,
        raw_context: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.query_generator is None:
            return None
        return self.query_generator(sequence, context, padding_mask, raw_context)
