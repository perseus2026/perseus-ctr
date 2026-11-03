"""Cross attention between sequence and context/query carriers."""

import typing as t

import torch
from torch import nn

from perseus.core.backbone.context.base import Layout
from perseus.core.backbone.history.interaction.base import Interaction


class CrossAttention(nn.Module):
    """Cross-attention primitive with the projections used by EST, HyFormer and MixFormer."""

    _PROJECTIONS: t.ClassVar[set[str]] = {"per_token", "compact", "shared", "per_query_kv"}

    def __init__(
        self, dim: int, reader_tokens: int, num_heads: int, projection: str, /, *, padding: str = "mask",
    ) -> None:
        super().__init__()
        if type(num_heads) is not int or num_heads < 1:
            raise ValueError("num_heads must be a positive integer")
        if type(reader_tokens) is not int or reader_tokens < 1:
            raise ValueError("reader_tokens must be a positive integer")
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        if projection not in self._PROJECTIONS:
            raise ValueError(f"unknown cross_attention projection `{projection}`")
        if padding not in {"mask", "zero"}:
            raise ValueError(f"unknown cross_attention padding `{padding}`")
        self.padding = padding
        self.projection = projection
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.reader_tokens = reader_tokens
        if projection == "per_token":
            self.attention_norm = nn.LayerNorm(dim)
            self.query_weight = nn.Parameter(torch.empty(reader_tokens, dim, dim))
            self.key = nn.Linear(dim, dim, bias=False)
            self.value = nn.Linear(dim, dim, bias=False)
            self.output = nn.Parameter(torch.empty(reader_tokens, dim, dim))
            nn.init.xavier_uniform_(self.query_weight)
            nn.init.xavier_uniform_(self.output)
        elif projection == "compact":
            self.attention_norm = nn.LayerNorm(dim)
            self.query = nn.Linear(dim, dim, bias=False)
            self.key = nn.Linear(dim, dim, bias=False)
            self.value = nn.Linear(dim, dim, bias=False)
            self.output = nn.Linear(dim, dim, bias=False)
        elif projection == "shared":
            self.norm = nn.LayerNorm(dim)
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.output = nn.Linear(dim, dim)
        else:
            self.key = nn.Linear(dim, dim * reader_tokens)
            self.value = nn.Linear(dim, dim * reader_tokens)

    def forward(self, reader: torch.Tensor, kv: torch.Tensor, padding_mask: torch.Tensor, /) -> torch.Tensor:
        empty = kv.size(1) == 0
        if empty:
            kv = nn.functional.pad(kv, (0, 0, 0, 1))
            padding_mask = nn.functional.pad(padding_mask, (0, 1), value=True)
        valid = ~padding_mask
        has_kv = valid.any(dim=-1)
        safe_keys = valid.clone()
        safe_keys[:, 0] |= ~has_kv
        attention_mask = safe_keys[:, None, None, :]
        has_kv = has_kv[:, None, None]
        zero_padding = self.padding == "zero"
        if zero_padding:
            kv = kv * valid.unsqueeze(-1)
            attention_mask = None
            # Zero padding keeps projected biases in the attention, even on cold rows.
            if not empty:
                has_kv = torch.ones_like(has_kv)

        if self.projection in {"per_token", "compact"}:
            batch, length, dim = reader.shape
            reader = self.attention_norm(reader)
            kv = self.attention_norm(kv)
            if zero_padding:
                kv = kv * valid.unsqueeze(-1)
            if self.projection == "per_token":
                query = torch.einsum("bld,ldh->blh", reader, self.query_weight)
            else:
                query = self.query(reader)
            query = query.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
            key = self.key(kv).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
            value = self.value(kv).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
            attended = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
            attended = attended.transpose(1, 2).reshape(batch, length, dim)
            if self.projection == "per_token":
                contribution = torch.einsum("bld,ldh->blh", attended, self.output)
            else:
                contribution = self.output(attended)
            return torch.where(has_kv, contribution, torch.zeros_like(contribution))

        if self.projection == "shared":
            batch, length, dim = reader.shape
            query = self.query(self.norm(reader)).view(
                batch, length, self.num_heads, self.head_dim,
            ).transpose(1, 2)
            norm_kv = self.norm(kv)
            key = self.key(norm_kv).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
            value = self.value(norm_kv).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
            attended = nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
            attended = attended.transpose(1, 2).reshape(batch, length, dim)
            contribution = self.output(attended)
            return reader + torch.where(has_kv, contribution, torch.zeros_like(contribution))

        batch, _, dim = reader.shape
        keys = self.key(kv).view(batch, -1, self.reader_tokens, dim).permute(0, 2, 1, 3)
        values = self.value(kv).view(batch, -1, self.reader_tokens, dim).permute(0, 2, 1, 3)
        attended = nn.functional.scaled_dot_product_attention(
            reader.unsqueeze(2), keys, values, attn_mask=attention_mask, scale=dim**-0.5,
        )
        contribution = attended.squeeze(2)
        return torch.where(has_kv, contribution, torch.zeros_like(contribution))


class CrossAttentionInteraction(Interaction):
    """Update the context or persistent query carrier from the history sequence."""

    def __init__(
        self,
        dim: int,
        num_context_tokens: int,
        params: dict[str, t.Any],
        /,
        num_layers: int = 1,
        *,
        context_fields: list[str] | None = None,
        context_layout: Layout | None = None,
        readout: bool = True,
        schedule: str = "layered",
        independent_layers: bool = False,
    ) -> None:
        del context_fields, context_layout
        reader = params.get("reader", "context")
        projection = params.get("projection", "per_token")
        write = params.get("write", "add")
        num_heads = params.get("num_heads", 2)
        if "reverse_context" in params:
            if type(params["reverse_context"]) is not bool:
                raise ValueError("reverse_context must be boolean")
            if params["reverse_context"] and reader != "queries":
                raise ValueError("reverse_context requires a query reader")
            if params["reverse_context"] and num_context_tokens == 0:
                raise ValueError("reverse_context requires context tokens")
            if params["reverse_context"] and (schedule != "layered" or not independent_layers):
                raise ValueError("reverse_context requires layered schedule and independent layers")
        if reader not in {"context", "queries"}:
            raise ValueError(f"unknown cross_attention reader `{reader}`")
        if write not in {"add", "replace"}:
            raise ValueError(f"unknown cross_attention write mode `{write}`")
        if projection == "shared" and write == "add":
            raise ValueError("projection `shared` returns its residual; set write to `replace`")
        if reader == "queries":
            reader_tokens = params["num_tokens"]
        else:
            reader_tokens = num_context_tokens + int(readout)
            if reader_tokens == 0:
                raise ValueError("cross_attention context reader requires context or a readout token")

        query_tokens = reader_tokens if reader == "queries" else None
        super().__init__(
            "queries" if query_tokens is not None else None, query_tokens,
            nonsequence_groups=["queries", "context"] if query_tokens is not None else ["context"],
            nonsequence_tokens=num_context_tokens + int(readout) + (query_tokens or 0),
            query_tokens=query_tokens,
        )
        self.reader = reader
        self.write = write
        self.readout = readout
        self.layers = nn.ModuleList([
            CrossAttention(dim, reader_tokens, num_heads, projection, padding=params.get("padding", "mask"))
            for _ in range(num_layers)
        ])
        self.reverse_layers = nn.ModuleList([
            CrossAttention(dim, num_context_tokens, num_heads, "compact", padding=params.get("padding", "mask"))
            for _ in range(num_layers)
        ]) if params.get("reverse_context", False) else None
        self.requires_sequence_step = self.reverse_layers is not None

    def forward(
        self, groups: dict[str, torch.Tensor], padding_mask: torch.Tensor, layer_index: int = 0, /,
    ) -> dict[str, torch.Tensor]:
        reader = groups[self.reader]
        update = self.layers[min(layer_index, len(self.layers) - 1)](
            reader, groups["sequence"], padding_mask,
        )
        value = reader + update if self.write == "add" else update
        return {**groups, self.reader: value}

    def update_sequence(self, sequence_state: torch.Tensor, groups: dict[str, torch.Tensor], padding_mask: torch.Tensor,
                layer_index: int = 0, /) -> torch.Tensor:
        if self.reverse_layers is None:
            return sequence_state
        context = groups["context"][:, : -1 if self.readout else None]
        context_padding = torch.zeros(context.shape[0], context.shape[1], dtype=torch.bool, device=context.device)
        delta = self.reverse_layers[min(layer_index, len(self.reverse_layers) - 1)](
            sequence_state, context, context_padding,
        )
        valid = ~padding_mask
        return torch.where(valid.unsqueeze(-1), sequence_state + delta, sequence_state)
