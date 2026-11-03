"""Tokenizers of the history interaction components."""

import torch
from torch import nn

from perseus.core.backbone.history.interaction.cross_attention import CrossAttention

from perseus.core.backbone.history.tokenizer import AutoSplitTokenizer


class ChunkTokenizer(nn.Module):
    """Split flattened features into channel chunks, one MLP per token."""

    def __init__(self, input_dim: int, token_dim: int, num_tokens: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_tokens = num_tokens
        self.chunk_dim = -(-input_dim // num_tokens)
        self.token_mlps = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.chunk_dim, token_dim), nn.LayerNorm(token_dim), nn.SiLU(),
            )
            for _ in range(num_tokens)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.chunk_dim * self.num_tokens != self.input_dim:
            features = nn.functional.pad(features, (0, self.chunk_dim * self.num_tokens - self.input_dim))
        chunks = features.view(features.size(0), self.num_tokens, self.chunk_dim)
        return torch.stack(
            [mlp(chunk) for mlp, chunk in zip(self.token_mlps, chunks.unbind(dim=1), strict=True)],
            dim=1,
        )


class MixedHeteroAttention(nn.Module):
    """Per-query projection attention pooling over one history window (HeMix)."""

    def __init__(self, dim: int, num_queries: int, num_heads: int, projection: str = "per_query_q") -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.projection = projection
        if projection == "per_query_kv":
            self.attention = CrossAttention(dim, num_queries, num_heads, projection, padding="zero")
            return
        if projection != "per_query_q":
            raise ValueError(f"unknown window projection {projection!r}")
        self.query_weight = nn.Parameter(torch.empty(num_queries, dim, dim))
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        nn.init.xavier_uniform_(self.query_weight)

    def forward(self, history: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        if history.size(1) == 0:
            return queries
        if self.projection == "per_query_kv":
            padding = torch.zeros(history.shape[:2], dtype=torch.bool, device=history.device)
            return self.attention(queries, history, padding)
        batch, length, dim = queries.shape
        query = torch.einsum("bld,ldh->blh", queries, self.query_weight)
        query = query.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(history).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(history).view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)
        output = nn.functional.scaled_dot_product_attention(query, key, value, scale=self.head_dim**-0.5)
        return self.output(output.transpose(1, 2).contiguous().view(batch, length, dim))


class MixedQueriesTokenizer(nn.Module):
    """HeMix tokenizer: dynamic ns tokens and learned CLS queries attending their history windows."""

    def __init__(
        self, input_dim: int, dim: int, num_dynamic_queries: int, num_learned_queries: int, attention_heads: int,
        projection: str = "per_query_q",
    ) -> None:
        super().__init__()
        self.num_dynamic_queries = num_dynamic_queries
        self.token_mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(input_dim, dim), nn.LayerNorm(dim), nn.SiLU())
            for _ in range(2 * num_dynamic_queries)
        )
        self.cls_tokens_earlier = nn.Parameter(torch.empty(1, num_learned_queries, dim))
        self.cls_tokens_recent = nn.Parameter(torch.empty(1, num_learned_queries, dim))
        nn.init.xavier_uniform_(self.cls_tokens_earlier)
        nn.init.xavier_uniform_(self.cls_tokens_recent)
        queries_per_window = num_dynamic_queries + num_learned_queries
        self.attention_earlier = MixedHeteroAttention(dim, queries_per_window, attention_heads, projection)
        self.attention_recent = MixedHeteroAttention(dim, queries_per_window, attention_heads, projection)

    def forward(self, features: torch.Tensor, recent: torch.Tensor, earlier: torch.Tensor) -> torch.Tensor:
        ns_tokens = torch.stack([mlp(features) for mlp in self.token_mlps], dim=1)
        dynamic = self.num_dynamic_queries
        recent_queries = torch.cat(
            [ns_tokens[:, dynamic:], self.cls_tokens_recent.expand(features.size(0), -1, -1)], dim=1,
        )
        earlier_queries = torch.cat(
            [ns_tokens[:, :dynamic], self.cls_tokens_earlier.expand(features.size(0), -1, -1)], dim=1,
        )
        return torch.cat(
            [
                self.attention_recent(recent, recent_queries),
                self.attention_earlier(earlier, earlier_queries),
                ns_tokens,
            ],
            dim=1,
        )


class FieldTaskTokenizer(nn.Module):
    """HiFormer tokenizer: per-field MLPs plus persistent task tokens."""

    def __init__(self, embedding_dim: int, dim: int, num_fields: int, num_task_tokens: int) -> None:
        super().__init__()
        self.task_tokens = nn.Parameter(torch.empty(1, num_task_tokens, dim))
        nn.init.xavier_uniform_(self.task_tokens)
        self.token_mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(embedding_dim, dim), nn.LayerNorm(dim), nn.SiLU()) for _ in range(num_fields)
        )

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        tokens = torch.stack(
            [mlp(field) for mlp, field in zip(self.token_mlps, fields.unbind(dim=1), strict=True)], dim=1,
        )
        return torch.cat([tokens, self.task_tokens.expand(fields.size(0), -1, -1)], dim=1)


class GlobalGroupTokenize(nn.Module):
    """TokenMixer tokenizer: one global MLP over the flat fields plus chunked group MLPs."""

    def __init__(self, dim: int, field_dim: int, num_group_tokens: int, /) -> None:
        super().__init__()
        self.global_tokenizer = AutoSplitTokenizer(field_dim, dim, 1)
        self.group_tokenizer = ChunkTokenizer(field_dim, dim, num_group_tokens)

    def forward(self, group: torch.Tensor, pooled: torch.Tensor, /) -> torch.Tensor:
        features = torch.cat([group.flatten(start_dim=1), pooled], dim=-1)
        return torch.cat([self.global_tokenizer(features), self.group_tokenizer(features)], dim=1)


class FieldTaskTokenize(nn.Module):
    """HiFormer tokenizer: per-field MLPs over context, readout and pooled tokens plus task tokens."""

    def __init__(self, dim: int, num_fields: int, num_tasks: int, /) -> None:
        super().__init__()
        self.tokenizer = FieldTaskTokenizer(dim, dim, num_fields, num_tasks)

    def forward(self, group: torch.Tensor, pooled: torch.Tensor, /) -> torch.Tensor:
        return self.tokenizer(torch.cat([group, pooled.unsqueeze(1)], dim=1))
