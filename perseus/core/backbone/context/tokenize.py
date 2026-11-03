import torch
from torch import nn

from perseus.core.backbone.context import base


class Aggregator(base.Aggregator):
    """Partition tokenizer over raw field embeddings.

    Flattens the uniform field embeddings, splits them into `num_tokens`
    sequential chunks, zero-pads the last chunk when the split is uneven and
    projects every chunk with its own Linear/LayerNorm/SiLU. `num_tokens`
    defaults to the field count, which turns one chunk into exactly one field.
    """

    raw_fields = True

    def __init__(
        self,
        dim: int,
        num_features: int,
        /,
        *,
        embedding_dim: int,
        num_tokens: int | None = None,
    ) -> None:
        super().__init__(dim, num_features)
        if num_tokens is not None and (type(num_tokens) is not int or num_tokens < 1):
            raise ValueError(f"context tokenizer `num_tokens` must be an integer >= 1, got {num_tokens!r}")
        self.num_fields = num_features
        self.embedding_dim = embedding_dim
        self.num_tokens = num_features if num_tokens is None else num_tokens
        self.input_dim = num_features * embedding_dim
        self.chunk_dim = -(-self.input_dim // self.num_tokens)
        self.output_shape = (self.num_tokens, dim)
        self.token_mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(self.chunk_dim, dim), nn.LayerNorm(dim), nn.SiLU())
            for _ in range(self.num_tokens)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3 or features.size(1) != self.num_fields or features.size(2) != self.embedding_dim:
            raise ValueError(
                f"context tokenizer expects shape [B, {self.num_fields}, {self.embedding_dim}], "
                f"got {tuple(features.shape)}",
            )
        flattened = features.flatten(start_dim=1)
        padded = self.chunk_dim * self.num_tokens - self.input_dim
        if padded:
            flattened = nn.functional.pad(flattened, (0, padded))
        chunks = flattened.view(features.size(0), self.num_tokens, self.chunk_dim)
        return torch.stack(
            [mlp(chunk) for mlp, chunk in zip(self.token_mlps, chunks.unbind(dim=1), strict=True)],
            dim=1,
        )
