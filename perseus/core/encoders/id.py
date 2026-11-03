import json
import typing as t
from functools import cached_property
from pathlib import Path

import polars as pl
import torch
from torch import nn

from perseus.core.encoders import base


class Observer(base.Observer):
    def __init__(self) -> None:
        self.value_counts = pl.DataFrame()

    def observe(self, values: pl.Series, /) -> None:
        counts = values.drop_nulls().rename("value").value_counts(name="count")
        if self.value_counts.is_empty():
            self.value_counts = counts
        else:
            self.value_counts = pl.concat([self.value_counts, counts]).group_by("value").agg(pl.col("count").sum())


class Preprocessor(base.Preprocessor):
    def __init__(self, values: pl.DataFrame, /, *, has_unk: bool = False) -> None:
        self.values = values
        self.has_unk = has_unk

    @cached_property
    def cardinality(self) -> int:
        return self.values["index"].max() or 0

    @classmethod
    def fit(
        cls,
        observer: Observer,
        /,
        *,
        top_k: int | None = None,
        min_count: int | None = None,
    ) -> t.Self:
        value_counts = observer.value_counts
        if min_count is not None:
            value_counts = value_counts.filter(pl.col("count") >= min_count)
        value_counts = value_counts.sort("count", "value", descending=[True, False])
        if top_k is not None:
            value_counts = value_counts.head(top_k)
        return cls(
            value_counts.select("value").with_row_index("index", offset=1),  # offset for padding
            has_unk=value_counts.is_empty() or top_k is not None or min_count is not None,
        )

    def static_transform(self, values: pl.Series, /) -> pl.Series:
        return (
            values.to_frame("value")
            .lazy()
            .join(self.values.lazy(), how="left", on="value", maintain_order="left")
            .select(pl.col("index").fill_null(self.cardinality + 1))
            .collect()
            .to_series()
        )

    def dynamic_transform(self, values: pl.Series, /) -> torch.Tensor:
        return self.dynamic_transform_tensor(values.to_torch())

    @property
    def tensor_transform(self) -> t.Callable[[torch.Tensor], torch.Tensor] | None:
        return self.dynamic_transform_tensor if type(self) is Preprocessor else None

    def dynamic_transform_tensor(self, values: torch.Tensor, /) -> torch.Tensor:
        return values.to(torch.int32)

    def dynamic_transform_batch(self, values: pl.Series, lengths: list[int], /) -> list[torch.Tensor]:
        if type(self).dynamic_transform is not Preprocessor.dynamic_transform:
            return super().dynamic_transform_batch(values, lengths)
        return list(self.dynamic_transform(values).split(lengths))

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.values.write_parquet(path / "values.pq")
        (path / "kwargs.json").write_text(json.dumps({"has_unk": self.has_unk}))

    @classmethod
    def load(cls, path: Path, /) -> t.Self:
        kwargs = json.loads((path / "kwargs.json").read_text())
        return cls(pl.read_parquet(path / "values.pq"), **kwargs)


class Embedder(base.Embedder):
    def __init__(self, num: int, dim: int, /, *, has_unk: bool = False, prepared: bool = False) -> None:
        super().__init__()

        self.embeddings = nn.Embedding(num + 1 + int(has_unk), dim, padding_idx=0)
        # Prepared IDs fit the table only when it includes the reserved unknown row.
        self.prepared = prepared and has_unk

    @classmethod
    def init(cls, preprocessor: Preprocessor, dim: int, /, *, prepared: bool = False) -> t.Self:
        """Use prepared=True only for IDs produced by the corresponding static_transform."""
        return cls(preprocessor.cardinality, dim, has_unk=preprocessor.has_unk, prepared=prepared)

    @property
    def unknown_embedding(self) -> torch.Tensor:
        return self.embeddings.weight[1:].mean(dim=0)

    def forward(self, indices: torch.Tensor, /) -> torch.Tensor:
        if self.prepared:
            return self.embeddings(indices)
        is_unknown = indices >= self.embeddings.num_embeddings
        if is_unknown.any():
            embeddings = self.embeddings(indices * is_unknown.logical_not())
            return embeddings + self.unknown_embedding.repeat(*indices.size(), 1) * is_unknown.unsqueeze(dim=-1)
        return self.embeddings(indices)


encoder = base.Encoder(Observer, Preprocessor, Embedder)
