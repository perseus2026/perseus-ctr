import typing as t
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch

from perseus import utils


class Observer(ABC):
    @abstractmethod
    def observe(self, values: pl.Series, /) -> None: ...


class Preprocessor(ABC):
    @property
    def tensor_transform(self) -> t.Callable[[torch.Tensor], torch.Tensor] | None:
        """Independent column transform, when supported without per-sample state."""
        return None

    @classmethod
    @abstractmethod
    def fit(cls, observer: Observer, /, **kwargs: t.Any) -> t.Self: ...

    @abstractmethod
    def static_transform(self, values: pl.Series, /) -> pl.Series: ...

    @abstractmethod
    def dynamic_transform(self, values: pl.Series, /) -> torch.Tensor: ...

    def dynamic_transform_batch(self, values: pl.Series, lengths: list[int], /) -> list[torch.Tensor]:
        """Transform concatenated histories, preserving per-history semantics."""
        result = []
        offset = 0
        for length in lengths:
            result.append(self.dynamic_transform(values.slice(offset, length)))
            offset += length
        return result

    @abstractmethod
    def save(self, path: Path, /) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path, /) -> t.Self: ...


class Embedder(utils.torch.Module, ABC):
    @classmethod
    @abstractmethod
    def init(cls, preprocessor: Preprocessor, dim: int, /, **kwargs: t.Any) -> t.Self: ...

    @abstractmethod
    def forward(self, values: torch.Tensor, /) -> torch.Tensor: ...


@dataclass(frozen=True)
class Encoder:
    observer: type[Observer]
    preprocessor: type[Preprocessor]
    embedder: type[Embedder]
