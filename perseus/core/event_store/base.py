import datetime as dt
import logging
import typing as t
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager
from pathlib import Path

import polars as pl

from perseus import utils

logger = logging.getLogger(__name__)


class EventsWriter(AbstractContextManager):
    def __init__(self, path: Path, /) -> None:
        self.path = path

    def __enter__(self) -> t.Self:
        return self

    @utils.timer(logger, "writing events to event store")
    def write(self, events: pl.DataFrame, /) -> None:
        return self._write(events)

    @abstractmethod
    def _write(self, events: pl.DataFrame, /) -> None:
        raise NotImplementedError


class EventsReader(ABC):
    def __init__(self, path: Path, /, *, max_events_per_sequence: int, allow_empty_sequences: bool = False) -> None:
        self.path = path
        self.max_events_per_sequence = max_events_per_sequence
        self.allow_empty_sequences = allow_empty_sequences

    def read(
        self,
        client_id: str,
        /,
        *,
        before: int | dt.date | dt.datetime | None = None,
    ) -> pl.DataFrame:
        """Read a client's history; integer boundaries are epoch nanoseconds."""
        events = self._read_events(client_id)
        if before is not None:
            boundary = (
                pl.col("timestamp").dt.epoch(time_unit="ns") < before
                if isinstance(before, int)
                else pl.col("timestamp") < before
            )
            events = (
                events.lazy()
                .filter(boundary)
                .with_columns(position=pl.col("position") - pl.col("position").min() + 1)
                .collect()
            )

        return events.filter(pl.col("position") <= self.max_events_per_sequence)

    def read_batch(
        self,
        client_ids: list[str],
        boundaries: list[int | None],
        /,
        *,
        features: list[str] | None = None,
    ) -> tuple[pl.DataFrame, list[int]]:
        """Read histories before epoch-nanosecond boundaries in request order.

        Return the concatenated rows and their counts, one count per requested client.
        """
        frames = [
            self.read(client_id, before=boundary) for client_id, boundary in zip(client_ids, boundaries, strict=True)
        ]
        if features is None:
            features = [name for name in frames[0].columns if name not in ("client_id", "timestamp", "position")]
        columns = ["position", "timestamp", *features]
        return pl.concat([frame.select(columns) for frame in frames]), [frame.height for frame in frames]

    @abstractmethod
    def _read_events(self, client_id: str) -> pl.DataFrame:
        raise NotImplementedError
