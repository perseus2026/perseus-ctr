import logging
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import torch

from perseus.core.event_store import base

logger = logging.getLogger(__name__)

_META_FIELDS = ("client_id", "timestamp", "position")


class EventsReader(base.EventsReader):
    def __init__(self, path: Path, /, *, max_events_per_sequence: int, allow_empty_sequences: bool = False) -> None:
        super().__init__(
            path, max_events_per_sequence=max_events_per_sequence, allow_empty_sequences=allow_empty_sequences,
        )
        index = pl.scan_parquet(self.path / "index.pq", row_index_name="batch").select("client_id", "batch").collect()
        reader = pa.ipc.open_file(self.path / "data.arrow")
        sizes = np.array([reader.get_batch(i).num_rows for i in range(reader.num_record_batches)], dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        self._clients = {
            client: (int(starts[batch]), int(starts[batch] + sizes[batch])) for client, batch in index.iter_rows()
        }
        events = pl.read_ipc(self.path / "data.arrow")
        self._schema = events.schema
        token_columns = [name for name in events.columns if name not in _META_FIELDS]
        for name in token_columns:
            if not events.schema[name].is_numeric():
                raise ValueError(f"cached event store supports numeric event columns only: {name}")
        if any(events[name].null_count() for name in events.columns):
            raise ValueError("cached event store does not support null values")
        self._timestamps = events["timestamp"].dt.epoch("ns").to_torch()
        self._positions = events["position"].to_torch()
        self._columns = {name: events[name].to_torch() for name in token_columns}
        try:
            self._timestamps.share_memory_()
            self._positions.share_memory_()
            for tensor in self._columns.values():
                tensor.share_memory_()
        except RuntimeError:
            logger.warning("torch shared memory is unavailable, the event store copy stays process-local")
        for start, end in self._clients.values():
            if np.any(np.diff(self._timestamps[start:end].numpy()) < 0):
                raise ValueError("cached event store requires events sorted by timestamp within each client")

    def _bounds(self, client_id: str) -> tuple[int, int]:
        if (bounds := self._clients.get(client_id)) is None:
            if self.allow_empty_sequences:
                return 0, 0
            raise KeyError(f"unknown {client_id = }")
        return bounds

    def read_batch(
        self,
        client_ids: list[str],
        boundaries: list[int | None],
        /,
        *,
        features: list[str] | None = None,
    ) -> tuple[pl.DataFrame, list[int]]:
        columns, lengths = self.read_batch_tensors(client_ids, boundaries, features=features)
        return pl.DataFrame({
            name: pl.Series(values.numpy()).cast(pl.Datetime("ns")).cast(self._schema[name])
            if name == "timestamp"
            else pl.Series(values.numpy()).cast(pl.Int64 if name == "position" else self._schema[name])
            for name, values in columns.items()
        }), lengths

    def read_batch_tensors(
        self,
        client_ids: list[str],
        boundaries: list[int | None],
        /,
        *,
        features: list[str] | None = None,
    ) -> tuple[dict[str, torch.Tensor], list[int]]:
        """Return numeric columns and history lengths; timestamps are epoch nanoseconds."""
        names = list(self._columns) if features is None else features
        timestamps = self._timestamps.numpy()
        stored_positions = self._positions.numpy()
        indices, positions, lengths = [], [], []
        for client, boundary in zip(client_ids, boundaries, strict=True):
            start, stop = self._bounds(client)
            if boundary is not None:
                stop = start + int(np.searchsorted(timestamps[start:stop], boundary, side="left"))
            position = stored_positions[start:stop]
            if boundary is not None and position.size:
                position = position - position.min() + 1
            keep = position <= self.max_events_per_sequence
            indices.append(start + np.flatnonzero(keep))
            positions.append(position[keep])
            lengths.append(int(keep.sum()))

        selected = torch.from_numpy(np.concatenate(indices))
        events = {
            "position": torch.from_numpy(np.concatenate(positions)).long(),
            "timestamp": self._timestamps[selected],
            **{name: self._columns[name][selected] for name in names},
        }
        return events, lengths

    def _read_events(self, client_id: str) -> pl.DataFrame:
        start, end = self._bounds(client_id)
        return pl.DataFrame(
            {
                "client_id": pl.Series([client_id] * (end - start), dtype=self._schema["client_id"]),
                **{
                    name: pl.Series(self._columns[name][start:end].numpy()).cast(self._schema[name])
                    for name in self._columns
                },
                "position": pl.Series(self._positions[start:end].numpy()),
                "timestamp": pl.Series(self._timestamps[start:end].numpy())
                .cast(pl.Datetime("ns"))
                .cast(self._schema["timestamp"]),
            },
        ).select(self._schema.names())
