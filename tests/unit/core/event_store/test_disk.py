"""Tests for the disk event store backend (arrow file)."""

import datetime as dt

import polars as pl
import pytest

from perseus.core.event_store import EventStore


def _events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "client_id": ["c1", "c1", "c2"],
            "timestamp": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 1)],
            "position": [1, 2, 1],
            "event": [10, 20, 30],
        },
    )


@pytest.fixture
def store(tmp_path) -> EventStore:
    store = EventStore(tmp_path / "store", backend="disk")
    with store.open_writer() as writer:
        writer.write(_events())
    return store


def test_writer_creates_arrow_and_index(store: EventStore) -> None:
    assert (store.path / "data.arrow").exists()
    assert (store.path / "index.pq").exists()


def test_reader_returns_full_client_slice(store: EventStore) -> None:
    events = store.open_reader(128).read("c1")
    assert events["event"].to_list() == [10, 20]
    assert events["position"].to_list() == [1, 2]
    utc = dt.UTC
    assert events["timestamp"].dt.epoch(time_unit="ns").to_list() == [
        int(dt.datetime(2024, 1, 1, tzinfo=utc).timestamp() * 1e9),
        int(dt.datetime(2024, 1, 2, tzinfo=utc).timestamp() * 1e9),
    ]


def test_reader_unknown_client_raises(store: EventStore) -> None:
    with pytest.raises(KeyError, match="unknown"):
        store.open_reader(128).read("nope")
