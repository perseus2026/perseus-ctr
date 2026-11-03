"""Tests for the RAM event store backend and the EventStore dispatcher."""

import datetime as dt

import polars as pl
import pytest

from perseus.core.event_store import EventStore


def _events() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "client_id": ["c1", "c1", "c1", "c2"],
            "timestamp": [
                dt.datetime(2024, 1, 1),
                dt.datetime(2024, 1, 2),
                dt.datetime(2024, 1, 3),
                dt.datetime(2024, 1, 1),
            ],
            "position": [1, 2, 3, 1],
            "event": ["a", "b", "c", "d"],
        },
    )


@pytest.fixture
def store(tmp_path) -> EventStore:
    path = tmp_path / "store"
    path.mkdir()
    store = EventStore(path, backend="ram")
    with store.open_writer() as writer:
        writer.write(_events())
    return store


def test_reader_returns_full_client_slice(store: EventStore) -> None:
    assert store.open_reader(128).read("c1")["event"].to_list() == ["a", "b", "c"]
    assert store.open_reader(128).read("c2")["event"].to_list() == ["d"]


def test_reader_unknown_client_raises(store: EventStore) -> None:
    with pytest.raises(KeyError, match="unknown"):
        store.open_reader(128).read("does-not-exist")


def test_writer_removes_path_on_exception(tmp_path) -> None:
    store = EventStore(tmp_path / "store", backend="ram")
    with pytest.raises(RuntimeError, match="boom"), store.open_writer() as writer:  # noqa: PT012
        writer.write(_events())
        raise RuntimeError("boom")
    assert not (tmp_path / "store").exists()


def test_default_path_is_tempdir() -> None:
    store = EventStore(backend="ram")
    assert store.path.is_dir()
    with store.open_writer() as writer:
        writer.write(_events())
    assert store.open_reader(2).read("c2")["event"].to_list() == ["d"]


def test_move_relocates_store(store: EventStore, tmp_path) -> None:
    previous = store.path
    destination = tmp_path / "moved"
    store.move(destination)
    assert store.path == destination
    assert not previous.exists()
    assert store.open_reader(2).read("c2")["event"].to_list() == ["d"]


def test_unknown_backend_raises(tmp_path) -> None:
    store = EventStore(tmp_path, backend="ram")
    store.backend = "bogus"
    with pytest.raises(RuntimeError, match="unknown backend"):
        store.open_writer()
    with pytest.raises(RuntimeError, match="unknown backend"):
        store.open_reader(2)


def test_boundary_reindexes_positions_before_applying_budget(tmp_path) -> None:
    events = _events().with_columns(position=pl.Series([7, 5, 1, 1]))
    store = EventStore(tmp_path / "store", backend="ram")
    with store.open_writer() as writer:
        writer.write(events)
    reader = store.open_reader(2)
    # Without a boundary the budget uses stored positions directly.
    assert reader.read("c1")["event"].to_list() == ["c"]
    boundary = dt.datetime(2024, 1, 3)
    bounded = reader.read("c1", before=boundary)
    assert bounded["event"].to_list() == ["b"]
    assert bounded["position"].to_list() == [1]
    boundaries = pl.Series([boundary, dt.datetime(2024, 1, 1)]).dt.epoch("ns").to_list()
    events, lengths = reader.read_batch(["c1", "c1"], boundaries)
    assert lengths == [1, 0]
    assert events["event"].to_list() == ["b"]
    assert events["position"].to_list() == [1]
