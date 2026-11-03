"""Tests for the cached event store backend: a tensor reader over the disk format."""

import datetime as dt

import numpy as np
import polars as pl
import pytest
import torch

from perseus.core.event_store import EventStore


def _events() -> pl.DataFrame:
    rng = np.random.default_rng(0)
    base = dt.datetime(2024, 1, 1)
    frames = []
    for client, n in zip(["c1", "c2", "c3"], [150, 3, 40], strict=False):
        frames.append(
            pl.DataFrame(
                {
                    "client_id": [client] * n,
                    "timestamp": [base + dt.timedelta(hours=int(h)) for h in np.sort(rng.integers(0, 24 * 60, n))],
                    "position": list(range(n, 0, -1)),
                    "event": pl.Series(rng.integers(1, 1000, n).tolist(), dtype=pl.UInt32()),
                },
            ),
        )
    frames.append(
        pl.DataFrame(
            {
                "client_id": ["c4"] * 6,
                "timestamp": [base + dt.timedelta(hours=h) for h in range(6)],
                "position": [5, 5, 5, 3, 3, 1],
                "event": pl.Series([1, 2, 3, 4, 5, 6], dtype=pl.UInt32()),
            },
        ),
    )
    frames.append(
        pl.DataFrame(
            {
                "client_id": ["c5"] * 4,
                "timestamp": [base + dt.timedelta(hours=h) for h in range(4)],
                "position": [10, 7, 3, 1],
                "event": pl.Series([7, 8, 9, 10], dtype=pl.UInt32()),
            },
        ),
    )
    return pl.concat(frames)


@pytest.fixture(scope="module")
def store(tmp_path_factory) -> EventStore:
    path = tmp_path_factory.mktemp("cached_store")
    store = EventStore(path / "store", backend="disk")
    with store.open_writer() as writer:
        writer.write(_events())
    return store


def _features(reader, client: str) -> dict[str, list]:
    events = reader.read(client)
    return {name: events[name].to_list() for name in events.columns if name != "client_id"}


def test_cached_features_match_disk_on_grid(store: EventStore) -> None:
    disk_reader = store.open_reader(128)
    cached_reader = EventStore(store.path, backend="cached").open_reader(128)
    for client in ["c1", "c2", "c3", "c4", "c5"]:
        expected = _features(disk_reader, client)
        got = _features(cached_reader, client)
        assert sorted(got) == sorted(expected)
        for name in expected:
            assert got[name] == expected[name], (client, name)


def test_cached_read_preserves_disk_schema(store: EventStore) -> None:
    disk_reader = store.open_reader(128)
    cached_reader = EventStore(store.path, backend="cached").open_reader(128)
    for client in ["c1", "c2", "c4"]:
        expected = disk_reader.read(client)
        actual = cached_reader.read(client)
        assert actual.schema == expected.schema
        assert actual.to_dict(as_series=False) == expected.to_dict(as_series=False)


def test_cached_windows_support_unsigned_event_ids(store: EventStore) -> None:
    reader = EventStore(store.path, backend="cached").open_reader(128)
    events, lengths = reader.read_batch(["c1"], [None], features=["event"])
    assert events["event"].dtype == pl.UInt32
    assert len(events) == lengths[0]


def test_tensor_windows_match_scalar_history_selection(store: EventStore) -> None:
    reader = EventStore(store.path, backend="cached", allow_empty_sequences=True).open_reader(3)
    clients = ["c4", "c2", "c4", "unknown", "c5"]
    boundary = int(pl.Series([dt.datetime(2024, 1, 1, 3)]).dt.epoch("ns").item())
    bounds = [None, boundary, boundary, boundary, None]
    expected = [reader.read(client, before=before) for client, before in zip(clients, bounds, strict=True)]
    columns, lengths = reader.read_batch_tensors(clients, bounds, features=["event"])
    assert lengths == [len(frame) for frame in expected]
    joined = pl.concat(expected)
    for name in ("position", "timestamp", "event"):
        values = joined[name].dt.epoch("ns") if name == "timestamp" else joined[name]
        torch.testing.assert_close(columns[name], values.to_torch(), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["torch"])
def test_cached_windows_support_timezone_aware_boundary(tmp_path, kind) -> None:
    events = pl.DataFrame({
        "client_id": ["a", "a"],
        "timestamp": pl.Series(
            [dt.datetime(2024, 1, 1, tzinfo=dt.UTC), dt.datetime(2024, 1, 2, tzinfo=dt.UTC)],
            dtype=pl.Datetime("ns", "UTC"),
        ),
        "position": pl.Series([2, 1], dtype=pl.Int64),
        "event": pl.Series([1, 2], dtype=pl.Int32),
    })
    store = EventStore(tmp_path / "store", backend="cached")
    with store.open_writer() as writer:
        writer.write(events)
    reader = store.open_reader(10)
    boundary = dt.datetime(2024, 1, 2, tzinfo=dt.UTC)
    expected = reader.read("a", before=boundary)
    boundary_ns = int(pl.Series([boundary]).dt.epoch("ns").item())
    events, _ = reader.read_batch(["a"], [boundary_ns])
    assert events["event"].to_list() == expected["event"].to_list()
    assert events["position"].to_list() == expected["position"].to_list()


@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
@pytest.mark.parametrize("boundary", [dt.date(2024, 1, 2), dt.datetime(2024, 1, 2, microsecond=500)])
def test_time_boundary_matches_original_polars_comparison(tmp_path, unit, boundary) -> None:
    events = pl.DataFrame({
        "client_id": ["a", "a", "a"],
        "timestamp": pl.Series(
            [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2), dt.datetime(2024, 1, 3)],
            dtype=pl.Datetime(unit),
        ),
        "position": pl.Series([3, 2, 1], dtype=pl.Int64),
        "event": pl.Series([1, 2, 3], dtype=pl.Int32),
    })
    store = EventStore(tmp_path / "store", backend="disk")
    with store.open_writer() as writer:
        writer.write(events)
    expected = events.filter(pl.col("timestamp") < boundary)["event"].to_list()
    for backend in ("disk", "cached"):
        reader = EventStore(store.path, backend=backend).open_reader(10)
        assert reader.read("a", before=boundary)["event"].to_list() == expected
        timestamp_dtype = reader.read("a").schema["timestamp"]
        boundary_ns = int(pl.Series([boundary]).cast(timestamp_dtype).dt.epoch("ns").item())
        batch_events, _ = reader.read_batch(["a"], [boundary_ns])
        assert batch_events["event"].to_list() == expected


def test_cached_unknown_client_raises(store: EventStore) -> None:
    reader = EventStore(store.path, backend="cached").open_reader(128)
    with pytest.raises(KeyError, match="unknown"):
        reader.read("nope")


def test_cached_rejects_non_numeric_columns(tmp_path) -> None:
    path = tmp_path / "string_store"
    store = EventStore(path, backend="disk")
    with store.open_writer() as writer:
        writer.write(
            pl.DataFrame(
                {
                    "client_id": ["c1"],
                    "timestamp": [dt.datetime(2024, 1, 1)],
                    "position": [1],
                    "event": ["click"],
                },
            ),
        )
    with pytest.raises(ValueError, match="numeric"):
        EventStore(path, backend="cached").open_reader(128)


def test_cached_holed_and_duplicated_positions_survive(store: EventStore) -> None:
    reader = EventStore(store.path, backend="cached").open_reader(128)
    assert reader.read("c4")["position"].to_list() == [5, 5, 5, 3, 3, 1]
    assert reader.read("c5")["position"].to_list() == [10, 7, 3, 1]


def _sharing_available() -> bool:
    try:
        torch.empty(1).share_memory_()
    except RuntimeError:
        return False
    return True


@pytest.mark.skipif(not _sharing_available(), reason="torch shared memory unavailable")
def test_cached_reader_tensors_are_shared(store: EventStore) -> None:
    reader = EventStore(store.path, backend="cached").open_reader(128)
    assert reader._timestamps.is_shared()
    assert reader._positions.is_shared()
    assert all(column.is_shared() for column in reader._columns.values())


def _bump_first_event(reader, queue) -> None:
    reader._columns["event"][0] += 1
    queue.put(int(reader._columns["event"][0]))


@pytest.mark.skipif(not _sharing_available(), reason="torch shared memory unavailable")
def test_cached_reader_memory_is_shared_across_processes(store: EventStore) -> None:
    import torch.multiprocessing  # noqa: PLC0415

    reader = EventStore(store.path, backend="cached").open_reader(128)
    before = int(reader._columns["event"][0])
    ctx = torch.multiprocessing.get_context("spawn")
    try:
        queue = ctx.Queue()
    except (ImportError, OSError):
        pytest.skip("semaphores unavailable")
    process = ctx.Process(target=_bump_first_event, args=(reader, queue))
    process.start()
    assert queue.get(timeout=120) == before + 1
    process.join(timeout=120)
    assert int(reader._columns["event"][0]) == before + 1
