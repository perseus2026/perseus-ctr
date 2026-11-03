import datetime as dt

import polars as pl
import pytest

from perseus.core.event_store import EventStore


@pytest.mark.parametrize("backend", ["disk", "ram", "cached"])
def test_missing_client_is_empty_only_when_enabled(tmp_path, backend):
    events = pl.DataFrame(
        {
            "client_id": ["known", "known"],
            "timestamp": [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)],
            "position": pl.Series([2, 1], dtype=pl.Int64),
            "item": pl.Series([3, 4], dtype=pl.UInt32),
        },
    )
    store = EventStore(tmp_path / "events", backend=backend, allow_empty_sequences=True)
    with store.open_writer() as writer:
        writer.write(events)
    reader = store.open_reader(2)
    assert reader.read("missing").schema == reader.read("known").schema
    assert reader.read("missing").is_empty()
    events, lengths = reader.read_batch(["missing", "known", "missing"], [None, None, None])
    assert lengths == [0, 2, 0]
    assert events["position"].to_list() == [2, 1]
    assert events["item"].to_list() == [3, 4]
    strict = EventStore(store.path, backend=backend).open_reader(2)
    with pytest.raises(KeyError):
        strict.read("missing")


@pytest.mark.parametrize("backend", ["disk", "ram", "cached"])
def test_all_empty_store_keeps_schema(tmp_path, backend):
    events = pl.DataFrame(
        schema={
            "client_id": pl.String,
            "timestamp": pl.Datetime("ns"),
            "position": pl.Int64,
            "item": pl.UInt32,
        },
    )
    store = EventStore(tmp_path / "events", backend=backend, allow_empty_sequences=True)
    with store.open_writer() as writer:
        writer.write(events)
    reader = store.open_reader(2)
    assert reader.read("missing").schema == events.schema
    _, lengths = reader.read_batch(["missing"], [None])
    assert lengths == [0]
