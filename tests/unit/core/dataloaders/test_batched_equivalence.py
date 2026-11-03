"""Public batch-mode parity, including transformed padding and exact ns cutoffs."""

import copy
import datetime as dt
import importlib
from unittest.mock import Mock

import polars as pl
import pytest
import torch

from perseus.core.dataloaders.samples import _collate_samples, _SamplesDataset
from perseus.core.encoders.base import Preprocessor as EncoderPreprocessor
from perseus.core.encoders.bow import Preprocessor as BowPreprocessor
from perseus.core.encoders.id import Preprocessor as IdPreprocessor
from perseus.core.encoders.ple import Preprocessor as PlePreprocessor
from perseus.core.event_store import EventStore
from perseus.core.tasks.classification.target import Preprocessor as TargetPreprocessor

samples_api = importlib.import_module("perseus.core.dataloaders.samples").samples
ORIGIN = 1_700_000_000_000_000_000


@pytest.mark.parametrize("mode", ["torch"])
def test_disk_batch_ignores_unused_string_features(tmp_path, mode):
    events = pl.DataFrame({
        "client_id": ["a", "a"],
        "timestamp": pl.Series([ORIGIN + 100, ORIGIN + 200]).cast(pl.Datetime("ns")),
        "position": pl.Series([2, 1], dtype=pl.Int64),
        "value": pl.Series([1, 2], dtype=pl.Int32),
        "unused": ["first", "second"],
    })
    store = EventStore(tmp_path / "events", backend="disk")
    with store.open_writer() as writer:
        writer.write(events)
    reader = store.open_reader(2)
    samples = pl.DataFrame({
        "client_id": ["a"],
        "timestamp": pl.Series([ORIGIN + 201]).cast(pl.Datetime("ns")),
    })
    preprocessor = IdPreprocessor(pl.DataFrame({"value": ["unused"], "index": [1]}))
    loader = samples_api(
        samples, reader, {"value": preprocessor}, {}, fixed_padding=False, batch_size=1,
    )
    batch = next(iter(loader))
    assert batch.events_features["value"].tolist() == [[1, 2]]
    assert batch.events_positions.tolist() == [[2, 1]]


def _fixture(tmp_path, backend, encoder, cold):
    # Positions count recency levels, not rows: the last three rows fit budget2.
    events = pl.DataFrame({
        "client_id": ["a"] * 5 + ["b"],
        "timestamp": pl.Series([ORIGIN + n for n in (100, 200, 300, 400, 500, 100)], dtype=pl.Int64)
        .cast(pl.Datetime("ns")),
        "position": pl.Series([5, 4, 2, 2, 1, 1], dtype=pl.Int64),
        "value": pl.Series([11, 12, 13, 14, 15, 21], dtype=pl.Int32),
    })
    store = EventStore(tmp_path / backend, backend=backend)
    with store.open_writer() as writer:
        writer.write(events)
    reader = store.open_reader(2)
    clients = ["a", "a", "a", "b", "a"]
    offsets = [100] * 5 if cold else [300, 301, 601, 100, 601]
    frame = pl.DataFrame({
        "client_id": clients,
        "timestamp": pl.Series([ORIGIN + n for n in offsets], dtype=pl.Int64).cast(pl.Datetime("ns")),
        "context": [{"city": n + 1, "price": float(n) - 1.5} for n in range(5)],
        "target": [{"click": n % 2, "buy": (n + 1) % 2} for n in range(5)],
        "sample_key": ["s0", "s1", "s2", "s3", "s4"],
        "split": ["train", "valid", "test", "test", "train"],
    })
    identity = IdPreprocessor(pl.DataFrame({"value": ["unused"], "index": [1]}))
    pre = identity if encoder == "id" else PlePreprocessor(
        [-2.0, -1.0] if encoder == "ple1" else [-2.0, -1.0, 1.0, 30.0],
    )
    return reader, frame, {"value": pre}, {"city": identity, "price": PlePreprocessor([-2.0, 0.0, 3.0])}


def _batch(reader, frame, events, context, fixed, **mode):
    loader = samples_api(
        frame, reader, events, context,
        task_preprocessor=TargetPreprocessor({"click": ["0", "1"], "buy": ["0", "1"]}),
        extras=["sample_key", "split"], fixed_padding=fixed,
        batch_size=len(frame), num_workers=0, shuffle=False,
    )
    if mode.get("batch_mode") == "off":
        dataset = _SamplesDataset(frame, reader, events, context,
                                  TargetPreprocessor({"click": ["0", "1"], "buy": ["0", "1"]}),
                                  ["sample_key", "split"])
        return _collate_samples(
            [dataset[i] for i in range(len(frame))], task_preprocessor=dataset.task_preprocessor,
            min_event_length=reader.max_events_per_sequence if fixed else int(reader.allow_empty_sequences),
        )
    return next(iter(loader))


def _assert_equal(actual, expected):
    for field in ("events_features", "context_features"):
        assert getattr(actual, field).keys() == getattr(expected, field).keys()
        for name, value in getattr(expected, field).items():
            other = getattr(actual, field)[name]
            assert other.dtype == value.dtype, (field, name)
            assert other.shape == value.shape, (field, name)
            torch.testing.assert_close(other, value, rtol=0, atol=0)
    for field in ("events_positions", "events_timestamps", "sample_timestamp", "target"):
        other, value = getattr(actual, field), getattr(expected, field)
        assert other.dtype == value.dtype, field
        assert other.shape == value.shape, field
        torch.testing.assert_close(other, value, rtol=0, atol=0)
    assert actual.extras == expected.extras


def test_getitems_reads_once_and_transforms_each_unpadded_history(tmp_path, monkeypatch):
    reader, frame, _, context = _fixture(tmp_path, "cached", "id", False)

    class RecordingPreprocessor:
        dynamic_transform_batch = EncoderPreprocessor.dynamic_transform_batch

        def __init__(self):
            self.calls = []

        def dynamic_transform(self, values):
            self.calls.append((values.name, values.dtype, values.to_list()))
            return values.to_torch().int()

    preprocessor = RecordingPreprocessor()
    loader = samples_api(
        frame, reader, {"value": preprocessor}, context,
        task_preprocessor=TargetPreprocessor({"click": ["0", "1"], "buy": ["0", "1"]}),
        extras=["sample_key", "split"], fixed_padding=True, batch_size=5,
    )
    dataset = loader.dataset
    indices = [4, 0, 4, 3, 1]
    expected = [dataset[index] for index in indices]
    expected_calls = preprocessor.calls.copy()
    preprocessor.calls.clear()

    original = reader.read_batch
    calls = []

    def read_batch(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(reader, "read_batch", read_batch)
    actual = dataset.__getitems__(indices)
    assert len(calls) == 1
    assert preprocessor.calls == expected_calls
    collate = loader.collate_fn
    _assert_equal(collate(actual), collate(expected))
    calls.clear()
    preprocessor.calls.clear()
    next(iter(loader))
    assert len(calls) == 1
    assert len(preprocessor.calls) == len(frame)


def test_bow_empty_history_has_valid_shape():
    preprocessor = BowPreprocessor(None)
    values = pl.Series("value", [], dtype=pl.List(pl.UInt32))
    result = preprocessor.dynamic_transform(values)
    assert result.shape == (0, 1)
    assert result.dtype == torch.int32


def test_bow_empty_history_through_dataloader(tmp_path):
    events = pl.DataFrame({
        "client_id": ["a"],
        "timestamp": pl.Series([ORIGIN + 100]).cast(pl.Datetime("ns")),
        "position": pl.Series([1], dtype=pl.Int64),
        "value": pl.Series([[4, 5]], dtype=pl.List(pl.UInt32)),
    })
    store = EventStore(tmp_path / "bow", backend="disk")
    with store.open_writer() as writer:
        writer.write(events)
    samples = pl.DataFrame({
        "client_id": ["a", "a"],
        "timestamp": pl.Series([ORIGIN + 100, ORIGIN + 101]).cast(pl.Datetime("ns")),
    })
    loader = samples_api(
        samples, store.open_reader(2), {"value": BowPreprocessor(None)}, {},
        batch_size=2, fixed_padding=False,
    )
    batch = next(iter(loader))
    assert batch.events_features["value"].tolist() == [[[0, 0]], [[4, 5]]]
    assert batch.events_positions.tolist() == [[0], [1]]


def test_batched_fetch_matches_scalar_training_step(tmp_path):
    reader, frame, events, context = _fixture(tmp_path, "cached", "id", False)
    batches = [
        _batch(reader, frame, events, context, True, batch_mode="off"),
        _batch(reader, frame, events, context, True),
    ]
    torch.manual_seed(12)
    original = torch.nn.Sequential(torch.nn.Embedding(32, 4, padding_idx=0), torch.nn.Flatten(), torch.nn.Linear(12, 1))
    other = copy.deepcopy(original)
    for model, batch in zip((original, other), batches, strict=True):
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        logits = model(batch.events_features["value"].long())
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch.target[:, :1].float())
        loss.backward()
        optimizer.step()
    for left, right in zip(original.parameters(), other.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_id_subclass_keeps_per_history_transform():
    class CenteredId(IdPreprocessor):
        def dynamic_transform(self, values):
            result = super().dynamic_transform(values)
            return result - result.min() if result.numel() else result

    preprocessor = CenteredId(pl.DataFrame({"value": ["unused"], "index": [1]}))
    values = pl.Series([1, 3, 10, 14], dtype=pl.Int32)
    result = preprocessor.dynamic_transform_batch(values, [2, 0, 2])
    assert [row.tolist() for row in result] == [[0, 2], [], [0, 4]]


def test_context_id_column_is_transformed_once_per_batch(tmp_path, monkeypatch):
    reader, frame, events, context = _fixture(tmp_path, "cached", "id", False)
    context["city"] = IdPreprocessor(context["city"].values)
    loader = samples_api(
        frame, reader, events, context,
        task_preprocessor=TargetPreprocessor({"click": ["0", "1"], "buy": ["0", "1"]}),
        extras=["sample_key", "split"], fixed_padding=True, batch_size=5,
    )
    indices = [4, 0, 4, 3, 1]
    expected = _batch(reader, frame[indices], events, context, True, batch_mode="off")
    transform = Mock(wraps=context["city"].dynamic_transform_tensor)
    monkeypatch.setattr(context["city"], "dynamic_transform_tensor", transform)
    actual = loader.collate_fn(indices)
    _assert_equal(actual, expected)
    assert [call.args[0].tolist() for call in transform.call_args_list] == [[5, 1, 5, 4, 2]]


def test_context_custom_transform_keeps_single_sample_semantics(tmp_path):
    class CenteredId(IdPreprocessor):
        def dynamic_transform(self, values):
            result = super().dynamic_transform(values)
            return result - result.min()

    reader, frame, events, context = _fixture(tmp_path, "cached", "id", False)
    context["city"] = CenteredId(pl.DataFrame({"value": ["unused"], "index": [1]}))
    expected = _batch(reader, frame, events, context, True, batch_mode="off")
    actual = _batch(reader, frame, events, context, True)
    _assert_equal(actual, expected)
    assert actual.context_features["city"].tolist() == [0] * len(frame)


@pytest.mark.parametrize("targets", [[0.5, 1.0, 2.0], [{"a": 1}, {"a": 2}, {"a": 3}], [[1, 2], [], [3]]])
def test_batch_preserves_target_and_extra_scalar_types(tmp_path, targets):
    class Task:
        def dynamic_transform(self, value):
            return value

    reader, frame, events, context = _fixture(tmp_path, "cached", "id", False)
    frame = frame.head(3).with_columns(target=pl.Series(targets), payload=pl.Series([[1, 2], [], [3]]))
    loader = samples_api(frame, reader, events, context, task_preprocessor=Task(), extras=["payload"])
    indices = [2, 0, 2, 1]
    expected = [loader.dataset[index] for index in indices]
    actual = loader.dataset.__getitems__(indices)
    for left, right in zip(actual, expected, strict=True):
        assert type(left.target) is type(right.target)
        if isinstance(left.target, pl.Series):
            assert left.target.equals(right.target)
        else:
            assert left.target == right.target
        assert left.extras["payload"].equals(right.extras["payload"])


def test_default_padding_matches_original_perseus(tmp_path):
    reader, frame, events, context = _fixture(tmp_path, "cached", "id", False)
    loader = samples_api(frame[1:2], reader, events, context, batch_size=1)
    assert next(iter(loader)).events_positions.shape == (1, 1)


@pytest.mark.parametrize("backend", ["cached", "disk"])
@pytest.mark.parametrize("mode", ["torch"])
@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("encoder", ["id", "ple1", "ple3"])
@pytest.mark.parametrize("cold", [False, True])
def test_batch_mode_matches_off(tmp_path, backend, mode, fixed, encoder, cold):
    reader, frame, events, context = _fixture(tmp_path, backend, encoder, cold)
    expected = _batch(reader, frame, events, context, fixed, batch_mode="off")
    actual = _batch(reader, frame, events, context, fixed)
    _assert_equal(actual, expected)
    # Independent source-order checks: off-mode must also preserve ns cutoffs.
    counts = actual.events_positions.ne(0).sum(1).tolist()
    assert counts == ([0] * 5 if cold else [2, 1, 3, 0, 3])
    width = (2 if fixed else 0) if cold else 3
    assert actual.events_positions.shape == (5, width)
    if not cold:
        assert actual.events_positions[2].tolist() == [2, 2, 1]
        assert actual.extras["sample_key"] == ["s0", "s1", "s2", "s3", "s4"]
    values = actual.events_features["value"]
    assert values.shape == ((5, width) if encoder == "id" else (5, width, 1 if encoder == "ple1" else 3))
    # Raw zero maps to a nonzero PLE code; padded entries must still be zero.
    if encoder != "id":
        assert events["value"].dynamic_transform(pl.Series([0.0])).ne(0).any()
    assert torch.equal(values[actual.events_positions == 0], torch.zeros_like(values[actual.events_positions == 0]))




@pytest.mark.parametrize("mode", ["torch"])
def test_bow_uses_safe_per_sample_fallback(tmp_path, monkeypatch, mode):
    # Prepared BOW values are token lists. Cached storage only supports numeric
    # scalar columns, so this exercises the supported disk reader instead.
    events = pl.DataFrame({
        "client_id": ["a", "a", "a"],
        "timestamp": pl.Series([ORIGIN + n for n in (100, 200, 300)], dtype=pl.Int64)
        .cast(pl.Datetime("ns")),
        "position": pl.Series([3, 2, 1], dtype=pl.Int64),
        "value": pl.Series([[1, 2], [3], [4, 5, 6]], dtype=pl.List(pl.UInt32)),
    })
    store = EventStore(tmp_path / "bow", backend="disk")
    with store.open_writer() as writer:
        writer.write(events)
    reader = store.open_reader(2)
    frame = pl.DataFrame({
        "client_id": ["a", "a"],
        "timestamp": pl.Series([ORIGIN + 201, ORIGIN + 301], dtype=pl.Int64).cast(pl.Datetime("ns")),
        "target": [{"click": 0, "buy": 1}, {"click": 1, "buy": 0}],
        "sample_key": ["one", "two"],
        "split": ["test", "test"],
    })
    # Tokenizer is used by static_transform only; these values are prepared.
    pre = BowPreprocessor(None)
    expected = _batch(reader, frame, {"value": pre}, {}, False, batch_mode="off")
    actual = _batch(reader, frame, {"value": pre}, {}, False)
    # Context is intentionally absent in this fallback test.
    assert actual.context_features is expected.context_features is None
    for name in actual.events_features:
        assert torch.equal(actual.events_features[name], expected.events_features[name])
    assert torch.equal(actual.events_positions, expected.events_positions)
    assert torch.equal(actual.target, expected.target)
    assert actual.extras == expected.extras


@pytest.mark.parametrize("backend", ["cached", "disk"])
@pytest.mark.parametrize("mode", ["torch"])
def test_reader_none_boundary_nonmonotonic_positions_and_float64(tmp_path, backend, mode):
    events = pl.DataFrame({
        "client_id": ["a"] * 5,
        "timestamp": pl.Series([ORIGIN + n for n in (100, 200, 300, 400, 500)], dtype=pl.Int64)
        .cast(pl.Datetime("ns")),
        "position": pl.Series([3, 9, 2, 2, 5], dtype=pl.Int64),
        # A Float32 intermediate would erase these increments.
        "precision": pl.Series([1.0 + n * 2**-40 for n in range(5)], dtype=pl.Float64),
    })
    store = EventStore(tmp_path / backend, backend=backend)
    with store.open_writer() as writer:
        writer.write(events)
    reader = store.open_reader(2)
    bounds = [None, ORIGIN + 601, ORIGIN + 100, ORIGIN + 301, None]
    frames = [reader.read("a", before=bound) for bound in bounds]
    expected_counts = [2, 3, 0, 2, 2]
    assert [len(frame) for frame in frames] == expected_counts
    events, lengths = reader.read_batch(["a"] * len(bounds), bounds)
    assert lengths == expected_counts
    assert events["precision"].dtype == pl.Float64
    assert events.equals(pl.concat([frame.select("position", "timestamp", "precision") for frame in frames]))


@pytest.mark.parametrize("mode", ["torch"])
def test_batch_mode_in_spawn_worker(tmp_path, mode):
    reader, frame, events, context = _fixture(tmp_path, "cached", "ple3", False)
    expected = _batch(reader, frame, events, context, True, batch_mode="off")
    loader = samples_api(
        frame, reader, events, context,
        task_preprocessor=TargetPreprocessor({"click": ["0", "1"], "buy": ["0", "1"]}),
        extras=["sample_key", "split"], fixed_padding=True,
        batch_size=len(frame), num_workers=1, multiprocessing_context="spawn", shuffle=False,
    )
    _assert_equal(next(iter(loader)), expected)


def test_timestamp_dtype_is_independent_of_default_float(tmp_path):
    reader, frame, events, context = _fixture(tmp_path, "cached", "id", False)
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        expected = _batch(reader, frame, events, context, False, batch_mode="off")
        actual = _batch(reader, frame, events, context, False)
        assert expected.sample_timestamp.dtype == torch.float32
        _assert_equal(actual, expected)
    finally:
        torch.set_default_dtype(original)


@pytest.mark.parametrize("mode", ["torch"])
def test_timezone_boundary_preserves_read_contract(tmp_path, mode):
    reader, _, _, _ = _fixture(tmp_path, "cached", "id", False)
    boundary = dt.datetime.fromtimestamp(ORIGIN // 10**9, tz=dt.UTC) + dt.timedelta(microseconds=1)
    with pytest.raises(pl.exceptions.SchemaError):
        reader.read("a", before=boundary)
    boundary_ns = int(pl.Series([boundary]).dt.epoch("ns").item())
    events, _ = reader.read_batch(["a"], [boundary_ns])
    assert events.height == 3
    assert events["timestamp"].dt.epoch("ns").max() < boundary_ns
