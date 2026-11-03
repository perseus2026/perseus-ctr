"""Direct cached batches preserve the scalar loader's inputs and fallback semantics."""

from decimal import Decimal
from unittest.mock import Mock

import polars as pl
import pytest
import torch

from perseus.core.dataloaders.samples import _collate_samples, _SamplesDataset, samples
from perseus.core.tasks.classification.target import Preprocessor
from tests.unit.core.dataloaders.test_batched_equivalence import _assert_equal, _batch, _fixture


def test_direct_batch_skips_event_dataframe_and_scalar_assembly(tmp_path, monkeypatch):
    reader, frame, events, context = _fixture(tmp_path, "cached", "ple3", False)
    indices = [4, 0, 4, 3, 1]
    expected = _batch(reader, frame[indices], events, context, True, batch_mode="off")
    loader = samples(
        frame[indices],
        reader,
        events,
        context,
        task_preprocessor=Preprocessor({"click": ["0", "1"], "buy": ["0", "1"]}),
        extras=["sample_key", "split"],
        fixed_padding=True,
        batch_size=5,
    )
    forbidden = Mock(side_effect=AssertionError("scalar assembly or DataFrame roundtrip"))
    monkeypatch.setattr(reader, "read_batch", forbidden)
    monkeypatch.setattr(_SamplesDataset, "__getitems__", forbidden)
    _assert_equal(next(iter(loader)), expected)
    forbidden.assert_not_called()


def test_direct_target_order_and_list_extras(tmp_path):
    reader, frame, events, context = _fixture(tmp_path, "cached", "id", False)
    frame = frame.with_columns(
        target=pl.Series([{"buy": index % 3, "click": index % 2} for index in range(len(frame))]),
        payload=pl.Series([[1, 2], [], [3], [4, 5, 6], []]),
    )
    task = Preprocessor({"click": ["0", "1"], "buy": ["0", "1", "2"]})
    indices = [4, 1, 4, 0]
    reference = _SamplesDataset(frame, reader, events, context, task, ["payload"])
    expected = _collate_samples([reference[index] for index in indices], task_preprocessor=task, min_event_length=2)
    loader = samples(frame, reader, events, context, task_preprocessor=task, extras=["payload"], fixed_padding=True)
    actual = loader.collate_fn(indices)
    torch.testing.assert_close(actual.target, expected.target, rtol=0, atol=0)
    assert actual.target.tolist() == [[0, 1], [1, 1], [0, 1], [0, 0]]
    for left, right in zip(actual.extras["payload"], expected.extras["payload"], strict=True):
        assert type(left) is type(right) is pl.Series
        assert left.equals(right)


@pytest.mark.parametrize("override", ["dynamic_transform", "collate"])
def test_custom_classification_keeps_general_path(tmp_path, override):
    class CustomTask(Preprocessor):
        pass

    if override == "dynamic_transform":
        CustomTask.dynamic_transform = lambda self, value: (value["buy"], value["click"])
    else:
        CustomTask.collate = lambda self, values: torch.tensor(values, dtype=torch.float64)
    reader, frame, events, context = _fixture(tmp_path, "cached", "id", False)
    task = CustomTask({"click": ["0", "1"], "buy": ["0", "1"]})
    loader = samples(frame, reader, events, context, task_preprocessor=task, batch_size=5)
    assert isinstance(loader.dataset, _SamplesDataset)
    reference = _SamplesDataset(frame, reader, events, context, task)
    expected = _collate_samples([reference[index] for index in range(5)], task_preprocessor=task)
    actual = next(iter(loader))
    torch.testing.assert_close(actual.target, expected.target, rtol=0, atol=0)


def test_unprepared_scalar_target_keeps_general_path(tmp_path):
    reader, frame, events, context = _fixture(tmp_path, "cached", "id", False)
    frame = frame.with_columns(target=pl.col("target").struct.field("click"))
    loader = samples(frame, reader, events, context, task_preprocessor=Preprocessor({"target": ["0", "1"]}))
    assert isinstance(loader.dataset, _SamplesDataset)


@pytest.mark.parametrize("null_kind", ["unknown_class", "null_struct"])
def test_direct_batch_rejects_null_targets(tmp_path, null_kind):
    reader, frame, events, context = _fixture(tmp_path, "cached", "id", False)
    task = Preprocessor({"click": ["0", "1"], "buy": ["0", "1"]})
    rows = [{"click": 0, "buy": 1}, {"click": 1, "buy": 0}, {"click": 1, "buy": 1}]
    rows[1] = {"click": 1, "buy": 2} if null_kind == "unknown_class" else None
    target = task.static_transform(pl.Series("target", rows))
    if null_kind == "null_struct":
        target = pl.Series("target", [target[0], None, target[2]], dtype=target.dtype)
    frame = frame.head(3).with_columns(target=target)
    loader = samples(frame, reader, events, context, task_preprocessor=task, batch_size=3)
    assert isinstance(loader.dataset, range)
    with pytest.raises(ValueError, match=r"classification targets.*null"):
        next(iter(loader))


@pytest.mark.parametrize(
    "prices",
    [
        pl.Series([Decimal("-1.25"), Decimal(0), Decimal("0.75"), Decimal(2), Decimal(5)], dtype=pl.Decimal(18, 2)),
        pl.Series([0, 1, 2**63, 2**63 + 1, 2**64 - 1], dtype=pl.UInt64),
    ],
)
def test_context_ple_casts_before_tensor_conversion(tmp_path, prices):
    reader, frame, events, context = _fixture(tmp_path, "cached", "id", False)
    frame = frame.with_columns(
        context=pl.DataFrame(
            {
                "city": frame["context"].struct.field("city"),
                "price": prices,
            },
        ).to_struct("context"),
    )
    expected = _batch(reader, frame, events, context, True, batch_mode="off")
    loader = samples(
        frame,
        reader,
        events,
        context,
        task_preprocessor=Preprocessor({"click": ["0", "1"], "buy": ["0", "1"]}),
        extras=["sample_key", "split"],
        fixed_padding=True,
        batch_size=len(frame),
    )
    assert isinstance(loader.dataset, range)
    _assert_equal(next(iter(loader)), expected)
