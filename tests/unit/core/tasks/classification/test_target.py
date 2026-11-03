"""Classification encoding, independent task heads and checkpoint contracts."""

import polars as pl
import pytest
import torch
from torch.nn import functional

from perseus.core.tasks.classification.target import Artifacts, Criterion, Head, Layer, Preprocessor


def fit(values):
    return Preprocessor.fit(values, Artifacts(), {})


def batch(processor, values):
    encoded = processor.static_transform(values)
    return processor.collate([processor.dynamic_transform(value) for value in encoded.to_list()])


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (pl.Series([30, 10, 20]), [2, 0, 1]),
        (pl.Series(["c", "a", "b"]), [2, 0, 1]),
        (pl.Series([2**40 + 1, 2**40]), [1, 0]),
        (pl.Series([True, False]), [1, 0]),
        (pl.Series([1e-7, 1e-6]), [0, 1]),
    ],
)
def test_static_transform_encodes_once(values, expected, tmp_path):
    processor = fit(values)
    transformed = processor.static_transform(values)
    assert transformed.to_list() == [{"target": value} for value in expected]
    assert transformed.dtype == pl.Struct({"target": pl.UInt32})
    assert processor.dynamic_transform(transformed[0]) == (expected[0],)
    processor.save(tmp_path)
    assert Preprocessor.load(tmp_path).static_transform(values).to_list() == transformed.to_list()


def test_struct_encoding_preserves_fitted_field_order():
    values = pl.Series(
        [{"binary": "yes", "category": 30}, {"binary": "no", "category": 10}, {"binary": "yes", "category": 20}],
    )
    processor = fit(values)
    assert list(processor.classes) == ["binary", "category"]
    assert processor.static_transform(values).dtype == pl.Struct({"binary": pl.UInt32, "category": pl.UInt32})
    assert processor.static_transform(values).to_list() == [
        {"binary": 1, "category": 2},
        {"binary": 0, "category": 0},
        {"binary": 1, "category": 1},
    ]
    assert processor.dynamic_transform({"category": 2, "binary": 1}) == (1, 2)
    targets = batch(processor, values)
    assert targets.dtype == torch.long
    torch.testing.assert_close(targets, torch.tensor([[1, 2], [0, 0], [1, 1]]))


@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("hidden", [None, [16, 8]])
def test_mixed_heads_use_independent_bce_and_ce_with_gradients(reduction, hidden):
    values = pl.Series([{"click": 0, "category": "a"}, {"click": 1, "category": "b"}, {"click": 1, "category": "c"}])
    processor = fit(values)
    layer = Layer.init(processor, 32, hidden=hidden)
    inputs = torch.randn(3, 2, 32, requires_grad=True)
    targets = batch(processor, values)
    logits = layer(inputs, None)
    assert [part.shape for part in logits] == [(3, 1), (3, 3)]
    expected_logits = [layer.layers[i](inputs[:, i]) for i in range(2)]
    torch.testing.assert_close(logits, expected_logits)
    losses = torch.stack(
        [
            functional.binary_cross_entropy_with_logits(logits[0][:, 0], targets[:, 0].float()),
            functional.cross_entropy(logits[1], targets[:, 1]),
        ],
    )
    loss = Criterion.init(processor, layer, task_reduction=reduction)(inputs, None, targets)
    torch.testing.assert_close(loss, getattr(losses, reduction)())
    loss.backward()
    assert (inputs.grad.abs().sum(dim=(0, 2)) > 0).all()
    assert all(parameter.grad is not None for parameter in layer.parameters())


@pytest.mark.parametrize(("labels", "width"), [(["a"], 1), (["a", "b"], 1), (["a", "b", "c"], 3)])
def test_single_target_head_probabilities_and_loss(labels, width):
    values = pl.Series(labels)
    processor = fit(values)
    layer = Layer.init(processor, 4)
    inputs = torch.randn(len(labels), 4)
    assert [part.shape for part in layer(inputs, None)] == [(len(labels), width)]
    loss = Criterion.init(processor, layer)(inputs, None, batch(processor, values))
    assert torch.isfinite(loss)
    prediction = Head.init(processor, layer).predict(inputs, None, {})
    assert prediction.dtype == pl.Struct({"target": pl.Struct(dict.fromkeys(labels, pl.Float32))})
    assert all(sum(row["target"].values()) == pytest.approx(1) for row in prediction.to_list())
    if len(labels) == 1:
        assert loss.item() == 0


@pytest.mark.parametrize("hidden", [None, [16, 8]])
def test_nested_head_probabilities_checkpoint_and_empty_batch(tmp_path, hidden):
    values = pl.Series([{"click": 0, "category": "a"}, {"click": 1, "category": "b"}, {"click": 1, "category": "c"}])
    processor = fit(values)
    head = Head.init(processor, Layer.init(processor, 4, hidden=hidden))
    inputs = torch.randn(3, 2, 4)
    expected = head.predict(inputs, None, {})
    assert expected.dtype == pl.Struct(
        {
            "click": pl.Struct({"0": pl.Float32, "1": pl.Float32}),
            "category": pl.Struct({"a": pl.Float32, "b": pl.Float32, "c": pl.Float32}),
        },
    )
    assert head.predict(torch.empty(0, 2, 4), None, {}).dtype == expected.dtype
    head.save(tmp_path)
    loaded = Head.load(tmp_path)
    assert loaded.classes == head.classes
    assert loaded.predict(inputs, None, {}).to_list() == expected.to_list()


def test_bfloat16_loss_is_float32():
    values = pl.Series([{"a": 0, "b": 1}, {"a": 1, "b": 0}])
    processor = fit(values)
    layer = Layer.init(processor, 4)
    inputs = torch.randn(2, 2, 4, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = Criterion.init(processor, layer)(inputs, None, batch(processor, values))
    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(inputs.grad).all()


def test_bfloat16_categorical_loss_is_float32():
    values = pl.Series(["a", "b", "c", "a"])
    processor = fit(values)
    layer = Layer.init(processor, 4)
    inputs = torch.randn(4, 4, requires_grad=True)
    targets = batch(processor, values)

    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = Criterion.init(processor, layer)(inputs, None, targets)

    assert loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(inputs.grad).all()


def test_invalid_state_width_and_reduction():
    processor = fit(pl.Series([{"a": 0, "b": 1}, {"a": 1, "b": 0}]))
    layer = Layer.init(processor, 4)
    with pytest.raises(ValueError, match="task states"):
        layer(torch.randn(2, 3, 4), None)
    with pytest.raises(ValueError, match="target width"):
        Criterion.init(processor, layer)(torch.randn(2, 4), None, torch.zeros(2, 3))
    with pytest.raises(ValueError, match="task_reduction"):
        Criterion.init(processor, layer, task_reduction="invalid")
