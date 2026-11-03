"""Probability mode applies sigmoid inside AMP forward and uses FP32 BCE exactly once."""

import copy
import json

import pytest
import torch
from torch.nn import functional

from perseus.core.tasks.classification.target import Criterion, Head, Layer


@pytest.mark.parametrize("binary_output", ["logits", "probabilities"])
@pytest.mark.parametrize("reduction", ["mean", "sum"])
@pytest.mark.parametrize("bf16", [False, True])
def test_loss_and_gradients_match_explicit_binary_and_categorical_computation(binary_output, reduction, bf16):
    layer = Layer(8, [2, 3], hidden=[4], binary_output=binary_output)
    reference = copy.deepcopy(layer)
    values = torch.randn(3, 2, 8, requires_grad=True)
    reference_values = values.detach().clone().requires_grad_()
    targets = torch.tensor([[0, 2], [1, 0], [1, 1]])
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        binary = reference.layers[0](reference_values[:, 0])
        category = reference.layers[1](reference_values[:, 1])
        if binary_output == "probabilities":
            binary = binary.sigmoid()
        category_loss = functional.cross_entropy(category.float(), targets[:, 1])
    if binary_output == "probabilities":
        binary_loss = functional.binary_cross_entropy(binary.float()[:, 0], targets[:, 0].float())
    else:
        binary_loss = functional.binary_cross_entropy_with_logits(binary.float()[:, 0], targets[:, 0].float())
    expected = getattr(torch.stack([binary_loss, category_loss]), reduction)()
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        actual = Criterion(layer, task_reduction=reduction)(values, None, targets)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    torch.testing.assert_close(values.grad, reference_values.grad, rtol=0, atol=0)
    for left, right in zip(layer.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)


def test_probability_head_does_not_apply_sigmoid_twice_and_survives_checkpoint(tmp_path):
    layer = Layer(4, [2, 3], binary_output="probabilities")
    head = Head(layer, {"click": ["0", "1"], "category": ["a", "b", "c"]})
    values = torch.randn(3, 2, 4)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        binary = layer.layers[0](values[:, 0]).sigmoid()
        assert binary.dtype == torch.bfloat16
        expected = binary.float()
        assert not torch.equal(expected, layer.layers[0](values[:, 0]).float().sigmoid())
        prediction = head.predict(values, None, {})
    torch.testing.assert_close(
        prediction.struct.field("click").struct.field("1").to_torch(),
        expected[:, 0],
        rtol=0,
        atol=0,
    )
    head.save(tmp_path)
    loaded = Head.load(tmp_path)
    assert loaded.layer.binary_output == "probabilities"
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert loaded.predict(values, None, {}).equals(prediction)


def test_old_layer_checkpoint_defaults_to_logits(tmp_path):
    layer = Layer(4, [2])
    layer.save(tmp_path)
    path = tmp_path / "config.json"
    config = json.loads(path.read_text())
    config["kwargs"].pop("binary_output")
    path.write_text(json.dumps(config))
    loaded = Layer.load(tmp_path)
    assert loaded.binary_output == "logits"
    values = torch.randn(3, 4)
    torch.testing.assert_close(loaded(values, None)[0], layer.layers[0](values), rtol=0, atol=0)


def test_invalid_output_mode_is_rejected():
    with pytest.raises(ValueError, match="binary_output must be logits or probabilities"):
        Layer(4, [2], binary_output="invalid")
