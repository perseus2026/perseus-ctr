"""Independent classification layers for each target field."""

import json

import pytest
import torch

from perseus.core.tasks.classification.target import Layer


@pytest.mark.parametrize("hidden", [None, [], [16], [64, 16]])
def test_targets_have_independent_layers_and_class_widths(hidden) -> None:
    layer = Layer(32, [2, 3], hidden=hidden)
    inputs = torch.randn(3, 32, requires_grad=True)

    logits = layer(inputs, None)

    assert len(layer.layers) == 2
    assert {id(p) for p in layer.layers[0].parameters()}.isdisjoint(id(p) for p in layer.layers[1].parameters())
    assert [part.shape for part in logits] == [(3, 1), (3, 3)]
    torch.testing.assert_close(logits[0], layer.layers[0](inputs))
    torch.testing.assert_close(logits[1], layer.layers[1](inputs))
    sum(part.sum() for part in logits).backward()
    assert all(parameter.grad is not None for target_layer in layer.layers for parameter in target_layer.parameters())


@pytest.mark.parametrize("hidden", [None, [64, 16]])
def test_targets_bind_to_matching_backbone_states(hidden) -> None:
    layer = Layer(32, [2, 3], hidden=hidden)
    states = torch.randn(3, 2, 32, requires_grad=True)

    logits = layer(states, None)

    torch.testing.assert_close(logits[0], layer.layers[0](states[:, 0]))
    torch.testing.assert_close(logits[1], layer.layers[1](states[:, 1]))
    logits[0].sum().backward()
    assert torch.count_nonzero(states.grad[:, 1]) == 0
    assert all(parameter.grad is None for parameter in layer.layers[1].parameters())


def test_backbone_state_count_matches_target_count() -> None:
    layer = Layer(4, [2, 3])
    with pytest.raises(ValueError, match="3 task states for 2 targets"):
        layer(torch.randn(3, 3, 4), None)


@pytest.mark.parametrize("hidden", [None, [], [16, 8]])
def test_target_layers_checkpoint_roundtrip(tmp_path, hidden) -> None:
    layer = Layer(4, [2, 3], hidden=hidden).eval()
    inputs = torch.randn(3, 2, 4)
    expected = layer(inputs, None)

    layer.save(tmp_path)
    loaded = Layer.load(tmp_path)

    torch.testing.assert_close(loaded(inputs, None), expected)


def test_load_linear_checkpoint_without_hidden_option(tmp_path) -> None:
    layer = Layer(4, [2, 3])
    layer.save(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["kwargs"].pop("hidden")
    config_path.write_text(json.dumps(config))

    loaded = Layer.load(tmp_path)
    assert set(loaded.state_dict()) == {"layers.0.weight", "layers.0.bias", "layers.1.weight", "layers.1.bias"}
    inputs = torch.randn(3, 4)
    torch.testing.assert_close(loaded(inputs, None), layer(inputs, None))
