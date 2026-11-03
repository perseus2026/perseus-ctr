"""Persistent query generation preserves tokenizer checkpoints and numerical behavior."""

import pytest
import torch

from perseus.core.backbone.context.base import Layout
from perseus.core.backbone.history.nonsequence.identity import Identity
from perseus.core.backbone.history.pooling import masked_mean
from perseus.core.backbone.history.tokenizer import AutoSplitTokenizer


@pytest.mark.parametrize("raw_fields", [False, True])
@pytest.mark.parametrize("readout", [False, True])
@pytest.mark.parametrize("rank", [None, 3])
@pytest.mark.parametrize("padding_mode", ["full", "mixed", "empty"])
def test_query_generator_preserves_seed_checkpoint_outputs_and_gradients(raw_fields, readout, rank, padding_mode):
    layout = Layout(3, 4, 2, False, raw_fields)
    width = (12 if raw_fields else 16) + (8 if readout else 0) + 8
    torch.manual_seed(42)
    reference = AutoSplitTokenizer(width, 8, 2, rank)
    rng = torch.random.get_rng_state()
    torch.manual_seed(42)
    encoder = Identity(
        8, 2, ["queries"], 2,
        params={"queries": {"type": "persistent", "params": {"num_tokens": 2, "rank": rank}}},
        num_layers=1, independent_layers=False, context_layout=layout, readout=readout,
    )
    generator = encoder.query_generator
    assert torch.equal(rng, torch.random.get_rng_state())
    assert list(generator.state_dict()) == list(reference.state_dict())
    for actual, expected in zip(generator.parameters(), reference.parameters(), strict=True):
        assert torch.equal(actual, expected)
    generator.load_state_dict(reference.state_dict(), strict=True)
    history = torch.randn(2, 4, 8, requires_grad=True)
    context = torch.randn(2, 2 + int(readout), 8, requires_grad=True)
    raw = torch.randn(2, 3, 4, requires_grad=True)
    padding = torch.zeros(2, 4, dtype=torch.bool)
    if padding_mode == "mixed":
        padding[0, :2] = True
    elif padding_mode == "empty":
        padding[:] = True
    features = [raw.flatten(1)] if raw_fields else [context.flatten(1)]
    if raw_fields and readout:
        features.append(context[:, -1])
    features.append(masked_mean(history, padding))
    expected = reference(torch.cat(features, dim=-1))
    actual = encoder.create_queries(history, context, padding, raw)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    inputs = (history, raw, context)
    expected_gradients = torch.autograd.grad(
        expected.square().sum(), (*inputs, *reference.parameters()), allow_unused=True, retain_graph=True,
    )
    actual_gradients = torch.autograd.grad(
        actual.square().sum(), (*inputs, *generator.parameters()), allow_unused=True,
    )
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        if expected_gradient is None:
            assert actual_gradient is None
        else:
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)
