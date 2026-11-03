"""Empty KV contributes zero while shared attention preserves its reader residual."""

import pytest
import torch

from perseus.core.backbone.history.interaction.cross_attention import CrossAttention


@pytest.mark.parametrize("projection", ["per_token", "compact", "shared", "per_query_kv"])
@pytest.mark.parametrize("padding", ["mask", "zero"])
@pytest.mark.parametrize("length", [0, 5])
def test_empty_history_has_finite_gradients_and_preserves_padding_contract(projection, padding, length):
    attention = CrossAttention(8, 3, 2, projection, padding=padding)
    reader = torch.randn(2, 3, 8, requires_grad=True)
    history = torch.randn(2, length, 8, requires_grad=True)
    mask = torch.ones(2, length, dtype=torch.bool)
    output = attention(reader, history, mask)
    expected = reader if projection == "shared" else torch.zeros_like(reader)
    if padding == "zero" and length:
        expected = attention(reader, torch.zeros_like(history), torch.zeros_like(mask))
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.square().sum().backward()
    assert reader.grad is not None
    assert history.grad is not None
    assert torch.isfinite(reader.grad).all()
    assert torch.isfinite(history.grad).all()
    for parameter in attention.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    torch.testing.assert_close(history.grad, torch.zeros_like(history))


@pytest.mark.parametrize("projection", ["per_token", "compact", "shared", "per_query_kv"])
def test_masked_values_cannot_change_empty_or_nonempty_history(projection):
    attention = CrossAttention(8, 3, 2, projection)
    reader = torch.randn(2, 3, 8)
    history = torch.randn(2, 5, 8)
    mask = torch.tensor([[True] * 5, [False, False, True, True, True]])
    changed = torch.where(mask.unsqueeze(-1), torch.randn_like(history) * 100, history)
    output = attention(reader, history, mask)
    torch.testing.assert_close(output, attention(reader, changed, mask), rtol=0, atol=0)
    expected = reader[0] if projection == "shared" else torch.zeros_like(reader[0])
    torch.testing.assert_close(output[0], expected, rtol=0, atol=0)
