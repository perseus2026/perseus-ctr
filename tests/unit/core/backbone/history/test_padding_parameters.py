"""Padding choices are explicit per component and do not depend on the environment."""

import pytest
import torch

from perseus.core.backbone.history.interaction.cross_attention import CrossAttention, CrossAttentionInteraction
from perseus.core.backbone.history.nonsequence import registry
from perseus.core.backbone.history.sequence.ffn import Aggregator as FeedForward
from perseus.core.backbone.history.sequence.self_attention import Aggregator as AttentionEncoder
from perseus.core.backbone.history.sequence.self_attention import SequenceAttention


def test_sequence_zero_padding_is_explicit(monkeypatch):
    torch.manual_seed(1)
    normal = FeedForward(4, 3)
    zeroed = FeedForward(4, 3, zero_padding=True)
    zeroed.load_state_dict(normal.state_dict())
    values = torch.ones(1, 3, 4)
    positions = torch.tensor([[0, 2, 1]])
    timestamps = torch.zeros_like(positions)
    mask = torch.zeros(1, 3, 3, dtype=torch.bool)
    monkeypatch.setenv("PERSEUS_PROBE_UNIRANK_PADDING", "1")
    assert torch.count_nonzero(normal(values, positions, timestamps, mask)[:, 0]) > 0
    assert torch.count_nonzero(zeroed(values, positions, timestamps, mask)[:, 0]) == 0


@pytest.mark.parametrize("mask_padding", [True, False])
def test_self_attention_mask_choice(mask_padding, monkeypatch):
    module = SequenceAttention(4, 1, mask_padding=mask_padding)
    calls = []
    original = torch.nn.functional.scaled_dot_product_attention

    def record(*args, **kwargs):
        calls.append(kwargs["attn_mask"])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", record)
    module(torch.randn(1, 3, 4), torch.tensor([[True, False, False]]))
    assert (calls[0] is not None) == mask_padding
    if mask_padding:
        assert calls[0].flatten().tolist() == [False, True, True]


@pytest.mark.parametrize("padding", ["mask", "zero"])
def test_cross_attention_padding_choice(padding, monkeypatch):
    module = CrossAttention(4, 1, 1, "compact", padding=padding)
    calls = []
    original = torch.nn.functional.scaled_dot_product_attention

    def record(*args, **kwargs):
        calls.append((args[1], args[2], kwargs["attn_mask"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", record)
    module(torch.randn(1, 1, 4), torch.randn(1, 3, 4), torch.tensor([[True, False, False]]))
    keys, values, mask = calls[0]
    assert (mask is not None) == (padding == "mask")
    if padding == "zero":
        assert torch.count_nonzero(keys[:, :, 0]) == 0
        assert torch.count_nonzero(values[:, :, 0]) == 0


def test_unknown_cross_attention_padding_rejected():
    with pytest.raises(ValueError, match="unknown cross_attention padding"):
        CrossAttention(4, 1, 1, "compact", padding="unexpected")


def test_reverse_attention_requires_context_tokens():
    with pytest.raises(ValueError, match="requires context tokens"):
        CrossAttentionInteraction(4, 0, {"reader": "queries", "num_tokens": 2, "reverse_context": True})


@pytest.mark.parametrize("invalid", [0, -1, True])
def test_component_dimensions_must_be_positive_integers(invalid):
    with pytest.raises(ValueError, match="num_layers"):
        FeedForward(4, 3, num_layers=invalid)
    with pytest.raises(ValueError, match="expansion_factor"):
        FeedForward(4, 3, expansion_factor=invalid)
    with pytest.raises(ValueError, match="num_layers"):
        AttentionEncoder(4, 3, num_layers=invalid)
    with pytest.raises(ValueError, match="num_heads"):
        SequenceAttention(4, invalid)
    with pytest.raises(ValueError, match="num_heads"):
        CrossAttention(4, 1, invalid, "compact")
    with pytest.raises(ValueError, match="reader_tokens"):
        CrossAttention(4, invalid, 1, "compact")

    with pytest.raises(ValueError, match="query rank"):
        registry["identity"](4, 1, ["queries"], 1,
                             params={"queries": {"type": "persistent", "params": {"num_tokens": 1, "rank": invalid}}},
                             num_layers=1, independent_layers=False)
