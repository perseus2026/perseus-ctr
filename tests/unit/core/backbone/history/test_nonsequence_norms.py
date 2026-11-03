"""Normalization semantics and checkpoint compatibility of nonsequence blocks."""

import pytest
import torch
from torch import nn

from perseus.core.backbone.history.nonsequence.attention import LowRankAttention, low_rank_attention
from perseus.core.backbone.history.nonsequence.stack import GroupedResidual, PreNormResidual, bind_norm_groups
from perseus.core.backbone.history.nonsequence.token_mixer import (
    LowRankTokenMixing,
    MultiHeadTokenMixing,
    ReversibleTokenMixing,
    token_mixing,
)


@pytest.fixture(params=["attention", "low_rank", "permutation"])
def block_factory(request):
    if request.param == "attention":
        return (
            lambda **params: low_rank_attention(8, 2, num_heads=2, qkv_rank=4, **params),
            lambda: LowRankAttention(8, 2, 2, 4),
        )
    if request.param == "low_rank":
        return (
            lambda **params: token_mixing(8, 2, kernel="low_rank", num_mix_heads=2, low_rank_dim=4, **params),
            lambda: LowRankTokenMixing(8, 2, 2, 4),
        )
    return (
        lambda **params: token_mixing(8, 2, kernel="permutation", **params),
        lambda: MultiHeadTokenMixing(8, 2),
    )


def test_default_preserves_initialization_checkpoint_output_and_rng(block_factory):
    factory, operation = block_factory
    torch.manual_seed(37)
    legacy = PreNormResidual(8, operation())
    legacy_rng = torch.get_rng_state()
    torch.manual_seed(37)
    actual = factory()
    assert torch.equal(torch.get_rng_state(), legacy_rng)
    assert actual.state_dict().keys() == legacy.state_dict().keys()
    for key, value in actual.state_dict().items():
        assert torch.equal(value, legacy.state_dict()[key])
    actual.load_state_dict(legacy.state_dict(), strict=True)
    tokens = torch.randn(3, 2, 8)
    torch.testing.assert_close(actual(tokens), legacy(tokens), rtol=0, atol=0)


def test_none_uses_raw_tokens_without_norm(block_factory):
    factory, _ = block_factory
    block = factory(norm="none")
    assert not any(isinstance(module, nn.LayerNorm) for module in block.modules())
    tokens = torch.randn(3, 2, 8)
    torch.testing.assert_close(block(tokens), tokens + block.op(tokens))


def test_group_uses_one_actual_shared_norm_with_gradients(block_factory):
    factory, _ = block_factory
    first = factory(norm_group="joint")
    second = factory(norm_group="joint")
    assert isinstance(first, GroupedResidual)
    holders = bind_norm_groups([first, second])
    assert len(holders) == 1
    norm = holders[0].norm
    with torch.no_grad():
        norm.weight.fill_(2)
        norm.bias.fill_(0.4)
    tokens = torch.randn(3, 2, 8)
    calls = []
    handle = norm.register_forward_hook(lambda *_: calls.append(1))
    output = first(tokens) + second(tokens)
    handle.remove()
    assert len(calls) == 2
    expected = 2 * tokens + first.op(norm(tokens)) + second.op(norm(tokens))
    torch.testing.assert_close(output, expected)
    output.square().sum().backward()
    assert norm.weight.grad is not None
    assert torch.isfinite(norm.weight.grad).all()
    assert norm.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("params", [{"norm": "invalid"}, {"norm": "none", "norm_group": "joint"}])
def test_invalid_normalization_is_rejected(block_factory, params):
    factory, _ = block_factory
    with pytest.raises(ValueError, match="norm"):
        factory(**params)


@pytest.mark.parametrize("params", [{"norm": "none"}, {"norm": "invalid"}, {"norm_group": "joint"}])
def test_reversible_rejects_incompatible_normalization(params):
    with pytest.raises(ValueError, match="reversible token_mixing"):
        token_mixing(8, 2, kernel="permutation", reversible=True, **params)


def test_reversible_default_remains_compatible():
    torch.manual_seed(37)
    legacy = ReversibleTokenMixing(8, 2, 2, 0.0)
    legacy_rng = torch.get_rng_state()
    torch.manual_seed(37)
    actual = token_mixing(8, 2, kernel="permutation", reversible=True, norm="layer_norm")
    assert torch.equal(torch.get_rng_state(), legacy_rng)
    actual.load_state_dict(legacy.state_dict(), strict=True)
    tokens = torch.randn(3, 2, 8)
    torch.testing.assert_close(actual(tokens), legacy(tokens), rtol=0, atol=0)
