"""Tests for backbone event and context aggregators and their registries."""

import pytest
import torch

from perseus.core import backbone
from perseus.core.backbone import context, event, history


class TestEventAggregators:
    @pytest.mark.parametrize("name", ["sum", "weighted_sum", "concat"])
    def test_forward_reduces_feature_dim(self, name: str) -> None:
        # Input shape (B, L, num_features, dim); output shape (B, L, dim).
        aggregator = event.registry[name](4, 2)
        out = aggregator(torch.randn(2, 3, 2, 4))
        assert out.shape == (2, 3, 4)

    def test_sum_is_feature_sum(self) -> None:
        aggregator = event.registry["sum"](4, 2)
        features = torch.randn(2, 3, 2, 4)
        assert torch.allclose(aggregator(features), features.sum(dim=2))


class TestContextAggregator:
    def test_identity_returns_input(self) -> None:
        aggregator = context.registry["identity"](4, 2)
        features = torch.randn(2, 1, 4)
        assert torch.allclose(aggregator(features), features)


def test_registries_have_expected_keys() -> None:
    assert set(event.registry) == {"sum", "weighted_sum", "concat", "concat_linear"}
    assert set(context.registry) == {"identity", "tokenizer"}
    assert {"bert", "modern_bert", "hstu", "mamba", "danet", "ligr"} <= set(history.registry)
def test_backbone_public_exports_exist():

    assert all(hasattr(backbone, name) for name in backbone.__all__)


def test_concat_linear_projects_when_input_and_output_width_match():
    aggregator = event.registry["concat_linear"](4, 2, embedding_dim=2)
    with torch.no_grad():
        aggregator.projection.weight.zero_()
        aggregator.projection.bias.fill_(2)
    features = torch.ones(1, 3, 2, 2)
    torch.testing.assert_close(aggregator(features), torch.full((1, 3, 4), 2.0))
