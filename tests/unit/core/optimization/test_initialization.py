"""Initialization recipe: module rules, shared weights, protected tensors, PAD rows, measurement."""

import torch
from torch import nn

from perseus.core.config import Initialization, InitSpec
from perseus.core.optimization import apply_initialization


class _Container(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)
        self.embedding = nn.Embedding(5, 3, padding_idx=0)
        self.query = nn.Parameter(torch.zeros(3, 3))
        self.norm = nn.LayerNorm(3)


class TestApplyInitialization:
    def test_linear_rule_and_zero_bias(self) -> None:
        module = _Container()
        apply_initialization(
            Initialization(linear=InitSpec(kind="normal", std=0.5), embedding=InitSpec(kind="normal", std=0.01)),
            module,
        )
        assert module.linear.bias.abs().sum() == 0
        assert 0.01 < module.linear.weight.std() < 1.0

    def test_embedding_rule_and_padding_row_zero(self) -> None:
        module = _Container()
        apply_initialization(
            Initialization(linear=InitSpec(kind="normal", std=0.5), embedding=InitSpec(kind="normal", std=0.1)),
            module,
        )
        assert module.embedding.weight[1:].abs().sum() > 0
        assert module.embedding.weight[0].abs().sum() == 0

    def test_protected_tensor_parameters_and_norms_untouched(self) -> None:
        module = _Container()
        probe = module.query.clone()
        norm_weight = module.norm.weight.clone()
        apply_initialization(
            Initialization(linear=InitSpec(kind="normal", std=0.5), embedding=InitSpec(kind="normal", std=0.1)),
            module,
        )
        assert torch.equal(module.query, probe)
        assert torch.equal(module.norm.weight, norm_weight)

    def test_shared_weight_initialized_once(self) -> None:
        first = nn.Linear(4, 3, bias=False)
        second = nn.Linear(4, 3, bias=False)
        second.weight = first.weight

        torch.manual_seed(100)
        probe = torch.empty(3, 4)
        nn.init.normal_(probe, std=1.0)

        torch.manual_seed(100)
        apply_initialization(Initialization(linear=InitSpec(kind="normal", std=1.0)), first, second)
        torch.testing.assert_close(first.weight.data, probe)

    def test_bias_keep_preserves_bias(self) -> None:
        module = _Container()
        with torch.no_grad():
            module.linear.bias.fill_(7.0)
        apply_initialization(
            Initialization(linear=InitSpec(kind="normal", std=0.5), bias="keep"),
            module,
        )
        torch.testing.assert_close(module.linear.bias, torch.full_like(module.linear.bias, 7.0), rtol=0, atol=0)

    def test_shared_bias_initialized_once(self, monkeypatch) -> None:
        first = nn.Linear(4, 3)
        second = nn.Linear(4, 3)
        second.bias = first.bias
        calls = []
        original = nn.init.zeros_

        def observe(tensor):
            if tensor is first.bias:
                calls.append(tensor)
            return original(tensor)

        monkeypatch.setattr(nn.init, "zeros_", observe)
        apply_initialization(Initialization(linear=InitSpec(kind="xavier_normal")), first, second)
        assert len(calls) == 1
        assert torch.equal(first.bias, torch.zeros_like(first.bias))
        assert first.bias is second.bias
