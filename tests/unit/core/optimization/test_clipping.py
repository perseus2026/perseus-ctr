"""Dense-only gradient clipping: norm bound, scope, deduplication."""

import torch
from torch import nn

from perseus.core.optimization import clip_gradients, clipping


class _Module(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.embedding = nn.Embedding(3, 2)


def test_clip_bounds_total_norm() -> None:
    module = _Module()
    for parameter in module.parameters():
        parameter.grad = torch.full_like(parameter, 10.0)
    total_norm = clipping.create([module], "dense").clip(max_norm=1.0)
    assert total_norm > 1.0
    dense = [module.linear.weight, module.linear.bias]
    clipped_norm = float(torch.norm(torch.stack([p.grad.norm() for p in dense])))
    assert clipped_norm <= 1.0 + 1e-6


def test_dense_scope_leaves_embedding_gradients() -> None:
    module = _Module()
    for parameter in module.parameters():
        parameter.grad = torch.full_like(parameter, 10.0)
    clipping.create([module], "dense").clip(max_norm=1.0)
    assert module.embedding.weight.grad.abs().sum() == 10.0 * module.embedding.weight.numel()


def test_shared_parameter_counted_once() -> None:
    first = nn.Linear(2, 2, bias=False)
    second = nn.Linear(2, 2, bias=False)
    second.weight = first.weight
    first.weight.grad = torch.full_like(first.weight, 3.0)
    total_norm = clip_gradients([first.weight, second.weight], max_norm=100.0)
    expected = float((first.weight.grad * first.weight.grad).sum().sqrt())
    assert abs(total_norm - expected) < 1e-6


def test_params_without_grads_are_skipped() -> None:
    module = nn.Linear(2, 2)
    module.weight.grad = torch.full_like(module.weight, 1.0)
    total_norm = clip_gradients(list(module.parameters()), max_norm=10.0)
    assert abs(total_norm - 2.0) < 1e-6


def test_nonfinite_norm_preserves_all_optimizer_gradients() -> None:
    for nonfinite in (float("inf"), float("nan")):
        module = _Module()
        module.linear.weight.grad = torch.full_like(module.linear.weight, 3.0)
        module.embedding.weight.grad = torch.full_like(module.embedding.weight, nonfinite)
        before = [p.grad.clone() if p.grad is not None else None for p in module.parameters()]
        total_norm = clip_gradients(list(module.parameters()), max_norm=1.0)
        assert not torch.isfinite(torch.tensor(total_norm))
        for parameter, gradient in zip(module.parameters(), before, strict=True):
            if gradient is not None:
                torch.testing.assert_close(parameter.grad, gradient, equal_nan=True)
