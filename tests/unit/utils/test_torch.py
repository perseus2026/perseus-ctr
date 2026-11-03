from contextlib import contextmanager
from unittest.mock import Mock

import torch
from torch import nn

from perseus.core.config import Initialization, InitSpec
from perseus.core.optimization import apply_initialization
from perseus.utils.torch import Module


def test_wrapped_module_exposes_sync_context_and_preserves_forward():
    class Model(Module):
        def forward(self, value):
            return value * 2

    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.sync_enabled = True

        def forward(self, value):
            return self.model(value)

        @contextmanager
        def no_sync(self):
            self.sync_enabled = False
            try:
                yield
            finally:
                self.sync_enabled = True

    model = Model()
    wrapper = Wrapper(model)
    accelerator = Mock()
    accelerator.prepare_model.return_value = wrapper
    assert model.wrapped_module is model

    model.wrap(accelerator)
    assert model.wrapped_module is wrapper
    with model.wrapped_module.no_sync():
        assert not wrapper.sync_enabled
        torch.testing.assert_close(model(torch.tensor(3)), torch.tensor(6))
    assert wrapper.sync_enabled
    assert list(model.state_dict()) == []

    model.unwrap(accelerator)
    accelerator.unwrap_model.assert_called_once_with(wrapper, keep_fp32_wrapper=False)
    assert model.wrapped_module is model
    torch.testing.assert_close(model(torch.tensor(4)), torch.tensor(8))


def test_wrap_selects_ddp_only_for_trainable_parameters():
    class Model(Module):
        def __init__(self, trainable):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1), requires_grad=trainable)
            self.register_buffer("offset", torch.ones(1))

        def forward(self, value):
            return value * self.weight + self.offset

    for trainable in (False, True):
        model = Model(trainable)
        accelerator = Mock()
        accelerator.prepare_model.return_value = model
        model.wrap(accelerator)
        accelerator.prepare_model.assert_called_once_with(model, evaluation_mode=not trainable)
        model.unwrap(accelerator)


def test_configured_initialization_overrides_defaults_and_preserves_bias():
    class Model(Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32)

        def forward(self, value):
            return self.linear(value)

    spec = Initialization(linear=InitSpec(kind="normal", std=1e-8), bias="keep")
    model = Model()
    assert 0.01 < model.linear.weight.std() < 0.04
    torch.testing.assert_close(model.linear.bias, torch.zeros_like(model.linear.bias), rtol=0, atol=0)
    with torch.no_grad():
        model.linear.bias.fill_(7)
    original_bias = model.linear.bias.clone()
    apply_initialization(spec, model)
    assert torch.equal(model.linear.bias, original_bias)
    assert model.linear.weight.abs().max() < 1e-7

    model = Model()
    torch.testing.assert_close(model.linear.bias, torch.zeros_like(model.linear.bias), rtol=0, atol=0)
