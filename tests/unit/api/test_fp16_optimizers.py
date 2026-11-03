"""Exercise shared GradScaler state using real CPU AMP and Accelerate wrappers."""

from types import SimpleNamespace

import pytest
import torch
from accelerate import Accelerator

from perseus.api.fit_model import _create_scheduler
from perseus.core.optimization import clip_gradients
from perseus.core.optimization import optimizers as optimizer_api


@pytest.mark.parametrize("scope", [None, "all", "dense"])
@pytest.mark.parametrize("overflow", [(), (0,), (1,), (0, 1)])
def test_shared_scaler_updates_once_and_tracks_each_optimizer(overflow, scope):
    accelerator = Accelerator(cpu=True)
    scaler = torch.amp.GradScaler("cpu", init_scale=8.0, growth_interval=1)
    parameters = [torch.nn.Parameter(torch.tensor(2.0)), torch.nn.Parameter(torch.tensor(3.0))]
    accelerator.scaler = scaler
    optimizers = [accelerator.prepare_optimizer(torch.optim.SGD([p], lr=0.1)) for p in parameters]
    checkpoint = SimpleNamespace(
        config=SimpleNamespace(
            training=SimpleNamespace(
                scheduler=SimpleNamespace(type="linear", params={"num_warmup_steps": 0, "num_training_steps": 2}),
            ),
        ),
    )
    schedulers = _create_scheduler(accelerator, optimizers, checkpoint)
    assert all(scheduler.optimizers == [optimizer] for scheduler, optimizer in zip(schedulers, optimizers, strict=True))
    scaler.scale(sum(p.square() for p in parameters)).backward()
    for index in overflow:
        parameters[index].grad.fill_(float("inf"))
    for optimizer in optimizers:
        scaler.unscale_(optimizer.optimizer)
    original_gradients = [p.grad.clone() for p in parameters]
    # Index 0 represents embedding parameters; index 1 represents dense parameters.
    if scope is not None:
        clip_gradients(parameters if scope == "all" else parameters[1:], max_norm=100.0)
    optimizer_api.step(accelerator, optimizers)
    for scheduler in schedulers:
        scheduler.step()
    for index, (parameter, optimizer, scheduler) in enumerate(zip(parameters, optimizers, schedulers, strict=True)):
        expected = float(index + 2) if index in overflow else float(index + 2) * 0.8
        assert parameter.item() == pytest.approx(expected)
        torch.testing.assert_close(parameter.grad, original_gradients[index])
        assert optimizer.step_was_skipped == (index in overflow)
        assert scheduler.get_last_lr() == [0.1 if index in overflow else 0.05]
        assert optimizer.scaler is scaler
    assert scaler.get_scale() == (4.0 if overflow else 16.0)
