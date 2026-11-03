"""Exercise automatic window assembly through the training API."""

import polars as pl
import pytest
import torch

from perseus.api.fit_model import _train_epoch, _unwrap
from perseus.core import trackers
from perseus.core.event_store import EventStore
from tests.unit.api.test_fit_model import _full_setup
from tests.unit.api.test_fit_model import accelerator as accelerator_fixture

accelerator = accelerator_fixture


@pytest.mark.parametrize("backend", ["ram", "cached"])
def test_batch_mode_one_training_update(
    accelerator, prepared_checkpoint, dataset, tmp_path, monkeypatch, backend,
):
    config = prepared_checkpoint.config
    config.max_events_per_sequence = 4
    config.training.dataloader.batch_size = 2
    config.training.dataloader.num_workers = 0
    config.training.test_dataloader.num_workers = 0
    config.training.log_every_n_train_steps = 100
    if backend == "cached":
        # Reuse the four prepared fixture events, rather than preparing data.
        events = pl.read_parquet(dataset.event_store.path / "data.pq")
        store = EventStore(tmp_path / "cached", backend="cached")
        with store.open_writer() as writer:
            writer.write(events)
        dataset.event_store = store
        config.training.event_store_backend = "cached"

    calls = []
    reader_type = type(dataset.event_store.open_reader(4))
    method = "read_batch_tensors" if backend == "cached" else "read_batch"
    original = getattr(reader_type, method)
    def observe_batch(self, clients, boundaries, *, features=None):
        calls.append(len(clients))
        return original(self, clients, boundaries, features=features)
    monkeypatch.setattr(reader_type, method, observe_batch)

    (
        embedders, backbone, layer, criterion, _head, _evaluator,
        optimizers, schedulers, train_loader, eval_loader, train_artifacts, _test_artifacts,
    ) = _full_setup(accelerator, prepared_checkpoint, dataset)
    try:
        # Also assert evaluation receives the configured mode through the API.
        evaluation = next(iter(eval_loader))
        # Evaluation joins labels by _index rather than collating targets.
        assert evaluation.target is None
        assert evaluation.events_positions.shape[0] == 1
        assert evaluation.extras["_index"] == [0]
        assert calls == [1]
        before = [parameter.detach().clone() for parameter in layer.parameters()]
        step, losses = _train_epoch(
            1, 0, [], accelerator, embedders, backbone, layer, criterion,
            train_loader, train_artifacts, optimizers, schedulers,
            prepared_checkpoint, dataset, trackers.registry["logging"](),
        )
        assert step == 1
        assert len(losses) == 1
        assert torch.isfinite(losses[0]).all()
        assert any(
            not torch.equal(previous, current.detach())
            for previous, current in zip(before, layer.parameters(), strict=True)
        )
        assert any(optimizer.state for optimizer in optimizers)
        assert calls == [1, 2]
    finally:
        _unwrap(accelerator, embedders, backbone, layer)
