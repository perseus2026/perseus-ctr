"""Step-by-step tests of api.fit_model (init and create steps; no full training loop)."""

import copy
import logging
from unittest.mock import Mock

import polars as pl
import pytest
import torch
from accelerate import Accelerator, DataLoaderConfiguration

from perseus.api.fit_model import (
    _create_dataloaders,
    _create_early_stopping,
    _create_optimizer,
    _create_scheduler,
    _dynamic_transform_artifacts,
    _init_backbone,
    _init_embedders,
    _init_task_entities,
    _load_model,
    _log_parameters,
    _renormalize_partial_window,
    _save_model,
    _split_test_samples,
    _test_epoch,
    _train_epoch,
    _unwrap,
    _wrap,
)
from perseus.core import checkpoint as ckpt
from perseus.core import trackers
from perseus.core.backbone import Backbone
from perseus.core.config import Config, Feature, LocatedIn, Metric, Optimizer
from perseus.core.tasks.classification.target import Artifacts as ClsArtifacts
from perseus.core.tasks.classification.target import Preprocessor as ClsPreprocessor
from perseus.core.tasks.ranking import target as ranking_target
from perseus.core.tasks.retrieval import target as retrieval_target
from tests.conftest import MINIMAL_CONFIG
from tests.unit.api.conftest import make_id_preprocessor


@pytest.fixture
def accelerator() -> Accelerator:
    return Accelerator(dataloader_config=DataLoaderConfiguration(non_blocking=True))


def test_init_embedders_unwrapped(accelerator, prepared_checkpoint) -> None:
    embedders = _init_embedders(prepared_checkpoint)
    assert "event" in embedders
    assert "_wrapped" not in embedders["event"].__dict__


def test_wrap(accelerator, prepared_checkpoint, dataset) -> None:
    embedders = _init_embedders(prepared_checkpoint)
    backbone = _init_backbone(prepared_checkpoint)
    layer, criterion, _head, _evaluator = _init_task_entities(prepared_checkpoint, dataset, backbone)
    _wrap(accelerator, embedders, backbone, layer, criterion)
    assert "_wrapped" in embedders["event"].__dict__
    assert "_wrapped" in backbone.__dict__
    for embedder in embedders.values():
        embedder.unwrap(accelerator)
    backbone.unwrap(accelerator)
    layer.unwrap(accelerator)


def test_init_backbone(prepared_checkpoint) -> None:
    backbone = _init_backbone(prepared_checkpoint)
    assert isinstance(backbone, Backbone)
    assert "_wrapped" not in backbone.__dict__


@pytest.mark.parametrize("hidden", [None, [16, 8]])
def test_init_task_entities(prepared_checkpoint, dataset, hidden) -> None:
    prepared_checkpoint.config.task.layer = {"hidden": hidden}
    backbone = _init_backbone(prepared_checkpoint)
    layer, _criterion, head, evaluator = _init_task_entities(prepared_checkpoint, dataset, backbone)
    assert all(isinstance(part, torch.nn.Sequential if hidden else torch.nn.Linear) for part in layer.layers)
    assert set(head.classes) == {"label"}
    assert "logloss" in evaluator.name_to_metric


def test_create_optimizer_and_scheduler(accelerator, prepared_checkpoint, dataset) -> None:
    embedders = _init_embedders(prepared_checkpoint)
    backbone = _init_backbone(prepared_checkpoint)
    layer, criterion, _, _ = _init_task_entities(prepared_checkpoint, dataset, backbone)
    criterion = _wrap(accelerator, embedders, backbone, layer, criterion)
    optimizers = _create_optimizer(accelerator, embedders, backbone, layer, criterion, prepared_checkpoint)
    schedulers = _create_scheduler(accelerator, optimizers, prepared_checkpoint)
    assert len(optimizers) == 1
    assert len(schedulers) == 1
    _log_parameters(embedders, backbone, layer, criterion)
    _unwrap(accelerator, embedders, backbone, layer)
    assert "_wrapped" not in backbone.__dict__


def test_named_optimizers_split_embeddings(accelerator, prepared_checkpoint, dataset) -> None:
    embedders = _init_embedders(prepared_checkpoint)
    backbone = _init_backbone(prepared_checkpoint)
    layer, criterion, _, _ = _init_task_entities(prepared_checkpoint, dataset, backbone)
    criterion = _wrap(accelerator, embedders, backbone, layer, criterion)
    prepared_checkpoint.config.training.optimizers.embeddings = Optimizer(type="adagrad", params={"lr": 0.05})
    optimizers = _create_optimizer(accelerator, embedders, backbone, layer, criterion, prepared_checkpoint)
    assert len(optimizers) == 2
    dense, sparse = optimizers
    modules = (*embedders.values(), backbone, layer, criterion)
    num_embedding_params = sum(
        p.numel()
        for m in modules
        for sub in m.modules()
        if isinstance(sub, torch.nn.Embedding)
        for p in sub.parameters()
    )
    num_total = sum(p.numel() for m in modules for p in m.parameters())
    sparse_numel = sum(p.numel() for group in sparse.param_groups for p in group["params"])
    dense_numel = sum(p.numel() for group in dense.param_groups for p in group["params"])
    assert sparse_numel == num_embedding_params
    assert dense_numel + sparse_numel == num_total
    _unwrap(accelerator, embedders, backbone, layer)


def test_create_early_stopping_none(prepared_checkpoint) -> None:
    assert _create_early_stopping(prepared_checkpoint) is None


def test_create_early_stopping_configured() -> None:
    config = Config.model_validate(
        {**MINIMAL_CONFIG, "training": {"early_stopping": {"metric": "logloss/label"}}},
    )
    prepared = ckpt.Prepared(config, {"event": make_id_preprocessor()})
    early_stopping = _create_early_stopping(prepared)
    assert early_stopping is not None
    assert early_stopping.metric == "logloss/label"


def test_dynamic_transform_artifacts(prepared_checkpoint, dataset) -> None:
    train, test = _dynamic_transform_artifacts(prepared_checkpoint, dataset)
    assert train is None
    assert test is None


def test_create_dataloaders(accelerator, prepared_checkpoint, dataset, caplog) -> None:
    with caplog.at_level(logging.INFO):
        train_dl, test_dl, _final_dl = _create_dataloaders(accelerator, prepared_checkpoint, dataset)
    train_batches = list(train_dl)
    assert all(batch.target.dtype == torch.long for batch in train_batches)
    assert sorted(row for batch in train_batches for row in batch.target.tolist()) == [[0], [1]]
    test_batch = next(iter(test_dl))
    assert "_index" in test_batch.extras


def _full_setup(accelerator, prepared_checkpoint, dataset):
    embedders = _init_embedders(prepared_checkpoint)
    backbone = _init_backbone(prepared_checkpoint)
    layer, criterion, head, evaluator = _init_task_entities(prepared_checkpoint, dataset, backbone)
    criterion = _wrap(accelerator, embedders, backbone, layer, criterion)
    optimizers = _create_optimizer(accelerator, embedders, backbone, layer, criterion, prepared_checkpoint)
    schedulers = _create_scheduler(accelerator, optimizers, prepared_checkpoint)
    train_dl, test_dl, _final_dl = _create_dataloaders(accelerator, prepared_checkpoint, dataset)
    train_art, test_art = _dynamic_transform_artifacts(prepared_checkpoint, dataset)
    return (
        embedders,
        backbone,
        layer,
        criterion,
        head,
        evaluator,
        optimizers,
        schedulers,
        train_dl,
        test_dl,
        train_art,
        test_art,
    )


@pytest.mark.parametrize("task_type", ["ranking", "retrieval"])
def test_train_epoch_with_tuple_target(accelerator, prepared_checkpoint, dataset, task_type) -> None:
    """Train through the real item-task target, artifacts, and criterion contracts."""
    task = ranking_target if task_type == "ranking" else retrieval_target
    items = pl.DataFrame({"item": ["a", "b"], "item_feature": [1, 0]})
    known_items = pl.DataFrame({"item": ["a", "b"], "index": [0, 1], "logq": [-0.69, -0.69]})
    if task_type == "ranking":
        preprocessor = task.Preprocessor(["label"], known_items, 1)
        targets = pl.Series(
            [[{"item": "a", "labels": {"label": 1}}], [{"item": "b", "labels": {"label": 0}}]],
        )
    else:
        preprocessor = task.Preprocessor(known_items, 1, sample_negatives=None)
        targets = pl.Series([["a"], ["b"]])
    dataset.task_preprocessor = preprocessor
    dataset.train_samples = dataset.train_samples.with_columns(preprocessor.static_transform(targets).alias("target"))
    dataset.train_artifacts = task.Artifacts(items)
    dataset.test_artifacts = task.Artifacts(items)
    prepared_checkpoint.config.task.type = task_type
    prepared_checkpoint.config.task.metrics = {}
    prepared_checkpoint.config.training.dataloader.batch_size = 2
    prepared_checkpoint.config.features["item_feature"] = Feature(
        located_in=LocatedIn(artifacts=True), encoder="event",
    )
    embedders, backbone, layer, criterion, _, _, optimizers, schedulers, train_dl, _, train_art, _ = _full_setup(
        accelerator, prepared_checkpoint, dataset,
    )
    assert isinstance(next(iter(train_dl)).target, tuple)
    before = layer.state_dict() if task_type == "ranking" else backbone.state_dict()
    before = {name: value.clone() for name, value in before.items()}
    try:
        total_step, losses = _train_epoch(
            1, 0, [], accelerator, embedders, backbone, layer, criterion, train_dl, train_art,
            optimizers, schedulers, prepared_checkpoint, dataset, trackers.registry["logging"](),
        )
        assert total_step == 1
        assert len(losses) == 1
        assert torch.isfinite(losses[0])
        after = layer.state_dict() if task_type == "ranking" else backbone.state_dict()
        assert any(not torch.equal(before[name], value) for name, value in after.items())
    finally:
        _unwrap(accelerator, embedders, backbone, layer)


@pytest.mark.parametrize("accumulation", [2, 3])
@pytest.mark.parametrize("num_labels", [1, 2])
def test_ranking_accumulation_weights_item_pairs(accelerator, prepared_checkpoint, dataset, accumulation, num_labels):
    """One positive pair and three negative pairs must give the merged-batch SGD update."""
    labels = ["label", "second_label"][:num_labels]
    preprocessor = ranking_target.Preprocessor(
        labels, pl.DataFrame({"item": ["a", "b"], "index": [0, 1]}), 1,
    )
    targets = pl.Series([
        [{"item": "a", "labels": dict.fromkeys(labels, 1)}],
        [{"item": "b", "labels": dict.fromkeys(labels, 0)}] * 3,
    ])
    dataset.task_preprocessor = preprocessor
    dataset.train_samples = dataset.train_samples.with_columns(preprocessor.static_transform(targets).alias("target"))
    items = pl.DataFrame({"item": ["a", "b"], "item_feature": [1, 0]})
    dataset.train_artifacts = ranking_target.Artifacts(items)
    dataset.test_artifacts = ranking_target.Artifacts(items)
    config = prepared_checkpoint.config
    config.task.type = "ranking"
    config.task.metrics = {}
    config.features["item_feature"] = Feature(located_in=LocatedIn(artifacts=True), encoder="event")
    config.training.optimizers.default = Optimizer(type="sgd", params={"lr": 0.1})
    config.training.dataloader.shuffle = False
    results = []
    try:
        for batch_size, steps in [(1, accumulation), (2, 1)]:
            torch.manual_seed(37)
            accelerator.gradient_accumulation_steps = steps
            prepared = copy.deepcopy(prepared_checkpoint)
            data = copy.deepcopy(dataset)
            prepared.config.training.dataloader.batch_size = batch_size
            embedders, backbone, layer, criterion, _, _, optimizers, schedulers, loader, _, artifacts, _ = _full_setup(
                accelerator, prepared, data,
            )
            with torch.no_grad():
                layer.linear.weight.zero_()
                layer.linear.bias.zero_()
            for module in backbone.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0.0
            try:
                _train_epoch(
                    1, 0, [], accelerator, embedders, backbone, layer, criterion, loader, artifacts,
                    optimizers, schedulers, prepared, data, trackers.registry["logging"](),
                )
                results.append({name: value.clone() for name, value in layer.state_dict().items()})
            finally:
                _unwrap(accelerator, embedders, backbone, layer)
    finally:
        accelerator.gradient_accumulation_steps = 1
    torch.testing.assert_close(results[0]["linear.bias"], torch.full((num_labels,), -0.025 / num_labels))
    for name in results[0]:
        torch.testing.assert_close(results[0][name], results[1][name], rtol=1e-5, atol=1e-7)


def test_train_test_epoch_and_save_load(accelerator, prepared_checkpoint, dataset, tmp_path) -> None:
    tracker = trackers.registry["logging"]()

    (
        embedders,
        backbone,
        layer,
        criterion,
        head,
        evaluator,
        optimizers,
        schedulers,
        train_dl,
        test_dl,
        train_art,
        test_art,
    ) = _full_setup(accelerator, prepared_checkpoint, dataset)

    total_step, _loss_buffer = _train_epoch(
        1,
        0,
        [],
        accelerator,
        embedders,
        backbone,
        layer,
        criterion,
        train_dl,
        train_art,
        optimizers,
        schedulers,
        prepared_checkpoint,
        dataset,
        tracker,
    )
    assert total_step >= 1

    _save_model(1, tmp_path, accelerator, embedders, backbone, layer)
    assert (tmp_path / "states" / "epoch=0001" / "backbone").exists()

    metrics = _test_epoch(
        1,
        tmp_path,
        accelerator,
        embedders,
        backbone,
        layer,
        head,
        evaluator,
        test_dl,
        test_art,
        prepared_checkpoint,
        dataset,
        tracker,
    )
    assert "logloss/label" in metrics
    assert "overall" in metrics["logloss/label"]

    _unwrap(accelerator, embedders, backbone, layer)
    _load_model(1, tmp_path, embedders, backbone, layer)  # must not raise


def test_test_epoch_evaluates_each_classification_target(
    accelerator,
    prepared_checkpoint,
    dataset,
    tmp_path,
) -> None:
    target_values = pl.Series(
        "target",
        [
            {"click": 0, "category": "a"},
            {"click": 1, "category": "b"},
            {"click": 0, "category": "c"},
        ],
    )
    preprocessor = ClsPreprocessor.fit(target_values, ClsArtifacts(), {})
    timestamp = dataset.train_samples["timestamp"][0]
    clients = ["c1", "c2", "c3"]
    dataset.task_preprocessor = preprocessor
    dataset.train_samples = pl.DataFrame(
        {
            "client_id": clients,
            "timestamp": [timestamp] * len(clients),
            "target": preprocessor.static_transform(target_values),
        },
    )
    dataset.test_samples = pl.DataFrame(
        {
            "client_id": clients,
            "timestamp": [timestamp] * len(clients),
            "target": target_values,
            "context": [{"item_id": index} for index in range(len(clients))],
            "source_row": list(range(len(clients))),
            "source_block": ["test-000.pq"] * len(clients),
        },
    )
    prepared_checkpoint.config.task.metrics = {
        "logloss": Metric(type="logloss"),
    }
    prepared_checkpoint.config.training.dataloader.batch_size = 2

    (
        embedders,
        backbone,
        layer,
        _criterion,
        head,
        evaluator,
        _optimizers,
        _schedulers,
        _train_dl,
        test_dl,
        _train_art,
        test_art,
    ) = _full_setup(accelerator, prepared_checkpoint, dataset)
    evaluator.calculate_metrics = Mock(wraps=evaluator.calculate_metrics)
    try:
        metrics = _test_epoch(
            1,
            tmp_path,
            accelerator,
            embedders,
            backbone,
            layer,
            head,
            evaluator,
            test_dl,
            test_art,
            prepared_checkpoint,
            dataset,
            trackers.registry["logging"](),
            predictions_dir=tmp_path / "export",
        )
    finally:
        _unwrap(accelerator, embedders, backbone, layer)

    evaluator.calculate_metrics.assert_called_once()
    evaluated_samples, artifacts = evaluator.calculate_metrics.call_args.args
    assert evaluated_samples["target"].struct.fields == ["click", "category"]
    assert artifacts is dataset.test_artifacts
    exported = pl.read_parquet(tmp_path / "export" / "predictions.parquet")
    assert exported.equals(evaluated_samples)
    assert exported["source_row"].to_list() == [0, 1, 2]
    assert "_index" not in exported.columns
    assert set(metrics) == {"logloss/click", "logloss/category"}
    assert metrics["logloss/category"]["overall"] >= 0.0
    assert metrics["logloss/click"]["overall"] >= 0.0


def test_create_dataloaders_shares_one_loader_without_split(accelerator, prepared_checkpoint, dataset) -> None:
    _, eval_dl, final_dl = _create_dataloaders(accelerator, prepared_checkpoint, dataset)
    assert eval_dl is final_dl


def test_create_dataloaders_evaluates_valid_then_test(accelerator, prepared_checkpoint, dataset) -> None:
    dataset.test_samples = pl.concat(
        [
            dataset.test_samples.with_columns(split=pl.lit("valid", dtype=pl.String)),
            dataset.test_samples.with_columns(split=pl.lit("test", dtype=pl.String)),
        ],
    )
    _, eval_dl, final_dl = _create_dataloaders(accelerator, prepared_checkpoint, dataset)
    assert eval_dl is not final_dl
    eval_indexes = [index for batch in eval_dl for index in batch.extras["_index"]]
    final_indexes = [index for batch in final_dl for index in batch.extras["_index"]]
    assert eval_indexes == [0]
    assert final_indexes == [1]


def test_split_test_samples_rejects_foreign_values() -> None:
    frame = pl.DataFrame({"split": ["valid", "train"]})
    with pytest.raises(ValueError, match="unexpected values"):
        _split_test_samples(frame)


def test_split_test_samples_requires_both_sides() -> None:
    frame = pl.DataFrame({"split": ["test", "test"]})
    with pytest.raises(ValueError, match="both valid and test"):
        _split_test_samples(frame)


@pytest.mark.parametrize("split_embeddings", [False, True])
def test_optimizer_groups_cover_shared_trainable_parameters_once(accelerator, prepared_checkpoint, split_embeddings):
    embedder = torch.nn.Embedding(5, 3)
    projection = torch.nn.Linear(3, 5)
    projection.weight = embedder.weight
    projection.bias.requires_grad_(False)
    backbone = torch.nn.Linear(3, 3)
    criterion = torch.nn.Linear(3, 1)
    configs = prepared_checkpoint.config.training.optimizers
    configs.default = Optimizer(type="sgd", params={"lr": 0.2})
    if split_embeddings:
        configs.embeddings = Optimizer(type="adagrad", params={"lr": 0.05})
    optimizers = _create_optimizer(
        accelerator, {"first": embedder, "shared": embedder}, backbone, projection, criterion, prepared_checkpoint,
    )
    parameter_ids = [id(p) for optimizer in optimizers for group in optimizer.param_groups for p in group["params"]]
    expected = {
        id(p)
        for module in (embedder, backbone, projection, criterion)
        for p in module.parameters()
        if p.requires_grad
    }
    assert len(parameter_ids) == len(set(parameter_ids))
    assert set(parameter_ids) == expected
    assert optimizers[0].param_groups[0]["lr"] == 0.2
    if split_embeddings:
        assert optimizers[1].param_groups[0]["params"] == [embedder.weight]
        assert optimizers[1].param_groups[0]["lr"] == 0.05
    schedulers = _create_scheduler(accelerator, optimizers, prepared_checkpoint)
    assert len(schedulers) == len(optimizers)


@pytest.mark.parametrize("kind", ["parameterless", "frozen", "trainable"])
def test_wrap_prepares_criterion_without_ddp_for_nontrainable_modules(kind):

    criterion = torch.nn.Linear(1, 1) if kind != "parameterless" else torch.nn.Identity()
    criterion.register_buffer("offset", torch.ones(1))
    if kind == "frozen":
        criterion.requires_grad_(False)
    accelerator = Mock()
    accelerator.prepare_model.return_value = criterion
    result = _wrap(accelerator, {}, Mock(), Mock(), criterion)
    assert result is criterion
    accelerator.prepare_model.assert_called_once_with(criterion, evaluation_mode=kind != "trainable")


@pytest.mark.parametrize("accumulation", [1, 2, 3])
def test_accumulation_normalization_uses_global_observation_count(accumulation):
    """DDP averages rank gradients, even when ranks contain unequal pair counts."""
    accelerator = Mock(num_processes=2, device=torch.device("cpu"))
    accelerator.reduce.return_value = torch.tensor(8)
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    # Rank zero has one positive pair; rank one has seven negative pairs.
    # BCE bias gradients at zero are -0.5 and +0.5 per pair, respectively.
    batch_size = 2
    parameter.grad = torch.tensor((-0.5 + 7 * 0.5) / (2 * batch_size * accumulation))
    _renormalize_partial_window(accelerator, [parameter], accumulation * batch_size, 1)
    torch.testing.assert_close(parameter.grad, torch.tensor(3 / 8))
    weight = accelerator.reduce.call_args.args[0]
    assert weight.item() == 1
    assert weight.dtype == torch.int64
    assert accelerator.reduce.call_args.kwargs == {"reduction": "sum"}
