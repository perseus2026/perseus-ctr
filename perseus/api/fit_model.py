import contextlib
import logging
import shutil
import typing as t
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import torch
from accelerate import Accelerator, DataLoaderConfiguration, load_checkpoint_in_model
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import broadcast_object_list, set_seed
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from perseus import core
from perseus.utils.torch import Module

logger = logging.getLogger(__name__)


def fit_model(
    checkpoint: core.checkpoint.Prepared,
    dataset: core.Dataset,
    /,
    *,
    fixed_padding: bool = True,
    test_predictions_dir: Path | None = None,
) -> core.checkpoint.Trained:
    accelerator = Accelerator(
        dataloader_config=DataLoaderConfiguration(non_blocking=True),
        mixed_precision=checkpoint.config.training.mixed_precision,
        gradient_accumulation_steps=checkpoint.config.training.accumulation or 1,
    )

    tracker = None
    if accelerator.is_main_process:
        tracker_config = checkpoint.config.training.tracker
        if tracker_config.type not in core.trackers.registry:
            raise ValueError(
                f"unknown tracker `{tracker_config.type}`; available: {sorted(core.trackers.registry)} "
                f"(some trackers require extra dependencies, e.g. `pip install perseus[clearml]`)",
            )
        tracker = core.trackers.registry[tracker_config.type](**tracker_config.params)
    set_seed(checkpoint.config.training.seed)

    encoder_to_embedder = _init_embedders(checkpoint)
    backbone = _init_backbone(checkpoint)
    task_layer, criterion, head, evaluator = _init_task_entities(checkpoint, dataset, backbone)
    if checkpoint.config.training.initialization is not None:
        core.optimization.apply_initialization(
            checkpoint.config.training.initialization,
            *encoder_to_embedder.values(), backbone, task_layer, criterion,
        )
    criterion = _wrap(accelerator, encoder_to_embedder, backbone, task_layer, criterion)
    if accelerator.is_main_process:
        _log_parameters(encoder_to_embedder, backbone, task_layer, criterion)

    optimizers = _create_optimizer(accelerator, encoder_to_embedder, backbone, task_layer, criterion, checkpoint)
    schedulers = _create_scheduler(accelerator, optimizers, checkpoint)
    early_stopping = _create_early_stopping(checkpoint)

    train_dataloader, eval_dataloader, final_dataloader = _create_dataloaders(
        accelerator, checkpoint, dataset, fixed_padding=fixed_padding,
    )
    split_evaluations = final_dataloader is not eval_dataloader
    train_transformed_artifacts, test_transformed_artifacts = _dynamic_transform_artifacts(checkpoint, dataset)

    with contextlib.ExitStack() as stack:
        cache_dir = Path(stack.enter_context(TemporaryDirectory())) if accelerator.is_main_process else None
        [cache_dir] = broadcast_object_list([cache_dir], from_process=0)
        epoch_predictions_dir = cache_dir / "test_predictions" if test_predictions_dir is not None else None

        total_step = 0
        loss_buffer = []
        for epoch in range(1, checkpoint.config.training.num_epochs + 1):
            total_step, loss_buffer = _train_epoch(
                epoch,
                total_step,
                loss_buffer,
                accelerator,
                encoder_to_embedder,
                backbone,
                task_layer,
                criterion,
                train_dataloader,
                train_transformed_artifacts,
                optimizers,
                schedulers,
                checkpoint,
                dataset,
                tracker,
            )
            if early_stopping is not None:
                _save_model(epoch, cache_dir, accelerator, encoder_to_embedder, backbone, task_layer)
            metrics = _test_epoch(
                epoch,
                cache_dir,
                accelerator,
                encoder_to_embedder,
                backbone,
                task_layer,
                head,
                evaluator,
                eval_dataloader,
                test_transformed_artifacts,
                checkpoint,
                dataset,
                tracker,
                label="valid" if split_evaluations else None,
                predictions_dir=None if split_evaluations else epoch_predictions_dir,
            )
            should_stop = early_stopping is not None and early_stopping.should_stop(metrics)
            if (epoch_predictions_dir is not None and not split_evaluations and accelerator.is_main_process
                    and (early_stopping is None or early_stopping.best_epoch == epoch)):
                test_predictions_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(epoch_predictions_dir / "predictions.parquet", test_predictions_dir / "predictions.parquet")
            if should_stop:
                break

        _unwrap(accelerator, encoder_to_embedder, backbone, task_layer)
        final_epoch = checkpoint.config.training.num_epochs
        if early_stopping is not None:
            _load_model(early_stopping.best_epoch, cache_dir, encoder_to_embedder, backbone, task_layer)
            final_epoch = early_stopping.best_epoch
        if split_evaluations:
            _test_epoch(
                final_epoch,
                cache_dir,
                accelerator,
                encoder_to_embedder,
                backbone,
                task_layer,
                head,
                evaluator,
                final_dataloader,
                test_transformed_artifacts,
                checkpoint,
                dataset,
                tracker,
                label="test",
                predictions_dir=test_predictions_dir,
            )
        accelerator.wait_for_everyone()

    return core.checkpoint.Trained(
        checkpoint.config,
        checkpoint.encoder_to_preprocessor,
        encoder_to_embedder,
        backbone,
        head,
    )


def _init_embedders(checkpoint: core.checkpoint.Prepared) -> dict[str, core.encoders.Embedder]:
    return {
        encoder: core.encoders.registry[desc.type].embedder.init(
            checkpoint.encoder_to_preprocessor[encoder],
            checkpoint.config.embedding_dim,
            **desc.embedder,
        )
        for encoder, desc in checkpoint.config.encoders.items()
    }


def _init_backbone(checkpoint: core.checkpoint.Prepared) -> core.Backbone:
    event_aggregator_config = {
        "num_features": len(checkpoint.config.feature_to_encoder(event=True)),
        **checkpoint.config.backbone.event_aggregator.model_dump(),
    }
    history_aggregator_config = {
        "max_events_per_sequence": checkpoint.config.max_events_per_sequence,
        **checkpoint.config.backbone.history_aggregator.model_dump(),
    }
    context_fields = list(checkpoint.config.feature_to_encoder(context=True))
    context_aggregator_config = None
    if checkpoint.config.backbone.context_aggregator is not None:
        context_aggregator_config = {
            "num_features": len(context_fields),
            **checkpoint.config.backbone.context_aggregator.model_dump(),
        }
    return core.Backbone(
        checkpoint.config.backbone.dim,
        history_aggregator_config=history_aggregator_config,
        event_aggregator_config=event_aggregator_config,
        context_aggregator_config=context_aggregator_config,
        embedding_dim=checkpoint.config.embedding_dim,
        mask_context_from_history=checkpoint.config.backbone.mask_context_from_history,
        context_fields=context_fields,
        readout_token=checkpoint.config.backbone.readout_token,
        zero_context_features=checkpoint.config.backbone.zero_context_features,
    )


def _init_task_entities(
    checkpoint: core.checkpoint.Prepared,
    dataset: core.Dataset,
    backbone: core.Backbone,
) -> tuple[core.tasks.Layer, core.tasks.Criterion, core.tasks.Head, core.tasks.Evaluator]:
    task_layer = core.tasks.registry[checkpoint.config.task.type].layer.init(
        dataset.task_preprocessor,
        backbone.output_shape[-1],
        **checkpoint.config.task.layer,
    )

    criterion = core.tasks.registry[checkpoint.config.task.type].criterion.init(
        dataset.task_preprocessor,
        task_layer,
        **checkpoint.config.task.criterion,
    )

    head = core.tasks.registry[checkpoint.config.task.type].head.init(
        dataset.task_preprocessor,
        task_layer,
        **checkpoint.config.task.head,
    )

    metric_type_to_cls = core.tasks.registry[checkpoint.config.task.type].metrics
    name_to_metric = {
        name: metric_type_to_cls[metric_desc.type](**metric_desc.params)
        for name, metric_desc in checkpoint.config.task.metrics.items()
    }
    evaluator = core.tasks.registry[checkpoint.config.task.type].evaluator(name_to_metric)

    return task_layer, criterion, head, evaluator


def _wrap(
    accelerator: Accelerator,
    encoder_to_embedder: dict[str, core.encoders.Embedder],
    backbone: core.Backbone,
    task_layer: core.tasks.Layer,
    criterion: core.tasks.Criterion,
) -> core.tasks.Criterion:
    for embedder in encoder_to_embedder.values():
        embedder.wrap(accelerator)
    backbone.wrap(accelerator)
    task_layer.wrap(accelerator)
    return accelerator.prepare_model(
        criterion, evaluation_mode=not any(parameter.requires_grad for parameter in criterion.parameters()),
    )


def _log_parameters(
    encoder_to_embedder: dict[str, core.encoders.Embedder],
    backbone: core.Backbone,
    task_layer: core.tasks.Layer,
    criterion: core.tasks.Criterion,
) -> None:
    encoder_num_params_total = 0
    for encoder, embedder in encoder_to_embedder.items():
        encoder_num_params = sum(p.numel() for p in embedder.parameters() if p.requires_grad)
        logger.info("encoder %s has %s trainable parameters", encoder, encoder_num_params)
        encoder_num_params_total += encoder_num_params
    logger.info("encoders has %s trainable parameters in total", encoder_num_params_total)

    backbone_num_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    logger.info("backbone has %s trainable parameters", backbone_num_params)

    task_layer_num_params = sum(p.numel() for p in task_layer.parameters() if p.requires_grad)
    logger.info("task layer has %s trainable parameters", task_layer_num_params)

    total_model_num_params = encoder_num_params_total + backbone_num_params + task_layer_num_params
    logger.info("model has %s trainable parameters in total", total_model_num_params)

    criterion_num_params = sum(p.numel() for p in criterion.parameters() if p.requires_grad)
    logger.info("criterion has %s trainable parameters", criterion_num_params)


def _create_optimizer(
    accelerator: Accelerator,
    encoder_to_embedder: dict[str, core.encoders.Embedder],
    backbone: core.Backbone,
    task_layer: core.tasks.Layer,
    criterion: core.tasks.Criterion,
    checkpoint: core.checkpoint.Prepared,
) -> list[optim.Optimizer]:
    modules = (*encoder_to_embedder.values(), backbone, task_layer, criterion)
    return core.optimization.optimizers.create(accelerator, modules, checkpoint.config.training.optimizers)


def _create_scheduler(
    accelerator: Accelerator,
    optimizers: list[optim.Optimizer],
    checkpoint: core.checkpoint.Prepared,
) -> list[optim.lr_scheduler.LRScheduler]:
    return [
        accelerator.prepare_scheduler(
            core.optimization.schedulers.registry[checkpoint.config.training.scheduler.type](
                optimizer.optimizer if isinstance(optimizer, AcceleratedOptimizer) else optimizer,
                **checkpoint.config.training.scheduler.params,
            ),
        )
        for optimizer in optimizers
    ]


def _create_early_stopping(checkpoint: core.checkpoint.Prepared) -> core.optimization.EarlyStopping | None:
    if checkpoint.config.training.early_stopping is None:
        return None
    return core.optimization.EarlyStopping(**checkpoint.config.training.early_stopping.model_dump())


def _create_dataloaders(
    accelerator: Accelerator,
    checkpoint: core.checkpoint.Prepared,
    dataset: core.Dataset,
    *,
    fixed_padding: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    event_feature_to_preprocessor = {
        feature: checkpoint.encoder_to_preprocessor[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(event=True).items()
    }
    context_feature_to_preprocessor = {
        feature: checkpoint.encoder_to_preprocessor[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(context=True).items()
    }
    events_reader = dataset.event_store.open_reader(checkpoint.config.max_events_per_sequence)

    train_dataloader = core.dataloaders.samples(
        dataset.train_samples,
        events_reader,
        event_feature_to_preprocessor,
        context_feature_to_preprocessor,
        task_preprocessor=dataset.task_preprocessor,
        fixed_padding=fixed_padding,
        batch_size=checkpoint.config.training.dataloader.batch_size,
        shuffle=checkpoint.config.training.dataloader.shuffle,
        num_workers=checkpoint.config.training.dataloader.num_workers,
        pin_memory=checkpoint.config.training.dataloader.pin_memory,
        persistent_workers=checkpoint.config.training.dataloader.num_workers > 0,
    )
    train_dataloader = accelerator.prepare_data_loader(train_dataloader)

    dataset.test_samples = dataset.test_samples.with_row_index("_index")
    eval_samples, final_samples = _split_test_samples(dataset.test_samples)

    def test_dataloader(samples: pl.DataFrame) -> DataLoader:
        return accelerator.prepare_data_loader(
            core.dataloaders.samples(
                samples,
                events_reader,
                event_feature_to_preprocessor,
                context_feature_to_preprocessor,
                extras=["_index", *core.tasks.registry[checkpoint.config.task.type].head.extras],
                fixed_padding=fixed_padding,
                batch_size=checkpoint.config.training.test_dataloader.batch_size,
                shuffle=False,
                num_workers=checkpoint.config.training.test_dataloader.num_workers,
                pin_memory=checkpoint.config.training.test_dataloader.pin_memory,
                persistent_workers=checkpoint.config.training.test_dataloader.num_workers > 0,
            ),
        )

    if eval_samples is final_samples:
        shared_dataloader = test_dataloader(final_samples)
        return train_dataloader, shared_dataloader, shared_dataloader
    return train_dataloader, test_dataloader(eval_samples), test_dataloader(final_samples)


def _split_test_samples(samples: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    if "split" not in samples.schema:
        return samples, samples
    if foreign := set(samples["split"].unique().to_list()) - {"valid", "test"}:
        raise ValueError(f"test samples `split` column holds unexpected values: {sorted(foreign)}")
    eval_samples = samples.filter(pl.col("split") == "valid")
    final_samples = samples.filter(pl.col("split") == "test")
    if eval_samples.is_empty() or final_samples.is_empty():
        raise ValueError("test samples `split` column must hold both valid and test rows")
    return eval_samples, final_samples


def _dynamic_transform_artifacts(checkpoint: core.checkpoint.Prepared, dataset: core.Dataset) -> tuple[t.Any, t.Any]:
    artifacts_feature_to_preprocessor = {
        feature: checkpoint.encoder_to_preprocessor[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(artifacts=True).items()
    }
    train_transformed_artifacts = dataset.train_artifacts.dynamic_transform(artifacts_feature_to_preprocessor)
    test_transformed_artifacts = dataset.test_artifacts.dynamic_transform(artifacts_feature_to_preprocessor)
    return train_transformed_artifacts, test_transformed_artifacts


def _renormalize_partial_window(
    accelerator: Accelerator,
    parameters: list[nn.Parameter],
    nominal_window_weight: int,
    window_weight: int,
) -> None:
    """Normalize accumulated mean losses by their total observation count across ranks.

    Each local loss was multiplied by weight / batch_size, then Accelerator
    divided by accumulation and DDP averaged over ranks. Multiplying by
    accumulation * batch_size * world_size / global_weight recovers the mean
    over the actual observations, including full windows and uneven tails.
    """
    world_size = accelerator.num_processes
    if world_size > 1:
        weight = torch.tensor(window_weight, dtype=torch.int64, device=accelerator.device)
        window_weight = int(accelerator.reduce(weight, reduction="sum").item())
    factor = nominal_window_weight * world_size / window_weight
    if factor != 1.0:
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(factor)


def _train_epoch(
    epoch: int,
    total_step: int,
    loss_buffer: list[torch.Tensor],
    accelerator: Accelerator,
    encoder_to_embedder: dict[str, core.encoders.Embedder],
    backbone: core.Backbone,
    task_layer: core.tasks.Layer,
    criterion: core.tasks.Criterion,
    train_dataloader: DataLoader,
    train_transformed_artifacts: t.Any,
    optimizers: list[optim.Optimizer],
    schedulers: list[optim.lr_scheduler.LRScheduler],
    checkpoint: core.checkpoint.Prepared,
    dataset: core.Dataset,
    tracker: core.trackers.Tracker | None,
) -> tuple[int, list[torch.Tensor]]:
    accelerator.wait_for_everyone()
    for embedder in encoder_to_embedder.values():
        embedder.train()
    backbone.train()
    task_layer.train()

    training = checkpoint.config.training
    modules = (*encoder_to_embedder.values(), backbone, task_layer, criterion)
    sync_modules = tuple(module.wrapped_module if isinstance(module, Module) else module for module in modules)
    update_parameters = core.optimization.optimizers.parameters(modules)
    clipping = training.gradient_clipping
    clipper = None if clipping is None else core.optimization.clipping.create(modules, clipping.scope)

    accumulation = accelerator.gradient_accumulation_steps
    nominal_window_weight = accumulation * training.dataloader.batch_size
    window_weight = 0
    normalize_loss = accumulation > 1 or accelerator.num_processes > 1
    loss_criterion = accelerator.unwrap_model(criterion)
    normalization_weight = getattr(loss_criterion, "normalization_weight", None)

    backbone_feature_to_embedder = {
        feature: encoder_to_embedder[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(event=True, context=True).items()
    }
    artifacts_feature_to_embedder = {
        feature: encoder_to_embedder[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(artifacts=True).items()
    }
    for batch in tqdm(train_dataloader, desc=f"train {epoch = } / {training.num_epochs}"):
        rows = int(batch.events_positions.shape[0])
        weight = rows if normalization_weight is None else normalization_weight(batch.target, rows)
        loss_scale = weight / training.dataloader.batch_size if normalize_loss else 1.0
        with accelerator.accumulate(*sync_modules):
            with accelerator.autocast():
                backbone_embeddings = backbone(
                    backbone_feature_to_embedder,
                    batch.events_features,
                    batch.events_positions,
                    batch.events_timestamps,
                    batch.sample_timestamp,
                    batch.context_features,
                )
                embedded_artifacts = dataset.train_artifacts.embed(
                    train_transformed_artifacts,
                    artifacts_feature_to_embedder,
                    batch.target,
                )
                loss = criterion(backbone_embeddings, embedded_artifacts, batch.target)

            accelerator.backward(loss * loss_scale)
            if accelerator.sync_gradients:
                window_weight += weight
                accelerator.unscale_gradients()
                if normalize_loss:
                    _renormalize_partial_window(
                        accelerator,
                        update_parameters,
                        nominal_window_weight,
                        window_weight,
                    )
                if clipping is not None:
                    clipper.clip(clipping.max_norm, clipping.norm_type)
                core.optimization.optimizers.step(accelerator, optimizers)
                for scheduler in schedulers:
                    scheduler.step()
                for optimizer in optimizers:
                    optimizer.zero_grad()
                window_weight = 0

        total_step += 1
        if not accelerator.sync_gradients:
            window_weight += weight
        loss_buffer.append(loss.detach())
        if accelerator.is_main_process and len(loss_buffer) == training.log_every_n_train_steps:
            tracker.log_metrics(
                {
                    "loss": {"train": torch.stack(loss_buffer).mean().item()},
                    "lr": {"train": schedulers[0].get_last_lr()[0]},
                },
                iteration=total_step,
            )
            loss_buffer.clear()
    accelerator.wait_for_everyone()

    return total_step, loss_buffer


def _save_model(
    epoch: int,
    cache_dir: Path,
    accelerator: Accelerator,
    encoder_to_embedder: dict[str, core.encoders.Embedder],
    backbone: core.Backbone,
    task_layer: core.tasks.Layer,
) -> None:
    accelerator.wait_for_everyone()
    state_dir = cache_dir / "states" / f"{epoch=:04}"
    for encoder, embedder in encoder_to_embedder.items():
        if sum(p.numel() for p in embedder.state_dict().values()) > 0:
            accelerator.save_model(embedder, state_dir / "encoders" / encoder)
    if sum(p.numel() for p in backbone.state_dict().values()) > 0:
        accelerator.save_model(backbone, state_dir / "backbone")
    if sum(p.numel() for p in task_layer.state_dict().values()) > 0:
        accelerator.save_model(task_layer, state_dir / "task_layer")


@torch.no_grad()
def _test_epoch(
    epoch: int,
    cache_dir: Path,
    accelerator: Accelerator,
    encoder_to_embedder: dict[str, core.encoders.Embedder],
    backbone: core.Backbone,
    task_layer: core.tasks.Layer,
    head: core.tasks.Head,
    evaluator: core.tasks.Evaluator,
    test_dataloader: DataLoader,
    test_transformed_artifacts: t.Any,
    checkpoint: core.checkpoint.Prepared,
    dataset: core.Dataset,
    tracker: core.trackers.Tracker | None,
    *,
    label: str | None = None,
    predictions_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    accelerator.wait_for_everyone()
    for embedder in encoder_to_embedder.values():
        embedder.eval()
    backbone.eval()
    task_layer.eval()

    backbone_feature_to_embedder = {
        feature: encoder_to_embedder[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(event=True, context=True).items()
    }
    artifacts_feature_to_embedder = {
        feature: encoder_to_embedder[encoder]
        for feature, encoder in checkpoint.config.feature_to_encoder(artifacts=True).items()
    }

    with accelerator.autocast():
        embedded_artifacts = dataset.test_artifacts.embed(
            test_transformed_artifacts,
            artifacts_feature_to_embedder,
        )

    predictions = pl.DataFrame()
    desc = f"test {epoch = } / {checkpoint.config.training.num_epochs}"
    if label:
        desc = f"test/{label} {epoch = } / {checkpoint.config.training.num_epochs}"
    for batch in tqdm(test_dataloader, desc=desc):
        with accelerator.autocast():
            backbone_embeddings = backbone(
                backbone_feature_to_embedder,
                batch.events_features,
                batch.events_positions,
                batch.events_timestamps,
                batch.sample_timestamp,
                batch.context_features,
            )
            prediction = head.predict(
                backbone_embeddings,
                embedded_artifacts,
                batch.extras,
                **evaluator.predict_kwargs,
            )
        predictions.vstack(
            pl.DataFrame(
                {
                    "_index": pl.Series("_index", batch.extras["_index"], dtype=pl.UInt32),
                    "prediction": prediction,
                },
            ),
            in_place=True,
        )
    rank_predictions_dir = cache_dir / "predictions"
    rank_predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(rank_predictions_dir / f"rank={accelerator.local_process_index}.pq")

    accelerator.wait_for_everyone()
    samples = (
        dataset.test_samples.lazy()
        .join(pl.scan_parquet(rank_predictions_dir), how="inner", on="_index")
        .unique("_index")
        .sort("_index")
        .drop("_index")
        .collect()
    )
    metrics = evaluator.calculate_metrics(samples, dataset.test_artifacts)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if predictions_dir is not None:
            predictions_dir.mkdir(parents=True, exist_ok=True)
            samples.write_parquet(predictions_dir / "predictions.parquet")
        shutil.rmtree(rank_predictions_dir)
        logged = metrics
        if label:
            logged = {
                name: {f"{label}/{group}": value for group, value in groups.items()}
                for name, groups in metrics.items()
            }
        tracker.log_metrics(logged, iteration=epoch)
    accelerator.wait_for_everyone()
    return metrics


def _unwrap(
    accelerator: Accelerator,
    encoder_to_embedder: dict[str, core.encoders.Embedder],
    backbone: core.Backbone,
    task_layer: core.tasks.Layer,
) -> None:
    for embedder in encoder_to_embedder.values():
        embedder.unwrap(accelerator)
    backbone.unwrap(accelerator)
    task_layer.unwrap(accelerator)


def _load_model(
    epoch: int,
    cache_dir: Path,
    encoder_to_embedder: dict[str, core.encoders.Embedder],
    backbone: core.Backbone,
    task_layer: core.tasks.Layer,
) -> None:
    state_dir = cache_dir / "states" / f"{epoch=:04}"
    for encoder, embedder in encoder_to_embedder.items():
        if sum(p.numel() for p in embedder.state_dict().values()) > 0:
            load_checkpoint_in_model(embedder, state_dir / "encoders" / encoder)
    if sum(p.numel() for p in backbone.state_dict().values()) > 0:
        load_checkpoint_in_model(backbone, state_dir / "backbone")
    if sum(p.numel() for p in task_layer.state_dict().values()) > 0:
        load_checkpoint_in_model(task_layer, state_dir / "task_layer")
