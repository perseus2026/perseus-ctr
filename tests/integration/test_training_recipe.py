"""Training recipe integration: lifecycle, accumulation/tail equivalence, env contracts."""

import copy
import datetime as dt
import logging
import typing as t
from unittest.mock import patch

import polars as pl
import pytest
import torch

from perseus import core
from perseus.api.fit_model import _test_epoch, fit_model
from perseus.api.make_head_predictions import make_head_predictions
from perseus.api.prepare_dataset import prepare_dataset
from perseus.core.dataset import Dataset
from perseus.core.event_store import EventStore
from perseus.core.tasks.classification.target import Artifacts as ClsArtifacts
from perseus.core.tasks.classification.target import Preprocessor as ClsPreprocessor
from tests.conftest import MINIMAL_CONFIG
from tests.integration.conftest import setup_purchase_source
from tests.unit.api.conftest import DIM, make_id_preprocessor

EVENT_DATE = dt.datetime(2024, 1, 1)
SAMPLE_DATE = dt.datetime(2024, 2, 1)
LR = 0.5
SEED = 42


@pytest.fixture(autouse=True)
def _clean_accelerate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", raising=False)
    monkeypatch.delenv("ACCELERATE_MIXED_PRECISION", raising=False)


@pytest.fixture(autouse=True)
def _capture_fit_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="perseus.api.fit_model")


def _config(batch_size: int, **training: t.Any) -> core.Config:
    base = copy.deepcopy(MINIMAL_CONFIG)
    base["backbone"] = {
        "dim": DIM,
        "history_aggregator": {"sequence_encoder": {"type": "bert", "params": {"dropout": 0.0}}},
    }
    overrides: dict[str, t.Any] = {
        "seed": SEED,
        "optimizers": {"default": {"type": "sgd", "params": {"lr": LR}}},
        "dataloader": {"batch_size": batch_size, "shuffle": False, "num_workers": 0},
    }
    overrides.update(training)
    base["training"] = overrides
    return core.Config.model_validate(base)


def _dataset(tmp_path, labels: list[int]) -> Dataset:
    clients = [f"c{index}" for index in range(len(labels))]
    store = EventStore(tmp_path / "events", backend="disk")
    with store.open_writer() as writer:
        writer.write(
            pl.DataFrame(
                {
                    "client_id": [client for client in clients for _ in range(2)],
                    "timestamp": [EVENT_DATE, EVENT_DATE + dt.timedelta(days=1)] * len(clients),
                    "position": [1, 2] * len(clients),
                    "event": [1, 2] * len(clients),
                },
            ),
        )
    return Dataset(
        train_samples=pl.DataFrame(
            {
                "client_id": clients,
                "timestamp": [SAMPLE_DATE] * len(clients),
                "target": [{"label": label} for label in labels],
            },
        ),
        train_artifacts=ClsArtifacts(),
        test_samples=pl.DataFrame(
            {"client_id": clients[:1], "timestamp": [SAMPLE_DATE], "target": [{"label": labels[0]}]},
        ),
        test_artifacts=ClsArtifacts(),
        event_store=store,
        task_preprocessor=ClsPreprocessor.fit(
            pl.DataFrame({"target": [{"label": 0}, {"label": 1}]})["target"],
            ClsArtifacts(),
            {},
        ),
    )


def _state_close(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert set(left) == set(right)
    for name, tensor in left.items():
        torch.testing.assert_close(tensor, right[name], rtol=1e-5, atol=1e-6)


class TestAccumulationEquivalence:
    """Window updates must equal the update of one equivalent full batch (same seed, one update)."""

    def _fit(self, tmp_path, labels: list[int], batch_size: int, accumulation: int | None) -> core.checkpoint.Trained:
        training = {} if accumulation is None else {"accumulation": accumulation}
        prepared = core.checkpoint.Prepared(
            _config(batch_size, **training),
            {"event": make_id_preprocessor()},
        )
        return fit_model(prepared, _dataset(tmp_path, labels))

    def test_uneven_microbatches_equal_merged_batch(self, tmp_path) -> None:
        labels = [0, 1, 1, 0]
        accumulated = self._fit(tmp_path / "a", labels, batch_size=3, accumulation=2)
        merged = self._fit(tmp_path / "b", labels, batch_size=4, accumulation=1)
        _state_close(accumulated.backbone.state_dict(), merged.backbone.state_dict())
        _state_close(accumulated.head.layer.state_dict(), merged.head.layer.state_dict())
        for name, embedder in accumulated.encoder_to_embedder.items():
            _state_close(embedder.state_dict(), merged.encoder_to_embedder[name].state_dict())

    def test_partial_last_window_equals_full_batch(self, tmp_path) -> None:
        labels = [0, 1, 1]
        accumulated = self._fit(tmp_path / "a", labels, batch_size=2, accumulation=3)
        merged = self._fit(tmp_path / "b", labels, batch_size=3, accumulation=1)
        _state_close(accumulated.backbone.state_dict(), merged.backbone.state_dict())
        _state_close(accumulated.head.layer.state_dict(), merged.head.layer.state_dict())
        for name, embedder in accumulated.encoder_to_embedder.items():
            _state_close(embedder.state_dict(), merged.encoder_to_embedder[name].state_dict())

    def test_full_window_accumulation_equals_single_batch_update(self, tmp_path) -> None:
        labels = [0, 1, 1, 0, 1, 0]
        accumulated = self._fit(tmp_path / "a", labels, batch_size=3, accumulation=2)
        single = self._fit(tmp_path / "b", labels, batch_size=6, accumulation=1)
        _state_close(accumulated.backbone.state_dict(), single.backbone.state_dict())
        _state_close(accumulated.head.layer.state_dict(), single.head.layer.state_dict())
        for name, embedder in accumulated.encoder_to_embedder.items():
            _state_close(embedder.state_dict(), single.encoder_to_embedder[name].state_dict())



class TestAccumulationContract:
    @pytest.mark.parametrize(("configured", "effective"), [(None, 2), (2, 4)])
    def test_env_accumulation_matches_explicit_config(self, tmp_path, monkeypatch, configured, effective) -> None:
        labels = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0]
        monkeypatch.setenv("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", str(effective))
        prepared = core.checkpoint.Prepared(
            _config(2, accumulation=configured), {"event": make_id_preprocessor()},
        )
        actual = fit_model(prepared, _dataset(tmp_path / "env", labels))

        monkeypatch.delenv("ACCELERATE_GRADIENT_ACCUMULATION_STEPS")
        prepared = core.checkpoint.Prepared(
            _config(2, accumulation=effective), {"event": make_id_preprocessor()},
        )
        expected = fit_model(prepared, _dataset(tmp_path / "explicit", labels))
        _state_close(actual.backbone.state_dict(), expected.backbone.state_dict())
        _state_close(actual.head.layer.state_dict(), expected.head.layer.state_dict())
        for name, embedder in actual.encoder_to_embedder.items():
            _state_close(embedder.state_dict(), expected.encoder_to_embedder[name].state_dict())


class TestPrecisionContract:
    def test_explicit_precision_overrides_env(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "bf16")
        dataset = _dataset(tmp_path, [0, 1, 1])
        prepared = core.checkpoint.Prepared(_config(2, mixed_precision="no"), {"event": make_id_preprocessor()})
        assert fit_model(prepared, dataset) is not None

    def test_env_drives_when_field_absent(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("ACCELERATE_MIXED_PRECISION", "no")
        prepared = core.checkpoint.Prepared(_config(2), {"event": make_id_preprocessor()})
        assert fit_model(prepared, _dataset(tmp_path, [0, 1, 1])) is not None


class TestLifecycle:
    def test_fit_with_recipe_save_load_predict(self, tmp_path, caplog) -> None:
        caplog.set_level(0)
        dataset = _dataset(tmp_path, [0, 1, 1, 0])
        prepared = core.checkpoint.Prepared(
            _config(
                2,
                accumulation=2,
                mixed_precision="no",
                initialization={
                    "linear": {"kind": "xavier_normal"},
                    "embedding": {"kind": "normal", "std": 1e-4},
                },
                gradient_clipping={"max_norm": 10.0, "scope": "dense"},
                early_stopping={"metric": "logloss/label", "mode": "min"},
            ),
            {"event": make_id_preprocessor()},
        )
        prepared.config.task.layer = {"hidden": [16, 8]}
        trained = fit_model(prepared, dataset)

        repeated = fit_model(prepared, _dataset(tmp_path / "repeated", [0, 1, 1, 0]))
        _state_close(trained.backbone.state_dict(), repeated.backbone.state_dict())
        _state_close(trained.head.layer.state_dict(), repeated.head.layer.state_dict())
        for name, embedder in trained.encoder_to_embedder.items():
            _state_close(embedder.state_dict(), repeated.encoder_to_embedder[name].state_dict())

        trained.save(tmp_path / "trained")
        loaded = core.checkpoint.Trained.load(tmp_path / "trained")

        samples = pl.DataFrame(
            {"_index": [0, 1], "client_id": ["c0", "c1"], "timestamp": [SAMPLE_DATE] * 2},
        )
        embeddings = pl.DataFrame(
            {"_index": [0, 1], "embeddings": [[0.1] * DIM, [0.2] * DIM]},
            schema_overrides={"embeddings": pl.Array(pl.Float32, DIM)},
        )
        first = make_head_predictions(samples, embeddings, trained)
        second = make_head_predictions(samples, embeddings, loaded)
        assert first["prediction"].to_list() == second["prediction"].to_list()


class TestMulticlassEndToEnd:
    def _prepare(self, tmp_path, monkeypatch, labels: list[str]) -> tuple[core.checkpoint.Prepared, core.Dataset]:
        monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(tmp_path / "hub"))
        clients = [f"c{index}" for index in range(len(labels))]
        setup_purchase_source(tmp_path / "hub", clients)
        config = _config(2)
        train = pl.DataFrame(
            {
                "client_id": clients,
                "timestamp": [SAMPLE_DATE] * len(clients),
                "target": labels,
            },
        )
        test = pl.DataFrame({"client_id": clients[:2], "timestamp": [SAMPLE_DATE] * 2, "target": labels[:2]})
        return prepare_dataset(config, train, test)

    @pytest.mark.parametrize("labels", [["10", "20", "30", "10"], ["a", "b", "c", "a"]])
    def test_prepare_fit_evaluate_save_load(self, tmp_path, monkeypatch, labels) -> None:
        prepared, dataset = self._prepare(tmp_path, monkeypatch, labels)
        assert dataset.task_preprocessor.classes["target"] == sorted(set(labels), key=str)
        assert {row["target"] for row in dataset.train_samples["target"].to_list()} == set(
            range(len(set(labels))),
        )
        assert set(dataset.test_samples["target"].to_list()) <= set(labels)

        trained = fit_model(prepared, dataset)
        assert trained.head.classes["target"] == sorted(set(labels), key=str)

        trained.save(tmp_path / "trained")
        loaded = core.checkpoint.Trained.load(tmp_path / "trained")
        assert loaded.head.classes["target"] == sorted(set(labels), key=str)

        samples = pl.DataFrame(
            {"_index": [0, 1], "client_id": [f"c{index}" for index in range(2)], "timestamp": [SAMPLE_DATE] * 2},
        )
        embeddings = pl.DataFrame(
            {"_index": [0, 1], "embeddings": [[0.1] * DIM, [0.2] * DIM]},
            schema_overrides={"embeddings": pl.Array(pl.Float32, DIM)},
        )
        predictions = make_head_predictions(samples, embeddings, loaded)
        fields = set(predictions["prediction"].struct.fields)
        assert fields == {"target"}
        assert set(predictions["prediction"].struct.field("target").struct.fields) == set(labels)


@pytest.mark.parametrize("split", [False, True])
def test_export_reuses_evaluation_and_matches_best_weights(tmp_path, split):
    config = _config(2, num_epochs=2, early_stopping={
        "metric": "logloss/label", "mode": "min", "min_delta": 1e9, "patience": 1,
    })
    config.max_events_per_sequence = 4
    prepared = core.checkpoint.Prepared(config, {"event": make_id_preprocessor()})
    dataset = _dataset(tmp_path / "data", [0, 1, 1, 0])
    dataset.test_samples = pl.concat([dataset.test_samples, dataset.test_samples]).with_columns(
        source_row=pl.Series([0, 1]), source_block=pl.lit("test-000.pq"),
        context=pl.Series([{"item_id": 10}, {"item_id": 11}]),
    )
    if split:
        dataset.test_samples = dataset.test_samples.with_columns(split=pl.Series(["valid", "test"]))
    export_dir = tmp_path / "predictions" / "test"
    with patch("perseus.api.fit_model._test_epoch", wraps=_test_epoch) as evaluate:
        trained = fit_model(prepared, dataset, test_predictions_dir=export_dir)
    assert evaluate.call_count == (3 if split else 2)
    exported = pl.read_parquet(export_dir / "predictions.parquet")
    expected_samples = dataset.test_samples
    if split:
        expected_samples = expected_samples.filter(pl.col("split") == "test")
    assert exported.drop("prediction").equals(expected_samples.drop("_index"))
    assert list(export_dir.iterdir()) == [export_dir / "predictions.parquet"]

    loader = core.dataloaders.samples(
        exported, dataset.event_store.open_reader(config.max_events_per_sequence),
        {"event": prepared.encoder_to_preprocessor["event"]}, {}, fixed_padding=True, batch_size=2,
    )
    batch = next(iter(loader))
    with torch.no_grad():
        embeddings = trained.backbone(
            {"event": trained.encoder_to_embedder["event"]}, batch.events_features, batch.events_positions,
            batch.events_timestamps, batch.sample_timestamp, batch.context_features,
        )
        expected = trained.head.predict(embeddings, None, {})
    assert exported["prediction"].to_list() == expected.to_list()
