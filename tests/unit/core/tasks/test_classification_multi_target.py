"""Classification registry, structured checkpoints and binary target contracts."""

import json

import polars as pl
import pytest
import torch

from perseus.core import tasks
from perseus.core.tasks.classification.target import Artifacts, Criterion, Head, Layer, Preprocessor

TARGETS = ["like", "cart"]


def _series(rows: list[dict[str, int]]) -> pl.Series:
    return pl.DataFrame({"target": rows})["target"]


def _fit(rows: list[dict[str, int]] | None = None) -> Preprocessor:
    return Preprocessor.fit(_series(rows or [{"like": 1, "cart": 0}, {"like": 0, "cart": 1}]), Artifacts(), {})


class TestRegistry:
    def test_multi_target_registered(self) -> None:
        task = tasks.registry["classification"]
        assert task.preprocessor is Preprocessor
        assert task.layer is Layer
        assert task.criterion is Criterion
        assert task.head is Head
        assert {
            "roc_auc", "logloss", "accuracy", "precision", "recall", "f1", "pr_auc",
        } <= set(task.metrics)


class TestPreprocessor:
    def test_structured_checkpoint_preserves_target_order(self, tmp_path) -> None:
        pre = _fit()
        pre.save(tmp_path)
        loaded = Preprocessor.load(tmp_path)
        assert loaded.classes == pre.classes
        assert list(loaded.classes) == TARGETS

    def test_load_reads_classification_schema(self, tmp_path) -> None:
        (tmp_path / "classification.json").write_text(json.dumps({
            "targets": {"like": ["0", "1"], "cart": ["0", "1"]},
        }))
        loaded = Preprocessor.load(tmp_path)
        assert list(loaded.classes) == TARGETS

    def test_load_rejects_empty_directory(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            Preprocessor.load(tmp_path)


class TestLayer:
    def test_builds_independent_target_layers(self) -> None:
        layer = Layer.init(_fit(), 8)
        assert len(layer.layers) == 2
        assert layer.layers[0].weight.data_ptr() != layer.layers[1].weight.data_ptr()
        assert [(target_layer.in_features, target_layer.out_features) for target_layer in layer.layers] == [
            (8, 1),
            (8, 1),
        ]
        assert [part.shape for part in layer(torch.randn(2, 8), None)] == [(2, 1), (2, 1)]


class TestHead:
    def test_binary_probabilities_and_checkpoint(self, tmp_path) -> None:
        pre = _fit()
        head = Head.init(pre, Layer.init(pre, 4))
        inputs = torch.randn(5, 4)
        out = head.predict(inputs, None, {})
        assert list(head.classes) == TARGETS
        assert out.dtype == pl.Struct(dict.fromkeys(TARGETS, pl.Struct({"0": pl.Float32, "1": pl.Float32})))
        for probas in out.to_list():
            assert all(sum(value.values()) == pytest.approx(1.0) for value in probas.values())

        head.save(tmp_path)
        loaded = Head.load(tmp_path)
        assert loaded.classes == head.classes
        assert loaded.predict(inputs, None, {}).to_list() == out.to_list()
