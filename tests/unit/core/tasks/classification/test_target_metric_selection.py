from threading import Lock

import polars as pl

from perseus.core.tasks import base
from perseus.core.tasks.classification.evaluation import Accuracy, Evaluator, RocAuc


def test_metrics_apply_to_each_scalar_target() -> None:
    samples = pl.DataFrame(
        {
            "target": [{"click": 1, "buy": 0}, {"click": 0, "buy": 1}],
            "prediction": [
                {"click": {"0": 0.1, "1": 0.9}, "buy": {"0": 0.9, "1": 0.1}},
                {"click": {"0": 0.8, "1": 0.2}, "buy": {"0": 0.1, "1": 0.9}},
            ],
        },
    ).with_columns(pl.struct(pl.lit("test").alias("source")).alias("groups"))

    configured_metrics = {"auc": RocAuc(pos_label="1"), "accuracy": Accuracy(threshold=0.5, pos_label="1")}
    evaluator = Evaluator(configured_metrics)
    assert Evaluator.calculate_metrics is base.Evaluator.calculate_metrics
    result = evaluator.calculate_metrics(samples, None)
    assert set(result) == {"auc/click", "auc/buy", "accuracy/click", "accuracy/buy"}
    for target_name in ("click", "buy"):
        assert result[f"auc/{target_name}"]["overall"] == 1.0
        assert result[f"auc/{target_name}"]["source = test"] == 1.0
        assert result[f"accuracy/{target_name}"]["overall"] == 1.0
    assert all(isinstance(value, float) for metric in result.values() for value in metric.values())
    assert evaluator.calculate_metrics(samples, None) == result
    assert set(configured_metrics) == {"auc", "accuracy"}
    assert evaluator.name_to_metric["auc/click"] is not evaluator.name_to_metric["auc/buy"]

    renamed = samples.with_columns(
        pl.col("target").struct.rename_fields(["like", "cart"]),
        pl.col("prediction").struct.rename_fields(["like", "cart"]),
    )
    renamed_result = evaluator.calculate_metrics(renamed, None)
    assert set(renamed_result) == {"auc/like", "auc/cart", "accuracy/like", "accuracy/cart"}
    assert evaluator.calculate_metrics(samples, None) == result


def test_custom_metric_receives_scalar_target_and_original_artifacts() -> None:
    artifacts = object()

    class MeanTarget(base.Metric):
        def __init__(self):
            self.calls = []
            self.lock = Lock()

        def calculate(self, samples, actual_artifacts):
            assert actual_artifacts is artifacts
            assert samples["target"].dtype == pl.String
            assert samples["most_probable_label"].dtype == pl.String
            with self.lock:
                self.calls.append(samples.height)
            return float(samples["target"].cast(pl.Float64).mean())

    metric = MeanTarget()
    evaluator = Evaluator({"mean": metric})
    samples = pl.DataFrame({
        "target": [{"click": 1, "buy": 0}, {"click": 1, "buy": 1}],
        "prediction": [
            {"click": {"0": 0.1, "1": 0.9}, "buy": {"0": 0.9, "1": 0.1}},
            {"click": {"0": 0.2, "1": 0.8}, "buy": {"0": 0.1, "1": 0.9}},
        ],
    })
    assert evaluator.calculate_metrics(samples, artifacts) == {
        "mean/click": {"overall": 1.0}, "mean/buy": {"overall": 0.5},
    }
    assert metric.calls == [2, 2]
