"""Tests for classification metrics and the Evaluator."""

import math

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import accuracy_score, average_precision_score, log_loss, roc_auc_score

from perseus.core.tasks.classification.evaluation import Accuracy, Evaluator, LogLoss, PrAuc, RocAuc


@pytest.fixture
def samples() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "target": [
                {"click": 1, "like": 0},
                {"click": 0, "like": 1},
                {"click": 1, "like": 1},
                {"click": 0, "like": 0},
            ],
            "prediction": [
                {"click": {"0": 0.1, "1": 0.9}, "like": {"0": 0.9, "1": 0.1}},
                {"click": {"0": 0.8, "1": 0.2}, "like": {"0": 0.2, "1": 0.8}},
                {"click": {"0": 0.3, "1": 0.7}, "like": {"0": 0.4, "1": 0.6}},
                {"click": {"0": 0.7, "1": 0.3}, "like": {"0": 0.3, "1": 0.7}},
            ],
        },
    )


def test_roc_auc_per_target(samples: pl.DataFrame) -> None:
    values = _calculate(RocAuc(pos_label="1"), samples, None)
    assert set(values) == {"click", "like"}
    assert all(0.0 <= value <= 1.0 for value in values.values())


def test_log_loss_per_target(samples: pl.DataFrame) -> None:
    values = _calculate(LogLoss(), samples, None)
    assert set(values) == {"click", "like"}
    assert all(value > 0 for value in values.values())


def test_constant_target_roc_auc_is_nan() -> None:
    samples = pl.DataFrame(
        {
            "target": [{"click": 1}, {"click": 1}],
            "prediction": [{"click": {"0": 0.1, "1": 0.9}}, {"click": {"0": 0.9, "1": 0.1}}],
        },
    )
    assert math.isnan(_calculate(RocAuc(pos_label="1"), samples, None)["click"])


def test_evaluator_overall_and_groups(samples: pl.DataFrame) -> None:
    grouped = samples.with_columns(pl.struct(pl.lit("ctx").alias("source")).alias("groups"))
    result = Evaluator(
        {"roc_auc": RocAuc(pos_label="1"), "logloss": LogLoss()},
    ).calculate_metrics(grouped, None)
    assert result["logloss/click"]["source = ctx"] == result["logloss/click"]["overall"]
    assert set(result["roc_auc/click"]) == {"overall", "source = ctx"}
    for metric_values in result.values():
        assert all(isinstance(value, float) for value in metric_values.values())


def test_target_named_overall_has_unambiguous_group_name() -> None:
    samples = pl.DataFrame(
        {
            "target": [{"overall": 1}, {"overall": 0}],
            "prediction": [{"overall": {"0": 0.1, "1": 0.9}}, {"overall": {"0": 0.9, "1": 0.1}}],
        },
    )
    result = Evaluator({"roc_auc": RocAuc(pos_label="1")}).calculate_metrics(samples, None)
    assert result["roc_auc/overall"]["overall"] == 1.0


@pytest.mark.parametrize("structured", [False, True])
@pytest.mark.parametrize("classes", [["no", "yes"], ["a", "b", "c"]])
def test_scalar_and_structured_classes_match_sklearn(structured, classes):

    probabilities = np.full((len(classes) * 2, len(classes)), 0.1 / (len(classes) - 1))
    labels = classes * 2
    for index in range(len(labels)):
        probabilities[index, index % len(classes)] = 0.9
    # Include an incorrect confident prediction and an ambiguous correct one.
    probabilities[0] = np.roll(probabilities[0], 1)
    probabilities[-1] = 0.4 / (len(classes) - 1)
    probabilities[-1, -1] = 0.6
    predictions = [dict(zip(classes, row, strict=True)) for row in probabilities]
    target_name = "task" if structured else "target"
    frame = pl.DataFrame(
        {
            "target": [{"task": value} for value in labels] if structured else labels,
            "prediction": [{target_name: value} for value in predictions],
        },
    )
    metrics = {
        "accuracy": Accuracy(),
        "logloss": LogLoss(),
        "roc": RocAuc(pos_label=classes[-1], multi_class="ovr"),
        "pr": PrAuc(pos_label=classes[-1]),
    }
    truth = np.array(labels)[:, None] == np.array(classes)
    expected = {
        "accuracy": accuracy_score(labels, np.array(classes)[probabilities.argmax(axis=1)]),
        "logloss": log_loss(labels, probabilities, labels=classes),
        "roc": roc_auc_score(truth[:, -1], probabilities[:, -1])
        if len(classes) == 2
        else roc_auc_score(labels, probabilities, labels=classes, multi_class="ovr"),
        "pr": average_precision_score(truth[:, -1], probabilities[:, -1])
        if len(classes) == 2
        else average_precision_score(truth, probabilities),
    }
    result = Evaluator(metrics).calculate_metrics(frame, None)
    for name, value in expected.items():
        assert result[f"{name}/{target_name}"]["overall"] == pytest.approx(value)


def _calculate(metric, samples, artifacts):
    result = Evaluator({"metric": metric}).calculate_metrics(samples, artifacts)
    return {name.removeprefix("metric/"): groups["overall"] for name, groups in result.items()}


@pytest.mark.parametrize("structured", [False, True])
def test_numeric_class_order_matches_sklearn(structured):
    labels = [2, 10, 30, 2, 30, 10]
    probabilities = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
            [0.2, 0.1, 0.7],
            [0.1, 0.8, 0.1],
        ],
    )
    predictions = [dict(zip(["2", "10", "30"], row, strict=True)) for row in probabilities]
    target_name = "task" if structured else "target"
    samples = pl.DataFrame(
        {
            "target": [{"task": label} for label in labels] if structured else labels,
            "prediction": [{target_name: prediction} for prediction in predictions],
        },
    )
    result = Evaluator({"logloss": LogLoss(), "roc": RocAuc(multi_class="ovr")}).calculate_metrics(
        samples,
        None,
    )
    assert result[f"logloss/{target_name}"]["overall"] == pytest.approx(
        log_loss(labels, probabilities, labels=[2, 10, 30]),
    )
    assert result[f"roc/{target_name}"]["overall"] == pytest.approx(
        roc_auc_score(labels, probabilities, multi_class="ovr"),
    )


def test_binary_logloss_preserves_precision_for_extreme_probabilities() -> None:
    scores = np.array([2.0611537e-9, 2.0611537e-9], dtype=np.float32)
    samples = pl.DataFrame(
        {
            "target": ["0", "1"],
            "prediction": pl.Series(
                [{"0": float(1 - p), "1": float(p)} for p in scores],
                dtype=pl.Struct({"0": pl.Float32, "1": pl.Float32}),
            ),
        },
    )
    expected = log_loss([False, True], scores.astype(np.float64), labels=[False, True])
    assert LogLoss().calculate(samples, None) == pytest.approx(expected, abs=1e-12)


def _predictions_with_correct_probability(samples, probability):
    return pl.Series("prediction", [
        {name: {"0": 1 - score, "1": score} for name, label in row.items()
         for score in [probability if label else 1 - probability]}
        for row in samples["target"].to_list()
    ])


@pytest.mark.parametrize(("correct_probability", "expected"), [(0.9, 1.0), (0.1, 0.0)])
def test_perfect_and_inverted_rankings(correct_probability, expected, samples):
    frame = samples.with_columns(_predictions_with_correct_probability(samples, correct_probability))
    assert _calculate(RocAuc(pos_label="1"), frame, None) == {"click": expected, "like": expected}


@pytest.mark.parametrize("correct_probability", [0.5, 0.9])
def test_binary_logloss_known_value(correct_probability, samples):
    frame = samples.with_columns(_predictions_with_correct_probability(samples, correct_probability))
    for value in _calculate(LogLoss(), frame, None).values():
        assert value == pytest.approx(-math.log(correct_probability))
