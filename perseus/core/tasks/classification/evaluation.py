import inspect
import math
import typing as t

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from perseus.core.tasks import base
from perseus.core.tasks.classification.target import Artifacts


class Evaluator(base.Evaluator):
    _configured_metrics: dict[str, base.Metric]

    @property
    def predict_kwargs(self) -> dict[str, t.Any]:
        return {}

    def _prepare_samples(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> pl.DataFrame:
        if not isinstance(samples.schema["target"], pl.Struct):
            samples = samples.with_columns(pl.struct("target").alias("target"))
        targets = samples["prediction"].struct.fields
        if not hasattr(self, "_configured_metrics"):
            self._configured_metrics = self.name_to_metric
        self.name_to_metric = {
            f"{name}/{target}": TargetMetric(metric, target)
            for name, metric in self._configured_metrics.items()
            for target in targets
        }
        return samples.with_columns(
            pl.struct(pl.col("target").struct.field(target).cast(pl.String).alias(target) for target in targets)
            .alias("target"),
            most_probable_label=pl.struct(
                pl.concat_list(pl.col("prediction").struct.field(target).struct.field("*"))
                .list.arg_max()
                .replace_strict(dict(enumerate(samples["prediction"].struct.field(target).struct.fields)))
                .alias(target)
                for target in targets
            ),
        )


class TargetMetric(base.Metric):
    def __init__(self, metric: base.Metric, target: str, /) -> None:
        self.metric = metric
        self.target = target

    def calculate(self, samples: pl.DataFrame, artifacts: Artifacts, /) -> float:
        return self.metric.calculate(
            samples.with_columns(
                pl.col(name).struct.field(self.target).alias(name)
                for name in ("target", "prediction", "most_probable_label")
            ),
            artifacts,
        )


class SklearnScorer(t.Protocol):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs: t.Any) -> float: ...


class SklearnMetric(base.Metric):
    _score_fn: t.ClassVar[SklearnScorer]

    def __init__(self, threshold: float | None = None, **kwargs: t.Any) -> None:
        self.threshold = threshold
        self.kwargs = kwargs

    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        score_fn = type(self)._score_fn  # noqa: SLF001

        if self.threshold is not None:
            if len(samples["prediction"].struct.fields) != 2:
                raise ValueError("threshold can be used only in binary classification case")
            if "pos_label" not in self.kwargs:
                raise ValueError("`pos_label` must be set in binary classification case to use `threshold`")
            pos_label = self.kwargs["pos_label"]
            y_true = (samples["target"] == pos_label).to_numpy()
            y_pred = (samples["prediction"].struct.field(pos_label) > self.threshold).to_numpy()
            kwargs = self.kwargs.copy()
            if "pos_label" in inspect.signature(score_fn).parameters:
                kwargs["pos_label"] = True
            else:
                kwargs.pop("pos_label")
        else:
            y_true = samples["target"].to_numpy()
            y_pred = samples["most_probable_label"].to_numpy()
            kwargs = self.kwargs

        return score_fn(y_true, y_pred, **kwargs)


class Accuracy(SklearnMetric):
    _score_fn = accuracy_score


class Recall(SklearnMetric):
    _score_fn = recall_score


class Precision(SklearnMetric):
    _score_fn = precision_score


class F1(SklearnMetric):
    _score_fn = f1_score


class RocAuc(base.Metric):
    def __init__(self, *, pos_label: str | None = None, **kwargs: t.Any) -> None:
        super().__init__()

        self.pos_label = pos_label
        self.kwargs = kwargs

    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        if len(samples["prediction"].struct.fields) == 2:
            if self.pos_label is None:
                raise ValueError("in binary classification case pos_label must be set")
            y_true = (samples["target"] == self.pos_label).to_numpy()
            y_pred = samples["prediction"].struct.field(self.pos_label).to_numpy()
            labels = None
        else:
            y_true = samples["target"].to_numpy()
            y_pred = (
                samples["prediction"].struct.unnest().select(sorted(samples["prediction"].struct.fields)).to_numpy()
            )
            labels = sorted(samples["prediction"].struct.fields)

        if samples["target"].n_unique() < len(samples["prediction"].struct.fields):
            return math.nan
        kwargs = self.kwargs.copy()
        kwargs.setdefault("labels", labels)
        return roc_auc_score(y_true, y_pred, **kwargs)


class PrAuc(base.Metric):
    def __init__(self, *, pos_label: str | None = None, **kwargs: t.Any) -> None:
        super().__init__()

        self.pos_label = pos_label
        self.kwargs = kwargs

    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        if len(samples["prediction"].struct.fields) == 2:
            if self.pos_label is None:
                raise ValueError("in binary classification case pos_label must be set")
            y_true = (samples["target"] == self.pos_label).to_numpy()
            y_pred = samples["prediction"].struct.field(self.pos_label).to_numpy()
            pos_label = True
        else:
            labels = sorted(samples["prediction"].struct.fields)
            y_true = np.column_stack([(samples["target"] == label).to_numpy() for label in labels])
            y_pred = (
                samples["prediction"].struct.unnest().select(sorted(samples["prediction"].struct.fields)).to_numpy()
            )
            pos_label = 1

        return average_precision_score(y_true, y_pred, pos_label=pos_label, **self.kwargs)


class LogLoss(base.Metric):
    def __init__(self, **kwargs: t.Any) -> None:
        self.kwargs = kwargs

    def calculate(self, samples: pl.DataFrame, _artifacts: Artifacts, /) -> float:
        kwargs = self.kwargs.copy()
        kwargs.setdefault("labels", sorted(samples["prediction"].struct.fields))
        scores = samples["prediction"].struct.unnest().select(
            sorted(samples["prediction"].struct.fields),
        ).to_numpy().astype(np.float64)
        if scores.shape[1] == 2:
            scores = scores[:, 1]
        return log_loss(samples["target"].to_numpy(), scores, **kwargs)
