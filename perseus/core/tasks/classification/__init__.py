from perseus.core.tasks import base
from perseus.core.tasks.classification.evaluation import (
    F1,
    Accuracy,
    Evaluator,
    LogLoss,
    PrAuc,
    Precision,
    Recall,
    RocAuc,
)
from perseus.core.tasks.classification.target import Artifacts, Criterion, Head, Layer, Preprocessor

__all__ = [
    "task",
]

task = base.Task(
    Artifacts,
    Preprocessor,
    Layer,
    Criterion,
    Head,
    Evaluator,
    {
        "roc_auc": RocAuc,
        "logloss": LogLoss,
        "accuracy": Accuracy,
        "precision": Precision,
        "recall": Recall,
        "f1": F1,
        "pr_auc": PrAuc,
    },
)
