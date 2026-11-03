import json
import typing as t
from pathlib import Path

import polars as pl
import torch
from torch import nn

from perseus.core import encoders
from perseus.core.tasks import base

TASK_REDUCTIONS = ("mean", "sum")


class Artifacts(base.Artifacts):
    def static_transform(self, _feature_to_preprocessor: dict[str, encoders.Preprocessor]) -> t.Self:
        return self

    def dynamic_transform(self, _feature_to_preprocessor: dict[str, encoders.Preprocessor]) -> None:
        return

    def embed(
        self,
        _transformed: t.Any,
        _feature_to_embedder: dict[str, encoders.Embedder],
        _target: t.Any | None = None,
        /,
    ) -> None:
        return

    def save(self, _path: Path, /) -> None:
        return

    @classmethod
    def load(cls, _path: Path, /) -> t.Self:
        return cls()


class Preprocessor(base.Preprocessor):
    """Fit independent binary or categorical targets in a fixed field order."""

    def __init__(self, classes: dict[str, list[str]], /) -> None:
        self.classes = classes

    @classmethod
    def fit(
        cls,
        values: pl.Series,
        _artifacts: Artifacts,
        _feature_to_observer: dict[str, encoders.Observer],
        /,
    ) -> t.Self:
        fields = {"target": values} if not isinstance(values.dtype, pl.Struct) else {
            name: values.struct.field(name) for name in values.struct.fields
        }
        classes = {name: field.unique().sort().cast(pl.String).to_list() for name, field in fields.items()}
        return cls(classes)

    def static_transform(self, values: pl.Series, /) -> pl.Series:
        if not isinstance(values.dtype, pl.Struct):
            values = pl.DataFrame({"target": values}).to_struct(values.name)
        return pl.DataFrame(
            [
                values.struct.field(name)
                .cast(pl.String)
                .replace_strict(
                    {label: index for index, label in enumerate(labels)},
                    default=None,
                    return_dtype=pl.UInt32,
                )
                .alias(name)
                for name, labels in self.classes.items()
            ],
        ).to_struct(values.name)

    def dynamic_transform(self, value: t.Any) -> t.Any:
        return tuple(value[name] for name in self.classes)

    def collate(self, values: list[t.Any], /) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.long)

    @property
    def batch_transform(self) -> t.Callable[[pl.Series], torch.Tensor] | None:
        return self.dynamic_transform_batch if type(self) is Preprocessor else None

    def dynamic_transform_batch(self, values: pl.Series, /) -> torch.Tensor:
        targets = values.struct.unnest().select(list(self.classes))
        if any(targets.null_count().row(0)):
            raise ValueError("classification targets must not contain null values")
        return targets.to_torch().long()

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        (path / "classification.json").write_text(json.dumps({"targets": self.classes}))

    @classmethod
    def load(cls, path: Path, /) -> t.Self:
        schema = json.loads((path / "classification.json").read_text())
        return cls(schema["targets"])


class Layer(base.Layer):
    def __init__(
        self, dim: int, num_classes: list[int], /, *, hidden: list[int] | None = None,
        binary_output: t.Literal["logits", "probabilities"] = "logits",
    ) -> None:
        super().__init__()
        if binary_output not in {"logits", "probabilities"}:
            raise ValueError("binary_output must be logits or probabilities")
        self.binary_output = binary_output
        self.num_classes = num_classes
        self.layers = nn.ModuleList()
        for count in num_classes:
            blocks = []
            width = dim
            for hidden_dim in hidden or []:
                blocks.extend([nn.Linear(width, hidden_dim), nn.ReLU()])
                width = hidden_dim
            output = nn.Linear(width, 1 if count == 2 else count)
            self.layers.append(nn.Sequential(*blocks, output) if blocks else output)

    @classmethod
    def init(
        cls, preprocessor: Preprocessor, dim: int, /, *, hidden: list[int] | None = None,
        binary_output: t.Literal["logits", "probabilities"] = "logits",
    ) -> t.Self:
        return cls(
            dim, [len(labels) for labels in preprocessor.classes.values()],
            hidden=hidden, binary_output=binary_output,
        )

    def forward(self, backbone_embeddings: torch.Tensor, _embedded_artifacts: t.Any, /) -> list[torch.Tensor]:
        if backbone_embeddings.ndim == 3:
            if backbone_embeddings.shape[1] != len(self.layers):
                raise ValueError(
                    f"backbone produced {backbone_embeddings.shape[1]} task states for {len(self.layers)} targets",
                )
            states = backbone_embeddings.unbind(dim=1)
        else:
            states = [backbone_embeddings] * len(self.layers)
        predictions = [layer(state) for layer, state in zip(self.layers, states, strict=True)]
        if self.binary_output == "probabilities":
            predictions = [
                prediction.sigmoid() if count == 2 else prediction
                for count, prediction in zip(self.num_classes, predictions, strict=True)
            ]
        return predictions


class Criterion(base.Criterion):
    def __init__(self, layer: Layer, /, *, task_reduction: str = "mean") -> None:
        super().__init__()
        if task_reduction not in TASK_REDUCTIONS:
            raise ValueError(f"unknown task_reduction `{task_reduction}`; available: {TASK_REDUCTIONS}")
        self.task_reduction = task_reduction
        self._layer = [layer]

    @classmethod
    def init(
        cls,
        _preprocessor: Preprocessor,
        layer: Layer,
        /,
        *,
        task_reduction: str = "mean",
    ) -> t.Self:
        return cls(layer, task_reduction=task_reduction)

    def forward(
        self,
        backbone_embeddings: torch.Tensor,
        embedded_artifacts: t.Any,
        target: torch.Tensor,
        /,
    ) -> torch.Tensor:
        layer = self._layer[0]
        predictions = layer(backbone_embeddings, embedded_artifacts)
        if target.ndim != 2 or target.shape[-1] != len(predictions):
            raise ValueError("classification target width does not match the number of task columns")
        losses = []
        for count, prediction, values in zip(layer.num_classes, predictions, target.unbind(dim=1), strict=True):
            prediction = prediction.float()
            if count == 2:
                if layer.binary_output == "probabilities":
                    with torch.autocast(device_type=prediction.device.type, enabled=False):
                        loss = nn.functional.binary_cross_entropy(prediction[:, 0], values.float())
                else:
                    loss = nn.functional.binary_cross_entropy_with_logits(prediction[:, 0], values.float())
            else:
                loss = nn.functional.cross_entropy(prediction, values)
            losses.append(loss)
        losses = torch.stack(losses)
        return losses.mean() if self.task_reduction == "mean" else losses.sum()


class Head(base.Head):
    def __init__(self, layer: Layer, classes: dict[str, list[str]], /) -> None:
        self.layer = layer
        self.classes = classes

    @classmethod
    def init(cls, preprocessor: Preprocessor, layer: Layer, /) -> t.Self:
        return cls(layer, preprocessor.classes)

    def predict(
        self,
        backbone_embeddings: torch.Tensor,
        embedded_artifacts: t.Any,
        _extras: dict[str, t.Any],
        /,
    ) -> pl.Series:
        self.layer.eval().to(backbone_embeddings.device)

        predictions = self.layer(backbone_embeddings, embedded_artifacts)
        columns = []
        for (name, labels), prediction in zip(self.classes.items(), predictions, strict=True):
            prediction = prediction.float()
            if len(labels) == 2:
                positive = prediction
                if self.layer.binary_output == "logits":
                    positive = positive.sigmoid()
                probabilities = torch.cat([1 - positive, positive], dim=-1)
            else:
                probabilities = prediction.softmax(dim=-1)
            columns.append(
                pl.DataFrame(probabilities.tolist(), schema=dict.fromkeys(labels, pl.Float32), orient="row")
                .to_struct(name),
            )
        return pl.DataFrame(columns).to_struct("prediction")

    def save(self, path: Path, /) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.layer.save(path / "layer")
        (path / "classification.json").write_text(json.dumps({"targets": self.classes}))

    @classmethod
    def load(cls, path: Path, /) -> t.Self:
        layer = Layer.load(path / "layer")
        schema = json.loads((path / "classification.json").read_text())
        return cls(layer, schema["targets"])
