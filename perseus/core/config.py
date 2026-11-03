import logging
import typing as t
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


class Metric(BaseModel):
    type: str | None = None
    params: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]


class Task(BaseModel):
    type: str
    metrics: dict[str, Metric | None]
    preprocessor: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]
    layer: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]
    criterion: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]
    head: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]

    def model_post_init(self, _context: t.Any) -> None:
        self.metrics = {name: (Metric() if metric is None else metric) for name, metric in self.metrics.items()}
        for name, metric in self.metrics.items():
            if metric.type is None:
                metric.type = name


class Attribute(BaseModel):
    multi: bool = False
    field: str | None = None


class Event(BaseModel):
    attributes: t.Annotated[dict[str, Attribute | None], Field(default_factory=dict)]
    max_tokens: int | None = None
    priority: int = 0
    max_events_per_sequence: int | None = None
    max_duration_per_sequence: str | None = None
    source: str = "event_hub"

    def model_post_init(self, _context: t.Any) -> None:
        self.attributes = {name: (Attribute() if attr is None else attr) for name, attr in self.attributes.items()}
        for name, attr in self.attributes.items():
            if attr.field is None:
                attr.field = name
        self.attributes["event"] = Attribute()

        if any(attr.multi for attr in self.attributes.values()) and self.max_tokens is None:
            raise ValueError("max_tokens must be set if at least one attribute is multi")


class Encoder(BaseModel):
    type: str
    preprocessor: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]
    embedder: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]


class LocatedIn(BaseModel):
    event: bool = False
    context: bool = False
    artifacts: bool = False

    def model_post_init(self, _context: t.Any) -> None:
        if not any([self.event, self.context, self.artifacts]):
            raise ValueError("at least 1 location of event, context or artifacts must be specified")


class Feature(BaseModel):
    located_in: LocatedIn
    encoder: str | Encoder


class Aggregator(BaseModel):
    type: str
    params: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]


HistoryPool = t.Literal["mean", "flatten", "readout"] | t.Annotated[
    dict[t.Literal["select"], t.Annotated[int, Field(gt=0)]], Field(min_length=1, max_length=1),
] | None


class HistoryComponent(BaseModel):
    type: str = "identity"
    params: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]
    finalize_in_stream: bool = False
    pool: HistoryPool = None

class HistoryInteraction(BaseModel):
    schedule: t.Literal["all_as_sequence", "once", "layered"] = "all_as_sequence"
    type: str = "identity"
    params: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]
    independent_layers: bool = False
    pool: HistoryPool = None

    @model_validator(mode="after")
    def _layer_schedule(self) -> "HistoryInteraction":
        if self.independent_layers and self.schedule != "layered":
            raise ValueError("independent_layers requires layered schedule")
        return self



class HistoryAggregator(BaseModel):
    sequence_encoder: HistoryComponent | None = None
    nonsequence_encoder: HistoryComponent | None = None
    interaction: HistoryInteraction | None = None
    num_layers: t.Annotated[int, Field(gt=0)] | None = None
    sequence_state_fp32: bool = False
    layer_order: list[
        t.Literal["sequence_encoder", "nonsequence_encoder", "interaction"]
        | dict[t.Literal["nonsequence_encoder"], list[t.Annotated[int, Field(strict=True, ge=0)]]]
    ] | None = None

    @model_validator(mode="after")
    def _non_empty(self) -> "HistoryAggregator":
        if all(component is None for component in (self.sequence_encoder, self.nonsequence_encoder, self.interaction)):
            raise ValueError("history_aggregator requires at least one component")
        schedule = self.interaction.schedule if self.interaction else "all_as_sequence"
        if self.num_layers is not None and schedule != "layered":
            raise ValueError("history num_layers requires layered schedule")
        if self.layer_order is not None and schedule != "layered":
            raise ValueError("layer_order requires schedule layered")
        if self.layer_order is not None:
            if not self.layer_order:
                raise ValueError("layer_order must not be empty")
            if any(isinstance(item, dict) and len(item) != 1 for item in self.layer_order):
                raise ValueError("each sliced operation must name one nonsequence encoder")
            whole = [item for item in self.layer_order if isinstance(item, str)]
            if len(set(whole)) != len(whole):
                raise ValueError("layer_order repeats a component")
            slices = [item["nonsequence_encoder"] for item in self.layer_order if isinstance(item, dict)]
            if slices and "nonsequence_encoder" in whole:
                raise ValueError("nonsequence slices cannot be mixed with a whole step")
        if schedule == "all_as_sequence" and any(
            component and component.pool is not None
            for component in (self.sequence_encoder, self.nonsequence_encoder, self.interaction)
        ):
            raise ValueError("all_as_sequence does not use output pools")
        if self.nonsequence_encoder and self.nonsequence_encoder.finalize_in_stream:
            raise ValueError("finalize_in_stream applies only to sequence_encoder")
        return self

class Prepare(BaseModel):
    allow_empty_sequences: bool = False


class Backbone(BaseModel):
    dim: int
    mask_context_from_history: bool = False
    readout_token: bool = True
    zero_context_features: t.Annotated[list[str], Field(default_factory=list)]
    history_aggregator: HistoryAggregator | None = None
    event_aggregator: t.Annotated[Aggregator, Field(default_factory=lambda: Aggregator(type="weighted_sum"))]
    context_aggregator: Aggregator | None = None

    def model_post_init(self, _context: t.Any) -> None:
        if self.history_aggregator is None:
            self.history_aggregator = HistoryAggregator(sequence_encoder=HistoryComponent(type="modern_bert"))


class Tracker(BaseModel):
    type: str = "logging"
    params: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]


class DataLoader(BaseModel):
    batch_size: int = 1
    num_workers: int = 0
    pin_memory: bool = False


class TrainDataLoader(DataLoader):
    shuffle: bool = False


class InitSpec(BaseModel):
    kind: t.Literal["xavier_normal", "xavier_uniform", "normal", "trunc_normal"]
    std: t.Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None

    @model_validator(mode="after")
    def _validate_std(self) -> "InitSpec":
        if self.kind in ("normal", "trunc_normal") and self.std is None:
            raise ValueError(f"`std` is required for initialization kind `{self.kind}`")
        if self.kind in ("xavier_normal", "xavier_uniform") and self.std is not None:
            raise ValueError(f"`std` is not supported for initialization kind `{self.kind}`")
        return self

class Initialization(BaseModel):
    linear: InitSpec | None = None
    embedding: InitSpec | None = None
    bias: t.Literal["zeros", "keep"] = "zeros"

    @model_validator(mode="after")
    def _non_empty(self) -> "Initialization":
        if self.linear is None and self.embedding is None:
            raise ValueError("initialization requires at least one of `linear` and `embedding` rules")
        return self

class GradientClipping(BaseModel):
    max_norm: t.Annotated[float, Field(gt=0, allow_inf_nan=False)] = 10.0
    scope: t.Literal["dense", "all"] = "dense"
    norm_type: t.Annotated[float, Field(gt=0)] = 2.0


class Optimizer(BaseModel):
    type: str = "adamw"
    params: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]


class Optimizers(BaseModel):
    default: t.Annotated[Optimizer, Field(default_factory=Optimizer)]
    embeddings: Optimizer | None = None


class Scheduler(BaseModel):
    type: str = "constant"
    params: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]


class EarlyStopping(BaseModel):
    metric: str
    group: str = "overall"
    mode: t.Literal["min", "max"] = "max"
    patience: int = 1
    min_delta: float = 0.0


class Training(BaseModel):
    num_epochs: int = 1
    log_every_n_train_steps: int = 10_000
    seed: int = 42
    event_store_backend: t.Literal["disk", "cached"] = "disk"

    tracker: t.Annotated[Tracker, Field(default_factory=Tracker)]
    dataloader: t.Annotated[TrainDataLoader, Field(default_factory=TrainDataLoader)]
    test_dataloader: DataLoader | None = None
    optimizers: t.Annotated[Optimizers, Field(default_factory=Optimizers)]
    scheduler: t.Annotated[Scheduler, Field(default_factory=Scheduler)]
    early_stopping: EarlyStopping | None = None

    initialization: Initialization | None = None
    gradient_clipping: GradientClipping | None = None
    accumulation: t.Annotated[int, Field(gt=0)] | None = None
    mixed_precision: t.Literal["no", "fp16", "bf16"] | None = None


    def model_post_init(self, _context: t.Any) -> None:
        if self.test_dataloader is None:
            self.test_dataloader = DataLoader(
                batch_size=self.dataloader.batch_size,
                num_workers=self.dataloader.num_workers,
                pin_memory=self.dataloader.pin_memory,
            )


class Inference(BaseModel):
    event_store_backend: t.Literal["disk", "ram", "cached"] = "ram"
    predict_kwargs: t.Annotated[dict[str, t.Any], Field(default_factory=dict)]
    backbone_dataloader: t.Annotated[DataLoader, Field(default_factory=DataLoader)]
    head_dataloader: t.Annotated[DataLoader, Field(default_factory=DataLoader)]


class Config(BaseModel):
    task: Task
    events: dict[str, Event | None]
    max_events_per_sequence: int = 512
    max_duration_per_sequence: str | None = None
    embedding_dim: int | None = None
    encoders: t.Annotated[dict[str, Encoder], Field(default_factory=dict)]
    features: t.Annotated[dict[str, Feature], Field(default_factory=dict)]
    backbone: Backbone
    prepare: t.Annotated[Prepare, Field(default_factory=Prepare)]
    training: t.Annotated[Training, Field(default_factory=Training)]
    inference: t.Annotated[Inference, Field(default_factory=Inference)]

    def model_post_init(self, _context: t.Any) -> None:
        if self.embedding_dim is None:
            self.embedding_dim = self.backbone.dim

        self.events = {event: (Event() if desc is None else desc) for event, desc in self.events.items()}

        self.encoders.setdefault("event", Encoder(type="id"))
        self.features["event"] = Feature(located_in=LocatedIn(event=True), encoder="event")
        for feature, desc in self.features.items():
            if isinstance(desc.encoder, Encoder):
                if feature in self.encoders:
                    raise ValueError(f"encoder `{feature}` already exists")
                self.encoders[feature] = desc.encoder
                desc.encoder = feature

        declared_attributes = {name for event in self.events.values() for name in event.attributes}
        feature_attributes = set(self.features.keys())
        if unused_attributes := declared_attributes - feature_attributes:
            raise ValueError(f"event attributes are declared, but unused: {', '.join(unused_attributes)}")

        declared_encoders = set(self.encoders.keys())
        feature_encoders = {feature.encoder for feature in self.features.values()}
        if unused_encoders := declared_encoders - feature_encoders:
            raise ValueError(f"encoders are declared, but unused: {', '.join(unused_encoders)}")
        if unknown_encoders := feature_encoders - declared_encoders:
            raise ValueError(f"unknown encoders: {', '.join(unknown_encoders)}")

        num_context_features = len(self.feature_to_encoder(context=True))
        if num_context_features > 0 and self.backbone.context_aggregator is None:
            raise ValueError("declared context features, but context aggregator is not defined")
        if num_context_features == 0 and self.backbone.context_aggregator is not None:
            raise ValueError("declared no context features, but context aggregator is defined")

    def feature_to_encoder(
        self,
        *,
        event: bool = False,
        context: bool = False,
        artifacts: bool = False,
    ) -> dict[str, str]:
        if not any([event, context, artifacts]):
            raise ValueError("must set at least one location from `event`, `context` and `artifacts`")
        return {
            feature: desc.encoder
            for feature, desc in self.features.items()
            if (desc.located_in.event and event)
            or (desc.located_in.context and context)
            or (desc.located_in.artifacts and artifacts)
        }

    def save(self, path: Path, /) -> None:
        with path.open("w") as f:
            yaml.safe_dump(self.model_dump(), f, indent=2, sort_keys=False, allow_unicode=True)

    @classmethod
    def load(cls, path: Path, /, *, overrides: dict[str, t.Any] | None = None) -> t.Self:
        with path.open() as f:
            base = yaml.safe_load(f)
        if overrides:
            base = _deep_merge(base, overrides)
        config = cls.model_validate(base)
        logger.info(
            "loaded config:\n%s",
            yaml.safe_dump(config.model_dump(), indent=2, sort_keys=False, allow_unicode=True),
        )
        return config


def _deep_merge(base: dict[str, t.Any], override: dict[str, t.Any], /) -> dict[str, t.Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out
