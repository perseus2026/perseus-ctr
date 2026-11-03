import logging
import os
import typing as t
from collections import defaultdict
from functools import partial

import polars as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from perseus.core.encoders import Preprocessor as EncoderPreprocessor
from perseus.core.event_store import EventsReader
from perseus.core.tasks import Preprocessor as TaskPreprocessor

logger = logging.getLogger(__name__)


def samples(
    samples: pl.DataFrame,
    events_reader: EventsReader,
    event_feature_to_preprocessor: dict[str, EncoderPreprocessor],
    context_feature_to_preprocessor: dict[str, EncoderPreprocessor],
    /,
    *,
    task_preprocessor: TaskPreprocessor | None = None,
    extras: list[str] | None = None,
    fixed_padding: bool = False,
    **kwargs: t.Any,
) -> DataLoader:
    from perseus.core.dataloaders import cached  # noqa: PLC0415 -- assembler shares the sample types below

    minimum = events_reader.max_events_per_sequence if fixed_padding else int(events_reader.allow_empty_sequences)
    dataset = _SamplesDataset(
        samples.select(
            col for col in ["client_id", "timestamp", "context", "target"] + (extras or []) if col in samples.schema
        ),
        events_reader,
        event_feature_to_preprocessor,
        context_feature_to_preprocessor,
        task_preprocessor,
        extras,
    )
    if cached.supports(dataset):
        logger.info("using direct cached batch assembly")
        data = range(len(dataset))
        collate = partial(cached.collate, dataset, min_event_length=minimum)
    else:
        data = dataset
        collate = partial(_collate_samples, task_preprocessor=task_preprocessor, min_event_length=minimum)
    return DataLoader(
        data,
        collate_fn=collate,
        worker_init_fn=_init_worker_samples,
        **kwargs,
    )


class _Sample(t.NamedTuple):
    events_features: dict[str, torch.Tensor]
    events_positions: torch.Tensor
    events_timestamps: torch.Tensor
    sample_timestamp: float
    context_features: dict[str, torch.Tensor] | None
    extras: dict[str, t.Any] | None
    target: t.Any


class _SamplesDataset(Dataset):
    def __init__(
        self,
        samples: pl.DataFrame,
        events_reader: EventsReader,
        event_feature_to_preprocessor: dict[str, EncoderPreprocessor],
        context_feature_to_preprocessor: dict[str, EncoderPreprocessor],
        task_preprocessor: TaskPreprocessor | None = None,
        extras: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.samples = samples
        self.events_reader = events_reader
        self.event_feature_to_preprocessor = event_feature_to_preprocessor
        self.context_feature_to_preprocessor = context_feature_to_preprocessor
        self.task_preprocessor = task_preprocessor
        self.extras = extras

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> _Sample:
        sample = self.samples[index]
        boundary = sample["timestamp"].dt.epoch(time_unit="ns").item()
        events = self.events_reader.read(sample["client_id"].item(), before=boundary)
        features = {
            feature: preprocessor.dynamic_transform(events[feature])
            for feature, preprocessor in self.event_feature_to_preprocessor.items()
        }
        return self._sample(
            sample, features, events["position"].to_torch().int(),
            (events["timestamp"].dt.epoch(time_unit="ns") / 1e9).to_torch().float(),
        )

    def __getitems__(self, indices: list[int]) -> list[_Sample]:
        if not indices:
            return []
        samples = self.samples[indices]
        boundaries = samples["timestamp"].dt.epoch("ns").to_list()
        events, lengths = self.events_reader.read_batch(
            samples["client_id"].to_list(), boundaries,
            features=list(self.event_feature_to_preprocessor),
        )
        features = {
            feature: preprocessor.dynamic_transform_batch(events[feature], lengths)
            for feature, preprocessor in self.event_feature_to_preprocessor.items()
        }
        positions = events["position"].to_torch().int().split(lengths)
        timestamps = (events["timestamp"].dt.epoch("ns") / 1e9).to_torch().float().split(lengths)
        context_features = {}
        if self.context_feature_to_preprocessor:
            context = samples["context"].struct.unnest()
            context_lengths = [1] * len(indices)
            context_features = {
                name: preprocessor.dynamic_transform_batch(context[name], context_lengths)
                for name, preprocessor in self.context_feature_to_preprocessor.items()
            }
        extras = {name: samples[name] for name in self.extras or [] if name in samples.schema}
        targets = samples["target"] if self.task_preprocessor is not None else None
        return [
            _Sample(
                {name: values[row] for name, values in features.items()},
                position,
                timestamp,
                boundaries[row] / 1e9,
                {name: values[row].squeeze(dim=0) for name, values in context_features.items()} or None,
                {name: values[row] for name, values in extras.items()} if self.extras else None,
                self.task_preprocessor.dynamic_transform(targets[row]) if targets is not None else None,
            )
            for row, (position, timestamp) in enumerate(zip(positions, timestamps, strict=True))
        ]

    def _sample(
        self, sample: pl.DataFrame, events_features: dict[str, torch.Tensor],
        events_positions: torch.Tensor, events_timestamps: torch.Tensor,
    ) -> _Sample:
        boundary = sample["timestamp"].dt.epoch(time_unit="ns").item()
        sample_timestamp = boundary / 1e9
        if self.context_feature_to_preprocessor:
            context = sample["context"].struct.unnest()
            context_features = {
                feature: preprocessor.dynamic_transform(context[feature]).squeeze(dim=0)
                for feature, preprocessor in self.context_feature_to_preprocessor.items()
            }
        else:
            context_features = None

        if self.extras:
            extras = {extra: sample[extra].item() for extra in self.extras or [] if extra in sample.schema}
        else:
            extras = None

        if self.task_preprocessor is not None:
            target = self.task_preprocessor.dynamic_transform(sample["target"].item())
        else:
            target = None

        return _Sample(
            events_features,
            events_positions,
            events_timestamps,
            sample_timestamp,
            context_features,
            extras,
            target,
        )


class _CollatedSamples(t.NamedTuple):
    events_features: dict[str, torch.Tensor]
    events_positions: torch.Tensor
    events_timestamps: torch.Tensor
    sample_timestamp: torch.Tensor
    context_features: dict[str, torch.Tensor] | None
    extras: dict[str, list[t.Any]] | None
    target: t.Any


def _collate_samples(
    samples: list[_Sample],
    /,
    *,
    task_preprocessor: TaskPreprocessor | None = None,
    min_event_length: int | None = None,
) -> _CollatedSamples:
    events_features_chunks = defaultdict(list)
    events_positions_chunks = []
    events_timestamps_chunks = []
    sample_timestamp_chunks = []
    context_features_chunks = defaultdict(list)
    extras_chunks = defaultdict(list)
    target_chunks = []
    for sample in samples:
        for feature, values in sample.events_features.items():
            events_features_chunks[feature].append(values)
        events_positions_chunks.append(sample.events_positions)
        events_timestamps_chunks.append(sample.events_timestamps)

        sample_timestamp_chunks.append(sample.sample_timestamp)
        if sample.context_features:
            for feature, values in sample.context_features.items():
                context_features_chunks[feature].append(values)
        if sample.extras:
            for name, value in sample.extras.items():
                extras_chunks[name].append(value)
        if task_preprocessor is not None:
            target_chunks.append(sample.target)

    events_features = {
        feature: _pad_sequences(values, min_length=min_event_length)
        for feature, values in events_features_chunks.items()
    }
    events_positions = _pad_sequences(events_positions_chunks, min_length=min_event_length)
    events_timestamps = _pad_sequences(events_timestamps_chunks, min_length=min_event_length)

    sample_timestamp = torch.tensor(sample_timestamp_chunks, dtype=torch.float32)
    context_features = {feature: torch.stack(values) for feature, values in context_features_chunks.items()} or None
    extras = dict(extras_chunks) if extras_chunks else None

    target = task_preprocessor.collate(target_chunks) if task_preprocessor is not None else None

    return _CollatedSamples(
        events_features,
        events_positions,
        events_timestamps,
        sample_timestamp,
        context_features,
        extras,
        target,
    )


def _pad_sequences(
    sequences: list[torch.Tensor],
    /,
    *,
    value: float = 0,
    side: t.Literal["left", "right"] = "left",
    min_length: int | None = None,
) -> torch.Tensor:
    max_sizes = list(sequences[0].size())
    for seq in sequences[1:]:
        for dim in range(len(max_sizes)):
            max_sizes[dim] = max(seq.size(dim), max_sizes[dim])
    if min_length:
        max_sizes[0] = max(max_sizes[0], min_length)

    padded_sequences = []
    for seq in sequences:
        pad_dims = []
        for dim in range(seq.dim() - 1, -1, -1):
            pad_needed = max_sizes[dim] - seq.size(dim)
            if dim == 0:
                pad_dims.extend([pad_needed, 0] if side == "left" else [0, pad_needed])
            else:
                pad_dims.extend([0, pad_needed])

        padded_sequences.append(nn.functional.pad(seq, pad_dims, value=value) if any(pad_dims) else seq)

    return torch.stack(padded_sequences, dim=0)


def _init_worker_samples(_: int) -> None:
    os.environ["POLARS_MAX_THREADS"] = "1"
