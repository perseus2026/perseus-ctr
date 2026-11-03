"""Assemble numeric cached histories directly into a model batch."""

import polars as pl
import torch

from perseus.core.dataloaders.samples import _CollatedSamples, _SamplesDataset
from perseus.core.event_store.cached import EventsReader


def supports(dataset: _SamplesDataset, /) -> bool:
    if type(dataset.events_reader) is not EventsReader:
        return False
    preprocessors = [*dataset.event_feature_to_preprocessor.values(), *dataset.context_feature_to_preprocessor.values()]
    if any(getattr(preprocessor, "tensor_transform", None) is None for preprocessor in preprocessors):
        return False
    if dataset.context_feature_to_preprocessor:
        schema = dataset.samples.schema["context"].to_schema()
        if any(not schema[name].is_numeric() for name in dataset.context_feature_to_preprocessor):
            return False
    if dataset.task_preprocessor is None:
        return True
    return (
        isinstance(dataset.samples.schema.get("target"), pl.Struct)
        and getattr(dataset.task_preprocessor, "batch_transform", None) is not None
    )


def collate(dataset: _SamplesDataset, indices: list[int], /, *, min_event_length: int) -> _CollatedSamples:
    samples = dataset.samples[indices]
    boundaries = samples["timestamp"].dt.epoch("ns").to_list()
    events, lengths = dataset.events_reader.read_batch_tensors(
        samples["client_id"].to_list(),
        boundaries,
        features=list(dataset.event_feature_to_preprocessor),
    )
    lengths = torch.tensor(lengths, dtype=torch.long)
    width = max(min_event_length, int(lengths.max()))
    offsets = lengths.cumsum(0) - lengths
    within = torch.arange(width)[None, :] - (width - lengths[:, None])
    valid = within >= 0
    gather = (offsets[:, None] + within).clamp_min(0)

    def pad(values: torch.Tensor) -> torch.Tensor:
        if not values.size(0):
            return values.new_zeros((len(indices), width, *values.shape[1:]))
        padded = values[gather]
        mask = valid.reshape(*valid.shape, *([1] * (padded.ndim - 2)))
        return padded.masked_fill(~mask, 0)

    features = {
        name: pad(preprocessor.tensor_transform(events[name]))
        for name, preprocessor in dataset.event_feature_to_preprocessor.items()
    }
    context = None
    if dataset.context_feature_to_preprocessor:
        columns = samples["context"].struct.unnest()
        context = {
            name: preprocessor.dynamic_transform(columns[name])
            for name, preprocessor in dataset.context_feature_to_preprocessor.items()
        }
    extras = (
        {name: [samples[name][row] for row in range(len(indices))] for name in dataset.extras if name in samples.schema}
        if dataset.extras
        else None
    )
    return _CollatedSamples(
        events_features=features,
        events_positions=pad(events["position"].int()),
        events_timestamps=pad((events["timestamp"].double() / 1e9).float()),
        sample_timestamp=torch.tensor([value / 1e9 for value in boundaries], dtype=torch.float32),
        context_features=context,
        extras=extras or None,
        target=dataset.task_preprocessor.batch_transform(samples["target"])
        if dataset.task_preprocessor is not None
        else None,
    )
