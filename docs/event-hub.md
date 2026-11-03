# Event Hub and input data

[Back to the README](../README.md) · [Full GPU workflow](training.md)

Perseus-CTR inherits the local Event Hub from Perseus. This guide adapts the [original Event Hub documentation](https://github.com/perseus2026/perseus#event-hub) to this release. Each source is a directory of Parquet files partitioned by event type, date, and client partition; no external event service is required.

## Connect a source

Run commands from the repository root after installation. Export the storage path in every shell used for preparation or inference, or put it in the repository's `.env` file:

```bash
export INTERNAL_STORAGE_EVENT_HUB=/path/to/event-hub
```

The variable name is case-insensitive. The default source name is `event_hub`. To add another source:

```bash
export INTERNAL_STORAGE_CUSTOM_EVENTS=/path/to/custom-events
```

Select it with `source: custom_events` in an event configuration, and with `--source custom_events` in Event Hub commands. The directory must be accessible to each process that reads it.

## Register and inspect events

An input event table must have these columns:

| Column | Type | Meaning |
|---|---|---|
| `client_id` | Non-null string | The user identifier, matching the basis tables. |
| `timestamp` | Non-null `datetime[ns]`, without timezone metadata | When the event happened. Use one time convention across all tables, for example UTC. |
| Other columns | Types accepted by their feature encoders | Event attributes, such as `item`, `amount`, or a list of categorical values. |

For example, a `purchase` file can contain `client_id`, `timestamp`, and a string `item` column. Register it under the name used in the model configuration:

```bash
uv run python -m perseus event-hub add-events events.pq --name purchase
uv run python -m perseus event-hub list-events
```

The listing reports the available dates and attributes. For a custom source:

```bash
uv run python -m perseus event-hub add-events custom-events.pq \
  --name purchase --source custom_events
uv run python -m perseus event-hub list-events --source custom_events
```

`add-events` writes each supplied date/client partition to its destination file. Re-importing a partition replaces that file; it does not append to or merge with its existing rows. Supply the complete partition contents when updating a source.

## Reference events and features

This fragment uses a shared item encoder for history and candidate context:

```yaml
events:
  purchase:
    attributes:
      item: {field: item}
    source: event_hub
    max_events_per_sequence: 100
    max_duration_per_sequence: 30d

max_events_per_sequence: 100

features:
  item:
    located_in: {event: true, context: true}
    encoder:
      type: id
      preprocessor: {min_count: 1}
```

`field` maps a feature name to its column in the event table; it defaults to the feature name. `source` defaults to `event_hub`. Every declared attribute needs a feature definition. The built-in `event` feature is added automatically. Context features also require a context aggregator in the backbone, as shown in the [complete training configuration](training.md#2-write-a-complete-configuration).

Use `multi: true` for list-valued attributes. An event with any multi-valued attribute must also set `max_tokens`:

```yaml
events:
  basket:
    attributes:
      items: {multi: true}
    max_tokens: 25
features:
  items:
    located_in: {event: true}
    encoder: {type: id}
```

The top-level `max_events_per_sequence` bounds the combined history. Per-event count and duration limits allow different policies for different event types. The reader excludes events at or after the prediction timestamp. You still need to construct features and targets without future information and choose an appropriate temporal split.

## Basis tables

The basis describes prediction examples, separately from historical events:

| Column | Training | Inference | Meaning |
|---|---|---|---|
| `client_id` | Required | Required | User whose history is read. |
| `timestamp` | Required | Set explicitly for reproducible predictions | Prediction time; inference uses the current time if omitted. |
| `target` | Required | Not required for classification predictions | Task-specific target. Binary classification can use a struct such as `{"label": 1}`. |
| `context` | Required when context features are configured | Same feature schema as training | Struct of candidate/context fields, for example `{"item": "item-17"}`. |

Use string user IDs and nanosecond timestamps consistently. A shared feature encoder, such as `item` above, expects compatible value types in event and context columns.

Training reads `basis/train/samples.pq` and `basis/test/samples.pq`. To separate model selection from final evaluation, the test file can include a `split` column containing both `valid` and `test` rows. Without that column, the same test fold is used for epoch evaluation and final predictions. Retrieval and ranking may additionally require task-specific artifacts under each fold's `artifacts/` directory.

Inference reads `basis/samples.pq`. See the [GPU workflow](training.md) for directory layouts, a complete classification config, and commands through prediction export.

## Delete events

Delete an event type within an inclusive date range:

```bash
uv run python -m perseus event-hub delete-events --name purchase \
  --date-from 2025-01-01 --date-to 2025-06-30
```

Omitting both dates deletes that event type entirely. All management commands accept `--source`.

## Event storage and training caches

Event Hub is the source of raw history. Dataset preparation reads it and writes transformed data to the training work directory. `training.event_store_backend: cached` holds prepared events in each training process's RAM; `disk` reads prepared events from disk. These are separate from the Event Hub source directory. DDP processes each need memory for their own cache.

Inference reads Event Hub again using the trained preprocessors. Keep its source variables configured and its required history available after training.
