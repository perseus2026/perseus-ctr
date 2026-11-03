# Full GPU training and inference workflow

[Back to the README](../README.md) · [Event Hub and input data](event-hub.md) · [Configuration reference](configuration.md)

This workflow trains a binary classifier on your own data, evaluates it, saves a checkpoint, and predicts on new examples. It covers one NVIDIA GPU and single-node DDP. Dataset size, vocabulary, history length, and hardware determine run time; the settings below are a starting configuration, not a benchmark recipe.

## 1. Install and prepare the input files

Use the [installation instructions](../README.md#installation). Check CUDA from the same environment that will run training:

```bash
unset ACCELERATE_USE_CPU
uv run python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0)); print("BF16:", torch.cuda.is_bf16_supported())'
```

The configuration below uses BF16. If the GPU does not support it, use `mixed_precision: "no"` in the training config and `--mixed_precision no` in the DDP launcher. The CPU quickstart shell script explicitly forces CPU execution; use the commands here for GPU training.

Lay out your files:

```text
events.pq                       # Historical purchases
workdir/
├── config.yaml
└── basis/
    ├── train/samples.pq
    └── test/samples.pq
```

This example expects:

- Events: string `client_id`, `timestamp` as `datetime[ns]`, and string `item`.
- Basis: string `client_id`, prediction `timestamp`, `target` as a struct with integer `label` (0 or 1), and `context` as a struct with string `item` for the candidate.
- Train and test folds containing both classes for ROC AUC. Construct the temporal split before preparation.

See [input schemas and source management](event-hub.md) for details. Register the events once:

```bash
export INTERNAL_STORAGE_EVENT_HUB=/path/to/event-hub
uv run python -m perseus event-hub add-events events.pq --name purchase
uv run python -m perseus event-hub list-events
```

## 2. Write a complete configuration

Save this as `workdir/config.yaml`:

```yaml
task:
  type: classification
  metrics:
    logloss: {}
    roc_auc:
      params: {pos_label: '1'}
  layer:
    hidden: [128, 64]

events:
  purchase:
    attributes:
      item: {}

max_events_per_sequence: 100
embedding_dim: 16

features:
  item:
    located_in: {event: true, context: true}
    encoder:
      type: id
      preprocessor: {min_count: 1}

backbone:
  dim: 128
  readout_token: true
  event_aggregator: {type: sum}
  context_aggregator: {type: tokenizer}
  history_aggregator:
    sequence_encoder:
      type: modern_bert
      params: {num_layers: 2, num_heads: 4, dropout: 0.0}
    interaction:
      type: identity
      schedule: all_as_sequence

training:
  num_epochs: 1
  seed: 42
  log_every_n_train_steps: 100
  event_store_backend: cached
  mixed_precision: bf16
  accumulation: 4
  dataloader:
    batch_size: 256
    num_workers: 4
    pin_memory: true
    shuffle: true
  test_dataloader:
    batch_size: 256
    num_workers: 4
    pin_memory: true
  optimizers:
    default:
      type: adamw
      params: {lr: 0.0001, weight_decay: 0.0}
    embeddings:
      type: adagrad
      params: {lr: 0.05}

inference:
  event_store_backend: ram
  backbone_dataloader: {batch_size: 256, num_workers: 0, pin_memory: true}
  head_dataloader: {batch_size: 1024, num_workers: 0, pin_memory: true}
```

Choose batch sizes that fit the GPU. The item vocabulary determines embedding-table size; `min_count: 1` retains rare IDs too. Replace the backbone with a [starting architecture](../README.md#choose-a-starting-architecture) to explore other compositions.

## 3. Prepare the dataset

Run preparation once, as a single CPU process:

```bash
CUDA_VISIBLE_DEVICES="" uv run python -m perseus train prepare-dataset --workdir workdir/
```

This writes transformed data to `workdir/dataset/` and the prepared config/preprocessors to `workdir/checkpoint/`. GPU training consumes those outputs.

`fit-model` reads **`checkpoint/config.yaml`**, not the top-level `config.yaml`. Use `--overrides` for training-only changes after preparation, as in the DDP example below. For a different feature schema or architecture experiment, prepare a fresh work directory. Each independent run needs its own work directory.

## 4. Train on one GPU or with DDP

Choose one of the following launches. A single process uses GPU 0 and the BF16/accumulation settings from the config:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m perseus train fit-model --workdir workdir/
```

For four GPUs on one machine, launch four processes and reduce accumulation from 4 to 1 to preserve the effective batch size:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run accelerate launch \
  --multi_gpu --num_processes 4 --num_machines 1 --machine_rank 0 \
  --num_cpu_threads_per_process 2 --mixed_precision bf16 --dynamo_backend no \
  -m perseus train fit-model --workdir workdir/ \
  --overrides '{"training":{"accumulation":1}}'
```

These commands are alternative runs from a prepared checkpoint. To compare them, copy the prepared work directory to separate locations before either training run and pass the corresponding path to each command. See the [Accelerate CLI reference](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch) for launcher options.

Batch size and workers are **per process**:

| Launch | Batch per process | Accumulation | Effective batch | Training workers in total |
|---|---:|---:|---:|---:|
| 1 GPU / 1 process | 256 | 4 | 1,024 | 4 |
| 4 GPUs / 4 processes | 256 | 1 | 1,024 | 16 |

For full accumulation windows, `effective batch = batch_size × process count × accumulation`. Partial windows at epoch end can be smaller. A machine allocated eight GPUs can run four processes on four selected devices; the unused GPUs do not contribute compute or GPU memory to those processes. Allocate workers and CPU threads within the machine's CPU/RAM budget. The cached event store occupies memory separately in each training process.

Training evaluates the configured fold, logs metrics, saves the trained checkpoint, and exports test predictions:

```text
workdir/
├── checkpoint/
└── predictions/test/predictions.parquet
```

Inspect the prediction table:

```bash
uv run python -c 'import polars as pl; p = pl.read_parquet("workdir/predictions/test/predictions.parquet"); print(p.shape); print(p.head())'
```

## 5. Predict on new examples

Use a separate inference directory containing `basis/samples.pq`, with the same user ID and context schema as training. Supply explicit prediction timestamps. Classification inference does not require `target`.

```text
inference/
├── basis/samples.pq
└── checkpoint/                 # Copy of the trained checkpoint
```

Create a fresh inference directory and copy the checkpoint and your prediction examples:

```bash
mkdir -p inference/basis
cp -R workdir/checkpoint inference/checkpoint
cp /path/to/new-samples.pq inference/basis/samples.pq
```

Keep `INTERNAL_STORAGE_EVENT_HUB` set to the source holding the relevant historical events. Partition the samples, compute backbone embeddings, then run the classification head:

```bash
CUDA_VISIBLE_DEVICES="" uv run python -m perseus inference distribute-samples --workdir inference/
CUDA_VISIBLE_DEVICES=0 uv run python -m perseus inference make-backbone-embeddings --workdir inference/
CUDA_VISIBLE_DEVICES=0 uv run python -m perseus inference make-head-predictions --workdir inference/
```

The embedding and head stages select the visible CUDA device automatically. Their batch sizes come from the checkpoint's `inference` settings. The outputs are partitioned Parquet files under `inference/embeddings/` and `inference/predictions/`; prediction rows are joined back to the input sample columns.

For multi-GPU inference, run separate processes with disjoint partition bins using `--partitions-num-bins` and `--partitions-bin-idx` (1-based). For example, two processes use bins 1 and 2 with `--partitions-num-bins 2`, each bound to a different GPU. Wait for all embedding processes to finish before launching the head stage with the same bin assignments. The inference CLI partitions work explicitly; launching identical commands through DDP would repeat work on the same output files.
