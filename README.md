# Perseus-CTR

**Composable sequence and feature interaction for recommendation research.**

Perseus-CTR builds on [Perseus](https://github.com/perseus2026/perseus) and makes the architecture of a recommendation model configurable along three axes: sequence encoding, non-sequential feature processing, and interaction between the two. Keep the data and training workflow, then assemble and compare different model recipes in YAML.

[Quickstart](#quickstart) · [GPU training](#train-on-your-own-data-gpu) · [Starting architectures](#choose-a-starting-architecture) · [Extending models](#extend-a-model)

## What comes from Perseus

Perseus provides the foundation for turning heterogeneous user events into representations and task predictions. This project retains its main abstractions and workflow:

| Foundation | What you configure |
|---|---|
| **Basis** | Prediction examples identified by `client_id` and `timestamp`, with a target and optional context or evaluation groups. |
| **Event Hub** | Local, client-partitioned event storage; each source is a directory containing historical events. |
| **Features and encoders** | Categorical IDs (`id`), numerical features (`ple`), and text (`bow`), drawn from events, context, or artifacts. |
| **Tasks** | Classification, regression, retrieval, and ranking, with their heads, losses, and metrics. |
| **Training and inference** | Dataset preparation, checkpoints, predictions, and distributed execution through Accelerate. |

See the **[original Perseus README](https://github.com/perseus2026/perseus/blob/main/README.md)** for the framework overview, Event Hub operations, and task concepts. The original framework is described in [Perseus: A Demo of Modular Personalization over Heterogeneous Event Sequences](https://doi.org/10.1145/3773078.3841295).

The examples below use the configuration schema of this release. In particular, an upstream `history_aggregator.type` / `params` selection becomes a `history_aggregator.sequence_encoder` selection here, with explicit interaction and non-sequential components when needed.

## What Perseus-CTR adds

The history stage separates three architectural choices:

| Axis | Role | Examples available in this release |
|---|---|---|
| `sequence_encoder` | Process the event history. | `identity`, `ffn`, `self_attention`, `modern_bert`, `bert`, `ligr`, `hstu`, `danet`, `mamba` |
| `nonsequence_encoder` | Process context, feature, or query tokens. | `identity`, `ffn`, `token_mixer`, or a `stack` of FFN, mixing, and attention blocks |
| `interaction` | Connect history to context or query tokens. | `identity`, `cross_attention`, `target_attention`, `window_attention` |

A recipe also selects **when** these components run:

- **`all_as_sequence`** jointly encodes history, context, and optional readout tokens with the sequence encoder.
- **`once`** encodes history, applies interaction, then processes the resulting non-sequential tokens.
- **`layered`** repeats the components in an explicit `layer_order`.

Pooling, depth, parameter sharing, normalization, and positional signals are configurable. See the [configuration guide](docs/configuration.md) for execution semantics and training controls.

## Installation

Use **Python 3.12**.[^platforms] From the repository root, install the pinned environment with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --frozen
uv run python -m perseus --help
```

Alternatively, activate a Python 3.12 virtual environment and run `python -m pip install -e .`; omit `uv run` from the commands below.

The default setup uses local files and a logging tracker. GPU training requires a compatible GPU, driver, and PyTorch installation. Mamba additionally requires `mamba-ssm` and `causal-conv1d`; ClearML is an optional tracker available through `pip install -e '.[clearml]'`.

[^platforms]: The documented local workflows use Linux/macOS; CUDA training uses Linux. On Windows, run the benchmarks under WSL2 with repositories and data in the Linux filesystem (for example, `~/...`, not `/mnt/c/...`).

## Quickstart

Run a complete synthetic example on CPU:

```bash
uv run bash examples/quickstart.sh /tmp/perseus-quickstart
```

The script generates twelve users and their events, registers the local event source, prepares train/test data, trains a small classification model for one epoch, and exports test probabilities. It needs no downloaded dataset or pretrained weights.

```text
/tmp/perseus-quickstart/
├── config.yaml
├── basis/                   # Generated train/test examples
├── event-hub/               # Registered historical events
├── dataset/                 # Prepared training data
├── checkpoint/              # Trained model
└── predictions/test/predictions.parquet
```

Use the generated `config.yaml` as a minimal starting point. The labels are synthetic, so its metrics only demonstrate that the workflow runs. The [generator](examples/quickstart.py) shows how the synthetic inputs are created. For real datasets, follow the [Event Hub and input-data guide](docs/event-hub.md).

## Train on your own data (GPU)

The [full GPU workflow](docs/training.md) covers input tables, a complete YAML configuration, dataset preparation, single-GPU or DDP training, evaluation, checkpoint export, and inference on new examples. The [Event Hub guide](docs/event-hub.md) explains storage, feature schemas, and event management.

With events registered and `workdir/config.yaml`, `basis/train/samples.pq`, and `basis/test/samples.pq` prepared as described there, the single-GPU launch is:

```bash
export INTERNAL_STORAGE_EVENT_HUB=/path/to/event-hub
unset ACCELERATE_USE_CPU
CUDA_VISIBLE_DEVICES="" uv run python -m perseus train prepare-dataset --workdir workdir/
CUDA_VISIBLE_DEVICES=0 uv run python -m perseus train fit-model --workdir workdir/
```

The example config uses `training.mixed_precision: bf16` on a BF16-capable GPU. Preparation runs on CPU; training and evaluation use the selected GPU. Outputs are `workdir/checkpoint/` and `workdir/predictions/test/predictions.parquet`.

For multiple GPUs, use the [DDP launch and batch-size example](docs/training.md#4-train-on-one-gpu-or-with-ddp). Worker counts and batch sizes are per process. After preparation, `fit-model` loads the config from `checkpoint/`; use `--overrides` for training-only changes.

## Choose a starting architecture

Start with the composition pattern that matches the question you want to study:

| Starting point | Composition | Useful for exploring |
|---|---|---|
| **Perseus-style sequence model** | `modern_bert` with `all_as_sequence` | A sequence-centered baseline using the inherited workflow. |
| **EST** | Sequence FFN + context FFN + layered cross-attention | Repeated exchange between history and contextual features. |
| **HeMix** | Identity sequence + window attention + non-sequential mixing stack | Recent/earlier history summaries followed by feature mixing. |
| **HyFormer** | Sequence self-attention + persistent query tokens + layered cross-attention | Learned queries that read the encoded history across layers. |

The composition API also supports MixFormer, TokenMixer, and HiFormer recipes. Dataset-specific benchmark configurations and reproduction commands are supplied in the companion experiment package.

The following are **backbone examples** to use inside a complete dataset configuration. They illustrate the composition patterns; benchmark reproduction requires the complete corresponding recipe, including features, padding, initialization, and training settings.

<details open>
<summary><strong>Sequence baseline: ModernBERT</strong></summary>

```yaml
backbone:
  dim: 256
  readout_token: true
  context_aggregator: {type: tokenizer}
  history_aggregator:
    sequence_encoder:
      type: modern_bert
      params: {num_layers: 2, num_heads: 4, dropout: 0.0}
    interaction:
      type: identity
      schedule: all_as_sequence
```

</details>

<details>
<summary><strong>EST: interleave history and context processing</strong></summary>

This example uses context features, such as candidate attributes, declared in `features` and supplied in the basis tables.

```yaml
backbone:
  dim: 256
  readout_token: false
  context_aggregator: {type: tokenizer}
  history_aggregator:
    num_layers: 3
    layer_order: [interaction, sequence_encoder, nonsequence_encoder]
    sequence_encoder:
      type: ffn
      params: {expansion_factor: 4, activation: relu, norm_group: est_ffn}
      pool: mean
    nonsequence_encoder:
      type: ffn
      params: {expansion_factor: 4, activation: relu, norm_group: est_ffn}
      pool: flatten
    interaction:
      type: cross_attention
      schedule: layered
      independent_layers: true
      params: {projection: per_token, reader: context, write: add, num_heads: 2}
```

</details>

<details>
<summary><strong>HeMix: summarize windows, then mix feature tokens</strong></summary>

This example consumes raw context feature embeddings. Declare context features in the configuration and provide their values in each basis example.

```yaml
backbone:
  dim: 256
  readout_token: false
  context_aggregator: {type: identity}
  history_aggregator:
    sequence_encoder: {type: identity}
    interaction:
      type: window_attention
      schedule: once
      params:
        context_source: raw
        recent_length: 20
        num_dynamic_queries: 2
        num_learned_queries: 2
        attention_heads: 2
      pool: mean
    nonsequence_encoder:
      type: stack
      params:
        num_layers: 3
        blocks:
          - type: token_mixing
            params: {kernel: low_rank, num_mix_heads: 2, low_rank_dim: 16}
          - type: per_token_ffn
            params: {expansion_factor: 4, activation: gelu}
```

</details>

## Extend a model

### Reuse the available blocks

Change one architectural axis while keeping the dataset, remaining components, and training protocol fixed. For example, replace HeMix's `sequence_encoder` section with:

```yaml
sequence_encoder:
  type: modern_bert
  params:
    num_layers: 2
    num_heads: 4
    dropout: 0.0
    positional: true
    rope: false
```

This adds history encoding before the existing window attention. `positional` and `rope` are independent: the example uses absolute sinusoidal positions and disables rotary positions. Under `once`, set depth inside the component's `params`; under `layered`, use `history_aggregator.num_layers` for the interleaved depth.

For the non-sequential axis, `stack` accepts `token_mixing`, `per_token_ffn`, `shared_ffn`, `low_rank_attention`, and `self_attention` blocks. For the interaction axis, choose among cross-attention, target-conditioned pooling, and window attention. Replace the old block's parameters along with its type, and keep token counts, widths, masks, and output pooling compatible.

Window attention also supports `projection: per_query_kv`, which uses separate K/V projections for each query. Its default is `per_query_q`. In the `per_query_kv` implementation, `attention_heads` does not change the computation; it is not a head-count tuning parameter for that mode.

### Register your own component

Implement the relevant interface, then register the class under a new name before constructing the model:

| Component | Interface | Registry |
|---|---|---|
| Sequence encoder | [`Aggregator`](perseus/core/backbone/history/sequence/base.py): `prepare`, `step`, `finalize` | [`sequence.registry`](perseus/core/backbone/history/sequence/__init__.py) |
| Non-sequential encoder | [`Encoder`](perseus/core/backbone/history/nonsequence/base.py) | [`nonsequence.registry`](perseus/core/backbone/history/nonsequence/__init__.py) |
| Interaction | [`Interaction`](perseus/core/backbone/history/interaction/base.py) | [`interaction.registry`](perseus/core/backbone/history/interaction/__init__.py) |

For a class implemented in your own package:

```python
from perseus.core.backbone.history import sequence
from my_project.encoders import MySequenceEncoder

sequence.registry["my_sequence"] = MySequenceEncoder
```

Select it with `sequence_encoder.type: my_sequence`. Import the registration code in every training or inference process before model construction; a registry change in a separate Python session does not carry over to the CLI. Feature encoders and tasks remain extensible through the inherited Perseus interfaces.

## Documentation and development

- [Event Hub, sources, and input schemas](docs/event-hub.md)
- [Full GPU workflow: training, DDP, and inference](docs/training.md)
- [Architecture composition and training controls](docs/configuration.md)
- [Contributing and running tests](CONTRIBUTING.md)
- [Original Perseus README](https://github.com/perseus2026/perseus/blob/main/README.md)

## License

Licensed under the [Apache License 2.0](LICENSE).
