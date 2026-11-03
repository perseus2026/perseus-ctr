# Architecture composition and training

A complete configuration defines `task`, `events`, optional `features`/`encoders`, `backbone` and `training`. The built-in `event` categorical feature is added automatically.

## History stage

Under `backbone.history_aggregator`, `sequence_encoder`, `nonsequence_encoder` and `interaction` each select an operator with `type` and keyword arguments with `params`.

- `all_as_sequence` (the default interaction schedule) jointly encodes event, context and optional readout tokens. Component output pools are not used.
- `once` encodes history, applies interaction, then processes the resulting non-sequential tokens.
- `layered` interleaves operators according to `layer_order`, repeated `num_layers` times. Sliced non-sequential steps can run on either side of interaction.

Readout pools include `mean`, `flatten`, `readout` and configured selected states. A pool changes the prediction representation; it is not interchangeable with an encoder. `readout_token` controls the optional learned readout token. Named normalisation groups and parameter sharing are explicit recipe settings.

Sequence encoders expose `prepare`, `step` and `finalize`. Custom operators can be registered through the corresponding component registries; consult the implementations in `perseus/core/backbone/history/` for contracts. Compatibility requires both matching shapes and matching interpretation of states and masks.

ModernBERT supports `rope` and `positional` independently. `positional: true` adds an absolute sinusoidal signal; `rope: false` disables rotary positions. The benchmark baseline and modified recipes must specify the intended combination.

## Training controls

The [full GPU workflow](training.md) provides the complete training configuration, one-GPU and DDP commands, effective-batch calculation, and inference stages. The [Event Hub guide](event-hub.md) defines the input schemas and distinguishes event sources from prepared training caches.

Within `training`, configure `seed`, `num_epochs`, `optimizers`, `scheduler`, `early_stopping`, `initialization`, and `gradient_clipping` for the experiment. `accumulation` and `mixed_precision` control optimization and precision; `dataloader` and `test_dataloader` control per-process batches and workers. Avoid conflicting `ACCELERATE_*` environment overrides when reproducing a fixed recipe.

Multi-feedback classification evaluates each target separately. Complete benchmark protocols and paired-seed statistical tests belong to the companion experiment package.

## Optional components

ClearML is optional (`pip install -e '.[clearml]'`) and only required when explicitly selected as tracker. Logging is the default tracker. Mamba requires an appropriate separate `mamba-ssm`/`causal-conv1d` installation and CUDA build environment. Other operators and the CPU quickstart do not require either integration.
