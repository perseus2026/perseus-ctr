"""Opt-in context ablation and sequence precision preserve field layout and autocast."""

import copy
import json

import polars as pl
import pytest
import torch

from perseus.api.fit_model import _init_backbone
from perseus.core import Backbone, Config
from perseus.core.backbone.history import HistoryAggregator
from perseus.core.checkpoint import Prepared
from perseus.core.encoders.id import Embedder, Observer, Preprocessor
from tests.unit.core.backbone.history.test_ctr_v1_composition import EST, HY, inputs


@pytest.mark.parametrize("shared_encoder", [False, True])
def test_zero_context_slot_survives_weight_updates_and_checkpoint(tmp_path, shared_encoder):
    observer = Observer()
    observer.observe(pl.Series(["constant", "constant"]))
    preprocessor = Preprocessor.fit(observer)
    encoded = preprocessor.dynamic_transform(preprocessor.static_transform(pl.Series(["constant", "constant"])))
    assert encoded.tolist() == [1, 1]
    candidate = Embedder.init(preprocessor, 4)
    assert candidate(encoded).count_nonzero() > 0
    event = candidate if shared_encoder else Embedder(2, 4)
    model = Backbone(
        4,
        readout_token=False,
        zero_context_features=["candidate_action"],
        context_fields=["candidate_action", "item"],
        event_aggregator_config={"type": "sum", "num_features": 1, "params": {}},
        context_aggregator_config={"type": "identity", "num_features": 2, "params": {}},
        history_aggregator_config={
            "max_events_per_sequence": 2,
            "sequence_encoder": {"type": "identity", "pool": "mean"},
            "nonsequence_encoder": {"type": "identity", "pool": "flatten"},
            "interaction": {"type": "identity", "schedule": "once"},
        },
    )
    embedders = {"event": event, "candidate_action": candidate, "item": Embedder(2, 4)}
    data = (
        {"event": torch.ones(2, 2, dtype=torch.int32)},
        torch.ones(2, 2, dtype=torch.int32),
        torch.ones(2, 2),
        torch.ones(2),
        {"candidate_action": encoded, "item": encoded},
    )
    seen = []

    def capture(_module, arguments):
        seen.append(arguments[0].detach().clone())

    handle = model.context_aggregator.register_forward_pre_hook(capture)
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.01, weight_decay=0.1)
    try:
        for _ in range(2):
            optimizer.zero_grad()
            output = model(embedders, *data)
            assert seen[-1].shape == (2, 2, 4)
            assert seen[-1][:, 0].count_nonzero() == 0
            torch.testing.assert_close(seen[-1][:, 1], embedders["item"](encoded), rtol=0, atol=0)
            torch.testing.assert_close(output[:, -4:], event(data[0]["event"]).mean(1), rtol=0, atol=0)
            output.square().sum().backward()
            assert candidate.embeddings.weight.grad is not None
            if not shared_encoder:
                assert candidate.embeddings.weight.grad.count_nonzero() == 0
            optimizer.step()
    finally:
        handle.remove()
    model.save(tmp_path)
    loaded = Backbone.load(tmp_path)
    assert loaded.zero_context_features == ["candidate_action"]
    torch.testing.assert_close(loaded(embedders, *data), model(embedders, *data), rtol=0, atol=0)
    # Old backbone checkpoints have neither option.
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text())
    config["kwargs"].pop("zero_context_features")
    config_path.write_text(json.dumps(config))
    assert Backbone.load(tmp_path).zero_context_features == []


@pytest.mark.parametrize("recipe", [EST, HY], ids=["est", "hyformer"])
@pytest.mark.parametrize("fp32", [False, True])
def test_sequence_precision_leaves_autocast_enabled(recipe, fp32):
    config = {"max_events_per_sequence": 12, **copy.deepcopy(recipe), "sequence_state_fp32": fp32}
    model = HistoryAggregator(config, 16, 2)
    data = list(inputs())
    data[0] = data[0].bfloat16().requires_grad_()
    data[1][0, :12] = 0
    states, projections = [], []

    def capture_state(_module, arguments, output):
        states.append((arguments[0]["sequence"].dtype, output["sequence"].dtype))

    def capture_projection(_module, _arguments, output):
        projections.append(output.dtype)

    handles = [layer.register_forward_hook(capture_state) for layer in model.sequence.layers]
    handles += [layer.key.register_forward_hook(capture_projection) for layer in model.interaction.layers]
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(*data)
        output.float().square().sum().backward()
    finally:
        for handle in handles:
            handle.remove()
    expected = torch.float32 if fp32 else torch.bfloat16
    assert states == [(expected, expected)] * 3
    assert projections == [torch.bfloat16] * 3
    assert torch.isfinite(output).all()
    assert torch.isfinite(data[0].grad).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_sequence_fp32_once_and_invalid_all_as_sequence():
    config = {
        "max_events_per_sequence": 3,
        "sequence_state_fp32": True,
        "sequence_encoder": {"type": "identity", "pool": "mean"},
        "interaction": {"type": "identity", "schedule": "once"},
    }
    model = HistoryAggregator(config, 4, 0, readout=False)
    values = torch.randn(2, 3, 4, dtype=torch.bfloat16)
    positions = torch.ones(2, 3, dtype=torch.int32)
    assert model(values, positions, positions.float(), None).dtype == torch.float32
    config["interaction"]["schedule"] = "all_as_sequence"
    with pytest.raises(ValueError, match="sequence_state_fp32 requires a separate sequence stream"):
        HistoryAggregator(config, 4, 0, readout=False)


def test_zero_context_feature_typo_is_rejected():
    with pytest.raises(ValueError, match="zero_context_features must name existing context fields"):
        Backbone(
            4,
            event_aggregator_config={"type": "sum", "num_features": 1, "params": {}},
            context_fields=["candidate_action"],
            zero_context_features=["missing"],
        )


def test_options_flow_through_config_and_checkpoint(tmp_path):
    config = Config.model_validate(
        {
            "task": {"type": "classification", "metrics": {}, "layer": {"binary_output": "probabilities"}},
            "events": {"purchase": None},
            "encoders": {"candidate_action": {"type": "id"}},
            "features": {"candidate_action": {"located_in": {"context": True}, "encoder": "candidate_action"}},
            "backbone": {
                "dim": 16,
                "zero_context_features": ["candidate_action"],
                "context_aggregator": {"type": "identity"},
                "history_aggregator": {**copy.deepcopy(EST), "sequence_state_fp32": True},
            },
        },
    )
    path = tmp_path / "config.yaml"
    config.save(path)
    loaded = Config.load(path)
    assert loaded.task.layer["binary_output"] == "probabilities"
    model = _init_backbone(Prepared(loaded, {}))
    model.save(tmp_path / "backbone")
    restored = Backbone.load(tmp_path / "backbone")
    assert restored.zero_context_features == ["candidate_action"]
    assert restored.history_aggregator.sequence_state_fp32 is True
