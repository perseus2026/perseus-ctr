"""Context-to-history mask ablation: history stops reading context, prediction may still change."""

import pytest
import torch
import yaml

from perseus.core.backbone import Backbone
from perseus.core.config import Config


def make_backbone(mask_context_from_history: bool) -> Backbone:
    torch.manual_seed(3)
    return Backbone(
        8,
        history_aggregator_config={
            "max_events_per_sequence": 6,
            "sequence_encoder": {"type": "modern_bert", "params": {"num_layers": 1}},
        },
        event_aggregator_config={"type": "sum", "num_features": 1, "params": {}},
        context_aggregator_config={"type": "identity", "num_features": 1, "params": {}},
        mask_context_from_history=mask_context_from_history,
    )


def capture_history(backbone: Backbone, context_index: int) -> tuple[torch.Tensor, torch.Tensor]:
    events_embedder = torch.nn.Embedding(7, 8)
    context_embedder = torch.nn.Embedding(5, 8)
    with torch.no_grad():
        events_embedder.weight.copy_(torch.arange(56, dtype=torch.float32).reshape(7, 8))
        context_embedder.weight.copy_(torch.arange(40, dtype=torch.float32).reshape(5, 8))
    events_features = {"event": torch.tensor([[1, 2, 3, 4]])}
    context_features = {"city": torch.tensor([context_index])}
    positions = torch.tensor([[1, 2, 3, 4]])
    timestamps = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    sample_timestamp = torch.tensor([0.5])
    seen = {}
    handle = backbone.history_aggregator.register_forward_hook(
        lambda module, arguments, output: seen.setdefault("history", output),
    )
    try:
        with torch.no_grad():
            out = backbone(
                {"event": events_embedder, "city": context_embedder},
                events_features, positions, timestamps, sample_timestamp, context_features,
            )
    finally:
        handle.remove()
    return seen["history"], out


@pytest.mark.parametrize("mask_context_from_history", [False, True])
def test_history_rows_and_prediction(mask_context_from_history: bool) -> None:
    backbone = make_backbone(mask_context_from_history)
    backbone.eval()
    history_a, prediction_a = capture_history(backbone, context_index=1)
    history_b, prediction_b = capture_history(backbone, context_index=3)
    assert history_a.shape == history_b.shape
    events = 4
    if mask_context_from_history:
        assert torch.equal(history_a[:, :events], history_b[:, :events])
        assert not torch.equal(prediction_a, prediction_b)
    else:
        assert not torch.equal(history_a[:, :events], history_b[:, :events])


def test_default_backbone_does_not_mask_context() -> None:
    backbone = Backbone(
        8,
        history_aggregator_config={"max_events_per_sequence": 6,
                                   "sequence_encoder": {"type": "modern_bert"}},
        event_aggregator_config={"type": "sum", "num_features": 1, "params": {}},
        context_aggregator_config={"type": "identity", "num_features": 1, "params": {}},
    )
    assert backbone.mask_context_from_history is False


def test_config_flag_round_trips() -> None:
    config = Config.model_validate(
        {
            "task": {"type": "classification", "metrics": {"logloss": None}},
            "events": {"purchase": None},
            "backbone": {"dim": 8, "mask_context_from_history": True,
                         "history_aggregator": {"sequence_encoder": {"type": "modern_bert"}}},
            "features": {},
        },
    )
    dumped = yaml.safe_load(yaml.safe_dump(config.model_dump()))
    assert Config.model_validate(dumped).backbone.mask_context_from_history is True
