"""Contract for the unified sequence encoder interface."""

import torch

from perseus.core.backbone.history import sequence
from perseus.core.backbone.history.sequence.base import Aggregator


def test_sequence_encoders_share_the_aggregator_registry() -> None:
    for name in ("identity", "ffn", "self_attention", "modern_bert"):
        assert issubclass(sequence.registry[name], Aggregator)
    assert not hasattr(sequence, "native")
    assert not hasattr(sequence, "native_signal")


def test_identity_position_signal_is_internal() -> None:
    encoder = sequence.registry["identity"](8, 4, positional=True)
    embeddings = torch.zeros(1, 4, 8)
    positions = torch.tensor([[0, 0, 2, 1]])
    mask = (positions == 0).unsqueeze(1).expand(-1, 4, -1)
    output = encoder(embeddings, positions, torch.zeros_like(positions), mask)
    assert torch.equal(output[:, :2], embeddings[:, :2])
    assert not torch.equal(output[:, 2:], embeddings[:, 2:])
