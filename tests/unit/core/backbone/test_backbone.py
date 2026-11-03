"""Backbone tests: aggregator assembly, forward with/without context, helpers, and save/load."""

import torch

from perseus.core.backbone import Backbone
from perseus.core.encoders.id import Embedder

DIM = 8
EMBEDDING_DIM = 4
MAX_EVENTS = 16


def _embedders() -> dict[str, Embedder]:
    return {"event": Embedder(10, DIM), "city": Embedder(10, DIM)}


def _backbone(*, with_context: bool = False) -> Backbone:
    return Backbone(
        DIM,
        history_aggregator_config={
            "sequence_encoder": {"type": "bert", "params": {"num_heads": 2, "dropout": 0.0}},
            "max_events_per_sequence": MAX_EVENTS,
        },
        event_aggregator_config={"type": "sum", "num_features": 1, "params": {}},
        context_aggregator_config=({"type": "identity", "num_features": 1, "params": {}} if with_context else None),
    )


def _events() -> tuple:
    events_features = {"event": torch.randint(1, 5, (2, 3), dtype=torch.int32)}
    positions = torch.tensor([[1, 2, 3], [1, 2, 0]], dtype=torch.int32)
    timestamps = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 0.0]])
    sample_timestamp = torch.tensor([4.0, 3.0])
    return events_features, positions, timestamps, sample_timestamp


def test_init_builds_components() -> None:
    backbone = _backbone()
    assert backbone.output_shape == (DIM,)
    assert backbone.event_projection is None
    assert hasattr(backbone, "history_aggregator")
    assert hasattr(backbone, "event_aggregator")
    assert backbone.readout_token.shape == (DIM,)
    assert not hasattr(backbone, "context_aggregator")


def test_init_with_context_builds_context_aggregator() -> None:
    assert hasattr(_backbone(with_context=True), "context_aggregator")


def test_forward_without_context() -> None:
    backbone = _backbone().eval()
    out = backbone(_embedders(), *_events())
    assert out.shape == (2, DIM)
    assert torch.isfinite(out).all()


def test_forward_with_context() -> None:
    backbone = _backbone(with_context=True).eval()
    events_features, positions, timestamps, sample_timestamp = _events()
    context_features = {"city": torch.randint(1, 5, (2,), dtype=torch.int32)}
    out = backbone(_embedders(), events_features, positions, timestamps, sample_timestamp, context_features)
    assert out.shape == (2, DIM)
    assert torch.isfinite(out).all()


def _small_backbone(embedding_dim: int) -> Backbone:
    return Backbone(
        DIM,
        history_aggregator_config={
            "sequence_encoder": {"type": "bert", "params": {"num_heads": 2, "dropout": 0.0}},
            "max_events_per_sequence": MAX_EVENTS,
        },
        event_aggregator_config={"type": "sum", "num_features": 1, "params": {}},
        context_aggregator_config={"type": "identity", "num_features": 1, "params": {}},
        embedding_dim=embedding_dim,
    )


def test_forward_with_smaller_embedding_dim() -> None:
    backbone = _small_backbone(EMBEDDING_DIM).eval()
    assert backbone.event_projection is not None
    assert backbone.context_projection is not None
    embedders = {"event": Embedder(10, EMBEDDING_DIM), "city": Embedder(10, EMBEDDING_DIM)}
    events_features, positions, timestamps, sample_timestamp = _events()
    context_features = {"city": torch.randint(1, 5, (2,), dtype=torch.int32)}
    out = backbone(embedders, events_features, positions, timestamps, sample_timestamp, context_features)
    assert out.shape == (2, DIM)
    assert torch.isfinite(out).all()


def test_equal_embedding_dim_keeps_master_path() -> None:
    backbone = _small_backbone(DIM).eval()
    assert backbone.event_projection is None
    assert backbone.context_projection is None
    out = backbone(_embedders(), *_events())
    assert out.shape == (2, DIM)
    assert torch.isfinite(out).all()


class TestHelpers:
    def test_build_positions_without_context(self) -> None:
        backbone = _backbone()
        positions = torch.tensor([[1, 2, 3], [1, 2, 0]], dtype=torch.int32)
        out = backbone._build_positions(positions, 0)
        assert out.shape == (2, 4)
        assert out[:, -1].tolist() == [0, 0]

    def test_build_positions_with_context(self) -> None:
        backbone = _backbone()
        positions = torch.tensor([[1, 2, 3]], dtype=torch.int32)
        out = backbone._build_positions(positions, 2)
        assert out.shape == (1, 6)

    def test_build_timestamps(self) -> None:
        backbone = _backbone()
        timestamps = torch.tensor([[1.0, 2.0, 3.0]])
        sample_timestamp = torch.tensor([9.0])
        out = backbone._build_timestamps(timestamps, sample_timestamp, 0)
        assert out.shape == (1, 4)
        assert out[0, -1].item() == 9.0  # The readout timestamp equals the sample timestamp.

    def test_build_mask_shape_and_diagonal(self) -> None:
        backbone = _backbone()
        positions = torch.tensor([[1, 2, 3], [1, 2, 0]], dtype=torch.int32)
        mask = backbone._build_mask(positions, 0)
        assert mask.shape == (2, 4, 4)
        diag = torch.arange(4)
        assert not mask[:, diag, diag].any()


def test_save_load_roundtrip(tmp_path) -> None:
    backbone = _backbone().eval()
    embedders = _embedders()
    events = _events()
    with torch.no_grad():
        expected = backbone(embedders, *events)
    backbone.save(tmp_path)
    loaded = Backbone.load(tmp_path).eval()
    with torch.no_grad():
        actual = loaded(embedders, *events)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_forward_all_empty_history_with_physical_padding() -> None:
    backbone = _backbone().eval()
    events_features = {"event": torch.zeros(2, MAX_EVENTS, dtype=torch.int32)}
    positions = torch.zeros(2, MAX_EVENTS, dtype=torch.int32)
    timestamps = torch.zeros(2, MAX_EVENTS)
    sample_timestamp = torch.tensor([4.0, 3.0])
    out = backbone(_embedders(), events_features, positions, timestamps, sample_timestamp)
    assert out.shape == (2, DIM)
    assert torch.isfinite(out).all()
