"""Pinned hook and accelerate behavior of the composed history runner."""

import accelerate
import torch

from perseus.core import Backbone
from perseus.core.backbone.history import HistoryAggregator
from perseus.core.encoders.id import Embedder

DIM = 16
MAX_EVENTS = 12
NUM_CONTEXT = 2
EVENT_AGGREGATOR = {"type": "sum", "num_features": 1, "params": {}}

FACADE = {"num_layers": 2,
    "sequence_encoder": {"type": "modern_bert", "params": {}, "pool": "readout"},
    "interaction": {"schedule": "layered", "type": "cross_attention",
                    "params": {"projection": "shared", "num_heads": 2, "write": "replace"}},
}


def _composed():
    return HistoryAggregator({"max_events_per_sequence": MAX_EVENTS, **FACADE}, DIM, NUM_CONTEXT)


def _backbone():
    return Backbone(
        DIM,
        history_aggregator_config={**FACADE, "max_events_per_sequence": MAX_EVENTS},
        event_aggregator_config=EVENT_AGGREGATOR,
    )


def _composed_inputs(batch=2):
    generator = torch.Generator().manual_seed(7)
    length = MAX_EVENTS + NUM_CONTEXT + 1
    embeddings = torch.randn(batch, length, DIM, generator=generator)
    positions = torch.arange(length).unsqueeze(0).repeat(batch, 1)
    timestamps = positions.float()
    mask = torch.ones(batch, length, length, dtype=torch.bool)
    return embeddings, positions, timestamps, mask


def _backbone_inputs(batch=2):
    generator = torch.Generator().manual_seed(7)
    events_features = {"event": torch.randint(1, 5, (batch, MAX_EVENTS), dtype=torch.int32, generator=generator)}
    positions = torch.arange(1, MAX_EVENTS + 1).unsqueeze(0).repeat(batch, 1)
    timestamps = positions.float()
    sample_timestamp = torch.full((batch,), float(MAX_EVENTS) + 1.0)
    return events_features, positions, timestamps, sample_timestamp


def test_composed_runner_keeps_inner_aggregator_hooks_silent():
    module = _composed().eval()
    assert module.num_layers == len(module.sequence.layers) == 2
    fired: list[str] = []
    inner = module.sequence.register_forward_hook(lambda _m, _i, _o: fired.append("inner"))
    outer = module.register_forward_hook(lambda _m, _i, _o: fired.append("runner"))
    try:
        output = module(*_composed_inputs())
        output.sum().backward()
    finally:
        inner.remove()
        outer.remove()
    assert fired == ["runner"]
    grads = [p.grad for p in module.sequence.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_bare_aggregator_keeps_forward_hooks():
    module = HistoryAggregator(
        {"max_events_per_sequence": MAX_EVENTS, "sequence_encoder": {"type": "identity"}}, DIM, NUM_CONTEXT,
    )
    fired: list[str] = []
    handle = module.register_forward_hook(lambda _m, _i, _o: fired.append("aggregator"))
    try:
        module(*_composed_inputs())
    finally:
        handle.remove()
    assert fired == ["aggregator"]


def test_once_calls_sequence_forward_hooks():
    model = HistoryAggregator({
        "max_events_per_sequence": MAX_EVENTS,
        "sequence_encoder": {"type": "identity", "pool": "mean"},
        "interaction": {"type": "identity", "schedule": "once"},
    }, DIM, NUM_CONTEXT)
    fired = []
    handle = model.sequence.register_forward_hook(lambda *_: fired.append(True))
    try:
        model(*_composed_inputs())
    finally:
        handle.remove()
    assert fired == [True]


def test_batch_with_fully_padded_row_stays_finite_through_backward():
    plain = _backbone().eval()
    events_features, positions, timestamps, sample_timestamp = _backbone_inputs(batch=3)
    positions[2] = 0
    out = plain({"event": Embedder(10, DIM)}, events_features, positions, timestamps, sample_timestamp)
    assert torch.isfinite(out).all()
    out.sum().backward()
    grads = [p.grad for p in plain.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(g).all() for g in grads)


def test_prepared_backbone_matches_the_plain_one_bitwise():
    plain = _backbone().eval()
    prepared = _backbone().eval()
    prepared.load_state_dict(plain.state_dict())
    accelerator = accelerate.Accelerator(cpu=True)
    prepared.wrap(accelerator)
    fired: list[str] = []
    handle = prepared.register_forward_hook(lambda _m, _i, _o: fired.append("outer"))
    events_features, positions, timestamps, sample_timestamp = _backbone_inputs()
    embedders = {"event": Embedder(10, DIM)}
    try:
        expected = plain(embedders, events_features, positions, timestamps, sample_timestamp)
        actual = prepared(embedders, events_features, positions, timestamps, sample_timestamp)
        actual.sum().backward()
        expected.sum().backward()
    finally:
        handle.remove()
    assert torch.equal(actual, expected)
    assert fired == ["outer"]
    for parameter in prepared.parameters():
        assert parameter.device == accelerator.device
    plain_grads = [p.grad for p in plain.parameters() if p.grad is not None]
    prepared_grads = [p.grad for p in prepared.parameters() if p.grad is not None]
    assert plain_grads
    assert prepared_grads
    for one, other in zip(plain_grads, prepared_grads, strict=True):
        assert torch.equal(one, other)
