"""Conditioning and execution-order checks for once-scheduled history models."""

import copy

import pytest
import torch

from perseus.core.backbone.history import HistoryAggregator
from perseus.core.backbone.history.interaction.readout_attention import QueryPooling, TargetAttentionPooling
from perseus.core.backbone.history.sequence.position import SinPositionSignal
from perseus.core.backbone.history.sequence.self_attention import SequenceAttention
from perseus.core.config import HistoryAggregator as HistoryConfig

DIM = 16
LENGTH = 24
CONTEXT = 4


def section(kind="global_group", *, positional=False, candidate=True, sequence=False):
    params = {} if kind == "windows" else {
        "tokenize": {"type": kind, "params": {}},
    }
    if kind == "field_task":
        params["tokenize"]["params"]["num_tasks"] = 4
    if kind != "windows" and candidate:
        params["query_context_indices"] = [0, 2]
    encoder = {"type": "identity", "params": {"positional": positional}}
    if sequence:
        encoder = {"type": "self_attention", "params": {
            "num_layers": 1, "num_heads": 2, "positional": positional,
            "ffn": {"expansion_factor": 2, "activation": "relu"},
        }}
    return {
        "max_events_per_sequence": LENGTH,
        "sequence_encoder": encoder,
        "nonsequence_encoder": {"type": "stack", "params": {
            "num_layers": 3, "blocks": [{"type": "shared_ffn", "params": {"expansion_factor": 2}}],
        }},
        "interaction": {
            "schedule": "once", "type": "window_attention" if kind == "windows" else "target_attention",
            "params": params,
                        "pool": {"select": 4} if kind == "field_task" else "mean"},
    }


def batch(dtype=torch.float32):
    embeddings = torch.randn(3, LENGTH + CONTEXT + 1, DIM, dtype=dtype)
    positions = torch.zeros(3, LENGTH + CONTEXT + 1, dtype=torch.long)
    positions[1, LENGTH - 3:LENGTH] = torch.tensor([3, 2, 1])
    positions[2, :LENGTH] = torch.arange(LENGTH, 0, -1)
    padding = positions[:, :LENGTH] == 0
    mask = torch.zeros(3, positions.size(1), positions.size(1), dtype=torch.bool)
    mask[:, :, :LENGTH] = padding.unsqueeze(1)
    return embeddings, positions, positions.float(), mask


@pytest.mark.parametrize("kind", ["field_task", "global_group"])
def test_selected_candidate_conditions_pool_before_tokenization(kind):
    model = HistoryAggregator(section(kind), DIM, CONTEXT).eval()
    arguments = batch()
    arguments[0].requires_grad_()
    pools = [m for m in model.modules() if isinstance(m, TargetAttentionPooling)]
    assert len(pools) == 1
    captured = []
    handle = pools[0].register_forward_hook(lambda _m, _args, out: captured.append(out))
    logits = model(*arguments)
    original = captured[-1]
    gradient = torch.autograd.grad(original.square().sum(), arguments[0], retain_graph=True)[0]
    assert gradient[:, LENGTH + 0].abs().sum() > 0
    assert gradient[:, LENGTH + 2].abs().sum() > 0
    assert torch.count_nonzero(gradient[:, [LENGTH + 1, LENGTH + 3, LENGTH + 4]]) == 0
    assert torch.equal(original[0], torch.zeros_like(original[0]))

    selected = arguments[0].detach().clone()
    selected[:, LENGTH] += torch.randn(DIM)
    model(selected, *arguments[1:])
    assert not torch.equal(original[1:], captured[-1][1:])

    nonselected = arguments[0].detach().clone()
    nonselected[:, [LENGTH + 1, LENGTH + 3, LENGTH + 4]] += torch.randn(DIM)
    model(nonselected, *arguments[1:])
    assert torch.equal(original, captured[-1])
    handle.remove()
    logits.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_default_query_preserves_legacy_operator_and_checkpoint():
    pool = TargetAttentionPooling(DIM, 0.0)
    selected = TargetAttentionPooling(DIM, 0.0, query_context_indices=(0, 2))
    selected.load_state_dict(pool.state_dict(), strict=True)
    assert list(pool.state_dict()) == list(selected.state_dict())
    embeddings, positions, _timestamps, _mask = batch()
    history, context = embeddings[:, :LENGTH], embeddings[:, LENGTH:]
    padding = positions[:, :LENGTH] == 0
    assert torch.equal(pool(history, context, padding), pool.pooling(context[:, -1], history, padding))
    assert torch.equal(
        selected(history, context, padding), pool.pooling(context[:, [0, 2]].sum(1), history, padding),
    )


@pytest.mark.parametrize("indices", [[], [0, 0], [-1], [CONTEXT], [True], [1.0], ["0"], None, "0"])
def test_reject_invalid_query_indices(indices):
    config = section()
    config["interaction"]["params"]["query_context_indices"] = indices
    HistoryConfig.model_validate({k: v for k, v in config.items() if k != "max_events_per_sequence"})
    with pytest.raises(ValueError, match=r"query_context_indices|query context index"):
        HistoryAggregator(config, DIM, CONTEXT)


def test_windows_ignores_unused_candidate_indices():
    config = section("windows")
    config["interaction"]["params"]["query_context_indices"] = [0]
    model = HistoryAggregator(config, DIM, CONTEXT).eval()
    reference = HistoryAggregator(section("windows"), DIM, CONTEXT).eval()
    reference.load_state_dict(model.state_dict(), strict=True)
    assert model.output_shape == reference.output_shape
    arguments = batch()
    torch.testing.assert_close(model(*arguments), reference(*arguments), rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["global_group", "windows"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_position_only_runs_once_before_pool_and_preserves_cold(kind, dtype):
    torch.manual_seed(23)
    plain = HistoryAggregator(section(kind), DIM, CONTEXT).to(dtype).eval()
    rng = torch.random.get_rng_state()
    torch.manual_seed(23)
    positioned = HistoryAggregator(section(kind, positional=True), DIM, CONTEXT).to(dtype).eval()
    assert torch.equal(rng, torch.random.get_rng_state())
    for left, right in zip(plain.parameters(), positioned.parameters(), strict=True):
        assert torch.equal(left, right)

    signal = next(m for m in positioned.modules() if isinstance(m, SinPositionSignal))
    pool = next(m for m in positioned.modules() if isinstance(m, (TargetAttentionPooling, QueryPooling)))
    trace, seen = [], []
    signal_handle = signal.register_forward_hook(lambda _m, _args, _out: trace.append("position"))
    def capture(_module, args):
        trace.append("pool")
        seen.append(args[0])

    pool_handle = pool.register_forward_pre_hook(capture)
    arguments = batch(dtype)
    expected = signal(arguments[0][:, :LENGTH], arguments[1][:, :LENGTH] == 0)
    trace.clear()
    output = positioned(*arguments)
    assert trace == ["position", "pool"]
    assert seen[0].dtype == dtype
    assert torch.equal(seen[0], expected)
    padding = arguments[1][:, :LENGTH] == 0
    assert torch.equal(seen[0][padding], arguments[0][:, :LENGTH][padding])
    assert torch.equal(output[0], plain(*arguments)[0])
    assert not torch.equal(output[1:], plain(*arguments)[1:])
    output.float().square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in positioned.parameters() if p.grad is not None)
    signal_handle.remove()
    pool_handle.remove()


def test_position_precedes_native_sequence_attention_and_pool():
    model = HistoryAggregator(section(positional=True, sequence=True), DIM, CONTEXT)
    trace = []
    handles = [
        module.register_forward_hook(lambda m, _args, _out: trace.append(type(m).__name__))
        for module in model.modules()
        if isinstance(module, (SinPositionSignal, SequenceAttention, TargetAttentionPooling))
    ]
    model(*batch())
    assert trace == ["SinPositionSignal", "SequenceAttention", "TargetAttentionPooling"]
    for handle in handles:
        handle.remove()


@pytest.mark.parametrize("value", ["false", 1, None])
def test_position_only_requires_boolean(value):
    config = section()
    config["sequence_encoder"]["params"]["positional"] = value
    with pytest.raises(ValueError, match="positional must be boolean"):
        HistoryAggregator(config, DIM, CONTEXT)


def test_identity_accepts_position_but_not_other_params():
    config = {"max_events_per_sequence": LENGTH, "sequence_encoder": {
        "type": "identity", "params": {"positional": True},
    }}
    model = HistoryAggregator(config, DIM, CONTEXT)
    assert model.sequence.position is not None
    invalid = section()
    invalid["sequence_encoder"]["params"]["unknown"] = 3
    with pytest.raises(TypeError):
        HistoryAggregator(invalid, DIM, CONTEXT)


def test_explicit_position_off_matches_absent_flag():
    config = section(candidate=False)
    legacy = copy.deepcopy(config)
    del legacy["sequence_encoder"]
    torch.manual_seed(41)
    old = HistoryAggregator(legacy, DIM, CONTEXT)
    torch.manual_seed(41)
    explicit = HistoryAggregator(config, DIM, CONTEXT)
    explicit.load_state_dict(old.state_dict(), strict=True)
    inputs = batch()
    assert torch.equal(old(*inputs), explicit(*inputs))
