"""Window bounds, query/head independence and legacy contract of the HeMix windows pooling."""

import io

import pytest
import torch

from perseus.core.backbone.history import HistoryAggregator
from perseus.core.backbone.history.interaction.readout_attention import WINDOW_DEFAULTS, QueryPooling
from perseus.core.backbone.history.interaction.tokenize import MixedHeteroAttention
from perseus.core.config import HistoryInteraction

DIM = 8
CONTEXT = 3
LENGTH = 100


def test_interaction_schema_does_not_restrict_registry_extensions():
    config = HistoryInteraction(type="custom", schedule="once", params={"custom_option": 7})
    assert config.type == "custom"
    assert config.params == {"custom_option": 7}


def pool(**overrides):
    return QueryPooling(DIM, CONTEXT, **(WINDOW_DEFAULTS | overrides))


def batch(length, valid_counts=None, seed=7, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    counts = (length, max(length - 2, 0), 0) if valid_counts is None else valid_counts
    counts = tuple(min(count, length) for count in counts)
    rows = len(counts)
    history = torch.randn(rows, length, DIM, generator=generator, dtype=dtype)
    group = torch.randn(rows, CONTEXT + 1, DIM, generator=generator, dtype=dtype)
    positions = torch.arange(length, 0, -1).unsqueeze(0).expand(rows, length).clone()
    padding = positions > torch.tensor(counts, dtype=positions.dtype).unsqueeze(1)
    return history, group, padding


def ns_tokens(module, group):
    features = group.flatten(start_dim=1)
    return torch.stack([mlp(features) for mlp in module.tokenizer.token_mlps], dim=1)


def legacy_reference(module, history, group, padding):
    """The base-revision forward of QueryPooling at defaults, replicated op by op."""
    tokenizer = module.tokenizer
    valid = (~padding).unsqueeze(-1).to(history.dtype)
    recent = history[:, -20:] * valid[:, -20:]
    earlier = history[:, :-20] * valid[:, :-20]
    ns = ns_tokens(module, group)
    rows = history.size(0)
    recent_queries = torch.cat([ns[:, 2:], tokenizer.cls_tokens_recent.expand(rows, -1, -1)], dim=1)
    earlier_queries = torch.cat([ns[:, :2], tokenizer.cls_tokens_earlier.expand(rows, -1, -1)], dim=1)

    def attend(attention, window, queries):
        if window.size(1) == 0:
            return queries
        batch, length, dim = queries.shape
        query = torch.einsum("bld,ldh->blh", queries, attention.query_weight)
        query = query.view(batch, length, 2, dim // 2).transpose(1, 2)
        key = attention.key(window).view(batch, -1, 2, dim // 2).transpose(1, 2)
        value = attention.value(window).view(batch, -1, 2, dim // 2).transpose(1, 2)
        pooled = torch.nn.functional.scaled_dot_product_attention(query, key, value, scale=(dim // 2) ** -0.5)
        return attention.output(pooled.transpose(1, 2).contiguous().view(batch, length, dim))

    return torch.cat(
        [
            attend(tokenizer.attention_recent, recent, recent_queries),
            attend(tokenizer.attention_earlier, earlier, earlier_queries),
            ns,
        ],
        dim=1,
    )


HEMIX = {
    "nonsequence_encoder": {"type": "stack", "params": {"num_layers": 3, "blocks": [
        {"type": "token_mixing", "params": {"kernel": "low_rank", "num_mix_heads": 2, "low_rank_dim": 16}},
        {"type": "per_token_ffn", "params": {"expansion_factor": 4, "activation": "gelu"}}]}},
    "interaction": {"schedule": "once", "type": "window_attention",
                    "params": {}, "pool": "mean"},
}


def history_config(window_params_override):
    config = {"max_events_per_sequence": LENGTH, **HEMIX}
    config["interaction"] = {
        **HEMIX["interaction"],
        "params": {**window_params_override},
    }
    return config


def composed_inputs(length=LENGTH, seed=23):
    generator = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(2, length + CONTEXT + 1, DIM, generator=generator)
    positions = torch.zeros(2, length + CONTEXT + 1, dtype=torch.long)
    positions[:, -CONTEXT - 1:] = 1
    mask = positions[:, :length, None] == 0
    return embeddings, positions, positions.float(), mask


def test_window_defaults_and_token_count():
    assert WINDOW_DEFAULTS == {
        "recent_length": 20, "num_dynamic_queries": 2, "num_learned_queries": 2, "attention_heads": 2,
    }
    for dynamic, learned in ((1, 1), (2, 2), (3, 1), (1, 4), (4, 2)):
        module = pool(num_dynamic_queries=dynamic, num_learned_queries=learned)
        history, group, padding = batch(LENGTH)
        assert module(history, group, padding).shape == (3, 4 * dynamic + 2 * learned, DIM)


@pytest.mark.parametrize(("dynamic", "learned"), [(2, 2), (3, 1), (1, 4)])
def test_output_layout_keeps_attention_rows_then_ns(dynamic, learned):
    module = QueryPooling(
        DIM, CONTEXT, recent_length=20, num_dynamic_queries=dynamic,
        num_learned_queries=learned, attention_heads=2,
    ).eval()
    history, group, padding = batch(100)
    out = module(history, group, padding)
    rows = 2 * (dynamic + learned)
    assert out.shape == (3, rows + 2 * dynamic, DIM)
    assert torch.equal(out[:, rows:], ns_tokens(module, group))


@pytest.mark.parametrize("length", [0, 1, 5, 20, 21, 25, 100])
def test_matches_legacy_computation_at_defaults(length):
    module = pool().eval()
    history, group, padding = batch(length)
    assert torch.equal(module(history, group, padding), legacy_reference(module, history, group, padding))


def test_fully_empty_history_returns_queries_from_both_windows():
    module = pool().eval()
    history, group, padding = batch(0)
    out = module(history, group, padding)
    rows = group.size(0)
    ns = ns_tokens(module, group)
    recent_queries = torch.cat([ns[:, 2:], module.tokenizer.cls_tokens_recent.expand(rows, -1, -1)], dim=1)
    earlier_queries = torch.cat([ns[:, :2], module.tokenizer.cls_tokens_earlier.expand(rows, -1, -1)], dim=1)
    assert torch.equal(out[:, 0:4], recent_queries)
    assert torch.equal(out[:, 4:8], earlier_queries)
    assert torch.equal(out[:, 8:12], ns)


def test_recent_length_beyond_history_leaves_earlier_window_queries():
    module = QueryPooling(
        DIM, CONTEXT, recent_length=30, num_dynamic_queries=2, num_learned_queries=2, attention_heads=2,
    ).eval()
    history, group, padding = batch(25, valid_counts=(25, 5, 0))
    out = module(history, group, padding)
    rows = group.size(0)
    ns = ns_tokens(module, group)
    earlier_queries = torch.cat([ns[:, :2], module.tokenizer.cls_tokens_earlier.expand(rows, -1, -1)], dim=1)
    assert torch.equal(out[:, 4:8], earlier_queries)
    assert torch.equal(out[:, 8:12], ns)


def test_window_split_routes_events_by_position():
    module = pool().eval()
    length = 30
    history, group, padding = batch(length, valid_counts=(length,) * 3)
    baseline = module(history, group, padding)
    sections = (slice(0, 4), slice(4, 8), slice(8, 12))
    for position, changed_section in ((5, 1), (25, 0)):
        moved = history.clone()
        moved[:, position] += torch.randn(DIM, generator=torch.Generator().manual_seed(position))
        out = module(moved, group, padding)
        assert not torch.equal(out[:, sections[changed_section]], baseline[:, sections[changed_section]])
        for other in range(3):
            if other != changed_section:
                assert torch.equal(out[:, sections[other]], baseline[:, sections[other]])
    regrouped = group.clone()
    regrouped[:, 0] += 1.0
    out = module(history, regrouped, padding)
    assert not torch.equal(out[:, 8:12], baseline[:, 8:12])


def test_padded_values_are_zeroed_but_pad_rows_stay_in_attention():
    module = pool().eval()
    count, extra, length = 30, 10, 40
    generator = torch.Generator().manual_seed(3)
    events = torch.randn(1, count, DIM, generator=generator)
    group = torch.randn(1, CONTEXT + 1, DIM, generator=generator)
    history = torch.zeros(1, length, DIM)
    history[0, -count:] = events
    padding = torch.zeros(1, length, dtype=torch.bool)
    padding[0, : length - count] = True
    baseline = module(history, group, padding)
    garbled = history.clone()
    garbled[0, : length - count] = torch.randn(length - count, DIM, generator=generator) * 50
    assert torch.equal(baseline, module(garbled, group, padding))
    longer_length = length + extra
    longer = torch.zeros(1, longer_length, DIM)
    longer[0, -count:] = events
    longer_padding = torch.zeros(1, longer_length, dtype=torch.bool)
    longer_padding[0, : longer_length - count] = True
    changed = module(longer, group, longer_padding)
    assert torch.equal(baseline[:, 0:4], changed[:, 0:4])
    assert not torch.equal(baseline[:, 4:8], changed[:, 4:8])


@pytest.mark.parametrize("length", [0, 1, 5, 20, 21, 25, 100])
def test_forward_backward_finite_on_mixed_batch(length):
    module = pool().train()
    counts = (length, max(length - 1, 0), 0, min(3, length))
    history, group, padding = batch(length, valid_counts=counts, seed=11)
    history.requires_grad_()
    out = module(history, group, padding)
    assert out.shape == (len(counts), 12, DIM)
    assert torch.isfinite(out).all()
    out.square().mean().backward()
    if length:
        assert history.grad is not None
        assert torch.isfinite(history.grad).all()
        assert history.grad[~padding].abs().sum() > 0
    active = module.tokenizer.named_parameters()
    for name, parameter in active:
        # An empty earlier window bypasses its attention projections.
        if name.startswith("attention_earlier.") and length <= module.recent_length:
            continue
        if name.startswith("attention_recent.") and length == 0:
            continue
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_attention_heads_are_independent_of_query_counts():
    torch.manual_seed(5)
    two_heads = pool().eval()
    variants = {
        heads: QueryPooling(
            DIM, CONTEXT, recent_length=20, num_dynamic_queries=2, num_learned_queries=2, attention_heads=heads,
        ).eval()
        for heads in (1, 4)
    }
    reference = two_heads.state_dict()
    for module in variants.values():
        module.load_state_dict(reference)
        assert list(module.state_dict()) == list(reference)
        assert all(torch.equal(module.state_dict()[name], tensor) for name, tensor in reference.items())
    history, group, padding = batch(25, seed=13)
    outputs = {heads: module(history, group, padding) for heads, module in variants.items()}
    outputs[2] = two_heads(history, group, padding)
    assert all(output.shape == (3, 12, DIM) for output in outputs.values())
    assert not torch.equal(outputs[1], outputs[2])
    assert not torch.equal(outputs[2], outputs[4])
    assert not torch.equal(outputs[1], outputs[4])


def test_query_counts_change_output_rows_with_fixed_heads():
    module = QueryPooling(
        DIM, CONTEXT, recent_length=20, num_dynamic_queries=3, num_learned_queries=1, attention_heads=2,
    ).eval()
    history, group, padding = batch(25)
    assert module(history, group, padding).shape == (3, 14, DIM)
    assert module.tokenizer.attention_earlier.query_weight.shape == (4, DIM, DIM)
    assert module.tokenizer.attention_earlier.num_heads == 2


def test_history_build_applies_window_params():
    default = HistoryAggregator(history_config({}), DIM, CONTEXT).eval()
    pools = [m for m in default.modules() if isinstance(m, QueryPooling)]
    assert len(pools) == 1
    assert pools[0].recent_length == 20
    custom = HistoryAggregator(
        history_config({"recent_length": 30, "num_dynamic_queries": 3,
                        "num_learned_queries": 1, "attention_heads": 4}), DIM, CONTEXT,
    ).eval()
    pool_custom = next(m for m in custom.modules() if isinstance(m, QueryPooling))
    assert pool_custom.recent_length == 30
    assert pool_custom.tokenizer.num_dynamic_queries == 3
    assert pool_custom.tokenizer.attention_earlier.num_heads == 4
    assert pool_custom.tokenizer.cls_tokens_earlier.shape == (1, 1, DIM)
    assert default.output_shape == (DIM,)
    assert custom.output_shape == (DIM,)
    inputs = composed_inputs()
    assert default(*inputs).shape == (2, DIM)
    assert custom(*inputs).shape == (2, DIM)
    assert torch.isfinite(default(*inputs)).all()
    assert torch.isfinite(custom(*inputs)).all()


@pytest.mark.parametrize("params", [
    {"recent_length": 0}, {"recent_length": -5}, {"recent_length": True}, {"recent_length": 2.0},
    {"num_dynamic_queries": 0}, {"num_learned_queries": None}, {"attention_heads": True},
])
def test_rejects_invalid_window_params(params):
    with pytest.raises(ValueError, match=r"recent_length|num_dynamic_queries|num_learned_queries|attention_heads"):
        HistoryAggregator(history_config(params), DIM, CONTEXT)


def test_rejects_head_indivisible_dim():
    with pytest.raises(ValueError, match="divisible"):
        MixedHeteroAttention(6, 4, 4)
    with pytest.raises(ValueError, match="divisible"):
        pool(attention_heads=3)


def test_rejects_window_params_outside_windows_pooling():
    config = history_config({"recent_length": 30})
    config["interaction"]["type"] = "target_attention"
    config["interaction"]["params"]["tokenize"] = {"type": "global_group", "params": {}}
    with pytest.raises(ValueError, match="require window_attention interaction"):
        HistoryAggregator(config, DIM, CONTEXT)
    assert HistoryAggregator(history_config({"recent": 5}), DIM, CONTEXT).output_shape is not None


def test_state_dict_roundtrip_preserves_outputs():
    module = pool().eval()
    keys = set(module.state_dict())
    assert {key for key in keys if key.startswith("tokenizer.attention_")} == {
        "tokenizer.attention_earlier.query_weight", "tokenizer.attention_earlier.key.weight",
        "tokenizer.attention_earlier.value.weight", "tokenizer.attention_earlier.output.weight",
        "tokenizer.attention_recent.query_weight", "tokenizer.attention_recent.key.weight",
        "tokenizer.attention_recent.value.weight", "tokenizer.attention_recent.output.weight",
    }
    history, group, padding = batch(50, seed=17)
    baseline = module(history, group, padding)
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    buffer.seek(0)
    restored = pool()
    restored.load_state_dict(torch.load(buffer))
    assert torch.equal(restored(history, group, padding), baseline)


def test_bfloat16_smoke_on_cpu():
    module = pool().to(torch.bfloat16).eval()
    for length in (0, 21, 100):
        history, group, padding = batch(length, seed=19, dtype=torch.bfloat16)
        out = module(history, group, padding)
        assert out.dtype == torch.bfloat16
        assert torch.isfinite(out).all()
