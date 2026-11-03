"""Behavioral contracts of the direct history aggregator."""
import pytest
from pydantic import ValidationError
import torch
from torch import nn

from perseus.core.backbone.history import HistoryAggregator, interaction, nonsequence, sequence
from perseus.core.backbone.history.nonsequence import registry as nonsequence_registry
from perseus.core.backbone.history.nonsequence.base import Encoder
from perseus.core.backbone.history.nonsequence.stack import GroupTransform
from perseus.core.backbone.history.sequence.position import SinPositionSignal
from perseus.core.backbone.history.sequence.self_attention import self_attention
from perseus.core.config import HistoryAggregator as HistoryConfig


def inputs(readout=False):
    embeddings = torch.tensor([[[0., 0.], [2., 4.], [4., 6.], [8., 10.]]])
    if readout:
        embeddings = torch.cat([embeddings, torch.tensor([[[11., 13.]]])], 1)
    positions = torch.tensor([[0, 3, 2, 1] + ([0] if readout else [])])
    timestamps = torch.zeros_like(positions)
    mask = torch.zeros(1, positions.shape[1], positions.shape[1], dtype=torch.bool)
    return embeddings, positions, timestamps, mask


def test_standalone_uses_encoder_forward():
    model = HistoryAggregator({
        "max_events_per_sequence": 4,
        "sequence_encoder": {"type": "modern_bert", "params": {"num_heads": 1, "dropout": 0.0}},
    }, 2, 0, readout=True).eval()
    args = inputs(readout=True)
    assert model.output_shape is None
    torch.testing.assert_close(model(*args), model.sequence(*args))


def test_standalone_attention_uses_full_mask_and_keeps_context_valid():
    model = HistoryAggregator({
        "max_events_per_sequence": 2,
        "sequence_encoder": {"type": "self_attention", "params": {"num_heads": 1}},
    }, 4, 1, readout=True).eval()
    torch.manual_seed(9)
    tokens = torch.randn(1, 4, 4)
    positions = torch.tensor([[0, 1, 0, 0]])
    timestamps = torch.zeros_like(positions)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    mask[:, :, 0] = True
    mask[:, 1, 2:] = True
    diagonal = torch.arange(4)
    mask[:, diagonal, diagonal] = False
    original = model(tokens, positions, timestamps, mask)
    altered = tokens.clone()
    altered[:, 0] += 20
    torch.testing.assert_close(model(altered, positions, timestamps, mask)[:, 1:], original[:, 1:])
    altered = tokens.clone()
    altered[:, 2] += torch.tensor([1., -2., 3., -1.])
    changed = model(altered, positions, timestamps, mask)
    torch.testing.assert_close(changed[:, 1], original[:, 1])
    assert not torch.allclose(changed[:, 3], original[:, 3])


def test_standalone_attention_rejects_ambiguous_unmasked_padding():
    model = HistoryAggregator({
        "max_events_per_sequence": 4,
        "sequence_encoder": {"type": "self_attention", "params": {"num_heads": 1, "mask_padding": False}},
    }, 2, 0, readout=False)
    with pytest.raises(ValueError, match="requires mask_padding=true"):
        model(*inputs())


def test_context_attention_ignores_history_padding_even_when_lengths_match():


    layer = GroupTransform(["context"], nn.ModuleList([self_attention(4, num_heads=1)])).eval()
    context = {"context": torch.randn(1, 3, 4)}
    empty_history = torch.tensor([[True, True, True]])
    full_history = torch.tensor([[False, False, False]])
    torch.testing.assert_close(layer(context, empty_history)["context"], layer(context, full_history)["context"])


def test_position_ranks_are_exact_in_bfloat16():

    signal = SinPositionSignal(8, 512)
    padding = torch.zeros(1, 512, dtype=torch.bool)
    actual = signal(torch.zeros(1, 512, 8, dtype=torch.bfloat16), padding)
    expected = signal(torch.zeros(1, 512, 8), padding).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("encoder", ["identity", "ffn"])
def test_standalone_position_encoding_includes_context_and_readout(encoder):
    params = {"positional": True}
    if encoder == "ffn":
        params["zero_padding"] = True
    model = HistoryAggregator({
        "max_events_per_sequence": 4,
        "sequence_encoder": {"type": encoder, "params": params},
    }, 2, 1, readout=True).eval()
    embeddings = torch.zeros(1, 6, 2)
    positions = torch.tensor([[0, 0, 0, 0, 0, 0]])
    mask = torch.zeros(1, 6, 6, dtype=torch.bool)
    output = model(embeddings, positions, torch.zeros_like(positions), mask)
    assert torch.count_nonzero(output[:, -2:]) > 0


def test_unknown_schedule_is_rejected_by_config():
    with pytest.raises(ValidationError):
        HistoryConfig(interaction={"schedule": "unknown"})


def test_identity_once_pools_only_nonpadded_history():
    model = HistoryAggregator({
        "max_events_per_sequence": 4,
        "sequence_encoder": {"type": "identity", "pool": "mean"},
        "interaction": {"type": "identity", "schedule": "once"},
    }, 2, 0, readout=False)
    torch.testing.assert_close(model(*inputs()), torch.tensor([[14 / 3, 20 / 3]]))
    x, pos, ts, mask = inputs()
    torch.testing.assert_close(model(x, pos * 0, ts, mask), torch.zeros(1, 2))


def test_once_finalizes_sequence_before_pooling():
    config = {
        "max_events_per_sequence": 4,
        "sequence_encoder": {"type": "modern_bert", "params": {"num_heads": 1, "dropout": 0.0}, "pool": "mean"},
        "interaction": {"type": "identity", "schedule": "once"},
    }
    model = HistoryAggregator(config, 2, 0, readout=False).eval()
    embeddings, positions, timestamps, mask = inputs()
    encoded = model.sequence(embeddings, positions, timestamps, mask)
    expected = encoded[:, 1:].mean(dim=1)
    torch.testing.assert_close(model(embeddings, positions, timestamps, mask), expected)


def test_readout_pool_shape():
    model = HistoryAggregator({
        "max_events_per_sequence": 4,
        "sequence_encoder": {"type": "identity"},
        "nonsequence_encoder": {"type": "identity", "pool": "readout"},
        "interaction": {"type": "identity", "schedule": "once"},
    }, 2, 0, readout=True)
    assert model.output_shape == (1, 2)
    torch.testing.assert_close(model(*inputs(readout=True)), torch.tensor([[[11., 13.]]]))


def test_readout_pool_requires_readout_token():
    config = {
        "max_events_per_sequence": 4,
        "sequence_encoder": {"type": "identity"},
        "nonsequence_encoder": {"type": "identity", "pool": "readout"},
        "interaction": {"type": "identity", "schedule": "once"},
    }
    with pytest.raises(ValueError, match="requires the backbone readout token"):
        HistoryAggregator(config, 2, 0, readout=False)


def test_layered_component_counts_use_history_count():
    config = {
        "max_events_per_sequence": 4,
        "sequence_encoder": {"type": "ffn", "params": {"num_layers": 2}, "pool": "mean"},
        "nonsequence_encoder": {"type": "ffn", "params": {"num_layers": 3}},
        "interaction": {"type": "identity", "schedule": "layered"},
    }
    model = HistoryAggregator(config, 2, 1, readout=False)
    assert model.num_layers == len(model.sequence.layers) == 1


def test_query_interaction_requires_explicit_generator():
    config = {
        "max_events_per_sequence": 4,
        "nonsequence_encoder": {"type": "ffn", "params": {}},
        "interaction": {"type": "cross_attention", "schedule": "layered", "pool": "mean",
                        "params": {"reader": "queries", "num_tokens": 2, "num_heads": 1}},
    }
    with pytest.raises(ValueError, match="query generator"):
        HistoryAggregator(config, 2, 1, readout=False)
    config["nonsequence_encoder"]["params"]["queries"] = {"type": "persistent", "params": {"num_tokens": 1}}
    with pytest.raises(ValueError, match="same number"):
        HistoryAggregator(config, 2, 1, readout=False)
    config["nonsequence_encoder"]["params"]["queries"]["params"] = {"num_tokens": 2, "rank": 0}
    with pytest.raises(ValueError, match="query rank"):
        HistoryAggregator(config, 2, 1, readout=False)


@pytest.mark.parametrize("params", [{"num_layers": 0}, {"expansion_factor": 0}])
def test_nonsequence_rejects_zero_dimensions(params):
    config = {
        "max_events_per_sequence": 4,
        "nonsequence_encoder": {"type": "ffn", "params": params, "pool": "flatten"},
        "interaction": {"type": "identity", "schedule": "once"},
    }
    with pytest.raises(ValueError, match="positive integer"):
        HistoryAggregator(config, 2, 1, readout=False)


def test_layered_history_count_overrides_component_counts():
    config = {
        "max_events_per_sequence": 4,
        "num_layers": 2,
        "sequence_encoder": {"type": "ffn", "params": {"num_layers": 3}, "pool": "mean"},
        "nonsequence_encoder": {"type": "ffn", "params": {"num_layers": 4}},
        "interaction": {"type": "identity", "schedule": "layered", "independent_layers": True},
    }
    model = HistoryAggregator(config, 2, 1, readout=False)
    assert model.num_layers == len(model.sequence.layers) == len(model.nonsequence.layers) == 2


def test_sequence_sharing_is_explicit():
    config = {
        "max_events_per_sequence": 4,
        "num_layers": 2,
        "sequence_encoder": {"type": "ffn", "params": {"shared_layers": True}, "pool": "mean"},
        "interaction": {"type": "identity", "schedule": "layered", "independent_layers": True},
    }
    model = HistoryAggregator(config, 2, 0, readout=False)
    assert model.num_layers == 2
    assert len(model.sequence.layers) == 1


def test_query_creation_does_not_require_identity_step_in_layer_order():
    model = HistoryAggregator({
        "max_events_per_sequence": 4,
        "num_layers": 1,
        "layer_order": ["interaction"],
        "sequence_encoder": {"type": "identity"},
        "nonsequence_encoder": {
            "type": "identity",
            "params": {"queries": {"type": "persistent", "params": {"num_tokens": 1}}},
        },
        "interaction": {
            "type": "cross_attention", "schedule": "layered", "pool": "mean",
            "params": {"reader": "queries", "num_tokens": 1, "num_heads": 1},
        },
    }, 2, 0, readout=True)
    assert model(*inputs(readout=True)).shape == (1, 2)


def test_reverse_interaction_requires_sequence_step():
    config = {
        "max_events_per_sequence": 4,
        "num_layers": 1,
        "layer_order": ["interaction"],
        "sequence_encoder": {"type": "identity"},
        "nonsequence_encoder": {
            "type": "identity",
            "params": {"queries": {"type": "persistent", "params": {"num_tokens": 1}}},
        },
        "interaction": {
            "type": "cross_attention", "schedule": "layered", "independent_layers": True, "pool": "mean",
            "params": {"reader": "queries", "num_tokens": 1, "num_heads": 1, "reverse_context": True},
        },
    }
    with pytest.raises(ValueError, match="requires sequence_encoder in layer_order"):
        HistoryAggregator(config, 2, 1, readout=True)


def test_shared_sequence_cannot_bind_different_norms_to_one_layer():
    config = {
        "max_events_per_sequence": 4,
        "num_layers": 2,
        "sequence_encoder": {
            "type": "ffn",
            "params": {"shared_layers": True, "norm_group": "shared"},
            "pool": "mean",
        },
        "nonsequence_encoder": {"type": "ffn", "params": {"norm_group": "shared"}},
        "interaction": {"type": "identity", "schedule": "layered", "independent_layers": True},
    }
    with pytest.raises(ValueError, match="already bound"):
        HistoryAggregator(config, 2, 1, readout=False)


def test_layered_history_count_with_identity_sequence():
    config = {
        "max_events_per_sequence": 4,
        "num_layers": 2,
        "sequence_encoder": {"type": "identity", "pool": "mean"},
        "interaction": {"type": "identity", "schedule": "layered"},
    }
    model = HistoryAggregator(config, 2, 0, readout=False)
    assert model.num_layers == 2
    torch.testing.assert_close(model(*inputs()), torch.tensor([[14 / 3, 20 / 3]]))


def test_target_attention_without_tokenizer_returns_vector():
    config = {
        "max_events_per_sequence": 4,
        "sequence_encoder": {"type": "identity"},
        "interaction": {"type": "target_attention", "schedule": "once", "pool": "mean"},
    }
    model = HistoryAggregator(config, 2, 0, readout=True)
    result = model(*inputs(readout=True))
    assert result.shape == (1, 2)
    assert model.output_shape == (2,)


@pytest.mark.parametrize("params", [
    {"sequence_encoder": {"type": "ffn", "params": {"norm_group": "shared"}}},
    {"interaction": {"type": "cross_attention", "schedule": "once", "params": {"reverse_context": True}}},
])
def test_rejects_operations_not_supported_by_once(params):
    params.setdefault("interaction", {"type": "identity", "schedule": "once"})
    def validate_and_build():
        HistoryConfig.model_validate(params)
        HistoryAggregator({"max_events_per_sequence": 4, **params}, 2, 1, readout=False)

    with pytest.raises(ValueError, match=r"norm_group|norm group|reverse_context"):
        validate_and_build()


def test_est_shares_normalization_only_within_each_layer():
    config = {
        "num_layers": 2,
        "max_events_per_sequence": 4,
        "sequence_encoder": {"type": "ffn", "params": {"norm_group": "ffn"}, "pool": "mean"},
        "nonsequence_encoder": {"type": "ffn", "params": {"norm_group": "ffn"}, "pool": "flatten"},
        "interaction": {"type": "cross_attention", "schedule": "layered", "independent_layers": True,
                        "params": {"projection": "per_token", "num_heads": 1}},
        "layer_order": ["interaction", "sequence_encoder", "nonsequence_encoder"],
    }
    model = HistoryAggregator(config, 2, 1, readout=False)
    for layer in range(2):
        sequence_norm = model.sequence.layers[layer].blocks[0]._norm[0]
        context_norm = model.nonsequence.layers[layer].blocks[0]._norm[0]
        assert sequence_norm is context_norm
    assert model.sequence.layers[0].blocks[0]._norm[0] is not model.sequence.layers[1].blocks[0]._norm[0]
    values, positions, timestamps, mask = inputs(readout=True)
    model(values, positions, timestamps, mask).sum().backward()
    assert all(group.norm.weight.grad is not None for group in model.shared_norms)


@pytest.mark.parametrize("schedule", ["layered", "all_as_sequence"])
def test_noop_registry_aliases_follow_runtime_contract(monkeypatch, schedule):

    for registry in (sequence.registry, interaction.registry, nonsequence.registry):
        monkeypatch.setitem(registry, "passthrough", registry["identity"])
    config = {
        "sequence_encoder": {"type": "passthrough", "pool": "mean" if schedule == "layered" else None},
        "interaction": {"type": "passthrough", "schedule": schedule},
        "nonsequence_encoder": {"type": "passthrough"},
    }
    if schedule == "layered":
        config["layer_order"] = ["sequence_encoder"]
    HistoryConfig.model_validate(config)
    model = HistoryAggregator({"max_events_per_sequence": 4, **config}, 2, 0, readout=False)
    expected = torch.tensor([[14 / 3, 20 / 3]]) if schedule == "layered" else inputs()[0]
    torch.testing.assert_close(model(*inputs()), expected)


@pytest.mark.parametrize(("component", "spec"), [
    ("sequence_encoder", {"type": "ffn"}),
    ("nonsequence_encoder", {"type": "ffn"}),
    ("interaction", {"type": "cross_attention", "params": {"num_heads": 1}}),
])
def test_layer_order_cannot_omit_active_components(component, spec):
    config = {
        "sequence_encoder": {"type": "identity", "pool": "mean"},
        "interaction": {"type": "identity", "schedule": "layered"},
        "layer_order": [name for name in ("sequence_encoder", "nonsequence_encoder", "interaction")
                        if name != component],
    }
    config[component] = {**config.get(component, {}), **spec}
    HistoryConfig.model_validate(config)
    with pytest.raises(ValueError, match="each non-noop component"):
        HistoryAggregator({"max_events_per_sequence": 4, **config}, 2, 1, readout=False)


@pytest.mark.parametrize(("component", "spec"), [
    ("nonsequence_encoder", {"type": "ffn"}),
    ("interaction", {"type": "cross_attention", "params": {"num_heads": 1}}),
    ("nonsequence_encoder", {"type": "identity", "params": {
        "queries": {"type": "persistent", "params": {"num_tokens": 1}},
    }}),
])
def test_all_as_sequence_cannot_ignore_active_components(component, spec):
    config = {"sequence_encoder": {"type": "identity"}, component: spec}
    HistoryConfig.model_validate(config)
    with pytest.raises(ValueError, match="does not use interaction or nonsequence"):
        HistoryAggregator({"max_events_per_sequence": 4, **config}, 2, 1, readout=False)


@pytest.mark.parametrize("name", ["identity", "ffn", "stack", "token_mixer"])
def test_nonsequence_registry_contains_concrete_classes(name):

    implementation = nonsequence_registry[name]
    assert isinstance(implementation, type)
    assert issubclass(implementation, Encoder)
    assert implementation is not Encoder


@pytest.mark.parametrize("depth", [1, 2])
def test_reversible_token_mixer_expands_width_before_ffn(depth):
    mixing = {"kernel": "permutation", "reversible": True}
    ffn = {"expansion_factor": 2, "activation": "gelu"}
    common = {"num_layers": 1, "independent_layers": False, "once": True, "readout": False}
    model = nonsequence_registry["token_mixer"](
        4, 3, ["context"], 3, params={"num_layers": depth, **mixing, "ffn": ffn}, **common,
    )
    reference = nonsequence_registry["stack"](
        4, 3, ["context"], 3,
        params={"num_layers": depth, "blocks": [
            {"type": "token_mixing", "params": mixing},
            {"type": "per_token_ffn", "params": ffn},
        ]}, **common,
    )
    reference.load_state_dict(model.state_dict(), strict=True)
    assert model.output_dim == reference.output_dim == 6
    tokens = torch.randn(2, 3, 4, requires_grad=True)
    reference_tokens = tokens.detach().clone().requires_grad_()
    padding = torch.zeros(2, 5, dtype=torch.bool)
    actual = model({"context": tokens}, padding)["context"]
    expected = reference({"context": reference_tokens}, padding)["context"]
    assert actual.shape == (2, 3, 6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(tokens.grad, reference_tokens.grad, rtol=0, atol=0)
    for parameter, reference_parameter in zip(model.parameters(), reference.parameters(), strict=True):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=0, atol=0)
