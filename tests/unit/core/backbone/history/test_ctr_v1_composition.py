import re
from copy import deepcopy

import pytest
import torch
from pydantic import ValidationError
from torch import nn
from torch.nn import functional

from perseus.core.backbone.history import HistoryAggregator
from perseus.core.backbone.history.interaction.cross_attention import CrossAttention
from perseus.core.backbone.history.nonsequence.ffn import PerTokenSwiGLU, per_token_ffn
from perseus.core.backbone.history.nonsequence.stack import (
    GroupedResidual,
    PreNormResidual,
    Residual,
    bind_norm_groups,
)
from perseus.core.backbone.history.sequence.ffn import shared_ffn
from perseus.core.config import Config
from perseus.core.config import HistoryAggregator as HistoryConfig

DIM = 16
MAX_EVENTS = 12
NUM_CONTEXT = 2
NS_TOKENS = NUM_CONTEXT + 1

EST = {"num_layers": 3,
    "layer_order": ["interaction", "sequence_encoder", "nonsequence_encoder"],
    "sequence_encoder": {"type": "ffn", "params": {"expansion_factor": 4,
                                                   "activation": "relu",
                                                   "norm": "layer_norm", "norm_group": "est_ffn"},
                         "pool": "mean"},
    "nonsequence_encoder": {"type": "ffn", "params": {"expansion_factor": 4,
                                                      "activation": "relu",
                                                      "norm": "layer_norm", "norm_group": "est_ffn"},
                            "pool": "flatten"},
    "interaction": {"schedule": "layered", "type": "cross_attention",
                    "independent_layers": True,
                    "params": {"projection": "per_token", "num_heads": 4}},
}

HY = {"num_layers": 3,
    "layer_order": ["sequence_encoder", "interaction", "nonsequence_encoder"],
    "sequence_encoder": {"type": "self_attention",
                         "params": {"num_heads": 4,
                                    "ffn": {"expansion_factor": 2, "activation": "relu", "norm": "none"}}},
    "nonsequence_encoder": {"type": "token_mixer",
                            "params": {"queries": {"type": "persistent", "params": {"num_tokens": 2}},
                                       "kernel": "permutation",
                                       "ffn": {"expansion_factor": 2, "activation": "gelu"}}},
    "interaction": {"schedule": "layered", "type": "cross_attention",
                    "independent_layers": True,
                    "params": {"reader": "queries", "num_tokens": 2, "projection": "shared",
                               "num_heads": 4, "write": "replace"}, "pool": "mean"},
}

MIX = {"num_layers": 3,
    "layer_order": [{"nonsequence_encoder": [0, 1]}, "sequence_encoder", "interaction",
                    {"nonsequence_encoder": [2]}],
    "sequence_encoder": {"type": "ffn", "params": {"expansion_factor": 4,
                                                   "activation": "swiglu"}},
    "nonsequence_encoder": {"type": "stack", "params": {"blocks": [
        {"type": "token_mixing", "params": {"kernel": "permutation"}},
        {"type": "per_token_ffn", "params": {"expansion_factor": 4, "activation": "swiglu"}},
        {"type": "per_token_ffn", "params": {"expansion_factor": 4, "activation": "swiglu"}}]}, "pool": "mean"},
    "interaction": {"schedule": "layered", "type": "cross_attention",
                    "independent_layers": True,
                    "params": {"projection": "per_query_kv"}},
}

SPECS = {"est": EST, "hy": HY, "mix": MIX}


def _single_layer(spec):
    out = {**spec, "num_layers": 1, "layer_order": list(spec["layer_order"])}
    out["sequence_encoder"] = {**spec["sequence_encoder"],
                               "params": dict(spec["sequence_encoder"]["params"])}
    out["nonsequence_encoder"] = {**spec["nonsequence_encoder"],
                                  "params": dict(spec["nonsequence_encoder"]["params"])}
    return out


def _strip_norm_keys(spec):
    out = {key: value for key, value in spec.items() if key != "layer_order"}
    for component in ("sequence_encoder", "nonsequence_encoder"):
        params = {k: v for k, v in out[component]["params"].items() if k not in {"norm", "norm_group"}}
        if isinstance(params.get("ffn"), dict):
            params["ffn"] = {k: v for k, v in params["ffn"].items() if k not in {"norm", "norm_group"}}
        out[component] = {**out[component], "params": params}
    return out


def history(spec, num_context_tokens=NUM_CONTEXT):
    return HistoryAggregator({"max_events_per_sequence": MAX_EVENTS, **spec}, DIM, num_context_tokens)


def inputs(batch=2):
    generator = torch.Generator().manual_seed(7)
    length = MAX_EVENTS + NUM_CONTEXT + 1
    embeddings = torch.randn(batch, length, DIM, generator=generator)
    positions = torch.arange(length).unsqueeze(0).repeat(batch, 1)
    timestamps = positions.float()
    mask = torch.zeros(batch, length, length, dtype=torch.bool)
    return embeddings, positions, timestamps, mask


def group_holders(module):
    return list(module.shared_norms)


def grouped_consumers(module):
    return [m for m in module.modules() if isinstance(m, GroupedResidual)]


def layer_norms(module):
    return [m for m in module.modules() if isinstance(m, nn.LayerNorm)]


@pytest.mark.parametrize("name", sorted(SPECS))
def test_reference_graphs_build_forward_and_backward(name):
    module = history(SPECS[name]).eval()
    embeddings, positions, timestamps, mask = inputs()
    output = module(embeddings, positions, timestamps, mask)
    assert torch.isfinite(output).all()
    output.pow(2).sum().backward()
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


@pytest.mark.parametrize("name", sorted(SPECS))
def test_reference_graphs_handle_padding_and_cold(name):
    module = history(SPECS[name]).eval()
    embeddings, positions, timestamps, mask = inputs()
    positions = positions.clone()
    positions[:, :5] = 0
    warm = module(embeddings, positions, timestamps, mask)
    assert torch.isfinite(warm).all()
    cold = module(embeddings, positions * 0, timestamps * 0, mask)
    assert torch.isfinite(cold).all()


def test_execution_order_matches_layer_order():
    module = history(MIX).eval()
    executed = [block for layer in range(3) for block in (
        module.nonsequence.layers[layer][0], module.nonsequence.layers[layer][1],
        module.sequence.layers[layer], module.interaction.layers[layer], module.nonsequence.layers[layer][2],
    )]
    order = []
    handles = [
        block.register_forward_hook(lambda _m, _i, _o, block=block: order.append(block))
        for block in executed
    ]
    module(*inputs())
    for handle in handles:
        handle.remove()
    per_layer = ["GroupTransform", "GroupTransform", "GroupTransform", "CrossAttention", "GroupTransform"]
    assert [type(block).__name__ for block in executed] == per_layer * 3
    assert order == executed


def test_slice_uses_two_distinct_ffn_instances():
    module = history(MIX).eval()
    swiglus = [m for m in module.modules() if isinstance(m, PerTokenSwiGLU)]
    assert len(swiglus) == 6
    first_ffn, second_ffn = swiglus[0], swiglus[1]
    assert all(p0.data_ptr() != p1.data_ptr() for p0, p1 in zip(first_ffn.parameters(),
                                                                second_ffn.parameters(), strict=False))
    tensors = list(module.parameters())
    assert len({p.data_ptr() for p in tensors}) == len(tensors)


def test_wrong_order_changes_the_output():
    torch.manual_seed(11)
    correct = history(EST).eval()
    torch.manual_seed(11)
    wrong = history({**EST, "layer_order": ["sequence_encoder", "interaction", "nonsequence_encoder"]}).eval()
    embeddings, positions, timestamps, mask = inputs()
    assert not torch.allclose(correct(embeddings, positions, timestamps, mask),
                              wrong(embeddings, positions, timestamps, mask))
    torch.manual_seed(12)
    mix = history(MIX).eval()
    torch.manual_seed(12)
    swapped = history({**MIX, "layer_order": [{"nonsequence_encoder": [2]}, "sequence_encoder",
                                              "interaction", {"nonsequence_encoder": [0, 1]}]}).eval()
    assert not torch.allclose(mix(embeddings, positions, timestamps, mask),
                              swapped(embeddings, positions, timestamps, mask))


def test_registry_encoder_step_follows_layer_order():
    spec = {"num_layers": 2,
        "layer_order": ["interaction", "sequence_encoder", "nonsequence_encoder"],
        "sequence_encoder": {"type": "modern_bert", "params": {}, "finalize_in_stream": True},
        "nonsequence_encoder": {"type": "ffn", "params": {"expansion_factor": 2},
                                "pool": "flatten"},
        "interaction": {"schedule": "layered", "type": "cross_attention",
                        "params": {"projection": "per_token", "num_heads": 4}},
    }
    module = history(spec).eval()
    order = []
    handles = [block.register_forward_hook(lambda *_, name=name: order.append(name)) for name, block in (
        ("interaction", module.interaction.layers[0]),
        ("sequence", module.sequence.layers[0]), ("sequence", module.sequence.layers[1]),
        ("finalize", module.sequence.norm),
        ("nonsequence", module.nonsequence.layers[0]),
    )]
    try:
        module(*inputs())
    finally:
        for handle in handles:
            handle.remove()
    assert order == ["interaction", "sequence", "nonsequence", "interaction", "sequence", "finalize", "nonsequence"]


def test_est_ffn_norm_group_single_owner_per_layer():
    module = history(EST).eval()
    holders = group_holders(module)
    consumers = grouped_consumers(module)
    assert len(holders) == 3
    assert len(consumers) == 6
    for index in range(3):
        assert module.sequence.layers[index].blocks[0]._norm[0] is module.nonsequence.layers[index].blocks[0]._norm[0]
    owner_norms = {id(holder.norm) for holder in holders}
    assert all(id(consumer._norm[0]) in owner_norms for consumer in consumers)
    assert len({holder.norm.weight.data_ptr() for holder in holders}) == 3
    attention_norms = [m.attention_norm for m in module.modules() if isinstance(m, CrossAttention)]
    assert len(attention_norms) == 3
    assert {id(norm) for norm in attention_norms}.isdisjoint(owner_norms)
    assert len(layer_norms(module)) == 6


def test_norm_group_without_layer_order_and_shared_layers():
    spec = _strip_norm_keys(EST)
    spec["interaction"] = {k: v for k, v in spec["interaction"].items() if k != "independent_layers"}
    spec["sequence_encoder"]["params"]["norm_group"] = "est_ffn"
    spec["nonsequence_encoder"]["params"]["norm_group"] = "est_ffn"
    module = history(spec).eval()
    assert len(group_holders(module)) == 1
    consumers = grouped_consumers(module)
    assert len({id(consumer._norm[0]) for consumer in consumers}) == 1


def _one_layer_est(with_group):
    spec = _single_layer(EST)
    if not with_group:
        for component in ("sequence_encoder", "nonsequence_encoder"):
            params = {k: v for k, v in spec[component]["params"].items() if k not in {"norm", "norm_group"}}
            spec[component] = {**spec[component], "params": params}
    return spec


def test_est_shared_norm_gradient_sums_both_branches():
    torch.manual_seed(3)
    grouped = history(_one_layer_est(True)).eval()
    torch.manual_seed(3)
    split = history(_one_layer_est(False)).eval()
    embeddings, positions, timestamps, mask = inputs()
    grouped(embeddings, positions, timestamps, mask).pow(2).sum().backward()
    split(embeddings, positions, timestamps, mask).pow(2).sum().backward()
    shared = group_holders(grouped)[0].norm.weight.grad
    seq_norm = split.sequence.layers[0].blocks[0].norm
    ns_norm = split.nonsequence.layers[0].blocks[0].norm
    torch.testing.assert_close(shared, seq_norm.weight.grad + ns_norm.weight.grad)
    for name, param in split.named_parameters():
        if param.grad is None or param.grad.shape != shared.shape:
            continue
        assert not torch.allclose(param.grad, shared), name


def test_est_shared_norm_alias_survives_step_and_reload():
    torch.manual_seed(5)
    module = history(_one_layer_est(True)).eval()
    embeddings, positions, timestamps, mask = inputs()
    module(embeddings, positions, timestamps, mask).pow(2).sum().backward()
    holder = group_holders(module)[0]
    assert holder.norm.weight.grad is not None
    assert holder.norm.weight.grad.abs().sum() > 0
    optimizer = torch.optim.SGD(module.parameters(), lr=0.05)
    optimizer.step()
    assert (holder.norm.weight - 1.0).abs().sum() > 0
    trained = module(embeddings, positions, timestamps, mask)
    with torch.no_grad():
        holder.norm.weight += 1.0
    perturbed = module(embeddings, positions, timestamps, mask)
    assert not torch.allclose(perturbed, trained)
    state = {key: value.clone() for key, value in module.state_dict().items()}
    fresh = history(_one_layer_est(True)).eval()
    fresh.load_state_dict(state)
    torch.testing.assert_close(fresh(embeddings, positions, timestamps, mask), perturbed)
    fresh_holder = group_holders(fresh)[0]
    assert all(consumer._norm[0] is fresh_holder.norm for consumer in grouped_consumers(fresh))
    norm_weight_keys = [key for key in fresh.state_dict() if key.endswith("norm.weight")]
    assert sorted(norm_weight_keys) == ["interaction.layers.0.attention_norm.weight", "shared_norms.0.norm.weight"]


def test_est_layer_matches_unirank_math():
    torch.manual_seed(17)
    module = history(_one_layer_est(True)).eval()
    inter, seq_gt, ns_gt = module.interaction.layers[0], module.sequence.layers[0], module.nonsequence.layers[0]
    ffn_norm = group_holders(module)[0].norm
    generator = torch.Generator().manual_seed(1)
    s = torch.randn(2, 7, DIM, generator=generator)
    ns = torch.randn(2, NS_TOKENS, DIM, generator=generator)
    padding = torch.zeros(2, 7, dtype=torch.bool)

    ns1 = ns + inter(ns, s, padding)
    s1 = seq_gt({"sequence": s}, padding)["sequence"]
    ns2 = ns_gt({"context": ns1}, padding)["context"]

    num_heads = EST["interaction"]["params"]["num_heads"]
    att_norm = inter.attention_norm
    head_dim = DIM // num_heads
    k = inter.key(att_norm(s)).view(2, 7, num_heads, head_dim).transpose(1, 2)
    v = inter.value(att_norm(s)).view(2, 7, num_heads, head_dim).transpose(1, 2)
    q = torch.einsum("bld,ldh->blh", att_norm(ns), inter.query_weight)
    q = q.view(2, NS_TOKENS, num_heads, head_dim).transpose(1, 2)
    attended = functional.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(2, NS_TOKENS, DIM)
    ns1_ref = ns + torch.einsum("bld,ldh->blh", attended, inter.output)

    seq_op = seq_gt.blocks[0].op
    ns_op = ns_gt.blocks[0].op
    s_out = seq_op.net[3](functional.relu(seq_op.net[0](ffn_norm(s))))
    hidden = functional.relu(torch.einsum("bld,ldh->blh", ffn_norm(ns1_ref), ns_op.first_weight) + ns_op.first_bias)
    ns_out = torch.einsum("blh,lhd->bld", hidden, ns_op.second_weight) + ns_op.second_bias
    s1_ref = s + s_out
    ns2_ref = ns1_ref + ns_out

    torch.testing.assert_close(ns1, ns1_ref)
    torch.testing.assert_close(s1, s1_ref)
    torch.testing.assert_close(ns2, ns2_ref)


class UTokenMixing(nn.Module):
    def __init__(self, input_dim, num_token):
        super().__init__()
        self.num_token = num_token
        self.input_dim = input_dim
        self.pad_dim = (-input_dim) % num_token
        self.padded_input_dim = input_dim + self.pad_dim
        self.head_dim = self.padded_input_dim // num_token
        valid = torch.arange(self.padded_input_dim) < input_dim
        valid = valid.expand(num_token, -1)
        valid = valid.reshape(num_token, num_token, self.head_dim).transpose(0, 1).reshape(-1)
        self.register_buffer("valid_index", valid.nonzero(as_tuple=False).flatten(), persistent=False)

    def forward(self, x):
        if self.pad_dim:
            x = functional.pad(x, (0, self.pad_dim))
        heads = torch.tensor_split(x, self.num_token, dim=-1)
        mixed = torch.stack(heads, dim=1).flatten(start_dim=2)
        mixed = mixed.flatten(start_dim=1).index_select(1, self.valid_index)
        return mixed.reshape(x.size(0), self.num_token, self.input_dim)


def u_swiglu(x, w1, w2, w3):
    return w3(functional.gelu(w1(x)) * w2(x))


def u_per_token_swiglu(x, module):
    h1 = functional.gelu(torch.einsum("btd,tdk->btk", x, module.w1) + module.b1)
    h2 = torch.einsum("btd,tdk->btk", x, module.w2) + module.b2
    return torch.einsum("btk,tkd->btd", h1 * h2, module.w3)


def test_mix_layer_stages_match_unirank_math():
    torch.manual_seed(23)
    module = history(_single_layer(MIX)).eval()
    ns_mixer_gt, ns_ffn_gt, ns_out_gt = module.nonsequence.layers[0]
    seq_gt, inter = module.sequence.layers[0], module.interaction.layers[0]
    mixer_res = ns_mixer_gt.blocks[0]
    mixer_ffn_res = ns_ffn_gt.blocks[0]
    seq_ffn_res = seq_gt.blocks[0]
    out_ffn_res = ns_out_gt.blocks[0]
    generator = torch.Generator().manual_seed(2)
    s = torch.randn(2, 7, DIM, generator=generator)
    ns = torch.randn(2, NS_TOKENS, DIM, generator=generator)
    padding = torch.zeros(2, 7, dtype=torch.bool)

    ns1 = ns_mixer_gt({"sequence": s, "context": ns}, padding)["context"]
    ns2 = ns_ffn_gt({"sequence": s, "context": ns1}, padding)["context"]
    s1 = seq_gt({"sequence": s}, padding)["sequence"]
    ns3 = ns2 + inter(ns2, s1, padding)
    ns4 = ns_out_gt({"sequence": s, "context": ns3}, padding)["context"]

    reference_mixer = UTokenMixing(DIM, NS_TOKENS)
    torch.testing.assert_close(mixer_res.op(ns), reference_mixer(ns))
    ns1_ref = reference_mixer(mixer_res.norm(ns)) + ns
    ns2_ref = u_per_token_swiglu(mixer_ffn_res.norm(ns1_ref), mixer_ffn_res.op) + ns1_ref
    s1_ref = u_swiglu(seq_ffn_res.norm(s), seq_ffn_res.op.net.w1, seq_ffn_res.op.net.w2,
                      seq_ffn_res.op.net.w3) + s
    keys = inter.key(s1_ref).chunk(NS_TOKENS, dim=-1)
    values = inter.value(s1_ref).chunk(NS_TOKENS, dim=-1)
    attended = functional.scaled_dot_product_attention(
        ns2_ref.unsqueeze(2), torch.stack(keys, dim=1), torch.stack(values, dim=1), scale=DIM ** -0.5,
    )
    ns3_ref = attended.squeeze(2) + ns2_ref
    ns4_ref = u_per_token_swiglu(out_ffn_res.norm(ns3_ref), out_ffn_res.op) + ns3_ref

    torch.testing.assert_close(ns1, ns1_ref)
    torch.testing.assert_close(ns2, ns2_ref)
    torch.testing.assert_close(s1, s1_ref)
    torch.testing.assert_close(ns3, ns3_ref)
    torch.testing.assert_close(ns4, ns4_ref)


def test_hy_sequence_ffn_without_norm_matches_unirank_math():
    torch.manual_seed(19)
    module = history(_single_layer(HY)).eval()
    attention, ffn_res = module.sequence.layers[0].blocks
    assert type(ffn_res) is Residual
    generator = torch.Generator().manual_seed(3)
    x = torch.randn(2, 7, DIM, generator=generator)

    norm_x = attention.norm(x)
    num_heads = HY["sequence_encoder"]["params"]["num_heads"]
    head_dim = DIM // num_heads
    attended = functional.scaled_dot_product_attention(
        attention.op.query(norm_x).view(2, 7, num_heads, head_dim).transpose(1, 2),
        attention.op.key(norm_x).view(2, 7, num_heads, head_dim).transpose(1, 2),
        attention.op.value(norm_x).view(2, 7, num_heads, head_dim).transpose(1, 2),
    ).transpose(1, 2).reshape(2, 7, DIM)
    after_attention = attention.op.output(attended) + x

    ours = ffn_res(after_attention, None)
    reference = ffn_res.op.net[3](functional.relu(ffn_res.op.net[0](after_attention))) + after_attention
    torch.testing.assert_close(ours, reference)

    legacy_ffn = shared_ffn(DIM, expansion_factor=2, activation="relu")
    with torch.no_grad():
        legacy_ffn.op.net[0].weight.copy_(ffn_res.op.net[0].weight)
        legacy_ffn.op.net[0].bias.copy_(ffn_res.op.net[0].bias)
        legacy_ffn.op.net[3].weight.copy_(ffn_res.op.net[3].weight)
        legacy_ffn.op.net[3].bias.copy_(ffn_res.op.net[3].bias)
        legacy_ffn.norm.weight.copy_(attention.norm.weight)
        legacy_ffn.norm.bias.copy_(attention.norm.bias)
    legacy = legacy_ffn(after_attention, None)
    assert not torch.allclose(legacy, reference)


def test_hy_default_keeps_ffn_norm():
    torch.manual_seed(31)
    legacy = history(_strip_norm_keys(HY)).eval()
    torch.manual_seed(31)
    without_norm = history(_strip_norm_keys(HY) | {
        "sequence_encoder": {"type": "self_attention",
                             "params": {"num_heads": 4,
                                        "ffn": {"expansion_factor": 2, "activation": "relu", "norm": "none"}}}}).eval()
    legacy_count = sum(p.numel() for p in legacy.parameters())
    none_count = sum(p.numel() for p in without_norm.parameters())
    assert legacy_count - none_count == 3 * 2 * DIM
    assert type(legacy.sequence.layers[0].blocks[1]) is PreNormResidual
    assert type(without_norm.sequence.layers[0].blocks[1]) is Residual


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("optimizer_cls", [torch.optim.AdamW, torch.optim.Adagrad])
def test_shared_cross_attention_residual_and_gradients_match_reference(bf16, optimizer_cls):
    torch.manual_seed(29)
    attention = CrossAttention(DIM, NS_TOKENS, 2, "shared").eval()
    reference = deepcopy(attention)
    optimizer = optimizer_cls(attention.parameters(), lr=1e-3)
    reference_optimizer = optimizer_cls(reference.parameters(), lr=1e-3)
    generator = torch.Generator().manual_seed(4)
    reader = torch.randn(2, NS_TOKENS, DIM, generator=generator)
    kv = torch.randn(2, 7, DIM, generator=generator)
    padding = torch.zeros(2, 7, dtype=torch.bool)
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
        out = attention(reader, kv, padding)
        norm_reader = reference.norm(reader)
        norm_kv = reference.norm(kv)
        num_heads, head_dim = reference.num_heads, reference.head_dim
        q = reference.query(norm_reader).view(2, NS_TOKENS, num_heads, head_dim).transpose(1, 2)
        k = reference.key(norm_kv).view(2, 7, num_heads, head_dim).transpose(1, 2)
        v = reference.value(norm_kv).view(2, 7, num_heads, head_dim).transpose(1, 2)
        attended = functional.scaled_dot_product_attention(q, k, v, attn_mask=(~padding)[:, None, None, :])
        delta = reference.output(attended.transpose(1, 2).reshape(2, NS_TOKENS, DIM))
        expected = reader + delta
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    assert not torch.allclose(out, reader + delta + delta)
    out.square().mean().backward()
    expected.square().mean().backward()
    for actual, wanted in zip(attention.parameters(), reference.parameters(), strict=True):
        assert actual.grad is not None
        assert wanted.grad is not None
        torch.testing.assert_close(actual.grad, wanted.grad, rtol=0, atol=0)
    optimizer.step()
    reference_optimizer.step()
    torch.testing.assert_close(attention.state_dict(), reference.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(optimizer.state_dict(), reference_optimizer.state_dict(), rtol=0, atol=0)
    with pytest.raises(ValueError, match=r"returns.*residual"):
        history({"num_layers": 1,
            "sequence_encoder": {"type": "ffn", "params": {"expansion_factor": 2}},
            "nonsequence_encoder": {"type": "ffn", "params": {"expansion_factor": 2}},
            "interaction": {"schedule": "layered", "type": "cross_attention",
                            "params": {"projection": "shared", "num_heads": 2}},
        })


def test_norm_and_group_validation():
    bad_group = _single_layer(EST)
    bad_group["nonsequence_encoder"]["params"]["norm"] = "none"
    with pytest.raises(ValueError, match="cannot be combined"):
        history(bad_group)
    with pytest.raises(ValueError, match="unknown history norm `elu`"):
        per_token_ffn(DIM, NS_TOKENS, norm="elu")
    with pytest.raises(ValueError, match="must be a non-empty string"):
        per_token_ffn(DIM, NS_TOKENS, norm_group="")
    wrapper = per_token_ffn(DIM, NS_TOKENS, norm_group="g")
    assert isinstance(wrapper, GroupedResidual)
    with pytest.raises(RuntimeError, match="has no owner"):
        wrapper(torch.randn(2, NS_TOKENS, DIM), None)
    other = shared_ffn(DIM + 1, norm_group="g")
    with pytest.raises(ValueError, match="different widths"):
        bind_norm_groups([nn.ModuleList([wrapper]), nn.ModuleList([other])])


@pytest.mark.parametrize(("spec", "expected"), [
    ({**EST, "interaction": {"schedule": "once", "type": "cross_attention",
                             "params": {"projection": "per_token", "num_heads": 4}}},
     "history num_layers requires layered schedule"),
    ({**EST, "layer_order": ["interaction", "sequence_decoder", "nonsequence_encoder"]},
     "layer_order.1.literal"),
    ({**EST, "layer_order": ["interaction", "interaction", "sequence_encoder", "nonsequence_encoder"]},
     "repeats a component"),
    ({**EST, "layer_order": ["sequence_encoder", "nonsequence_encoder"]},
     "layer_order must include each non-noop component"),
    ({**EST, "layer_order": ["interaction", "sequence_encoder", "nonsequence_encoder", "interaction"]},
     "repeats a component"),
    ({**MIX, "layer_order": [{"nonsequence_encoder": [0, 1]}, "sequence_encoder", "interaction",
                             {"nonsequence_encoder": [0, 2]}]},
     "nonsequence slices must cover every block exactly once"),
    ({**MIX, "layer_order": [{"nonsequence_encoder": [0]}, "sequence_encoder", "interaction",
                             {"nonsequence_encoder": [1]}]}, "nonsequence slices must cover every block exactly once"),
    ({**MIX, "layer_order": [{"nonsequence_encoder": [0, 1]}, "sequence_encoder", "interaction",
                             {"nonsequence_encoder": [3]}]}, "nonsequence slices must cover every block exactly once"),
    ({**MIX, "layer_order": [{"nonsequence_encoder": []}, "sequence_encoder", "interaction",
                             {"nonsequence_encoder": [0, 1, 2]}]},
     "nonsequence slices must cover every block exactly once"),
    ({**MIX, "layer_order": [{"nonsequence_encoder": [0, 1]}, "sequence_encoder", "interaction",
                             {"nonsequence_encoder": ["2"]}]}, "layer_order.3.literal"),
    ({**MIX, "layer_order": [{"sequence_encoder": [0]}, "interaction",
                             {"nonsequence_encoder": [0, 1, 2]}]}, "layer_order.0.literal"),
    ({**MIX, "layer_order": [{"nonsequence_encoder": [0, 1]}, "sequence_encoder", "interaction",
                             "nonsequence_encoder", {"nonsequence_encoder": [2]}]},
     "nonsequence slices cannot be mixed with a whole step"),
    ({**EST, "layer_order": [{"nonsequence_encoder": [0]}, "sequence_encoder", "interaction"]},
     "nonsequence slices require a stack"),
    ({**EST, "layer_order": []}, "layer_order must not be empty"),
    ({**EST, "layer_order": "interaction"}, "Input should be a valid list"),
])
def test_layer_order_validation(spec, expected):
    def validate_and_build():
        HistoryConfig.model_validate(spec)
        history(spec)

    with pytest.raises(ValueError, match=re.escape(expected)):
        validate_and_build()


def test_identity_component_step_is_allowed():
    spec = {"num_layers": 2,
        "layer_order": ["interaction", "sequence_encoder", "nonsequence_encoder"],
        "sequence_encoder": {"type": "ffn", "params": {"expansion_factor": 2}, "pool": "mean"},
        "interaction": {"schedule": "layered", "type": "identity"},
    }
    HistoryConfig.model_validate(spec)
    module = history(spec).eval()
    assert torch.isfinite(module(*inputs())).all()


def test_legacy_est_plan_is_unchanged():
    spec = _strip_norm_keys(EST)
    spec["interaction"] = {**spec["interaction"], "independent_layers": True}
    module = history(spec).eval()
    assert len(module.sequence.layers) == 3
    assert len(module.interaction.layers) == 3
    assert len(module.nonsequence.layers) == 3
    assert len(layer_norms(module)) == 9
    embeddings, positions, timestamps, mask = inputs()
    assert torch.isfinite(module(embeddings, positions, timestamps, mask)).all()


REVERSE = {"num_layers": 2,
    "sequence_encoder": {"type": "ffn", "params": {"expansion_factor": 2}, "pool": "mean"},
    "nonsequence_encoder": {"type": "ffn", "params": {"expansion_factor": 2,
                            "queries": {"type": "persistent", "params": {"num_tokens": 2}}},
                            "pool": "flatten"},
    "interaction": {"schedule": "layered", "type": "cross_attention", "independent_layers": True,
                    "params": {"reader": "queries", "num_tokens": 2, "num_heads": 2, "write": "replace",
                               "reverse_context": True}, "pool": "mean"},
}


def test_explicit_order_registers_reverse_context_modules():
    torch.manual_seed(41)
    explicit = history({**REVERSE, "layer_order": ["sequence_encoder", "interaction",
                                                   "nonsequence_encoder"]}).eval()
    torch.manual_seed(41)
    default = history(dict(REVERSE)).eval()
    assert len(explicit.state_dict()) == len(default.state_dict())
    explicit_shapes = sorted(str(tuple(value.shape)) for value in explicit.state_dict().values())
    default_shapes = sorted(str(tuple(value.shape)) for value in default.state_dict().values())
    assert explicit_shapes == default_shapes
    attention_modules = [m for m in explicit.modules() if isinstance(m, CrossAttention)]
    assert len(attention_modules) == 4
    state_ptrs = {value.data_ptr() for value in explicit.state_dict().values()}
    param_ptrs = {p.data_ptr() for p in explicit.parameters()}
    assert state_ptrs == param_ptrs
    embeddings, positions, timestamps, mask = inputs()
    explicit.double()
    embeddings64 = embeddings.double()
    assert torch.isfinite(explicit(embeddings64, positions, timestamps, mask)).all()
    optimizer = torch.optim.SGD(explicit.parameters(), lr=0.01)
    explicit(embeddings64, positions, timestamps, mask).pow(2).sum().backward()
    for attention in attention_modules:
        grads = [p.grad for p in attention.parameters()]
        assert grads
        assert any(grad is not None and grad.abs().sum() > 0 for grad in grads)
    optimizer.step()


def test_norm_group_requires_layered_schedule():
    spec = {
        "sequence_encoder": {"type": "ffn", "params": {"num_layers": 3, "expansion_factor": 2,
                                                       "norm_group": "g"}, "pool": "mean"},
        "interaction": {"schedule": "once", "type": "target_attention",
                        "params": {}, "pool": "mean"},
    }
    with pytest.raises(ValueError, match="requires layered schedule"):
        history(spec)


def test_config_schema_accepts_layer_order():
    payload = {
        "task": {"type": "classification", "metrics": {}},
        "events": {"purchase": None},
        "max_events_per_sequence": 8,
        "backbone": {"dim": DIM, "history_aggregator": EST,
                     "context_aggregator": {"type": "identity", "params": {}}},
        "features": {"city": {"located_in": {"context": True}, "encoder": {"type": "id"}}},
        "training": {"num_epochs": 1},
    }
    config = Config.model_validate(payload)
    assert config.backbone.history_aggregator.layer_order == EST["layer_order"]
    dumped = config.model_dump()
    assert dumped["backbone"]["history_aggregator"]["layer_order"] == EST["layer_order"]
    with pytest.raises(ValidationError):
        Config.model_validate(payload | {"backbone": {"dim": DIM,
                                                      "history_aggregator": {**EST, "layer_order": "interaction"}}})


def test_est_parameter_delta_matches_removed_norms():
    torch.manual_seed(37)
    grouped = history(_one_layer_est(True))
    torch.manual_seed(37)
    split = history(_one_layer_est(False))
    assert sum(p.numel() for p in split.parameters()) - sum(p.numel() for p in grouped.parameters()) == 2 * DIM
