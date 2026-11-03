"""Execution shortcuts preserve EST outputs and gradients on prepared model inputs."""

import copy
from types import MethodType
from unittest.mock import Mock

import pytest
import torch

from perseus.core import Backbone
from perseus.core.backbone.history.nonsequence.stack import GroupTransform, apply_group_block
from perseus.core.encoders.id import Embedder


def _legacy_group_forward(self, groups, padding_mask):
    keys = self.groups
    sizes = [groups[key].size(1) for key in keys]
    tokens = torch.cat([groups[key] for key in keys], dim=1)
    mask = padding_mask if keys == ["sequence"] else None
    zero_padding = mask is not None and self.zero_padding
    if zero_padding:
        tokens = tokens.masked_fill(mask.unsqueeze(-1), 0)
    for block in self.blocks:
        tokens = apply_group_block(block, tokens, mask, zero_padding=zero_padding)
    return {**groups, **dict(zip(keys, torch.split(tokens, sizes, dim=1), strict=True))}


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_est_shortcuts_match_previous_execution(empty, dtype):
    backbone = Backbone(
        16,
        embedding_dim=4,
        readout_token=False,
        event_aggregator_config={"type": "concat_linear", "num_features": 2, "params": {}},
        context_aggregator_config={"type": "tokenizer", "num_features": 3, "params": {}},
        history_aggregator_config={
            "max_events_per_sequence": 6,
            "num_layers": 2,
            "layer_order": ["interaction", "sequence_encoder", "nonsequence_encoder"],
            "sequence_encoder": {"type": "ffn", "params": {"norm_group": "est_ffn"}, "pool": "mean"},
            "nonsequence_encoder": {"type": "ffn", "params": {"norm_group": "est_ffn"}, "pool": "flatten"},
            "interaction": {
                "type": "cross_attention",
                "schedule": "layered",
                "independent_layers": True,
                "params": {"projection": "per_token", "num_heads": 2},
            },
        },
    )
    embedders = {f"f{i}": Embedder(7, 4, has_unk=True, prepared=True) for i in range(5)}
    reference, reference_embedders = copy.deepcopy((backbone, embedders))
    reference.history_aggregator.sequence.requires_attention_mask = True
    for module in reference.modules():
        if isinstance(module, GroupTransform):
            module.forward = MethodType(_legacy_group_forward, module)
    for embedder in reference_embedders.values():
        embedder.prepared = False
    backbone._build_mask = Mock(side_effect=AssertionError("FFN does not consume the full attention mask"))
    positions = torch.tensor([[0] * 6, [0, 0, 4, 3, 2, 1], [6, 5, 4, 3, 2, 1]])
    if empty:
        positions.zero_()
    events = {f"f{i}": torch.randint(0, 9, (3, 6), dtype=torch.int32).masked_fill(positions == 0, 0) for i in range(2)}
    context = {f"f{i}": torch.randint(0, 9, (3,), dtype=torch.int32) for i in range(2, 5)}
    data = (events, positions, positions.float(), torch.full((3,), 10.0), context)
    with torch.autocast("cpu", dtype=dtype, enabled=dtype == torch.bfloat16):
        expected = reference(reference_embedders, *data)
        actual = backbone(embedders, *data)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    expected.float().square().sum().backward()
    actual.float().square().sum().backward()
    for left, right in [(reference, backbone), *zip(reference_embedders.values(), embedders.values(), strict=True)]:
        for (name, a), (_, b) in zip(left.named_parameters(), right.named_parameters(), strict=True):
            assert (a.grad is None) == (b.grad is None), name
            if a.grad is not None:
                torch.testing.assert_close(b.grad, a.grad, rtol=0, atol=0, msg=name)


def test_safe_key_update_matches_all_small_masks():
    bits = torch.arange(256)[:, None]
    valid = ((bits >> torch.arange(8)) & 1).bool().reshape(256, 2, 4)
    expected, actual = valid.clone(), valid.clone()
    has_keys = valid.any(-1)
    expected.view(-1, 4)[~has_keys.flatten(), 0] = True
    actual[:, :, 0] |= ~has_keys
    assert torch.equal(actual, expected)
