import typing as t

from torch import nn

from perseus.core.backbone.history.nonsequence import attention, ffn, token_mixer
from perseus.core.backbone.history.nonsequence.base import Encoder
from perseus.core.backbone.history.nonsequence.stack import width_after
from perseus.core.backbone.history.sequence import ffn as sequence_ffn
from perseus.core.backbone.history.sequence import self_attention


class Stack(Encoder):
    supports_slices = True

    def build_blocks(self, dim: int, num_tokens: int, params: dict[str, t.Any]) -> list[nn.Module]:
        block_types = {
            "per_token_ffn": ffn.per_token_ffn,
            "token_mixing": token_mixer.token_mixing,
            "low_rank_attention": attention.low_rank_attention,
            "shared_ffn": sequence_ffn.shared_ffn,
            "self_attention": self_attention.self_attention,
        }
        blocks = []
        for spec in params["blocks"]:
            block_type, block_params = spec["type"], spec.get("params", {})
            block = block_types[block_type](dim, num_tokens, **block_params)
            dim = width_after(dim, block)
            blocks.append(block)
        return blocks
