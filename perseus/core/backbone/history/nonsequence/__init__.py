"""Non-sequential encoders registered by their concrete implementation."""

from perseus.core.backbone.history.nonsequence.base import Encoder
from perseus.core.backbone.history.nonsequence.ffn import FeedForward
from perseus.core.backbone.history.nonsequence.identity import Identity
from perseus.core.backbone.history.nonsequence.stack_encoder import Stack
from perseus.core.backbone.history.nonsequence.token_mixer import TokenMixer

__all__ = ["Encoder", "registry"]

registry = {
    "identity": Identity,
    "ffn": FeedForward,
    "stack": Stack,
    "token_mixer": TokenMixer,
}
