"""Interactions between history, context and query tokens."""

from perseus.core.backbone.history.interaction import cross_attention, readout_attention
from perseus.core.backbone.history.interaction.base import Identity, Interaction

registry: dict[str, type[Interaction]] = {
    "identity": Identity,
    "cross_attention": cross_attention.CrossAttentionInteraction,
    "target_attention": readout_attention.TargetAttentionInteraction,
    "window_attention": readout_attention.WindowAttentionInteraction,
}
