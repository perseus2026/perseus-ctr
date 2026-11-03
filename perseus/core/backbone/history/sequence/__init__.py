"""Sequence stream encoders registered as history aggregators."""

from perseus.core.backbone.history.sequence import (
    base,
    bert,
    danet,
    ffn,
    hstu,
    identity,
    ligr,
    mamba,
    modern_bert,
    self_attention,
)

registry: dict[str, type[base.Aggregator]] = {
    "identity": identity.Aggregator,
    "ffn": ffn.Aggregator,
    "self_attention": self_attention.Aggregator,
    "bert": bert.Aggregator,
    "modern_bert": modern_bert.Aggregator,
    "hstu": hstu.Aggregator,
    "mamba": mamba.Aggregator,
    "danet": danet.Aggregator,
    "ligr": ligr.Aggregator,
}
