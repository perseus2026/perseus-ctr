from perseus.core.backbone.context import identity, tokenize
from perseus.core.backbone.context.base import Aggregator, Layout

__all__ = [
    "Aggregator",
    "Layout",
    "registry",
]

registry: dict[str, type[Aggregator]] = {
    "identity": identity.Aggregator,
    "tokenizer": tokenize.Aggregator,
}
