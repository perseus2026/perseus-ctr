

from perseus.core.backbone.history.nonsequence.base import Encoder


class Identity(Encoder):
    step_is_noop = True
