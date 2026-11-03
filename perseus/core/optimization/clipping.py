import math
import typing as t

from torch import nn

from perseus.core.optimization.optimizers import parameters


def clip_gradients(parameters: t.Sequence[nn.Parameter], max_norm: float, norm_type: float = 2.0) -> float:
    seen: set[int] = set()
    with_grads = []
    for parameter in parameters:
        if parameter.grad is not None and id(parameter) not in seen:
            seen.add(id(parameter))
            with_grads.append(parameter)
    total_norm = nn.utils.get_total_norm([parameter.grad for parameter in with_grads], norm_type=norm_type)
    # Preserve finite optimizers when another optimizer overflowed before clipping.
    if math.isfinite(float(total_norm)):
        nn.utils.clip_grads_with_norm_(with_grads, max_norm, total_norm)
    return float(total_norm)


class GradientClipper:
    def __init__(self, parameters: t.Sequence[nn.Parameter]) -> None:
        self.parameters = parameters

    def clip(self, max_norm: float, norm_type: float = 2.0) -> float:
        return clip_gradients(self.parameters, max_norm, norm_type)


def create(modules: t.Sequence[nn.Module], scope: str) -> GradientClipper:
    group = "dense" if scope == "dense" else "all"
    return GradientClipper(parameters(modules, group))
