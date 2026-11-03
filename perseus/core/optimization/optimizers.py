import typing as t

from accelerate import Accelerator
from torch import nn, optim

from perseus.core.config import Optimizers

registry: dict[str, type[optim.Optimizer]] = {
    "adadelta": optim.Adadelta,
    "adafactor": optim.Adafactor,
    "adagrad": optim.Adagrad,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "adamax": optim.Adamax,
    "asgd": optim.ASGD,
    "nadam": optim.NAdam,
    "radam": optim.RAdam,
    "rmsprop": optim.RMSprop,
    "rprop": optim.Rprop,
    "sgd": optim.SGD,
}


def create(
    accelerator: Accelerator,
    modules: t.Sequence[nn.Module],
    config: Optimizers,
) -> list[optim.Optimizer]:
    if config.embeddings is None:
        groups = [(config.default, parameters(modules))]
    else:
        groups = [
            (config.default, parameters(modules, "dense")),
            (config.embeddings, parameters(modules, "embeddings")),
        ]
    return [
        accelerator.prepare_optimizer(registry[spec.type](parameters, **spec.params))
        for spec, parameters in groups
        if parameters
    ]


def parameters(modules: t.Sequence[nn.Module], group: str = "all") -> list[nn.Parameter]:
    embeddings = {
        id(parameter)
        for module in modules
        for submodule in module.modules()
        if isinstance(submodule, nn.Embedding)
        for parameter in submodule.parameters(recurse=False)
        if parameter.requires_grad
    }
    seen: set[int] = set()
    parameters = []
    for module in modules:
        for parameter in module.parameters():
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            is_embedding = id(parameter) in embeddings
            if (group == "dense" and is_embedding) or (group == "embeddings" and not is_embedding):
                continue
            seen.add(id(parameter))
            parameters.append(parameter)
    return parameters


def step(accelerator: Accelerator, optimizers: t.Sequence[optim.Optimizer]) -> None:
    if accelerator.scaler is None or len(optimizers) == 1:
        for optimizer in optimizers:
            optimizer.step()
        return

    scaler = accelerator.scaler
    deferred = _DeferredScalerUpdate(scaler)
    try:
        for optimizer in optimizers:
            optimizer.scaler = deferred
        for optimizer in optimizers:
            optimizer.step()
    finally:
        for optimizer in optimizers:
            optimizer.scaler = scaler
    scaler.update()


class _DeferredScalerUpdate:
    def __init__(self, scaler: t.Any) -> None:
        self.scaler = scaler

    def step(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return self.scaler.step(*args, **kwargs)

    def update(self) -> None:
        pass
