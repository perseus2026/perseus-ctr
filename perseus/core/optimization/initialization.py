import torch
from torch import nn

from perseus.core.config import Initialization, InitSpec

INIT_WEIGHTS_STD = 0.02


def initialize_module(module: nn.Module) -> None:
    if isinstance(module, nn.Embedding):
        nn.init.trunc_normal_(module.weight, std=INIT_WEIGHTS_STD)
        if module.padding_idx is not None:
            with torch.no_grad():
                module.weight[module.padding_idx].zero_()
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=INIT_WEIGHTS_STD)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.RMSNorm):
        if module.weight is not None:
            nn.init.ones_(module.weight)
    elif isinstance(module, nn.MultiheadAttention):
        for weight in (module.in_proj_weight, module.q_proj_weight, module.k_proj_weight, module.v_proj_weight):
            if weight is not None:
                nn.init.trunc_normal_(weight, std=INIT_WEIGHTS_STD)
        for bias in (module.in_proj_bias, module.bias_k, module.bias_v):
            if bias is not None:
                nn.init.zeros_(bias)


def _fill(tensor: torch.Tensor, spec: InitSpec) -> None:
    if spec.kind == "xavier_normal":
        nn.init.xavier_normal_(tensor)
    elif spec.kind == "xavier_uniform":
        nn.init.xavier_uniform_(tensor)
    elif spec.kind == "normal":
        nn.init.normal_(tensor, std=spec.std)
    elif spec.kind == "trunc_normal":
        nn.init.trunc_normal_(tensor, std=spec.std)
    else:
        raise ValueError(f"unknown initialization kind `{spec.kind}`")


def apply_initialization(spec: Initialization, *modules: nn.Module) -> None:
    touched: set[int] = set()
    for module in modules:
        for sub in module.modules():
            if isinstance(sub, nn.Embedding):
                if id(sub.weight) not in touched:
                    touched.add(id(sub.weight))
                    if spec.embedding is None:
                        nn.init.trunc_normal_(sub.weight, std=INIT_WEIGHTS_STD)
                    else:
                        _fill(sub.weight, spec.embedding)
                if sub.padding_idx is not None:
                    with torch.no_grad():
                        sub.weight[sub.padding_idx].zero_()
            elif isinstance(sub, nn.Linear):
                if id(sub.weight) not in touched:
                    touched.add(id(sub.weight))
                    if spec.linear is None:
                        nn.init.trunc_normal_(sub.weight, std=INIT_WEIGHTS_STD)
                    else:
                        _fill(sub.weight, spec.linear)
                if sub.bias is not None and spec.bias == "zeros" and id(sub.bias) not in touched:
                    touched.add(id(sub.bias))
                    nn.init.zeros_(sub.bias)
            else:
                initialize_module(sub)
