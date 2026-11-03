import typing as t

import torch
from torch import nn


class PreNormResidual(nn.Module):
    """Pre-norm residual sublayer around a padding-agnostic operation."""

    def __init__(self, dim: int, op: nn.Module, /) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.op = op

    def forward(self, tokens: torch.Tensor, _padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return tokens + self.op(self.norm(tokens))


class PaddedPreNormResidual(nn.Module):
    """Pre-norm residual sublayer around a padding-aware operation."""

    def __init__(self, dim: int, op: nn.Module, /) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.op = op

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        return tokens + self.op(self.norm(tokens), padding_mask)


class Residual(nn.Module):
    def __init__(self, op: nn.Module, /) -> None:
        super().__init__()
        self.op = op

    def forward(self, tokens: torch.Tensor, _padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return tokens + self.op(tokens)


class GroupedResidual(nn.Module):
    def __init__(self, dim: int, op: nn.Module, group: str, /) -> None:
        super().__init__()
        self.dim = dim
        self.group = group
        self.op = op
        self._norm: tuple[nn.Module, ...] = ()

    @property
    def is_bound(self) -> bool:
        return bool(self._norm)

    def bind(self, norm: nn.Module, /) -> None:
        if self._norm:
            if self._norm[0] is not norm:
                raise ValueError(f"norm group `{self.group}` is already bound to another norm")
            return
        if tuple(norm.normalized_shape) != (self.dim,):
            raise ValueError(
                f"norm group `{self.group}` normalizes {tuple(norm.normalized_shape)}, "
                f"consumer needs {(self.dim,)}",
            )
        self._norm = (norm,)

    def forward(self, tokens: torch.Tensor, _padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if not self._norm:
            raise RuntimeError(f"norm group `{self.group}` has no owner in this layer")
        return tokens + self.op(self._norm[0](tokens))


class NormGroup(nn.Module):
    def __init__(self, name: str, norm: nn.Module, /) -> None:
        super().__init__()
        self.name = name
        self.norm = norm


def bind_norm_groups(modules: list[nn.Module], /) -> list[nn.Module]:
    groups: dict[str, list[GroupedResidual]] = {}
    for holder in modules:
        for sub in holder.modules():
            if isinstance(sub, GroupedResidual):
                groups.setdefault(sub.group, []).append(sub)
    holders: list[nn.Module] = []
    for name, consumers in groups.items():
        width = consumers[0].dim
        if any(consumer.dim != width for consumer in consumers):
            raise ValueError(f"history norm group `{name}` consumers have different widths")
        norm = nn.LayerNorm(width)
        for consumer in consumers:
            consumer.bind(norm)
        holders.append(NormGroup(name, norm))
    return holders


def check_norm_groups(modules: list[nn.Module], /) -> None:
    for module in modules:
        for child in module.modules():
            if isinstance(child, GroupedResidual) and not child.is_bound:
                raise ValueError(f"norm group `{child.group}` has no owner; norm_group requires layered schedule")


class GroupTransform(nn.Module):
    def __init__(self, groups: list[str], blocks_: nn.ModuleList, /, *, zero_padding: bool = False) -> None:
        super().__init__()
        self.groups = groups
        self.blocks = blocks_
        self.zero_padding = zero_padding

    def forward(self, groups: dict[str, torch.Tensor], padding_mask: torch.Tensor, /) -> dict[str, torch.Tensor]:
        keys = self.groups
        sizes = [groups[key].size(1) for key in keys]
        tokens = groups[keys[0]] if len(keys) == 1 else torch.cat([groups[key] for key in keys], dim=1)
        mask = padding_mask if keys == ["sequence"] else None
        zero_padding = mask is not None and self.zero_padding
        if zero_padding:
            tokens = tokens.masked_fill(mask.unsqueeze(-1), 0)
        for block in self.blocks:
            tokens = apply_group_block(block, tokens, mask, zero_padding=zero_padding)
        if len(keys) == 1:
            if tokens.size(1) != sizes[0]:
                raise ValueError("group transform must preserve token count")
            return {**groups, keys[0]: tokens}
        return {**groups, **dict(zip(keys, torch.split(tokens, sizes, dim=1), strict=True))}


def apply_group_block(
    block: nn.Module,
    tokens: torch.Tensor,
    padding_mask: torch.Tensor | None,
    /,
    *,
    zero_padding: bool = False,
) -> torch.Tensor:
    if zero_padding:
        tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0)
    tokens = block(tokens, padding_mask)
    if zero_padding:
        tokens = tokens.masked_fill(padding_mask.unsqueeze(-1), 0)
    return tokens


TOKEN_MIXING_KERNELS = {"permutation", "low_rank"}
PER_TOKEN_ACTIVATIONS = {"relu", "gelu", "swiglu"}
SHARED_ACTIVATIONS = {"relu", "gelu", "swiglu"}

_SCOPE = "history"
_NORMS = {"layer_norm", "none"}


def _activation(value: t.Any, allowed: set[str], /) -> str:
    if value not in allowed:
        raise ValueError(f"unknown {_SCOPE} activation `{value}`; available: {sorted(allowed)}")
    return value


def residual_wrap(dim: int, op: nn.Module, params: dict[str, t.Any], /) -> nn.Module:
    kind = params.get("norm", "layer_norm")
    if kind not in _NORMS:
        raise ValueError(f"unknown {_SCOPE} norm `{kind}`; available: {sorted(_NORMS)}")
    group = params.get("norm_group")
    if group is not None:
        if kind == "none":
            raise ValueError(f"{_SCOPE} `norm: none` cannot be combined with `norm_group`")
        if not isinstance(group, str) or not group:
            raise ValueError(f"{_SCOPE} `norm_group` must be a non-empty string, got {group!r}")
        return GroupedResidual(dim, op, group)
    if kind == "none":
        return Residual(op)
    return PreNormResidual(dim, op)


def width_after(width: int, module: nn.Module, /) -> int:
    return getattr(module, "output_dim", width)
