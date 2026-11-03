"""Pooling history and context tokens into task inputs."""

import typing as t

import torch

Pool = str | dict[str, int] | None


def masked_mean(tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    valid = (~padding_mask).unsqueeze(-1).to(tokens.dtype)
    return (tokens * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

def output_shape(
    pools: list[tuple[Pool, str | None]],
    lengths: dict[str, int | None],
    widths: dict[str, int],
    /,
    *,
    readout: bool,
) -> tuple[int, ...]:
    selected = [(pool, source) for pool, source in pools if pool is not None]
    if not selected:
        raise ValueError("history requires an output pool")
    for pool, source in selected:
        if source is None or lengths[source] is None:
            raise ValueError("output pool requires a token group produced by the component")
        if pool == "readout" and not readout:
            raise ValueError("readout pool requires the backbone readout token")
        if source != "sequence" and lengths[source] == 0:
            raise ValueError(f"output pool source `{source}` is empty: add context features or the readout token")
        if isinstance(pool, dict) and pool["select"] > lengths[source]:
            raise ValueError("selected output exceeds the number of tokens")
    selects = [(pool, source) for pool, source in selected if isinstance(pool, dict) or pool == "readout"]
    if selects and len(selected) > 1:
        raise ValueError("select or readout cannot be combined with other output pools")
    if len(selected) == 1 and (isinstance(selected[0][0], dict) or selected[0][0] == "readout"):
        pool, source = selected[0]
        return (pool["select"] if isinstance(pool, dict) else 1, widths[source])
    return (sum(widths[source] * (lengths[source] if pool == "flatten" else 1) for pool, source in selected),)


def pool(
    groups: dict[str, torch.Tensor], pools: list[tuple[Pool, str | None]], padding_mask: torch.Tensor, /,
) -> torch.Tensor:
    """Apply configured pools. The readout pool selects the last token of its group."""
    outputs = []
    for operation, source in pools:
        if operation is None:
            continue
        tokens = groups[t.cast("str", source)]
        if operation == "mean":
            outputs.append(masked_mean(tokens, padding_mask) if source == "sequence" else tokens.mean(dim=1))
        elif operation == "flatten":
            outputs.append(tokens.flatten(start_dim=1))
        elif operation == "readout":
            outputs.append(tokens[:, -1:])
        else:
            outputs.append(tokens[:, -operation["select"]:])
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=-1)
