"""Shared fixtures for all Perseus tests."""

import copy
import typing as t
from pathlib import Path

import pytest
import torch
import yaml

# Minimal valid configuration: one classification task, one event, and a backbone.
# model_post_init adds the "event" encoder and feature automatically.
MINIMAL_CONFIG: dict[str, t.Any] = {
    "task": {
        "type": "classification",
        "metrics": {"logloss": None},
    },
    "events": {"purchase": None},
    "backbone": {"dim": 8},
}


@pytest.fixture
def minimal_config_dict() -> dict[str, t.Any]:
    """Return a fresh copy of the minimal valid configuration dictionary."""
    return copy.deepcopy(MINIMAL_CONFIG)


@pytest.fixture
def minimal_config_path(tmp_path: Path, minimal_config_dict: dict[str, t.Any]) -> Path:
    """Write the minimal configuration to a temporary YAML file."""
    path = tmp_path / "config.yaml"
    with path.open("w") as f:
        yaml.safe_dump(minimal_config_dict, f, allow_unicode=True, sort_keys=False)
    return path


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    """Seed PyTorch for deterministic tests."""
    torch.manual_seed(0)
