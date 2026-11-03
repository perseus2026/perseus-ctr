"""Training schema: new typed fields, optimizer defaults, validation."""

import pytest
from pydantic import ValidationError

from perseus.core.config import Config, Optimizer, Training


def _config(training: dict) -> Config:
    return Config.model_validate(
        {
            "task": {"type": "classification", "metrics": {"logloss": None}},
            "events": {"purchase": None},
            "backbone": {"dim": 8},
            "training": training,
        },
    )


class TestLegacyDefaults:
    def test_new_fields_default_to_legacy_behavior(self) -> None:
        training = Training()
        assert training.initialization is None
        assert training.gradient_clipping is None
        assert training.accumulation is None
        assert training.mixed_precision is None

    def test_legacy_config_loads_without_new_fields(self) -> None:
        config = _config({})
        assert config.training.accumulation is None
        assert config.training.initialization is None


class TestNewFields:
    def test_recipe_and_clipping_parse(self) -> None:
        config = _config(
            {
                "initialization": {
                    "linear": {"kind": "xavier_normal"},
                    "embedding": {"kind": "normal", "std": 1e-4},
                },
                "gradient_clipping": {"max_norm": 10.0, "scope": "dense"},
                "accumulation": 4,
                "mixed_precision": "bf16",
            },
        )
        assert config.training.initialization.linear.kind == "xavier_normal"
        assert config.training.initialization.embedding.std == 1e-4
        assert config.training.gradient_clipping.max_norm == 10.0
        assert config.training.accumulation == 4
        assert config.training.mixed_precision == "bf16"

    def test_recipe_fields_persist_through_save_load(self, tmp_path) -> None:
        config = _config({"gradient_clipping": {"max_norm": 5.0}, "accumulation": 2})
        config.save(tmp_path / "config.yaml")
        loaded = Config.load(tmp_path / "config.yaml")
        assert loaded.training.gradient_clipping.max_norm == 5.0
        assert loaded.training.accumulation == 2

    def test_new_fields_round_trip_model_dump(self) -> None:
        training = Training(accumulation=3)
        restored = Training.model_validate(training.model_dump())
        assert restored.accumulation == 3


class TestFieldTypes:
    def test_accumulation_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            Training(accumulation=0)

    def test_unknown_mixed_precision_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Training(mixed_precision="fp8")

    def test_inference_accepts_cached_event_store(self) -> None:
        config = Config.model_validate(
            {
                "task": {"type": "classification", "metrics": {"logloss": None}},
                "events": {"purchase": None},
                "backbone": {"dim": 8},
                "inference": {"event_store_backend": "cached"},
            },
        )
        assert config.inference.event_store_backend == "cached"


class TestLegacyOptimizerDefaults:
    def test_empty_params_keep_torch_defaults(self) -> None:
        assert Optimizer(type="adamw").params == {}
        assert Optimizer(type="adagrad").params == {}
        assert Optimizer(type="sgd").params == {}

    def test_explicit_params_pass_through(self) -> None:
        params = Optimizer(type="adamw", params={"lr": 1e-4, "weight_decay": 0.0, "eps": 1e-8}).params
        assert params["lr"] == 1e-4
        assert params["weight_decay"] == 0.0
        assert params["eps"] == 1e-8

    def test_reference_recipe_params_are_explicit_in_config(self) -> None:
        config = _config(
            {
                "optimizers": {
                    "default": {
                        "type": "adamw",
                        "params": {"lr": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0},
                    },
                    "embeddings": {
                        "type": "adagrad",
                        "params": {"lr": 0.05, "eps": 1e-10, "initial_accumulator_value": 0.1},
                    },
                },
            },
        )
        assert config.training.optimizers.default.params["lr"] == 1e-4
        assert config.training.optimizers.embeddings.params["initial_accumulator_value"] == 0.1


def test_unrelated_unknown_training_fields_keep_master_behavior() -> None:
    training = Training.model_validate(
        {"unrelated_option": True, "optimizer": {"type": "sgd"}, "sparse_optimizer": {"type": "adagrad"}},
    )
    assert training.optimizers.default.type == "adamw"
