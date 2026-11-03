"""Real two-rank CPU/Gloo lifecycle checks, isolated from pytest's Accelerator state."""

import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest
import torch
from accelerate import Accelerator

from perseus import api, core
from perseus.cli.train import fit_model
from tests.integration.test_training_recipe import _config, _dataset
from tests.unit.api.conftest import make_id_preprocessor


def _prepare(workdir, backend, *, reproduction=False):
    config = _config(2, accumulation=2, event_store_backend=backend)
    if backend == "cached":
        config.backbone.history_aggregator = core.config.HistoryAggregator.model_validate({
            "num_layers": 2,
            "sequence_encoder": {"type": "ffn"},
            "nonsequence_encoder": {"type": "ffn", "pool": "flatten"},
            "interaction": {"type": "cross_attention", "schedule": "layered",
                            "params": {"projection": "compact"}},
        })
    config.max_events_per_sequence = 4
    config.training.test_dataloader.num_workers = 0
    config.training.test_dataloader.batch_size = 2
    preprocessors = {"event": make_id_preprocessor()}
    dataset = _dataset(workdir, [0, 1, 1, 0, 1, 0, 0, 1, 0])
    if reproduction:
        config.training.mixed_precision = "bf16"
        config.task.layer["binary_output"] = "probabilities"
        config.backbone.zero_context_features = ["candidate_action"]
        config.backbone.history_aggregator.sequence_state_fp32 = True
        config.backbone.event_aggregator = core.config.Aggregator(type="concat_linear")
        config.backbone.context_aggregator = core.config.Aggregator(type="tokenizer")
        config.encoders["candidate_action"] = core.config.Encoder(type="id")
        config.features["candidate_action"] = core.config.Feature(
            located_in=core.config.LocatedIn(context=True), encoder="candidate_action",
        )
        preprocessors["candidate_action"] = make_id_preprocessor()
        dataset.train_samples = dataset.train_samples.with_columns(context=pl.struct(candidate_action=pl.lit(1)))
    prepared = core.checkpoint.Prepared(config, preprocessors)
    # Multiple batches and a tail exercise distributed eval de-duplication.
    dataset.test_samples = dataset.train_samples.head(5).with_row_index("source_row")
    prepared.save(workdir / "checkpoint")
    dataset.save(workdir / "dataset")


def _run(tmp_path, mode, backend):
    _prepare(tmp_path, backend, reproduction=mode == "reproduction")
    env = os.environ.copy()
    env.update(ACCELERATE_USE_CPU="true", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               POLARS_MAX_THREADS="1", PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    env.pop("ACCELERATE_GRADIENT_ACCUMULATION_STEPS", None)
    env.pop("ACCELERATE_MIXED_PRECISION", None)
    log_path = tmp_path / "torchrun.log"
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with log_path.open("w") as log:
        process = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "torch.distributed.run", "--master-addr=127.0.0.1",
         f"--master-port={port}", "--nproc_per_node=2",
         str(Path(__file__).resolve()), "--worker", str(tmp_path), mode],
            env=env, stdout=log, stderr=subprocess.STDOUT, text=True, start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
            raise AssertionError(f"two-rank smoke timed out; log:\n{log_path.read_text()}") from None
    assert returncode == 0, log_path.read_text()


@pytest.mark.parametrize("backend", ["disk", "cached"])
def test_two_rank_fit_accumulation_eval_and_weights(tmp_path, backend):
    _run(tmp_path, "api", backend)
    left = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    right = torch.load(tmp_path / "rank-1.pt", weights_only=True)
    assert left.keys() == right.keys()
    for key in left:
        assert torch.isfinite(left[key]).all(), key
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    reports = json.loads((tmp_path / "metrics-0.json").read_text())
    assert any("logloss/label" in report for report in reports)
    assert not (tmp_path / "metrics-1.json").exists()
    exported = pl.read_parquet(tmp_path / "predictions" / "test" / "predictions.parquet")
    assert exported["source_row"].to_list() == list(range(5))
    assert "_index" not in exported.columns


def test_two_rank_cli_publishes_checkpoint_once(tmp_path):
    _run(tmp_path, "cli", "cached")
    writers = sorted(path.name for path in tmp_path.glob("save-called-*.txt"))
    assert writers == ["save-called-0.txt"], f"shared checkpoint writers: {writers}"
    core.checkpoint.Trained.load(tmp_path / "checkpoint")
    exported = pl.read_parquet(tmp_path / "predictions" / "test" / "predictions.parquet")
    assert exported["source_row"].to_list() == list(range(5))


def test_two_rank_bf16_reproduction_options(tmp_path):
    _run(tmp_path, "reproduction", "cached")
    left = torch.load(tmp_path / "rank-0.pt", weights_only=True)
    right = torch.load(tmp_path / "rank-1.pt", weights_only=True)
    for key in left:
        assert torch.isfinite(left[key]).all(), key
        torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)
    exported = pl.read_parquet(tmp_path / "predictions" / "test" / "predictions.parquet")
    assert exported["source_row"].to_list() == list(range(5))


def _worker(workdir, mode):
    # Accelerate passes GPU device_ids to CPU barriers too; recent PyTorch
    # interprets them as MPS devices on Macs. Keep this CPU-only smoke on Gloo.
    if sys.platform == "darwin":
        Accelerator.wait_for_everyone = lambda self: torch.distributed.barrier()

    rank = int(os.environ["RANK"])
    observed = []
    if mode == "reproduction":
        layer_cls = core.tasks.registry["classification"].layer
        original_forward = layer_cls.forward
        original_bce = torch.nn.functional.binary_cross_entropy

        def check_forward(self, *args):
            predictions = original_forward(self, *args)
            assert all(value.dtype == torch.bfloat16 for value in predictions)
            observed.append("bf16 sigmoid")
            return predictions

        def check_bce(prediction, target, *args, **kwargs):
            assert not torch.is_autocast_enabled("cpu")
            assert prediction.dtype == target.dtype == torch.float32
            observed.append("fp32 bce")
            return original_bce(prediction, target, *args, **kwargs)

        layer_cls.forward = check_forward
        torch.nn.functional.binary_cross_entropy = check_bce

    class RecordingTracker:
        def __init__(self, **kwargs):
            self.reports = []

        def log_metrics(self, metrics, *, iteration):
            self.reports.append(metrics)
            (workdir / f"metrics-{rank}.json").write_text(json.dumps(self.reports))

    core.trackers.registry["ddp-smoke"] = RecordingTracker
    overrides = {"training": {"tracker": {"type": "ddp-smoke", "params": {}}}}
    if mode == "cli":
        original_save = core.checkpoint.Trained.save

        def recording_save(self, path):
            (workdir / f"save-called-{rank}.txt").write_text(str(path))
            return original_save(self, path)

        core.checkpoint.Trained.save = recording_save
        fit_model(workdir=workdir, overrides=overrides)
    else:
        prepared = core.checkpoint.Prepared.load(workdir / "checkpoint", overrides=overrides)
        dataset = core.Dataset.load(workdir / "dataset", prepared.config)
        trained = api.fit_model(prepared, dataset, test_predictions_dir=workdir / "predictions" / "test")
        if mode == "reproduction":
            assert {"bf16 sigmoid", "fp32 bce"} <= set(observed)
        state = {}
        for name, module in [("backbone", trained.backbone), ("head", trained.head.layer),
                             *trained.encoder_to_embedder.items()]:
            state.update({f"{name}.{key}": value.cpu() for key, value in module.state_dict().items()})
        torch.save(state, workdir / f"rank-{rank}.pt")
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    _worker(Path(sys.argv[2]), sys.argv[3])
