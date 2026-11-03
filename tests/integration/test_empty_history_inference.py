import polars as pl
import pytest

from perseus.api.make_backbone_embeddings import make_backbone_embeddings
from perseus.api.prepare_dataset import prepare_dataset
from tests.integration.conftest import SAMPLE_DATE, build_trained, partition_of, setup_purchase_source
from tests.unit.api.conftest import make_config


@pytest.mark.parametrize("backend", ["disk", "ram", "cached"])
def test_inference_keeps_clients_without_events(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(tmp_path / "hub"))
    setup_purchase_source(tmp_path / "hub", ["known"])
    trained = build_trained()
    trained.config.prepare.allow_empty_sequences = True
    trained.config.inference.event_store_backend = backend
    samples = pl.DataFrame({"_index": [7], "client_id": ["missing"], "timestamp": [SAMPLE_DATE]})
    result = make_backbone_embeddings(partition_of("missing"), samples, trained)
    assert result is not None
    assert result["_index"].to_list() == [7]
    assert result["embeddings"].to_list()[0]


def test_prepare_preserves_clients_without_events(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(tmp_path / "hub"))
    setup_purchase_source(tmp_path / "hub", ["known"])
    config = make_config()
    config.prepare.allow_empty_sequences = True
    train = pl.DataFrame({
        "client_id": ["known", "missing"], "timestamp": [SAMPLE_DATE, SAMPLE_DATE],
        "target": [{"label": 0}, {"label": 1}],
    })
    _prepared, dataset = prepare_dataset(config, train, train.head(0))
    assert set(dataset.train_samples["client_id"].to_list()) == {"known", "missing"}
    assert dataset.event_store.open_reader(2).read("missing").is_empty()


def test_prepare_all_empty_id_histories(tmp_path, monkeypatch):
    monkeypatch.setenv("INTERNAL_STORAGE_EVENT_HUB", str(tmp_path / "hub"))
    setup_purchase_source(tmp_path / "hub", ["known"])
    config = make_config()
    config.prepare.allow_empty_sequences = True
    train = pl.DataFrame({
        "client_id": ["missing"], "timestamp": [SAMPLE_DATE], "target": [{"label": 0}],
    })
    prepared, dataset = prepare_dataset(config, train, train.head(0))
    assert dataset.train_samples.height == 1
    assert dataset.event_store.open_reader(2).read("missing").is_empty()
    assert prepared.encoder_to_preprocessor["event"].cardinality == 0
