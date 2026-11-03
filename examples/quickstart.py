"""Create deterministic synthetic data and a small CPU-compatible recipe."""
import argparse
import datetime as dt
from pathlib import Path

import polars as pl
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    workdir = args.workdir
    clients = [f'user-{index}' for index in range(12)]
    events = pl.DataFrame({
        'client_id': [client for client in clients for _ in range(2)],
        'timestamp': [dt.datetime(2024, 1, 1), dt.datetime(2024, 1, 2)] * len(clients),
    }).with_columns(pl.col('timestamp').cast(pl.Datetime('ns')))
    workdir.mkdir(parents=True, exist_ok=True)
    events.write_parquet(workdir / 'events.pq')
    for split, selected in [('train', clients[:8]), ('test', clients[8:])]:
        target = workdir / 'basis' / split
        target.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            'client_id': selected,
            'timestamp': [dt.datetime(2024, 2, 1)] * len(selected),
            'target': [{'label': index % 2} for index in range(len(selected))],
        }).with_columns(pl.col('timestamp').cast(pl.Datetime('ns'))).write_parquet(target / 'samples.pq')
    config = {
        'task': {'type': 'classification', 'metrics': {'logloss': None, 'roc_auc': {'params': {'pos_label': '1'}}}},
        'events': {'purchase': None},
        'max_events_per_sequence': 4,
        'backbone': {'dim': 8, 'history_aggregator': {
            'sequence_encoder': {'type': 'ffn', 'params': {'num_layers': 1}},
        }},
        'training': {'num_epochs': 1, 'seed': 42, 'mixed_precision': 'no',
                     'event_store_backend': 'cached',
                     'dataloader': {'batch_size': 4, 'num_workers': 0},
                     'test_dataloader': {'batch_size': 4, 'num_workers': 0}},
    }
    (workdir / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    print(f'Created synthetic events, basis and configuration in {workdir}')


if __name__ == '__main__':
    main()
