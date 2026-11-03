#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
workdir="${1:-$repo_root/workdir/quickstart}"
mkdir -p "$workdir"
workdir="$(cd -- "$workdir" && pwd)"
export INTERNAL_STORAGE_EVENT_HUB="$workdir/event-hub"
export ACCELERATE_USE_CPU=true
python "$repo_root/examples/quickstart.py" --workdir "$workdir"
python -m perseus event-hub add-events "$workdir/events.pq" --name purchase
python -m perseus train prepare-dataset --workdir "$workdir"
python -m perseus train fit-model --workdir "$workdir"
echo "Checkpoint: $workdir/checkpoint"
echo "Test predictions: $workdir/predictions/test/predictions.parquet"
