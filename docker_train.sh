#!/usr/bin/env bash
# Build and run training inside Docker.
# Usage:
#   bash docker_train.sh
#   bash docker_train.sh --epochs 50 --batch-size 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p training_output

echo "=== Building training image ==="
docker compose build train

echo "=== Starting training container ==="
if [[ $# -gt 0 ]]; then
    docker compose run --rm train "$@"
else
    docker compose run --rm train
fi
