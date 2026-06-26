#!/usr/bin/env bash
# Pack the project for transfer to a fresh server.
# Creates a tarball with code + dataset, excluding caches and old checkpoints.
#
# Usage: bash pack_for_transfer.sh
# Output: shotSegregation_transfer.tar.gz (in parent directory)

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(dirname "$PROJECT_DIR")"
ARCHIVE="$PARENT_DIR/shotSegregation_transfer.tar.gz"

echo "=== Packing project for transfer ==="
echo "Source:  $PROJECT_DIR"
echo "Archive: $ARCHIVE"
echo "(Includes wheels/ for offline install)"
echo ""

cd "$PARENT_DIR"

# Exclude __pycache__, old training outputs/checkpoints, .git, ipynb checkpoints
tar czf "$ARCHIVE" \
    --exclude='__pycache__' \
    --exclude='.ipynb_checkpoints' \
    --exclude='*.pyc' \
    --exclude='test_training' \
    --exclude='training_output/checkpoints/*.pt' \
    --exclude='training_output/samples' \
    --exclude='.git' \
    --exclude='output' \
    --exclude='venv' \
    "$(basename "$PROJECT_DIR")"

ARCHIVE_SIZE=$(du -sh "$ARCHIVE" | cut -f1)
echo ""
echo "=== Done ==="
echo "Archive: $ARCHIVE ($ARCHIVE_SIZE)"
echo ""
echo "Transfer to your server with:"
echo "  scp $ARCHIVE user@your-server:~/"
echo ""
echo "Then on the server, run:"
echo "  tar xzf shotSegregation_transfer.tar.gz"
echo "  cd shotSegregation"
echo "  bash setup_and_train.sh"
