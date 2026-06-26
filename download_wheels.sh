#!/usr/bin/env bash
# Download all Python dependencies (CPU-only) as wheel files for offline install.
# Run THIS on a machine with internet access.
#
# Usage: bash download_wheels.sh
#        bash download_wheels.sh --python 3.11   # target a specific Python version
# Output: ./wheels/ directory with all .whl files
#
# Then transfer the wheels/ folder to the offline server along with the code.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Default to server's Python version (3.11)
TARGET_PYTHON="3.11"
if [[ "${1:-}" == "--python" ]]; then
    TARGET_PYTHON="${2:-3.11}"
fi

WHEELS_DIR="$SCRIPT_DIR/wheels"
rm -rf "$WHEELS_DIR"
mkdir -p "$WHEELS_DIR"

echo "=== Downloading CPU-only wheels for offline install ==="
echo "Target: $WHEELS_DIR"
echo "Target Python: $TARGET_PYTHON"
echo "Target platform: manylinux_2_28_x86_64"
echo ""

PLATFORM_ARGS="--python-version $TARGET_PYTHON --platform manylinux_2_28_x86_64 --only-binary=:all:"

# Download pip itself (pure python, no platform needed)
echo "[1/4] Downloading pip, setuptools, wheel..."
python3 -m pip download pip setuptools wheel -d "$WHEELS_DIR" -q

# Download CPU-only PyTorch for the target platform
echo "[2/4] Downloading PyTorch (CPU-only) for Python $TARGET_PYTHON..."
python3 -m pip download torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu \
    $PLATFORM_ARGS \
    -d "$WHEELS_DIR" -q

# Download other training deps for target platform
echo "[3/4] Downloading Pillow, numpy, tqdm..."
python3 -m pip download Pillow numpy tqdm \
    $PLATFORM_ARGS \
    -d "$WHEELS_DIR" -q

# Download any pure-python deps that might be needed (typing_extensions, etc.)
echo "[4/4] Downloading remaining pure-python deps..."
python3 -m pip download typing_extensions sympy networkx jinja2 markupsafe filelock fsspec mpmath \
    -d "$WHEELS_DIR" -q 2>/dev/null || true

echo ""
echo "=== Done ==="
TOTAL_SIZE=$(du -sh "$WHEELS_DIR" | cut -f1)
NUM_FILES=$(find "$WHEELS_DIR" -name '*.whl' | wc -l)
echo "Downloaded $NUM_FILES wheel files ($TOTAL_SIZE)"
echo "Target: Python $TARGET_PYTHON on Linux x86_64"
echo ""
echo "Transfer wheels/ to the server along with the code, then run:"
echo "  bash setup_and_train.sh"
