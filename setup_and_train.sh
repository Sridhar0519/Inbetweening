#!/usr/bin/env bash
# Setup and launch CPU training on a fresh server.
#
# This script:
#   1. Creates a Python virtual environment
#   2. Installs CPU-only PyTorch + dependencies
#   3. Starts training with CPU-friendly defaults
#
# Usage: bash setup_and_train.sh [--epochs 100] [--batch-size 2] [--image-size 128]
#
# Prerequisites: Python 3.10+ and pip

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults (CPU-friendly)
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-2}"
DATASET="${DATASET:-./output}"
OUTPUT="${OUTPUT:-./training_output}"

# Parse optional CLI overrides
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)      EPOCHS="$2"; shift 2 ;;
        --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
        --image-size)  IMAGE_SIZE="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --dataset)     DATASET="$2"; shift 2 ;;
        --output)      OUTPUT="$2"; shift 2 ;;
        --resume)      RESUME="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Shot Segregation GAN — CPU Training Setup ==="
echo ""

# --- 1. Python Environment ---
if [ ! -d "venv" ]; then
    echo "[1/3] Creating virtual environment..."
    # Try standard venv first, fall back to manual creation without pip
    if ! python3 -m venv venv 2>/dev/null; then
        echo "  python3-venv not available, creating manual virtualenv..."
        mkdir -p venv/bin venv/lib
        # Create a minimal venv by symlinking python
        ln -sf "$(which python3)" venv/bin/python3
        ln -sf "$(which python3)" venv/bin/python
        # Create activation script
        cat > venv/bin/activate << 'ACTIVATE'
export VIRTUAL_ENV="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="$VIRTUAL_ENV/bin:$PATH"
export PYTHONPATH="$VIRTUAL_ENV/lib/python3/site-packages:${PYTHONPATH:-}"
unset PYTHONHOME
ACTIVATE
        mkdir -p "venv/lib/python3/site-packages"
    fi
else
    echo "[1/3] Virtual environment already exists."
fi

source venv/bin/activate
echo "  Python: $(python --version)"

# Ensure pip is available in the venv
if ! python -m pip --version &>/dev/null; then
    echo "  pip not found, bootstrapping from wheels..."
    WHEELS_DIR="$SCRIPT_DIR/wheels"
    if [ -d "$WHEELS_DIR" ]; then
        # Direct install pip from its wheel
        PIP_WHL=$(find "$WHEELS_DIR" -name 'pip-*.whl' | head -1)
        if [ -n "$PIP_WHL" ]; then
            # Extract and run pip to install itself
            python "$PIP_WHL/pip" install --no-index --find-links "$WHEELS_DIR" pip setuptools wheel 2>/dev/null || {
                # Fallback: unzip the wheel and use it
                TMP_PIP="/tmp/_pip_bootstrap_$$"
                python -c "import zipfile; zipfile.ZipFile('$PIP_WHL').extractall('$TMP_PIP')"
                PYTHONPATH="$TMP_PIP" python -m pip install --no-index --find-links "$WHEELS_DIR" \
                    --prefix "$(python -c 'import sys; print(sys.prefix)')" pip setuptools wheel -q 2>/dev/null || \
                PYTHONPATH="$TMP_PIP" python -m pip install --no-index --find-links "$WHEELS_DIR" \
                    pip setuptools wheel -q
                rm -rf "$TMP_PIP"
            }
        fi
    else
        echo "  ERROR: No wheels/ directory found and no internet available."
        echo "  Run 'bash download_wheels.sh' on a machine with internet first."
        exit 1
    fi
fi

# --- 2. Install Dependencies (CPU-only PyTorch) ---
WHEELS_DIR="$SCRIPT_DIR/wheels"
if [ -d "$WHEELS_DIR" ] && [ "$(find "$WHEELS_DIR" -name '*.whl' | wc -l)" -gt 0 ]; then
    echo "[2/3] Installing dependencies from local wheels (offline)..."
    python -m pip install --no-index --find-links "$WHEELS_DIR" \
        torch torchvision Pillow numpy tqdm setuptools wheel -q
else
    echo "[2/3] Installing dependencies (CPU-only PyTorch)..."
    python -m pip install --upgrade pip -q
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu -q
    python -m pip install Pillow numpy tqdm -q
fi

echo "  Installed: torch $(python -c 'import torch; print(torch.__version__)')"
echo "  CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# --- 3. Start Training ---
LOG_DIR="$OUTPUT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "[3/3] Starting training..."
echo ""
echo "  Config:"
echo "    Epochs:      $EPOCHS"
echo "    Batch size:  $BATCH_SIZE"
echo "    Image size:  $IMAGE_SIZE"
echo "    Workers:     $NUM_WORKERS"
echo "    Dataset:     $DATASET"
echo "    Output:      $OUTPUT"
echo "    Log file:    $LOG_FILE"
echo ""

TRAIN_CMD="python -u -m model.train \
    --dataset $DATASET \
    --output $OUTPUT \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --image-size $IMAGE_SIZE \
    --num-workers $NUM_WORKERS \
    --save-every 5 \
    --sample-every 5 \
    --fast-cpu"

if [ -n "${RESUME:-}" ]; then
    TRAIN_CMD="$TRAIN_CMD --resume $RESUME"
    echo "  Resuming from: $RESUME"
fi

echo "Running: $TRAIN_CMD"
echo "Logging to: $LOG_FILE"
echo ""

# Run training — output goes to both terminal and log file
eval "$TRAIN_CMD" 2>&1 | tee "$LOG_FILE"
