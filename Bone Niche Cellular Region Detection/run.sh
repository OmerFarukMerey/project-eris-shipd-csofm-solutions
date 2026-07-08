#!/usr/bin/env bash
# Sets up a venv, installs deps (with a CUDA-wheel fallback for very new GPUs
# like the RTX 50-series / Blackwell), and runs solution.py end to end.
#
# Usage:
#   ./run.sh
#   BONE_NUM_EPOCHS=20 ./run.sh          # override epoch count
#   BONE_BATCH_SIZE=32 ./run.sh          # override batch size
# (solution.py auto-scales batch size/workers/lr for CUDA already; these env
# vars just let you override its defaults without editing the file.)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtualenv at $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip >/dev/null

echo "Installing dependencies..."
pip install -r requirements.txt

cuda_ok=$(python - <<'PY'
import torch
print("yes" if torch.cuda.is_available() else "no")
PY
)

if [ "$cuda_ok" != "yes" ]; then
    echo "CUDA not detected with the default torch build."
    echo "Retrying with the latest stable CUDA 12.6 wheels from pytorch.org ..."
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu126 || true
    cuda_ok=$(python - <<'PY'
import torch
print("yes" if torch.cuda.is_available() else "no")
PY
    )
fi

if [ "$cuda_ok" != "yes" ]; then
    echo "WARNING: torch still does not see a CUDA device."
    echo "Very new GPUs (e.g. RTX 50-series / Blackwell, sm_120) may need a"
    echo "newer PyTorch than what's on stable pip yet. Try the nightly build:"
    echo "  pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128"
    echo "solution.py will still run on CPU if you skip this, just much slower."
fi

python - <<'PY'
import torch
print(f"torch {torch.__version__} | cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

echo "Starting training + inference (solution.py) ..."
python solution.py

echo "Done. Submission written to working/submission.csv"
