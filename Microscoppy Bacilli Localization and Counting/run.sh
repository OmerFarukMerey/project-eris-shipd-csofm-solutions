#!/usr/bin/env bash
# ======================================================================================
# run.sh — FAST run that finishes end-to-end in ~20-26 min (hard cap 30 min)
# --------------------------------------------------------------------------------------
# yolo11m @1280 -> trains, then predicts, then writes submission.csv, all in one process.
#
# RUN IT INSIDE tmux so a dropped SSH connection can't kill it:
#     tmux new -s job          # start persistent session
#     bash run.sh              # (detach: Ctrl-b then d ; reattach: tmux attach -t job)
#
# NOTE: this trains a FRESH model and overwrites any previous checkpoint in _work/.
#       If you want to keep an earlier best.pt, copy it out first:
#         cp _work/runs/detector/weights/best.pt ~/prev_best.pt
#       Or, to just submit an existing checkpoint without retraining, skip this and run:
#         DEVICE=0 python infer_only.py
# ======================================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "==================================================================="
echo " Bacilli detection — FAST 30-min run"
echo "==================================================================="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "!! nvidia-smi failed — are you on a GPU pod?"; exit 1; }

if [ ! -f dataset/public/train.csv ]; then
  echo "!! dataset/public/train.csv not found. Upload the dataset/ folder next to run.sh."; exit 1
fi

rm -rf _work   # clean, predictable dataset + fresh run

# --- FAST configuration (all read by solution.py via env vars) ---
export MODEL=yolo11m.pt      # ~4x faster than yolo11x, still strong
export IMGSZ=1280            # good resolution for the small rods
export BATCH=16              # fits easily on any >=16 GB GPU (no OOM at yolo11m/1280)
export EPOCHS=50             # finishes well inside the budget; guard caps it if slow
export VAL_FRAC=0.10

export N_SYNTH_BLUE=400      # keep solid domain augmentation (blue-shift is 0.14 of the score)
export N_RODPASTE=120
export N_VIGNETTE=60

export INFER_IMGSZ=1536      # inference resolution (warm domain)
export BLUE_IMGSZ=1536       # blue 640px domain: upscale tiny rods
export PNG_IMGSZ=1280

export USE_COUNT_REGRESSOR=0 # OFF for a guaranteed fast finish (detector-count is fine)
export WALL_BUDGET_H=0.5     # HARD CAP 30 min; stops training at ~24 min, then infers
export DEVICE=0

echo "Config: MODEL=$MODEL IMGSZ=$IMGSZ BATCH=$BATCH EPOCHS=$EPOCHS BUDGET=${WALL_BUDGET_H}h"
echo "        INFER=$INFER_IMGSZ BLUE=$BLUE_IMGSZ PNG=$PNG_IMGSZ  regressor=$USE_COUNT_REGRESSOR"
echo "-------------------------------------------------------------------"

# deps are usually already installed from a prior run; these are quick no-ops if so
python -m pip install -q ultralytics ensemble-boxes opencv-python-headless albumentations || true

echo "Self-check (grader anchors) ..."
python solution.py --selfcheck

echo "-------------------------------------------------------------------"
echo "Full run starting (train -> predict -> submission.csv) ..."
time python solution.py

echo "==================================================================="
echo " DONE.  ->  $(pwd)/submission.csv"
wc -l submission.csv || true
echo "==================================================================="
