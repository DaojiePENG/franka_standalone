#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# LeapBot Inference Server — launch script (GPU machine)
#
# Prerequisites:
#   1. conda env 'vj_fw' with fastwam_infer, torch, fastapi, uvicorn, opencv-python
#   2. TensorRT engine (or PyTorch) checkpoint ready under $ASSET_ROOT
#
# Usage:
#   bash launch_server.sh                           # defaults
#   TASK=press_three_buttons bash launch_server.sh  # override task
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Activate conda environment ────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-vj_fw}"

# Source conda.sh so `conda activate` works inside a script
_CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
if [[ -f "${_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${_CONDA_BASE}/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source "/opt/conda/etc/profile.d/conda.sh"
else
    echo "[WARNING] conda.sh not found — trying 'conda activate' directly"
fi

conda activate "$CONDA_ENV"
echo "[launch_server] conda env: $CONDA_ENV ($(python --version 2>&1))"

# ── Configurable variables (override via environment) ─────────────────────────
ASSET_ROOT="${ASSET_ROOT:-/home/aihub/daojie/LeapBot-inference-asset}"
TASK="${TASK:-move_objects_into_box}"
INFER_ROOT="${INFER_ROOT:-/home/aihub/daojie/LeapBot-inference-only}"
DEVICE="${DEVICE:-cuda:0}"
BACKEND="${BACKEND:-tensorrt}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
# Set ENABLE_FISHEYE=1 to turn on KB4 undistortion for wrist_image
ENABLE_FISHEYE="${ENABLE_FISHEYE:-0}"

# ── Build args ────────────────────────────────────────────────────────────────
ARGS=(
    --asset_root "$ASSET_ROOT"
    --task       "$TASK"
    --infer_root "$INFER_ROOT"
    --device     "$DEVICE"
    --backend    "$BACKEND"
    --port       "$PORT"
    --host       "$HOST"
)
if [[ "$ENABLE_FISHEYE" == "1" ]]; then
    ARGS+=(--enable_fisheye_undistortion)
fi

# ── Info banner ───────────────────────────────────────────────────────────────
echo "================================================================="
echo "  LeapBot Inference Server"
echo "  CONDA ENV   : $CONDA_ENV"
echo "  ASSET_ROOT  : $ASSET_ROOT"
echo "  TASK        : $TASK"
echo "  DEVICE      : $DEVICE"
echo "  BACKEND     : $BACKEND"
echo "  ENDPOINT    : http://${HOST}:${PORT}"
echo "  FISHEYE     : ${ENABLE_FISHEYE}"
echo "================================================================="
echo ""

# ── Launch ────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"
exec python server.py "${ARGS[@]}"
