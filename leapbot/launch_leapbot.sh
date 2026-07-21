#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# LeapBot NUC-side launcher — starts franka_server.py + leapbot_control.py
#
# This script runs on the NUC / robot control host. It:
#   1. Starts franka_server.py (ZeroRPC, polymetis + libfranka)
#   2. Waits for the Franka server to come up
#   3. Starts leapbot_control.py (camera + inference + control loop)
#
# Prerequisites (on this machine):
#   - polymetis env active (conda activate <env>)
#   - franka_server.py dependencies installed (zerorpc, polymetis, panda_py, scipy)
#   - leapbot_control.py dependencies installed (requests, opencv-python, pynput)
#   - GPU inference server running on $SERVER_IP:$SERVER_PORT
#
# Usage:
#   SERVER_IP=192.168.1.100 bash launch_leapbot.sh
#   SERVER_IP=192.168.1.100 TASK=press_three_buttons bash launch_leapbot.sh
#   SERVER_IP=192.168.1.100 CAM_GLOBAL=zed_1 CAM_WRIST=fisheye bash launch_leapbot.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FRANKA_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Activate conda environment ────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-polymetis}"
_CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
if [[ -f "${_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${_CONDA_BASE}/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"
echo "[launch_leapbot] conda env: $CONDA_ENV ($(python --version 2>&1))"

# ── Configurable variables (override via environment) ─────────────────────────

# Franka server (polymetis / libfranka)
ROBOT_IP="${ROBOT_IP:-localhost}"
GRIPPER_ROBOT_IP="${GRIPPER_ROBOT_IP:-10.168.1.200}"
FRANKA_PORT="${FRANKA_PORT:-4242}"

# GPU inference server
SERVER_IP="${SERVER_IP:?ERROR: Set SERVER_IP to the GPU machine address}"
SERVER_PORT="${SERVER_PORT:-8000}"
TASK="${TASK:-move_objects_into_box}"

# Control
FREQUENCY="${FREQUENCY:-10}"
ACTION_HORIZON="${ACTION_HORIZON:-4}"

# Camera → model view mapping (which physical camera → which model input)
CAM_GLOBAL="${CAM_GLOBAL:-zed_0}"
CAM_WRIST="${CAM_WRIST:-fisheye}"
CAM_VIS="${CAM_VIS:-zed_1,l515_0}"

# Safety limits
MAX_DELTA_POS="${MAX_DELTA_POS:-0.08}"
MAX_DELTA_ROT="${MAX_DELTA_ROT:-0.3}"

# Workspace bounds (from calibrate_workspace.py output)
SAFETY_X_MIN="${SAFETY_X_MIN:-0.20}"
SAFETY_X_MAX="${SAFETY_X_MAX:-0.70}"
SAFETY_Y_MIN="${SAFETY_Y_MIN:--0.40}"
SAFETY_Y_MAX="${SAFETY_Y_MAX:-0.40}"
SAFETY_Z_MIN="${SAFETY_Z_MIN:-0.05}"
SAFETY_Z_MAX="${SAFETY_Z_MAX:-0.60}"

SKIP_HOME="${SKIP_HOME:-0}"

# ── Info banner ───────────────────────────────────────────────────────────────
echo "================================================================="
echo "  LeapBot NUC Launcher"
echo "  CONDA ENV    : ${CONDA_ENV}"
echo "  Franka server: ${ROBOT_IP}:${FRANKA_PORT}"
echo "  Gripper IP   : ${GRIPPER_ROBOT_IP}"
echo "  GPU server   : ${SERVER_IP}:${SERVER_PORT}"
echo "  Task         : ${TASK}"
echo "  Frequency    : ${FREQUENCY} Hz"
echo "  Horizon      : ${ACTION_HORIZON}"
echo "  global_image : ${CAM_GLOBAL}"
echo "  wrist_image  : ${CAM_WRIST}"
echo "  vis extras   : ${CAM_VIS}"
echo "  Safety       : pos≤${MAX_DELTA_POS}m  rot≤${MAX_DELTA_ROT}rad"
echo "================================================================="
echo ""

# ── Pre-flight: check GPU server ──────────────────────────────────────────────
echo "[Pre-flight] Checking GPU server at ${SERVER_IP}:${SERVER_PORT} ..."
if curl -sf "http://${SERVER_IP}:${SERVER_PORT}/ready" >/dev/null 2>&1; then
    echo "[Pre-flight] GPU server ready ✓"
else
    echo "[WARNING] GPU server not ready at ${SERVER_IP}:${SERVER_PORT}"
    echo "          Make sure launch_server.sh is running on the GPU machine."
    echo "          Continuing anyway — the control loop will retry ..."
    echo ""
fi

# ── Start franka_server.py ────────────────────────────────────────────────────
echo ""
echo "[Step 1] Starting franka_server.py ..."
cd "$FRANKA_DIR"
python franka_server.py \
    --robot_ip         "$ROBOT_IP" \
    --gripper_robot_ip "$GRIPPER_ROBOT_IP" \
    --port             "$FRANKA_PORT" \
    &
FRANKA_PID=$!

# Wait for franka_server to come up
echo "[Step 1] Waiting for franka_server on port ${FRANKA_PORT} ..."
for i in $(seq 1 30); do
    if python -c "import zerorpc; c=zerorpc.Client(heartbeat=5); c.connect('tcp://localhost:${FRANKA_PORT}'); c.close()" 2>/dev/null; then
        echo "[Step 1] franka_server ready ✓"
        break
    fi
    sleep 2
    if [ "$i" -eq 30 ]; then
        echo "[ERROR] franka_server did not start after 60s"
        kill $FRANKA_PID 2>/dev/null || true
        exit 1
    fi
done

# ── Start leapbot_control.py ─────────────────────────────────────────────────
echo ""
echo "[Step 2] Starting leapbot_control.py ..."
cd "$SCRIPT_DIR"

CONTROLLER_ARGS=(
    --robot_ip       "$ROBOT_IP"
    --robot_port     "$FRANKA_PORT"
    --server_ip      "$SERVER_IP"
    --server_port    "$SERVER_PORT"
    --task           "$TASK"
    --frequency      "$FREQUENCY"
    --action_horizon "$ACTION_HORIZON"
    --cam_global     "$CAM_GLOBAL"
    --cam_wrist      "$CAM_WRIST"
    --cam_vis        "$CAM_VIS"
    --max_delta_pos  "$MAX_DELTA_POS"
    --max_delta_rot  "$MAX_DELTA_ROT"
    --safety_x_min   "$SAFETY_X_MIN"
    --safety_x_max   "$SAFETY_X_MAX"
    --safety_y_min   "$SAFETY_Y_MIN"
    --safety_y_max   "$SAFETY_Y_MAX"
    --safety_z_min   "$SAFETY_Z_MIN"
    --safety_z_max   "$SAFETY_Z_MAX"
)
if [[ "$SKIP_HOME" == "1" ]]; then
    CONTROLLER_ARGS+=(--skip_home)
fi

python leapbot_control.py "${CONTROLLER_ARGS[@]}" &
CONTROL_PID=$!

# ── Cleanup on exit ──────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[Cleanup] Stopping controller (PID=$CONTROL_PID) ..."
    kill $CONTROL_PID 2>/dev/null || true
    wait $CONTROL_PID 2>/dev/null || true
    echo "[Cleanup] Stopping franka_server (PID=$FRANKA_PID) ..."
    kill $FRANKA_PID 2>/dev/null || true
    wait $FRANKA_PID 2>/dev/null || true
    echo "[Cleanup] Done."
}
trap cleanup EXIT INT TERM

# ── Wait ─────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  LeapBot is running!"
echo ""
echo "  Keyboard controls (in the controller terminal):"
echo "    S        Start / resume policy"
echo "    Space    Pause / resume toggle"
echo "    B        Stop policy → IDLE"
echo "    N / →    Single-step one action"
echo "    Esc      Emergency stop (+ open gripper)"
echo "    H        Reset to home"
echo "    Z / X    Close / open gripper"
echo "    + / -    Adjust action horizon"
echo "    V        Toggle camera viz"
echo "    Q        Quit gracefully"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

wait $CONTROL_PID
EXIT_CODE=$?
echo ""
echo "Controller exited with code $EXIT_CODE"
exit $EXIT_CODE
