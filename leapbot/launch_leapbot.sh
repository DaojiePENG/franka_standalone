#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# LeapBot launcher — connects to an already-running franka_server.py
#
# franka_server.py must be running on the NUC beforehand (started via
# polymetis / launch_robot.py or equivalent).  This script only launches
# leapbot_control.py which connects to the Franka server via ZeroRPC.
#
# 配置优先级：环境变量 > leapbot_config.py > 脚本内置默认值
#
# 环境变量可临时覆盖：
#   CONFIG_FILE    — 配置文件路径（默认 ./leapbot_config.py）
#   SERVER_IP      — GPU 推理服务器 IP（覆盖 config 中的 SERVER_IP）
#   TASK           — 任务 ID（覆盖 config 中的 TASK）
#   CONDA_ENV      — conda 环境名（默认 umi）
#
# 使用方式：
#   bash launch_leapbot.sh                                           # 最简
#   CONFIG_FILE=config_press_three_buttons.py bash launch_leapbot.sh # 指定配置
#   SERVER_IP=1.2.3.4 TASK=press_three_buttons bash launch_leapbot.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Activate conda environment ────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-umi}"
_CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
if [[ -f "${_CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    source "${_CONDA_BASE}/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
    source "/opt/conda/etc/profile.d/conda.sh"
fi
set +u  # conda activate.d scripts reference unset vars (e.g. MKL_INTERFACE_LAYER)
conda activate "$CONDA_ENV"
set -u

# ── Franka server settings ────────────────────────────────────────────────────
ROBOT_IP="${ROBOT_IP:-192.168.3.2}"
FRANKA_PORT="${FRANKA_PORT:-4242}"

# ── Config file ───────────────────────────────────────────────────────────────
CONFIG_FILE="${CONFIG_FILE:-}"

# ── Build CLI overrides (only set if env var is non-empty) ────────────────────
OVERRIDES=()
[[ -n "${SERVER_IP:-}" ]]        && OVERRIDES+=(--server_ip "$SERVER_IP")
[[ -n "${TASK:-}" ]]             && OVERRIDES+=(--task "$TASK")

# ── Info ──────────────────────────────────────────────────────────────────────
echo "================================================================="
echo "  LeapBot Launcher"
echo "  CONDA ENV    : ${CONDA_ENV}"
echo "  Franka server: ${ROBOT_IP}:${FRANKA_PORT}"
echo "  Config file  : ${CONFIG_FILE:-<default leapbot_config.py>}"
[[ ${#OVERRIDES[@]} -gt 0 ]] && echo "  CLI overrides: ${OVERRIDES[*]}"
echo "================================================================="
echo ""

# ── Pre-flight: check franka_server is reachable ──────────────────────────────
echo "[Pre-flight] Checking franka_server on ${ROBOT_IP}:${FRANKA_PORT} ..."
if ! python -c "
import zerorpc, sys
c = zerorpc.Client(timeout=5, heartbeat=5)
try:
    c.connect('tcp://${ROBOT_IP}:${FRANKA_PORT}')
    c.close()
except Exception as e:
    print(f'[ERROR] Cannot reach franka_server: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1; then
    echo "[ERROR] franka_server not reachable at ${ROBOT_IP}:${FRANKA_PORT}."
    echo "        Start franka_server.py on the NUC first."
    exit 1
fi
echo "[Pre-flight] franka_server reachable ✓"

# ── Start leapbot_control.py ─────────────────────────────────────────────────
echo ""
echo "[Step 1] Starting leapbot_control.py ..."
cd "$SCRIPT_DIR"

CONTROLLER_ARGS=(
    --robot_ip   "$ROBOT_IP"
    --robot_port "$FRANKA_PORT"
)
[[ -n "$CONFIG_FILE" ]] && CONTROLLER_ARGS+=(--config "$CONFIG_FILE")
CONTROLLER_ARGS+=("${OVERRIDES[@]}")

python leapbot_control.py "${CONTROLLER_ARGS[@]}" &
CONTROL_PID=$!

# ── Cleanup ───────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "[Cleanup] Stopping controller (PID=$CONTROL_PID) ..."
    kill $CONTROL_PID 2>/dev/null || true
    wait $CONTROL_PID 2>/dev/null || true
    echo "[Cleanup] Done."
}
trap cleanup EXIT INT TERM

# ── Wait ─────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "  LeapBot is running!"
echo "  Keyboard controls:"
echo "    S=Start  Space=Pause  B=Stop  N=Step  Esc=E-stop"
echo "    H=Home  Z/X=Gripper  +/-=Horizon  V=Viz  Q=Quit"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

wait $CONTROL_PID
EXIT_CODE=$?
echo ""
echo "Controller exited with code $EXIT_CODE"
exit $EXIT_CODE
