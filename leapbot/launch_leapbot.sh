#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# LeapBot NUC-side launcher
#
# 配置优先级：环境变量 > leapbot_config.py > 脚本内置默认值
#
# 大部分参数在 leapbot_config.py 中配置即可，以下环境变量可临时覆盖：
#   CONFIG_FILE    — 配置文件路径（默认 ./leapbot_config.py）
#   SERVER_IP      — GPU 推理服务器 IP（覆盖 config 中的 SERVER_IP）
#   TASK           — 任务 ID（覆盖 config 中的 TASK）
#   CONDA_ENV      — NUC 侧 conda 环境名（默认 polymetis）
#
# 使用方式：
#   SERVER_IP=192.168.1.100 bash launch_leapbot.sh                    # 最简
#   CONFIG_FILE=./my_exp_config.py bash launch_leapbot.sh             # 指定配置
#   SERVER_IP=1.2.3.4 TASK=press_three_buttons bash launch_leapbot.sh # 临时覆盖
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

# ── Franka server settings ────────────────────────────────────────────────────
ROBOT_IP="${ROBOT_IP:-localhost}"
GRIPPER_ROBOT_IP="${GRIPPER_ROBOT_IP:-10.168.1.200}"
FRANKA_PORT="${FRANKA_PORT:-4242}"

# ── Config file ───────────────────────────────────────────────────────────────
CONFIG_FILE="${CONFIG_FILE:-}"

# ── Build CLI overrides (only set if env var is non-empty) ────────────────────
OVERRIDES=()
[[ -n "${SERVER_IP:-}" ]]        && OVERRIDES+=(--server_ip "$SERVER_IP")
[[ -n "${TASK:-}" ]]             && OVERRIDES+=(--task "$TASK")

# ── Info ──────────────────────────────────────────────────────────────────────
echo "================================================================="
echo "  LeapBot NUC Launcher"
echo "  CONDA ENV    : ${CONDA_ENV}"
echo "  Franka server: ${ROBOT_IP}:${FRANKA_PORT}"
echo "  Config file  : ${CONFIG_FILE:-<default leapbot_config.py>}"
[[ ${#OVERRIDES[@]} -gt 0 ]] && echo "  CLI overrides: ${OVERRIDES[*]}"
echo "================================================================="
echo ""

# ── Start franka_server.py ────────────────────────────────────────────────────
echo "[Step 1] Starting franka_server.py ..."
cd "$FRANKA_DIR"
python franka_server.py \
    --robot_ip         "$ROBOT_IP" \
    --gripper_robot_ip "$GRIPPER_ROBOT_IP" \
    --port             "$FRANKA_PORT" \
    &
FRANKA_PID=$!

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
