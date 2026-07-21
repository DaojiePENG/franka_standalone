# ──────────────────────────────────────────────────────────────────────────────
# LeapBot 配置 — press_three_buttons
#
# 相机：zed_1（正前方第三视角）+ fisheye（夹爪第一视角）
# 使用：
#   python leapbot_control.py --config config_press_three_buttons.py
#   CONFIG_FILE=config_press_three_buttons.py bash launch_leapbot.sh
# ──────────────────────────────────────────────────────────────────────────────

# 网络
SERVER_IP   = "10.7.0.131"
SERVER_PORT = 8000
ROBOT_IP    = "192.168.3.2"
ROBOT_PORT  = 4242
TIMEOUT     = 5.0

# 任务
TASK            = "press_three_buttons"
FREQUENCY       = 10
ACTION_HORIZON  = 4

# 相机映射（正前方视角 + 夹爪第一视角）
CAM_GLOBAL = "zed_1"            # 正前方第三视角 → global_image
CAM_WRIST  = "fisheye"          # 夹爪第一视角   → wrist_image
CAM_VIS    = "zed_0,l515_0"     # 可视化

NO_L515    = False
NO_FISHEYE = False
NO_ZED     = False

# 安全限制
MAX_DELTA_POS = 0.08
MAX_DELTA_ROT = 0.3

# 工作空间边界（标定后填入）
SAFETY_X_MIN = 0.25
SAFETY_X_MAX = 0.70
SAFETY_Y_MIN = -0.30
SAFETY_Y_MAX = 0.30
SAFETY_Z_MIN = 0.03
SAFETY_Z_MAX = 0.40

# 启动行为
SKIP_HOME = False
