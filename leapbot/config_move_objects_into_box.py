# ──────────────────────────────────────────────────────────────────────────────
# LeapBot 配置 — move_objects_into_box
#
# 相机：zed_0（左侧第三视角）+ fisheye（夹爪第一视角）
# 使用：
#   python leapbot_control.py --config config_move_objects_into_box.py
#   CONFIG_FILE=config_move_objects_into_box.py bash launch_leapbot.sh
# ──────────────────────────────────────────────────────────────────────────────

# 网络
SERVER_IP   = "10.7.0.131"
SERVER_PORT = 8000
ROBOT_IP    = "localhost"
ROBOT_PORT  = 4242
TIMEOUT     = 5.0

# 任务
TASK            = "move_objects_into_box"
FREQUENCY       = 10
ACTION_HORIZON  = 4

# 相机映射
CAM_GLOBAL = "zed_0"            # 左侧第三视角 → global_image
CAM_WRIST  = "fisheye"          # 夹爪第一视角 → wrist_image
CAM_VIS    = "zed_1,l515_0"     # 可视化

NO_L515    = False
NO_FISHEYE = False
NO_ZED     = False

# 安全限制
MAX_DELTA_POS = 0.08
MAX_DELTA_ROT = 0.3

# 工作空间边界（标定后填入）
SAFETY_X_MIN = 0.20
SAFETY_X_MAX = 0.70
SAFETY_Y_MIN = -0.40
SAFETY_Y_MAX = 0.40
SAFETY_Z_MIN = 0.05
SAFETY_Z_MAX = 0.60

# 启动行为
SKIP_HOME = False
