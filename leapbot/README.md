# LeapBot 真机实验指南

本文档指导你在真实 Franka 机器人上运行 LeapBot 策略推理和闭环控制。

---

## 系统架构

```
GPU 机器（本机）                    NUC / 机器人控制主机
conda env: vj_fw                   conda env: polymetis (或 umi)
┌────────────────────────┐        ┌─────────────────────────────────────┐
│  server.py             │◄─HTTP─ │  leapbot_control.py                 │
│  FastAPI + FastWAM     │        │  相机采集 + 安全检查 + 键盘控制     │
│  TensorRT / PyTorch    │        │         │                            │
│  端口: 8000            │        │  ZeroRPC│                            │
└────────────────────────┘        │         ▼                            │
                                  │  franka_server.py                    │
                                  │  polymetis + libfranka               │
                                  │  端口: 4242                          │
                                  │         │                            │
                                  │         ▼                            │
                                  │  Franka Panda 机器人 + 夹爪          │
                                  └─────────────────────────────────────┘

相机布局（连接在 NUC 上）：
  zed_0    → 机器人左侧第三视角   → 默认 global_image
  zed_1    → 机器人正前方视角      → 可视化 / 可配置为 global/wrist
  fisheye  → 夹爪第一视角         → 默认 wrist_image
  l515_0   → 机器人左前方第三视角  → 可视化 / 可配置为 global/wrist（含深度）
```

---

## 1. 环境准备

### 1.1 GPU 机器（本机）— conda env: `vj_fw`

```bash
conda activate vj_fw

# 确认推理环境可用
cd /home/aihub/daojie/LeapBot-inference-only
python -c "from fastwam_infer import FastWAMInference; print('OK')"

# 安装额外依赖（如未安装）
pip install fastapi uvicorn opencv-python requests
```

确认 TensorRT engine 已构建（或使用 PyTorch 后端）。

### 1.2 NUC / 机器人控制主机 — conda env: polymetis（或 umi）

```bash
conda activate polymetis   # 或你的 franka 环境

# 确认依赖
python -c "import zerorpc, cv2, requests, pynput, numpy, scipy"

# 相机相关
python -c "import pyrealsense2"    # L515
python -c "import pyzed.sl"        # ZED（如使用）

# 安装缺失依赖
pip install requests opencv-python pynput
```

### 1.3 网络连通性

```bash
# 从 NUC 测试 GPU 机器
curl http://<GPU机器IP>:8000/health
# 期望: {"status":"ok"}
# 示例，aihub GPU 机器 IP: 10.7.0.131
curl http://10.7.0.131:8000/health
```

---

## 2. 配置

### 2.1 config.py（NUC 侧）

编辑 `franka_standalone/config.py`，确认与硬件一致：

```python
ROBOT_IP = '192.168.3.2'       # NUC IP
ROBOT_PORT = 4242

L515_SERIALS = ['f1480807']
FISHEYE_USB_ID = '32e4:9230'
FISHEYE_RESOLUTION = (640, 480)

ZED_SERIALS = [17791214, 17160981]   # zed_0, zed_1（升序）
ZED_RESOLUTION = 'VGA'
ZED_FPS = 30

EE_HOME_POSE = np.array([0.4, 0.0, 0.3, np.pi, 0.0, 0.0])
FRANKA_HOME_JOINTS = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785])
```

### 2.2 相机 → 模型视图映射（可配置）

模型需要两个图像输入，映射关系通过 CLI 参数指定（**不同实验可使用不同映射**）：

| 物理相机 | 默认用途 | CLI 参数 |
|---------|---------|---------|
| zed_0   | global_image（主第三视角） | `--cam_global zed_0` |
| fisheye | wrist_image（夹爪第一视角） | `--cam_wrist fisheye` |
| zed_1   | 可视化 | `--cam_vis zed_1,l515_0` |
| l515_0  | 可视化（含深度） | `--cam_vis zed_1,l515_0` |

**切换相机示例**：

```bash
# 实验 A：用 zed_0 做全局视角，fisheye 做手腕视角（默认）
--cam_global zed_0 --cam_wrist fisheye

# 实验 B：用 zed_1 做全局视角，l515_0 做手腕视角
--cam_global zed_1 --cam_wrist l515_0

# 实验 C：只用两个 ZED，不用 fisheye 和 L515
--cam_global zed_0 --cam_wrist zed_1 --no_fisheye --no_l515
```

### 2.3 安全限制（工作空间标定）

默认的工作空间边界是保守估计，**实验前必须标定你实际的工作空间**。

**标定方法**：使用 `calibrate_workspace.py` 工具：

```bash
# 先启动 polymetis + franka_server.py
conda activate polymetis
cd /path/to/franka_standalone/leapbot
python calibrate_workspace.py --robot_ip localhost --margin 0.02
```

标定流程：
1. 机器人自动回到 Home 位
2. 用 WASD/QE 键手动移动机器人到工作空间的**最远允许位置**：
   - W/S：推到最远前方 (x_max) 和最后方 (x_min)
   - A/D：推到最左 (y_max) 和最右 (y_min)
   - Q/E：推到最高 (z_max) 和最低 (z_min)
3. 在每个极限位置停留约 1 秒
4. 按 **T** 预览当前记录的边界
5. 按 **P** 输出最终边界（可直接复制粘贴的代码）
6. 按 **H** 随时归位，**Esc** 退出

工具会自动输出如下格式，直接粘贴到 `FrankaPoseSafetyChecker` 或通过 CLI 传入：

```python
# calibrate_workspace.py 输出示例：
    x_min=0.2500, x_max=0.6200,
    y_min=-0.3500, y_max=0.3200,
    z_min=0.0800, z_max=0.5000,
```

```bash
# 或通过 CLI 传入标定结果
python leapbot_control.py --server_ip <IP> \
    --safety_x_min 0.25 --safety_x_max 0.62 \
    --safety_y_min -0.35 --safety_y_max 0.32 \
    --safety_z_min 0.08 --safety_z_max 0.50
```

**标定参数说明**：
- `margin`（默认 0.02m）：在你到达的极限位置基础上再往内缩 2cm，避免在物理极限处运行
- 标定时按住 **Shift** 可以 3x 速度移动，加快标定
- 建议标定 2-3 次取交集，确保边界稳定

---

## 3. 启动步骤

### 步骤 1：启动 Franka 底层服务（NUC 终端 1）

```bash
conda activate polymetis
cd /path/to/franka_standalone
bash launch_polymetis.sh     # 保持运行
```

### 步骤 2：启动 GPU 推理服务器（GPU 机器终端）

```bash
conda activate vj_fw
cd /home/aihub/daojie/franka_standalone/leapbot

# 默认配置
bash launch_server.sh

# 自定义配置
ASSET_ROOT=/path/to/assets \
TASK=move_objects_into_box \
bash launch_server.sh
```

等待看到：

```
[Server] Model loaded successfully!
[Server] Starting uvicorn on 0.0.0.0:8000
```

验证：

```bash
curl http://localhost:8000/ready
```

### 步骤 3：启动机器人闭环控制（NUC 终端 2）

```bash
conda activate polymetis
cd /path/to/franka_standalone/leapbot

# 方式 A：使用启动脚本（自动启动 franka_server.py）
SERVER_IP=<GPU机器IP> bash launch_leapbot.sh

# 方式 B：手动启动（如 franka_server.py 已在运行）
python leapbot_control.py \
    --server_ip <GPU机器IP> \
    --task move_objects_into_box
```

启动后进入 **IDLE** 状态，显示所有可用键盘控制：

```
═══════════════════════════════════════════════════════════════════
  Keyboard controls:
    S        Start / resume policy
    Space    Pause / resume toggle
    B        Stop policy → IDLE
    N / →    Single-step one action
    Esc      Emergency stop (+ open gripper)
    H        Reset to home
    Z / X    Close / open gripper
    + / -    Adjust action horizon
    V        Toggle camera viz
    M        Print current state
    Q        Quit gracefully
═══════════════════════════════════════════════════════════════════

  State: IDLE  —  press S to start
```

---

## 4. 键盘控制详解

### 策略控制

| 按键 | 功能 | 可用状态 | 说明 |
|------|------|---------|------|
| **S** | 启动 / 恢复策略 | IDLE, PAUSED | 进入 AUTO 模式，开始闭环推理 |
| **Space** | 暂停 / 恢复 | AUTO ↔ PAUSED | 立即暂停/恢复，保持当前位姿 |
| **B** | 停止策略 | AUTO, PAUSED | 返回 IDLE，机器人保持当前位置 |
| **N** 或 **→** | 单步执行 | IDLE, PAUSED | 执行一次推理的一个动作，然后自动暂停 |
| **Esc** | 急停 | 任何状态 | **立即**停止策略 + 打开夹爪，进入 STOPPED |
| **H** | 归位 | IDLE, PAUSED, STOPPED | 关节空间回到 Home 位姿 |

### 夹爪控制

| 按键 | 功能 | 说明 |
|------|------|------|
| **Z** | 闭合夹爪 | 手动二值闭合（不影响策略状态） |
| **X** | 张开夹爪 | 手动二值张开（不影响策略状态） |

### 调节控制

| 按键 | 功能 | 说明 |
|------|------|------|
| **+** | 增大 action_horizon | 推理频率降低，GPU 负载降低 |
| **-** | 减小 action_horizon | 推理频率升高，策略更灵活（最小为 1） |
| **V** | 切换可视化 | 开/关相机实时预览窗口 |
| **M** | 打印状态 | 在终端显示当前位姿、夹爪、状态 |
| **Q** | 退出 | 安全退出，自动清理 |

### 状态机

```
                S
    IDLE ─────────────► AUTO
     ▲                   │  ▲
     │ B                 │  │ S / N / →
     │                   ▼  │
    STOPPED ◄────── PAUSED ─┘
     │  ▲            ▲
     │  │ Esc        │ Esc
     └──┘            │
                      │
    任何状态 ──Esc──► STOPPED
    IDLE/PAUSED/STOPPED ──H──► IDLE
```

| 状态 | 含义 | 策略 | 机器人 |
|------|------|------|--------|
| **IDLE** | 等待用户启动 | 不推理 | 阻抗控制，保持位姿 |
| **AUTO** | 策略运行中 | 持续推理执行 | 按策略输出运动 |
| **PAUSED** | 策略暂停 | 不推理 | 阻抗控制，保持位姿 |
| **STOPPED** | 急停 | 不推理 | 阻抗控制，夹爪已打开 |

---

## 5. 推荐实验流程

### 首次实验（保守模式）

```bash
# 启动时用小 horizon 和小 delta 限制
SERVER_IP=192.168.1.100 \
ACTION_HORIZON=1 \
MAX_DELTA_POS=0.03 \
MAX_DELTA_ROT=0.15 \
bash launch_leapbot.sh
```

1. 系统启动后进入 **IDLE**，确认相机画面正常
2. 按 **M** 查看当前位姿，确认在安全范围内
3. 按 **S** 启动策略，观察机器人动作
4. **随时准备按 Esc** 急停
5. 如果动作合理，按 **Space** 暂停，用 **+** 增大 horizon
6. 按 **S** 继续，逐步放宽限制

### 标准实验

```bash
SERVER_IP=192.168.1.100 \
TASK=move_objects_into_box \
ACTION_HORIZON=4 \
bash launch_leapbot.sh
```

1. 按 **H** 归位
2. 按 **S** 启动策略
3. 观察机器人执行任务
4. 按 **Space** 暂停检查，**S** 继续
5. 按 **B** 停止策略（保持位姿），**Q** 退出

### 切换相机配置

```bash
# 实验 A
SERVER_IP=192.168.1.100 \
CAM_GLOBAL=zed_0 \
CAM_WRIST=fisheye \
bash launch_leapbot.sh

# 实验 B（不同视角组合）
SERVER_IP=192.168.1.100 \
CAM_GLOBAL=zed_1 \
CAM_WRIST=l515_0 \
bash launch_leapbot.sh
```

---

## 6. 启动参数速查

### GPU 服务器（launch_server.sh / server.py）

| 环境变量 / 参数 | 默认值 | 说明 |
|----------------|--------|------|
| `CONDA_ENV` | vj_fw | conda 环境名 |
| `ASSET_ROOT` | /home/aihub/.../assets | FastWAM 资产目录 |
| `TASK` | move_objects_into_box | 任务 ID |
| `INFER_ROOT` | /home/aihub/.../LeapBot-inference-only | 推理包路径 |
| `DEVICE` | cuda:0 | 推理设备 |
| `BACKEND` | tensorrt | 推理后端 |
| `PORT` | 8000 | HTTP 端口 |
| `ENABLE_FISHEYE` | 0 | 鱼眼去畸变（1=开启）|

### 机器人控制器（launch_leapbot.sh / leapbot_control.py）

| 环境变量 / 参数 | 默认值 | 说明 |
|----------------|--------|------|
| `CONDA_ENV` | polymetis | conda 环境名 |
| `SERVER_IP` | **必填** | GPU 机器 IP |
| `SERVER_PORT` | 8000 | GPU 推理端口 |
| `ROBOT_IP` | localhost | franka_server.py 地址 |
| `FRANKA_PORT` | 4242 | ZeroRPC 端口 |
| `TASK` | move_objects_into_box | 任务 ID |
| `FREQUENCY` | 10 | 控制频率 (Hz) |
| `ACTION_HORIZON` | 4 | 每次推理执行步数（运行时可 +/-）|
| `CAM_GLOBAL` | zed_0 | global_image 相机名 |
| `CAM_WRIST` | fisheye | wrist_image 相机名 |
| `CAM_VIS` | zed_1,l515_0 | 可视化额外相机 |
| `MAX_DELTA_POS` | 0.08 | 单步位移限制 (m) |
| `MAX_DELTA_ROT` | 0.3 | 单步旋转限制 (rad) |
| `SKIP_HOME` | 0 | 跳过归零（1=跳过）|

---

## 7. 切换任务

```bash
# GPU 服务器：重启并指定新任务
TASK=press_three_buttons bash launch_server.sh

# NUC 控制器：使用相同 TASK
SERVER_IP=<IP> TASK=press_three_buttons bash launch_leapbot.sh
```

---

## 8. 推理 API（自定义控制脚本）

```
POST http://<GPU_IP>:8000/infer
{
    "global_image": "<base64 JPEG>",
    "wrist_image":  "<base64 JPEG>",
    "proprio_7d":   [x,y,z,rx,ry,rz,gripper_width],
    "task":         "move_objects_into_box"
}

Response:
{
    "action_chunk": [[dx,dy,dz,drx,dry,drz,grip], ...],  // [32,7]
    "latency_ms":   45.2
}
```

Python 客户端：

```python
from leapbot_client import LeapbotClient

client = LeapbotClient("192.168.1.100", 8000)
result = client.infer(global_rgb, wrist_rgb, proprio_7d, "move_objects_into_box")
print(result["action_chunk"].shape)  # (32, 7)
```

---

## 9. 故障排查

| 问题 | 解决方案 |
|------|---------|
| `GPU server not ready` | 确认 `server.py` 已启动；`curl http://<IP>:8000/ready` |
| `Missing camera frame(s)` | 检查相机连接；用 `--no_zed` / `--no_fisheye` 禁用；确认 `--cam_global` / `--cam_wrist` 名称正确 |
| `Inference failed` | GPU 服务器日志；确认 `--task` 匹配 |
| `SAFETY: blocked` | 调整 `--max_delta_pos` / `--max_delta_rot`；检查工作空间边界 |
| `polymetis not ready` | 先启动 `launch_polymetis.sh` |
| 动作抖动 | 减小 `--action_horizon`（按 `-` 键） |
| 推理太慢 | 确认 `--backend tensorrt`；检查 GPU 使用率 |
| `pyzed` 导入失败 | 确认在 umi/polymetis 环境中 |
| 键盘无响应 | 确认焦点在终端窗口上；检查 pynput 是否有 X11 权限 |

---

## 10. 文件清单

```
franka_standalone/leapbot/
├── README.md               # 本文档
├── server.py               # GPU 推理服务器（FastAPI）
├── launch_server.sh        # GPU 服务器启动脚本（conda vj_fw）
├── leapbot_client.py       # NUC 侧推理 HTTP 客户端
├── leapbot_control.py      # NUC 侧闭环控制主脚本（完整键盘控制）
└── launch_leapbot.sh       # NUC 侧启动脚本（franka_server + 控制器）
```

---

## 11. 安全注意事项

**启动闭环控制前务必确认：**

1. **工作空间边界**：根据实际桌面调整 `FrankaPoseSafetyChecker`
2. **首次实验**：使用 `ACTION_HORIZON=1` + `MAX_DELTA_POS=0.03`
3. **急停准备**：手放在急停按钮附近，**Esc** 键可立即停止策略并打开夹爪
4. **单步验证**：先用 **N** 单步执行，确认动作合理后再用 **S** 连续运行
5. **遮挡物**：确认运动范围内无障碍物和线缆
6. **网络**：使用有线局域网，避免 WiFi 延迟抖动
