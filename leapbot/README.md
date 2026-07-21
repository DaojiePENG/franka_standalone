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

**已预置的配置文件**（按任务区分相机）：

| 配置文件 | 任务 | global_image | wrist_image |
|---------|------|-------------|-------------|
| `config_move_objects_into_box.py` | move_objects_into_box | zed_0（左侧） | fisheye |
| `config_press_three_buttons.py` | press_three_buttons | zed_1（正前方） | fisheye |

```bash
# 直接使用预置配置
python leapbot_control.py --config config_move_objects_into_box.py
python leapbot_control.py --config config_press_three_buttons.py

# CLI 临时覆盖
--cam_global zed_1 --cam_wrist l515_0
--cam_global zed_0 --cam_wrist zed_1 --no_fisheye --no_l515
```

### 2.3 配置文件

所有可配置参数集中在 `leapbot/leapbot_config.py` 中，修改后直接启动即可，无需加命令行参数。

```bash
# 直接编辑
vim leapbot_config.py

# 关键参数：
SERVER_IP   = "192.168.1.100"   # ← 必须改为你的 GPU 机器 IP
TASK        = "move_objects_into_box"
CAM_GLOBAL  = "zed_0"
CAM_WRIST   = "fisheye"
SAFETY_X_MIN = 0.25             # ← 来自标定工具输出
SAFETY_X_MAX = 0.62             # ← 来自标定工具输出
...
```

**优先级**：`CLI 参数 > leapbot_config.py > 硬编码默认值`

可用时指定其他配置文件：

```bash
python leapbot_control.py --config ./my_experiment_config.py
```

### 2.4 工作空间标定

默认的工作空间边界是保守估计，**首次实验前必须标定**。

```bash
conda activate polymetis
cd /path/to/franka_standalone/leapbot
python calibrate_workspace.py --robot_ip localhost --margin 0.02
```

流程：
1. 机器人自动归零，用 **WASD/QE** 推到每个极限位置（W=前方, S=后方, A=左, D=右, Q=上, E=下）
2. 每个极限位置**停留 1 秒**
3. 按 **T** 预览边界，按 **P** 输出最终结果
4. 将输出值填入 `leapbot_config.py` 的 `SAFETY_*` 字段

---

## 3. 启动步骤（快速参考）

完整的端到端实验流程见 [第 5 节](#5-实验流程)，以下是各组件的单独启动说明。

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
bash launch_server.sh        # 可通过环境变量自定义，见第 6 节
```

验证：`curl http://localhost:8000/ready`

### 步骤 3：启动机器人闭环控制（NUC 终端 2）

```bash
conda activate polymetis
cd /path/to/franka_standalone/leapbot
SERVER_IP=<GPU机器IP> bash launch_leapbot.sh
```

启动后进入 **IDLE** 状态，显示键盘控制说明（见第 4 节）。

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

## 5. 实验流程

完整实验流程按顺序分为 5 个阶段。**首次使用必须从阶段 1 开始。**

---

### 阶段 1：工作空间标定（仅首次 / 每次调整工位后）

**前置条件**：polymetis + franka_server.py 已启动（步骤 1）

```bash
conda activate polymetis
cd /path/to/franka_standalone/leapbot
python calibrate_workspace.py --robot_ip localhost --margin 0.02
```

操作流程：

1. 机器人自动归零到 Home 位
2. 用 **WASD/QE** 手动推到工作空间的每个极限位置：
   - **W** 推到最远前方 → 记录 x_max
   - **S** 推到最后方 → 记录 x_min
   - **A** 推到最左 → 记录 y_max
   - **D** 推到最右 → 记录 y_min
   - **Q** 推到最高 → 记录 z_max
   - **E** 推到最低 → 记录 z_min
3. 每个极限位置**停留 1 秒**，让跟踪器记录
4. 按 **T** 预览当前边界，按 **R** 可重置重来
5. 全部标定完毕后按 **P** 输出结果：

```
============================================================
    x_min=0.2500, x_max=0.6200,
    y_min=-0.3500, y_max=0.3200,
    z_min=0.0800, z_max=0.5000,
============================================================
```

6. 按 **Esc** 退出标定工具

**记录这些值**，后续启动控制器时通过环境变量传入（见阶段 3）。

> `--margin 0.02` 会在你到达的极限位置基础上再往内缩 2cm。
> 建议标定 2-3 次取交集，确保边界稳定。

---

### 阶段 2：启动 GPU 推理服务器

在 **GPU 机器**上（只需启动一次，切换任务时重启）：

```bash
conda activate vj_fw
cd /home/aihub/daojie/franka_standalone/leapbot
bash launch_server.sh
```

等待看到：

```
[Server] Model loaded successfully!
[Server] Starting uvicorn on 0.0.0.0:8000
```

---

### 阶段 3：启动闭环控制器（首次实验 — 保守模式）

首次实验建议先临时修改 `leapbot_config.py` 使用保守参数：

```python
# leapbot_config.py — 首次实验临时改为：
ACTION_HORIZON  = 1      # 纯 receding horizon，最灵活
MAX_DELTA_POS   = 0.03   # 小步长限制
MAX_DELTA_ROT   = 0.15
# SAFETY_* 填入阶段 1 标定结果
```

然后启动：

```bash
conda activate polymetis
cd /path/to/franka_standalone/leapbot
SERVER_IP=<GPU机器IP> bash launch_leapbot.sh
```

确认策略行为稳定后，改回正常参数（`ACTION_HORIZON=4`，放宽 delta 限制）。

---

### 阶段 4：验证与调试

系统启动后进入 **IDLE** 状态，按以下顺序验证：

1. **检查相机画面** — 确认 4 个相机图像正常显示
2. **按 M** — 查看当前末端位姿，确认在标定的安全范围内
3. **按 N（单步）** — 执行一个动作，观察机器人是否按预期运动
4. 单步合理后，**按 S** — 进入连续运行模式
5. **随时准备按 Esc** — 急停（立即停止 + 打开夹爪）
6. 动作稳定后按 **Space** 暂停，用 **+** 增大 horizon，再按 **S** 继续

---

### 阶段 5：正式实验

确认策略行为合理后，恢复 `leapbot_config.py` 中的正常参数，然后启动：

```bash
SERVER_IP=<GPU机器IP> bash launch_leapbot.sh
```

或指定不同的任务/配置文件：

```bash
SERVER_IP=<GPU机器IP> TASK=press_three_buttons bash launch_leapbot.sh
CONFIG_FILE=./exp_config_b.py SERVER_IP=<GPU机器IP> bash launch_leapbot.sh
```

1. 按 **H** 归位
2. 按 **S** 启动策略
3. 观察机器人执行任务
4. 按 **Space** 暂停检查，**S** 继续
5. 任务完成或需要停止时，按 **B** 停止策略（保持位姿），**Q** 退出

---

### 切换相机配置

不同实验需要不同视角时，准备多个配置文件或用 CLI 覆盖：

```bash
# 方式 A：为不同实验准备不同配置文件
cp leapbot_config.py exp_A_config.py   # 编辑 CAM_GLOBAL, CAM_WRIST 等
cp leapbot_config.py exp_B_config.py
CONFIG_FILE=exp_A_config.py SERVER_IP=<IP> bash launch_leapbot.sh

# 方式 B：CLI 临时覆盖
SERVER_IP=<IP> --cam_global zed_1 --cam_wrist l515_0 bash launch_leapbot.sh
```

### Mock 推理测试（不连接机器人）

在实际控制机器人之前，用 Mock 模式验证推理是否正常工作：

```bash
# 方式 A：CLI 传入 --mock
python leapbot_control.py --config config_move_objects_into_box.py --mock

# 方式 B：在配置文件中设置 MOCK = True
# leapbot_config.py:
MOCK = True
python leapbot_control.py
```

Mock 模式行为：
- ✅ 采集所有相机画面（与真实模式完全一致）
- ✅ 连接 GPU 推理服务器，发送推理请求
- ✅ 可视化相机画面窗口
- ❌ **不连接** Franka 机器人
- ❌ **不执行** 任何动作（只打印预测结果）

按 **S** 启动策略后，终端输出每次推理的预测动作：

```
[Mock] Inference 52.3ms, chunk shape=(32, 7), showing 4 steps:
       step      dx      dy      dz     drx     dry     drz   grip  target_x target_y target_z
          0  +0.0012 -0.0003 +0.0008 +0.0010 -0.0005 +0.0002  0.080  +0.4012 -0.0003 +0.3008
          1  +0.0015 -0.0002 +0.0010 +0.0008 -0.0004 +0.0003  0.080  +0.4027 -0.0005 +0.3018
          2  +0.0018 -0.0001 +0.0012 +0.0006 -0.0003 +0.0004  0.080  +0.4045 -0.0006 +0.3030
          3  +0.0020 +0.0000 +0.0015 +0.0005 -0.0002 +0.0005  0.080  +0.4065 -0.0006 +0.3045
```

检查要点：
1. **相机画面**：global_image 和 wrist_image 是否正确对应
2. **推理延迟**：infer 时间是否合理（<100ms 为佳）
3. **动作幅度**：dx/dy/dz 是否在合理范围（不应出现 >0.05 的突变）
4. **动作方向**：delta 方向是否符合预期（如抓取任务应向下移动）
5. **安全检查**：UNSAFE 标记表示动作超出安全边界，需要调整
6. **夹爪值**：grip 列是否在 0~0.08 范围内

验证合理后再进入阶段 1（标定）和后续的真实机器人实验。

---

## 6. 配置参数速查

### GPU 服务器（launch_server.sh / server.py）

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `CONDA_ENV` | vj_fw | conda 环境名 |
| `ASSET_ROOT` | /home/aihub/.../assets | FastWAM 资产目录 |
| `TASK` | move_objects_into_box | 任务 ID |
| `INFER_ROOT` | /home/aihub/.../LeapBot-inference-only | 推理包路径 |
| `DEVICE` | cuda:0 | 推理设备 |
| `BACKEND` | tensorrt | 推理后端 |
| `PORT` | 8000 | HTTP 端口 |
| `ENABLE_FISHEYE` | 0 | 鱼眼去畸变（1=开启）|

### 机器人控制器（leapbot_config.py + launch_leapbot.sh）

**主要配置**：编辑 `leapbot_config.py`，参数及默认值见该文件内注释。

**launch_leapbot.sh 环境变量**（仅用于临时覆盖）：

| 环境变量 | 说明 |
|---------|------|
| `CONFIG_FILE` | 指定其他配置文件（默认 ./leapbot_config.py）|
| `SERVER_IP` | 覆盖 config 中的 GPU 服务器 IP |
| `TASK` | 覆盖 config 中的任务 ID |
| `CONDA_ENV` | NUC 侧 conda 环境名（默认 polymetis）|

**CLI 覆盖**（优先级最高，直接传给 `leapbot_control.py`）：

```bash
# 任何 leapbot_config.py 中的参数都可以用 --参数名 覆盖
python leapbot_control.py --server_ip 1.2.3.4 --action_horizon 1
python leapbot_control.py --config ./other.py --cam_global zed_1
```

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
├── README.md                             # 本文档
├── leapbot_config.py                     # 默认控制器配置
├── config_move_objects_into_box.py       # 任务配置：zed_0 + fisheye
├── config_press_three_buttons.py         # 任务配置：zed_1 + fisheye
├── server.py                             # GPU 推理服务器（FastAPI）
├── launch_server.sh                      # GPU 服务器启动脚本（conda vj_fw）
├── leapbot_client.py                     # NUC 侧推理 HTTP 客户端
├── leapbot_control.py                    # NUC 侧闭环控制（支持 --mock）
├── launch_leapbot.sh                     # NUC 侧启动脚本
└── calibrate_workspace.py                # 工作空间标定工具
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
