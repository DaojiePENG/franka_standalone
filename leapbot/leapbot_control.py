#!/usr/bin/env python3
"""
LeapBot Real Robot Controller — runs on the NUC / robot control host.

Full closed-loop script with interactive keyboard control, safety mechanisms,
and configurable camera-to-model-view mapping.

Keyboard Controls:
    S       Start / resume policy execution
    Space   Pause / resume  (toggle)
    B       Stop policy, return to IDLE (keeps current pose)
    N / →   Single-step one action then auto-pause
    Esc     Emergency stop: immediately stop policy, open gripper
    H       Reset to home position (only when paused/idle)
    Z       Close gripper (manual, binary)
    X       Open  gripper (manual, binary)
    +/-     Increase/decrease action horizon by 1
    V       Toggle camera visualization
    M       Print current state to terminal
    Q       Quit gracefully (terminate controller, close cameras)

Policy States:
    IDLE    → S → AUTO
    AUTO    → Space → PAUSED  |  B → IDLE  |  Esc → STOPPED
    PAUSED  → S/N → AUTO     |  B → IDLE  |  Esc → STOPPED
    STOPPED → H → IDLE       |  Q → exit

Camera → model view mapping is configurable via CLI flags:
    --cam_global zed_0        # which camera is global_image
    --cam_wrist  fisheye      # which camera is wrist_image
    --cam_vis zed_1,l515_0    # extra cameras for the visualization grid

Requires:
    - franka_server.py running on this machine (ZeroRPC, port 4242)
    - leapbot server.py running on the GPU machine (HTTP, port 8000)

Usage:
    python leapbot_control.py --server_ip 192.168.1.100
    python leapbot_control.py --server_ip 192.168.1.100 --task press_three_buttons
    python leapbot_control.py --server_ip 192.168.1.100 --cam_global zed_1 --cam_wrist l515_0
"""
import argparse
import enum
import importlib.util
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.spatial.transform as st
import zerorpc

# ── Import from franka_standalone ─────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    ROBOT_IP, ROBOT_PORT, CONTROL_FREQUENCY,
    MAX_GRIPPER_WIDTH, GRIPPER_SPEED,
    EE_HOME_POSE, HOME_MOVE_DURATION,
    FRANKA_HOME_JOINTS, JOINTS_HOME_DURATION,
    KX_DEFAULT, KXD_DEFAULT,
    TX_FLANGE_TIP, TX_TIP_FLANGE,
    L515_SERIALS, FISHEYE_USB_ID, FISHEYE_RESOLUTION,
    ZED_SERIALS, ZED_RESOLUTION, ZED_FPS,
)
from franka_collect import (
    L515Camera, FisheyeCamera, ZEDCamera,
    find_video_device_by_usb_id, detect_zed_serials,
    pose_to_mat, mat_to_pose, precise_wait,
)
from leapbot_client import LeapbotClient


# ═══════════════════════════════════════════════════════════════════════════════
#  Policy State Machine
# ═══════════════════════════════════════════════════════════════════════════════

class PolicyState(enum.Enum):
    IDLE    = "IDLE"       # not running, waiting for user to start
    AUTO    = "AUTO"       # policy running continuously
    PAUSED  = "PAUSED"     # policy paused, can resume / step / stop
    STOPPED = "STOPPED"    # emergency-stopped, needs home reset


# ═══════════════════════════════════════════════════════════════════════════════
#  Keyboard Controller (pynput, non-blocking)
# ═══════════════════════════════════════════════════════════════════════════════

class PolicyKeyboardController:
    """Non-blocking keyboard handler using pynput.

    All flags are consumed (reset to False) by the main loop after reading,
    EXCEPT ``shift_held`` which is a live mirror of the physical Shift key.
    Gripper Z/X uses the same debounce logic as franka_collect to handle
    X11 auto-repeat correctly.
    """

    def __init__(self):
        self._pk = None
        self._listener = None

        # ── Policy control (one-shot flags) ───────────────────────────────────
        self.policy_start   = False   # S: start / resume
        self.policy_stop    = False   # B: stop → IDLE
        self.pause_toggle   = False   # Space: toggle pause
        self.single_step    = False   # N / → : one action then auto-pause
        self.emergency_stop = False   # Esc: immediate stop + open gripper
        self.home_requested = False   # H: reset to home

        # ── Gripper (hold + debounce, same as franka_collect) ─────────────────
        self.gripper_close_held = False
        self.gripper_open_held  = False
        self._z_last_release = 0.0
        self._x_last_release = 0.0
        self._gripper_debounce = 0.15

        # ── Misc (one-shot) ───────────────────────────────────────────────────
        self.viz_toggle       = False   # V: toggle camera vis window
        self.print_state      = False   # M: print state
        self.horizon_increase = False   # +: horizon + 1
        self.horizon_decrease = False   # -: horizon - 1
        self.quit_requested   = False   # Q: graceful quit

        # ── Live state ────────────────────────────────────────────────────────
        self.shift_held = False

        try:
            from pynput import keyboard as pk
            self._pk = pk
            self._listener = pk.Listener(on_press=self._on_press,
                                         on_release=self._on_release)
            self._listener.start()
        except ImportError as e:
            print(f"[Keyboard] pynput unavailable ({e}) — keyboard input disabled")
            print("[Keyboard] Use Ctrl+C to quit, or set DISPLAY for interactive control")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _char(self, key):
        try:
            return key.char.lower() if hasattr(key, 'char') and key.char else None
        except AttributeError:
            return None

    def _on_press(self, key):
        pk = self._pk
        if key in (pk.Key.shift, pk.Key.shift_r):
            self.shift_held = True
            return

        c = self._char(key)

        # Policy control
        if c == 's':
            self.policy_start = True
        elif c == 'b':
            self.policy_stop = True
        elif c == 'n':
            self.single_step = True
        elif c == 'h':
            self.home_requested = True
        elif c == 'q':
            self.quit_requested = True
        elif c == 'v':
            self.viz_toggle = True
        elif c == 'm':
            self.print_state = True
        elif c == '+' or c == '=':
            self.horizon_increase = True
        elif c == '-':
            self.horizon_decrease = True

        # Gripper (debounced hold)
        elif c == 'z':
            now = time.monotonic()
            if now - self._z_last_release > self._gripper_debounce:
                self.gripper_close_held = True
        elif c == 'x':
            now = time.monotonic()
            if now - self._x_last_release > self._gripper_debounce:
                self.gripper_open_held = True

        # Special keys
        if key == pk.Key.space:
            self.pause_toggle = True
        elif key == pk.Key.esc:
            self.emergency_stop = True
        elif key == pk.Key.right:
            self.single_step = True

    def _on_release(self, key):
        pk = self._pk
        if key in (pk.Key.shift, pk.Key.shift_r):
            self.shift_held = False
            return
        c = self._char(key)
        if c == 'z':
            self.gripper_close_held = False
            self._z_last_release = time.monotonic()
        elif c == 'x':
            self.gripper_open_held = False
            self._x_last_release = time.monotonic()

    def stop(self):
        if self._listener is not None:
            self._listener.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  Franka ZeroRPC Client
# ═══════════════════════════════════════════════════════════════════════════════

class FrankaClient:
    """Thin ZeroRPC client matching franka_collect.FrankaClient."""

    def __init__(self, ip: str, port: int):
        self._c = zerorpc.Client(heartbeat=20)
        self._c.connect(f"tcp://{ip}:{port}")

    def get_ee_pose(self):
        return np.array(self._c.get_ee_pose())

    def get_tip_pose(self):
        flange = self.get_ee_pose()
        return mat_to_pose(pose_to_mat(flange) @ TX_FLANGE_TIP)

    def get_joint_positions(self):
        return np.array(self._c.get_joint_positions())

    def move_to_joint_positions(self, q, t):
        self._c.move_to_joint_positions(q.tolist(), float(t))

    def start_cartesian_impedance(self, Kx=None, Kxd=None):
        Kx = KX_DEFAULT if Kx is None else Kx
        Kxd = KXD_DEFAULT if Kxd is None else Kxd
        self._c.start_cartesian_impedance(Kx.tolist(), Kxd.tolist())

    def update_desired_ee_pose(self, pose):
        flange = mat_to_pose(pose_to_mat(pose) @ TX_TIP_FLANGE)
        self._c.update_desired_ee_pose(flange.tolist())

    def terminate_current_policy(self):
        self._c.terminate_current_policy()

    def get_gripper_state(self):
        return self._c.get_gripper_state()

    def gripper_grasp(self, speed=GRIPPER_SPEED, force=40.0):
        return self._c.gripper_grasp(float(speed), float(force))

    def gripper_release(self, speed=GRIPPER_SPEED):
        return self._c.gripper_release(float(speed))

    def close(self):
        self._c.close()


# ── Async gripper dispatch (fire-and-forget, same as franka_collect) ──────────

_gripper_lock = threading.Lock()

def _async_gripper_cmd(robot_ip, robot_port, action, *,
                       speed=GRIPPER_SPEED, force=40.0, width=0.0):
    def _run():
        with _gripper_lock:
            try:
                client = zerorpc.Client(heartbeat=20, timeout=5)
                client.connect(f"tcp://{robot_ip}:{int(robot_port)}")
                try:
                    if action == 'grasp':
                        client.gripper_grasp(float(speed), float(force))
                    elif action == 'release':
                        client.gripper_release(float(speed))
                    elif action == 'move':
                        client.gripper_move(float(width), float(speed))
                    elif action == 'stop':
                        client.gripper_stop()
                finally:
                    try:
                        client.close()
                    except Exception:
                        pass
            except Exception as e:
                print(f"[GRIPPER] async {action} failed: {e}")
    threading.Thread(target=_run, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
#  Safety Checker
# ═══════════════════════════════════════════════════════════════════════════════

class FrankaPoseSafetyChecker:
    """Workspace and single-step safety limits.

    All limits are in the Franka base frame. Adjust to your workspace.
    """

    def __init__(self,
                 x_min=0.20, x_max=0.70,
                 y_min=-0.40, y_max=0.40,
                 z_min=0.05, z_max=0.60,
                 max_delta_pos=0.08,
                 max_delta_rot=0.3):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.z_min, self.z_max = z_min, z_max
        self.max_delta_pos = max_delta_pos
        self.max_delta_rot = max_delta_rot

    def check(self, current_pose, target_pose):
        """Returns (is_safe, violations: list[str])."""
        violations = []
        x, y, z = target_pose[:3]

        if x < self.x_min or x > self.x_max:
            violations.append(f"x={x:.3f} outside [{self.x_min:.2f}, {self.x_max:.2f}]")
        if y < self.y_min or y > self.y_max:
            violations.append(f"y={y:.3f} outside [{self.y_min:.2f}, {self.y_max:.2f}]")
        if z < self.z_min or z > self.z_max:
            violations.append(f"z={z:.3f} outside [{self.z_min:.2f}, {self.z_max:.2f}]")

        dp = np.linalg.norm(target_pose[:3] - current_pose[:3])
        if dp > self.max_delta_pos:
            violations.append(f"delta_pos={dp:.4f}m > {self.max_delta_pos}m")

        r0 = st.Rotation.from_rotvec(current_pose[3:6])
        r1 = st.Rotation.from_rotvec(target_pose[3:6])
        dr_angle = (r0.inv() * r1).magnitude()
        if dr_angle > self.max_delta_rot:
            violations.append(f"delta_rot={dr_angle:.4f}rad > {self.max_delta_rot}rad")

        return len(violations) == 0, violations


# ═══════════════════════════════════════════════════════════════════════════════
#  Pose Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _interpolate_pose(start, end, alpha):
    pos = (1 - alpha) * start[:3] + alpha * end[:3]
    r0 = st.Rotation.from_rotvec(start[3:])
    r1 = st.Rotation.from_rotvec(end[3:])
    slerp = st.Slerp([0, 1], st.Rotation.concatenate([r0, r1]))
    rot = slerp(alpha).as_rotvec()
    return np.concatenate([pos, rot])


def reset_to_home(robot: FrankaClient, frequency: int = CONTROL_FREQUENCY):
    """Joint-space home → EE home with smooth interpolation."""
    print("[Home] joints → home ...")
    robot.terminate_current_policy()
    time.sleep(0.1)
    robot.move_to_joint_positions(FRANKA_HOME_JOINTS, JOINTS_HOME_DURATION)
    robot.start_cartesian_impedance()

    start_pose = robot.get_tip_pose()
    dt = 1.0 / frequency
    n_steps = max(int(HOME_MOVE_DURATION * frequency), 1)
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        waypoint = _interpolate_pose(start_pose, EE_HOME_POSE, alpha)
        robot.update_desired_ee_pose(waypoint)
        time.sleep(dt)
    print("[Home] Done.")


def apply_delta_action(current_pose, delta_action):
    """Apply 7D delta [dx,dy,dz,drx,dry,drz,grip] to current tip pose."""
    target = current_pose.copy()
    target[:3] += delta_action[:3]
    drot = st.Rotation.from_rotvec(delta_action[3:6])
    cur_rot = st.Rotation.from_rotvec(current_pose[3:6])
    target[3:6] = (drot * cur_rot).as_rotvec()
    return target


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Controller
# ═══════════════════════════════════════════════════════════════════════════════

class LeapbotController:
    """Full-featured LeapBot controller with keyboard interaction."""

    # Camera name constants (used for display and CLI defaults)
    ALL_CAM_NAMES = ('zed_0', 'zed_1', 'fisheye', 'l515_0')

    def __init__(self, args):
        self.args = args
        self.dt = 1.0 / args.frequency
        self.task = args.task
        self.action_horizon = args.action_horizon
        self.mock = getattr(args, 'mock', False)

        # Camera → model view mapping (resolved at camera init time)
        self.cam_global = args.cam_global    # e.g. "zed_0"
        self.cam_wrist  = args.cam_wrist     # e.g. "fisheye"
        self.cam_vis    = [s.strip() for s in args.cam_vis.split(',')
                           if s.strip()] if args.cam_vis else []

        # State
        self.state = PolicyState.IDLE
        self.gripper_width = MAX_GRIPPER_WIDTH
        self.viz_enabled = True

        # ── Cameras ───────────────────────────────────────────────────────────
        self.cameras = {}   # {name: camera_object}
        self._init_cameras()

        # ── Robot (skip in mock mode) ─────────────────────────────────────────
        if self.mock:
            print("[Mock] Skipping robot connection")
            self.robot = None
        else:
            self.robot = FrankaClient(args.robot_ip, args.robot_port)

        # ── Inference client ──────────────────────────────────────────────────
        self.leapbot = LeapbotClient(args.server_ip, args.server_port,
                                     timeout=args.timeout)

        # ── Safety ────────────────────────────────────────────────────────────
        self.safety = FrankaPoseSafetyChecker(
            x_min=args.safety_x_min, x_max=args.safety_x_max,
            y_min=args.safety_y_min, y_max=args.safety_y_max,
            z_min=args.safety_z_min, z_max=args.safety_z_max,
            max_delta_pos=args.max_delta_pos,
            max_delta_rot=args.max_delta_rot,
        )

        # ── Keyboard ──────────────────────────────────────────────────────────
        self.kb = PolicyKeyboardController()

    # ── Camera initialization ─────────────────────────────────────────────────

    def _init_cameras(self):
        args = self.args

        # L515 cameras
        if not args.no_l515:
            for serial in L515_SERIALS:
                name = f'l515_{len(self.cameras)}'
                try:
                    cam = L515Camera(serial)
                    self.cameras[name] = cam
                    # Re-key if the name was 'l515_0' already captured above
                    if name == 'l515_0':
                        pass  # already correct
                    print(f"[Camera] L515 {serial} → '{name}' ✓")
                except Exception as e:
                    print(f"[Camera] L515 {serial} failed: {e}")

        # Fisheye camera
        if not args.no_fisheye and FISHEYE_USB_ID:
            vid, pid = FISHEYE_USB_ID.split(':')
            dev = find_video_device_by_usb_id(vid, pid)
            if dev:
                try:
                    cam = FisheyeCamera(dev, *FISHEYE_RESOLUTION)
                    self.cameras['fisheye'] = cam
                    print(f"[Camera] Fisheye at {dev} → 'fisheye' ✓")
                except Exception as e:
                    print(f"[Camera] Fisheye failed: {e}")
            else:
                print(f"[Camera] Fisheye USB {FISHEYE_USB_ID} not found")

        # ZED cameras
        if not args.no_zed:
            if ZED_SERIALS:
                zed_serials = [int(s) for s in ZED_SERIALS]
            else:
                try:
                    zed_serials = detect_zed_serials()
                except Exception:
                    zed_serials = []
            for i, serial in enumerate(zed_serials):
                name = f'zed_{i}'
                try:
                    cam = ZEDCamera(serial, resolution=ZED_RESOLUTION,
                                    fps=ZED_FPS)
                    self.cameras[name] = cam
                    print(f"[Camera] ZED {serial} → '{name}' ✓")
                except Exception as e:
                    print(f"[Camera] ZED {serial} failed: {e}")

        # ── Validate mapping ──────────────────────────────────────────────────
        available = set(self.cameras.keys())
        print(f"\n[Camera] Available: {sorted(available)}")
        print(f"[Camera] Mapping:   global_image={self.cam_global}, "
              f"wrist_image={self.cam_wrist}")
        if self.cam_vis:
            print(f"[Camera] Vis extras: {self.cam_vis}")

        for required, label in [(self.cam_global, 'global_image'),
                                (self.cam_wrist,  'wrist_image')]:
            if required not in available:
                print(f"[WARNING] Camera '{required}' for {label} is NOT available!")
                print(f"          Available cameras: {sorted(available)}")
                print(f"          Use --cam_global / --cam_wrist to remap.")

    def _stop_cameras(self):
        for cam in self.cameras.values():
            cam.stop()

    def _get_frames(self):
        """Read all cameras → {name: rgb_uint8}."""
        frames = {}
        for name, cam in self.cameras.items():
            try:
                if isinstance(cam, L515Camera):
                    color, _ = cam.get()
                else:
                    color = cam.get()
                if color is not None:
                    frames[name] = color
            except Exception:
                pass
        return frames

    # ── Visualization ─────────────────────────────────────────────────────────

    def _show_vis(self, frames, current_tip, gripper_width):
        """Show a labeled camera grid with robot state overlay."""
        if not self.viz_enabled:
            return
        # Build ordered list: global, wrist, then vis extras
        vis_order = []
        vis_order.append((f'{self.cam_global} [global]', self.cam_global))
        vis_order.append((f'{self.cam_wrist}  [wrist]',  self.cam_wrist))
        for name in self.cam_vis:
            if name in frames and name not in (self.cam_global, self.cam_wrist):
                vis_order.append((name, name))

        items = [(label, frames[key]) for label, key in vis_order
                 if key in frames]
        if not items:
            return

        canvas = self._compose_camera_grid(items)
        if canvas is None:
            return

        h = canvas.shape[0]
        # State + info line
        state_color = {
            PolicyState.IDLE:    (200, 200, 200),
            PolicyState.AUTO:    (0, 255, 0),
            PolicyState.PAUSED:  (0, 200, 255),
            PolicyState.STOPPED: (0, 0, 255),
        }[self.state]
        cv2.putText(canvas, f"[{self.state.value}]", (10, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        info = (f"pos=[{current_tip[0]:.3f},{current_tip[1]:.3f},"
                f"{current_tip[2]:.3f}]  grip={gripper_width:.3f}m  "
                f"horizon={self.action_horizon}")
        cv2.putText(canvas, info, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        cv2.imshow('LeapBot Control', canvas)
        cv2.waitKey(1)

    @staticmethod
    def _compose_camera_grid(vis_items, cell_w=480, cell_h=360, cols=2):
        if not vis_items:
            return None
        cells = []
        for label, rgb in vis_items:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            s = min(cell_w / float(w), cell_h / float(h))
            nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
            resized = cv2.resize(bgr, (nw, nh))
            cell = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            y0, x0 = (cell_h - nh) // 2, (cell_w - nw) // 2
            cell[y0:y0 + nh, x0:x0 + nw] = resized
            cv2.putText(cell, label, (5, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cells.append(cell)
        while len(cells) % cols != 0:
            cells.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))
        rows = [np.hstack(cells[i:i + cols]) for i in range(0, len(cells), cols)]
        return np.vstack(rows)

    # ── Keyboard dispatch ─────────────────────────────────────────────────────

    def _dispatch_keyboard(self, robot, current_tip):
        """Read all keyboard flags and update state / issue commands."""
        kb = self.kb
        is_mock = self.mock

        # ── Emergency stop (highest priority, always checked) ─────────────────
        if kb.emergency_stop:
            kb.emergency_stop = False
            if self.state != PolicyState.STOPPED:
                print("\n[EMERGENCY STOP] Policy halted, opening gripper")
                self.state = PolicyState.STOPPED
                if not is_mock:
                    _async_gripper_cmd(self.args.robot_ip, self.args.robot_port,
                                       'release', speed=GRIPPER_SPEED)
                    try:
                        robot.terminate_current_policy()
                    except Exception:
                        pass
                    robot.start_cartesian_impedance()
                    robot.update_desired_ee_pose(current_tip)
                self.gripper_width = MAX_GRIPPER_WIDTH
            return True  # signal that state changed

        # ── Quit ──────────────────────────────────────────────────────────────
        if kb.quit_requested:
            kb.quit_requested = False
            print("\n[Quit] requested — exiting after cleanup")
            return False  # signal exit

        # ── Start / resume policy ─────────────────────────────────────────────
        if kb.policy_start:
            kb.policy_start = False
            if self.state in (PolicyState.IDLE, PolicyState.PAUSED):
                self.state = PolicyState.AUTO
                print(f"\n[State] → AUTO (horizon={self.action_horizon})")

        # ── Pause / resume toggle ─────────────────────────────────────────────
        if kb.pause_toggle:
            kb.pause_toggle = False
            if self.state == PolicyState.AUTO:
                self.state = PolicyState.PAUSED
                print(f"\n[State] → PAUSED")
            elif self.state == PolicyState.PAUSED:
                self.state = PolicyState.AUTO
                print(f"\n[State] → AUTO (resumed, horizon={self.action_horizon})")

        # ── Stop policy → IDLE ────────────────────────────────────────────────
        if kb.policy_stop:
            kb.policy_stop = False
            if self.state in (PolicyState.AUTO, PolicyState.PAUSED):
                self.state = PolicyState.IDLE
                print("\n[State] → IDLE (pose held)")

        # ── Single step ───────────────────────────────────────────────────────
        if kb.single_step:
            kb.single_step = False
            if self.state in (PolicyState.IDLE, PolicyState.PAUSED):
                self.state = PolicyState.AUTO
                self._single_step_pending = True
                print("\n[Step] executing one action ...")

        # ── Home (skip in mock mode) ──────────────────────────────────────────
        if kb.home_requested:
            kb.home_requested = False
            if is_mock:
                print("\n[Home] skipped (mock mode)")
            elif self.state in (PolicyState.IDLE, PolicyState.PAUSED,
                                PolicyState.STOPPED):
                print("\n[Home] resetting ...")
                reset_to_home(robot, self.args.frequency)
                self.state = PolicyState.IDLE
                self.gripper_width = MAX_GRIPPER_WIDTH
                return 'home'  # signal to re-read pose

        # ── Gripper (skip in mock mode) ───────────────────────────────────────
        if not is_mock:
            if kb.gripper_close_held:
                kb.gripper_close_held = False
                _async_gripper_cmd(self.args.robot_ip, self.args.robot_port,
                                   'grasp', speed=GRIPPER_SPEED, force=40.0)
                self.gripper_width = 0.0
                print("\n[Gripper] close")
            elif kb.gripper_open_held:
                kb.gripper_open_held = False
                _async_gripper_cmd(self.args.robot_ip, self.args.robot_port,
                                   'release', speed=GRIPPER_SPEED)
                self.gripper_width = MAX_GRIPPER_WIDTH
                print("\n[Gripper] open")
        else:
            kb.gripper_close_held = False
            kb.gripper_open_held = False

        # ── Horizon adjust ────────────────────────────────────────────────────
        if kb.horizon_increase:
            kb.horizon_increase = False
            self.action_horizon = min(self.action_horizon + 1, 32)
            print(f"\n[Horizon] → {self.action_horizon}")
        if kb.horizon_decrease:
            kb.horizon_decrease = False
            self.action_horizon = max(self.action_horizon - 1, 1)
            print(f"\n[Horizon] → {self.action_horizon}")

        # ── Visualization toggle ──────────────────────────────────────────────
        if kb.viz_toggle:
            kb.viz_toggle = False
            self.viz_enabled = not self.viz_enabled
            if not self.viz_enabled:
                cv2.destroyAllWindows()
            print(f"\n[Viz] {'ON' if self.viz_enabled else 'OFF'}")

        # ── Print state ───────────────────────────────────────────────────────
        if kb.print_state:
            kb.print_state = False
            p = current_tip
            print(f"\n[State] {self.state.value}  "
                  f"pos=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]  "
                  f"rot=[{p[3]:.3f},{p[4]:.3f},{p[5]:.3f}]  "
                  f"grip={self.gripper_width:.3f}m  "
                  f"horizon={self.action_horizon}")

        return True  # continue

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        args = self.args
        robot = self.robot
        leapbot = self.leapbot
        is_mock = self.mock

        # ── Pre-flight ────────────────────────────────────────────────────────
        mode_str = "MOCK (inference only, no robot)" if is_mock \
                   else "Real-Robot Controller"
        print("\n" + "=" * 60)
        print(f"  LeapBot {mode_str}")
        print("=" * 60)
        print("[Pre-flight] Checking GPU inference server ...")
        if not leapbot.ready():
            print("[ERROR] GPU server not ready. "
                  "Start launch_server.sh on the GPU machine first.")
            return
        print("[Pre-flight] GPU server ready ✓")

        if is_mock:
            # Mock mode: use home pose as fake proprio
            current_tip = EE_HOME_POSE.copy()
            self.gripper_width = MAX_GRIPPER_WIDTH
            print("[Pre-flight] Mock mode — using home pose as fake proprio")
        else:
            print("[Pre-flight] Connecting to Franka ...")
            robot.start_cartesian_impedance()
            print("[Pre-flight] Franka connected ✓")

            # ── Home ──────────────────────────────────────────────────────────
            if not args.skip_home:
                reset_to_home(robot, args.frequency)

            current_tip = robot.get_tip_pose()
            self.gripper_width = MAX_GRIPPER_WIDTH
            robot.gripper_release()
            time.sleep(0.5)

        # ── Banner ────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        if is_mock:
            print("  Mode           : MOCK (inference only)")
        else:
            print("  Franka server  : {}:{}".format(args.robot_ip, args.robot_port))
        print("  GPU server     : {}:{}".format(args.server_ip, args.server_port))
        print("  Task           : {}".format(self.task))
        print("  Frequency      : {} Hz".format(args.frequency))
        print("  global_image   : {}".format(self.cam_global))
        print("  wrist_image    : {}".format(self.cam_wrist))
        print("  Safety         : delta_pos≤{}m  delta_rot≤{}rad".format(
            args.max_delta_pos, args.max_delta_rot))
        print("=" * 60)
        print("  Controls:")
        print("    S        Start / resume policy")
        print("    Space    Pause / resume toggle")
        print("    B        Stop policy → IDLE")
        print("    N / →    Single-step one action")
        print("    Esc      Emergency stop")
        if not is_mock:
            print("    H        Reset to home")
            print("    Z / X    Close / open gripper")
        print("    + / -    Increase / decrease action horizon")
        print("    V        Toggle camera visualization")
        print("    M        Print current state")
        print("    Q        Quit")
        print("=" * 60)
        print(f"\n  State: {self.state.value}  —  press S to start\n")

        t_start = time.monotonic()
        iter_idx = 0
        vis_interval = 1.0 / 10.0
        last_vis = 0.0
        last_print = 0.0
        self._single_step_pending = False
        infer_ms = 0.0

        try:
            while True:
                t_cycle_end = t_start + (iter_idx + 1) * self.dt

                # ── 1. Read robot state (skip in mock) ────────────────────────
                if not is_mock:
                    try:
                        current_tip = robot.get_tip_pose()
                    except Exception:
                        pass
                    try:
                        gs = robot.get_gripper_state()
                        self.gripper_width = float(gs.get('width',
                                                          self.gripper_width))
                    except Exception:
                        pass

                # ── 2. Cameras (always read for vis, even if paused) ──────────
                cam_frames = self._get_frames()

                # ── 3. Keyboard dispatch ──────────────────────────────────────
                result = self._dispatch_keyboard(robot, current_tip)
                if result is False:
                    break   # quit requested
                if result == 'home':
                    if not is_mock:
                        current_tip = robot.get_tip_pose()
                    t_start = time.monotonic()
                    iter_idx = 0
                    continue

                # ── 4. Visualization (throttled) ──────────────────────────────
                now_m = time.monotonic()
                if (now_m - last_vis) >= vis_interval:
                    self._show_vis(cam_frames, current_tip, self.gripper_width)
                    last_vis = now_m

                # ── 5. Policy inference (only in AUTO state) ──────────────────
                should_infer = (self.state == PolicyState.AUTO)

                if should_infer:
                    global_img = cam_frames.get(self.cam_global)
                    wrist_img  = cam_frames.get(self.cam_wrist)

                    if global_img is None or wrist_img is None:
                        missing = []
                        if global_img is None:
                            missing.append(self.cam_global)
                        if wrist_img is None:
                            missing.append(self.cam_wrist)
                        print(f"\n[WARN] Missing camera: {missing} — holding")
                    else:
                        result = leapbot.infer(global_img, wrist_img,
                                               self._make_proprio(current_tip),
                                               self.task)
                        if result is None:
                            print("\n[WARN] Inference failed — holding")
                        else:
                            action_chunk = result["action_chunk"]
                            infer_ms = result["latency_ms"]

                            if is_mock:
                                # Mock: print predicted actions, don't execute
                                self._print_mock_actions(
                                    action_chunk, current_tip, infer_ms)
                            else:
                                n_exec = min(self.action_horizon,
                                             action_chunk.shape[0])
                                self._execute_horizon(
                                    robot, action_chunk, n_exec, current_tip)

                            # After single-step, auto-pause
                            if self._single_step_pending:
                                self._single_step_pending = False
                                self.state = PolicyState.PAUSED
                                print(f"  [Step] done, → PAUSED")

                # ── 6. Terminal print (2 Hz) ──────────────────────────────────
                now = time.monotonic()
                if now - last_print > 0.5:
                    p = current_tip
                    state_str = self.state.value
                    print(
                        f"\r[{state_str:6s}] "
                        f"pos=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]  "
                        f"grip={self.gripper_width:.3f}m  "
                        f"hz={self.action_horizon}  "
                        f"infer={infer_ms:.0f}ms   ",
                        end='', flush=True,
                    )
                    last_print = now

                # ── 7. Frequency regulation ───────────────────────────────────
                precise_wait(t_cycle_end)
                iter_idx += 1

        except KeyboardInterrupt:
            print("\n\n[Ctrl+C] Interrupted")

        finally:
            print("\n\nShutting down ...")
            self.kb.stop()
            cv2.destroyAllWindows()
            if not is_mock:
                try:
                    robot.terminate_current_policy()
                except Exception:
                    pass
                robot.close()
            self._stop_cameras()
            leapbot.close()
            print("Done.")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_proprio(self, tip_pose):
        proprio = np.zeros(7, dtype=np.float32)
        proprio[:6] = tip_pose
        proprio[6] = self.gripper_width
        return proprio

    def _print_mock_actions(self, action_chunk, current_tip, infer_ms):
        """Mock mode: print predicted actions and simulated targets."""
        n = min(self.action_horizon, action_chunk.shape[0])
        tip = current_tip.copy()
        print(f"\n[Mock] Inference {infer_ms:.1f}ms, "
              f"chunk shape={action_chunk.shape}, showing {n} steps:")
        print(f"       {'step':>4s}  "
              f"{'dx':>7s} {'dy':>7s} {'dz':>7s} "
              f"{'drx':>7s} {'dry':>7s} {'drz':>7s} "
              f"{'grip':>6s}  "
              f"{'target_x':>8s} {'target_y':>8s} {'target_z':>8s}")
        for i in range(n):
            a = action_chunk[i]
            if not np.isfinite(a).all():
                print(f"       {i:4d}  NON-FINITE — stopping")
                break
            target = apply_delta_action(tip, a)
            is_safe, violations = self.safety.check(tip, target)
            safe_str = "" if is_safe else f"  UNSAFE({'; '.join(violations[:2])})"
            print(f"       {i:4d}  "
                  f"{a[0]:+7.4f} {a[1]:+7.4f} {a[2]:+7.4f} "
                  f"{a[3]:+7.4f} {a[4]:+7.4f} {a[5]:+7.4f} "
                  f"{a[6]:6.3f}  "
                  f"{target[0]:+8.4f} {target[1]:+8.4f} {target[2]:+8.4f}"
                  f"{safe_str}")
            tip = target

    def _execute_horizon(self, robot, action_chunk, n_exec, current_tip):
        """Execute up to n_exec actions from the chunk with safety checks."""
        actions_done = 0
        tip = current_tip

        for step in range(n_exec):
            action = action_chunk[step]

            if not np.isfinite(action).all():
                print(f"\n[WARN] Non-finite action[{step}], stopping horizon")
                break

            target = apply_delta_action(tip, action)

            is_safe, violations = self.safety.check(tip, target)
            if not is_safe:
                print(f"\n[SAFETY] step {step} blocked: {'; '.join(violations)}")
                break

            robot.update_desired_ee_pose(target)
            actions_done += 1
            tip = target

            # Gripper (threshold-based binary)
            target_grip = float(action[6])
            if target_grip < 0.04:
                if self.gripper_width > 0.02:
                    robot.gripper_grasp(speed=GRIPPER_SPEED)
                    self.gripper_width = 0.0
            else:
                if self.gripper_width < 0.06:
                    robot.gripper_release(speed=GRIPPER_SPEED)
                    self.gripper_width = MAX_GRIPPER_WIDTH

            # Maintain frequency within horizon
            if step < n_exec - 1:
                time.sleep(self.dt)

        return actions_done


# ═══════════════════════════════════════════════════════════════════════════════
#  Config Loader
# ═══════════════════════════════════════════════════════════════════════════════

def _load_config(config_path):
    """Load a Python config file and return it as a namespace object.

    The file is loaded via importlib so it executes top-level assignments
    (SERVER_IP = "…", etc.) which become attributes of the returned object.
    Returns None if the file does not exist.
    """
    p = Path(config_path).expanduser()
    if not p.exists():
        return None
    if not p.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_leapbot_config", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _cfg(ns, attr, fallback=None):
    """Read an attribute from the config namespace, returning *fallback* if
    the attribute is missing or the namespace is None."""
    if ns is None:
        return fallback
    return getattr(ns, attr, fallback)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LeapBot real-robot controller with full keyboard control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Keyboard Controls (during execution):
  S        Start / resume policy execution
  Space    Pause / resume toggle
  B        Stop policy → IDLE (hold current pose)
  N / →    Single-step one action then auto-pause
  Esc      Emergency stop (stop policy + open gripper)
  H        Reset to home position
  Z / X    Close / open gripper (manual binary)
  + / -    Increase / decrease action horizon
  V        Toggle camera visualization window
  M        Print current state to terminal
  Q        Quit gracefully

Priority: CLI arguments > config file > built-in defaults.
Config file: --config  (default: ./leapbot_config.py)
""")

    # ── Parse only --config first ─────────────────────────────────────────────
    parser.add_argument(
        "--config", default=None,
        help="Path to leapbot_config.py (default: ./leapbot_config.py)")

    # ── Network ───────────────────────────────────────────────────────────────
    parser.add_argument("--robot_ip",    default=None)
    parser.add_argument("--robot_port",  type=int, default=None)
    parser.add_argument("--server_ip",   default=None)
    parser.add_argument("--server_port", type=int, default=None)
    parser.add_argument("--timeout",     type=float, default=None)

    # ── Task / control ────────────────────────────────────────────────────────
    parser.add_argument("--task",       default=None)
    parser.add_argument("--frequency",  type=int, default=None)
    parser.add_argument("--action_horizon", type=int, default=None)

    # ── Camera → model view mapping ───────────────────────────────────────────
    parser.add_argument("--cam_global", default=None)
    parser.add_argument("--cam_wrist",  default=None)
    parser.add_argument("--cam_vis",    default=None)

    # ── Camera disable flags ──────────────────────────────────────────────────
    parser.add_argument("--no_l515",    action="store_true", default=None)
    parser.add_argument("--no_fisheye", action="store_true", default=None)
    parser.add_argument("--no_zed",     action="store_true", default=None)

    # ── Safety ────────────────────────────────────────────────────────────────
    parser.add_argument("--max_delta_pos", type=float, default=None)
    parser.add_argument("--max_delta_rot", type=float, default=None)
    parser.add_argument("--skip_home", action="store_true", default=None)

    # ── Mock mode ─────────────────────────────────────────────────────────────
    parser.add_argument("--mock", action="store_true", default=None,
                        help="Mock mode: cameras + inference only, "
                             "no robot connection")

    # ── Workspace bounds ──────────────────────────────────────────────────────
    parser.add_argument("--safety_x_min", type=float, default=None)
    parser.add_argument("--safety_x_max", type=float, default=None)
    parser.add_argument("--safety_y_min", type=float, default=None)
    parser.add_argument("--safety_y_max", type=float, default=None)
    parser.add_argument("--safety_z_min", type=float, default=None)
    parser.add_argument("--safety_z_max", type=float, default=None)

    args = parser.parse_args()

    # ── Load config file ──────────────────────────────────────────────────────
    config_path = args.config
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "leapbot_config.py"
    cfg = _load_config(config_path)
    if cfg is not None:
        print(f"[Config] Loaded: {config_path}")
    else:
        if args.config is not None:
            print(f"[Config] WARNING: file not found: {config_path}")
        # No config file and no explicit --config: use built-in defaults
        cfg = _load_config(None)  # returns None; _cfg will use fallback

    # ── Fill None args from config, then from built-in defaults ───────────────
    def _get(cli_val, cfg_attr, builtin):
        """CLI → config → built-in."""
        if cli_val is not None:
            return cli_val
        return _cfg(cfg, cfg_attr, builtin)

    args.server_ip     = _get(args.server_ip,     "SERVER_IP",   None)
    args.server_port   = _get(args.server_port,   "SERVER_PORT", 8000)
    args.robot_ip      = _get(args.robot_ip,      "ROBOT_IP",    ROBOT_IP)
    args.robot_port    = _get(args.robot_port,     "ROBOT_PORT",  ROBOT_PORT)
    args.timeout       = _get(args.timeout,        "TIMEOUT",     5.0)
    args.task          = _get(args.task,           "TASK",        "move_objects_into_box")
    args.frequency     = _get(args.frequency,      "FREQUENCY",   CONTROL_FREQUENCY)
    args.action_horizon= _get(args.action_horizon, "ACTION_HORIZON", 4)
    args.cam_global    = _get(args.cam_global,     "CAM_GLOBAL",  "zed_0")
    args.cam_wrist     = _get(args.cam_wrist,      "CAM_WRIST",   "fisheye")
    args.cam_vis       = _get(args.cam_vis,        "CAM_VIS",     "zed_1,l515_0")
    args.max_delta_pos = _get(args.max_delta_pos,  "MAX_DELTA_POS", 0.08)
    args.max_delta_rot = _get(args.max_delta_rot,  "MAX_DELTA_ROT", 0.3)
    args.skip_home     = _get(args.skip_home,      "SKIP_HOME",   False)
    args.safety_x_min  = _get(args.safety_x_min,   "SAFETY_X_MIN", 0.20)
    args.safety_x_max  = _get(args.safety_x_max,   "SAFETY_X_MAX", 0.70)
    args.safety_y_min  = _get(args.safety_y_min,   "SAFETY_Y_MIN", -0.40)
    args.safety_y_max  = _get(args.safety_y_max,   "SAFETY_Y_MAX", 0.40)
    args.safety_z_min  = _get(args.safety_z_min,   "SAFETY_Z_MIN", 0.05)
    args.safety_z_max  = _get(args.safety_z_max,   "SAFETY_Z_MAX", 0.60)

    # Boolean flags: store_true with None default means the CLI flag
    # was NOT given.  Fall back to config, then False.
    args.no_l515    = args.no_l515    or _cfg(cfg, "NO_L515",    False)
    args.no_fisheye = args.no_fisheye or _cfg(cfg, "NO_FISHEYE", False)
    args.no_zed     = args.no_zed     or _cfg(cfg, "NO_ZED",     False)
    args.mock       = args.mock       or _cfg(cfg, "MOCK",       False)

    # ── Validate required ─────────────────────────────────────────────────────
    if args.server_ip is None:
        parser.error(
            "--server_ip is required (not set in CLI or config file).\n"
            "Set SERVER_IP in leapbot_config.py or pass --server_ip on CLI.")

    controller = LeapbotController(args)
    controller.run()


if __name__ == "__main__":
    main()
