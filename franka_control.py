#!/usr/bin/env python3
"""
Franka 上位机 — Keyboard teleop (NO camera).

Connects to franka_server.py via ZeroRPC and uses Cartesian impedance control.
Keyboard WASD controls the robot end-effector, Z/X controls the gripper.

Usage:
    python franka_control.py
    python franka_control.py --robot_ip 192.168.3.2 --robot_port 4242

Controls:
    W/S   Forward / Backward  (X)
    A/D   Left / Right        (Y)
    Q/E   Up / Down           (Z)
    J/L   Yaw left / right
    I/K   Pitch up / down
    U/O   Roll left / right
    Shift Hold for 3x speed
    Z     Close gripper (full)
    X     Open gripper (full)
    H     Reset to home
    Esc   Quit
"""
import argparse
import time
import sys
import threading

import numpy as np
import scipy.spatial.transform as st
import zerorpc
from pynput import keyboard as pynput_keyboard

from config import (
    ROBOT_IP, ROBOT_PORT, CONTROL_FREQUENCY,
    POS_SPEED, ROT_SPEED, MAX_GRIPPER_WIDTH, GRIPPER_SPEED,
    EE_HOME_POSE, HOME_MOVE_DURATION,
    FRANKA_HOME_JOINTS, JOINTS_HOME_DURATION,
    KX_DEFAULT, KXD_DEFAULT,
    TX_FLANGE_TIP, TX_TIP_FLANGE,
)


# ======================== Pose Utilities ========================

def pose_to_mat(pose):
    """[x,y,z,rx,ry,rz] -> 4x4 matrix."""
    mat = np.eye(4)
    mat[:3, 3] = pose[:3]
    mat[:3, :3] = st.Rotation.from_rotvec(pose[3:]).as_matrix()
    return mat

def mat_to_pose(mat):
    """4x4 matrix -> [x,y,z,rx,ry,rz]."""
    pos = mat[:3, 3]
    rot = st.Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    return np.concatenate([pos, rot])


# ======================== ZeroRPC Client ========================

class FrankaClient:
    """Thin ZeroRPC client wrapping the NUC server."""

    def __init__(self, ip: str, port: int):
        self._c = zerorpc.Client(heartbeat=20)
        self._c.connect(f"tcp://{ip}:{port}")

    # -- arm --
    def get_ee_pose(self):
        """Flange pose as [x,y,z,rx,ry,rz]."""
        return np.array(self._c.get_ee_pose())

    def get_tip_pose(self):
        """Tool-tip pose (flange + tip offset)."""
        flange = self.get_ee_pose()
        return mat_to_pose(pose_to_mat(flange) @ TX_FLANGE_TIP)

    def get_joint_positions(self):
        return np.array(self._c.get_joint_positions())

    def move_to_joint_positions(self, q, time_to_go):
        self._c.move_to_joint_positions(q.tolist(), float(time_to_go))

    def start_cartesian_impedance(self, Kx=None, Kxd=None):
        Kx = KX_DEFAULT if Kx is None else Kx
        Kxd = KXD_DEFAULT if Kxd is None else Kxd
        self._c.start_cartesian_impedance(Kx.tolist(), Kxd.tolist())

    def update_desired_ee_pose(self, pose):
        """Send tip pose; internally converts to flange frame."""
        flange = mat_to_pose(pose_to_mat(pose) @ TX_TIP_FLANGE)
        self._c.update_desired_ee_pose(flange.tolist())

    def terminate_current_policy(self):
        self._c.terminate_current_policy()

    # -- gripper --
    def get_gripper_state(self):
        return self._c.get_gripper_state()

    def gripper_grasp(self, speed=0.2, force=40.0):
        return self._c.gripper_grasp(float(speed), float(force))

    def gripper_release(self, speed=0.2):
        return self._c.gripper_release(float(speed))

    def gripper_move(self, width, speed=0.2):
        """Position move to width (m); server stops any previous motion itself."""
        return self._c.gripper_move(float(width), float(speed))

    def gripper_stop(self):
        """Interrupt in-flight gripper motion; returns settled state dict (width m)."""
        return self._c.gripper_stop()

    def close(self):
        self._c.close()


# ==================== Async Gripper Dispatch ====================
# Gripper grasp/release block on the server until the motion settles. Firing
# them on the shared control client would stall the pose loop for the whole
# motion, so (mirroring demo_franka_keyboard_wrist_L515_dual_zed.py's
# _async_gripper_cmd) each command runs fire-and-forget on a fresh per-call
# zerorpc client, serialized by a lock so concurrent presses never interleave.
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
                    else:
                        print(f"[GRIPPER] unknown async action: {action!r}")
                finally:
                    try:
                        client.close()
                    except Exception:
                        pass
            except Exception as e:
                print(f"[GRIPPER] async {action} failed: {e}")
    threading.Thread(target=_run, daemon=True).start()


# ======================== Keyboard Handler ========================

class KeyboardTeleop:
    """Reads WASD + rotation keys. Call get_velocity(dt) each cycle."""

    KEY_MAP = {
        'w': 'fwd', 's': 'bwd', 'a': 'left', 'd': 'right',
        'q': 'up',  'e': 'down',
        'j': 'yaw_l', 'l': 'yaw_r',
        'i': 'pit_u', 'k': 'pit_d',
        'u': 'rol_l', 'o': 'rol_r',
    }

    def __init__(self, pos_speed=0.08, rot_speed=0.3):
        self.pos_speed = pos_speed
        self.rot_speed = rot_speed
        self._states = {v: False for v in self.KEY_MAP.values()}
        # One-shot gripper requests: set True on a genuine Z/X DOWN-edge and
        # consumed (cleared) exactly once by the control loop. X11 key
        # auto-repeat replays a synthetic release+press pair per repeat while a
        # key is held, which naive press-edge detection would read as many
        # presses. We reject any press that lands within _gripper_debounce of
        # the last release of the same key, so one physical press = one request.
        self.gripper_close_req = False
        self.gripper_open_req = False
        self._z_last_release = 0.0
        self._x_last_release = 0.0
        self._gripper_debounce = 0.05
        self.shift_held = False
        self.quit_requested = False
        self.home_requested = False

        self._listener = pynput_keyboard.Listener(
            on_press=self._press, on_release=self._release)
        self._listener.start()

    def _char(self, key):
        try:
            return key.char.lower() if hasattr(key, 'char') and key.char else None
        except AttributeError:
            return None

    def _press(self, key):
        if key in (pynput_keyboard.Key.shift, pynput_keyboard.Key.shift_r):
            self.shift_held = True
            return
        c = self._char(key)
        if c in self.KEY_MAP:
            self._states[self.KEY_MAP[c]] = True
        elif c == 'z':
            if time.monotonic() - self._z_last_release > self._gripper_debounce:
                self.gripper_close_req = True
        elif c == 'x':
            if time.monotonic() - self._x_last_release > self._gripper_debounce:
                self.gripper_open_req = True
        elif c == 'h':
            self.home_requested = True
        if key == pynput_keyboard.Key.esc:
            self.quit_requested = True

    def _release(self, key):
        if key in (pynput_keyboard.Key.shift, pynput_keyboard.Key.shift_r):
            self.shift_held = False
            return
        c = self._char(key)
        if c in self.KEY_MAP:
            self._states[self.KEY_MAP[c]] = False
        elif c == 'z':
            self._z_last_release = time.monotonic()
        elif c == 'x':
            self._x_last_release = time.monotonic()

    def get_velocity(self, dt):
        mult = 3.0 if self.shift_held else 1.0
        s, ps, rs = self._states, self.pos_speed * mult, self.rot_speed * mult
        dp = np.zeros(3)
        if s['fwd']:   dp[0] += ps * dt
        if s['bwd']:   dp[0] -= ps * dt
        if s['left']:  dp[1] += ps * dt
        if s['right']: dp[1] -= ps * dt
        if s['up']:    dp[2] += ps * dt
        if s['down']:  dp[2] -= ps * dt

        dr = np.zeros(3)
        if s['rol_l']: dr[0] -= rs * dt
        if s['rol_r']: dr[0] += rs * dt
        if s['pit_u']: dr[1] -= rs * dt
        if s['pit_d']: dr[1] += rs * dt
        if s['yaw_l']: dr[2] += rs * dt
        if s['yaw_r']: dr[2] -= rs * dt
        return dp, dr

    def stop(self):
        self._listener.stop()


# ======================== Main Loop ========================

def _interpolate_pose(start, end, alpha):
    """Linearly interpolate position + SLERP rotation between two poses."""
    pos = (1 - alpha) * start[:3] + alpha * end[:3]
    r0 = st.Rotation.from_rotvec(start[3:])
    r1 = st.Rotation.from_rotvec(end[3:])
    slerp = st.Slerp([0, 1], st.Rotation.concatenate([r0, r1]))
    rot = slerp(alpha).as_rotvec()
    return np.concatenate([pos, rot])


def reset_to_home(robot: FrankaClient, frequency: int = CONTROL_FREQUENCY):
    """Joint-space home -> EE home with smooth interpolation."""
    print("[Home] Step 1: joints home ...")
    robot.terminate_current_policy()
    time.sleep(0.1)
    robot.move_to_joint_positions(FRANKA_HOME_JOINTS, JOINTS_HOME_DURATION)

    print("[Home] Step 2: starting impedance ...")
    robot.start_cartesian_impedance()

    print("[Home] Step 3: moving EE to home pose ...")
    start_pose = robot.get_tip_pose()
    dt = 1.0 / frequency
    n_steps = max(int(HOME_MOVE_DURATION * frequency), 1)
    for i in range(1, n_steps + 1):
        alpha = i / n_steps
        waypoint = _interpolate_pose(start_pose, EE_HOME_POSE, alpha)
        robot.update_desired_ee_pose(waypoint)
        time.sleep(dt)
    print("[Home] Done.")


def main():
    parser = argparse.ArgumentParser(description='Franka keyboard control (no camera)')
    parser.add_argument('--robot_ip', default=ROBOT_IP)
    parser.add_argument('--robot_port', type=int, default=ROBOT_PORT)
    parser.add_argument('--frequency', type=int, default=CONTROL_FREQUENCY)
    parser.add_argument('--pos_speed', type=float, default=POS_SPEED)
    parser.add_argument('--rot_speed', type=float, default=ROT_SPEED)
    parser.add_argument('--init_home', action='store_true', default=True,
                        help='Reset to home on start')
    args = parser.parse_args()

    print("=" * 55)
    print("  Franka Keyboard Control (上位机, no camera)")
    print(f"  Server: {args.robot_ip}:{args.robot_port}")
    print(f"  Freq:   {args.frequency} Hz")
    print("=" * 55)

    robot = FrankaClient(args.robot_ip, args.robot_port)
    teleop = KeyboardTeleop(pos_speed=args.pos_speed, rot_speed=args.rot_speed)

    try:
        # start impedance controller
        robot.start_cartesian_impedance()
        print("[Init] Cartesian impedance started.")

        if args.init_home:
            reset_to_home(robot, args.frequency)

        target_pose = robot.get_tip_pose()
        gripper_pos = MAX_GRIPPER_WIDTH
        robot.gripper_release()

        print("\nReady! WASD to move; Z close gripper, X open gripper; "
              "H home, Esc quit.")
        dt = 1.0 / args.frequency
        last_print = 0.0

        while not teleop.quit_requested:
            t0 = time.monotonic()

            # home
            if teleop.home_requested:
                teleop.home_requested = False
                reset_to_home(robot, args.frequency)
                target_pose = robot.get_tip_pose()
                gripper_pos = MAX_GRIPPER_WIDTH

            # keyboard velocity
            dpos, drot_xyz = teleop.get_velocity(dt)
            target_pose[:3] += dpos
            drot = st.Rotation.from_euler('xyz', drot_xyz)
            target_pose[3:] = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec()

            # send to robot
            robot.update_desired_ee_pose(target_pose)

            # gripper: full close / open, one dispatch per Z/X press edge. Z =
            # full force-close (grasp), X = full open (release). The DOWN-edge
            # is detected in KeyboardTeleop (auto-repeat debounced) and delivered
            # as a one-shot request, so a single press fires exactly one motion.
            # Dispatch is async (fire-and-forget) so the blocking gripper RPC
            # never stalls the pose loop; gripper_pos is a pure input-driven cmd,
            # feedback never writes it. Print only on an actual target change.
            if teleop.gripper_close_req:
                teleop.gripper_close_req = False
                _async_gripper_cmd(args.robot_ip, args.robot_port, 'grasp',
                                   speed=GRIPPER_SPEED, force=40.0)
                if gripper_pos != 0.0:
                    print("\n[GRIPPER] full force-close dispatched")
                gripper_pos = 0.0
            elif teleop.gripper_open_req:
                teleop.gripper_open_req = False
                _async_gripper_cmd(args.robot_ip, args.robot_port, 'release',
                                   speed=GRIPPER_SPEED)
                if gripper_pos != MAX_GRIPPER_WIDTH:
                    print("\n[GRIPPER] full open dispatched")
                gripper_pos = MAX_GRIPPER_WIDTH

            # terminal print (2 Hz)
            now = time.monotonic()
            if now - last_print > 0.5:
                p = target_pose
                print(f"\rpos=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]  "
                      f"rot=[{p[3]:.3f},{p[4]:.3f},{p[5]:.3f}]  "
                      f"gripper={gripper_pos:.3f}m", end='', flush=True)
                last_print = now

            # regulate frequency
            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nShutting down ...")
        teleop.stop()
        try:
            robot.terminate_current_policy()
        except Exception:
            pass
        robot.close()
        print("Done.")


if __name__ == '__main__':
    main()
