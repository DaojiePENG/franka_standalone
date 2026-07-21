#!/usr/bin/env python3
"""
Workspace Safety Bounds Calibration Tool

Uses keyboard teleop to jog the robot to each corner of your desired workspace.
The script tracks the min/max tip positions reached and prints the safety
bounds you should paste into leapbot_control.py's FrankaPoseSafetyChecker.

Workflow:
  1. Run this script (robot starts at home position)
  2. Use WASD/QE to move the robot to each boundary of your workspace:
     - Push to the FURTHEST forward  position you want to allow  (x_max)
     - Push to the FURTHEST backward position you want to allow  (x_min)
     - Push to the LEFT / RIGHT limits (y_min / y_max)
     - Push to the UP / DOWN limits    (z_min / z_max)
  3. At each boundary, hold the position for ~1 second so it registers
  4. Press T to print the current bounds
  5. Press P to print the FINAL bounds and copy-paste them
  6. Press H to reset to home at any time
  7. Press Esc to quit

The script uses MARGIN (default 0.02m) to shrink the bounds slightly so
you don't operate right at the physical limit.

Usage:
    python calibrate_workspace.py --robot_ip localhost --robot_port 4242
    python calibrate_workspace.py --robot_ip localhost --margin 0.03
"""
import argparse
import time
import sys
from pathlib import Path

import numpy as np
import scipy.spatial.transform as st
import zerorpc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    ROBOT_IP, ROBOT_PORT, CONTROL_FREQUENCY,
    POS_SPEED, ROT_SPEED, MAX_GRIPPER_WIDTH, GRIPPER_SPEED,
    EE_HOME_POSE, HOME_MOVE_DURATION,
    FRANKA_HOME_JOINTS, JOINTS_HOME_DURATION,
    KX_DEFAULT, KXD_DEFAULT,
    TX_FLANGE_TIP, TX_TIP_FLANGE,
)


# ── Pose utilities ───────────────────────────────────────────────────────────

def pose_to_mat(pose):
    mat = np.eye(4)
    mat[:3, 3] = pose[:3]
    mat[:3, :3] = st.Rotation.from_rotvec(pose[3:]).as_matrix()
    return mat

def mat_to_pose(mat):
    pos = mat[:3, 3]
    rot = st.Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    return np.concatenate([pos, rot])


# ── Franka Client ────────────────────────────────────────────────────────────

class FrankaClient:
    def __init__(self, ip, port):
        self._c = zerorpc.Client(heartbeat=20)
        self._c.connect(f"tcp://{ip}:{port}")

    def get_ee_pose(self):
        return np.array(self._c.get_ee_pose())

    def get_tip_pose(self):
        flange = self.get_ee_pose()
        return mat_to_pose(pose_to_mat(flange) @ TX_FLANGE_TIP)

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

    def gripper_release(self, speed=GRIPPER_SPEED):
        return self._c.gripper_release(float(speed))

    def close(self):
        self._c.close()


# ── Keyboard ─────────────────────────────────────────────────────────────────

class CalibrationKeyboard:
    """WASD/QE jog + special keys."""

    KEY_MAP = {
        'w': 'fwd', 's': 'bwd', 'a': 'left', 'd': 'right',
        'q': 'up',  'e': 'down',
        'j': 'yaw_l', 'l': 'yaw_r',
        'i': 'pit_u', 'k': 'pit_d',
        'u': 'rol_l', 'o': 'rol_r',
    }

    def __init__(self):
        from pynput import keyboard as pk
        self._pk = pk
        self._states = {v: False for v in self.KEY_MAP.values()}
        self.shift_held = False
        self.quit = False
        self.home = False
        self.print_bounds = False
        self.print_final = False
        self.reset_bounds = False

        self._listener = pk.Listener(on_press=self._press,
                                     on_release=self._release)
        self._listener.start()

    def _char(self, key):
        try:
            return key.char.lower() if hasattr(key, 'char') and key.char else None
        except AttributeError:
            return None

    def _press(self, key):
        if key in (self._pk.Key.shift, self._pk.Key.shift_r):
            self.shift_held = True
            return
        c = self._char(key)
        if c in self.KEY_MAP:
            self._states[self.KEY_MAP[c]] = True
        elif c == 'h':
            self.home = True
        elif c == 't':
            self.print_bounds = True
        elif c == 'p':
            self.print_final = True
        elif c == 'r':
            self.reset_bounds = True
        if key == self._pk.Key.esc:
            self.quit = True

    def _release(self, key):
        if key in (self._pk.Key.shift, self._pk.Key.shift_r):
            self.shift_held = False
            return
        c = self._char(key)
        if c in self.KEY_MAP:
            self._states[self.KEY_MAP[c]] = False

    def get_velocity(self, dt):
        mult = 3.0 if self.shift_held else 1.0
        s, ps, rs = self._states, POS_SPEED * mult, ROT_SPEED * mult
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


# ── Home ─────────────────────────────────────────────────────────────────────

def _interpolate_pose(start, end, alpha):
    pos = (1 - alpha) * start[:3] + alpha * end[:3]
    r0 = st.Rotation.from_rotvec(start[3:])
    r1 = st.Rotation.from_rotvec(end[3:])
    slerp = st.Slerp([0, 1], st.Rotation.concatenate([r0, r1]))
    rot = slerp(alpha).as_rotvec()
    return np.concatenate([pos, rot])


def reset_to_home(robot, frequency=CONTROL_FREQUENCY):
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


# ── Bounds Tracker ───────────────────────────────────────────────────────────

class BoundsTracker:
    """Tracks the min/max tip positions the user has jogged to."""

    def __init__(self, margin=0.02):
        self.margin = margin
        self.reset()

    def reset(self):
        self.x_min = +np.inf
        self.x_max = -np.inf
        self.y_min = +np.inf
        self.y_max = -np.inf
        self.z_min = +np.inf
        self.z_max = -np.inf

    def update(self, pose):
        x, y, z = pose[:3]
        self.x_min = min(self.x_min, x)
        self.x_max = max(self.x_max, x)
        self.y_min = min(self.y_min, y)
        self.y_max = max(self.y_max, y)
        self.z_min = min(self.z_min, z)
        self.z_max = max(self.z_max, z)

    @property
    def has_data(self):
        return self.x_min != +np.inf

    def raw_bounds(self):
        return {
            'x_min': round(self.x_min, 4), 'x_max': round(self.x_max, 4),
            'y_min': round(self.y_min, 4), 'y_max': round(self.y_max, 4),
            'z_min': round(self.z_min, 4), 'z_max': round(self.z_max, 4),
        }

    def safe_bounds(self):
        """Apply margin shrinkage for safe limits."""
        m = self.margin
        return {
            'x_min': round(self.x_min + m, 4), 'x_max': round(self.x_max - m, 4),
            'y_min': round(self.y_min + m, 4), 'y_max': round(self.y_max - m, 4),
            'z_min': round(self.z_min + m, 4), 'z_max': round(self.z_max - m, 4),
        }

    def format_print(self, label, bounds_dict):
        print(f"\n{'─' * 50}")
        print(f"  {label}")
        print(f"{'─' * 50}")
        for key in ('x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'):
            print(f"  {key:6s} = {bounds_dict[key]:+.4f}")
        print(f"{'─' * 50}")

    def format_paste(self):
        """Print the exact Python code to paste into the safety checker."""
        b = self.safe_bounds()
        print("\n" + "=" * 60)
        print("  COPY-PASTE into leapbot_control.py")
        print("  (in FrankaPoseSafetyChecker.__init__ defaults)")
        print("=" * 60)
        print()
        print(f"    x_min={b['x_min']}, x_max={b['x_max']},")
        print(f"    y_min={b['y_min']}, y_max={b['y_max']},")
        print(f"    z_min={b['z_min']}, z_max={b['z_max']},")
        print()
        print("  Or use as CLI arguments:")
        print()
        print(f"    --safety_x_min {b['x_min']} --safety_x_max {b['x_max']} \\")
        print(f"    --safety_y_min {b['y_min']} --safety_y_max {b['y_max']} \\")
        print(f"    --safety_z_min {b['z_min']} --safety_z_max {b['z_max']}")
        print("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate workspace safety bounds for LeapBot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. Robot starts at home position
  2. Jog with WASD/QE to each boundary of your desired workspace
  3. At each limit, hold the position briefly so it registers
  4. Press T to preview current bounds
  5. Press P to print final bounds (copy-paste ready)
  6. Press R to reset bounds tracking
  7. Press H to return to home
  8. Press Esc to quit

Keyboard:
  W/S    Forward / Backward   (X axis)
  A/D    Left / Right         (Y axis)
  Q/E    Up / Down            (Z axis)
  J/L    Yaw left / right
  I/K    Pitch up / down
  U/O    Roll left / right
  Shift  Hold for 3x speed
  T      Print current bounds
  P      Print FINAL bounds (copy-paste code)
  R      Reset bounds tracking
  H      Home
  Esc    Quit
""")
    parser.add_argument("--robot_ip", default=ROBOT_IP)
    parser.add_argument("--robot_port", type=int, default=ROBOT_PORT)
    parser.add_argument("--frequency", type=int, default=CONTROL_FREQUENCY)
    parser.add_argument("--margin", type=float, default=0.02,
                        help="Safety margin to shrink bounds (m, default: 0.02)")
    parser.add_argument("--skip_home", action="store_true")
    args = parser.parse_args()

    robot = FrankaClient(args.robot_ip, args.robot_port)
    kb = CalibrationKeyboard()
    tracker = BoundsTracker(margin=args.margin)
    dt = 1.0 / args.frequency

    print("=" * 60)
    print("  Workspace Calibration Tool")
    print(f"  Server: {args.robot_ip}:{args.robot_port}")
    print(f"  Margin: {args.margin}m")
    print("=" * 60)
    print("  Jog the robot to each workspace boundary.")
    print("  WASD/QE = move, Shift = fast, H = home")
    print("  T = preview bounds, P = print final, R = reset")
    print("  Esc = quit")
    print("=" * 60)

    try:
        robot.start_cartesian_impedance()
        if not args.skip_home:
            reset_to_home(robot, args.frequency)

        target_pose = robot.get_tip_pose()
        robot.gripper_release()
        last_print = 0.0

        print(f"\n  Home tip pose: "
              f"pos=[{target_pose[0]:.4f}, {target_pose[1]:.4f}, {target_pose[2]:.4f}]\n")

        while not kb.quit:
            t0 = time.monotonic()

            # Keyboard events
            if kb.home:
                kb.home = False
                reset_to_home(robot, args.frequency)
                target_pose = robot.get_tip_pose()

            if kb.print_bounds:
                kb.print_bounds = False
                if tracker.has_data:
                    tracker.format_print("RAW bounds (exactly where you jogged)",
                                         tracker.raw_bounds())
                    tracker.format_print("SAFE bounds (with margin)",
                                         tracker.safe_bounds())
                else:
                    print("[Bounds] No data yet — jog the robot first")

            if kb.print_final:
                kb.print_final = False
                if tracker.has_data:
                    tracker.format_paste()
                else:
                    print("[Bounds] No data yet — jog the robot first")

            if kb.reset_bounds:
                kb.reset_bounds = False
                tracker.reset()
                print("[Bounds] Reset — jog again to record new bounds")

            # Jog
            dpos, drot_xyz = kb.get_velocity(dt)
            target_pose[:3] += dpos
            drot = st.Rotation.from_euler('xyz', drot_xyz)
            target_pose[3:] = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec()
            robot.update_desired_ee_pose(target_pose)

            # Update tracker
            try:
                actual_tip = robot.get_tip_pose()
                tracker.update(actual_tip)
            except Exception:
                tracker.update(target_pose)

            # Terminal print (2 Hz)
            now = time.monotonic()
            if now - last_print > 0.5:
                p = target_pose
                tracked = ""
                if tracker.has_data:
                    b = tracker.raw_bounds()
                    tracked = (f"  bounds: x[{b['x_min']:.3f},{b['x_max']:.3f}] "
                               f"y[{b['y_min']:.3f},{b['y_max']:.3f}] "
                               f"z[{b['z_min']:.3f},{b['z_max']:.3f}]")
                print(f"\rpos=[{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}]  "
                      f"rot=[{p[3]:.3f},{p[4]:.3f},{p[5]:.3f}]{tracked}   ",
                      end='', flush=True)
                last_print = now

            # Frequency
            elapsed = time.monotonic() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\n\n[Ctrl+C]")
    finally:
        kb.stop()
        if tracker.has_data:
            print("\n")
            tracker.format_paste()
        try:
            robot.terminate_current_policy()
        except Exception:
            pass
        robot.close()
        print("\nDone.")


if __name__ == '__main__':
    main()
