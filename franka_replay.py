#!/usr/bin/env python3
"""
Franka episode replay utility.

Replays a recorded episode from franka_collect.py by streaming the saved tool-tip
pose and gripper command sequence back to franka_server.py over ZeroRPC.

Usage:
    python franka_replay.py D:\atten_vla_franka_data\franka_standalone\collected_data_basic_demo\episode_0000
    python franka_replay.py ./collected_data_basic_demo/episode_0000 --speed 2.0

Controls During Replay:
    Space      Pause / resume
    Right / N  Single-step one frame and stay paused
    Esc        Stop replay immediately
"""
import argparse
import threading
import time
from pathlib import Path

import numpy as np
import scipy.spatial.transform as st
import zerorpc
from pynput import keyboard as pynput_keyboard

from config import (
    ROBOT_IP, ROBOT_PORT, CONTROL_FREQUENCY,
    MAX_GRIPPER_WIDTH, GRIPPER_SPEED, GRIPPER_FORCE,
    EE_HOME_POSE, HOME_MOVE_DURATION,
    FRANKA_HOME_JOINTS, JOINTS_HOME_DURATION,
    KX_DEFAULT, KXD_DEFAULT,
    TX_FLANGE_TIP, TX_TIP_FLANGE,
)


def pose_to_mat(pose):
    mat = np.eye(4)
    mat[:3, 3] = pose[:3]
    mat[:3, :3] = st.Rotation.from_rotvec(pose[3:]).as_matrix()
    return mat


def mat_to_pose(mat):
    pos = mat[:3, 3]
    rot = st.Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    return np.concatenate([pos, rot])


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

    def gripper_grasp(self, speed=GRIPPER_SPEED, force=GRIPPER_FORCE):
        return self._c.gripper_grasp(float(speed), float(force))

    def gripper_release(self, speed=GRIPPER_SPEED):
        return self._c.gripper_release(float(speed))

    def close(self):
        self._c.close()


def _interpolate_pose(start, end, alpha):
    pos = (1 - alpha) * start[:3] + alpha * end[:3]
    r0 = st.Rotation.from_rotvec(start[3:])
    r1 = st.Rotation.from_rotvec(end[3:])
    slerp = st.Slerp([0, 1], st.Rotation.concatenate([r0, r1]))
    rot = slerp(alpha).as_rotvec()
    return np.concatenate([pos, rot])


def move_to_pose(robot, target_pose, duration, frequency):
    start_pose = robot.get_tip_pose()
    dt = 1.0 / frequency
    n_steps = max(int(duration * frequency), 1)
    for step_idx in range(1, n_steps + 1):
        alpha = step_idx / n_steps
        robot.update_desired_ee_pose(_interpolate_pose(start_pose, target_pose, alpha))
        time.sleep(dt)


def reset_to_home(robot, frequency=CONTROL_FREQUENCY):
    print("[Home] joints -> home ...")
    robot.terminate_current_policy()
    time.sleep(0.1)
    robot.move_to_joint_positions(FRANKA_HOME_JOINTS, JOINTS_HOME_DURATION)
    robot.start_cartesian_impedance()
    move_to_pose(robot, EE_HOME_POSE, HOME_MOVE_DURATION, frequency)
    print("[Home] Done.")


def resolve_episode_npz(episode_path):
    episode_path = Path(episode_path)
    if episode_path.is_dir():
        npz_path = episode_path / 'robot_data.npz'
    else:
        npz_path = episode_path
    if not npz_path.exists():
        raise FileNotFoundError(f"Episode file not found: {npz_path}")
    return npz_path


def load_episode(npz_path):
    with np.load(npz_path) as data:
        timestamps = np.asarray(data['timestamps'], dtype=np.float64)
        actions = np.asarray(data['actions'], dtype=np.float64)
        robot_states = np.asarray(data['robot_states'], dtype=np.float64)

    if timestamps.ndim != 1:
        raise ValueError(f"timestamps must be 1D, got shape {timestamps.shape}")
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"actions must be shaped (N, 7), got {actions.shape}")
    if robot_states.ndim != 2 or robot_states.shape[1] != 7:
        raise ValueError(f"robot_states must be shaped (N, 7), got {robot_states.shape}")
    if len(timestamps) != len(actions):
        raise ValueError("timestamps and actions length mismatch")
    if len(actions) == 0:
        raise ValueError("episode is empty")
    if np.any(np.diff(timestamps) < 0):
        raise ValueError("timestamps must be non-decreasing")

    return timestamps, actions, robot_states


def apply_gripper_command(robot, target_width, threshold):
    if target_width <= threshold:
        robot.gripper_grasp()
        return 0.0
    robot.gripper_release()
    return MAX_GRIPPER_WIDTH


class ReplayInterrupted(RuntimeError):
    pass


class ReplayKeyboardController:
    def __init__(self, poll_interval=0.02):
        self.poll_interval = poll_interval
        self._lock = threading.Lock()
        self.paused = False
        self.quit_requested = False
        self.step_requested = False
        self._listener = pynput_keyboard.Listener(on_press=self._on_press)
        self._listener.start()

    def _on_press(self, key):
        with self._lock:
            if key == pynput_keyboard.Key.space:
                self.paused = not self.paused
                state = 'paused' if self.paused else 'resumed'
                print(f"\n[Replay] {state}.")
            elif key == pynput_keyboard.Key.esc:
                self.quit_requested = True
                print("\n[Replay] interrupt requested.")
            elif key == pynput_keyboard.Key.right:
                self.paused = True
                self.step_requested = True
                print("\n[Replay] single-step requested.")
            else:
                try:
                    char = key.char.lower() if hasattr(key, 'char') and key.char else None
                except AttributeError:
                    char = None
                if char == 'n':
                    self.paused = True
                    self.step_requested = True
                    print("\n[Replay] single-step requested.")

    def wait(self, delta_seconds):
        remaining = max(float(delta_seconds), 0.0)
        last_tick = time.monotonic()
        while True:
            now = time.monotonic()
            elapsed = now - last_tick
            last_tick = now
            with self._lock:
                if self.quit_requested:
                    raise ReplayInterrupted("Replay interrupted by user")
                if self.step_requested:
                    self.step_requested = False
                    return 'step'
                paused = self.paused
            if paused:
                time.sleep(self.poll_interval)
                continue
            remaining -= elapsed
            if remaining <= 0:
                return 'play'
            time.sleep(min(self.poll_interval, remaining))

    def stop(self):
        self._listener.stop()


def replay_episode(robot, timestamps, actions, speed, gripper_threshold,
                   controller, print_every=50):
    last_gripper_closed = None

    for idx, (timestamp, action) in enumerate(zip(timestamps, actions)):
        if idx == 0:
            wait_result = controller.wait(0.0)
        else:
            delta_t = (timestamp - timestamps[idx - 1]) / speed
            wait_result = controller.wait(delta_t)

        robot.update_desired_ee_pose(action[:6])

        is_closed = action[6] <= gripper_threshold
        if is_closed != last_gripper_closed:
            apply_gripper_command(robot, action[6], gripper_threshold)
            last_gripper_closed = is_closed

        if idx == 0 or (idx + 1) % print_every == 0 or idx + 1 == len(actions):
            elapsed = timestamp - timestamps[0]
            mode = 'STEP' if wait_result == 'step' else 'PLAY'
            print(
                f"[Replay] {idx + 1}/{len(actions)}  "
                f"t={elapsed:.2f}s  mode={mode}  pos={np.array2string(action[:3], precision=3)}  "
                f"gripper={action[6]:.3f}"
            )


def build_parser():
    parser = argparse.ArgumentParser(description='Replay a recorded Franka episode')
    parser.add_argument('episode', help='Episode directory or robot_data.npz path')
    parser.add_argument('--robot_ip', default=ROBOT_IP)
    parser.add_argument('--robot_port', type=int, default=ROBOT_PORT)
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier, e.g. 2.0 = 2x faster')
    parser.add_argument('--frequency', type=int, default=CONTROL_FREQUENCY,
                        help='Interpolation frequency for pre-positioning moves')
    parser.add_argument('--gripper_threshold', type=float, default=0.04,
                        help='Widths at or below this are treated as closed')
    parser.add_argument('--move_to_start_duration', type=float, default=3.0,
                        help='Seconds used to move from current pose to the first recorded pose')
    parser.add_argument('--hold_final', type=float, default=1.0,
                        help='Seconds to hold the last pose after playback')
    parser.add_argument('--skip_home', action='store_true',
                        help='Skip the joint-home reset before moving to the start pose')
    parser.add_argument('--skip_move_to_start', action='store_true',
                        help='Start replay immediately from the current pose')
    parser.add_argument('--dry_run', action='store_true',
                        help='Only inspect the episode without commanding the robot')
    return parser


def main():
    args = build_parser().parse_args()
    if args.speed <= 0:
        raise ValueError('--speed must be > 0')
    if args.frequency <= 0:
        raise ValueError('--frequency must be > 0')

    npz_path = resolve_episode_npz(args.episode)
    timestamps, actions, robot_states = load_episode(npz_path)

    duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
    print("=" * 60)
    print("  Franka Episode Replay")
    print(f"  Episode:    {npz_path.parent}")
    print(f"  Frames:     {len(actions)}")
    print(f"  Duration:   {duration:.2f} s")
    print(f"  Speed:      {args.speed:.2f}x")
    print(f"  Start pose: {np.array2string(actions[0, :6], precision=3)}")
    print(f"  End pose:   {np.array2string(actions[-1, :6], precision=3)}")
    print("  Controls:   Space pause/resume | Right/N single-step | Esc interrupt")
    print("=" * 60)

    if args.dry_run:
        return

    robot = FrankaClient(args.robot_ip, args.robot_port)
    controller = ReplayKeyboardController()
    try:
        robot.start_cartesian_impedance()

        if not args.skip_home:
            reset_to_home(robot, args.frequency)

        if not args.skip_move_to_start:
            print("[Replay] Moving to first recorded pose ...")
            move_to_pose(robot, actions[0, :6], args.move_to_start_duration, args.frequency)

        apply_gripper_command(robot, actions[0, 6], args.gripper_threshold)
        time.sleep(0.5)

        print("[Replay] Starting episode replay ...")
        replay_episode(
            robot,
            timestamps,
            actions,
            args.speed,
            args.gripper_threshold,
            controller,
        )

        if args.hold_final > 0:
            robot.update_desired_ee_pose(actions[-1, :6])
            time.sleep(args.hold_final)
        print("[Replay] Done.")
    except ReplayInterrupted:
        print("[Replay] Stopped by user.")
    finally:
        controller.stop()
        try:
            robot.terminate_current_policy()
        except Exception:
            pass
        robot.close()


if __name__ == '__main__':
    main()