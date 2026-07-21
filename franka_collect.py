#!/usr/bin/env python3
"""
Franka 上位机 — Keyboard teleop + multi-camera data collection.

Connects to franka_server.py via ZeroRPC, streams L515 RGB-D + fisheye + ZED
stereo images, and records episodes to disk (numpy .npz per episode). Each ZED
runs in its OWN spawned process (pyzed is installed in the umi env) and
publishes the latest left-view RGB frame to the parent via shared memory, so
pyzed grab()/retrieve_image() never hold the parent GIL and can never starve
the gevent/zerorpc control client. ZEDs are auto-detected from the bus; see
ZEDCamera and _zed_worker.

Usage:
    python franka_collect.py -o ./data
    python franka_collect.py -o ./data --robot_ip 192.168.3.2

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
    C     Start recording episode
    V     Stop recording & save episode
    B     Drop (discard) current episode
    H     Reset to home
    Esc   Quit
"""
import argparse
import time
import threading
import subprocess
import multiprocessing as mp
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np
import scipy.spatial.transform as st
import zerorpc

from config import (
    ROBOT_IP, ROBOT_PORT, CONTROL_FREQUENCY,
    POS_SPEED, ROT_SPEED, MAX_GRIPPER_WIDTH, GRIPPER_SPEED,
    EE_HOME_POSE, HOME_MOVE_DURATION,
    FRANKA_HOME_JOINTS, JOINTS_HOME_DURATION,
    KX_DEFAULT, KXD_DEFAULT,
    TX_FLANGE_TIP, TX_TIP_FLANGE,
    L515_SERIALS, FISHEYE_USB_ID, FISHEYE_RESOLUTION,
    ZED_SERIALS, ZED_RESOLUTION, ZED_FPS,
)


# ======================== Pose Utilities ========================

def pose_to_mat(pose):
    mat = np.eye(4)
    mat[:3, 3] = pose[:3]
    mat[:3, :3] = st.Rotation.from_rotvec(pose[3:]).as_matrix()
    return mat

def mat_to_pose(mat):
    pos = mat[:3, 3]
    rot = st.Rotation.from_matrix(mat[:3, :3]).as_rotvec()
    return np.concatenate([pos, rot])


def precise_wait(t_end, slack_time=0.001):
    """Sleep until monotonic time reaches t_end, spinning for the final slack
    to minimise jitter (equivalent to umi.common.precise_sleep.precise_wait)."""
    t_wait = t_end - time.monotonic()
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time.monotonic() < t_end:
            pass


# ======================== ZeroRPC Client ========================

class FrankaClient:
    def __init__(self, ip, port):
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
# them on the shared control client would stall the 20 Hz pose loop for the
# whole motion, so (mirroring demo_franka_keyboard_wrist_L515_dual_zed.py's
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


# ======================== Cameras ========================

class L515Camera:
    """Thread-based Intel RealSense L515 camera reader."""

    # (color_w, color_h, depth_w, depth_h, fps) — ordered from preferred to fallback
    _PROFILES = [
        (960, 540, 640, 480, 30),   # matches l515_camera.py defaults
        (960, 540, 640, 480, 15),
        (640, 480, 640, 480, 30),
        (640, 480, 640, 480, 15),
        (320, 240, 320, 240, 30),
    ]

    def __init__(self, serial: str,
                 color_width=960, color_height=540,
                 depth_width=640, depth_height=480,
                 fps=30):
        import pyrealsense2 as rs
        self.serial = serial
        self._rs = rs
        self._lock = threading.Lock()
        self._color = None
        self._depth = None
        self._running = False

        self._pipeline = rs.pipeline()
        self._align = rs.align(rs.stream.color)

        profiles = [(color_width, color_height, depth_width, depth_height, fps)] + self._PROFILES
        started = False
        for cw, ch, dw, dh, f in profiles:
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color, cw, ch, rs.format.rgb8, f)
            cfg.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, f)
            if cfg.can_resolve(self._pipeline):
                self._pipeline.start(cfg)
                print(f"[L515 {serial}] opened: color {cw}x{ch}, depth {dw}x{dh} @ {f}fps")
                started = True
                break
        if not started:
            raise RuntimeError(
                f"No supported profile found. Check USB 3.0 connection and firmware.")

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=2000)
                aligned = self._align.process(frames)
                color = np.asarray(aligned.get_color_frame().get_data())
                depth = np.asarray(aligned.get_depth_frame().get_data())
                with self._lock:
                    self._color = color
                    self._depth = depth
            except Exception as e:
                print(f"[L515 {self.serial}] frame error: {e}")

    def get(self):
        """Return (color_rgb_uint8, depth_uint16) or (None, None)."""
        with self._lock:
            return (self._color.copy() if self._color is not None else None,
                    self._depth.copy() if self._depth is not None else None)

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self._pipeline.stop()


class FisheyeCamera:
    """Thread-based USB camera reader."""

    def __init__(self, device, width=640, height=480):
        self._lock = threading.Lock()
        self._color = None
        self._running = False

        # Force the V4L2 backend: a plain VideoCapture(str) opens via FFMPEG on
        # OpenCV 4.13, where CAP_PROP_FRAME_WIDTH/HEIGHT are silently ignored and
        # the capture keeps the driver's residual format. V4L2 + an explicit
        # FOURCC forces a fresh negotiation to the requested resolution.
        self._cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {device}")

        # Prefer YUYV (this camera natively supports YUYV 640x480@30); only fall
        # back to MJPG if the driver refuses the requested size under YUYV.
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, 30)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w != width or actual_h != height:
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = (int(self._cap.get(cv2.CAP_PROP_FOURCC)) & 0xFFFFFFFF
                  ).to_bytes(4, 'little').decode(errors='ignore')
        print(f"[Camera] Fisheye {device}: {fourcc} {actual_w}x{actual_h}")

        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            ret, frame = self._cap.read()
            if ret:
                with self._lock:
                    self._color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def get(self):
        """Return color_rgb_uint8 or None."""
        with self._lock:
            return self._color.copy() if self._color is not None else None

    def stop(self):
        self._running = False
        self._thread.join(timeout=2)
        self._cap.release()


def detect_zed_serials():
    """Return connected ZED serials in ascending order (empty on none/error)."""
    try:
        import pyzed.sl as sl
    except ImportError as e:
        raise RuntimeError(
            "pyzed is not importable in this environment; ZED capture requires "
            "pyzed installed in umi (e.g. `python -c 'import pyzed.sl'`)") from e
    serials = []
    for dev in sl.Camera.get_device_list():
        try:
            serials.append(int(dev.serial_number))
        except (TypeError, ValueError):
            pass
    return sorted(s for s in serials if s)


_ZED_RESOLUTION_MAP = {
    'HD2K': (2208, 1242), 'HD1080': (1920, 1080),
    'HD720': (1280, 720), 'VGA': (672, 376),
}

# Child-process status codes shared with the parent via a spawn-context Value.
_ZED_OPENING, _ZED_READY, _ZED_FAILED = 0, 1, -1


def _zed_worker(serial, resolution, fps, shm_name, shape, lock, seq, status,
                stop_event):
    """ZED grab loop, run in a dedicated spawn process (top-level so it is
    picklable under the 'spawn' start method).

    Opens the camera with a FRESH pyzed/CUDA context in this process, then
    continuously grabs the left view and copies the latest RGB frame into the
    shared-memory buffer under ``lock`` (bumping ``seq``). All the blocking C
    pyzed calls and the GIL they hold stay in THIS process, so the parent's
    gevent hub (which drives the zerorpc control client) is never starved."""
    try:
        import pyzed.sl as sl
    except Exception:
        status.value = _ZED_FAILED
        return

    cam = None
    try:
        res_enum = getattr(sl.RESOLUTION, str(resolution).strip().upper(), None)
        if res_enum is None:
            status.value = _ZED_FAILED
            return
        cam = sl.Camera()
        init = sl.InitParameters()
        init.camera_resolution = res_enum
        init.camera_fps = int(fps)
        init.depth_mode = sl.DEPTH_MODE.NONE
        try:
            init.coordinate_units = sl.UNIT.METER
        except Exception:
            pass
        if serial not in (None, ''):
            init.set_from_serial_number(int(serial))
        if cam.open(init) != sl.ERROR_CODE.SUCCESS:
            try:
                cam.close()
            except Exception:
                pass
            status.value = _ZED_FAILED
            return
    except Exception:
        if cam is not None:
            try:
                cam.close()
            except Exception:
                pass
        status.value = _ZED_FAILED
        return

    # Attach to the parent's segment. Under 'spawn' the child shares the
    # parent's resource_tracker, so the parent (creator) is the SOLE owner: it
    # unlinks on stop() and the tracker reclaims the segment if the parent is
    # hard-killed. The child only close()s its view (never unlink) to avoid a
    # double-unregister.
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)
    runtime = sl.RuntimeParameters()
    left_mat = sl.Mat()
    status.value = _ZED_READY
    try:
        while not stop_event.is_set():
            if cam.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue
            cam.retrieve_image(left_mat, sl.VIEW.LEFT)
            arr = np.asarray(left_mat.get_data())
            if arr.ndim == 3 and arr.shape[2] == 4:
                rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
            else:
                rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            if rgb.shape != shape:
                rgb = cv2.resize(rgb, (shape[1], shape[0]))
            with lock:
                buf[:] = rgb
                seq.value += 1
    finally:
        try:
            cam.close()
        except Exception:
            pass
        try:
            shm.close()
        except Exception:
            pass


class ZEDCamera:
    """Process-isolated ZED stereo camera reader.

    Each ZED is opened in its OWN 'spawn' subprocess (_zed_worker), which owns
    the pyzed/CUDA context and does all grab()/retrieve_image() work; the latest
    left-view RGB frame is published to this parent via a shared-memory buffer.
    The parent never touches pyzed, so the blocking C calls / GIL contention
    that previously starved the gevent-based zerorpc control client are gone.

    Public contract is unchanged from the old thread-based reader: construct by
    serial (+ resolution/fps), loop-free .get() returning the latest left RGB
    uint8 frame (or None), and .stop() that cleanly tears down the child."""

    _RESOLUTION_MAP = _ZED_RESOLUTION_MAP

    def __init__(self, serial=None, resolution='HD720', fps=30,
                 open_timeout=30.0):
        self.serial = serial
        res_name = str(resolution).strip().upper()
        if res_name not in self._RESOLUTION_MAP:
            raise ValueError(f"[ZED] unknown resolution {resolution!r}; "
                             f"expected one of {list(self._RESOLUTION_MAP)!r}")
        w, h = self._RESOLUTION_MAP[res_name]
        self._shape = (h, w, 3)
        self._proc = None
        self._shm = None

        # spawn (NOT fork): the child must initialise pyzed/CUDA fresh; forking
        # after any CUDA use in the parent can hang or crash.
        ctx = mp.get_context('spawn')
        self._lock = ctx.Lock()
        self._seq = ctx.Value('L', 0)
        self._status = ctx.Value('i', _ZED_OPENING)
        self._stop_event = ctx.Event()
        self._last_seq = 0

        nbytes = int(np.prod(self._shape))
        self._shm = shared_memory.SharedMemory(create=True, size=nbytes)
        self._buf = np.ndarray(self._shape, dtype=np.uint8, buffer=self._shm.buf)

        self._proc = ctx.Process(
            target=_zed_worker,
            args=(serial, res_name, int(fps), self._shm.name, self._shape,
                  self._lock, self._seq, self._status, self._stop_event),
            daemon=True)
        self._proc.start()

        deadline = time.monotonic() + float(open_timeout)
        while time.monotonic() < deadline:
            child_status = self._status.value
            if child_status == _ZED_READY:
                break
            if child_status == _ZED_FAILED or not self._proc.is_alive():
                self._cleanup()
                raise RuntimeError(
                    f"[ZED {serial}] child process failed to open camera "
                    f"(pyzed missing or camera busy)")
            time.sleep(0.05)
        else:
            self._cleanup()
            raise RuntimeError(
                f"[ZED {serial}] open timed out after {open_timeout:.0f}s")
        print(f"[ZED {serial}] opened (spawn process): {w}x{h} @ {int(fps)}fps")

    def get(self):
        """Return the latest left color_rgb_uint8 frame or None (non-blocking:
        just a locked memcpy out of shared memory, no pyzed calls)."""
        with self._lock:
            if self._seq.value == 0:
                return None
            return self._buf.copy()

    def _cleanup(self):
        try:
            self._stop_event.set()
        except Exception:
            pass
        if self._proc is not None:
            self._proc.join(timeout=2)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2)
            self._proc = None
        if self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
            try:
                self._shm.unlink()
            except Exception:
                pass
            self._shm = None

    def stop(self):
        self._cleanup()


def find_video_device_by_usb_id(vendor_id: str, product_id: str):
    """Locate /dev/videoN for a USB camera by vendor:product."""
    try:
        out = subprocess.run(
            ['v4l2-ctl', '--list-devices'],
            capture_output=True, text=True, timeout=5).stdout
        if not out.strip():
            return None
        current = None
        for line in out.split('\n'):
            if not line.startswith('\t'):
                current = line.strip()
            elif '/dev/video' in line and current:
                vpath = line.strip()
                try:
                    udev = subprocess.run(
                        ['udevadm', 'info', '--query=all', vpath],
                        capture_output=True, text=True, timeout=5)
                    if (udev.returncode == 0
                            and f'ID_VENDOR_ID={vendor_id}' in udev.stdout
                            and f'ID_MODEL_ID={product_id}' in udev.stdout):
                        return vpath
                except Exception:
                    pass
    except Exception as e:
        print(f"[Camera] find device error: {e}")
    return None


# ======================== Keyboard Handler ========================

class KeyboardTeleop:
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
        self.record_start = False
        self.record_stop = False
        self.record_drop = False

        # Lazy pynput import (needs an X display at import time). Deferring it
        # here keeps the module importable headlessly, which matters because the
        # spawned ZED child processes re-import this module and must NOT require
        # a display just to grab frames.
        from pynput import keyboard as pynput_keyboard
        self._pk = pynput_keyboard
        self._listener = pynput_keyboard.Listener(
            on_press=self._press, on_release=self._release)
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
        elif c == 'z':
            if time.monotonic() - self._z_last_release > self._gripper_debounce:
                self.gripper_close_req = True
        elif c == 'x':
            if time.monotonic() - self._x_last_release > self._gripper_debounce:
                self.gripper_open_req = True
        elif c == 'h':
            self.home_requested = True
        elif c == 'c':
            self.record_start = True
        elif c == 'v':
            self.record_stop = True
        elif c == 'b':
            self.record_drop = True
        if key == self._pk.Key.esc:
            self.quit_requested = True

    def _release(self, key):
        if key in (self._pk.Key.shift, self._pk.Key.shift_r):
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


# ======================== Episode Recorder ========================

class EpisodeRecorder:
    """Accumulates data in-memory; saves to .npz on finish."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._ep_idx = self._next_episode_idx()
        self.reset()

    def _next_episode_idx(self):
        existing = list(self.output_dir.glob('episode_*'))
        if not existing:
            return 0
        indices = []
        for p in existing:
            try:
                indices.append(int(p.stem.split('_')[1]))
            except (ValueError, IndexError):
                pass
        return max(indices) + 1 if indices else 0

    def reset(self):
        self.timestamps = []
        self.actions = []          # (7,) pose6d + gripper
        self.robot_states = []     # (7,) pose6d + gripper_width
        self.joint_positions = []  # (7,)
        self.camera_colors = {}    # cam_name -> list of rgb frames
        self.camera_depths = {}    # cam_name -> list of depth frames

    @property
    def is_empty(self):
        return len(self.timestamps) == 0

    def add(self, timestamp, action, robot_state, joint_pos,
            camera_frames: dict):
        """
        camera_frames: {cam_name: {'color': rgb, 'depth': depth_or_None}}
        """
        self.timestamps.append(timestamp)
        self.actions.append(action.copy())
        self.robot_states.append(robot_state.copy())
        self.joint_positions.append(joint_pos.copy())
        for name, frames in camera_frames.items():
            self.camera_colors.setdefault(name, []).append(frames['color'])
            if frames.get('depth') is not None:
                self.camera_depths.setdefault(name, []).append(frames['depth'])

    def save(self):
        """Save episode as .npz + camera video directories."""
        ep_dir = self.output_dir / f'episode_{self._ep_idx:04d}'
        ep_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            ep_dir / 'robot_data.npz',
            timestamps=np.array(self.timestamps),
            actions=np.array(self.actions),
            robot_states=np.array(self.robot_states),
            joint_positions=np.array(self.joint_positions),
        )

        for cam_name, frames in self.camera_colors.items():
            cam_dir = ep_dir / cam_name
            cam_dir.mkdir(exist_ok=True)
            for i, frame in enumerate(frames):
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(cam_dir / f'color_{i:05d}.jpg'), bgr)

        for cam_name, frames in self.camera_depths.items():
            cam_dir = ep_dir / cam_name
            cam_dir.mkdir(exist_ok=True)
            np.savez_compressed(
                cam_dir / 'depth.npz',
                depth=np.array(frames))

        n = len(self.timestamps)
        print(f"[Recorder] Saved episode_{self._ep_idx:04d} ({n} frames) -> {ep_dir}")
        self._ep_idx += 1
        self.reset()

    def drop(self):
        print(f"[Recorder] Dropped episode ({len(self.timestamps)} frames)")
        self.reset()


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
    print("[Home] joints -> home ...")
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


def compose_camera_grid(vis_items, cell_w=480, cell_h=360, cols=2):
    """Tile (label, rgb_image) pairs into a BGR grid (``cols`` columns,
    ceil(n/cols) rows). Each image is letterboxed into a uniform cell (aspect
    preserved, black bars fill the remainder) so hstack/vstack never misalign
    on cameras of differing resolution; missing trailing slots are black. Draws
    the per-camera label into each cell. Returns the montage or None."""
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


def main():
    parser = argparse.ArgumentParser(
        description='Franka keyboard teleop + multi-camera data collection')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory for episodes')
    parser.add_argument('--robot_ip', default=ROBOT_IP)
    parser.add_argument('--robot_port', type=int, default=ROBOT_PORT)
    parser.add_argument('--frequency', type=int, default=CONTROL_FREQUENCY)
    parser.add_argument('--pos_speed', type=float, default=POS_SPEED)
    parser.add_argument('--rot_speed', type=float, default=ROT_SPEED)
    parser.add_argument('--no_l515', action='store_true',
                        help='Disable L515 cameras')
    parser.add_argument('--no_fisheye', action='store_true',
                        help='Disable fisheye camera')
    parser.add_argument('--no_zed', action='store_true',
                        help='Disable ZED cameras')
    parser.add_argument('--no_depth', action='store_true',
                        help='Do not record depth images')
    parser.add_argument('--init_home', action='store_true', default=True)
    args = parser.parse_args()

    dt = 1.0 / args.frequency
    # Send each pose command command_latency before the cycle boundary so the
    # cadence stays steady regardless of how long the pre-command work took.
    command_latency = 1.0 / 100.0
    # Visualization (imshow/resize/grid) is expensive; cap it well below the
    # control rate so drawing never gates the control loop.
    vis_fps = 15.0
    vis_interval = 1.0 / vis_fps

    # ---- Cameras ----
    l515_cams = []
    if not args.no_l515:
        for serial in L515_SERIALS:
            try:
                print(f"[Camera] Starting L515 {serial} ...")
                l515_cams.append(L515Camera(serial))
            except Exception as e:
                print(f"[Camera] L515 {serial} failed: {e}")

    fisheye_cam = None
    if not args.no_fisheye and FISHEYE_USB_ID:
        vid, pid = FISHEYE_USB_ID.split(':')
        dev = find_video_device_by_usb_id(vid, pid)
        if dev:
            try:
                print(f"[Camera] Starting fisheye at {dev} ...")
                fisheye_cam = FisheyeCamera(dev, *FISHEYE_RESOLUTION)
            except Exception as e:
                print(f"[Camera] Fisheye failed: {e}")
        else:
            print(f"[Camera] Fisheye USB {FISHEYE_USB_ID} not found")

    zed_cams = []
    if not args.no_zed:
        if ZED_SERIALS:
            zed_serials = [int(s) for s in ZED_SERIALS]
        else:
            try:
                zed_serials = detect_zed_serials()
            except Exception as e:
                zed_serials = []
                print(f"[Camera] ZED enumeration failed: {e}")
            if zed_serials:
                print(f"[Camera] Auto-detected {len(zed_serials)} ZED(s): "
                      f"{zed_serials}")
            else:
                print("[Camera] No ZED cameras detected; skipping ZED capture")
        for serial in zed_serials:
            try:
                print(f"[Camera] Starting ZED {serial} ...")
                zed_cams.append(ZEDCamera(
                    serial, resolution=ZED_RESOLUTION, fps=ZED_FPS))
            except Exception as e:
                print(f"[Camera] ZED {serial} failed: {e}")

    # ---- Robot ----
    robot = FrankaClient(args.robot_ip, args.robot_port)
    teleop = KeyboardTeleop(pos_speed=args.pos_speed, rot_speed=args.rot_speed)
    recorder = EpisodeRecorder(args.output)

    print("=" * 60)
    print("  Franka Data Collector (上位机 + cameras)")
    print(f"  Server:    {args.robot_ip}:{args.robot_port}")
    print(f"  L515:      {len(l515_cams)} cameras")
    print(f"  Fisheye:   {'yes' if fisheye_cam else 'no'}")
    print(f"  ZED:       {len(zed_cams)} cameras")
    print(f"  Freq:      {args.frequency} Hz")
    print(f"  Output:    {args.output}")
    print("=" * 60)
    print("  C = start recording, V = stop & save, B = drop")
    print("  WASD = move, Z = close gripper, X = open gripper,")
    print("  H = home, Esc = quit")
    print("=" * 60)

    try:
        robot.start_cartesian_impedance()
        if args.init_home:
            reset_to_home(robot, args.frequency)

        target_pose = robot.get_tip_pose()
        gripper_pos = MAX_GRIPPER_WIDTH
        robot.gripper_release()
        time.sleep(1.0)  # let cameras warm up

        is_recording = False
        last_print = 0.0
        last_vis = 0.0
        gripper_meas = gripper_pos

        # Drift-free fixed-schedule loop: each cycle boundary is derived from a
        # monotonic origin (t_start + i*dt) so a heavy iteration never
        # accumulates drift, and the pose command is sent at a steady phase
        # (t_cycle_end - command_latency) decoupled from camera / vis work.
        t_start = time.monotonic()
        iter_idx = 0

        while not teleop.quit_requested:
            t_cycle_end = t_start + (iter_idx + 1) * dt
            t_sample = t_cycle_end - command_latency
            ts = time.time()

            # ---- events ----
            did_block = False
            if teleop.record_start and not is_recording:
                teleop.record_start = False
                is_recording = True
                recorder.reset()
                print("\n>>> RECORDING STARTED")
            if teleop.record_stop and is_recording:
                teleop.record_stop = False
                is_recording = False
                if not recorder.is_empty:
                    recorder.save()
                    did_block = True
                print(">>> RECORDING STOPPED")
            if teleop.record_drop:
                teleop.record_drop = False
                if is_recording:
                    is_recording = False
                    recorder.drop()
                    print(">>> EPISODE DROPPED")
            if teleop.home_requested and not is_recording:
                teleop.home_requested = False
                reset_to_home(robot, args.frequency)
                target_pose = robot.get_tip_pose()
                gripper_pos = MAX_GRIPPER_WIDTH
                did_block = True

            # A blocking event (episode save / homing) ran the wall clock far
            # past the current schedule; re-anchor the origin so the loop does
            # not fire a burst of catch-up pose commands.
            if did_block:
                t_start = time.monotonic()
                iter_idx = 0
                continue

            # ---- cameras (non-blocking cached reads) ----
            # Only touch the camera buffers when a frame is needed this tick
            # (recording, or a throttled vis tick); the expensive compositing is
            # deferred to compose_camera_grid and gated by vis_interval so
            # drawing never gates the control cadence.
            now_m = time.monotonic()
            need_vis = (now_m - last_vis) >= vis_interval
            cam_frames = {}
            vis_items = []
            if is_recording or need_vis:
                for i, cam in enumerate(l515_cams):
                    color, depth = cam.get()
                    if color is not None:
                        cam_frames[f'l515_{i}'] = {
                            'color': color,
                            'depth': depth if not args.no_depth else None,
                        }
                        if need_vis:
                            vis_items.append((f'L515-{i}', color))
                if fisheye_cam is not None:
                    color = fisheye_cam.get()
                    if color is not None:
                        cam_frames['fisheye'] = {'color': color, 'depth': None}
                        if need_vis:
                            vis_items.append(('Fisheye', color))
                for i, cam in enumerate(zed_cams):
                    color = cam.get()
                    if color is not None:
                        cam_frames[f'zed_{i}'] = {'color': color, 'depth': None}
                        if need_vis:
                            vis_items.append((f'ZED-{i}', color))

            # ---- visualize (throttled to ~vis_fps, off the control cadence) --
            if need_vis and vis_items:
                canvas = compose_camera_grid(vis_items)
                if canvas is not None:
                    mh = canvas.shape[0]
                    rec_color = (0, 0, 255) if is_recording else (200, 200, 200)
                    label = f'Ep {recorder._ep_idx}'
                    if is_recording:
                        label += f' [REC {len(recorder.timestamps)} frames]'
                    cv2.putText(canvas, label, (10, mh - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, rec_color, 2)
                    pos_txt = (f'pos=[{target_pose[0]:.3f},{target_pose[1]:.3f},'
                               f'{target_pose[2]:.3f}]  gripper={gripper_meas:.3f}m '
                               f'(cmd {gripper_pos:.3f})')
                    cv2.putText(canvas, pos_txt, (10, mh - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.imshow('Franka Collect', canvas)
                    if cv2.waitKey(1) == 27:
                        break
                last_vis = now_m

            # ---- wait for the scheduled command phase, then command the arm ---
            precise_wait(t_sample)

            # ---- teleop ----
            dpos, drot_xyz = teleop.get_velocity(dt)
            target_pose[:3] += dpos
            drot = st.Rotation.from_euler('xyz', drot_xyz)
            target_pose[3:] = (drot * st.Rotation.from_rotvec(target_pose[3:])).as_rotvec()
            robot.update_desired_ee_pose(target_pose)

            # ---- gripper: full close / open, one dispatch per Z/X press edge -
            # Z = full force-close (grasp), X = full open (release). The DOWN-edge
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

            # REAL measured gripper width (server-side poller makes this a
            # non-blocking cached read); fall back to commanded on any error.
            try:
                gripper_meas = float(robot.get_gripper_state().get('width', gripper_pos))
            except Exception:
                gripper_meas = gripper_pos

            # ---- record ----
            if is_recording and cam_frames:
                action = np.zeros(7)
                action[:6] = target_pose
                action[6] = gripper_pos

                joints = robot.get_joint_positions()
                robot_state = np.zeros(7)
                robot_state[:6] = robot.get_tip_pose()
                robot_state[6] = gripper_meas

                recorder.add(ts, action, robot_state, joints, cam_frames)

            # ---- terminal print (2 Hz) ----
            now = time.monotonic()
            if now - last_print > 0.5:
                p = target_pose
                rec_flag = ' [REC]' if is_recording else ''
                print(f"\rpos=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]  "
                      f"gripper={gripper_meas:.3f}m (cmd {gripper_pos:.3f}){rec_flag}   ",
                      end='', flush=True)
                last_print = now

            # ---- frequency regulation (drift-free fixed schedule) ----
            precise_wait(t_cycle_end)
            iter_idx += 1

    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nShutting down ...")
        teleop.stop()
        cv2.destroyAllWindows()

        if is_recording and not recorder.is_empty:
            print("[Recorder] Saving in-progress episode ...")
            recorder.save()

        for cam in l515_cams:
            cam.stop()
        if fisheye_cam:
            fisheye_cam.stop()
        for cam in zed_cams:
            cam.stop()

        try:
            robot.terminate_current_policy()
        except Exception:
            pass
        robot.close()
        print("Done.")


if __name__ == '__main__':
    main()
