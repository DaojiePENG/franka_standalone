#!/usr/bin/env python3
"""
LeapBot Inference HTTP Client — runs on the NUC / robot control host.

Sends camera frames and robot proprioception to the GPU-side inference server
and receives an action chunk in return.

Usage (standalone test):
    python leapbot_client.py --server_ip 192.168.1.100 --server_port 8000
"""
import argparse
import base64
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
import requests


class LeapbotClient:
    """Lightweight HTTP client for the LeapBot inference server."""

    def __init__(self, server_ip: str, server_port: int = 8000,
                 timeout: float = 5.0):
        """
        Args:
            server_ip:   IP address of the GPU inference server.
            server_port: Port the inference server listens on.
            timeout:     HTTP request timeout in seconds.
        """
        self.base_url = f"http://{server_ip}:{server_port}"
        self.timeout = timeout
        self._session = requests.Session()

    def health(self) -> bool:
        """Check if the server is alive."""
        try:
            r = self._session.get(
                f"{self.base_url}/health", timeout=self.timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def ready(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        try:
            r = self._session.get(
                f"{self.base_url}/ready", timeout=self.timeout)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def infer(self,
              global_image: np.ndarray,
              wrist_image: np.ndarray,
              proprio_7d: np.ndarray,
              task: str) -> Optional[Dict]:
        """
        Send one observation and receive an action chunk.

        Args:
            global_image: RGB uint8 [H,W,3] — third-person view
                         (mapped from zed_0 by default)
            wrist_image:  RGB uint8 [H,W,3] — gripper first-person view
                         (mapped from fisheye by default)
            proprio_7d:   float32 [7] — [x,y,z,rx,ry,rz,gripper_width]
            task:         task identifier string

        Returns:
            dict with keys:
                'action_chunk' : np.ndarray [32,7] float32
                'latency_ms'   : float (server-side inference time)
            or None on failure.
        """
        payload = {
            "global_image": _encode_image_jpeg(global_image),
            "wrist_image":  _encode_image_jpeg(wrist_image),
            "proprio_7d":   proprio_7d.tolist(),
            "task":         task,
        }
        try:
            r = self._session.post(
                f"{self.base_url}/infer",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
            return {
                "action_chunk": np.array(data["action_chunk"], dtype=np.float32),
                "latency_ms":   float(data["latency_ms"]),
            }
        except requests.RequestException as e:
            print(f"[LeapbotClient] Request failed: {e}")
            return None

    def close(self):
        self._session.close()


# ──────────────────────────── Helpers ─────────────────────────────────────────

def _encode_image_jpeg(rgb: np.ndarray, quality: int = 85) -> str:
    """Encode an RGB uint8 numpy array to a base64 JPEG string.

    Using JPEG keeps the payload small (~30-50 KB for 640x480) which reduces
    network latency significantly compared to raw PNG (~300 KB) or uncompressed
    data (~900 KB). Quality 85 is visually lossless for policy inference.
    """
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr,
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ──────────────────────────── CLI Test ────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Test the LeapBot inference server connection")
    parser.add_argument("--server_ip", required=True)
    parser.add_argument("--server_port", type=int, default=8000)
    parser.add_argument("--task", default="move_objects_into_box")
    args = parser.parse_args()

    client = LeapbotClient(args.server_ip, args.server_port)

    print(f"Server: {args.server_ip}:{args.server_port}")
    print(f"Health: {client.health()}")
    print(f"Ready : {client.ready()}")

    if not client.ready():
        print("Server not ready — aborting test")
        client.close()
        return

    # Send a dummy black image + zero proprio
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    proprio = np.zeros(7, dtype=np.float32)
    proprio[6] = 0.08  # open gripper

    print("\nSending dummy inference request ...")
    result = client.infer(dummy_img, dummy_img, proprio, args.task)
    if result is not None:
        chunk = result["action_chunk"]
        print(f"  action_chunk shape : {chunk.shape}")
        print(f"  action_chunk dtype : {chunk.dtype}")
        print(f"  action[0]          : {np.array2string(chunk[0], precision=4)}")
        print(f"  server latency     : {result['latency_ms']:.1f} ms")
    else:
        print("  Inference failed")

    client.close()


if __name__ == "__main__":
    main()
