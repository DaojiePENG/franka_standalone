#!/usr/bin/env python3
"""
LeapBot Inference Server — runs on the GPU machine.

Loads the FastWAM model once at startup and exposes a synchronous HTTP
endpoint that the robot control host (NUC) calls each control cycle.

Protocol:
    POST /infer
    Request body (JSON):
        global_image   : base64-encoded RGB uint8 JPEG (third-person view)
        wrist_image    : base64-encoded RGB uint8 JPEG (gripper-first-person view)
        proprio_7d     : list of 7 floats  [x,y,z,rx,ry,rz,gripper_width]
        task           : str, e.g. "move_objects_into_box"

    Response (JSON):
        action_chunk   : list[list[float]], shape [32,7]
        latency_ms     : float, server-side inference time in milliseconds

Usage:
    python server.py --asset_root /path/to/assets --task move_objects_into_box
    python server.py --asset_root /path/to/assets --task move_objects_into_box --port 8000
"""
import argparse
import base64
import logging
import sys
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Server] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("leapbot_server")

# ──────────────────────────── Global State ────────────────────────────────────

engine = None  # FastWAMInference instance, created once at startup


def _decode_image_b64(b64_str: str) -> np.ndarray:
    """Decode a base64-encoded JPEG/PNG image to RGB uint8 numpy array."""
    raw = base64.b64decode(b64_str)
    buf = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode image (invalid JPEG/PNG data)")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ──────────────────────────── Request / Response ──────────────────────────────

class InferRequest(BaseModel):
    global_image: str        # base64-encoded JPEG
    wrist_image: str         # base64-encoded JPEG
    proprio_7d: List[float]  # [x,y,z,rx,ry,rz,gripper_width]
    task: str                # task identifier, must match loaded model


class InferResponse(BaseModel):
    action_chunk: List[List[float]]  # [32,7]
    latency_ms: float


# ──────────────────────────── FastAPI App ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook: model is loaded once and held in GPU memory."""
    global engine
    log.info("Server starting — model will be loaded during first /ready call")
    yield
    if engine is not None:
        log.info("Shutting down — releasing model")
        engine.close()
        engine = None


app = FastAPI(title="LeapBot Inference Server", lifespan=lifespan)


# ──────────────────────────── Endpoints ───────────────────────────────────────

@app.get("/health")
async def health():
    """Basic liveness probe — always returns 200."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness probe — returns 200 only if the model is loaded and ready."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return {"status": "ready", "task": engine.task_id}


@app.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    """
    Run one forward pass: images + proprio  →  action chunk [32,7].

    The model inference is synchronous (blocks the worker thread) because
    CUDA kernels cannot be awaited. FastAPI runs each endpoint in a thread
    pool by default when the function is defined as `async def` + uses
    blocking calls, so concurrent requests are handled safely.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    if req.task != engine.task_id:
        raise HTTPException(
            status_code=400,
            detail=f"Task mismatch: server loaded '{engine.task_id}', "
                   f"request has '{req.task}'",
        )

    try:
        global_rgb = _decode_image_b64(req.global_image)
        wrist_rgb  = _decode_image_b64(req.wrist_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image decode error: {e}")

    proprio = np.array(req.proprio_7d, dtype=np.float32)
    if proprio.shape != (7,):
        raise HTTPException(
            status_code=400,
            detail=f"proprio_7d must have 7 elements, got {proprio.shape[0]}",
        )

    try:
        t0 = time.perf_counter()
        action_chunk = engine.infer(
            images={"global_image": global_rgb, "wrist_image": wrist_rgb},
            states=proprio,
            task=req.task,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
    except Exception as e:
        log.error("Inference failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    log.debug("Inference done: %.1f ms, action[0]=%s",
              latency_ms, np.array2string(action_chunk[0], precision=4))

    return InferResponse(
        action_chunk=action_chunk.tolist(),
        latency_ms=round(latency_ms, 2),
    )


# ──────────────────────────── Entry Point ─────────────────────────────────────

def _load_engine(args):
    """Load the FastWAM engine into the global `engine` variable."""
    global engine
    # Ensure the inference package is importable
    infer_root = args.infer_root
    if infer_root and infer_root not in sys.path:
        sys.path.insert(0, infer_root)

    from fastwam_infer import FastWAMInference

    log.info("Loading FastWAM model ...")
    log.info("  asset_root : %s", args.asset_root)
    log.info("  task       : %s", args.task)
    log.info("  device     : %s", args.device)
    log.info("  backend    : %s", args.backend)

    fisheye_undistorters = None
    if args.enable_fisheye_undistortion:
        from fastwam_infer import make_fastwam_fisheye_undistorter
        fisheye_undistorters = {
            "wrist_image": make_fastwam_fisheye_undistorter(),
        }
        log.info("  fisheye    : enabled (wrist_image KB4 undistortion)")

    engine = FastWAMInference(
        asset_root=args.asset_root,
        task_id=args.task,
        device=args.device,
        backend=args.backend,
        output_mode="delta_7d",
        convert_proprio_7d=True,
        enable_fisheye_undistortion=args.enable_fisheye_undistortion,
        fisheye_undistorters=fisheye_undistorters,
    )
    log.info("Model loaded successfully!")
    log.info("  action_horizon : %d", engine.action_horizon)
    log.info("  action_dim     : %d", engine.action_dim)
    log.info("  image_hw       : %s", engine.image_hw)


def main():
    parser = argparse.ArgumentParser(description="LeapBot Inference Server")
    parser.add_argument(
        "--asset_root", required=True,
        help="Path to the FastWAM asset root (checkpoints, configs, stats, T5 cache)",
    )
    parser.add_argument(
        "--task", required=True,
        help="Task identifier, e.g. 'move_objects_into_box'",
    )
    parser.add_argument(
        "--infer_root",
        default=None,
        help="Path to the LeapBot-inference-only directory "
             "(added to sys.path for fastwam_infer imports). "
             "If None, assumes fastwam_infer is already importable.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", default="tensorrt",
                        choices=["tensorrt", "pytorch", "auto"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--enable_fisheye_undistortion", action="store_true", default=False,
        help="Enable KB4 fisheye undistortion for the wrist_image view",
    )
    args = parser.parse_args()

    _load_engine(args)

    import uvicorn
    log.info("Starting uvicorn on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
