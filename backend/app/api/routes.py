import hashlib
import json
import os
import threading
import time
from datetime import datetime

import cv2
import jwt
import numpy as np
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.auth_routes import get_current_user
from app.core.config import JWT_SECRET, RUNS_DIR, REPORTS_DIR
from app.core.database import log_output
from app.services.eval import run_inference

# Routes that need auth via Bearer token
router = APIRouter(dependencies=[Depends(get_current_user)])

# Stream endpoint uses a ?token= query param because <img src> can't send headers
stream_router = APIRouter()

# ── Prompts ───────────────────────────────────────────────────────────────────

ANALYZE_PROMPT = (
    "Describe what you see in this image. "
    "Identify the environment, the main subjects or objects present, and any notable details. "
    "Be specific and factual in two to three sentences."
)

INSPECT_PROMPT = (
    "Inspect this image for safety and maintenance concerns relevant to transportation or infrastructure. "
    "Look for: structural damage such as cracks, corrosion, or dents; "
    "hazards such as obstructions, spills, or unsecured items; "
    "damaged or missing safety equipment; and unusual environmental conditions. "
    "Describe any issues found and where in the image they appear. "
    "If nothing concerning is visible, say 'No issues detected.'"
)

# ── Shared app state (injected by server.py at startup) ───────────────────────

_state: dict = {}
_infer_lock = threading.Lock()

SNAPSHOTS_DIR  = os.path.join(os.path.dirname(RUNS_DIR), "snapshots")
RECORDINGS_DIR = os.path.join(os.path.dirname(RUNS_DIR), "recordings")
AUTOSCAN_DIR   = os.path.join(os.path.dirname(RUNS_DIR), "auto_scan")
SESSION_LOG    = os.path.join(os.path.dirname(RUNS_DIR), "session_log.json")


def init_state(state: dict):
    """Called by server.py lifespan to inject shared state."""
    _state.update(state)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _require_frame() -> bytes:
    jpeg = _state.get("camera") and _state["camera"].get_latest_jpeg()
    if not jpeg:
        raise HTTPException(503, "Camera not ready — no frames captured yet.")
    return jpeg


def _require_model():
    if not _state.get("llm"):
        raise HTTPException(503, "Model is still loading, please wait ~10 seconds.")


def _run_inference_locked(prompt: str, save_dir: str | None = None, source: str = "Analyze") -> dict:
    """Acquire inference lock, run inference, save a clean JPEG frame, release.
    Returns result dict including 'file_path' for the saved frame."""
    _require_model()
    if not _infer_lock.acquire(blocking=False):
        raise HTTPException(409, "Another inference is already running. Try again shortly.")
    try:
        jpeg = _require_frame()
        # Fingerprint the exact bytes we are about to analyze. Two consecutive
        # inferences with the same hash means the camera handed us identical
        # frames, which points at a frozen camera rather than the model.
        frame_hash = hashlib.md5(jpeg).hexdigest()[:12]
        prev_hash = _state.get("last_frame_hash")
        if prev_hash is not None and prev_hash == frame_hash:
            print(
                f"[camera] WARNING: two consecutive inferences used identical frame "
                f"bytes (hash {frame_hash}). The camera delivered the same frame twice; "
                f"a frozen or stalled camera is the likely cause, not the model."
            )
        _state["last_frame_hash"] = frame_hash
        text, tokens, elapsed = run_inference(_state["llm"], jpeg, prompt)
        result = {
            "text": text,
            "tokens": tokens,
            "elapsed_s": round(elapsed, 2),
            "timestamp": datetime.now().isoformat(),
            "frame_hash": frame_hash,
        }
        _state["last_result"] = {**result, "prompt": prompt, "source": source, "jpeg": jpeg}
        target_dir = save_dir or RUNS_DIR
        os.makedirs(target_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Save clean frame — Q&A is stored in the DB and shown in the Library preview
        file_path = os.path.join(target_dir, f"frame_{ts}.jpg")
        with open(file_path, "wb") as f:
            f.write(jpeg)
        return {**result, "file_path": file_path}
    finally:
        _infer_lock.release()


# ── Camera stream ─────────────────────────────────────────────────────────────

def _mjpeg_generator():
    while True:
        jpeg = _state.get("camera") and _state["camera"].get_latest_jpeg()
        if jpeg:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(1 / 30)


@stream_router.get("/stream")
def stream(token: str | None = None):
    if not token:
        raise HTTPException(401, "Token required.")
    try:
        jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid or expired token.")
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Status ────────────────────────────────────────────────────────────────────

def _memory_mb() -> int:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return -1


@router.get("/status")
def status():
    return {
        "model_ready": _state.get("llm") is not None,
        "camera_ready": (
            _state.get("camera") is not None
            and _state["camera"].get_latest_jpeg() is not None
        ),
        "recording": _state.get("recording", False),
        "recording_file": _state.get("recording_file"),
        "autoscan": _state.get("autoscan", False),
        "autoscan_interval_s": _state.get("autoscan_interval", 10),
        "memory_available_mb": _memory_mb(),
        "frame_age_s": (
            round(age, 1)
            if _state.get("camera") is not None
            and (age := _state["camera"].frame_age_s()) is not None
            else None
        ),
        "last_result": (
            {k: v for k, v in _state["last_result"].items() if k != "jpeg"}
            if _state.get("last_result")
            else None
        ),
    }


# ── Inference endpoints ───────────────────────────────────────────────────────

@router.post("/analyze")
def analyze(user: dict = Depends(get_current_user)):
    result = _run_inference_locked(ANALYZE_PROMPT, source="Analyze")
    file_path = result.pop("file_path")
    log_output(
        type="analyze",
        timestamp=result["timestamp"],
        file_path=file_path,
        prompt=ANALYZE_PROMPT,
        response=result["text"],
        tokens=result["tokens"],
        elapsed_s=result["elapsed_s"],
        user_id=int(user["sub"]),
        frame_hash=result.get("frame_hash"),
    )
    return result


@router.post("/inspect")
def inspect(user: dict = Depends(get_current_user)):
    result = _run_inference_locked(INSPECT_PROMPT, source="Inspect")
    file_path = result.pop("file_path")
    log_output(
        type="inspect",
        timestamp=result["timestamp"],
        file_path=file_path,
        prompt=INSPECT_PROMPT,
        response=result["text"],
        tokens=result["tokens"],
        elapsed_s=result["elapsed_s"],
        user_id=int(user["sub"]),
        frame_hash=result.get("frame_hash"),
    )
    return result


# ── Snapshot ──────────────────────────────────────────────────────────────────

@router.post("/snapshot")
def snapshot(user: dict = Depends(get_current_user)):
    jpeg = _require_frame()
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SNAPSHOTS_DIR, f"snapshot_{ts}.jpg")
    with open(path, "wb") as f:
        f.write(jpeg)
    timestamp = datetime.now().isoformat()
    log_output(
        type="snapshot",
        timestamp=timestamp,
        file_path=path,
        prompt=None,
        response=None,
        tokens=None,
        elapsed_s=None,
        user_id=int(user["sub"]),
    )
    return {"saved": path, "timestamp": ts}


# ── Recording ─────────────────────────────────────────────────────────────────

def _recording_loop(video_path: str, fps: int = 15):
    """Record camera frames to a video file at the requested fps.
    Frame dimensions are detected from the first captured frame so they
    always match what the camera is actually delivering."""
    writer = None
    interval = 1.0 / fps
    last_write = 0.0

    while _state.get("recording"):
        now = time.time()
        elapsed_since_last = now - last_write
        if elapsed_since_last < interval:
            time.sleep(interval - elapsed_since_last)
            continue

        jpeg = _state.get("camera") and _state["camera"].get_latest_jpeg()
        if jpeg:
            frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                if writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
                if writer.isOpened():
                    writer.write(frame)
                    last_write = time.time()

    if writer is not None and writer.isOpened():
        writer.release()


@router.post("/record/start")
def record_start(user: dict = Depends(get_current_user)):
    if _state.get("recording"):
        raise HTTPException(409, "Recording is already active.")
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(RECORDINGS_DIR, f"session_{ts}.mp4")
    _state["recording"] = True
    _state["recording_file"] = video_path
    _state["recording_user_id"] = int(user["sub"])
    t = threading.Thread(target=_recording_loop, args=(video_path,), daemon=True)
    t.start()
    return {"recording": True, "session_file": video_path}


@router.post("/record/stop")
def record_stop():
    if not _state.get("recording"):
        raise HTTPException(409, "No recording is active.")
    _state["recording"] = False
    video_path = _state.get("recording_file", "")
    user_id = _state.get("recording_user_id")
    _state["recording_file"] = None
    _state["recording_user_id"] = None
    if user_id and video_path:
        log_output(
            type="record",
            timestamp=datetime.now().isoformat(),
            file_path=video_path,
            prompt=None,
            response=None,
            tokens=None,
            elapsed_s=None,
            user_id=user_id,
        )
    return {"recording": False, "session_file": video_path}


# ── Auto-scan ─────────────────────────────────────────────────────────────────

def _autoscan_loop(interval: int, session_dir: str):
    while _state.get("autoscan"):
        try:
            result = _run_inference_locked(ANALYZE_PROMPT, save_dir=session_dir, source="Auto-Scan")
            file_path = result.pop("file_path")
            user_id = _state.get("autoscan_user_id")
            if user_id:
                log_output(
                    type="autoscan",
                    timestamp=result["timestamp"],
                    file_path=file_path,
                    prompt=ANALYZE_PROMPT,
                    response=result["text"],
                    tokens=result["tokens"],
                    elapsed_s=result["elapsed_s"],
                    user_id=user_id,
                    frame_hash=result.get("frame_hash"),
                )
        except HTTPException:
            pass  # busy or camera not ready — skip this tick
        time.sleep(interval)


@router.post("/autoscan/start")
def autoscan_start(interval: int = 10, user: dict = Depends(get_current_user)):
    if _state.get("autoscan"):
        raise HTTPException(409, "Auto-scan is already running.")
    _require_model()
    ts = datetime.now().strftime("%m%d%y_%H%M")
    session_dir = os.path.join(AUTOSCAN_DIR, f"scan_{ts}")
    os.makedirs(session_dir, exist_ok=True)
    _state["autoscan"] = True
    _state["autoscan_interval"] = interval
    _state["autoscan_user_id"] = int(user["sub"])
    t = threading.Thread(target=_autoscan_loop, args=(interval, session_dir), daemon=True)
    _state["autoscan_thread"] = t
    t.start()
    return {"autoscan": True, "interval_s": interval, "session_dir": session_dir}


@router.post("/autoscan/stop")
def autoscan_stop():
    if not _state.get("autoscan"):
        raise HTTPException(409, "Auto-scan is not running.")
    _state["autoscan"] = False
    _state["autoscan_thread"] = None
    _state["autoscan_user_id"] = None
    return {"autoscan": False}


# ── Flag ──────────────────────────────────────────────────────────────────────

@router.post("/flag")
def flag(user: dict = Depends(get_current_user)):
    last = _state.get("last_result")
    jpeg = _require_frame()
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    frame_path = os.path.join(SNAPSHOTS_DIR, f"flagged_{ts}.jpg")
    with open(frame_path, "wb") as f:
        f.write(jpeg)

    # Carry over the last inference result as the flag's context
    prompt   = last.get("prompt")   if last else None
    response = last.get("text")     if last else None
    tokens   = last.get("tokens")   if last else None
    elapsed  = last.get("elapsed_s") if last else None

    log_output(
        type="flag",
        timestamp=datetime.now().isoformat(),
        file_path=frame_path,
        prompt=prompt,
        response=response,
        tokens=tokens,
        elapsed_s=elapsed,
        user_id=int(user["sub"]),
    )

    entry = {
        "timestamp": ts,
        "flagged": True,
        "frame": frame_path,
        "last_inference": (
            {k: v for k, v in last.items() if k != "jpeg"} if last else None
        ),
    }

    # Append to session log
    log = []
    if os.path.exists(SESSION_LOG):
        try:
            with open(SESSION_LOG) as f:
                log = json.load(f)
        except (json.JSONDecodeError, OSError):
            log = []
    log.append(entry)
    with open(SESSION_LOG, "w") as f:
        json.dump(log, f, indent=2)

    return entry
