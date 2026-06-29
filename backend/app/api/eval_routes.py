"""
Evaluation routes for the web interface (Phase 12).

Runs the 5-prompt evaluation suite against the live camera feed in a
background thread and streams progress via GET /eval/status polling.
Results and annotated frames are saved under output/eval reports/.
"""

import json
import os
import re
import threading
from datetime import datetime
from urllib.parse import quote

import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.api.auth_routes import get_current_user, require_admin
from app.api.routes import _infer_lock, _state
from app.core.config import BASELINE_PATH, JWT_SECRET, REPORTS_DIR
from app.services.eval import (
    PROMPTS,
    generate_baseline_report,
    generate_report,
    run_inference,
)

# Admin-only router — all eval run/report routes require admin
eval_router = APIRouter(prefix="/eval", dependencies=[Depends(require_admin)])

# Separate router for frame image serving — uses ?token= query param because
# <img src> tags cannot send Authorization headers.
eval_frame_router = APIRouter(prefix="/eval")

# ── Eval run state (shared with GET /eval/status) ─────────────────────────────

_eval_state: dict = {
    "running": False,
    "progress": 0,          # number of prompts completed so far
    "total": len(PROMPTS),
    "current_label": None,  # label of the prompt currently running
    "results": [],          # grows as each prompt finishes
    "report_id": None,      # "report_YYYYMMDD_HHMMSS"
    "error": None,
}


# ── Background eval thread ────────────────────────────────────────────────────

def _run_eval() -> None:
    report_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_id = f"report_{report_ts}"
    report_dir = os.path.join(REPORTS_DIR, report_id)
    os.makedirs(report_dir, exist_ok=True)

    _eval_state["running"] = True
    _eval_state["progress"] = 0
    _eval_state["results"] = []
    _eval_state["report_id"] = report_id
    _eval_state["error"] = None
    _eval_state["current_label"] = None

    results: list[dict] = []

    for i, (label, prompt) in enumerate(PROMPTS):
        _eval_state["current_label"] = label
        _eval_state["progress"] = i

        jpeg = _state.get("camera") and _state["camera"].get_latest_jpeg()
        if not jpeg:
            _eval_state["error"] = "Camera not ready — no frames available."
            _eval_state["running"] = False
            return

        acquired = _infer_lock.acquire(timeout=15)
        if not acquired:
            _eval_state["error"] = f"Timed out waiting for inference lock on '{label}'."
            _eval_state["running"] = False
            return

        try:
            llm = _state.get("llm")
            if not llm:
                _eval_state["error"] = "Model not loaded."
                _eval_state["running"] = False
                return
            text, tokens, elapsed = run_inference(llm, jpeg, prompt)
        finally:
            _infer_lock.release()

        # Save raw frame — must happen before appending to results so the
        # frame file exists by the time the frontend renders the result card.
        frame_path = os.path.join(report_dir, f"{label}.jpg")
        with open(frame_path, "wb") as f:
            f.write(jpeg)

        result: dict = {
            "label": label,
            "prompt": prompt,
            "response": text,
            "tokens": tokens,
            "latency_s": round(elapsed, 2),
        }
        results.append(result)
        _eval_state["results"] = list(results)   # copy; avoids iterator/mutation races
        _eval_state["progress"] = i + 1

    # Persist structured results for the report viewer and future comparisons.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current = {"timestamp": timestamp, "results": results}

    with open(os.path.join(report_dir, "results.json"), "w") as f:
        json.dump(current, f, indent=2)

    # Read the current baseline snapshot once here; avoids any race if the user
    # calls /eval/set-baseline for a different report while this run finishes.
    previous = None
    if os.path.exists(BASELINE_PATH):
        try:
            with open(BASELINE_PATH) as f:
                previous = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    if previous:
        generate_report(previous, current, report_dir)
    else:
        generate_baseline_report(current, report_dir)

    _eval_state["running"] = False
    _eval_state["current_label"] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _valid_report_id(report_id: str) -> bool:
    return bool(re.match(r"^report_\d{8}_\d{6}$", report_id))


def _compute_stats(results: list[dict]) -> dict:
    latencies = [r["latency_s"] for r in results]
    fastest = min(results, key=lambda r: r["latency_s"])
    slowest = max(results, key=lambda r: r["latency_s"])
    return {
        "avg_latency_s": round(sum(latencies) / len(latencies), 2),
        "total_tokens": sum(r["tokens"] for r in results),
        "fastest": {"label": fastest["label"], "latency_s": fastest["latency_s"]},
        "slowest": {"label": slowest["label"], "latency_s": slowest["latency_s"]},
    }


def _compute_comparison(current: list[dict], previous: list[dict]) -> dict | None:
    if not previous or not current:
        return None
    prev_map = {r["label"]: r for r in previous}
    avg_prev = sum(r["latency_s"] for r in previous) / len(previous)
    avg_curr = sum(r["latency_s"] for r in current) / len(current)
    avg_delta = round(avg_curr - avg_prev, 2)
    per_prompt: dict[str, dict] = {}
    for r in current:
        p = prev_map.get(r["label"])
        if p:
            per_prompt[r["label"]] = {
                "latency_delta": round(r["latency_s"] - p["latency_s"], 2),
                "token_delta": r["tokens"] - p["tokens"],
            }
    return {
        "avg_latency_delta": avg_delta,
        "direction": "faster" if avg_delta < 0 else "slower" if avg_delta > 0 else "same",
        "per_prompt": per_prompt,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@eval_router.post("/run")
def eval_run():
    if _eval_state["running"]:
        raise HTTPException(409, "Evaluation already running.")
    if _state.get("autoscan"):
        raise HTTPException(409, "Stop Auto-Scan before running evaluation.")
    if not _state.get("llm"):
        raise HTTPException(503, "Model not loaded yet.")
    camera = _state.get("camera")
    if not camera or not camera.get_latest_jpeg():
        raise HTTPException(503, "Camera not ready — no frames yet.")
    threading.Thread(target=_run_eval, daemon=True).start()
    return {"started": True, "total": len(PROMPTS)}


@eval_router.get("/status")
def eval_status():
    return dict(_eval_state)


@eval_router.get("/reports")
def eval_reports():
    if not os.path.isdir(REPORTS_DIR):
        return {"reports": []}

    baseline_ts: str | None = None
    if os.path.exists(BASELINE_PATH):
        try:
            with open(BASELINE_PATH) as f:
                baseline_ts = json.load(f).get("timestamp")
        except (json.JSONDecodeError, OSError):
            pass

    entries: list[dict] = []
    for name in sorted(os.listdir(REPORTS_DIR), reverse=True):
        if not name.startswith("report_"):
            continue
        full = os.path.join(REPORTS_DIR, name)
        if not os.path.isdir(full):
            continue

        meta: dict = {
            "id": name,
            "has_report": os.path.exists(os.path.join(full, "report.md")),
        }
        results_path = os.path.join(full, "results.json")
        if os.path.exists(results_path):
            try:
                with open(results_path) as f:
                    data = json.load(f)
                res_list: list[dict] = data.get("results", [])
                meta["timestamp"] = data.get("timestamp")
                meta["result_count"] = len(res_list)
                meta["avg_latency_s"] = (
                    round(sum(r["latency_s"] for r in res_list) / len(res_list), 2)
                    if res_list else None
                )
                meta["is_baseline"] = (meta["timestamp"] == baseline_ts)
            except (json.JSONDecodeError, OSError):
                meta.update(timestamp=None, result_count=0, avg_latency_s=None, is_baseline=False)
        else:
            meta.update(timestamp=None, result_count=0, avg_latency_s=None, is_baseline=False, legacy=True)

        entries.append(meta)

    return {"reports": entries}


@eval_router.get("/report/{report_id}")
def eval_report(report_id: str):
    if not _valid_report_id(report_id):
        raise HTTPException(400, "Invalid report ID format.")
    report_dir = os.path.join(REPORTS_DIR, report_id)
    if not os.path.isdir(report_dir):
        raise HTTPException(404, "Report not found.")
    results_path = os.path.join(report_dir, "results.json")
    if not os.path.exists(results_path):
        raise HTTPException(404, "No structured data for this report (legacy CLI run).")

    with open(results_path) as f:
        data = json.load(f)

    is_baseline = False
    baseline_data: dict | None = None
    if os.path.exists(BASELINE_PATH):
        try:
            with open(BASELINE_PATH) as f:
                bl = json.load(f)
            is_baseline = bl.get("timestamp") == data.get("timestamp")
            if not is_baseline:
                baseline_data = bl
        except (json.JSONDecodeError, OSError):
            pass

    results = data.get("results", [])
    for r in results:
        # URL-encode the filename so the browser receives a valid URL.
        # FastAPI will decode %20 → space in the /frame/ route's filename param.
        r["frame_url"] = f"/eval/frame/{report_id}/{quote(r['label'] + '.jpg')}"

    stats = _compute_stats(results) if results else None
    comparison = (
        _compute_comparison(results, baseline_data["results"])
        if baseline_data and baseline_data.get("results") and results
        else None
    )

    return {
        "report_id": report_id,
        "timestamp": data.get("timestamp"),
        "results": results,
        "is_baseline": is_baseline,
        "stats": stats,
        "comparison": comparison,
    }


@eval_frame_router.get("/frame/{report_id}/{filename}")
def eval_frame(report_id: str, filename: str, token: str | None = None):
    if not token:
        raise HTTPException(401, "Token required.")
    try:
        jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid or expired token.")
    if not _valid_report_id(report_id):
        raise HTTPException(400, "Invalid report ID format.")
    # Only allow word chars + spaces + .jpg — blocks path traversal via ..
    if not re.match(r"^[\w\s]+\.jpg$", filename):
        raise HTTPException(400, "Invalid filename.")
    frame_path = os.path.join(REPORTS_DIR, report_id, filename)
    if not os.path.exists(frame_path):
        raise HTTPException(404, "Frame image not found.")
    return FileResponse(frame_path, media_type="image/jpeg")


@eval_router.post("/set-baseline/{report_id}")
def eval_set_baseline(report_id: str):
    if not _valid_report_id(report_id):
        raise HTTPException(400, "Invalid report ID format.")
    results_path = os.path.join(REPORTS_DIR, report_id, "results.json")
    if not os.path.exists(results_path):
        raise HTTPException(404, "Results not found for this report.")
    with open(results_path) as f:
        data = json.load(f)
    os.makedirs(os.path.dirname(BASELINE_PATH), exist_ok=True)
    with open(BASELINE_PATH, "w") as f:
        json.dump(data, f, indent=2)
    return {"baseline_set": True, "report_id": report_id, "timestamp": data.get("timestamp")}
