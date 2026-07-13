"""
FastAPI server — Phase 11 entry point.
Replaces the PyQt5 desktop app (main.py) with a web interface accessible
from any device on the local network.

Run from the backend/ directory:
    uv run uvicorn server:app --host 0.0.0.0 --port 8000
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.admin_routes import admin_router
from app.api.auth_routes import auth_router
from app.api.camera_thread import CameraThread
from app.api.eval_routes import eval_frame_router, eval_router
from app.api.library_routes import library_admin_router, library_download_router, library_router
from app.api.routes import init_state, router, stream_router
from app.core.config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS
from app.core.database import init_db
from app.services.capture import open_camera, release_camera
from app.services.eval import load_model

_FRONTEND_DIST = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
)

# ── Shared mutable state ──────────────────────────────────────────────────────

_state: dict = {
    "llm": None,
    "cap": None,
    "camera": None,
    "recording": False,
    "recording_file": None,
    "recording_user_id": None,
    "autoscan": False,
    "autoscan_interval": 10,
    "autoscan_thread": None,
    "autoscan_user_id": None,
    "last_result": None,
}


# ── Lifespan: startup and shutdown ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio

    # Initialise the SQLite database and seed the admin account on first run
    init_db()

    # Open camera first (fast) so the stream is available immediately
    _state["cap"] = open_camera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS)
    _state["camera"] = CameraThread(_state["cap"])
    _state["camera"].start()

    # Load model in a thread so the event loop stays responsive
    # (load_model blocks for ~10s on Jetson)
    _state["llm"] = await asyncio.to_thread(load_model)

    init_state(_state)
    print("Server ready — model loaded, camera streaming.")

    yield  # ── server is running ──

    # Shutdown — stop background loops; VideoWriter.release() is called inside the loop
    _state["autoscan"] = False
    _state["recording"] = False
    if _state["camera"]:
        _state["camera"].stop()
    if _state["cap"]:
        release_camera(_state["cap"])
    print("Server shutdown complete.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="VLM Interface", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # LAN-only; open CORS is fine for local network use
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)             # /auth/* — public (no Bearer required)
app.include_router(stream_router)           # /stream  — token via query param
app.include_router(eval_frame_router)       # /eval/frame/* — token via query param
app.include_router(library_download_router) # /library/download/* — token via query param
app.include_router(router)                  # all other routes — Bearer required
app.include_router(eval_router)             # /eval/*  — Bearer + admin required
app.include_router(library_router)          # /library/* — Bearer required
app.include_router(library_admin_router)    # /library/admin/* — Bearer + admin required
app.include_router(admin_router)            # /admin/* — Bearer + admin required

# Serve the built React frontend — must come AFTER the API router
if os.path.isdir(_FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(_FRONTEND_DIST, "assets")), name="assets")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        f = os.path.join(_FRONTEND_DIST, "favicon.ico")
        return FileResponse(f) if os.path.exists(f) else FileResponse(os.path.join(_FRONTEND_DIST, "index.html"))

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str):
        return FileResponse(os.path.join(_FRONTEND_DIST, "index.html"))
