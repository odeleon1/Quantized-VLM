"""
Library routes.

Users see only their own outputs (date then type in the sidebar).
Admins see all users' outputs (username → date → type).

Three routers:
  library_router          — Bearer auth; list current user's outputs
  library_admin_router    — Bearer + admin; list all users' outputs with username
  library_download_router — ?token= query param for browser GET requests
                            (img src thumbnail, full-size view, file download)
"""

import os

import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from app.api.auth_routes import get_current_user, require_admin
from app.core.config import JWT_SECRET
from app.core.database import (
    get_output,
    get_output_by_id,
    list_all_outputs_with_username,
    list_outputs,
)

library_router       = APIRouter(prefix="/library", dependencies=[Depends(get_current_user)])
library_admin_router = APIRouter(prefix="/library", dependencies=[Depends(require_admin)])
library_download_router = APIRouter(prefix="/library")


# ── Current-user list ─────────────────────────────────────────────────────────

@library_router.get("/outputs")
def library_list(type: str | None = None, user: dict = Depends(get_current_user)):
    user_id = int(user["sub"])
    rows = list_outputs(user_id=user_id, output_type=type or None)
    return {"outputs": rows}


# ── Admin list (all users, with username) ─────────────────────────────────────

@library_admin_router.get("/admin/outputs")
def library_admin_list(type: str | None = None):
    rows = list_all_outputs_with_username(output_type=type or None)
    return {"outputs": rows}


# ── File serving (token via query param) ─────────────────────────────────────

def _resolve_token(token: str | None) -> dict:
    """Decode and return the JWT payload, or raise 401."""
    if not token:
        raise HTTPException(401, "Token required.")
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid or expired token.")


def _get_row_for_token(output_id: int, payload: dict) -> dict:
    """Return the output row, enforcing ownership unless the caller is admin."""
    is_admin = bool(payload.get("is_admin"))
    user_id  = int(payload["sub"])
    row = get_output_by_id(output_id) if is_admin else get_output(output_id, user_id)
    if not row:
        raise HTTPException(404, "Output not found.")
    return row


@library_download_router.get("/view/{output_id}")
def library_view(output_id: int, token: str | None = None):
    """Serve the file inline — used for <img src> and in-browser preview."""
    payload = _resolve_token(token)
    row = _get_row_for_token(output_id, payload)
    file_path = row.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(404, "File not found on disk.")
    return FileResponse(file_path)


@library_download_router.get("/download/{output_id}")
def library_download(output_id: int, token: str | None = None):
    """Serve the file as a download attachment."""
    payload = _resolve_token(token)
    row = _get_row_for_token(output_id, payload)
    file_path = row.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(404, "File not found on disk.")
    filename = os.path.basename(file_path)
    return FileResponse(
        file_path,
        filename=filename,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
