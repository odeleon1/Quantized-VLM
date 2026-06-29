import re
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel

from app.core.config import JWT_EXPIRY_HOURS, JWT_SECRET
from app.core.database import (
    create_user,
    email_exists,
    find_user,
    find_user_by_id,
    update_password,
    username_exists,
)

auth_router = APIRouter(prefix="/auth")
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
_security = HTTPBearer()

# ── Password validation ───────────────────────────────────────────────────────

_SPECIAL = re.compile(r"[^A-Za-z0-9]")


def _validate_password(password: str) -> str | None:
    """Returns an error message, or None if the password is valid."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r"[A-Z]", password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must contain at least one lowercase letter."
    if not re.search(r"\d", password):
        return "Password must contain at least one number."
    if not _SPECIAL.search(password):
        return "Password must contain at least one special character."
    return None


# ── JWT helpers ───────────────────────────────────────────────────────────────

def _create_token(user: dict) -> str:
    payload = {
        "sub":      str(user["id"]),   # JWT spec requires sub to be a string
        "username": user["username"],
        "email":    user["email"],
        "is_admin": bool(user["is_admin"]),
        "exp":      datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


# ── Auth dependency (imported by routes.py and eval_routes.py) ────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> dict:
    try:
        return jwt.decode(credentials.credentials, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if not user.get("is_admin"):
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user


# ── Request models ────────────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    identifier: str   # username or email
    password: str


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


# ── Routes ────────────────────────────────────────────────────────────────────

@auth_router.post("/signup")
def signup(body: SignupRequest):
    username = body.username.strip()
    email    = body.email.strip().lower()

    if not username or not email or not body.password:
        raise HTTPException(400, "All fields are required.")
    if len(username) < 3:
        raise HTTPException(400, "Username must be at least 3 characters.")
    if "@" not in email:
        raise HTTPException(400, "Invalid email address.")

    err = _validate_password(body.password)
    if err:
        raise HTTPException(400, err)

    if username_exists(username):
        raise HTTPException(409, "Username is already taken.")
    if email_exists(email):
        raise HTTPException(409, "An account with that email already exists.")

    user = create_user(username, email, pwd_ctx.hash(body.password))
    return {"message": "Account created successfully.", "username": user["username"]}


@auth_router.post("/login")
def login(body: LoginRequest):
    user = find_user(body.identifier.strip())
    if not user or not pwd_ctx.verify(body.password, user["password_hash"]):
        raise HTTPException(401, "Invalid username/email or password.")
    return {
        "access_token": _create_token(user),
        "token_type":   "bearer",
        "user": {
            "id":       user["id"],
            "username": user["username"],
            "email":    user["email"],
            "is_admin": bool(user["is_admin"]),
        },
    }


@auth_router.get("/me")
def me(user: dict = Depends(get_current_user)):
    row = find_user_by_id(int(user["sub"]))
    if not row:
        raise HTTPException(401, "User not found.")
    return {
        "id":       row["id"],
        "username": row["username"],
        "email":    row["email"],
        "is_admin": bool(row["is_admin"]),
    }


@auth_router.post("/change-password")
def change_password(body: ChangePasswordRequest, user: dict = Depends(get_current_user)):
    row = find_user_by_id(int(user["sub"]))
    if not row or not pwd_ctx.verify(body.current_password, row["password_hash"]):
        raise HTTPException(401, "Current password is incorrect.")
    err = _validate_password(body.new_password)
    if err:
        raise HTTPException(400, err)
    update_password(user["sub"], pwd_ctx.hash(body.new_password))
    return {"message": "Password changed successfully."}
