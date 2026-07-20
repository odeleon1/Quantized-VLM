"""
Central configuration for the backend.

This file is committed to the repository and contains no secrets. Every
sensitive value is read from an environment variable, and where a secret is
needed but not provided, one is generated locally and stored outside git.

Paths are derived from the repository root so a fresh clone works without
editing anything. Override CAMERA_INDEX and the ADMIN_* / JWT_* values through
environment variables when the defaults do not match the deployment.
"""

import os
import secrets
import stat

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))

# ── Model and data paths ──────────────────────────────────────────────────────

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "moondream2-text-model-Q4_K_M.gguf")
MMPROJ_PATH = os.path.join(PROJECT_ROOT, "models", "moondream2-mmproj-f16.gguf")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "eval reports")
BASELINE_PATH = os.path.join(REPORTS_DIR, "results_latest.json")
RUNS_DIR = os.path.join(OUTPUT_DIR, "runs")
DB_PATH = os.path.join(OUTPUT_DIR, "vlmedge.db")

# ── Camera ────────────────────────────────────────────────────────────────────

CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "480"))
CAMERA_FPS = int(os.environ.get("CAMERA_FPS", "30"))

# ── Authentication secrets ────────────────────────────────────────────────────


def _load_or_create_jwt_secret() -> str:
    """
    Resolve the JWT signing secret without ever committing it.

    Order of preference:
      1. The JWT_SECRET environment variable, if set.
      2. A previously generated secret at output/.jwt_secret, so tokens stay
         valid across restarts.
      3. A freshly generated secret, written to output/.jwt_secret with owner
         only permissions.
    """
    env_secret = os.environ.get("JWT_SECRET")
    if env_secret:
        return env_secret

    secret_path = os.path.join(OUTPUT_DIR, ".jwt_secret")
    if os.path.exists(secret_path):
        with open(secret_path) as f:
            stored = f.read().strip()
        if stored:
            return stored

    generated = secrets.token_hex(32)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(secret_path, "w") as f:
        f.write(generated)
    os.chmod(secret_path, stat.S_IRUSR | stat.S_IWUSR)  # 0600
    return generated


JWT_SECRET = _load_or_create_jwt_secret()
JWT_EXPIRY_HOURS = int(os.environ.get("JWT_EXPIRY_HOURS", "24"))

# ── First boot admin account ──────────────────────────────────────────────────
# These seed the admin account only on first boot, when the users table is
# empty. If ADMIN_PASSWORD is not provided, a strong random one is generated and
# printed to the console by database.py so the account never has a guessable
# default. Losing that first boot console output means deleting output/vlmedge.db
# to re-seed.

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "admin@vlmedge.local")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD") or secrets.token_urlsafe(12)
