import os
import sqlite3
from datetime import datetime, timezone

from app.core.config import (
    ADMIN_EMAIL, ADMIN_PASSWORD, ADMIN_USERNAME, DB_PATH,
)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin      INTEGER NOT NULL DEFAULT 0,
            created_at    TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS outputs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            type       TEXT    NOT NULL,
            timestamp  TEXT    NOT NULL,
            file_path  TEXT,
            prompt     TEXT,
            response   TEXT,
            tokens     INTEGER,
            elapsed_s  REAL,
            user_id    INTEGER NOT NULL,
            frame_hash TEXT
        )
    """)
    # Databases created before frame_hash existed keep working: add the column
    # in place if it is missing rather than requiring a rebuild.
    existing_cols = {r[1] for r in conn.execute("PRAGMA table_info(outputs)").fetchall()}
    if "frame_hash" not in existing_cols:
        conn.execute("ALTER TABLE outputs ADD COLUMN frame_hash TEXT")
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count == 0:
        from app.core.security import hash_password
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO users (username, email, password_hash, is_admin, created_at) "
            "VALUES (?, ?, ?, 1, ?)",
            (ADMIN_USERNAME, ADMIN_EMAIL, hash_password(ADMIN_PASSWORD), now),
        )
        conn.commit()
        print(f"\n{'=' * 60}")
        print("[AUTH] Initial admin account created:")
        print(f"       Username : {ADMIN_USERNAME}")
        print(f"       Email    : {ADMIN_EMAIL}")
        print(f"       Password : {ADMIN_PASSWORD}")
        print("[AUTH] Change this password after first login.")
        print(f"{'=' * 60}\n")

    conn.close()


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def find_user(username_or_email: str) -> dict | None:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM users WHERE username = ? OR email = ?",
        (username_or_email, username_or_email),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def find_user_by_id(user_id: int) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def username_exists(username: str) -> bool:
    conn = get_db()
    row = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return row is not None


def email_exists(email: str) -> bool:
    conn = get_db()
    row = conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return row is not None


def create_user(username: str, email: str, password_hash: str) -> dict:
    conn = get_db()
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        "INSERT INTO users (username, email, password_hash, is_admin, created_at) "
        "VALUES (?, ?, ?, 0, ?)",
        (username, email, password_hash, now),
    )
    conn.commit()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
    conn.close()
    return dict(row)


def list_users() -> list[dict]:
    conn = get_db()
    rows = conn.execute("SELECT * FROM users ORDER BY created_at ASC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_admin(user_id: int, is_admin: bool) -> None:
    conn = get_db()
    conn.execute(
        "UPDATE users SET is_admin = ? WHERE id = ?",
        (1 if is_admin else 0, user_id),
    )
    conn.commit()
    conn.close()


def update_password(user_id: int, new_hash: str) -> None:
    conn = get_db()
    conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
    conn.commit()
    conn.close()


# ── Output CRUD ───────────────────────────────────────────────────────────────

def log_output(
    *,
    type: str,
    timestamp: str,
    file_path: str | None,
    prompt: str | None,
    response: str | None,
    tokens: int | None,
    elapsed_s: float | None,
    user_id: int,
    frame_hash: str | None = None,
) -> int:
    conn = get_db()
    cur = conn.execute(
        "INSERT INTO outputs (type, timestamp, file_path, prompt, response, tokens, elapsed_s, user_id, frame_hash) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (type, timestamp, file_path, prompt, response, tokens, elapsed_s, user_id, frame_hash),
    )
    conn.commit()
    output_id = cur.lastrowid
    conn.close()
    return output_id


def list_outputs(user_id: int, output_type: str | None = None) -> list[dict]:
    conn = get_db()
    if output_type:
        rows = conn.execute(
            "SELECT * FROM outputs WHERE user_id = ? AND type = ? ORDER BY timestamp DESC",
            (user_id, output_type),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM outputs WHERE user_id = ? ORDER BY timestamp DESC",
            (user_id,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_output(output_id: int, user_id: int) -> dict | None:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM outputs WHERE id = ? AND user_id = ?",
        (output_id, user_id),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_output_by_id(output_id: int) -> dict | None:
    """Fetch an output row without ownership check — for admin use."""
    conn = get_db()
    row = conn.execute("SELECT * FROM outputs WHERE id = ?", (output_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def list_all_outputs_with_username(output_type: str | None = None) -> list[dict]:
    """Return all outputs from all users, with each row's username attached.
    Used by the admin library view."""
    conn = get_db()
    if output_type:
        rows = conn.execute(
            "SELECT o.*, u.username FROM outputs o "
            "JOIN users u ON o.user_id = u.id "
            "WHERE o.type = ? ORDER BY o.timestamp DESC",
            (output_type,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT o.*, u.username FROM outputs o "
            "JOIN users u ON o.user_id = u.id "
            "ORDER BY o.timestamp DESC"
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
