"""
Password hashing helpers built directly on bcrypt.

passlib was removed because it is unmaintained and its bcrypt backend breaks
against modern bcrypt releases: it cannot read the version and its internal
probe passes an over length value that bcrypt 5.x rejects, so every hash and
verify call raises. Calling bcrypt directly avoids that whole failure mode.

bcrypt only uses the first 72 bytes of a password and, as of 5.x, raises if a
longer value is passed. We truncate to 72 bytes so long passwords are handled
the same way older deployments handled them and never crash. Existing hashes are
standard "$2b$" strings, so they verify unchanged.
"""

import bcrypt

_MAX_PASSWORD_BYTES = 72


def hash_password(password: str) -> str:
    pw = password.encode("utf-8")[:_MAX_PASSWORD_BYTES]
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    pw = password.encode("utf-8")[:_MAX_PASSWORD_BYTES]
    try:
        return bcrypt.checkpw(pw, password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False
