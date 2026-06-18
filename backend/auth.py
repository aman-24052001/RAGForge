"""
auth.py — HMAC token auth with per-session IDs.

Token format: {session_id}.{hmac_signature}
session_id = random UUID generated at login (unique per browser session)
signature  = HMAC-SHA256(APP_PASSWORD:session_id:day)

This means:
- Each login gets a unique session_id → separate RAG index
- Token expires daily (day-stamp in signature)
- Logout deletes the session from memory
"""

import os
import hmac
import hashlib
import time
import secrets
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel


def _secret() -> str:
    pwd = os.getenv("APP_PASSWORD", "")
    if not pwd:
        return "no-auth"
    return hmac.new(pwd.encode(), b"ragforge-v2", hashlib.sha256).hexdigest()

def _sign(session_id: str) -> str:
    day = str(int(time.time()) // 86400)
    return hmac.new(_secret().encode(),
                    f"{session_id}:{day}".encode(),
                    hashlib.sha256).hexdigest()

def _verify(token: str) -> str | None:
    """Returns session_id if valid, None if invalid."""
    pwd = os.getenv("APP_PASSWORD", "")
    if not pwd:
        # Auth disabled — use a fixed session per token (or 'open')
        return token if token else "open"
    try:
        session_id, sig = token.rsplit(".", 1)
    except ValueError:
        return None
    # Check today and yesterday
    for offset in [0, 1]:
        day = str(int(time.time()) // 86400 - offset)
        expected = hmac.new(_secret().encode(),
                            f"{session_id}:{day}".encode(),
                            hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, expected):
            return session_id
    return None


# ── Models ────────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    password: str

class LoginResponse(BaseModel):
    token: str
    session_id: str
    message: str

# ── FastAPI dependency ────────────────────────────────────────────────────────
_bearer = HTTPBearer(auto_error=False)

def get_session_id(
    credentials: HTTPAuthorizationCredentials = Security(_bearer)
) -> str:
    pwd = os.getenv("APP_PASSWORD", "")
    if not pwd:
        # No password set → single open session
        return "open"
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing token — please log in")
    session_id = _verify(credentials.credentials)
    if not session_id:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return session_id

# Keep require_auth as alias for backward compat
require_auth = get_session_id

# ── Login ─────────────────────────────────────────────────────────────────────
def create_login_token(password: str) -> LoginResponse:
    pwd = os.getenv("APP_PASSWORD", "")
    if not pwd:
        sid = "open"
        return LoginResponse(token="no-auth", session_id=sid, message="Auth disabled")
    if not hmac.compare_digest(password, pwd):
        raise HTTPException(status_code=401, detail="Wrong password")
    session_id = secrets.token_hex(16)  # unique per login
    token = f"{session_id}.{_sign(session_id)}"
    return LoginResponse(token=token, session_id=session_id, message="OK")
