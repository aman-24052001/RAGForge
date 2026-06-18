"""
auth.py — HMAC-SHA256 token auth with per-session IDs.
Token: {session_id}.{hmac_signature}
Expires: ~24h (day-stamp, ±1 day grace for midnight boundary)
"""

import os
import hmac
import hashlib
import time
import secrets
import asyncio
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

_lock = asyncio.Lock()  # protects nothing here but kept for future state

def _secret() -> bytes:
    pwd = os.getenv("APP_PASSWORD", "")
    if not pwd:
        return b"no-auth-open"
    return hmac.new(pwd.encode(), b"ragforge-v2", hashlib.sha256).digest()

def _sign(session_id: str) -> str:
    day = str(int(time.time()) // 86400)
    return hmac.new(_secret(), f"{session_id}:{day}".encode(), hashlib.sha256).hexdigest()

def _verify(token: str) -> str | None:
    """Returns session_id if valid, None otherwise."""
    pwd = os.getenv("APP_PASSWORD", "")
    if not pwd:
        return token if token else "open"
    try:
        session_id, sig = token.rsplit(".", 1)
    except ValueError:
        return None
    # Check today and yesterday (midnight boundary grace)
    for offset in [0, 1]:
        day = str(int(time.time()) // 86400 - offset)
        expected = hmac.new(_secret(), f"{session_id}:{day}".encode(), hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, expected):
            return session_id
    return None


class LoginRequest(BaseModel):
    password: str

class LoginResponse(BaseModel):
    token: str
    session_id: str
    message: str

_bearer = HTTPBearer(auto_error=False)

def get_session_id(
    credentials: HTTPAuthorizationCredentials = Security(_bearer)
) -> str:
    pwd = os.getenv("APP_PASSWORD", "")
    if not pwd:
        return "open"
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing token — please log in")
    sid = _verify(credentials.credentials)
    if not sid:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    return sid

require_auth = get_session_id

def create_login_token(password: str) -> LoginResponse:
    pwd = os.getenv("APP_PASSWORD", "")
    if not pwd:
        return LoginResponse(token="no-auth", session_id="open", message="Auth disabled")
    if not hmac.compare_digest(password.encode(), pwd.encode()):
        raise HTTPException(status_code=401, detail="Wrong password")
    session_id = secrets.token_hex(16)
    token = f"{session_id}.{_sign(session_id)}"
    return LoginResponse(token=token, session_id=session_id, message="OK")
