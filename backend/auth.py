"""
auth.py — simple shared-password protection via Bearer token.

Flow:
  1. Client POST /auth/login  { "password": "..." }
  2. Server checks against APP_PASSWORD env var
  3. If correct → returns a signed token (HMAC-SHA256)
  4. Client sends token in Authorization: Bearer <token> header
  5. Backend verifies token on every protected endpoint via FastAPI dependency
"""

import os
import hmac
import hashlib
import time
from fastapi import HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# ── Secret key: derived from APP_PASSWORD so no separate JWT secret needed ────
def _get_secret() -> str:
    pwd = os.getenv("APP_PASSWORD", "")
    if not pwd:
        return ""  # auth disabled if no password set
    # Derive a stable secret from the password
    return hmac.new(pwd.encode(), b"ragforge-token-secret", hashlib.sha256).hexdigest()

def _make_token(password: str) -> str:
    """Sign password + day-stamp so tokens expire daily."""
    day = str(int(time.time()) // 86400)  # changes every 24h
    secret = _get_secret()
    return hmac.new(secret.encode(), f"{password}:{day}".encode(), hashlib.sha256).hexdigest()

def _verify_token(token: str) -> bool:
    pwd = os.getenv("APP_PASSWORD", "")
    if not pwd:
        return True  # no password set → open access
    # Check today and yesterday (handles midnight boundary)
    for offset in [0, 1]:
        day = str(int(time.time()) // 86400 - offset)
        secret = _get_secret()
        expected = hmac.new(secret.encode(), f"{pwd}:{day}".encode(), hashlib.sha256).hexdigest()
        if hmac.compare_digest(token, expected):
            return True
    return False


# ── FastAPI models ─────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    password: str

class LoginResponse(BaseModel):
    token: str
    message: str

# ── Dependency: require valid token on protected routes ───────────────────────
_bearer = HTTPBearer(auto_error=False)

def require_auth(credentials: HTTPAuthorizationCredentials = Security(_bearer)):
    pwd = os.getenv("APP_PASSWORD", "")
    if not pwd:
        return  # auth disabled
    if not credentials or not _verify_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid or missing token")

def create_login_token(password: str) -> LoginResponse:
    pwd = os.getenv("APP_PASSWORD", "")
    if not pwd:
        return LoginResponse(token="no-auth", message="Auth disabled")
    if not hmac.compare_digest(password, pwd):
        raise HTTPException(status_code=401, detail="Wrong password")
    return LoginResponse(token=_make_token(password), message="OK")
