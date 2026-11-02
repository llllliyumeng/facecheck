"""Cloud access control, revocable opaque sessions and deployment validation.

Credentials are high-entropy API tokens, stored as SHA-256 digests in a private
runtime file. This is organization-level RBAC, not tenant or video ownership.
No cryptographic video format or recognition protocol is changed here.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

COOKIE = "facecheck_session"
SESSION_SECONDS = 8 * 60 * 60
ROLES = {"admin", "operator", "auditor"}
PUBLIC_PATHS = {"/health", "/api/v1/auth/status", "/api/v1/auth/login"}
AUDITOR_PATHS = {"/api/v1/auth/me", "/api/v1/audit", "/api/v1/security/summary",
                 "/api/v1/platform/audit/export", "/api/v1/platform/requirements",
                 "/api/v1/platform/benchmarks"}


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def secret_file(name: str) -> Optional[bytes]:
    """Binary secrets support arbitrary bytes and never appear in error text."""
    filename = os.getenv(name + "_FILE")
    if filename and os.getenv(name):
        raise ValueError(f"configure only {name} or {name}_FILE")
    if not filename:
        value = os.getenv(name)
        return value.encode("utf-8") if value else None
    try:
        path = Path(filename)
        if not path.is_file() or path.stat().st_size > 65536:
            raise OSError
        return path.read_bytes()
    except OSError:
        raise ValueError(f"{name}_FILE is unreadable or invalid") from None


@dataclass(frozen=True)
class Principal:
    id: str
    role: str
    credential_digest: str


class AccessPolicy:
    def __init__(self, repository):
        self.repository = repository
        self.deployment = os.getenv("FACECHECK_DEPLOYMENT", "local")
        if self.deployment not in {"local", "cloud"}:
            raise ValueError("FACECHECK_DEPLOYMENT must be local or cloud")
        raw_file = os.getenv("FACECHECK_AUTH_CREDENTIALS_FILE")
        self.credentials_file = Path(raw_file) if raw_file else None
        self.origin = os.getenv("FACECHECK_PUBLIC_ORIGIN", "").rstrip("/")
        self.cloud = self.deployment == "cloud"
        self.enabled = self.credentials_file is not None
        if self.cloud and (not self.enabled or not self.origin):
            raise ValueError("cloud deployment requires credentials and FACECHECK_PUBLIC_ORIGIN")
        parsed = urlsplit(self.origin)
        if self.enabled and (parsed.scheme not in {"https", "http"} or not parsed.hostname
                             or parsed.username or parsed.password or parsed.query or parsed.fragment
                             or parsed.path not in {"", "/"}):
            raise ValueError("FACECHECK_PUBLIC_ORIGIN must be an origin without path or credentials")
        if self.cloud and parsed.scheme != "https":
            raise ValueError("cloud deployment requires an HTTPS public origin")
        self.allowed_hosts = [parsed.hostname] if parsed.hostname else []
        self._attempt_lock = threading.Lock()
        self._attempts = {}  # bounded login abuse guard, independent from session correctness
        if self.enabled:
            self.credentials()  # fail closed before serving
            with self.connect() as db:
                db.execute('''CREATE TABLE IF NOT EXISTS access_sessions (
                    session_digest TEXT PRIMARY KEY, principal_id TEXT NOT NULL,
                    credential_digest TEXT NOT NULL, expires_at REAL NOT NULL)''')
                db.execute('''CREATE TABLE IF NOT EXISTS access_revocations (
                    credential_digest TEXT PRIMARY KEY, revoked_at REAL NOT NULL)''')

    def connect(self):
        db = sqlite3.connect(self.repository.path, timeout=20)
        db.row_factory = sqlite3.Row
        return db

    def credentials(self):
        try:
            if self.credentials_file.stat().st_size > 65536:
                raise ValueError
            data = json.loads(self.credentials_file.read_text(encoding="utf-8"))
            rows = data["principals"]
            if data.get("version") != 1 or not isinstance(rows, list) or not 1 <= len(rows) <= 100:
                raise ValueError
            result = []
            for row in rows:
                pid, role, token_hash = row["id"], row["role"], row["token_sha256"]
                if (not isinstance(pid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", pid)
                        or role not in ROLES or not isinstance(token_hash, str)
                        or not re.fullmatch(r"[0-9a-f]{64}", token_hash)):
                    raise ValueError
                result.append(Principal(pid, role, token_hash))
            if len({p.id for p in result}) != len(result) or len({p.credential_digest for p in result}) != len(result):
                raise ValueError
            return result
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            raise ValueError("access credentials are unavailable or invalid") from None

    def principal(self, request: Request):
        credentials = self.active_credentials()
        authorization = request.headers.get("authorization")
        if authorization:
            kind, _, value = authorization.partition(" ")
            if kind.lower() != "bearer" or not 32 <= len(value) <= 256:
                return None, "bearer"
            hashed = digest(value)
            return next((p for p in credentials if hmac.compare_digest(p.credential_digest, hashed)), None), "bearer"
        session = request.cookies.get(COOKIE)
        if not session or not 32 <= len(session) <= 128:
            return None, "cookie"
        with self.connect() as db:
            row = db.execute("SELECT * FROM access_sessions WHERE session_digest=? AND expires_at>?", (digest(session), time.time())).fetchone()
        if not row:
            return None, "cookie"
        return next((p for p in credentials if p.id == row["principal_id"] and hmac.compare_digest(p.credential_digest, row["credential_digest"])), None), "cookie"

    def active_credentials(self):
        credentials = self.credentials()  # removing/changing a token also revokes its sessions
        with self.connect() as db:
            revoked = {r[0] for r in db.execute("SELECT credential_digest FROM access_revocations")}
        return [p for p in credentials if p.credential_digest not in revoked]

    @staticmethod
    def permitted(principal, path, method):
        if path in {"/api/v1/auth/me", "/api/v1/auth/logout"}:
            return True
        if path.startswith("/api/v1/auth/admin/"):
            return principal.role == "admin"
        if path.startswith("/api/v1/platform/key-rotations"):
            return principal.role == "admin"
        if principal.role in {"admin", "operator"}:
            return True
        # Auditors cannot download plaintext videos, read queries or launch inference.
        return method == "GET" and (path in AUDITOR_PATHS or path.startswith("/api/v1/platform/benchmarks/"))

    def login_allowed(self, address):
        now = time.monotonic()
        with self._attempt_lock:
            # Bound memory even for distributed bogus logins. Pruning never touches sessions.
            self._attempts = {ip: ts for ip, ts in self._attempts.items() if ts[-1] > now - 60}
            if address not in self._attempts and len(self._attempts) >= 4096:
                return False
            values = [ts for ts in self._attempts.get(address, []) if ts > now - 60]
            if len(values) >= 10:
                return False
            self._attempts[address] = values + [now]
            return True

    def audit(self, event, principal=None, **detail):
        self.repository.record_event(None, event, json.dumps(dict(actor=principal.id if principal else None, role=principal.role if principal else None, **detail), ensure_ascii=False, sort_keys=True))


class AccessMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, policy):
        super().__init__(app)
        self.policy = policy

    async def dispatch(self, request, call_next):
        policy = self.policy
        path = request.url.path
        if policy.enabled and path != "/health":
            if policy.cloud and request.url.scheme != "https":
                return JSONResponse({"detail": "HTTPS is required"}, status_code=400)
            # Expose only the login page and public session status to unauthenticated clients.
            if path.startswith("/api/") and path not in PUBLIC_PATHS:
                try:
                    principal, source = policy.principal(request)
                except ValueError:
                    return JSONResponse({"detail": "access service unavailable"}, status_code=503)
                if principal is None:
                    policy.audit("authentication_rejected", method=request.method)
                    return JSONResponse({"detail": "authentication required"}, status_code=401, headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"})
                request.state.principal = principal
                if not policy.permitted(principal, path, request.method):
                    policy.audit("access_denied", principal, method=request.method)
                    return JSONResponse({"detail": "permission denied"}, status_code=403)
                if source == "cookie" and request.method not in {"GET", "HEAD", "OPTIONS"}:
                    if request.headers.get("origin") != policy.origin:
                        policy.audit("csrf_rejected", principal, method=request.method)
                        return JSONResponse({"detail": "same-origin request required"}, status_code=403)
            # Do not expose OpenAPI schema or interactive API UI without a session.
            elif path in {"/docs", "/redoc", "/openapi.json"}:
                return JSONResponse({"detail": "API documentation is disabled in authenticated mode"}, status_code=404)
        response = await call_next(request)
        if path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if policy.cloud and request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000"
        return response


def auth_router(policy):
    router = APIRouter(prefix="/api/v1/auth", tags=["access control"])

    @router.get("/status")
    def status():
        return dict(enabled=policy.enabled, deployment=policy.deployment)

    @router.get("/me")
    def me(request: Request):
        p = getattr(request.state, "principal", None)
        return dict(enabled=policy.enabled, id=p.id if p else "local", role=p.role if p else "local")

    @router.post("/login")
    async def login(request: Request):
        if not policy.enabled:
            raise HTTPException(404, "authentication is disabled")
        # Login itself also needs origin checking to prevent login-CSRF.
        if request.headers.get("origin") != policy.origin:
            raise HTTPException(403, "same-origin request required")
        if not policy.login_allowed(request.client.host if request.client else "unknown"):
            raise HTTPException(429, "too many login attempts; retry in one minute")
        try:
            raw = bytearray()
            async for block in request.stream():
                raw.extend(block)
                if len(raw) > 4096:
                    raise HTTPException(413, "login request too large")
            data = json.loads(raw)
            token = data.get("token")
            if not isinstance(token, str) or not 32 <= len(token) <= 256:
                raise ValueError
            credentials = policy.active_credentials()
        except (ValueError, AttributeError):
            raise HTTPException(401, "invalid credentials") from None
        p = next((p for p in credentials if hmac.compare_digest(p.credential_digest, digest(token))), None)
        if p is None:
            policy.audit("login_rejected")
            raise HTTPException(401, "invalid credentials")
        session = secrets.token_urlsafe(32)
        with policy.connect() as db:
            db.execute("DELETE FROM access_sessions WHERE expires_at<=?", (time.time(),))
            db.execute("INSERT INTO access_sessions VALUES(?,?,?,?)", (digest(session), p.id, p.credential_digest, time.time()+SESSION_SECONDS))
        policy.audit("login_succeeded", p)
        response = JSONResponse(dict(id=p.id, role=p.role))
        response.set_cookie(COOKIE, session, max_age=SESSION_SECONDS, httponly=True, secure=policy.cloud or policy.origin.startswith("https:"), samesite="strict", path="/")
        return response

    @router.post("/logout")
    def logout(request: Request):
        session = request.cookies.get(COOKIE)
        if policy.enabled and session:
            with policy.connect() as db:
                db.execute("DELETE FROM access_sessions WHERE session_digest=?", (digest(session),))
        response = JSONResponse({"logged_out": True})
        response.delete_cookie(COOKIE, path="/")
        return response

    @router.get("/admin/principals")
    def principals():
        if not policy.enabled:
            raise HTTPException(404, "authentication is disabled")
        active = {p.id for p in policy.active_credentials()}
        return {"items": [dict(id=p.id, role=p.role, active=p.id in active) for p in policy.credentials()]}

    @router.post("/admin/principals/{principal_id}/revoke")
    def revoke(principal_id: str, request: Request):
        if not policy.enabled:
            raise HTTPException(404, "authentication is disabled")
        target = next((p for p in policy.credentials() if p.id == principal_id), None)
        if target is None:
            raise HTTPException(404, "principal not found")
        with policy.connect() as db:
            db.execute("INSERT OR IGNORE INTO access_revocations VALUES(?,?)", (target.credential_digest, time.time()))
            db.execute("DELETE FROM access_sessions WHERE principal_id=?", (target.id,))
        policy.audit("credential_revoked", request.state.principal, target=target.id)
        return {"id": target.id, "revoked": True}

    return router


def validate_cloud_keys(policy, encryption_key, mac_key):
    if not policy.cloud:
        return
    if (not encryption_key or len(encryption_key) != 16 or not mac_key or len(mac_key) < 32
            or encryption_key in {b"0123456789abcdef", b"1234567890123456"}
            or len(set(encryption_key)) < 4 or encryption_key == mac_key):
        raise ValueError("cloud deployment requires independent high-entropy SM4 (16-byte) and MAC (at least 32-byte) secrets")
