#!/usr/bin/env python3
import hashlib
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

import jwt

HOST = os.getenv("SG_VERCEL_OIDC_HOST", "127.0.0.1")
PORT = int(os.getenv("SG_VERCEL_OIDC_PORT", "18791"))
TEAM_SLUG = os.getenv("SG_VERCEL_TEAM_SLUG", "blingsco-2487s-projects")
PROJECT_NAME = os.getenv("SG_VERCEL_PROJECT_NAME", "sg-mcp-relay")
PROJECT_ID = os.getenv("SG_VERCEL_PROJECT_ID", "prj_kiLZP5gpauARNvJ9xc1ulSej7pEQ")
OWNER_ID = os.getenv("SG_VERCEL_OWNER_ID", "team_3zZ8eE2s0UOSug0S0F4PuSwR")
ENVIRONMENT = os.getenv("SG_VERCEL_ENVIRONMENT", "production")
RELAY_CAP_SHA256 = os.getenv("SG_VERCEL_RELAY_CAP_SHA256", "5686f479efbb5963674552883026bb50e8dad8ed272e28d4608584da566c1d29")
AUDIENCE = f"https://vercel.com/{TEAM_SLUG}"
SUBJECT = f"owner:{TEAM_SLUG}:project:{PROJECT_NAME}:environment:{ENVIRONMENT}"
ALLOWED_ISSUERS = {
    "https://oidc.vercel.com": "https://oidc.vercel.com/.well-known/jwks",
    f"https://oidc.vercel.com/{TEAM_SLUG}": f"https://oidc.vercel.com/{TEAM_SLUG}/.well-known/jwks",
}
JWK_CLIENTS = {issuer: jwt.PyJWKClient(uri, cache_keys=True, lifespan=3600) for issuer, uri in ALLOWED_ISSUERS.items()}


def verify_capability(value: str) -> None:
    if not value or len(value) > 256:
        raise ValueError("missing capability")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    if not hmac.compare_digest(digest, RELAY_CAP_SHA256):
        raise ValueError("bad capability")


def verify_token(token: str) -> dict:
    header = jwt.get_unverified_header(token)
    if header.get("alg") != "RS256":
        raise jwt.InvalidAlgorithmError("expected RS256")
    unverified = jwt.decode(token, options={"verify_signature": False, "verify_aud": False})
    issuer = unverified.get("iss")
    if issuer not in ALLOWED_ISSUERS:
        raise jwt.InvalidIssuerError("untrusted issuer")
    signing_key = REDACTED
    payload = jwt.decode(
        token,
        signing_key.key,
        algorithms=["RS256"],
        audience=AUDIENCE,
        issuer=issuer,
        options={"require": ["exp", "iss", "aud", "sub"]},
    )
    if payload.get("sub") != SUBJECT:
        raise jwt.InvalidTokenError("subject mismatch")
    expected = {
        "owner": TEAM_SLUG,
        "owner_id": OWNER_ID,
        "project": PROJECT_NAME,
        "project_id": PROJECT_ID,
        "environment": ENVIRONMENT,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise jwt.InvalidTokenError(f"{key} mismatch")
    return payload


class Handler(BaseHTTPRequestHandler):
    server_version = "SGVercelOIDC/1.1"

    def _reply(self, code: int, body: dict | None = None):
        data = b"" if body is None else json.dumps(body, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Cache-Control", "no-store")
        if data:
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        if data and self.command != "HEAD":
            self.wfile.write(data)

    def _handle(self):
        path = urlsplit(self.path).path
        if path == "/health":
            self._reply(200, {
                "ok": True,
                "service": "sg-vercel-oidc-auth",
                "version": "1.1",
                "project": PROJECT_NAME,
                "environment": ENVIRONMENT,
                "caller_capability_required": True,
            })
            return
        if path != "/verify":
            self._reply(404, {"ok": False})
            return
        try:
            verify_capability(self.headers.get("X-SG-Relay-Capability", ""))
        except Exception:
            self._reply(401, {"ok": False, "error": "invalid_relay_capability"})
            return
        auth = self.headers.get("Authorization", "")
        if not auth.startswith("Bearer "):
            self._reply(401, {"ok": False, "error": "missing_bearer"})
            return
        token = REDACTED
        if not token or len(token) > 16384:
            self._reply(401, {"ok": False, "error": "invalid_bearer"})
            return
        try:
            verify_token(token)
        except Exception as exc:
            self._reply(401, {"ok": False, "error": exc.__class__.__name__})
            return
        self._reply(204)

    do_GET = _handle
    do_HEAD = _handle
    do_POST = _handle
    do_PUT = _handle
    do_PATCH = _handle
    do_DELETE = _handle
    do_OPTIONS = _handle

    def log_message(self, fmt, *args):
        print(f"{self.client_address[0]} {self.command} {urlsplit(self.path).path} {args[1] if len(args) > 1 else ''}", flush=True)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"sg-vercel-oidc-auth listening on {HOST}:{PORT}", flush=True)
    server.serve_forever()
