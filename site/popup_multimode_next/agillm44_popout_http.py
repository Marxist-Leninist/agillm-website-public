#!/usr/bin/env python3
"""Small stdlib HTTP surface for :mod:`agillm44_popout_service`."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import urlsplit

from agillm44_popout_contract import ContractError
from agillm44_popout_service import PopoutService, PreparedRequest


MAX_BODY_BYTES = 512 * 1024


def handler_factory(
    service: PopoutService,
    *,
    token: str = "",
    heartbeat_s: float = 10.0,
) -> type[BaseHTTPRequestHandler]:
    expected_token = token.strip()
    heartbeat_s = max(1.0, float(heartbeat_s))

    class Handler(BaseHTTPRequestHandler):
        server_version = "AGILLMPopout/1"

        def log_message(self, _format: str, *args: Any) -> None:
            return

        @property
        def route(self) -> str:
            return urlsplit(self.path).path.rstrip("/") or "/"

        def _authed(self) -> bool:
            return not expected_token or (self.headers.get("X-AGILLM-Token") or "").strip() == expected_token

        def _send_json(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _read_json(self) -> dict[str, Any]:
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as exc:
                raise ContractError("invalid_request", "invalid Content-Length") from exc
            if length < 0 or length > MAX_BODY_BYTES:
                raise ContractError("request_too_large", f"request body exceeds {MAX_BODY_BYTES} bytes", status=413)
            try:
                raw = json.loads(self.rfile.read(length) or b"{}")
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ContractError("invalid_json", "request body is not valid JSON") from exc
            if not isinstance(raw, dict):
                raise ContractError("invalid_request", "request body must be a JSON object")
            return raw

        def _require_auth(self) -> bool:
            if self._authed():
                return True
            self._send_json(
                401,
                {
                    "ok": False,
                    "error": "bad or missing X-AGILLM-Token",
                    "error_code": "unauthorized",
                },
            )
            return False

        def do_GET(self) -> None:
            if self.route in {"/health", "/v1/health"}:
                self._send_json(200, service.health())
                return
            self._send_json(404, {"ok": False, "error": "not found", "error_code": "not_found"})

        def do_POST(self) -> None:
            if self.route not in {
                "/infer",
                "/stream",
                "/cancel",
                "/v1/infer",
                "/v1/stream",
                "/v1/cancel",
            }:
                self._send_json(404, {"ok": False, "error": "not found", "error_code": "not_found"})
                return
            if not self._require_auth():
                return
            try:
                body = self._read_json()
            except ContractError as exc:
                self._send_json(exc.status, exc.payload())
                return

            if self.route.endswith("/cancel"):
                try:
                    status, result = service.cancel(body.get("request_id"))
                except ContractError as exc:
                    status, result = exc.status, exc.payload()
                self._send_json(status, result)
                return

            try:
                prepared = service.prepare(body)
            except ContractError as exc:
                if self.route.endswith("/stream"):
                    self._send_ndjson_error(exc.status, exc.payload())
                else:
                    self._send_json(exc.status, exc.payload())
                return

            if self.route.endswith("/stream"):
                self._run_stream(prepared)
            else:
                # Do not accumulate a token-by-token trace for buffered calls.
                # max_new is bounded too, but the terminal object is the only
                # buffered contract and keeps memory independent of event count.
                status, terminal = service.execute(prepared, lambda _event: None)
                self._send_json(status, terminal)

        def _send_ndjson_error(self, status: int, body: dict[str, Any]) -> None:
            payload = json.dumps({"event": "error", **body}, ensure_ascii=False).encode("utf-8") + b"\n"
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _run_stream(self, prepared: PreparedRequest) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache, no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            client_gone = threading.Event()
            stopped = threading.Event()
            terminal_sent = threading.Event()
            write_lock = threading.Lock()
            started_at = time.monotonic()

            def emit(obj: dict[str, Any]) -> None:
                event_type = obj.get("event")
                if client_gone.is_set() or (terminal_sent.is_set() and event_type == "heartbeat"):
                    return
                payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
                try:
                    with write_lock:
                        if client_gone.is_set():
                            return
                        if terminal_sent.is_set() and event_type == "heartbeat":
                            return
                        self.wfile.write(payload)
                        self.wfile.flush()
                        if event_type in {"done", "error"}:
                            terminal_sent.set()
                except (BrokenPipeError, ConnectionResetError):
                    # Continue draining the model worker so its JSONL pipe stays
                    # synchronized; explicit /cancel remains request-scoped.
                    client_gone.set()

            def heartbeat() -> None:
                while not stopped.wait(heartbeat_s):
                    if terminal_sent.is_set():
                        return
                    emit(
                        {
                            "event": "heartbeat",
                            "ok": True,
                            "request_id": prepared.request_id,
                            "elapsed_s": round(time.monotonic() - started_at, 1),
                        }
                    )

            thread = threading.Thread(target=heartbeat, name="agillm-popout-heartbeat", daemon=True)
            thread.start()
            try:
                service.execute(prepared, emit)
            finally:
                stopped.set()
                thread.join(timeout=1.0)

    return Handler


def serve(
    service: PopoutService,
    *,
    host: str,
    port: int,
    token: str = "",
    heartbeat_s: float = 10.0,
) -> None:
    handler = handler_factory(service, token=token, heartbeat_s=heartbeat_s)
    ThreadingHTTPServer((host, int(port)), handler).serve_forever()
