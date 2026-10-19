#!/usr/bin/env python3
"""Adapter for AGILLM's warm native `infer --server` JSONL protocol.

One adapter owns one child process.  A cancellation retires only that child;
it can never reach another mode or the standard 1.1B bridge.
"""

from __future__ import annotations

import json
import os
import select
import re
import subprocess
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from agillm44_popout_contract import ContractError, TELEMETRY_SCHEMA, validate_telemetry


_STAT_RE = re.compile(r"\[(?P<sec>[0-9.]+)s \| (?P<tok>[0-9]+) tokens \| (?P<tps>[0-9.]+) tok/s\]")
_STREAM_PREFIXES = ("[STREAM_AR] ", "[STREAM_SAT] ", "[STREAM_NAT] ")


class NativeJsonlBackend:
    """A serialized warm worker for one canonical decoding mode."""

    def __init__(
        self,
        *,
        mode: str,
        command: Sequence[str],
        env: Mapping[str, str] | None = None,
        startup_timeout_s: float = 600.0,
        request_timeout_s: float = 1800.0,
        queue_timeout_s: float = 30.0,
    ) -> None:
        if not command or not all(isinstance(part, str) and part for part in command):
            raise ValueError("command must be a non-empty list of non-empty strings")
        self.mode = mode
        self.command = tuple(command)
        self.env = {**os.environ, **dict(env or {})}
        self.startup_timeout_s = float(startup_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self.queue_timeout_s = float(queue_timeout_s)
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self._warm_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._busy = False
        self._ready = False
        self._last_error = ""
        self._loads = 0

    @property
    def configured(self) -> bool:
        return True

    def _alive_locked(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def health(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "ready": self._alive_locked() and self._ready,
                "protocol_ready": self._ready,
                "busy": self._busy,
                "loads": self._loads,
                "last_error": self._last_error,
            }

    def ensure_ready(self) -> bool:
        with self._warm_lock:
            with self._state_lock:
                if self._alive_locked() and self._ready:
                    return True
                if self._alive_locked() and not self._ready:
                    self._retire(self._proc, close_pipes=True)
                self._proc = None
                self._ready = False
                try:
                    proc = subprocess.Popen(
                        list(self.command),
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=self.env,
                        text=True,
                        bufsize=1,
                    )
                except Exception as exc:
                    self._last_error = f"worker spawn failed: {exc}"
                    return False
                self._proc = proc

            deadline = time.monotonic() + self.startup_timeout_s
            startup_tail: list[str] = []
            assert proc.stdout is not None
            while time.monotonic() < deadline:
                wait_s = max(0.0, min(1.0, deadline - time.monotonic()))
                readable, _, _ = select.select([proc.stdout], [], [], wait_s)
                if not readable:
                    if proc.poll() is not None:
                        break
                    continue
                line = proc.stdout.readline()
                if not line:
                    if proc.poll() is not None:
                        break
                    continue
                startup_tail.append(line.rstrip("\n")[:512])
                del startup_tail[:-8]
                if "[INFER_SERVER_READY]" in line:
                    with self._state_lock:
                        self._ready = True
                        self._last_error = ""
                        self._loads += 1
                    return True

            self._retire(proc, close_pipes=True)
            with self._state_lock:
                if self._proc is proc:
                    self._proc = None
                self._ready = False
                self._last_error = "worker exited or timed out before READY: " + " | ".join(startup_tail)
            return False

    @staticmethod
    def _close_pipes(proc: subprocess.Popen[str] | None) -> None:
        if proc is None:
            return
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            if pipe is not None:
                try:
                    pipe.close()
                except Exception:
                    pass

    @classmethod
    def _retire(cls, proc: subprocess.Popen[str] | None, *, close_pipes: bool = False) -> None:
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=2.0)
        except Exception:
            pass
        if close_pipes:
            cls._close_pipes(proc)

    def close(self) -> None:
        with self._state_lock:
            proc, self._proc = self._proc, None
            self._ready = False
        if proc is not None and proc.poll() is None:
            try:
                assert proc.stdin is not None
                proc.stdin.write('{"cmd":"quit"}\n')
                proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=2.0)
            except Exception:
                self._retire(proc)
            self._close_pipes(proc)

    def stream(
        self,
        payload: Mapping[str, Any],
        cancel_event: threading.Event,
    ) -> Iterator[dict[str, Any]]:
        if not self._lock.acquire(timeout=self.queue_timeout_s):
            raise ContractError(
                "worker_queue_timeout",
                f"{self.mode} worker queue timed out",
                status=503,
            )
        try:
            if cancel_event.is_set():
                raise ContractError("request_canceled", "request canceled", status=409)
            if not self.ensure_ready():
                raise ContractError(
                    "backend_unavailable",
                    f"{self.mode} worker is not ready",
                    status=503,
                    extra={"backend_health": self.health()},
                )
            with self._state_lock:
                proc = self._proc
                self._busy = True
            assert proc is not None and proc.stdin is not None and proc.stdout is not None

            finished = threading.Event()
            timed_out = threading.Event()

            def retire_on_cancel_or_timeout() -> None:
                deadline = time.monotonic() + self.request_timeout_s
                while not finished.wait(0.05):
                    if cancel_event.is_set():
                        self._retire(proc)
                        return
                    if time.monotonic() >= deadline:
                        timed_out.set()
                        self._retire(proc)
                        return

            watcher = threading.Thread(target=retire_on_cancel_or_timeout, daemon=True)
            watcher.start()
            try:
                proc.stdin.write(json.dumps(dict(payload), separators=(",", ":")) + "\n")
                proc.stdin.flush()
                yield from self._read_result(proc, payload, cancel_event, timed_out)
            finally:
                finished.set()
                watcher.join(timeout=1.0)
                with self._state_lock:
                    self._busy = False
                    if proc.poll() is not None and self._proc is proc:
                        self._proc = None
                        self._ready = False
                if proc.poll() is not None:
                    self._close_pipes(proc)
        finally:
            self._lock.release()

    def _read_result(
        self,
        proc: subprocess.Popen[str],
        payload: Mapping[str, Any],
        cancel_event: threading.Event,
        timed_out: threading.Event,
    ) -> Iterator[dict[str, Any]]:
        assert proc.stdout is not None
        started = False
        completed = False
        plain_lines: list[str] = []
        stream_tokens = 0
        legacy_stats: dict[str, Any] | None = None
        telemetry: Mapping[str, Any] | None = None

        def marker_json(text: str, prefix: str) -> Any:
            try:
                return json.loads(text[len(prefix) :])
            except json.JSONDecodeError as exc:
                raise ContractError(
                    "backend_protocol_error",
                    f"worker emitted invalid JSON after {prefix.strip()}",
                    status=502,
                ) from exc

        while True:
            line = proc.stdout.readline()
            if not line:
                if cancel_event.is_set():
                    raise ContractError("request_canceled", "request canceled", status=409)
                if timed_out.is_set():
                    raise ContractError("backend_timeout", "worker request timed out", status=504)
                if proc.poll() is not None:
                    raise ContractError("backend_died", "worker died mid-request", status=502)
                continue
            text = line.rstrip("\n")
            if "[INFER_SERVER_RESULT_START]" in text:
                started = True
                continue
            if "[INFER_SERVER_RESULT_END]" in text:
                completed = True
                break
            if "[INFER_SERVER_ERROR]" in text:
                raise ContractError("backend_error", text, status=502)
            if not started:
                continue
            if text.startswith("[STREAM_BEGIN] "):
                yield {"event": "stream_begin", "data": marker_json(text, "[STREAM_BEGIN] ")}
                continue
            prefix = next((item for item in _STREAM_PREFIXES if text.startswith(item)), None)
            if prefix:
                stream_tokens += 1
                yield {"event": "stream_tok", "data": marker_json(text, prefix)}
                continue
            if text.startswith("[nat-trace] "):
                yield {"event": "nat_trace", "data": marker_json(text, "[nat-trace] ")}
                continue
            if text.startswith("[sat-trace] "):
                yield {"event": "sat_trace", "data": marker_json(text, "[sat-trace] ")}
                continue
            if text.startswith("[decode-telemetry] "):
                telemetry = marker_json(text, "[decode-telemetry] ")
                continue
            match = _STAT_RE.search(text)
            if match:
                legacy_stats = {
                    "seconds": float(match.group("sec")),
                    "tokens": int(match.group("tok")),
                    "tok_s": float(match.group("tps")),
                }
                continue
            if text.startswith("[") or text.startswith("Generating"):
                continue
            plain_lines.append(text)

        if not completed:
            raise ContractError("backend_protocol_error", "worker result did not terminate", status=502)

        # The existing AR runtime predates proof telemetry, but an AR decode has
        # one committed token per step by construction. SAT/NAT are never
        # inferred: future runtimes must emit the explicit telemetry marker.
        if telemetry is None and self.mode == "ar":
            token_count = int((legacy_stats or {}).get("tokens", stream_tokens))
            elapsed = float((legacy_stats or {}).get("seconds", 0.0))
            tok_s = float((legacy_stats or {}).get("tok_s", token_count / elapsed if elapsed else 0.0))
            telemetry = {
                "schema": TELEMETRY_SCHEMA,
                "mode": "ar",
                "decode_steps": token_count,
                "committed_tokens": token_count,
                "gate_decisions": 0,
                "gate_commit_1": 0,
                "gate_commit_2": 0,
                "gate_commit_2_rate": 0.0,
                "sat_tokens_committed": 0,
                "ar_realign_tokens": 0,
                "prompt_align_tokens": 0,
                "total_emitted_tokens": token_count,
                "core_forwards": token_count,
                "refinement_steps": 0,
                "terminal_boundary_partial": False,
                "terminal_realign_skipped": False,
                "elapsed_s": elapsed,
                "tok_s": tok_s,
            }
        if telemetry is None:
            raise ContractError(
                "missing_backend_telemetry",
                f"{self.mode} runtime did not emit [decode-telemetry] proof",
                status=502,
            )
        validated = validate_telemetry(self.mode, telemetry)
        stats = dict(legacy_stats or {})
        stats["telemetry"] = validated
        yield {"event": "stats", "stats": stats}
        yield {
            "event": "backend_done",
            "text": "\n".join(plain_lines).strip(),
            "stats": stats,
        }
