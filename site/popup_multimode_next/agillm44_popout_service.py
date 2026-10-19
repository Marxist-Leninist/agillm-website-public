#!/usr/bin/env python3
"""HTTP-independent pop-out service core with strict four-mode routing."""

from __future__ import annotations

import threading
import time
import uuid
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from agillm44_popout_contract import (
    CANONICAL_MODES,
    CONTRACT_SCHEMA,
    MODE_SEMANTICS,
    TELEMETRY_SCHEMA,
    CheckpointManifest,
    ContractError,
    canonical_mode,
    parse_request_id,
    runtime_payload,
    validate_sampler,
    validate_telemetry,
)


class ModeBackend(Protocol):
    mode: str

    def health(self) -> Mapping[str, Any]: ...

    def ensure_ready(self) -> bool: ...

    def stream(
        self,
        payload: Mapping[str, Any],
        cancel_event: threading.Event,
    ): ...


@dataclass(frozen=True)
class PreparedRequest:
    request_id: str
    mode: str
    requested: Mapping[str, Any]
    runtime_payload: Mapping[str, Any]
    backend: ModeBackend
    cancel_event: threading.Event


class RequestRegistry:
    """Request-scoped cancellation with short cancel-before-register tombstones."""

    def __init__(self, tombstone_s: float = 120.0) -> None:
        self._lock = threading.RLock()
        self._active: dict[str, tuple[threading.Event, str]] = {}
        self._tombstones: dict[str, float] = {}
        self._tombstone_s = float(tombstone_s)

    def _prune(self) -> None:
        now = time.monotonic()
        for request_id, expiry in list(self._tombstones.items()):
            if expiry <= now:
                self._tombstones.pop(request_id, None)

    def register(self, request_id: str, mode: str) -> threading.Event:
        with self._lock:
            self._prune()
            if request_id in self._active:
                raise ContractError(
                    "duplicate_request_id",
                    "request_id is already active",
                    status=409,
                    extra={"request_id": request_id},
                )
            event = threading.Event()
            if request_id in self._tombstones:
                self._tombstones.pop(request_id, None)
                event.set()
            self._active[request_id] = (event, mode)
            return event

    def finish(self, request_id: str) -> None:
        with self._lock:
            self._active.pop(request_id, None)
            self._tombstones.pop(request_id, None)

    def cancel(self, request_id: str) -> tuple[int, dict[str, Any]]:
        with self._lock:
            self._prune()
            item = self._active.get(request_id)
            if item is not None:
                event, mode = item
                event.set()
                return 200, {
                    "ok": True,
                    "canceling": True,
                    "request_id": request_id,
                    "mode": mode,
                    "scope": "request_only",
                }
            self._tombstones[request_id] = time.monotonic() + self._tombstone_s
            return 202, {
                "ok": True,
                "canceling": True,
                "queued": True,
                "request_id": request_id,
                "scope": "request_only",
                "note": "cancel remembered before request registration",
            }

    def counts(self) -> dict[str, Any]:
        with self._lock:
            self._prune()
            by_mode = {mode: 0 for mode in CANONICAL_MODES}
            for _, mode in self._active.values():
                by_mode[mode] += 1
            return {"active": len(self._active), "active_by_mode": by_mode}


class PopoutService:
    def __init__(
        self,
        manifest: CheckpointManifest,
        backends: Mapping[str, ModeBackend],
    ) -> None:
        unknown = sorted(set(backends) - set(CANONICAL_MODES))
        if unknown:
            raise ContractError(
                "invalid_backend_config",
                "backend registry contains unknown modes",
                status=503,
                extra={"unknown_modes": unknown},
            )
        for mode, backend in backends.items():
            if getattr(backend, "mode", None) != mode:
                raise ContractError(
                    "invalid_backend_config",
                    "backend registry key and backend.mode differ",
                    status=503,
                    extra={"registry_mode": mode, "backend_mode": getattr(backend, "mode", None)},
                )
        self.manifest = manifest
        self.backends = dict(backends)
        self.requests = RequestRegistry()
        self.started_at = time.time()

    def capabilities(self) -> dict[str, Any]:
        report: dict[str, Any] = {}
        for mode in CANONICAL_MODES:
            declaration = self.manifest.modes[mode]
            backend = self.backends.get(mode)
            if backend is None:
                backend_health = {"ready": False}
            else:
                try:
                    backend_health = dict(backend.health())
                except Exception:
                    backend_health = {"ready": False, "health_error": True}
            backend_ready = bool(backend_health.get("ready")) and bool(backend_health.get("protocol_ready", True))
            backend_configured = backend is not None
            available = declaration.trained and backend_configured and backend_ready
            if not declaration.trained:
                reason_code = "checkpoint_capability_untrained"
            elif not backend_configured:
                reason_code = "backend_not_configured"
            elif not backend_ready:
                reason_code = "backend_not_ready"
            else:
                reason_code = "available"
            report[mode] = {
                "trained": declaration.trained,
                "state_binding": declaration.state_binding,
                "telemetry_contract": declaration.telemetry_contract,
                "training_receipt_id": declaration.training_receipt_id,
                "backend_configured": backend_configured,
                "worker_ready": backend_ready,
                "available": available,
                "reason_code": reason_code,
                "reason": declaration.reason,
                "semantics": dict(MODE_SEMANTICS[mode]),
                "backend_health": backend_health,
            }
        return report

    def health(self) -> dict[str, Any]:
        capabilities = self.capabilities()
        trained = [mode for mode in CANONICAL_MODES if capabilities[mode]["trained"]]
        available = [mode for mode in CANONICAL_MODES if capabilities[mode]["available"]]
        return {
            "ok": True,
            "contract_schema": CONTRACT_SCHEMA,
            "capability_schema": "agillm.http-capabilities.v1",
            "telemetry_schema": TELEMETRY_SCHEMA,
            "ready": bool(available),
            "fully_ready": bool(trained) and set(available) == set(trained),
            "ckpt": Path(self.manifest.artifact_path).name,
            "strict_no_fallback": True,
            "fallback": False,
            "canonical_modes": list(CANONICAL_MODES),
            "trained_modes": trained,
            "available_modes": available,
            "model": {
                **dict(self.manifest.model_metadata),
                "id": self.manifest.model_id,
                "artifact_path": self.manifest.artifact_path,
                "checkpoint_name": Path(self.manifest.artifact_path).name,
                "artifact_sha256": self.manifest.artifact_sha256,
                "export_schema": self.manifest.artifact_schema,
                "tokenizer_sha256": self.manifest.tokenizer_sha256,
            },
            "training_receipt": dict(self.manifest.training_receipt) if self.manifest.training_receipt else None,
            "capabilities": capabilities,
            "requests": self.requests.counts(),
            "uptime_s": round(time.time() - self.started_at, 3),
        }

    def _capability_error(self, mode: str) -> ContractError:
        capability = self.capabilities()[mode]
        return ContractError(
            "capability_unavailable",
            f"mode '{mode}' is recognized but not trained by this checkpoint",
            status=422,
            extra={
                "requested_mode": mode,
                "model_id": self.manifest.model_id,
                "export_schema": self.manifest.artifact_schema,
                "available_modes": self.health()["available_modes"],
                "capability": capability,
                "strict_no_fallback": True,
            },
        )

    def prepare(self, payload: Mapping[str, Any]) -> PreparedRequest:
        mode = canonical_mode(payload)
        validate_sampler(payload, mode)
        declaration = self.manifest.modes[mode]
        if not declaration.trained:
            # This check is intentionally before request registration or any
            # backend lookup/spawn. An untrained mode performs zero model work.
            raise self._capability_error(mode)
        # Fully validate and bound the public body before a worker health check
        # can spawn, warm, queue, or otherwise consume model resources.
        child_payload = runtime_payload(payload, mode)
        request_id = parse_request_id(payload.get("request_id")) or uuid.uuid4().hex
        backend = self.backends.get(mode)
        if backend is None:
            raise ContractError(
                "backend_unavailable",
                f"trained mode '{mode}' has no configured backend",
                status=503,
                extra={"requested_mode": mode, "strict_no_fallback": True},
            )
        if not backend.ensure_ready():
            raise ContractError(
                "backend_unavailable",
                f"trained mode '{mode}' backend is not ready",
                status=503,
                extra={
                    "requested_mode": mode,
                    "backend_health": dict(backend.health()),
                    "strict_no_fallback": True,
                },
            )

        cancel_event = self.requests.register(request_id, mode)
        if cancel_event.is_set():
            self.requests.finish(request_id)
            raise ContractError(
                "request_canceled",
                "request was canceled before registration",
                status=409,
                extra={"request_id": request_id},
            )
        return PreparedRequest(
            request_id=request_id,
            mode=mode,
            requested=dict(payload),
            runtime_payload=child_payload,
            backend=backend,
            cancel_event=cancel_event,
        )

    def execute(
        self,
        prepared: PreparedRequest,
        emit: Callable[[dict[str, Any]], None],
    ) -> tuple[int, dict[str, Any]]:
        mode = prepared.mode
        request_id = prepared.request_id
        terminal: dict[str, Any] | None = None
        last_stats: Mapping[str, Any] | None = None
        try:
            emit(
                {
                    "event": "start",
                    "ok": True,
                    "request_id": request_id,
                    "model_id": self.manifest.model_id,
                    "model_sha256": self.manifest.artifact_sha256,
                    "requested_mode": mode,
                    "effective_mode": mode,
                    "strict_no_fallback": True,
                }
            )
            for event in prepared.backend.stream(prepared.runtime_payload, prepared.cancel_event):
                if prepared.cancel_event.is_set():
                    raise ContractError("request_canceled", "request canceled", status=409)
                if not isinstance(event, Mapping):
                    raise ContractError("backend_protocol_error", "backend event must be an object", status=502)
                event_type = event.get("event")
                if event_type == "backend_done":
                    stats = event.get("stats")
                    if not isinstance(stats, Mapping) or not isinstance(stats.get("telemetry"), Mapping):
                        raise ContractError("missing_backend_telemetry", "backend done event lacks telemetry", status=502)
                    validated = validate_telemetry(mode, stats["telemetry"])
                    if last_stats is None:
                        last_stats = {**dict(stats), "telemetry": validated}
                        emit({"event": "stats", "stats": last_stats})
                    terminal = {
                        "event": "done",
                        "ok": True,
                        "request_id": request_id,
                        "model_id": self.manifest.model_id,
                        "model_sha256": self.manifest.artifact_sha256,
                        "requested_mode": mode,
                        "effective_mode": mode,
                        "text": str(event.get("text") or ""),
                        "stats": dict(last_stats),
                        "strict_no_fallback": True,
                    }
                    emit(terminal)
                    break
                if event_type == "stats":
                    stats = event.get("stats")
                    if not isinstance(stats, Mapping) or not isinstance(stats.get("telemetry"), Mapping):
                        raise ContractError("missing_backend_telemetry", "stats event lacks telemetry", status=502)
                    last_stats = {**dict(stats), "telemetry": validate_telemetry(mode, stats["telemetry"])}
                    emit({"event": "stats", "stats": last_stats})
                    continue
                if event_type == "sat_trace" and mode not in {"sat_fixed", "sat_variable"}:
                    raise ContractError("backend_protocol_error", "SAT trace emitted for non-SAT route", status=502)
                if event_type == "nat_trace" and mode != "nat":
                    raise ContractError("backend_protocol_error", "NAT trace emitted for non-NAT route", status=502)
                if event_type not in {"stream_begin", "stream_tok", "sat_trace", "nat_trace"}:
                    raise ContractError(
                        "backend_protocol_error",
                        f"unexpected backend event: {event_type!r}",
                        status=502,
                    )
                emit(dict(event))
            if terminal is None:
                raise ContractError("backend_protocol_error", "backend ended without done", status=502)
            return 200, terminal
        except ContractError as exc:
            terminal = {
                "event": "error",
                **exc.payload(),
                "request_id": request_id,
                "model_id": self.manifest.model_id,
                "requested_mode": mode,
                "effective_mode": mode,
                "strict_no_fallback": True,
            }
            emit(terminal)
            return exc.status, terminal
        except Exception as exc:
            terminal = {
                "event": "error",
                "ok": False,
                "error": "internal pop-out backend error",
                "error_code": "internal_error",
                "request_id": request_id,
                "model_id": self.manifest.model_id,
                "requested_mode": mode,
                "effective_mode": mode,
                "strict_no_fallback": True,
            }
            emit(terminal)
            return 500, terminal
        finally:
            self.requests.finish(request_id)

    def cancel(self, request_id_value: Any) -> tuple[int, dict[str, Any]]:
        request_id = parse_request_id(request_id_value)
        if request_id is None:
            raise ContractError(
                "request_id_required",
                "cancel requires request_id; global cancellation is forbidden",
            )
        return self.requests.cancel(request_id)
