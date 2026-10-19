#!/usr/bin/env python3
"""Launch a pinned, isolated AGILLM pop-out bridge."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
from typing import Mapping

from agillm44_native_jsonl_backend import NativeJsonlBackend
from agillm44_popout_contract import CANONICAL_MODES, CheckpointManifest, ContractError
from agillm44_popout_http import serve
from agillm44_popout_service import PopoutService


RUNTIME_CONFIG_SCHEMA = "agillm.popout-runtime-config.v1"


def _load_runtime_config(
    path: str,
    manifest: CheckpointManifest,
) -> dict[str, NativeJsonlBackend]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, Mapping) or raw.get("schema") != RUNTIME_CONFIG_SCHEMA:
        raise ContractError(
            "invalid_runtime_config",
            f"runtime config schema must be {RUNTIME_CONFIG_SCHEMA}",
            status=503,
        )
    preflight_raw = raw.get("preflight_command")
    if not isinstance(preflight_raw, list) or not preflight_raw:
        raise ContractError(
            "invalid_runtime_config",
            "a manifest-binding loader preflight_command is required",
            status=503,
        )
    if isinstance(preflight_raw, list) and preflight_raw:
        preflight = [
            str(part)
            .replace("{checkpoint}", manifest.artifact_path)
            .replace("{checkpoint_sha256}", manifest.artifact_sha256)
            .replace("{tokenizer_sha256}", manifest.tokenizer_sha256 or "")
            for part in preflight_raw
        ]
        try:
            result = subprocess.run(
                preflight,
                check=False,
                capture_output=True,
                text=True,
                timeout=float(raw.get("preflight_timeout_s", 300)),
                env={
                    **os.environ,
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_HUB_OFFLINE": "1",
                },
            )
        except Exception as exc:
            raise ContractError(
                "preflight_failed",
                f"checkpoint/tokenizer preflight could not run: {exc}",
                status=503,
            ) from exc
        if result.returncode != 0:
            raise ContractError(
                "preflight_failed",
                "checkpoint/tokenizer preflight returned non-zero",
                status=503,
                extra={"returncode": result.returncode},
            )
        try:
            report = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise ContractError(
                "preflight_failed",
                "checkpoint/tokenizer preflight did not emit final JSON",
                status=503,
            ) from exc
        manifest.validate_preflight_report(report)
    modes = raw.get("modes")
    if not isinstance(modes, Mapping):
        raise ContractError("invalid_runtime_config", "runtime config modes must be an object", status=503)
    unknown = sorted(set(modes) - set(CANONICAL_MODES))
    if unknown:
        raise ContractError(
            "invalid_runtime_config",
            "runtime config contains unknown modes",
            status=503,
            extra={"unknown_modes": unknown},
        )

    backends: dict[str, NativeJsonlBackend] = {}
    for mode, mode_raw in modes.items():
        if not manifest.modes[mode].trained:
            raise ContractError(
                "untrained_backend_configured",
                f"runtime config attempts to enable untrained mode '{mode}'",
                status=503,
            )
        if not isinstance(mode_raw, Mapping):
            raise ContractError("invalid_runtime_config", f"modes.{mode} must be an object", status=503)
        command_raw = mode_raw.get("command")
        if not isinstance(command_raw, list) or not command_raw:
            raise ContractError("invalid_runtime_config", f"modes.{mode}.command must be a non-empty list", status=503)
        command = [
            str(part).replace("{checkpoint}", manifest.artifact_path).replace("{canonical_mode}", mode)
            for part in command_raw
        ]
        if manifest.artifact_path not in command:
            raise ContractError(
                "invalid_runtime_config",
                f"modes.{mode}.command must load the manifest-pinned checkpoint via {{checkpoint}}",
                status=503,
            )
        env_raw = mode_raw.get("env", {})
        if not isinstance(env_raw, Mapping):
            raise ContractError("invalid_runtime_config", f"modes.{mode}.env must be an object", status=503)
        env = {str(key): str(value) for key, value in env_raw.items()}
        backends[mode] = NativeJsonlBackend(
            mode=mode,
            command=command,
            env=env,
            startup_timeout_s=float(mode_raw.get("startup_timeout_s", 600)),
            request_timeout_s=float(mode_raw.get("request_timeout_s", 1800)),
            queue_timeout_s=float(mode_raw.get("queue_timeout_s", 30)),
        )
    return backends


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--host", default=os.environ.get("AGILLM_POPOUT_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("AGILLM_POPOUT_PORT", "18597")))
    parser.add_argument(
        "--token-env",
        default="AGILLM_POPOUT_TOKEN",
        help="environment variable containing the bridge token",
    )
    parser.add_argument(
        "--heartbeat-s",
        type=float,
        default=float(os.environ.get("AGILLM_POPOUT_HEARTBEAT_S", "10")),
    )
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = CheckpointManifest.load(args.manifest)
    manifest.verify_artifact()
    backends = _load_runtime_config(args.runtime_config, manifest)
    service = PopoutService(manifest, backends)
    if args.verify_only:
        print(json.dumps(service.health(), indent=2, sort_keys=True))
        return 0
    token = (os.environ.get(args.token_env) or "").strip()
    if not token:
        raise SystemExit(f"refusing to serve without non-empty {args.token_env}")

    # Listen immediately so /health truthfully reports warming/not-ready while
    # each configured mode warms in isolation.
    for backend in backends.values():
        threading.Thread(target=backend.ensure_ready, daemon=True).start()
    serve(
        service,
        host=args.host,
        port=args.port,
        token=token,
        heartbeat_s=args.heartbeat_s,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
