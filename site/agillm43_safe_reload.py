#!/usr/bin/env python3
"""Fail-closed preflight and signal gate for AGILLM4.3 bridge reloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import signal
import sys


DEFAULT_ENV = pathlib.Path("/opt/agillm43_netserve/bridge.env")
ALLOWED_ROOTS = (
    pathlib.Path("/mnt/geth-vol1/agillm43/ckpts/artifacts"),
    pathlib.Path("/opt/agillm43_offload/ckpts"),
)
REJECTED_ROOTS = (
    pathlib.Path("/mnt/geth-vol1/agillm43_recovery_mirror"),
)
FULL_SHARD_NAMES = {
    "ar.pt",
    "nat.pt",
    "opt.pt",
    "sat.pt",
    "scaler.pt",
    *(f"core_block_{i:04d}.pt" for i in range(28)),
}
STEP_FILE = re.compile(
    r"^pretrain_step(?P<step>[0-9]{8})_from[0-9]{8}_[0-9]{8}T(?:[0-9]{4}Z|[0-9]{6}_[0-9]+_[0-9]+_[0-9]+)[.]pt$"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GateError(RuntimeError):
    pass


def load_env(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise GateError(f"invalid env line {number}")
        key, value = line.split("=", 1)
        if key in values:
            raise GateError(f"duplicate env key: {key}")
        values[key] = value
    return values


def under(path: pathlib.Path, root: pathlib.Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(env_path: pathlib.Path) -> dict[str, object]:
    values = load_env(env_path)
    checkpoint = values.get("AGILLM_BRIDGE_CKPT", "")
    pinned = values.get("AGILLM_BRIDGE_PINNED_CKPT", "")
    expected_sha = values.get("AGILLM_BRIDGE_PINNED_SHA256", "").lower()

    if not checkpoint or checkpoint != pinned:
        raise GateError("CKPT and PINNED_CKPT must be present and identical")
    if not SHA256.fullmatch(expected_sha):
        raise GateError("AGILLM_BRIDGE_PINNED_SHA256 is missing or invalid")

    raw_path = pathlib.Path(checkpoint)
    if not raw_path.is_absolute() or not raw_path.is_file():
        raise GateError("pinned checkpoint is not an absolute regular file")
    path = raw_path.resolve(strict=True)
    roots = tuple(root.resolve(strict=True) for root in ALLOWED_ROOTS)
    rejected = tuple(root.resolve(strict=True) for root in REJECTED_ROOTS)
    if any(under(path, root) for root in rejected):
        raise GateError("status-mirror paths can never serve inference")
    if not any(under(path, root) for root in roots):
        raise GateError("pinned checkpoint is outside approved serving roots")
    if path.stat().st_size < 1024 * 1024:
        raise GateError("pinned checkpoint is implausibly small")

    actual_sha = sha256_file(path)
    if actual_sha != expected_sha:
        raise GateError("pinned checkpoint sha256 does not match approval")

    match = STEP_FILE.fullmatch(path.name)
    if path.name.startswith("pretrain_step") and not match:
        raise GateError("unsupported rolling checkpoint filename; refusing legacy fallback")
    provenance_path = pathlib.Path(str(path)[:-3] + ".provenance.json")
    tokenizer_path = pathlib.Path(str(path) + ".tokenizer.json")
    provenance: dict[str, object] | None = None

    if match:
        for sidecar in (provenance_path, tokenizer_path):
            if not sidecar.is_file() or sidecar.stat().st_size <= 0:
                raise GateError(f"required sidecar missing: {sidecar.name}")
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise GateError("provenance sidecar is unreadable") from exc
        filename_step = int(match.group("step"))
        if int(provenance.get("step", -1)) != filename_step:
            raise GateError("provenance step does not match checkpoint filename")

    if provenance and provenance.get("checkpoint_type") == "full":
        shard_dir = pathlib.Path(str(path) + ".shards")
        if not shard_dir.is_dir():
            raise GateError("full checkpoint shard directory is missing")
        missing = sorted(
            name
            for name in FULL_SHARD_NAMES
            if not (shard_dir / name).is_file() or (shard_dir / name).stat().st_size <= 0
        )
        if missing:
            preview = ",".join(missing[:4])
            raise GateError(f"full checkpoint shards are incomplete: {preview}")

    return {
        "status": "pass",
        "checkpoint": path.name,
        "sha256": actual_sha,
        "checkpoint_type": provenance.get("checkpoint_type") if provenance else "legacy",
        "shards_required": bool(provenance and provenance.get("checkpoint_type") == "full"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=pathlib.Path, default=DEFAULT_ENV)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    try:
        receipt = validate(args.env)
        if not args.check_only:
            if not args.pid or args.pid <= 1:
                raise GateError("valid bridge pid is required")
            cmdline = pathlib.Path(f"/proc/{args.pid}/cmdline").read_bytes()
            if b"agillm43_http_bridge.py" not in cmdline:
                raise GateError("target pid is not the AGILLM bridge")
            os.kill(args.pid, signal.SIGHUP)
            receipt["signal"] = "SIGHUP"
            receipt["pid"] = args.pid
        print("AGILLM_RELOAD_GATE " + json.dumps(receipt, sort_keys=True), flush=True)
        return 0
    except (GateError, OSError, ValueError) as exc:
        print(f"AGILLM_RELOAD_GATE REFUSED: {exc}", file=sys.stderr, flush=True)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
