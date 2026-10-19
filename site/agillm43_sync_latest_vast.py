#!/usr/bin/env python3
"""Safely deploy the newest completed AGILLM4.3 checkpoint directly from Vast.

The trainer remains authoritative only for candidate discovery and immutable
checkpoint bytes. A candidate is exported and SHA-256 verified on Vast,
transferred into an isolated GETH staging directory, verified again on GETH,
and then handed to the existing transactional promotion/rollback gate.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import os
import pathlib
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import urllib.request
from typing import Any

NETSERVE = pathlib.Path("/opt/agillm43_netserve")
STATUS = NETSERVE / "agillm_hf_status.json"
PROMOTER = NETSERVE / "agillm43_promote_hf_verified.py"
SUPERVISOR = NETSERVE / "agillm43_deploy_supervisor.py"
ROUTE_STATE = pathlib.Path("/var/lib/agillm43-vast-route/current.json")
SSH_KEY = pathlib.Path("/root/.ssh/goddess_vast")
SAVE_DIR = pathlib.PurePosixPath("/workspace/agillm43_production_v14/production_quality_v18_hard_spike_gates/checkpoints")
SAVE_ROOT = pathlib.PurePosixPath("/workspace/agillm43_production_v14/production_quality_v18_hard_spike_gates")
SAVE_DIR_NAME_RE = re.compile(r"^checkpoints(?:_[A-Za-z0-9]+)*$")
# The trainer starts a new leg in a sibling directory (checkpoints_continue400B_from200B,
# etc.). SAVE_DIR above is only the legacy default; the live leg is discovered per run and
# must still be a direct child of SAVE_ROOT whose name matches SAVE_DIR_NAME_RE.
_ACTIVE_SAVE_DIR = SAVE_DIR


def active_save_dir() -> pathlib.PurePosixPath:
    return _ACTIVE_SAVE_DIR
EXPORT_ROOT = pathlib.PurePosixPath("/workspace/.agillm43_direct_serving_export")
BUILDER = pathlib.PurePosixPath("/workspace/agillm43-build-export-v2.py")
RUNTIME = pathlib.PurePosixPath(
    "/workspace/agillm43_inference_runtime_pins/runtime_64f2ac73bd7c3aff489ff3caf795ca5daffdefc28932d94a2e71805a6461bc57.py"
)
TRUSTED_BUILDER_SHA = "c5d3fbeacfcf15b42117b3599d2456c89c960491f57cf787afb8c6b255041371"
TRUSTED_RUNTIME_SHA = "64f2ac73bd7c3aff489ff3caf795ca5daffdefc28932d94a2e71805a6461bc57"
EXPECTED_LABEL = "agillm43-retrain-134B"
STAGING_ROOT = pathlib.Path("/mnt/geth-vol1/agillm43/ckpts/artifacts/full/vast_direct")
SERVE_ROOT = pathlib.Path("/mnt/geth-vol1/agillm43/ckpts/artifacts/full/hf_verified")
# Share the HF deployment lock so direct and archival promotion can never race.
LOCK = pathlib.Path("/run/lock/agillm43-hf-autodeploy.lock")
HOST_RE = re.compile(r"^ssh[0-9]+[.]vast[.]ai$")
IPV4_RE = re.compile(r"^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])[.]){3}(?:25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9]?[0-9])$")


def _host_is_allowed(host: str, address: str) -> bool:
    """Accept the Vast SSH proxy hostname, or a direct IPv4 endpoint that is
    exactly the IP the local Vast inventory observed for this instance.

    The direct route is written by agillm43-vast-route-refresh whenever the
    instance exposes a direct SSH port; rejecting it made autodeploy fail with
    "Vast SSH endpoint failed validation" on every direct-route refresh.
    Pinning host == observed ip keeps the identity check as strict as the DNS
    path (gethostbyname of a literal returns the literal)."""
    if HOST_RE.fullmatch(host):
        return True
    return bool(IPV4_RE.fullmatch(host)) and host == address
CHECKPOINT_RE = re.compile(r"^pretrain_step([0-9]{8})_from[0-9]{8}_[0-9]{8}T[0-9]{4}Z[.]pt$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
SETTLE_SECONDS = 180
ROUTE_MAX_AGE_SECONDS = 600
MIN_RESERVE_BYTES = 12 * 1024**3
MIN_SHARD_ENTRIES = 30
MAX_SHARD_ENTRIES = 128
GIB = 1024**3


class DirectSyncError(RuntimeError):
    pass


def now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def now_iso() -> str:
    return now().isoformat()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise DirectSyncError(f"unsafe or missing file: {path}")
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise DirectSyncError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def fsync_directory(path: pathlib.Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: pathlib.Path, value: object, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DirectSyncError(f"required JSON is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise DirectSyncError(f"malformed JSON: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DirectSyncError(f"JSON is not an object: {path}")
    return value


def parse_utc(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp has no timezone")
    return parsed.astimezone(dt.timezone.utc)


def validate_root_file(path: pathlib.Path, *, private: bool = False) -> os.stat_result:
    info = path.stat()
    if info.st_uid != 0 or not stat.S_ISREG(info.st_mode):
        raise DirectSyncError(f"file is not a root-owned regular file: {path}")
    forbidden = 0o077 if private else 0o022
    if stat.S_IMODE(info.st_mode) & forbidden:
        raise DirectSyncError(f"unsafe permissions on {path}: {oct(stat.S_IMODE(info.st_mode))}")
    return info


def load_route() -> dict[str, Any]:
    validate_root_file(ROUTE_STATE)
    validate_root_file(SSH_KEY, private=True)
    route = read_json(ROUTE_STATE)
    try:
        instance_id = int(route["instance_id"])
        host = str(route["ssh_host"])
        port = int(route["ssh_port"])
        label = str(route["label"])
        status = str(route["actual_status"])
        address = str(route["ip"])
        observed_at = parse_utc(str(route["observed_at"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise DirectSyncError(f"invalid Vast route state: {exc}") from exc
    if route.get("schema") != "agillm43.vast-route.v1":
        raise DirectSyncError("unexpected Vast route schema")
    if instance_id <= 0 or label != EXPECTED_LABEL or status != "running":
        raise DirectSyncError("Vast route does not identify the running AGILLM trainer")
    if not _host_is_allowed(host, address) or not (1 <= port <= 65535):
        raise DirectSyncError("Vast SSH endpoint failed validation")
    route_age = (now() - observed_at).total_seconds()
    if route_age < -60 or route_age > ROUTE_MAX_AGE_SECONDS:
        raise DirectSyncError(f"Vast route state is stale or future-dated: age={route_age:.1f}s")
    try:
        resolved = socket.gethostbyname(host)
    except OSError as exc:
        raise DirectSyncError(f"cannot resolve current Vast SSH host: {exc}") from exc
    if resolved != address:
        raise DirectSyncError(f"Vast route DNS identity mismatch: {resolved} != {address}")
    return {
        "instance_id": instance_id,
        "host": host,
        "port": port,
        "label": label,
        "ip": address,
        "observed_at": observed_at.isoformat(),
        "route_age_s": round(route_age, 3),
    }


def ssh_transport(route: dict[str, Any]) -> list[str]:
    return [
        "ssh",
        "-i", str(SSH_KEY),
        "-p", str(route["port"]),
        "-o", "BatchMode=yes",
        "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=15",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=8",
    ]


def ssh_run(route: dict[str, Any], command: str, *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    invocation = ssh_transport(route) + [f"root@{route['host']}", command]
    process = subprocess.run(invocation, text=True, capture_output=True, timeout=timeout, check=False)
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip()[-4000:]
        raise DirectSyncError(f"Vast command failed rc={process.returncode}: {detail}")
    return process


def parse_last_json(text: str, label: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise DirectSyncError(f"{label} returned no JSON object")


def remote_latest(route: dict[str, Any]) -> dict[str, Any]:
    code = f'''import json, pathlib, re, time
root = pathlib.Path({str(SAVE_ROOT)!r}).resolve()
name_ok = re.compile(r"^checkpoints(?:_[A-Za-z0-9]+)*$")
cands = []
for d in sorted(root.iterdir()):
    if d.is_symlink() or not d.is_dir() or not name_ok.fullmatch(d.name):
        continue
    pointer = d / "training_latest.json"
    if pointer.is_file() and not pointer.is_symlink():
        cands.append((pointer.stat().st_mtime, d.name, d))
if not cands:
    raise SystemExit("no approved save directory carries training_latest.json")
cands.sort()
save = cands[-1][2].resolve()
latest = save / "training_latest.json"
data = json.loads(latest.read_text())
checkpoint = pathlib.Path(str(data["checkpoint_path"])).resolve()
if checkpoint.parent != save:
    raise SystemExit("checkpoint escaped save directory")
provenance = checkpoint.with_name(checkpoint.name[:-3] + ".provenance.json")
tokenizer = pathlib.Path(str(checkpoint) + ".tokenizer.json")
shards = pathlib.Path(str(checkpoint) + ".shards")
prov = json.loads(provenance.read_text())
files = [p for p in shards.iterdir() if p.is_file() and not p.is_symlink()]
st = checkpoint.stat()
print(json.dumps({{
    "save_dir": str(save),
    "step": int(data["step"]),
    "path": str(checkpoint),
    "file": checkpoint.name,
    "size": st.st_size,
    "mtime": st.st_mtime,
    "age_s": time.time() - st.st_mtime,
    "checkpoint_type": prov.get("checkpoint_type"),
    "provenance_step": int(prov.get("step", -1)),
    "effective_seen_tok": int(prov.get("effective_seen_tok", prov.get("seen_tok", -1))),
    "tokenizer_size": tokenizer.stat().st_size,
    "provenance_size": provenance.stat().st_size,
    "shard_count": len(files),
    "shard_bytes": sum(p.stat().st_size for p in files),
}}, sort_keys=True))'''
    process = ssh_run(route, "python3 -c " + shlex.quote(code), timeout=120)
    latest = parse_last_json(process.stdout, "Vast latest-checkpoint query")
    try:
        step = int(latest["step"])
        checkpoint_path = pathlib.PurePosixPath(str(latest["path"]))
        filename = str(latest["file"])
        age = float(latest["age_s"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DirectSyncError(f"invalid Vast latest-checkpoint response: {exc}") from exc
    match = CHECKPOINT_RE.fullmatch(filename)
    if not match or int(match.group(1)) != step:
        raise DirectSyncError("Vast checkpoint filename and step disagree")
    save_dir = pathlib.PurePosixPath(str(latest.get("save_dir", "")))
    if save_dir.parent != SAVE_ROOT or not SAVE_DIR_NAME_RE.fullmatch(save_dir.name):
        raise DirectSyncError(f"Vast save directory is not approved: {save_dir}")
    global _ACTIVE_SAVE_DIR
    _ACTIVE_SAVE_DIR = save_dir
    if checkpoint_path.parent != save_dir or checkpoint_path.name != filename:
        raise DirectSyncError("Vast checkpoint path escaped the approved save directory")
    if latest.get("checkpoint_type") != "full" or int(latest.get("provenance_step", -1)) != step:
        raise DirectSyncError("Vast checkpoint provenance is not a matching full checkpoint")
    if int(latest.get("shard_count", -1)) < 30 or int(latest.get("shard_bytes", 0)) < 5 * GIB:
        raise DirectSyncError("Vast checkpoint shard set is incomplete")
    latest["age_s"] = age
    return latest


def current_status() -> dict[str, Any]:
    try:
        return read_json(STATUS)
    except DirectSyncError:
        return {}


def health() -> dict[str, Any]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:8493/health", timeout=5) as response:
            value = json.loads(response.read())
            return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def validate_live(status: dict[str, Any]) -> bool:
    try:
        step = int(status["step"])
        digest = str(status["checkpoint_sha256"])
        filename = str(status["file"])
        package = pathlib.Path(str(status["package_path"]))
        if status.get("status") != "serving_verified" or status.get("serving") is not True:
            return False
        if not SHA_RE.fullmatch(digest) or pathlib.PurePath(filename).name != filename:
            return False
        if package.is_symlink() or package.parent.resolve() != SERVE_ROOT.resolve():
            return False
        marker = read_json(package / "SERVING_PACKAGE.json")
        if int(marker.get("step", -1)) != step or marker.get("checkpoint_sha256") != digest:
            return False
        if not (package / "checkpoints" / filename).is_file():
            return False
        bridge = health()
        return bridge.get("ready") is True and bridge.get("warming") is False and bridge.get("ckpt") == filename
    except Exception:
        return False


def remote_export_path(step: int) -> pathlib.PurePosixPath:
    return EXPORT_ROOT / f"step{step}"


def build_and_verify_remote_export(
    route: dict[str, Any], latest: dict[str, Any]
) -> dict[str, Any]:
    step = int(latest["step"])
    checkpoint = pathlib.PurePosixPath(str(latest["path"]))
    output = remote_export_path(step)
    q = shlex.quote
    command = f'''set -Eeuo pipefail
umask 077
exec 9>/workspace/.agillm43_direct_export.lock
flock -w 1800 9
BUILDER={q(str(BUILDER))}
RUNTIME={q(str(RUNTIME))}
CKPT={q(str(checkpoint))}
OUT={q(str(output))}
SAVE={q(str(active_save_dir()))}
EXPECTED_STEP={step}
EXPECTED_NAME={q(checkpoint.name)}
test "$(sha256sum "$BUILDER" | awk '{{print $1}}')" = {q(TRUSTED_BUILDER_SHA)}
OBSERVED_RUNTIME_SHA=$(sha256sum "$RUNTIME" | awk '{{print $1}}')
if test "$OBSERVED_RUNTIME_SHA" != {q(TRUSTED_RUNTIME_SHA)}; then
  printf 'inference runtime hash mismatch: expected=%s observed=%s path=%s\n' {q(TRUSTED_RUNTIME_SHA)} "$OBSERVED_RUNTIME_SHA" "$RUNTIME" >&2
  exit 1
fi
test -f "$CKPT" -a ! -L "$CKPT"
test -d "$CKPT.shards" -a ! -L "$CKPT.shards"
test -f "$CKPT.tokenizer.json" -a ! -L "$CKPT.tokenizer.json"
test -f "${{CKPT%.pt}}.provenance.json" -a ! -L "${{CKPT%.pt}}.provenance.json"
AGE=$(( $(date +%s) - $(stat -c %Y "$CKPT") ))
test "$AGE" -ge {SETTLE_SECONDS}
if test -d "$OUT"; then
  test ! -L "$OUT"
  cd "$OUT"
  sha256sum -c SHA256SUMS >/dev/null
else
  mkdir -p {q(str(EXPORT_ROOT))}
  nice -n 10 ionice -c3 /opt/conda/bin/python "$BUILDER" \
    --save-dir "$SAVE" --runtime "$RUNTIME" --output "$OUT" --checkpoint "$CKPT"
  cd "$OUT"
  sha256sum -c SHA256SUMS >/dev/null
fi
python3 - "$OUT" "$EXPECTED_STEP" "$EXPECTED_NAME" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
step = int(sys.argv[2]); expected_name = sys.argv[3]
manifest = json.loads((root / "recovery/export_manifest.json").read_text())
if manifest.get("status") != "validated_complete" or int(manifest.get("step", -1)) != step:
    raise SystemExit("invalid export manifest")
if manifest.get("checkpoint") != expected_name:
    raise SystemExit("export checkpoint identity mismatch")
files = [p for p in root.rglob("*") if p.is_file() and not p.is_symlink()]
sums = root / "SHA256SUMS"
members = sorted(
    [{{"path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size}} for p in files],
    key=lambda item: item["path"],
)
print(json.dumps({{
    "ok": True,
    "step": step,
    "checkpoint": manifest["checkpoint"],
    "checkpoint_sha256": manifest["checkpoint_sha256"],
    "runtime_sha256": manifest["runtime_sha256"],
    "effective_seen_tok": int(manifest.get("effective_seen_tok", -1)),
    "entries": int(manifest.get("entries", -1)),
    "files": len(files),
    "bytes": sum(p.stat().st_size for p in files),
    "sum_entries": sum(1 for line in sums.read_text().splitlines() if line.strip()),
    "sha256sums_sha256": hashlib.sha256(sums.read_bytes()).hexdigest(),
    "export_path": str(root),
    "members": members,
}}, sort_keys=True))
PY'''
    process = ssh_run(route, command, timeout=2400)
    summary = parse_last_json(process.stdout, "Vast direct export")
    checks = {
        "ok": summary.get("ok") is True,
        "step": int(summary.get("step", -1)) == step,
        "checkpoint": summary.get("checkpoint") == checkpoint.name,
        "checkpoint_sha": bool(SHA_RE.fullmatch(str(summary.get("checkpoint_sha256") or ""))),
        "runtime_sha": summary.get("runtime_sha256") == TRUSTED_RUNTIME_SHA,
        "entries": MIN_SHARD_ENTRIES <= int(summary.get("entries", -1)) <= MAX_SHARD_ENTRIES and int(summary.get("entries", -1)) == int(latest.get("shard_count", -2)),
        "files": int(summary.get("files", 0)) == int(summary.get("sum_entries", -1)) + 1,
        "sum_entries": int(summary.get("entries", -1)) + 7 <= int(summary.get("sum_entries", 0)) <= int(summary.get("entries", -1)) + 11,
        "bytes": 5 * GIB < int(summary.get("bytes", 0)) < 20 * GIB,
        "sums_sha": bool(SHA_RE.fullmatch(str(summary.get("sha256sums_sha256") or ""))),
        "path": pathlib.PurePosixPath(str(summary.get("export_path"))) == output,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise DirectSyncError("Vast direct export failed validation: " + ",".join(failed))
    members = summary.get("members")
    if not isinstance(members, list) or len(members) != int(summary.get("files", -1)):
        raise DirectSyncError("Vast direct export member list is missing or incomplete")
    seen: set[str] = set()
    member_bytes = 0
    for item in members:
        if not isinstance(item, dict):
            raise DirectSyncError("Vast direct export member is not an object")
        relative = str(item.get("path") or "")
        pure = pathlib.PurePosixPath(relative)
        try:
            size = int(item.get("bytes"))
        except (TypeError, ValueError) as exc:
            raise DirectSyncError(f"invalid Vast export member size: {relative}") from exc
        if not relative or relative.startswith("/") or ".." in pure.parts or pure.as_posix() != relative:
            raise DirectSyncError(f"unsafe Vast export member path: {relative}")
        if relative in seen or size < 0:
            raise DirectSyncError(f"duplicate or invalid Vast export member: {relative}")
        seen.add(relative)
        member_bytes += size
    if member_bytes != int(summary["bytes"]) or "SHA256SUMS" not in seen:
        raise DirectSyncError("Vast export member list disagrees with export identity")
    return summary


def staging_path(step: int, checkpoint_sha: str) -> pathlib.Path:
    return STAGING_ROOT / f".incoming-step{step}-{checkpoint_sha[:16]}"


def safe_stage(path: pathlib.Path) -> None:
    STAGING_ROOT.mkdir(parents=True, exist_ok=True, mode=0o750)
    if STAGING_ROOT.is_symlink() or STAGING_ROOT.stat().st_uid != 0:
        raise DirectSyncError("unsafe Vast-direct staging root")
    if path.parent.resolve() != STAGING_ROOT.resolve() or not re.fullmatch(
        r"[.]incoming-step[0-9]+-[0-9a-f]{16}", path.name
    ):
        raise DirectSyncError(f"unsafe staging path: {path}")
    path.mkdir(mode=0o700, exist_ok=True)
    if path.is_symlink() or path.stat().st_uid != 0:
        raise DirectSyncError(f"unsafe staging directory: {path}")


def rsync_export(
    route: dict[str, Any],
    remote_path: str,
    destination: pathlib.Path,
    members: list[dict[str, Any]],
) -> dict[str, Any]:
    safe_stage(destination)
    remote_root = pathlib.PurePosixPath(remote_path)
    if remote_root.parent != EXPORT_ROOT or not re.fullmatch(r"step[0-9]+", remote_root.name):
        raise DirectSyncError(f"unsafe remote export path: {remote_path}")
    existing_bytes = sum(
        item.stat().st_size
        for item in destination.rglob("*")
        if item.is_file() and not item.is_symlink()
    )
    transport = ssh_transport(route)
    ssh_command = " ".join(shlex.quote(part) for part in transport)
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in members:
        if not isinstance(item, dict):
            raise DirectSyncError("rsync member is not an object")
        relative = str(item.get("path") or "")
        pure = pathlib.PurePosixPath(relative)
        try:
            size = int(item.get("bytes"))
        except (TypeError, ValueError) as exc:
            raise DirectSyncError(f"invalid rsync member size: {relative}") from exc
        if (
            not relative or relative.startswith("/") or ".." in pure.parts
            or pure.as_posix() != relative or relative in seen or size < 0
        ):
            raise DirectSyncError(f"unsafe or duplicate rsync member: {relative}")
        seen.add(relative)
        ordered.append({"path": relative, "bytes": size})
    if not (MIN_SHARD_ENTRIES + 8 <= len(ordered) <= MAX_SHARD_ENTRIES + 12) or "SHA256SUMS" not in seen:
        raise DirectSyncError("rsync member set is incomplete")
    ordered.sort(key=lambda item: (-int(item["bytes"]), str(item["path"])))

    def transfer_one(item: dict[str, Any]) -> dict[str, Any]:
        relative = str(item["path"])
        source = f"root@{route['host']}:{remote_path.rstrip('/')}/./{relative}"
        command = [
            "rsync", "-aR", "--numeric-ids", "--partial", "--append-verify",
            "--timeout=600", "--info=stats2",
            "--chmod=Du=rwx,Dgo=,Fu=rw,Fgo=", "-e", ssh_command,
            source, str(destination) + "/",
        ]
        process = subprocess.run(command, text=True, capture_output=True, timeout=3600, check=False)
        if process.returncode != 0:
            detail = (process.stderr or process.stdout).strip()[-3000:]
            raise DirectSyncError(
                f"Vast-to-GETH rsync failed for {relative} rc={process.returncode}: {detail}"
            )
        return {
            "path": relative,
            "bytes": int(item["bytes"]),
            "tail": process.stdout.strip().splitlines()[-5:],
        }

    worker_count = min(8, len(ordered))
    completed: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [executor.submit(transfer_one, item) for item in ordered]
        for future in concurrent.futures.as_completed(futures):
            completed.append(future.result())
    completed.sort(key=lambda item: str(item["path"]))
    return {
        "existing_bytes_before": existing_bytes,
        "parallel_workers": worker_count,
        "files_transferred": len(completed),
        "bytes_declared": sum(int(item["bytes"]) for item in completed),
        "rsync_tail": [
            {"path": item["path"], "tail": item["tail"]}
            for item in completed[-8:]
        ],
    }


def parse_sums(path: pathlib.Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            digest, relative = raw.split("  ", 1)
        except ValueError as exc:
            raise DirectSyncError(f"invalid SHA256SUMS line {number}") from exc
        pure = pathlib.PurePosixPath(relative)
        if not SHA_RE.fullmatch(digest) or relative.startswith("/") or ".." in pure.parts:
            raise DirectSyncError(f"unsafe SHA256SUMS line {number}")
        if relative in values:
            raise DirectSyncError(f"duplicate SHA256SUMS path: {relative}")
        if not (
            relative.startswith("checkpoints/")
            or relative == "recovery/export_manifest.json"
            or relative == "runtime/agillm41.py"
            or relative.startswith("state/")
        ):
            raise DirectSyncError(f"unapproved export path: {relative}")
        values[relative] = digest
    return values


def verify_local_export(root: pathlib.Path, expected: dict[str, Any]) -> dict[str, Any]:
    if root.is_symlink() or root.parent.resolve() != STAGING_ROOT.resolve():
        raise DirectSyncError("unsafe local export root")
    links = [str(path.relative_to(root)) for path in root.rglob("*") if path.is_symlink()]
    if links:
        raise DirectSyncError(f"local export contains symlinks: {links[:5]}")
    sums_path = root / "SHA256SUMS"
    sums_sha = sha256_file(sums_path)
    if sums_sha != expected["sha256sums_sha256"]:
        raise DirectSyncError("source and destination SHA256SUMS identities differ")
    sums = parse_sums(sums_path)
    if len(sums) != int(expected.get("sum_entries", -1)):
        raise DirectSyncError(f"unexpected SHA256SUMS entry count: {len(sums)}")
    total = sums_path.stat().st_size
    for relative, digest in sorted(sums.items()):
        path = root / pathlib.PurePosixPath(relative)
        actual = sha256_file(path)
        if actual != digest:
            raise DirectSyncError(f"destination SHA256 mismatch: {relative}")
        total += path.stat().st_size
    expected_files = set(sums) | {"SHA256SUMS"}
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    extras = sorted(actual_files - expected_files)
    missing = sorted(expected_files - actual_files)
    if extras or missing:
        raise DirectSyncError(f"destination file-set mismatch extras={extras[:5]} missing={missing[:5]}")
    manifest = read_json(root / "recovery/export_manifest.json")
    checks = {
        "status": manifest.get("status") == "validated_complete",
        "step": int(manifest.get("step", -1)) == int(expected["step"]),
        "checkpoint": manifest.get("checkpoint") == expected["checkpoint"],
        "checkpoint_sha": manifest.get("checkpoint_sha256") == expected["checkpoint_sha256"],
        "runtime_sha": manifest.get("runtime_sha256") == TRUSTED_RUNTIME_SHA,
        "bytes": total == int(expected["bytes"]),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise DirectSyncError("destination export manifest failed: " + ",".join(failed))
    return {
        "files_verified": len(actual_files),
        "bytes_verified": total,
        "sha256sums_sha256": sums_sha,
        "manifest": manifest,
    }


def metadata_path(stage: pathlib.Path) -> pathlib.Path:
    return stage / ".vast_metadata"


def build_metadata(
    stage: pathlib.Path,
    route: dict[str, Any],
    latest: dict[str, Any],
    summary: dict[str, Any],
    local_verification: dict[str, Any],
) -> pathlib.Path:
    destination = metadata_path(stage)
    if destination.exists():
        if destination.is_symlink() or destination.parent.resolve() != stage.resolve():
            raise DirectSyncError("unsafe existing direct metadata directory")
        shutil.rmtree(destination)
    destination.mkdir(mode=0o700)
    members = (
        "SHA256SUMS",
        "recovery/export_manifest.json",
        "runtime/agillm41.py",
        "state/latest.json",
        "state/run_state.json",
    )
    copied: dict[str, dict[str, Any]] = {}
    for relative in members:
        source = stage / pathlib.PurePosixPath(relative)
        if source.is_symlink() or not source.is_file():
            raise DirectSyncError(f"required direct metadata member is missing: {relative}")
        target = destination / pathlib.PurePosixPath(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied[relative] = {"bytes": target.stat().st_size, "sha256": sha256_file(target)}
    if copied["runtime/agillm41.py"]["sha256"] != TRUSTED_RUNTIME_SHA:
        raise DirectSyncError("direct metadata runtime is not the trusted inference runtime")

    verified_at = now_iso()
    content_identity = {
        "schema": "agillm43.vast-direct-content.v1",
        "step": int(summary["step"]),
        "checkpoint": summary["checkpoint"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "runtime_sha256": summary["runtime_sha256"],
        "sha256sums_sha256": summary["sha256sums_sha256"],
        "files": int(summary["files"]),
        "bytes": int(summary["bytes"]),
        "builder_sha256": TRUSTED_BUILDER_SHA,
    }
    content_commit = sha256_bytes(canonical_bytes(content_identity))
    receipt_seed = {
        "content_commit": content_commit,
        "instance_id": route["instance_id"],
        "route_observed_at": route["observed_at"],
        "source_checkpoint": latest["path"],
        "verified_at": verified_at,
    }
    receipt_commit = sha256_bytes(canonical_bytes(receipt_seed))
    prefix = f"vast-direct/instance{route['instance_id']}/step{summary['step']}/"
    repo = f"vast-direct/{EXPECTED_LABEL}/instance-{route['instance_id']}"
    receipt = {
        "schema": "agillm43.vast-direct-verification.v1",
        "ok": True,
        "source": "vast-direct",
        "repo": repo,
        "prefix": prefix,
        "step": int(summary["step"]),
        "checkpoint": summary["checkpoint"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "effective_seen_tok": int(summary.get("effective_seen_tok", -1)),
        "runtime_sha256": summary["runtime_sha256"],
        "files_verified": int(local_verification["files_verified"]),
        "verified_bytes": int(local_verification["bytes_verified"]),
        "sha256sums_sha256": summary["sha256sums_sha256"],
        "builder_sha256": TRUSTED_BUILDER_SHA,
        "content_commit": content_commit,
        "receipt_commit": receipt_commit,
        "verified_at": verified_at,
        "source_sha256_verified": True,
        "destination_sha256_verified": True,
        "route_label": route["label"],
        "route_instance_id": route["instance_id"],
        "route_observed_at": route["observed_at"],
        "source_host": route["host"],
        "source_port": route["port"],
        "source_checkpoint": latest["path"],
        "source_checkpoint_mtime": latest["mtime"],
        "source_export": summary["export_path"],
        "transport": "rsync-over-ssh",
    }
    metadata = {
        "schema": "agillm43.vast-serving-metadata.v1",
        "source": "vast-direct",
        "repo": repo,
        "prefix": prefix,
        "step": int(summary["step"]),
        "checkpoint": summary["checkpoint"],
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "runtime_sha256": summary["runtime_sha256"],
        "content_commit": content_commit,
        "receipt_commit": receipt_commit,
        "verified_at": verified_at,
        "files": copied,
    }
    atomic_json(destination / "VAST_VERIFIED.json", receipt)
    atomic_json(destination / "metadata.json", metadata)
    for path in sorted(destination.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for directory in sorted((path for path in destination.rglob("*") if path.is_dir()), reverse=True):
        fsync_directory(directory)
    fsync_directory(destination)
    return destination


def final_package(step: int, checkpoint_sha: str) -> pathlib.Path:
    return SERVE_ROOT / f"step{step}-{checkpoint_sha[:16]}"


def recoverable_final(path: pathlib.Path, step: int, checkpoint_sha: str) -> bool:
    try:
        if path.is_symlink() or path.parent.resolve() != SERVE_ROOT.resolve() or not path.is_dir():
            return False
        marker = read_json(path / "SERVING_PACKAGE.json")
        return (
            int(marker.get("step", -1)) == step
            and marker.get("checkpoint_sha256") == checkpoint_sha
            and (path / "metadata/VAST_VERIFIED.json").is_file()
            and (path / "metadata/metadata.json").is_file()
        )
    except Exception:
        return False


def promote(source: pathlib.Path, metadata: pathlib.Path, *, move_source: bool) -> dict[str, Any]:
    command = [sys.executable, str(PROMOTER), "--source", str(source), "--metadata", str(metadata)]
    if move_source:
        command.append("--move-source")
    process = subprocess.run(command, text=True, capture_output=True, timeout=2400, check=False)
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip()[-6000:]
        raise DirectSyncError(f"transactional promotion failed rc={process.returncode}: {detail}")
    try:
        value = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise DirectSyncError("transactional promoter returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise DirectSyncError("transactional promoter returned a non-object")
    if value.get("serving") is not True or value.get("source") != "vast-direct":
        raise DirectSyncError("transactional promoter did not confirm Vast-direct serving")
    return value


def attach_supervision(status: dict[str, Any]) -> dict[str, Any]:
    payload = {"promotion": status, "source": "vast-direct-autodeploy"}
    try:
        process = subprocess.run(
            [str(SUPERVISOR)], input=json.dumps(payload), text=True,
            capture_output=True, timeout=480, check=False,
        )
        lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
        if not lines:
            raise DirectSyncError(f"supervisor returned no JSON; rc={process.returncode}")
        supervision = json.loads(lines[-1])
        if not isinstance(supervision, dict):
            raise DirectSyncError("supervisor returned a non-object")
        supervision = dict(supervision)
        supervision["process_rc"] = process.returncode
        if process.returncode != 0:
            supervision["warning"] = "advisory model supervision unavailable; deterministic gates passed"
    except Exception as exc:
        supervision = {
            "ok": False,
            "advisory_only": True,
            "error": type(exc).__name__,
            "message": str(exc)[:800],
            "warning": "advisory model supervision unavailable; deterministic gates passed",
        }
    supervision["step"] = int(status["step"])
    supervision["checkpoint_sha256"] = status["checkpoint_sha256"]
    latest = current_status()
    if (
        int(latest.get("step", -1)) == int(status["step"])
        and latest.get("checkpoint_sha256") == status["checkpoint_sha256"]
    ):
        latest["supervision"] = supervision
        atomic_json(STATUS, latest)
    return supervision


def cleanup_remote_export(route: dict[str, Any], step: int) -> None:
    output = remote_export_path(step)
    command = (
        "set -Eeuo pipefail; OUT=" + shlex.quote(str(output)) + "; "
        "test \"$OUT\" = " + shlex.quote(str(EXPORT_ROOT / f"step{step}")) + "; "
        "if test -e \"$OUT\"; then test -d \"$OUT\" -a ! -L \"$OUT\"; rm -rf -- \"$OUT\"; fi"
    )
    ssh_run(route, command, timeout=120)


def sync(*, check_only: bool, expected_step: int | None, keep_export: bool) -> dict[str, Any]:
    route = load_route()
    latest = remote_latest(route)
    step = int(latest["step"])
    if expected_step is not None and step != expected_step:
        raise DirectSyncError(f"expected Vast step {expected_step}, found {step}")
    status = current_status()
    current_step = int(status.get("step", 0) or 0)
    result: dict[str, Any] = {
        "ok": True,
        "checked_at": now_iso(),
        "action": "check" if check_only else "sync",
        "vast_instance_id": route["instance_id"],
        "vast_step": step,
        "vast_file": latest["file"],
        "vast_age_s": round(float(latest["age_s"]), 3),
        "current_step": current_step,
        "current_file": status.get("file"),
    }
    if step <= current_step:
        if not validate_live(status):
            raise DirectSyncError("current serving package failed validation while Vast is not newer")
        result.update(
            action="noop",
            reason="already_serving_latest_vast_checkpoint" if step == current_step else "live_step_ahead_of_vast",
            current_source=status.get("source") or status.get("authoritative_source"),
        )
        return result
    if check_only:
        if float(latest["age_s"]) < SETTLE_SECONDS:
            result.update(action="settling", settle_remaining_s=round(SETTLE_SECONDS - float(latest["age_s"]), 3))
        else:
            result.update(action="update_available")
        return result
    if float(latest["age_s"]) < SETTLE_SECONDS:
        result.update(action="settling", settle_remaining_s=round(SETTLE_SECONDS - float(latest["age_s"]), 3))
        return result
    if not PROMOTER.is_file() or PROMOTER.is_symlink():
        raise DirectSyncError("transactional promoter is missing or unsafe")

    source_summary = build_and_verify_remote_export(route, latest)
    checkpoint_sha = str(source_summary["checkpoint_sha256"])
    stage = staging_path(step, checkpoint_sha)
    final = final_package(step, checkpoint_sha)

    if recoverable_final(final, step, checkpoint_sha):
        promotion = promote(final, final / "metadata", move_source=False)
        transfer = {"recovered_existing_final": True}
    else:
        safe_stage(stage)
        free = shutil.disk_usage(STAGING_ROOT).free
        existing = sum(
            path.stat().st_size
            for path in stage.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
        missing = max(0, int(source_summary["bytes"]) - existing)
        required = missing + MIN_RESERVE_BYTES
        if free < required:
            raise DirectSyncError(f"insufficient GETH space: free={free} required={required}")
        transfer = rsync_export(
            route, str(source_summary["export_path"]), stage, source_summary["members"]
        )
        post_transfer_summary = build_and_verify_remote_export(route, latest)
        identity_keys = (
            "step", "checkpoint", "checkpoint_sha256", "runtime_sha256",
            "files", "bytes", "sum_entries", "sha256sums_sha256", "export_path", "members",
        )
        if any(post_transfer_summary.get(key) != source_summary.get(key) for key in identity_keys):
            raise DirectSyncError("Vast export identity changed during transfer")
        local_verification = verify_local_export(stage, source_summary)
        metadata = build_metadata(stage, route, latest, source_summary, local_verification)
        promotion = promote(stage / "checkpoints", metadata, move_source=True)
        transfer.update({
            "destination_files_verified": local_verification["files_verified"],
            "destination_bytes_verified": local_verification["bytes_verified"],
        })

    if int(promotion.get("step", -1)) != step or promotion.get("checkpoint_sha256") != checkpoint_sha:
        raise DirectSyncError("promoter served an unexpected checkpoint identity")
    supervision = attach_supervision(promotion)
    if stage.exists():
        if stage.is_symlink() or stage.parent.resolve() != STAGING_ROOT.resolve():
            raise DirectSyncError("refusing unsafe staging cleanup")
        shutil.rmtree(stage)
    if not keep_export:
        cleanup_remote_export(route, step)
    result.update(
        action="promoted",
        deployed_at=promotion.get("deployed_at"),
        checkpoint_sha256=checkpoint_sha,
        source="vast-direct",
        integrity=promotion.get("integrity"),
        transfer=transfer,
        smoke=promotion.get("smoke"),
        supervision=supervision,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--keep-export", action="store_true")
    args = parser.parse_args()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"ok": True, "action": "noop", "reason": "another_checkpoint_sync_is_running"}, sort_keys=True))
            return 0
        result = sync(
            check_only=args.check_only,
            expected_step=args.expected_step,
            keep_export=args.keep_export,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({
            "ok": False,
            "error": type(exc).__name__,
            "message": str(exc),
            "failed_at": now_iso(),
        }, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    finally:
        os.close(descriptor)


if __name__ == "__main__":
    raise SystemExit(main())
