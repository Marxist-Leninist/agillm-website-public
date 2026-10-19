#!/usr/bin/env python3
"""Run the verified Vast deployer with a restricted Vast-to-GETH push transport.

The base deployer remains responsible for candidate discovery, source export,
source/destination SHA-256 verification, transactional promotion, rollback and
model-supervisor receipts. This wrapper replaces only the slow gateway pull:
Vast pushes into a write-only quarantine owned by a dedicated SSH identity,
then GETH atomically removes the completed tree from that identity's reach.
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
import os
import pathlib
import pwd
import re
import shlex
import shutil
import stat
import subprocess
import sys
from typing import Any

BASE = pathlib.Path("/opt/agillm43_netserve/agillm43_sync_latest_vast.py")
TRUSTED_BASE_SHA = "2eccb6787f05f72f3cc98a86a296cc53bbdb4cbc020c284755d50e4f70b9aeca"
PUSH_ROOT = pathlib.Path("/mnt/geth-vol1/agillm43/ckpts/artifacts/full/vast_push")
PUSH_USER = "agillm43push"
PUSH_AUTHORIZED_KEYS = pathlib.Path("/var/lib/agillm43-vast-push-user/.ssh/authorized_keys")
PUSH_FORCED_COMMAND = (
    'restrict,command="/usr/bin/rrsync -wo -no-del -no-lock -munge '
    '/mnt/geth-vol1/agillm43/ckpts/artifacts/full/vast_push" '
)
PUSH_KEY_FINGERPRINT = "SHA256:wkv3xQg8xa1PzsfvXoRVlSzIH0CgGhWHQh50Z9noB+w"
GETH_HOST = "x.x.x.x"
GETH_HOST_PUBLIC_KEY = pathlib.Path("/etc/ssh/ssh_host_ed25519_key.pub")
GETH_HOST_FINGERPRINT = "SHA256:hE2//6kXIPcirqLbro0cqumo3oTlSvG9+8Ah++AuqZc"
REMOTE_PUSH_KEY = pathlib.PurePosixPath("/workspace/.agillm43_geth_push_ed25519")
REMOTE_KNOWN_HOSTS = pathlib.PurePosixPath("/workspace/.agillm43_geth_known_hosts")
REMOTE_PUSH_WORKER = pathlib.PurePosixPath("/workspace/agillm43-push-export-v1.py")
TRUSTED_PUSH_WORKER_SHA = "1c92862656aac4f802ef4a632424406d1cbaf3db5c8a449e68d0058ba905bdbb"
DESTINATION_RE = re.compile(r"^[.]incoming-step[0-9]+-[0-9a-f]{16}$")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ):
        raise RuntimeError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def key_fingerprint(path: pathlib.Path) -> str:
    process = subprocess.run(
        ["ssh-keygen", "-lf", str(path), "-E", "sha256"],
        text=True, capture_output=True, timeout=10, check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"cannot fingerprint {path}")
    fields = process.stdout.split()
    if len(fields) < 2:
        raise RuntimeError(f"invalid fingerprint output for {path}")
    return fields[1]


def load_base() -> Any:
    if BASE.is_symlink() or not BASE.is_file() or sha256_file(BASE) != TRUSTED_BASE_SHA:
        raise RuntimeError("installed Vast deployment base failed its pinned SHA-256 check")
    spec = importlib.util.spec_from_file_location("agillm43_vast_sync_base", BASE)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load installed Vast deployment base")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_regular(path: pathlib.Path, owner: int, forbidden_mode: int = 0o077) -> os.stat_result:
    info = path.stat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode) or info.st_uid != owner:
        raise RuntimeError(f"unsafe regular file: {path}")
    if stat.S_IMODE(info.st_mode) & forbidden_mode:
        raise RuntimeError(f"unsafe permissions on {path}: {oct(stat.S_IMODE(info.st_mode))}")
    return info


def validate_directory(path: pathlib.Path, owner: int) -> os.stat_result:
    info = path.stat()
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode) or info.st_uid != owner:
        raise RuntimeError(f"unsafe directory: {path}")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise RuntimeError(f"unsafe permissions on {path}: {oct(stat.S_IMODE(info.st_mode))}")
    return info


def validate_receiver(base: Any) -> pwd.struct_passwd:
    account = pwd.getpwnam(PUSH_USER)
    validate_directory(PUSH_ROOT, account.pw_uid)
    validate_regular(PUSH_AUTHORIZED_KEYS, account.pw_uid)
    lines = [line.strip() for line in PUSH_AUTHORIZED_KEYS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1 or not lines[0].startswith(PUSH_FORCED_COMMAND):
        raise RuntimeError("restricted push authorized_keys command failed validation")
    if key_fingerprint(PUSH_AUTHORIZED_KEYS) != PUSH_KEY_FINGERPRINT:
        raise RuntimeError("restricted push public-key fingerprint mismatch")
    if PUSH_ROOT.stat().st_dev != base.STAGING_ROOT.stat().st_dev:
        raise RuntimeError("push quarantine and root staging are not on the same filesystem")
    return account


def host_key_entry() -> str:
    validate_regular(GETH_HOST_PUBLIC_KEY, 0, forbidden_mode=0o022)
    if key_fingerprint(GETH_HOST_PUBLIC_KEY) != GETH_HOST_FINGERPRINT:
        raise RuntimeError("GETH host-key fingerprint mismatch")
    parts = GETH_HOST_PUBLIC_KEY.read_text(encoding="utf-8").strip().split()
    if len(parts) < 2 or parts[0] != "ssh-ed25519" or not re.fullmatch(r"[A-Za-z0-9+/=]+", parts[1]):
        raise RuntimeError("GETH host public key is malformed")
    base64.b64decode(parts[1], validate=True)
    return f"{GETH_HOST} {parts[0]} {parts[1]}\n"


def ensure_remote_push_material(base: Any, route: dict[str, Any]) -> dict[str, Any]:
    encoded_host = base64.b64encode(host_key_entry().encode("ascii")).decode("ascii")
    q = shlex.quote
    command = f'''set -Eeuo pipefail
umask 077
KEY=REDACTED
KNOWN={q(str(REMOTE_KNOWN_HOSTS))}
WORKER={q(str(REMOTE_PUSH_WORKER))}
test -f "$KEY" -a ! -L "$KEY"
test "$(stat -c %u:%a "$KEY")" = "0:600"
test -f "$WORKER" -a ! -L "$WORKER"
test "$(stat -c %u "$WORKER")" = "0"
test "$(sha256sum "$WORKER" | awk '{{print $1}}')" = {q(TRUSTED_PUSH_WORKER_SHA)}
KEY_FP=$(ssh-keygen -y -f "$KEY" | ssh-keygen -lf /dev/stdin -E sha256 | awk '{{print $2}}')
test "$KEY_FP" = {q(PUSH_KEY_FINGERPRINT)}
TMP="${{KNOWN}}.tmp.$$"
trap 'rm -f "$TMP"' EXIT
printf %s {q(encoded_host)} | base64 -d > "$TMP"
chmod 600 "$TMP"
mv -f "$TMP" "$KNOWN"
trap - EXIT
test "$(stat -c %u:%a "$KNOWN")" = "0:600"
HOST_FP=$(ssh-keygen -lf "$KNOWN" -E sha256 | awk '{{print $2}}')
test "$HOST_FP" = {q(GETH_HOST_FINGERPRINT)}
printf '{{"ok":true,"key_fingerprint":"%s","host_fingerprint":"%s","worker_sha256":"%s"}}\n' "$KEY_FP" "$HOST_FP" {q(TRUSTED_PUSH_WORKER_SHA)}
'''
    process = base.ssh_run(route, command, timeout=60)
    value = base.parse_last_json(process.stdout, "Vast push-material validation")
    if (
        value.get("ok") is not True
        or value.get("key_fingerprint") != PUSH_KEY_FINGERPRINT
        or value.get("host_fingerprint") != GETH_HOST_FINGERPRINT
        or value.get("worker_sha256") != TRUSTED_PUSH_WORKER_SHA
    ):
        raise RuntimeError("Vast push material did not pass pinned validation")
    return value


def member_identity(base: Any, members: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, int]:
    clean: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in members:
        if not isinstance(item, dict):
            raise RuntimeError("push member is not an object")
        relative = str(item.get("path") or "")
        pure = pathlib.PurePosixPath(relative)
        try:
            size = int(item.get("bytes"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid push-member size: {relative}") from exc
        if (
            not relative or relative.startswith("/") or ".." in pure.parts
            or pure.as_posix() != relative or relative in seen or size < 0
        ):
            raise RuntimeError(f"unsafe or duplicate push member: {relative}")
        seen.add(relative)
        clean.append({"path": relative, "bytes": size})
    clean.sort(key=lambda value: str(value["path"]))
    if not (base.MIN_SHARD_ENTRIES + 8 <= len(clean) <= base.MAX_SHARD_ENTRIES + 12) or "SHA256SUMS" not in seen:
        raise RuntimeError("push member set is incomplete")
    identity = base.sha256_bytes(base.canonical_bytes(clean))
    return clean, identity, sum(int(value["bytes"]) for value in clean)


def tree_members(root: pathlib.Path) -> list[dict[str, Any]]:
    links = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink()]
    if links:
        raise RuntimeError(f"staging tree contains symlinks: {links[:5]}")
    values = [
        {"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size}
        for path in root.rglob("*") if path.is_file()
    ]
    values.sort(key=lambda item: str(item["path"]))
    return values


def chmod_chown_tree(root: pathlib.Path, uid: int, gid: int) -> None:
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_symlink():
            raise RuntimeError(f"refusing symlink in staging tree: {path}")
        os.chown(path, uid, gid)
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
    os.chown(root, uid, gid)
    os.chmod(root, 0o700)


def fsync_directory(path: pathlib.Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prepare_quarantine(
    base: Any,
    destination: pathlib.Path,
    expected: list[dict[str, Any]],
    account: pwd.struct_passwd,
) -> tuple[pathlib.Path, bool, int]:
    if destination.parent.resolve() != base.STAGING_ROOT.resolve() or not DESTINATION_RE.fullmatch(destination.name):
        raise RuntimeError(f"unsafe root staging destination: {destination}")
    quarantine = PUSH_ROOT / destination.name
    expected_paths = {str(item["path"]): int(item["bytes"]) for item in expected}

    if destination.exists():
        if destination.is_symlink() or not destination.is_dir() or destination.stat().st_uid != 0:
            raise RuntimeError("unsafe existing root staging destination")
        metadata = destination / ".vast_metadata"
        if metadata.exists():
            if metadata.is_symlink() or not metadata.is_dir():
                raise RuntimeError("unsafe stale metadata in root staging destination")
            shutil.rmtree(metadata)
        current = tree_members(destination)
        if current == expected:
            return quarantine, True, sum(int(item["bytes"]) for item in current)
        if quarantine.exists() and current:
            raise RuntimeError("both root staging and push quarantine contain partial candidates")
        for item in current:
            wanted = expected_paths.get(str(item["path"]))
            if wanted is None or int(item["bytes"]) > wanted:
                raise RuntimeError("root staging has an incompatible partial candidate")
        if current:
            os.rename(destination, quarantine)
            chmod_chown_tree(quarantine, account.pw_uid, account.pw_gid)
        else:
            destination.rmdir()

    if quarantine.exists():
        validate_directory(quarantine, account.pw_uid)
        current = tree_members(quarantine)
        for item in current:
            wanted = expected_paths.get(str(item["path"]))
            if wanted is None or int(item["bytes"]) > wanted:
                raise RuntimeError("push quarantine has an incompatible partial candidate")
        existing = sum(int(item["bytes"]) for item in current)
    else:
        quarantine.mkdir(mode=0o700)
        os.chown(quarantine, account.pw_uid, account.pw_gid)
        existing = 0
    return quarantine, False, existing


def transfer_via_push(
    route: dict[str, Any],
    remote_path: str,
    destination: pathlib.Path,
    members: list[dict[str, Any]],
) -> dict[str, Any]:
    base = BASE_MODULE
    account = validate_receiver(base)
    clean, identity, total = member_identity(base, members)
    remote_root = pathlib.PurePosixPath(remote_path)
    if remote_root.parent != base.EXPORT_ROOT or not re.fullmatch(r"step[0-9]+", remote_root.name):
        raise RuntimeError(f"unsafe remote export path: {remote_path}")
    quarantine, recovered, existing = prepare_quarantine(base, destination, clean, account)
    if recovered:
        return {
            "transport": "restricted-parallel-rsync-push",
            "recovered_existing_destination": True,
            "files_transferred": len(clean),
            "bytes_declared": total,
            "members_sha256": identity,
        }

    free = shutil.disk_usage(PUSH_ROOT).free
    missing = max(0, total - existing)
    required = missing + base.MIN_RESERVE_BYTES
    if free < required:
        raise RuntimeError(f"insufficient GETH space for push: free={free} required={required}")
    material = ensure_remote_push_material(base, route)
    q = shlex.quote
    command = (
        f"/opt/conda/bin/python {q(str(REMOTE_PUSH_WORKER))} "
        f"--export-root {q(str(remote_root))} "
        f"--destination-name {q(destination.name)} "
        f"--expected-members-sha256 {q(identity)} --workers 8"
    )
    process = base.ssh_run(route, command, timeout=3600)
    value = base.parse_last_json(process.stdout, "Vast restricted push worker")
    checks = {
        "ok": value.get("ok") is True,
        "schema": value.get("schema") == "agillm43.vast-push.v1",
        "export": pathlib.PurePosixPath(str(value.get("export_root"))) == remote_root,
        "destination": value.get("destination_name") == destination.name,
        "identity": value.get("members_sha256") == identity,
        "files": int(value.get("files", -1)) == len(clean),
        "bytes": int(value.get("bytes", -1)) == total,
        "workers": int(value.get("workers", 0)) == 8,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError("Vast restricted push worker failed validation: " + ",".join(failed))
    if not quarantine.exists() or quarantine.is_symlink() or tree_members(quarantine) != clean:
        raise RuntimeError("completed push quarantine does not match the declared member set")

    subprocess.run(["sync", "-f", str(quarantine)], check=True, timeout=300)
    if destination.exists():
        raise RuntimeError("root staging destination appeared during push")
    os.rename(quarantine, destination)
    chmod_chown_tree(destination, 0, 0)
    fsync_directory(base.STAGING_ROOT)
    fsync_directory(PUSH_ROOT)
    return {
        "transport": "restricted-parallel-rsync-push",
        "existing_bytes_before": existing,
        "parallel_workers": int(value["workers"]),
        "files_transferred": int(value["files"]),
        "bytes_declared": int(value["bytes"]),
        "elapsed_s": value.get("elapsed_s"),
        "aggregate_mib_s": value.get("aggregate_mib_s"),
        "members_sha256": identity,
        "key_fingerprint": material["key_fingerprint"],
        "host_fingerprint": material["host_fingerprint"],
    }


BASE_MODULE = load_base()
BASE_MODULE.rsync_export = transfer_via_push


if __name__ == "__main__":
    raise SystemExit(BASE_MODULE.main())
