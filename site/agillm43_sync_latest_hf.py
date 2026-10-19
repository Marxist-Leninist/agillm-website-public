#!/usr/bin/env python3
"""Deploy the newest remotely verified AGILLM4.3 rolling checkpoint.

The mutable Hub branch is used only to discover receipt paths. Candidate bytes
are downloaded from the immutable receipt commit, verified again by the
transactional promotion gate, and only then allowed to replace the live bridge.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, TypeVar

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_download, hf_hub_url, snapshot_download

REPO = "MarxistLeninist/AGILLM-4.3-repretrain-single-file-20260716T091441Z"
REVISION = "main"
NETSERVE = pathlib.Path("/opt/agillm43_netserve")
STATUS = NETSERVE / "agillm_hf_status.json"
PROMOTER = NETSERVE / "agillm43_promote_hf_verified.py"
SUPERVISOR = NETSERVE / "agillm43_deploy_supervisor.py"
METADATA_ROOT = NETSERVE / "hf_verified"
DOWNLOAD_ROOT = pathlib.Path("/mnt/geth-vol1/agillm43/ckpts/artifacts/full/hf_download")
SERVE_ROOT = pathlib.Path("/mnt/geth-vol1/agillm43/ckpts/artifacts/full/hf_verified")
LIVE_VAST_ROOT = pathlib.Path("/mnt/geth-vol1/agillm43/ckpts/artifacts/full/live_vast")
LOCK = pathlib.Path("/run/lock/agillm43-hf-autodeploy.lock")
TOKEN_PATHS = (
    pathlib.Path("/root/.cache/huggingface/token"),
    pathlib.Path("/root/.huggingface/token"),
    pathlib.Path("/root/.hf_token_agillm43"),
)
RECEIPT_RE = re.compile(r"^recovery/rolling/step([0-9]+)/HF_VERIFIED\.json$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40,64}$")
CHECKPOINT_RE = re.compile(r"^pretrain_step0*([0-9]+)_.*\.pt$")
GIB = 1024**3
T = TypeVar("T")


class SyncError(RuntimeError):
    pass


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            h.update(block)
        after = os.fstat(handle.fileno())
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise SyncError(f"file changed while hashing: {path}")
    return h.hexdigest()


def atomic_json(path: pathlib.Path, value: object, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp, mode)
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def retry(label: str, fn: Callable[[], T], attempts: int = 5) -> T:
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except BaseException as exc:
            last = exc
            if attempt == attempts:
                break
            delay = min(30, 2 ** attempt)
            print(json.dumps({"event": "retry", "operation": label, "attempt": attempt, "delay_s": delay}), flush=True)
            time.sleep(delay)
    assert last is not None
    raise SyncError(f"{label} failed after {attempts} attempts: {type(last).__name__}: {last}") from last


def load_token() -> str | None:
    for path in TOKEN_PATHS:
        try:
            value = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            continue
        if value:
            return value
    return None


def read_current() -> dict[str, Any]:
    try:
        value = json.loads(STATUS.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def validate_receipt(value: object, expected_step: int, expected_path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SyncError("HF receipt is not a JSON object")
    receipt: dict[str, Any] = value
    step = int(receipt.get("step", -1))
    expected_prefix = f"recovery/rolling/step{expected_step}/"
    checkpoint = str(receipt.get("checkpoint") or "")
    match = CHECKPOINT_RE.fullmatch(checkpoint)
    checks = {
        "ok": receipt.get("ok") is True,
        "schema": receipt.get("schema") == "agillm43.hf-verification.v2",
        "repo": receipt.get("repo") == REPO,
        "step": step == expected_step,
        "prefix": receipt.get("prefix") == expected_prefix,
        "checkpoint_step": bool(match and int(match.group(1)) == expected_step),
        "checkpoint_sha": bool(SHA_RE.fullmatch(str(receipt.get("checkpoint_sha256") or ""))),
        "content_commit": bool(COMMIT_RE.fullmatch(str(receipt.get("content_commit") or ""))),
        "receipt_commit": receipt.get("receipt_commit") in (None, "") or bool(COMMIT_RE.fullmatch(str(receipt.get("receipt_commit")))),
        "files": 44 <= int(receipt.get("files_verified", 0)) <= 45,
        "bytes": 5 * GIB <= int(receipt.get("verified_bytes", 0)) <= 20 * GIB,
        "path": expected_path == f"recovery/rolling/step{expected_step}/HF_VERIFIED.json",
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SyncError(f"HF receipt failed validation fields: {','.join(failed)}")
    return receipt


def fetch_json_at(repo_path: str, revision: str, token: str | None) -> tuple[dict[str, Any], bytes]:
    with tempfile.TemporaryDirectory(prefix=".hf-receipt-", dir=NETSERVE) as temp:
        def download() -> str:
            return hf_hub_download(
                repo_id=REPO,
                repo_type="model",
                filename=repo_path,
                revision=revision,
                token=token,
                local_dir=temp,
                force_download=True,
                etag_timeout=30,
            )
        path = pathlib.Path(retry(f"download receipt at {revision[:12]}", download))
        raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SyncError("downloaded HF receipt is invalid JSON") from exc
    if not isinstance(value, dict):
        raise SyncError("downloaded HF receipt is not an object")
    return value, raw


def discover_latest(token: str | None, expected_step: int | None) -> tuple[int, str, dict[str, Any], dict[str, Any]]:
    api = HfApi(token=token)
    files = retry(
        "list verified checkpoint receipts",
        lambda: api.list_repo_files(repo_id=REPO, repo_type="model", revision=REVISION),
    )
    candidates: list[tuple[int, str]] = []
    for repo_path in files:
        match = RECEIPT_RE.fullmatch(repo_path)
        if match:
            candidates.append((int(match.group(1)), repo_path))
    if expected_step is not None:
        candidates = [item for item in candidates if item[0] == expected_step]
        if not candidates:
            raise SyncError(f"step {expected_step} has no remotely verified receipt yet")
    if not candidates:
        raise SyncError("no remotely verified rolling checkpoint receipts found")
    step, repo_path = max(candidates)
    url = hf_hub_url(repo_id=REPO, repo_type="model", filename=repo_path, revision=REVISION)
    file_metadata = retry(
        "resolve immutable receipt commit",
        lambda: get_hf_file_metadata(url, token=token),
    )
    immutable_revision = str(file_metadata.commit_hash or "")
    if not COMMIT_RE.fullmatch(immutable_revision):
        raise SyncError("Hub did not return an immutable commit for the receipt")
    exact_value, _ = fetch_json_at(repo_path, immutable_revision, token)
    remote_receipt = validate_receipt(exact_value, step, repo_path)
    claimed_commit = remote_receipt.get("receipt_commit")
    if claimed_commit not in (None, "", immutable_revision):
        raise SyncError("receipt claims a different immutable commit")
    receipt = dict(remote_receipt)
    receipt["receipt_commit"] = immutable_revision
    return step, repo_path, receipt, remote_receipt


def install_metadata(package: pathlib.Path, receipt: dict[str, Any]) -> pathlib.Path:
    step = int(receipt["step"])
    final = METADATA_ROOT / f"step{step}"
    if final.exists():
        try:
            old = json.loads((final / "HF_VERIFIED.json").read_text(encoding="utf-8"))
        except Exception as exc:
            raise SyncError(f"existing metadata directory is invalid: {final}") from exc
        if old != receipt:
            raise SyncError(f"existing metadata receipt conflicts with candidate: {final}")
        return final

    staging = METADATA_ROOT / f".incoming-step{step}.{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, mode=0o755)
    required = (
        "HF_VERIFIED.json",
        "SHA256SUMS",
        "recovery/export_manifest.json",
        "runtime/agillm41.py",
        "state/latest.json",
        "state/run_state.json",
        "status.json",
    )
    files_meta: dict[str, dict[str, object]] = {}
    for rel in required:
        source = package / rel
        if source.is_symlink() or not source.is_file():
            raise SyncError(f"verified package metadata member missing or unsafe: {rel}")
        destination = staging / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if rel == "HF_VERIFIED.json":
            atomic_json(destination, receipt)
        else:
            shutil.copy2(source, destination)
        files_meta[rel] = {"bytes": destination.stat().st_size, "sha256": sha256_file(destination)}

    metadata = {
        "schema": "agillm43.hf-serving-metadata.v1",
        "repo": REPO,
        "prefix": receipt["prefix"],
        "step": step,
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "content_commit": receipt["content_commit"],
        "receipt_commit": receipt["receipt_commit"],
        "verified_at": receipt["verified_at"],
        "files": files_meta,
    }
    atomic_json(staging / "metadata.json", metadata)
    fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    METADATA_ROOT.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)
    fd = os.open(METADATA_ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return final


def supervise_deployment(payload: dict[str, object]) -> dict[str, object]:
    """Run advisory Token Plan reviewers; deterministic gates remain authoritative."""
    try:
        process = subprocess.run(
            [str(SUPERVISOR)], input=json.dumps(payload), text=True,
            capture_output=True, timeout=480,
        )
        lines = [line.strip() for line in process.stdout.splitlines() if line.strip()]
        if not lines:
            raise SyncError(f"supervisor returned no JSON; rc={process.returncode}")
        value = json.loads(lines[-1])
        if not isinstance(value, dict):
            raise SyncError("supervisor result is not an object")
        value["process_rc"] = process.returncode
        if process.returncode != 0:
            value["warning"] = "model supervision unavailable; deterministic deployment remains authoritative"
        return value
    except Exception as exc:
        return {
            "ok": False,
            "advisory_only": True,
            "error": type(exc).__name__,
            "message": str(exc)[:800],
            "warning": "model supervision unavailable; deterministic deployment remains authoritative",
        }


def attach_supervision(status: dict[str, Any], payload: dict[str, object]) -> dict[str, object]:
    supervision = supervise_deployment(payload)
    supervision = dict(supervision)
    supervision["step"] = int(status.get("step") or 0)
    checkpoint_sha = str(status.get("checkpoint_sha256") or "")
    if checkpoint_sha:
        supervision["checkpoint_sha256"] = checkpoint_sha
    updated = dict(status)
    updated["supervision"] = supervision
    atomic_json(STATUS, updated)
    return supervision


def live_valid_hf(current: dict[str, Any], current_step: int, current_sha: str) -> bool:
    """HF-promoted package layout: hf_verified/stepX-sha/{SERVING_PACKAGE.json,checkpoints/<file>}."""
    package = pathlib.Path(str(current.get("package_path") or ""))
    return (
        current.get("status") == "serving_verified"
        and current.get("serving") is True
        and SHA_RE.fullmatch(current_sha) is not None
        and package.is_dir()
        and not package.is_symlink()
        and package.parent.resolve() == SERVE_ROOT.resolve()
        and (package / "SERVING_PACKAGE.json").is_file()
        and (package / "checkpoints" / str(current.get("file") or "")).is_file()
    )


def live_valid_vast_direct(current: dict[str, Any], current_step: int, current_sha: str) -> bool:
    """Vast-direct hold-release layout: live_vast/stepX-sha/{DEPLOY_RECEIPT.json,<ckpt>,SOURCE_SHA256SUMS}."""
    filename = str(current.get("file") or "")
    if (
        current.get("status") != "serving_verified"
        or current.get("serving") is not True
        or not SHA_RE.fullmatch(current_sha)
        or pathlib.PurePath(filename).name != filename
        or not filename.endswith(".pt")
    ):
        return False
    package = pathlib.Path(str(current.get("package_path") or ""))
    if (
        not package.is_dir()
        or package.is_symlink()
        or package.parent.resolve() != LIVE_VAST_ROOT.resolve()
        or package.name != f"step{current_step}-{current_sha[:16]}"
    ):
        return False
    try:
        receipt = json.loads((package / "DEPLOY_RECEIPT.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(receipt, dict):
        return False
    if (
        receipt.get("status") != "complete_verified"
        or int(receipt.get("step", -1)) != current_step
        or receipt.get("checkpoint") != filename
        or receipt.get("checkpoint_sha256") != current_sha
        or str(receipt.get("destination") or "") != str(package)
    ):
        return False
    checkpoint = package / filename
    if not checkpoint.is_file() or checkpoint.is_symlink():
        return False
    try:
        sums_text = (package / "SOURCE_SHA256SUMS").read_text(encoding="utf-8")
    except OSError:
        return False
    sums = {}
    for line in sums_text.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and SHA_RE.fullmatch(parts[0]):
            sums[parts[1].lstrip("*").lstrip("./")] = parts[0]
    return sums.get(filename) == current_sha


def sync(expected_step: int | None, check_only: bool, keep_failed: bool) -> dict[str, Any]:
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    token = load_token()
    step, repo_path, receipt, remote_receipt = discover_latest(token, expected_step)
    current = read_current()
    current_step = int(current.get("step", 0) or 0)
    current_sha = str(current.get("checkpoint_sha256") or "")
    candidate_sha = str(receipt["checkpoint_sha256"])
    result: dict[str, Any] = {
        "ok": True,
        "checked_at": now_iso(),
        "repo": REPO,
        "latest_verified_step": step,
        "latest_verified_file": receipt["checkpoint"],
        "current_step": current_step,
        "action": "check" if check_only else "sync",
    }
    if step < current_step:
        live_valid = live_valid_hf(current, current_step, current_sha) or live_valid_vast_direct(
            current, current_step, current_sha
        )
        if not live_valid:
            raise SyncError(
                f"latest remote verified step {step} is behind an unvalidated live step {current_step}"
            )
        result.update(
            action="noop",
            reason="remote_archive_behind_verified_live",
            live_source=current.get("source") or current.get("authoritative_source"),
            live_file=current.get("file"),
        )
        return result
    if step == current_step:
        if current_sha and current_sha != candidate_sha:
            raise SyncError("live and remote checkpoint hashes disagree at the same step")
        supervision = current.get("supervision") if isinstance(current.get("supervision"), dict) else {}
        if (
            not check_only
            and (supervision.get("ok") is not True or int(supervision.get("step") or 0) != step)
        ):
            supervision = attach_supervision(current, {"promotion": current, "source": "geth-local-autodeploy"})
        result.update(
            action="noop", reason="already_serving_latest_verified",
            supervision=supervision or None,
        )
        return result
    if check_only:
        result.update(action="update_available")
        return result

    if not PROMOTER.is_file():
        raise SyncError(f"promotion gate missing: {PROMOTER}")
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(DOWNLOAD_ROOT).free
    verified_bytes = int(receipt["verified_bytes"])
    required_free = verified_bytes * 2 + 4 * GIB
    if free < required_free:
        raise SyncError(f"insufficient deploy disk: free={free} required={required_free}")

    work = DOWNLOAD_ROOT / f".incoming-step{step}-{candidate_sha[:16]}.{os.getpid()}"
    if work.exists():
        raise SyncError(f"refusing pre-existing work directory: {work}")
    work.mkdir(mode=0o700)
    prefix = str(receipt["prefix"]).rstrip("/")
    immutable_revision = str(receipt["receipt_commit"])
    promoted = False
    try:
        def download_snapshot() -> str:
            return snapshot_download(
                repo_id=REPO,
                repo_type="model",
                revision=immutable_revision,
                allow_patterns=[prefix + "/**"],
                local_dir=str(work),
                token=token,
                max_workers=8,
            )
        retry(f"download immutable checkpoint step {step}", download_snapshot, attempts=4)
        package = work / prefix
        if package.is_symlink() or not package.is_dir():
            raise SyncError(f"downloaded package root missing or unsafe: {package}")
        downloaded_receipt = json.loads((package / "HF_VERIFIED.json").read_text(encoding="utf-8"))
        if downloaded_receipt != remote_receipt:
            raise SyncError("snapshot receipt differs from preflight immutable receipt")
        source = package / "checkpoints"
        if source.is_symlink() or not source.is_dir():
            raise SyncError("downloaded checkpoint payload directory is missing or unsafe")
        checkpoint = source / str(receipt["checkpoint"])
        if checkpoint.is_symlink() or not checkpoint.is_file():
            raise SyncError("receipt checkpoint is absent from downloaded payload")
        metadata = install_metadata(package, receipt)
        command = [
            sys.executable,
            str(PROMOTER),
            "--source", str(source),
            "--metadata", str(metadata),
            "--move-source",
        ]
        promotion = subprocess.run(command, text=True, capture_output=True)
        if promotion.returncode != 0:
            detail = (promotion.stderr or promotion.stdout).strip()[-4000:]
            raise SyncError(f"transactional promotion failed: {detail}")
        promoted = True
        live = read_current()
        if int(live.get("step", -1)) != step or live.get("checkpoint_sha256") != candidate_sha or live.get("serving") is not True:
            raise SyncError("promotion returned success but live status does not match candidate")
        supervision = attach_supervision(
            live,
            {"promotion": live, "source": "geth-local-autodeploy"},
        )
        result.update(
            action="promoted",
            deployed_at=live.get("deployed_at"),
            checkpoint_sha256=candidate_sha,
            serving=live.get("serving"),
            smoke=live.get("smoke"),
            supervision=supervision,
        )
        return result
    finally:
        if promoted or not keep_failed:
            with contextlib.suppress(Exception):
                shutil.rmtree(work)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-step", type=int)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--keep-failed", action="store_true")
    args = parser.parse_args()
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(LOCK, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"ok": True, "action": "noop", "reason": "sync_already_running"}, sort_keys=True))
            return 0
        value = sync(args.expected_step, args.check_only, args.keep_failed)
        print(json.dumps(value, indent=2, sort_keys=True))
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
        os.close(lock_fd)


if __name__ == "__main__":
    raise SystemExit(main())
