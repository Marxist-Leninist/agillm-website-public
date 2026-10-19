#!/usr/bin/env python3
"""Hash-guarded, inference-only fallback rollout and rollback. NOT executed here.

Run on the inference host, not on a trainer shell selected by guesswork. Default
is read-only preflight; writes/signaling require --execute --admission-paused.
The latter asserts the caller has already paused upstream admission and drained
requests. A /health check alone cannot close the new-request/SIGTERM race.

Required path and /proc starttime arguments deliberately have no guessed values.
Example shape (replace every placeholder with freshly inspected exact values):
  python3 guarded_inference_rollout.py rollout --bridge-file /ABS/bridge.py \
    --env-file /ABS/bridge.env --bridge-starttime TICKS \
    --supervisor-starttime TICKS --trainer-starttime TICKS
Add --execute --admission-paused only after successful CPU canary and admission
pause. For rollback use the same paths/state-dir plus the NEW bridge PID and its
starttime. Rollback modifies only AGILLM_BRIDGE_RUNTIME in the current env, so
checkpoint/dtype/pin changes made since rollout are retained. Full env backup is
for recovery evidence only and is NEVER restored wholesale.

Only the exact verified bridge is sent SIGTERM, using a pidfd. No worker,
supervisor, trainer, process group, or unrelated process receives a signal.
Old worker identities must disappear through stdin pipe EOF; otherwise stop and
report, never escalate to a blanket kill. This fallback needs the existing
supervisor to restart the bridge. An unhealthy/warming rollback fails closed and
requires separately reviewed recovery, not broader automatic process control.
"""

import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import tempfile
import time
import urllib.request


BASELINE_BRIDGE_SHA = "2cf7fef6cd4b7cb7c6768f028719a8c2e4fe0d9b08c53dc7b828ea075102688d"
BASELINE_RUNTIME_SHA = "65193f4a845e12c458f6cda0bebed621d0ee6a2f532ca95fcb5e9036ed32f0e2"
CANDIDATE_BRIDGE_SHA = "ce9bd513c3ac3d5e0afe9769ad0ac26408a859a693e26357714f6dfe3b70898b"
CANDIDATE_RUNTIME_SHA = "957dbf31c22ec2bc568eb3ca91644bdf09b59a1f2527d82dd8589d8f54716ff3"
STAGE = "/workspace/agillm43_live_infer/quality_20260904T2257Z"
RUNTIME_KEY = "AGILLM_BRIDGE_RUNTIME"


class GuardError(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise GuardError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def regular(path):
    path = Path(path)
    require(path.is_absolute(), "All target paths must be absolute")
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), "Target must be a regular non-symlink file")
    return path, info


def read_checked(path, expected=None):
    path, info = regular(path)
    data = path.read_bytes()
    if expected:
        require(digest(data) == expected, "File hash precondition failed")
    return data, info


def runtime_value(data):
    matches = []
    for raw in data.decode("utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            if key.strip() == RUNTIME_KEY:
                matches.append(value.strip())
    require(len(matches) == 1, "Env must declare the runtime key exactly once")
    value = matches[0]
    require(value.startswith("/") and "\x00" not in value, "Runtime value must be an absolute path")
    return value


def replace_runtime(data, expected, replacement):
    require(runtime_value(data) == expected, "Env runtime compare-and-swap precondition failed")
    require(replacement.startswith("/") and not any(x in replacement for x in "\r\n\x00"), "Invalid replacement runtime path")
    output = []
    for raw in data.decode("utf-8").splitlines(keepends=True):
        stripped = raw.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped and stripped.split("=", 1)[0].strip() == RUNTIME_KEY:
            ending = "\r\n" if raw.endswith("\r\n") else ("\n" if raw.endswith("\n") else "")
            output.append(RUNTIME_KEY + "=" + replacement + ending)
        else:
            output.append(raw)
    return "".join(output).encode("utf-8")


def atomic_replace(path, data, expected_sha, mode=None):
    """Hash-check immediately before same-directory replace; preserve ownership.

    The lock used by this script serializes this rollout only. External env/code
    writers must be paused/coordinated too; filesystem replace is not a universal
    transaction against arbitrary writers that ignore that lock.
    """
    path = Path(path)
    previous, info = read_checked(path, expected_sha)
    fd, temporary = tempfile.mkstemp(prefix=".agillm-rollout-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), mode if mode is not None else stat.S_IMODE(info.st_mode))
            os.fchown(handle.fileno(), info.st_uid, info.st_gid)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        latest, latest_info = read_checked(path, expected_sha)
        require((latest_info.st_dev, latest_info.st_ino, latest_info.st_mtime_ns) ==
                (info.st_dev, info.st_ino, info.st_mtime_ns) and latest == previous,
                "Concurrent target write detected; refusing replacement")
        os.replace(temporary, path)
        dirfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)  # Only this exact mkstemp-created file.


def private_write(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def proc_identity(pid):
    try:
        raw = Path(f"/proc/{int(pid)}/stat").read_text()
        tail = raw[raw.rfind(")") + 2:].split()
        return {"pid": int(pid), "ppid": int(tail[1]), "starttime": int(tail[19]), "state": tail[0]}
    except FileNotFoundError:
        return None


def check_identity(pid, starttime, ppid=None):
    current = proc_identity(pid)
    require(current is not None and current["starttime"] == int(starttime), "Process identity changed or disappeared")
    require(current["state"] not in {"Z", "X"}, "Expected live process is not running")
    if ppid is not None:
        require(current["ppid"] == int(ppid), "Process parent precondition failed")
    return current


def stable_identity(identity):
    return identity["pid"], identity["ppid"], identity["starttime"]


def proc_args(pid):
    try:
        return [x.decode("utf-8", "surrogateescape") for x in Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0") if x]
    except FileNotFoundError:
        return []


def has_script(pid, path):
    argv = proc_args(pid)
    if not argv or not re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(argv[0]).name):
        return False
    index = 1
    while index < len(argv):
        argument = argv[index]
        if argument in {"-c", "-m"}:
            return False  # Not a directly executed Python script.
        if argument == "--":
            index += 1
            break
        if argument in {"-W", "-X"}:
            index += 2
            continue
        if argument.startswith("-"):
            if argument in {"-u", "-B", "-S", "-s", "-E", "-I", "-O", "-OO", "-q", "-v", "-vv", "-b", "-bb", "-x"} or argument.startswith(("-W", "-X")):
                index += 1
                continue
            return False
        break
    if index >= len(argv):
        return False
    try:
        script = Path(argv[index])
        if not script.is_absolute():
            script = Path(os.readlink(f"/proc/{pid}/cwd")) / script
        return script.resolve() == Path(path).resolve()
    except OSError:
        return False


def children(pid):
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        identity = proc_identity(int(entry.name))
        if identity and identity["ppid"] == pid:
            found.append(identity)
    return found


def owns_listener(pid, port):
    inodes = set()
    for fd in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(fd)
        except FileNotFoundError:
            continue
        match = re.fullmatch(r"socket:\[(\d+)\]", target)
        if match:
            inodes.add(match.group(1))
    for table in ("tcp", "tcp6"):
        for line in Path(f"/proc/{pid}/net/{table}").read_text().splitlines()[1:]:
            fields = line.split()
            if fields[3] == "0A" and int(fields[1].split(":")[-1], 16) == port and fields[9] in inodes:
                return True
    return False


def health(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
        payload = json.load(response)
    require(isinstance(payload, dict), "Health response is not an object")
    return payload


def idle_health(port):
    payload = health(port)
    require(payload.get("ok") is True and payload.get("ready") is True and payload.get("warming") is False,
            "Bridge must be ready and not warming")
    for key in ("busy", "active_request_count", "pending_request_count", "draining_slots"):
        require(type(payload.get(key)) is int and payload[key] == 0, "Bridge not provably idle/drained")
    require(type(payload.get("live_slots")) is int and payload["live_slots"] > 0, "Missing live worker count")
    return payload


def check_anchors(args):
    check_identity(args.supervisor_pid, args.supervisor_starttime)
    check_identity(args.trainer_pid, args.trainer_starttime)
    require("train" in proc_args(args.trainer_pid), "Pinned trainer PID no longer has train command")


def check_bridge(args, runtime):
    check_anchors(args)
    current = check_identity(args.bridge_pid, args.bridge_starttime, args.supervisor_pid)
    require(has_script(args.bridge_pid, args.bridge_file), "Bridge command does not contain exact target script")
    require(owns_listener(args.bridge_pid, args.port), "Exact bridge PID does not own expected listener")
    snapshot = idle_health(args.port)
    workers = children(args.bridge_pid)
    require(len(workers) == snapshot["live_slots"], "Worker set differs from idle health snapshot")
    for worker in workers:
        argv = proc_args(worker["pid"])
        require(runtime in argv and "infer" in argv and "--server" in argv,
                "Unexpected direct child; no process changes allowed")
        require("--device" in argv and argv[argv.index("--device") + 1] == "cpu",
                "Worker must be explicit CPU inference")
        require(worker["pid"] not in {args.trainer_pid, args.supervisor_pid}, "Protected PID appeared as worker")
    return current, workers


def wait_gone(identities, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = [entry for entry in identities if
                     (current := proc_identity(entry["pid"])) is not None and
                     current["starttime"] == entry["starttime"]]
        if not remaining:
            return
        time.sleep(0.5)
    raise GuardError("Old bridge/worker identities did not disappear; no additional signals sent")


def wait_replacement(args, runtime):
    deadline = time.monotonic() + args.ready_timeout
    while time.monotonic() < deadline:
        check_anchors(args)
        candidates = [entry for entry in children(args.supervisor_pid)
                      if has_script(entry["pid"], args.bridge_file) and
                      (entry["pid"], entry["starttime"]) != (args.bridge_pid, args.bridge_starttime)]
        require(len(candidates) <= 1, "Multiple replacement bridges found; refusing ambiguous success")
        if len(candidates) == 1:
            candidate = candidates[0]
            try:
                snapshot = idle_health(args.port)
                if owns_listener(candidate["pid"], args.port):
                    workers = children(candidate["pid"])
                    if len(workers) == snapshot["live_slots"] and all(
                            runtime in proc_args(worker["pid"]) and "--server" in proc_args(worker["pid"])
                            for worker in workers):
                        return candidate, workers
            except (OSError, ValueError, GuardError):
                pass
        time.sleep(0.5)
    raise GuardError("Replacement not verified ready before deadline; keep admission paused and inspect")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("rollout", "rollback"))
    parser.add_argument("--bridge-file", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, default=Path(STAGE))
    parser.add_argument("--state-dir", type=Path, default=Path(STAGE) / "rollout_backup")
    parser.add_argument("--bridge-pid", type=int, default=1314759)
    parser.add_argument("--bridge-starttime", type=int, required=True)
    parser.add_argument("--supervisor-pid", type=int, default=67)
    parser.add_argument("--supervisor-starttime", type=int, required=True)
    parser.add_argument("--trainer-pid", type=int, default=732670)
    parser.add_argument("--trainer-starttime", type=int, required=True)
    parser.add_argument("--port", type=int, default=8493)
    parser.add_argument("--ready-timeout", type=float, default=600)
    parser.add_argument("--runtime-sha", default=CANDIDATE_RUNTIME_SHA,
                        help="Expected candidate runtime SHA-256 (must match this reviewed rollout)")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--admission-paused", action="store_true")
    parser.add_argument("--external-writers-paused", action="store_true",
                        help="Confirm coordinated pause of env/code writers for the brief file transaction")
    return parser.parse_args()


def main(args):
    require(len({args.bridge_pid, args.supervisor_pid, args.trainer_pid}) == 3, "Protected process identities overlap")
    require(args.port == 8493, "This rollout is pinned to primary inference port 8493")
    require(args.runtime_sha == CANDIDATE_RUNTIME_SHA, "Runtime hash differs from the reviewed candidate")
    for path in (args.bridge_file, args.env_file, args.stage_dir, args.state_dir):
        require(path.is_absolute(), "Paths must be absolute")
    require(args.env_file.parent == args.bridge_file.parent and args.env_file.name == "bridge.env",
            "Bridge reads bridge.env beside its script; target paths must match")
    require(args.bridge_file != args.stage_dir / "bridge_candidate.py", "Staged candidate is not the live bridge target")
    candidate_runtime = str(args.stage_dir / "runtime_candidate.py")
    env_before, env_info = read_checked(args.env_file)
    require(stat.S_IMODE(env_info.st_mode) == 0o600, "bridge.env must already have mode 0600")

    if args.action == "rollout":
        require(not args.state_dir.exists(), "Rollout backup already exists; inspect prior transaction")
        original_runtime = runtime_value(env_before)
        read_checked(original_runtime, BASELINE_RUNTIME_SHA)
        original_bridge, original_info = read_checked(args.bridge_file, BASELINE_BRIDGE_SHA)
        destination_bridge, _ = read_checked(args.stage_dir / "bridge_candidate.py", CANDIDATE_BRIDGE_SHA)
        runtime_source, _ = read_checked(candidate_runtime, CANDIDATE_RUNTIME_SHA)
        ast.parse(destination_bridge.decode("utf-8"))
        ast.parse(runtime_source.decode("utf-8"))
        source_runtime, destination_runtime = original_runtime, candidate_runtime
        source_bridge_sha, destination_bridge_sha = BASELINE_BRIDGE_SHA, CANDIDATE_BRIDGE_SHA
    else:
        manifest_bytes, _ = read_checked(args.state_dir / "manifest.json")
        manifest = json.loads(manifest_bytes)
        require(manifest.get("bridge_file") == str(args.bridge_file) and manifest.get("env_file") == str(args.env_file),
                "Rollback target differs from recorded transaction")
        require(manifest.get("candidate_runtime") == candidate_runtime, "Rollback candidate path differs")
        original_runtime = manifest["original_runtime"]
        read_checked(original_runtime, BASELINE_RUNTIME_SHA)
        destination_bridge, _ = read_checked(args.state_dir / "bridge.original.py", BASELINE_BRIDGE_SHA)
        read_checked(args.bridge_file, CANDIDATE_BRIDGE_SHA)
        source_runtime, destination_runtime = candidate_runtime, original_runtime
        source_bridge_sha, destination_bridge_sha = CANDIDATE_BRIDGE_SHA, BASELINE_BRIDGE_SHA
    replace_runtime(env_before, source_runtime, destination_runtime)
    original_identity, workers = check_bridge(args, source_runtime)
    require(hasattr(os, "pidfd_open") and hasattr(signal, "pidfd_send_signal"), "pidfd signaling support required")
    if not args.execute:
        print(json.dumps({"ok": True, "action": args.action, "read_only_preflight": True,
                          "bridge_pid": args.bridge_pid, "worker_pids": [x["pid"] for x in workers],
                          "trainer_identity_verified": True, "requires_admission_pause": True,
                          "requires_external_writer_pause": True}, sort_keys=True))
        return
    require(args.admission_paused and args.external_writers_paused,
            "Execution requires confirmed upstream admission pause and coordinated env/code writer pause")
    lock_path = args.bridge_file.parent / ".agillm-inference-rollout.lock"
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    bridge_fd = os.pidfd_open(args.bridge_pid)
    try:
        # Revalidate after lock acquisition and pidfd pinning, before any write.
        current, current_workers = check_bridge(args, source_runtime)
        require(stable_identity(current) == stable_identity(original_identity) and
                sorted(map(stable_identity, current_workers)) == sorted(map(stable_identity, workers)),
                "Process set changed during preflight")
        if args.action == "rollout":
            args.state_dir.mkdir(mode=0o700)
            private_write(args.state_dir / "bridge.original.py", original_bridge)
            private_write(args.state_dir / "bridge.env.original.private", env_before)
            manifest = {"schema": 1, "bridge_file": str(args.bridge_file), "env_file": str(args.env_file),
                        "original_runtime": original_runtime, "candidate_runtime": candidate_runtime,
                        "baseline_bridge_sha256": BASELINE_BRIDGE_SHA, "candidate_bridge_sha256": CANDIDATE_BRIDGE_SHA,
                        "baseline_runtime_sha256": BASELINE_RUNTIME_SHA, "candidate_runtime_sha256": CANDIDATE_RUNTIME_SHA,
                        "original_bridge": original_identity, "original_workers": workers,
                        "trainer": check_identity(args.trainer_pid, args.trainer_starttime)}
            private_write(args.state_dir / "manifest.json", json.dumps(manifest, indent=2).encode() + b"\n")
        atomic_replace(args.bridge_file, destination_bridge, source_bridge_sha)
        # Fresh read preserves unrelated checkpoint/pin/dtype edits since preflight.
        env_current, _ = read_checked(args.env_file)
        env_updated = replace_runtime(env_current, source_runtime, destination_runtime)
        try:
            atomic_replace(args.env_file, env_updated, digest(env_current), mode=0o600)
        except Exception:
            # Process was not signaled. Restore only our bridge code replacement.
            source_bytes, _ = read_checked(args.state_dir / "bridge.original.py", BASELINE_BRIDGE_SHA) if args.action == "rollout" else read_checked(args.stage_dir / "bridge_candidate.py", CANDIDATE_BRIDGE_SHA)
            atomic_replace(args.bridge_file, source_bytes, destination_bridge_sha)
            raise
        try:
            read_checked(args.bridge_file, destination_bridge_sha)
            require(runtime_value(read_checked(args.env_file)[0]) == destination_runtime, "Runtime key changed before restart")
            # Old process/worker argv still identifies source runtime until SIGTERM.
            check_bridge(args, source_runtime)
        except Exception:
            # No signal has been sent: undo only our exact file changes. If an
            # outside writer changed either target, its CAS fails and we stop.
            latest_env, _ = read_checked(args.env_file)
            restored_env = replace_runtime(latest_env, destination_runtime, source_runtime)
            atomic_replace(args.env_file, restored_env, digest(latest_env), mode=0o600)
            source_bytes, _ = read_checked(args.state_dir / "bridge.original.py", BASELINE_BRIDGE_SHA) if args.action == "rollout" else read_checked(args.stage_dir / "bridge_candidate.py", CANDIDATE_BRIDGE_SHA)
            atomic_replace(args.bridge_file, source_bytes, destination_bridge_sha)
            raise GuardError("Final idle/identity guard failed; own file changes restored without signaling")
        signal.pidfd_send_signal(bridge_fd, signal.SIGTERM)
        private_write(args.state_dir / (args.action + ".signal-sent.json"),
                      json.dumps({"bridge_pid": args.bridge_pid, "starttime": args.bridge_starttime}).encode() + b"\n")
        wait_gone([original_identity] + workers, 60)
        replacement, replacement_workers = wait_replacement(args, destination_runtime)
        check_anchors(args)
        read_checked(args.bridge_file, destination_bridge_sha)
        require(runtime_value(read_checked(args.env_file)[0]) == destination_runtime, "Runtime changed during verification")
        result = {"ok": True, "action": args.action, "bridge": replacement,
                  "worker_pids": [x["pid"] for x in replacement_workers], "old_workers_gone": True,
                  "trainer_identity_unchanged": True, "admission_still_paused_by_caller": True,
                  "checkpoint_env_not_restored_from_backup": True}
        private_write(args.state_dir / (args.action + ".verified.json"), json.dumps(result, indent=2).encode() + b"\n")
        print(json.dumps(result, sort_keys=True))
    finally:
        os.close(bridge_fd)
        os.close(lock_fd)


if __name__ == "__main__":
    try:
        main(parse_args())
    except Exception as exc:
        # Never dump env data, command lines, response bodies, or exception args.
        message = str(exc) if isinstance(exc, GuardError) else "Operational failure; inspect local state securely"
        print(json.dumps({"ok": False, "error_type": type(exc).__name__, "error": message,
                          "automatic_extra_signals": False, "keep_admission_paused_if_execution_started": True}))
        raise SystemExit(1)
