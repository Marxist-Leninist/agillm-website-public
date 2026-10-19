"""Briefly pause only the verified serving refresher; always resume it."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

stage = Path("/workspace/agillm43_live_infer/quality_20260904T2257Z")
watcher_pid = 2871444
watcher = Path("/proc") / str(watcher_pid)
fd = os.pidfd_open(watcher_pid)
tail = (watcher / "stat").read_text().rsplit(")", 1)[1].split()
assert int(tail[19]) == 33933733 and int(tail[1]) == 2871442
assert (watcher / "cwd").resolve() == Path("/workspace/agillm43_live_infer")
assert b"./watch_latest.py" in (watcher / "cmdline").read_bytes().split(b"\0")
args = [sys.executable, str(stage / "guarded_inference_rollout.py"), "rollout",
        "--bridge-file", "/workspace/agillm43_live_infer/agillm43_http_bridge.py",
        "--env-file", "/workspace/agillm43_live_infer/bridge.env",
        "--bridge-starttime", "31073629", "--supervisor-starttime", "7776",
        "--trainer-starttime", "44400808", "--ready-timeout", "240",
        "--execute", "--admission-paused", "--external-writers-paused"]
try:
    signal.pidfd_send_signal(fd, signal.SIGSTOP)
    deadline = time.monotonic() + 3
    while (watcher / "stat").read_text().rsplit(")", 1)[1].split()[0] not in ("T", "t"):
        if time.monotonic() > deadline:
            raise RuntimeError("Serving refresher did not pause")
        time.sleep(.05)
    print("Serving refresher paused; trainer unaffected", flush=True)
    result = subprocess.run(args, timeout=315)
finally:
    signal.pidfd_send_signal(fd, signal.SIGCONT)
    os.close(fd)
    print("Serving refresher resumed", flush=True)
raise SystemExit(result.returncode)
