"""Temporarily route standard inference to the existing healthy backup only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("action", choices=["drain", "restore"])
a = p.parse_args()
config = Path("/opt/agillm43_netserve/agillm43-infer-edge.nginx.conf")
staging = Path("/opt/agillm43_netserve/quality_20260904T2257Z")
backup = staging / "edge.original.conf"
original_sha = "89d5e082f1ffb83b875c0fff638f07a7945257217c89d28faa39c9c0dda2737d"
needle = b"        server 127.0.0.1:8494 max_fails=1 fail_timeout=3s;"
replacement = b"        server 127.0.0.1:8494 max_fails=1 fail_timeout=3s down;"
current = config.read_bytes()
if a.action == "drain":
    with urllib.request.urlopen("http://127.0.0.1:8497/health", timeout=5) as r:
        health = json.load(r)
    assert health.get("ready") and not health.get("warming"), "Backup not ready"
    assert hashlib.sha256(current).hexdigest() == original_sha, "Edge configuration changed"
    assert current.count(needle) == 1 and not backup.exists(), "Unsafe or repeated staging"
    shutil.copy2(config, backup)
    os.chmod(backup, 0o600)
    desired = current.replace(needle, replacement)
else:
    original = backup.read_bytes()
    assert hashlib.sha256(original).hexdigest() == original_sha, "Invalid backup"
    assert current == original.replace(needle, replacement), "Edge changed during rollout"
    desired = original
temp = config.with_name(config.name + ".quality-tmp")
assert not temp.exists(), "Temporary edge file already exists"
temp.write_bytes(desired)
shutil.copymode(config, temp)
os.replace(temp, config)
nginx = ["/opt/nginx-goddess/nginx", "-c", str(config)]
try:
    subprocess.run(nginx + ["-t"], check=True, timeout=15)
    subprocess.run(nginx + ["-s", "reload"], check=True, timeout=15)
except Exception:
    temp.write_bytes(current)
    shutil.copymode(config, temp)
    os.replace(temp, config)
    subprocess.run(nginx + ["-t"], check=True, timeout=15)
    subprocess.run(nginx + ["-s", "reload"], check=True, timeout=15)
    raise
print(json.dumps({"action": a.action, "config_sha256": hashlib.sha256(desired).hexdigest(),
                  "backup_serving": a.action == "drain", "acl_changed": False}))
