#!/usr/bin/env bash
set -euo pipefail
SRC='/opt/agillm43_netserve/agillm43_http_bridge.py'
BACKUP='/opt/agillm43_netserve/agillm43_http_bridge.py.bak.20260712T1451Z-preready-log'
test -f "$SRC"
test ! -e "$BACKUP"
cp -a "$SRC" "$BACKUP"
python3 - "$SRC" <<'PY'
import os, pathlib, stat, sys
p = pathlib.Path(sys.argv[1])
st = p.stat()
s = p.read_text()
old_deadline = '    deadline = time.time() + 600\n    while time.time() < deadline:\n        line = proc.stdout.readline()\n'
new_deadline = '    # Keep a small, redacted tail so pre-READY child failures are diagnosable.\n    startup_tail = []\n    deadline = time.time() + 600\n    while time.time() < deadline:\n        line = proc.stdout.readline()\n        if line:\n            safe_line = re.sub(r"(?i)(token|secret|key)(\\s*[=:]\\s*)\\S+", r"\\1\\2<redacted>", line.strip())[:512]\n            startup_tail.append(safe_line)\n            del startup_tail[:-8]\n'
old_exit = '            if proc.poll() is not None:\n                return None, ckpt, threads, dtype\n'
new_exit = '            if proc.poll() is not None:\n                print("[bridge] warm child exited before READY rc=%s tail=%s" % (proc.returncode, json.dumps(startup_tail, ensure_ascii=True)[:4096]), flush=True)\n                return None, ckpt, threads, dtype\n'
old_kill = '    try:\n        proc.kill()\n    except Exception:\n        pass\n    return None, ckpt, threads, dtype\n'
new_kill = '    print("[bridge] warm child timed out before READY tail=%s" % json.dumps(startup_tail, ensure_ascii=True)[:4096], flush=True)\n    try:\n        proc.kill()\n    except Exception:\n        pass\n    return None, ckpt, threads, dtype\n'
for old, new, label in ((old_deadline,new_deadline,'deadline'),(old_exit,new_exit,'exit'),(old_kill,new_kill,'timeout')):
    if s.count(old) != 1:
        raise SystemExit(f'{label} pattern count={s.count(old)}')
    s = s.replace(old,new,1)
tmp = p.with_name(p.name + '.preready-log.tmp')
with tmp.open('w', encoding='utf-8', newline='') as fh:
    fh.write(s)
    fh.flush()
    os.fsync(fh.fileno())
os.chmod(tmp, stat.S_IMODE(st.st_mode))
os.chown(tmp, st.st_uid, st.st_gid)
os.replace(tmp,p)
PY
if ! python3 -m py_compile "$SRC"; then cp -a "$BACKUP" "$SRC"; exit 1; fi
printf 'bridge_backup=%s backup_sha256=%s\n' "$BACKUP" "$(sha256sum "$BACKUP" | awk '{print $1}')"
printf 'bridge_patched=%s patched_sha256=%s service_restarted=no\n' "$SRC" "$(sha256sum "$SRC" | awk '{print $1}')"
