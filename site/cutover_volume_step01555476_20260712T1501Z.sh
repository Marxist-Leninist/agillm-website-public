#!/usr/bin/env bash
set -euo pipefail
ENV='/opt/agillm43_netserve/bridge.env'
BACKUP='/opt/agillm43_netserve/bridge.env.bak.20260712T1501Z-pre-step01555476-volume'
BRIDGE='/opt/agillm43_netserve/agillm43_http_bridge.py'
BRIDGE_BACKUP='/opt/agillm43_netserve/agillm43_http_bridge.py.bak.20260712T1451Z-preready-log'
SERVICE='agillm43-bridge.service'
OLD='/opt/agillm43_offload/ckpts/pretrain_delta_step01419008_from01017712_20260710T1616Z.pt'
OLD_NAME='pretrain_delta_step01419008_from01017712_20260710T1616Z.pt'
NEW='/mnt/geth-vol1/agillm43/ckpts/pretrain_delta_step01555476_from01017712_20260711T1616Z.pt'
NEW_NAME='pretrain_delta_step01555476_from01017712_20260711T1616Z.pt'
EXPECTED_BYTES='5756618527'
EXPECTED_SHA='539f0ba881f4efee640e16bf34d227cec438ba987b3540ccc957f50364862ca6'

wait_ready() {
  local expected="$1" timeout="$2" start now body
  start=$(date +%s)
  while true; do
    body=$(curl -sS --max-time 5 http://127.0.0.1:8493/health 2>/dev/null || true)
    if HEALTH_BODY="$body" EXPECTED_CKPT="$expected" python3 -c 'import json,os,sys; d=json.loads(os.environ.get("HEALTH_BODY") or "{}"); sys.exit(0 if d.get("ready") is True and d.get("warming") is False and d.get("ckpt")==os.environ["EXPECTED_CKPT"] and int(d.get("live_slots") or 0)>=1 else 1)' 2>/dev/null; then
      printf 'warm_ready=%s at=%s health=%s\n' "$expected" "$(date -u +%FT%TZ)" "$body"
      return 0
    fi
    now=$(date +%s)
    if (( now - start >= timeout )); then printf 'warm_timeout=%ss last_health=%s\n' "$timeout" "$body" >&2; return 1; fi
    if (( (now - start) % 30 < 5 )); then printf 'warming elapsed=%ss health=%s\n' "$((now-start))" "$body"; fi
    sleep 5
  done
}

rollback() {
  trap - EXIT
  printf 'rollback_begin=%s env_backup=%s\n' "$(date -u +%FT%TZ)" "$BACKUP"
  cp -a "$BACKUP" "$ENV.rollback.tmp"
  mv -f "$ENV.rollback.tmp" "$ENV"
  systemctl restart "$SERVICE"
  if wait_ready "$OLD_NAME" 900; then printf 'rollback_result=PASS bridge_patch_retained=yes\n'; return 0; fi
  printf 'rollback_old_failed_restoring_bridge_source=yes\n' >&2
  cp -a "$BRIDGE_BACKUP" "$BRIDGE"
  python3 -m py_compile "$BRIDGE"
  systemctl restart "$SERVICE"
  wait_ready "$OLD_NAME" 900
  printf 'rollback_result=PASS bridge_patch_retained=no\n'
}

armed=0
success=0
on_exit() {
  local rc=$?
  trap - EXIT
  if (( armed == 1 && success == 0 )); then printf 'cutover_failure_rc=%s automatic_rollback=begin\n' "$rc" >&2; rollback || true; fi
  exit "$rc"
}
trap on_exit EXIT

test -f "$OLD"
test -f "$NEW"
test -f "$NEW.sha256"
test ! -e "$BACKUP"
test ! -e /mnt/geth-vol1/agillm43/candidate_step01555476/candidate-auth.pid.running
! ss -lnt '( sport = :8494 )' | tail -n +2 | grep -q .
actual_bytes=$(stat -c %s "$NEW")
actual_sha=$(sha256sum "$NEW" | awk '{print $1}')
sidecar_sha=$(awk 'NR==1{print $1}' "$NEW.sha256")
printf 'production_candidate_verified path=%s bytes=%s sha256=%s sidecar_sha256=%s\n' "$NEW" "$actual_bytes" "$actual_sha" "$sidecar_sha"
test "$actual_bytes" = "$EXPECTED_BYTES"
test "$actual_sha" = "$EXPECTED_SHA"
test "$sidecar_sha" = "$EXPECTED_SHA"
test "$(grep '^AGILLM_BRIDGE_CKPT=' "$ENV")" = "AGILLM_BRIDGE_CKPT=$OLD"
test "$(grep '^AGILLM_BRIDGE_PINNED_CKPT=' "$ENV")" = "AGILLM_BRIDGE_PINNED_CKPT=$OLD"
test "$(grep '^AGILLM_BRIDGE_REQUIRE_PIN=' "$ENV")" = 'AGILLM_BRIDGE_REQUIRE_PIN=1'
cp -a "$ENV" "$BACKUP"
printf 'env_backup=%s sha256=%s\n' "$BACKUP" "$(sha256sum "$BACKUP" | awk '{print $1}')"
armed=1
python3 - "$ENV" "$NEW" <<'PY'
import os,pathlib,stat,sys
p=pathlib.Path(sys.argv[1]); target=sys.argv[2]; st=p.stat(); lines=p.read_text().splitlines(keepends=True)
for key in ('AGILLM_BRIDGE_CKPT','AGILLM_BRIDGE_PINNED_CKPT'):
    idx=[i for i,line in enumerate(lines) if line.startswith(key+'=')]
    if len(idx)!=1: raise SystemExit(f'{key} count={len(idx)}')
    ending='\n' if lines[idx[0]].endswith('\n') else ''
    lines[idx[0]]=f'{key}={target}{ending}'
req=[line.rstrip('\n') for line in lines if line.startswith('AGILLM_BRIDGE_REQUIRE_PIN=')]
if req != ['AGILLM_BRIDGE_REQUIRE_PIN=1']: raise SystemExit('AGILLM_BRIDGE_REQUIRE_PIN must remain 1')
cache=[i for i,line in enumerate(lines) if line.startswith('AGILLM43_DECOMPRESS_CACHE=')]
if len(cache)>1: raise SystemExit('duplicate AGILLM43_DECOMPRESS_CACHE')
if cache:
    ending='\n' if lines[cache[0]].endswith('\n') else ''
    lines[cache[0]]='AGILLM43_DECOMPRESS_CACHE=1'+ending
else:
    if lines and not lines[-1].endswith('\n'): lines[-1]+='\n'
    lines.append('AGILLM43_DECOMPRESS_CACHE=1\n')
tmp=p.with_name(p.name+'.cutover.tmp')
with tmp.open('w',encoding='utf-8',newline='') as f: f.writelines(lines); f.flush(); os.fsync(f.fileno())
os.chmod(tmp,stat.S_IMODE(st.st_mode)); os.chown(tmp,st.st_uid,st.st_gid); os.replace(tmp,p)
PY
python3 - "$BACKUP" "$ENV" <<'PY'
import pathlib,sys
a=pathlib.Path(sys.argv[1]).read_text().splitlines(); b=pathlib.Path(sys.argv[2]).read_text().splitlines()
def keep(lines):
    return [x for x in lines if not x.startswith(('AGILLM_BRIDGE_CKPT=','AGILLM_BRIDGE_PINNED_CKPT=','AGILLM43_DECOMPRESS_CACHE='))]
if keep(a)!=keep(b): raise SystemExit('unexpected bridge.env change outside CKPT, PINNED_CKPT, DECOMPRESS_CACHE')
PY
test "$(grep '^AGILLM_BRIDGE_CKPT=' "$ENV")" = "AGILLM_BRIDGE_CKPT=$NEW"
test "$(grep '^AGILLM_BRIDGE_PINNED_CKPT=' "$ENV")" = "AGILLM_BRIDGE_PINNED_CKPT=$NEW"
test "$(grep '^AGILLM_BRIDGE_REQUIRE_PIN=' "$ENV")" = 'AGILLM_BRIDGE_REQUIRE_PIN=1'
test "$(grep '^AGILLM43_DECOMPRESS_CACHE=' "$ENV")" = 'AGILLM43_DECOMPRESS_CACHE=1'
printf 'env_change_scope=CKPT,PINNED_CKPT,DECOMPRESS_CACHE-only env_sha256=%s\n' "$(sha256sum "$ENV" | awk '{print $1}')"
before_pid=$(systemctl show "$SERVICE" -p MainPID --value)
before_restarts=$(systemctl show "$SERVICE" -p NRestarts --value)
systemctl restart "$SERVICE"
printf 'service_restart=%s at=%s before_pid=%s before_nrestarts=%s\n' "$SERVICE" "$(date -u +%FT%TZ)" "$before_pid" "$before_restarts"
wait_ready "$NEW_NAME" 900
systemctl is-active --quiet "$SERVICE"
after_pid=$(systemctl show "$SERVICE" -p MainPID --value)
after_restarts=$(systemctl show "$SERVICE" -p NRestarts --value)
printf 'service_state=active after_pid=%s after_nrestarts=%s old_checkpoint_preserved=%s bridge_sha256=%s\n' "$after_pid" "$after_restarts" "$(test -f "$OLD" && echo yes || echo no)" "$(sha256sum "$BRIDGE" | awk '{print $1}')"
success=1
printf 'production_cutover=PASS checkpoint=%s env_backup=%s\n' "$NEW" "$BACKUP"
