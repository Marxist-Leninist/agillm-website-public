#!/usr/bin/env bash
set -euo pipefail
PART='/opt/agillm43_offload/ckpts/pretrain_delta_step01555476_from01017712_20260711T1616Z.pt.part.20260712T1427Z'
FINAL='/opt/agillm43_offload/ckpts/pretrain_delta_step01555476_from01017712_20260711T1616Z.pt'
OLD='/opt/agillm43_offload/ckpts/pretrain_delta_step01419008_from01017712_20260710T1616Z.pt'
ENV='/opt/agillm43_netserve/bridge.env'
BACKUP='/opt/agillm43_netserve/bridge.env.bak.20260712T1430Z-step01419008'
SERVICE='agillm43-bridge.service'
EXPECTED_BYTES='5756618527'
EXPECTED_SHA='539f0ba881f4efee640e16bf34d227cec438ba987b3540ccc957f50364862ca6'
TARGET_NAME='pretrain_delta_step01555476_from01017712_20260711T1616Z.pt'
OLD_NAME='pretrain_delta_step01419008_from01017712_20260710T1616Z.pt'

wait_ready() {
  local expected="$1" timeout="$2" start now body
  start=$(date +%s)
  while true; do
    body=$(curl -sS --max-time 5 http://127.0.0.1:8493/health 2>/dev/null || true)
    if HEALTH_BODY="$body" EXPECTED_CKPT="$expected" python3 -c 'import json,os,sys; d=json.loads(os.environ.get("HEALTH_BODY") or "{}"); sys.exit(0 if d.get("ready") is True and d.get("warming") is False and d.get("ckpt")==os.environ["EXPECTED_CKPT"] else 1)' 2>/dev/null; then
      printf 'warm_ready=%s at=%s\n' "$expected" "$(date -u +%FT%TZ)"
      printf '%s\n' "$body"
      return 0
    fi
    now=$(date +%s)
    if (( now - start >= timeout )); then
      printf 'warm_timeout=%ss last_health=%s\n' "$timeout" "$body" >&2
      return 1
    fi
    if (( (now - start) % 30 < 5 )); then printf 'warming elapsed=%ss health=%s\n' "$((now-start))" "$body"; fi
    sleep 5
  done
}

rollback() {
  trap - EXIT
  printf 'rollback_begin=%s backup=%s\n' "$(date -u +%FT%TZ)" "$BACKUP"
  test -f "$BACKUP"
  cp -a "$BACKUP" "$ENV.rollback.tmp"
  mv -f "$ENV.rollback.tmp" "$ENV"
  systemctl restart "$SERVICE"
  if wait_ready "$OLD_NAME" 900; then
    printf 'rollback_result=PASS env=%s service=%s\n' "$ENV" "$SERVICE"
    return 0
  fi
  printf 'rollback_result=FAIL\n' >&2
  return 1
}

if [[ "${1:-promote}" == 'rollback' ]]; then rollback; exit $?; fi

armed=0
success=0
on_exit() {
  local rc=$?
  trap - EXIT
  if (( armed == 1 && success == 0 )); then
    printf 'promotion_failure_rc=%s automatic_rollback=begin\n' "$rc" >&2
    rollback || true
  fi
  exit "$rc"
}
trap on_exit EXIT

test -f "$OLD"
test -f "$ENV"
test ! -e "$BACKUP"
test ! -e "$FINAL"
actual_bytes=$(stat -c %s "$PART")
actual_sha=$(sha256sum "$PART" | awk '{print $1}')
printf 'verified_candidate bytes=%s sha256=%s\n' "$actual_bytes" "$actual_sha"
test "$actual_bytes" = "$EXPECTED_BYTES"
test "$actual_sha" = "$EXPECTED_SHA"
old_line=$(grep '^AGILLM_BRIDGE_CKPT=' "$ENV")
test "$old_line" = "AGILLM_BRIDGE_CKPT=$OLD"
test "$(grep -c '^AGILLM_BRIDGE_CKPT=' "$ENV")" = 1
cp -a "$ENV" "$BACKUP"
printf 'env_backup=%s backup_sha256=%s\n' "$BACKUP" "$(sha256sum "$BACKUP" | awk '{print $1}')"
armed=1
chown --reference="$OLD" "$PART"
chmod --reference="$OLD" "$PART"
mv "$PART" "$FINAL"
printf 'checkpoint_promoted=%s bytes=%s sha256=%s\n' "$FINAL" "$(stat -c %s "$FINAL")" "$(sha256sum "$FINAL" | awk '{print $1}')"
python3 - "$ENV" "$FINAL" <<'PY'
import os, pathlib, stat, sys
p = pathlib.Path(sys.argv[1])
value = sys.argv[2]
st = p.stat()
lines = p.read_text().splitlines(keepends=True)
idx = [i for i, line in enumerate(lines) if line.startswith('AGILLM_BRIDGE_CKPT=')]
if len(idx) != 1:
    raise SystemExit(f'expected exactly one AGILLM_BRIDGE_CKPT, found {len(idx)}')
ending = '\n' if lines[idx[0]].endswith('\n') else ''
lines[idx[0]] = f'AGILLM_BRIDGE_CKPT={value}{ending}'
tmp = p.with_name(p.name + '.promote.tmp')
with tmp.open('w', encoding='utf-8', newline='') as fh:
    fh.writelines(lines)
    fh.flush()
    os.fsync(fh.fileno())
os.chmod(tmp, stat.S_IMODE(st.st_mode))
os.chown(tmp, st.st_uid, st.st_gid)
os.replace(tmp, p)
PY
test "$(grep -c '^AGILLM_BRIDGE_CKPT=' "$ENV")" = 1
test "$(grep '^AGILLM_BRIDGE_CKPT=' "$ENV")" = "AGILLM_BRIDGE_CKPT=$FINAL"
test "$(grep -v '^AGILLM_BRIDGE_CKPT=' "$BACKUP" | sha256sum | awk '{print $1}')" = "$(grep -v '^AGILLM_BRIDGE_CKPT=' "$ENV" | sha256sum | awk '{print $1}')"
printf 'env_change_scope=AGILLM_BRIDGE_CKPT-only current_env_sha256=%s\n' "$(sha256sum "$ENV" | awk '{print $1}')"
systemctl restart "$SERVICE"
printf 'service_restart=%s at=%s\n' "$SERVICE" "$(date -u +%FT%TZ)"
wait_ready "$TARGET_NAME" 900
systemctl is-active --quiet "$SERVICE"
success=1
printf 'promotion_result=PASS service=%s checkpoint=%s backup=%s\n' "$SERVICE" "$FINAL" "$BACKUP"
