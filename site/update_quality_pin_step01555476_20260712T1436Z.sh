#!/usr/bin/env bash
set -euo pipefail
ENV='/opt/agillm43_netserve/bridge.env'
BACKUP='/opt/agillm43_netserve/bridge.env.bak.20260712T1436Z-before-quality-pin-update'
OLD='/opt/agillm43_offload/ckpts/pretrain_delta_step01419008_from01017712_20260710T1616Z.pt'
NEW='/opt/agillm43_offload/ckpts/pretrain_delta_step01555476_from01017712_20260711T1616Z.pt'
SERVICE='agillm43-bridge.service'
armed=0
success=0
on_exit() {
  local rc=$?
  trap - EXIT
  if (( armed == 1 && success == 0 )); then
    cp -a "$BACKUP" "$ENV.pin-rollback.tmp"
    mv -f "$ENV.pin-rollback.tmp" "$ENV"
    systemctl reload "$SERVICE" || true
    printf 'pin_update_rollback=attempted rc=%s\n' "$rc" >&2
  fi
  exit "$rc"
}
trap on_exit EXIT
test -f "$NEW"
test ! -e "$BACKUP"
test "$(grep -c '^AGILLM_BRIDGE_PINNED_CKPT=' "$ENV")" = 1
test "$(grep '^AGILLM_BRIDGE_PINNED_CKPT=' "$ENV")" = "AGILLM_BRIDGE_PINNED_CKPT=$OLD"
test "$(grep '^AGILLM_BRIDGE_CKPT=' "$ENV")" = "AGILLM_BRIDGE_CKPT=$NEW"
cp -a "$ENV" "$BACKUP"
armed=1
python3 - "$ENV" "$NEW" <<'PY'
import os, pathlib, stat, sys
p = pathlib.Path(sys.argv[1])
value = sys.argv[2]
st = p.stat()
lines = p.read_text().splitlines(keepends=True)
idx = [i for i, line in enumerate(lines) if line.startswith('AGILLM_BRIDGE_PINNED_CKPT=')]
if len(idx) != 1:
    raise SystemExit(f'expected exactly one AGILLM_BRIDGE_PINNED_CKPT, found {len(idx)}')
ending = '\n' if lines[idx[0]].endswith('\n') else ''
lines[idx[0]] = f'AGILLM_BRIDGE_PINNED_CKPT={value}{ending}'
tmp = p.with_name(p.name + '.pin-update.tmp')
with tmp.open('w', encoding='utf-8', newline='') as fh:
    fh.writelines(lines)
    fh.flush()
    os.fsync(fh.fileno())
os.chmod(tmp, stat.S_IMODE(st.st_mode))
os.chown(tmp, st.st_uid, st.st_gid)
os.replace(tmp, p)
PY
test "$(grep '^AGILLM_BRIDGE_PINNED_CKPT=' "$ENV")" = "AGILLM_BRIDGE_PINNED_CKPT=$NEW"
test "$(grep -v '^AGILLM_BRIDGE_PINNED_CKPT=' "$BACKUP" | sha256sum | awk '{print $1}')" = "$(grep -v '^AGILLM_BRIDGE_PINNED_CKPT=' "$ENV" | sha256sum | awk '{print $1}')"
printf 'pin_backup=%s backup_sha256=%s\n' "$BACKUP" "$(sha256sum "$BACKUP" | awk '{print $1}')"
printf 'pin_change_scope=AGILLM_BRIDGE_PINNED_CKPT-only env_sha256=%s\n' "$(sha256sum "$ENV" | awk '{print $1}')"
systemctl reload "$SERVICE"
success=1
printf 'bridge_reload=sent service=%s at=%s\n' "$SERVICE" "$(date -u +%FT%TZ)"
