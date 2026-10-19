#!/usr/bin/env bash
set -Eeuo pipefail

ENV_FILE=/opt/agillm43_netserve/bridge.env
SERVICE=agillm43-bridge.service
OLD_RUNTIME=/opt/agillm43_netserve/agillm41_runtime_083665a7.py
NEW_RUNTIME=/opt/agillm43_netserve/agillm41_runtime_d9860d5e.py
EXPECTED_SHA=d9860d5e96ba158db667b0493998a979f9a0e1e9b914041c485cdae44d735ed3
EXPECTED_CKPT=pretrain_delta_step01555476_from01017712_20260711T1616Z.pt
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
BACKUP="${ENV_FILE}.bak.${STAMP}-pre-d9860d5e"
HEALTH="/run/agillm43-d986-health.$$.json"
OLD_PID=$(systemctl show "$SERVICE" -p MainPID --value)
BACKUP_READY=0

health_matches() {
  curl -fsS --max-time 4 http://127.0.0.1:8493/health >"$HEALTH" 2>/dev/null || return 1
  /usr/bin/python3 -c 'import json,sys; h=json.load(open(sys.argv[1])); want=sys.argv[2]; raise SystemExit(0 if h.get("ready") is True and h.get("warming") is False and h.get("ckpt")==want and h.get("live_slots")==1 else 1)' "$HEALTH" "$EXPECTED_CKPT"
}

wait_ready() {
  local i
  for i in $(seq 1 90); do
    if health_matches; then
      return 0
    fi
    sleep 2
  done
  return 1
}

rollback() {
  local rc=$?
  trap - ERR
  if [[ "$BACKUP_READY" == 1 && -f "$BACKUP" ]]; then
    cp -a "$BACKUP" "$ENV_FILE"
    systemctl restart "$SERVICE" || true
    if wait_ready && pgrep -P "$(systemctl show "$SERVICE" -p MainPID --value)" -f "$OLD_RUNTIME" >/dev/null; then
      printf 'ROLLBACK_COMPLETE backup=%s restored_runtime=%s original_rc=%s\n' "$BACKUP" "$OLD_RUNTIME" "$rc"
    else
      printf 'ROLLBACK_ATTEMPTED_NOT_READY backup=%s original_rc=%s\n' "$BACKUP" "$rc"
    fi
  fi
  rm -f "$HEALTH"
  exit "$rc"
}
trap rollback ERR

systemctl is-active --quiet "$SERVICE"
systemctl is-enabled --quiet "$SERVICE"
test "$OLD_PID" -gt 1
pgrep -P "$OLD_PID" -f "$OLD_RUNTIME" >/dev/null
test ! -s /mnt/geth-vol1/agillm43/candidate_step01555476/candidate-auth.pid || ! kill -0 "$(cat /mnt/geth-vol1/agillm43/candidate_step01555476/candidate-auth.pid)" 2>/dev/null
! ss -lnt '( sport = :8494 )' | tail -n +2 | grep -q .
test "$(stat -c %s "$NEW_RUNTIME")" = 535212
test "$(sha256sum "$NEW_RUNTIME" | awk '{print $1}')" = "$EXPECTED_SHA"
test "$(stat -c %U:%G:%a "$NEW_RUNTIME")" = root:root:644
/usr/bin/python3 -m py_compile "$NEW_RUNTIME"
test "$(grep -c '^AGILLM_BRIDGE_RUNTIME=' "$ENV_FILE")" = 1
grep -q "^AGILLM_BRIDGE_RUNTIME=${OLD_RUNTIME}$" "$ENV_FILE"

cp -a "$ENV_FILE" "$BACKUP"
chmod 0600 "$BACKUP"
BACKUP_READY=1
sed -i "s#^AGILLM_BRIDGE_RUNTIME=.*#AGILLM_BRIDGE_RUNTIME=${NEW_RUNTIME}#" "$ENV_FILE"
test "$(grep -c '^AGILLM_BRIDGE_RUNTIME=' "$ENV_FILE")" = 1
grep -q "^AGILLM_BRIDGE_RUNTIME=${NEW_RUNTIME}$" "$ENV_FILE"
cmp -s <(grep -v '^AGILLM_BRIDGE_RUNTIME=' "$BACKUP") <(grep -v '^AGILLM_BRIDGE_RUNTIME=' "$ENV_FILE")

systemctl restart "$SERVICE"
wait_ready
NEW_PID=$(systemctl show "$SERVICE" -p MainPID --value)
test "$NEW_PID" -gt 1
test "$NEW_PID" != "$OLD_PID"
systemctl is-active --quiet "$SERVICE"
systemctl is-enabled --quiet "$SERVICE"
CHILD_PID=$(pgrep -P "$NEW_PID" -f "$NEW_RUNTIME" | head -n1)
test -n "$CHILD_PID"
grep -q "^AGILLM_BRIDGE_RUNTIME=${NEW_RUNTIME}$" "$ENV_FILE"

trap - ERR
rm -f "$HEALTH"
printf 'CUTOVER_PASS backup=%s old_pid=%s new_pid=%s child_pid=%s runtime=%s checkpoint=%s\n' \
  "$BACKUP" "$OLD_PID" "$NEW_PID" "$CHILD_PID" "$NEW_RUNTIME" "$EXPECTED_CKPT"
