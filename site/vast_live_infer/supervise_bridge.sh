#!/usr/bin/env bash
set -uo pipefail
cd /workspace/agillm43_live_infer
while true; do
  set -a
  . ./bridge.env
  set +a
  python3 -u ./agillm43_http_bridge.py &
  child=$!
  printf '%s\n' "$child" > bridge.pid
  wait "$child"
  rc=$?
  printf '[supervisor] bridge exited rc=%s at %s; restarting\n' "$rc" "$(date -Is)"
  sleep 2
done
