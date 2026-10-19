#!/usr/bin/env bash
set -uo pipefail
cd /workspace/agillm43_live_infer
printf '%s\n' "$$" > rollover-supervisor.pid
while true; do
  python3 -u ./watch_latest.py --poll 30 --min-age 60 --confirmations 2
  rc=$?
  printf '[rollover-supervisor] watcher exited rc=%s at %s; restarting\n' "$rc" "$(date -Is)"
  sleep 5
done
