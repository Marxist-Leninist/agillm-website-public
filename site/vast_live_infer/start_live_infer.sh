#!/usr/bin/env bash
set -uo pipefail
cd /workspace/agillm43_live_infer

pid_matches() {
  local pidfile=$1 needle=$2 pid cmd
  [ -r "$pidfile" ] || return 1
  pid=$(cat "$pidfile" 2>/dev/null) || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null) || return 1
  [[ "$cmd" == *"$needle"* ]]
}

if ! pid_matches supervisor.pid supervise_bridge.sh; then
  nohup ./supervise_bridge.sh >> bridge.log 2>&1 < /dev/null &
  printf '%s\n' "$!" > supervisor.pid
fi

if ! pid_matches rollover-supervisor.pid supervise_rollover.sh; then
  nohup ./supervise_rollover.sh >> rollover.log 2>&1 < /dev/null &
  printf '%s\n' "$!" > rollover-supervisor.pid
fi
