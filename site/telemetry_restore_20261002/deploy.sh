#!/bin/bash
# AGI44_TELEMETRY_RESTORE_20261002 deploy. Undo: bash undo.sh
set -euo pipefail
cd /opt/agillm43_netserve
TS=20261002T0751Z
D=telemetry_restore_20261002
FILES="agillm_curves_page.html agillm_curves_panel.html agillm_completion.html"
for f in $FILES; do
  cp -p "$f" "$D/$f.before-telemetry-restore-$TS"
  cp -p "$f" "$f.bak-before-telemetry-restore-$TS"
  python3 "$D/restore_telemetry_20261002.py" "$f" "$D/$f.patched"
done
sha256sum $D/*.patched
for f in $FILES; do
  cat "$D/$f.patched" > "$f"
done
sha256sum $FILES
stat -c '%n %a %U' $FILES
echo served_curves_marker=$(curl -s -m 15 http://127.0.0.1:8793/agillm-curves | grep -c AGI44_TELEMETRY_RESTORE_20261002)
