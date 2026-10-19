#!/bin/bash
# Undo everything from 2 Oct 2026:
#   1) AGI44_OBJECTIVE_TELEMETRY_20261002 backend + objective series (08:22Z) via backend/undo_objective_20261002.sh
#   2) AGI44_TELEMETRY_RESTORE_20261002 chart restore (07:51Z): put back the exact pre-restore bytes.
# Run `bash backend/undo_objective_20261002.sh` instead to undo only step 1.
set -euo pipefail
cd /opt/agillm43_netserve
if [ -x telemetry_restore_20261002/backend/undo_objective_20261002.sh ] || [ -f telemetry_restore_20261002/backend/undo_objective_20261002.sh ]; then
  bash telemetry_restore_20261002/backend/undo_objective_20261002.sh
fi
TS=20261002T0751Z
D=telemetry_restore_20261002
for f in agillm_curves_page.html agillm_curves_panel.html agillm_completion.html; do
  cat "$D/$f.before-telemetry-restore-$TS" > "$f"
done
sha256sum agillm_curves_page.html agillm_curves_panel.html agillm_completion.html
# expected after undo:
# 77bae1cc5468624a0f69f4ad6800362d47962100e49b276524157816f3df6fde  agillm_curves_page.html
# 88ff118cfa277b0b1a91aed945663a7b1126465a89839f4d1104cf27f1b32709  agillm_curves_panel.html
# 87506c8fec18a7077a4cc0951f3b3e6caa120891378903d2921a7dca6ad7e8c6  agillm_completion.html
