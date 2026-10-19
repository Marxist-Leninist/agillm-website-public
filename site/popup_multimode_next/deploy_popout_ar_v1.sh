#!/usr/bin/env bash
# Deploy a new AR-only 49.1M popout artifact to the GETH popout service.
# Usage: deploy_popout_ar_v1.sh <parent_step> <artifact_url_or_path> <expected_sha256> [notes]
# Owner-authorised 2026-09-02. Downloads (or copies) the artifact into
# /mnt/geth-vol1/agillm44/detachable_50m/step<N>/, verifies sha256, writes
# manifests/agillm44_step<N>_ar_only.json, repoints manifests/current.json, updates the
# EnvironmentFile pins, restarts agillm44-popup-multimode-staged and waits for health.
set -euo pipefail
STEP=$1; SRC=$2; SHA=$3; NOTES=${4:-"AR-only v1-schema export (agillm5.detachable-50m.ar-export.v1) produced from a v2 multimode checkpoint with the receipt gate bypassed on owner authorisation 2026-09-02; no held-out receipt; SAT/NAT/gate untrained and unqualified."}
D=/opt/agillm43_netserve/popup_multimode_next; ENVF=/opt/agillm43_netserve/popup_49m_20260831/bridge.env
UNIT=/etc/systemd/system/agillm44-popup-multimode-staged.service
PAD=$(printf '%08d' "$STEP"); DIR=/mnt/geth-vol1/agillm44/detachable_50m/step$PAD; ART=$DIR/detachable_50m_step$PAD.pt
mkdir -p "$DIR"; chmod 750 "$DIR"
if [ ! -f "$ART" ]; then
  case "$SRC" in http*) curl -fsSL --retry 3 -o "$ART.part" "$SRC";; *) cp "$SRC" "$ART.part";; esac
  mv "$ART.part" "$ART"
fi
chmod 600 "$ART"
ACT=$(sha256sum "$ART" | cut -d' ' -f1); [ "$ACT" = "$SHA" ] || { echo "SHA MISMATCH $ACT != $SHA"; rm -f "$ART"; exit 2; }
TOKSHA=$(python3 -c "import json;print(json.load(open('$D/manifests/agillm44_step02570067_ar_only.json'))['tokenizer_sha256'])")
MAN=$D/manifests/agillm44_step${PAD}_ar_only.json
python3 - "$MAN" "$STEP" "$ART" "$SHA" "$TOKSHA" "$NOTES" <<'PY'
import json,sys
man,step,art,sha,tok,notes=sys.argv[1:]
json.dump({"model_id":f"agillm44-popout-49.1m-step{int(step):08d}",
 "artifact":{"path":art,"sha256":sha,"schema":"agillm5.detachable-50m.ar-export.v1"},
 "tokenizer_sha256":tok,
 "model_metadata":{"label":"Pop-out 49.1M","parent_step":int(step),"core_ar_unique_parameters":49096704,"unique_parameters":49097218,"standalone":True,"nat_head":False},
 "notes":notes}, open(man,"w"), indent=2); print("manifest",man)
PY
# preflight exactly as the service does, before touching the live unit
/usr/bin/python3 $D/agillm44_detachable_50m_multimode.py --checkpoint "$ART" --expected-sha256 "$SHA" --expected-tokenizer-sha256 "$TOKSHA" --verify-only >/dev/null && echo "preflight OK"
ln -sfn "$MAN" $D/manifests/current.json
cp -p "$UNIT" "$UNIT.bak-$(date -u +%Y%m%dT%H%M%SZ)"
sed -i -E "s#--manifest [^ ]+#--manifest $D/manifests/current.json#" "$UNIT"
cp -p "$ENVF" "$ENVF.bak-$(date -u +%Y%m%dT%H%M%SZ)"
sed -i -E "s#^(AGILLM_BRIDGE_PINNED_CKPT|AGILLM_BRIDGE_CKPT)=.*#\1=$ART#" "$ENVF"
# Pre-warm the decompressed-weights cache OUTSIDE the hardened unit (ProtectSystem=strict makes
# /mnt/geth-vol1 read-only for the service; the worker cannot create .agillm43_decompressed_cache itself).
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 timeout 600 /usr/bin/python3 -u /opt/agillm43_netserve/runtime_derived/agillm41_runtime_938597eb7764_logits_sat_v13.py infer --device cpu --cpu_threads 4 --ckpt "$ART" --mode ar --max_new 1 --min_new 0 --temperature 0.0 --top_p 1.0 --greedy --plain-output --infer_dtype fp32 --prompt warm >/dev/null 2>&1 || true
test -d "$DIR/.agillm43_decompressed_cache" && echo "decompressed cache pre-warmed" || echo "WARNING: cache not created"
systemctl daemon-reload; systemctl restart agillm44-popup-multimode-staged
for i in $(seq 1 90); do
  sleep 5
  H=$(curl -s -m 5 http://127.0.0.1:18597/health || true)
  if printf '%s' "$H" | python3 -c "import sys,json; d=json.load(sys.stdin); raise SystemExit(0 if d.get('ckpt') == 'detachable_50m_step$PAD.pt' and d.get('ready') is True else 1)" 2>/dev/null; then
    echo "HEALTH OK after ~$((i*5))s"
    printf '%s' "$H" | python3 -c "import sys,json;d=json.load(sys.stdin);print('ckpt',d.get('ckpt'),'parent_step',(d.get('model') or {}).get('parent_step'),'sha',(d.get('model') or {}).get('artifact_sha256'),'trained',d.get('trained_modes'))"
    exit 0
  fi
done
echo "HEALTH NOT READY after 450s"; systemctl status agillm44-popup-multimode-staged --no-pager | tail -5; tail -20 $D/bridge.log; exit 3
