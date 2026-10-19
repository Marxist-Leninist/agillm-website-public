#!/usr/bin/env bash
set -euo pipefail
SRC='/opt/agillm43_offload/ckpts/pretrain_delta_step01555476_from01017712_20260711T1616Z.pt'
VOL_ROOT='/mnt/geth-vol1/agillm43'
CKPT_DIR="$VOL_ROOT/ckpts"
CAND_DIR="$VOL_ROOT/candidate_step01555476"
NAME='pretrain_delta_step01555476_from01017712_20260711T1616Z.pt'
FINAL="$CKPT_DIR/$NAME"
PART="$FINAL.part.20260712T1453Z"
SIDECAR="$FINAL.sha256"
EXPECTED_BYTES='5756618527'
EXPECTED_SHA='539f0ba881f4efee640e16bf34d227cec438ba987b3540ccc957f50364862ca6'
PROD_ENV='/opt/agillm43_netserve/bridge.env'
PROD_BRIDGE='/opt/agillm43_netserve/agillm43_http_bridge.py'

test -f "$SRC"
test "$(stat -c %s "$SRC")" = "$EXPECTED_BYTES"
test "$(sha256sum "$SRC" | awk '{print $1}')" = "$EXPECTED_SHA"
test "$(stat -c %d /mnt/geth-vol1)" != "$(stat -c %d "$SRC")"
avail=$(df -B1 --output=avail /mnt/geth-vol1 | tail -n 1 | tr -d ' ')
test "$avail" -gt 15000000000
test ! -e "$FINAL"
test ! -e "$PART"
test ! -e "$SIDECAR"
test ! -e "$CAND_DIR"
install -d -m 755 -o root -g root "$CKPT_DIR" "$CAND_DIR"
printf 'copy_start=%s source=%s target=%s volume_avail=%s\n' "$(date -u +%FT%TZ)" "$SRC" "$PART" "$avail"
cp --reflink=never --sparse=always --preserve=mode,ownership,timestamps "$SRC" "$PART"
sync -f "$PART"
actual_bytes=$(stat -c %s "$PART")
actual_sha=$(sha256sum "$PART" | awk '{print $1}')
printf 'copy_verified bytes=%s sha256=%s\n' "$actual_bytes" "$actual_sha"
test "$actual_bytes" = "$EXPECTED_BYTES"
test "$actual_sha" = "$EXPECTED_SHA"
mv "$PART" "$FINAL"
printf '%s  %s\n' "$EXPECTED_SHA" "$NAME" >"$SIDECAR.part"
chmod 644 "$SIDECAR.part"
chown root:root "$SIDECAR.part"
sync -f "$SIDECAR.part"
mv "$SIDECAR.part" "$SIDECAR"
cp -a "$PROD_ENV" "$CAND_DIR/bridge.env"
cp -a "$PROD_BRIDGE" "$CAND_DIR/agillm43_http_bridge.py"
python3 - "$CAND_DIR/bridge.env" "$FINAL" <<'PY'
import os, pathlib, stat, sys
p=pathlib.Path(sys.argv[1]); target=sys.argv[2]; st=p.stat(); lines=p.read_text().splitlines(keepends=True)
for key in ('AGILLM_BRIDGE_CKPT','AGILLM_BRIDGE_PINNED_CKPT'):
    idx=[i for i,line in enumerate(lines) if line.startswith(key+'=')]
    if len(idx)!=1: raise SystemExit(f'{key} count={len(idx)}')
    ending='\n' if lines[idx[0]].endswith('\n') else ''
    lines[idx[0]]=f'{key}={target}{ending}'
req=[line for line in lines if line.startswith('AGILLM_BRIDGE_REQUIRE_PIN=')]
if req != ['AGILLM_BRIDGE_REQUIRE_PIN=1\n'] and req != ['AGILLM_BRIDGE_REQUIRE_PIN=1']:
    raise SystemExit('AGILLM_BRIDGE_REQUIRE_PIN must remain 1')
cache_idx=[i for i,line in enumerate(lines) if line.startswith('AGILLM43_DECOMPRESS_CACHE=')]
if len(cache_idx)>1: raise SystemExit('duplicate AGILLM43_DECOMPRESS_CACHE')
if cache_idx:
    ending='\n' if lines[cache_idx[0]].endswith('\n') else ''
    lines[cache_idx[0]]='AGILLM43_DECOMPRESS_CACHE=1'+ending
else:
    if lines and not lines[-1].endswith('\n'): lines[-1]+='\n'
    lines.append('AGILLM43_DECOMPRESS_CACHE=1\n')
tmp=p.with_name(p.name+'.tmp')
with tmp.open('w',encoding='utf-8',newline='') as f: f.writelines(lines); f.flush(); os.fsync(f.fileno())
os.chmod(tmp,stat.S_IMODE(st.st_mode)); os.chown(tmp,st.st_uid,st.st_gid); os.replace(tmp,p)
PY
python3 - "$CAND_DIR/agillm43_http_bridge.py" <<'PY'
import os,pathlib,stat,sys
p=pathlib.Path(sys.argv[1]); st=p.stat(); s=p.read_text()
old='    print("[bridge] v4 listening on 0.0.0.0:" + str(PORT) + " while pool warms", flush=True)\n    threading.Thread(target=_warm_initial_pool, daemon=True).start()\n    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()'
new='    bind_host = os.environ.get("AGILLM_BRIDGE_BIND", "0.0.0.0")\n    print("[bridge] v4 listening on " + bind_host + ":" + str(PORT) + " while pool warms", flush=True)\n    threading.Thread(target=_warm_initial_pool, daemon=True).start()\n    ThreadingHTTPServer((bind_host, PORT), H).serve_forever()'
if s.count(old)!=1: raise SystemExit(f'bind block count={s.count(old)}')
s=s.replace(old,new,1); tmp=p.with_name(p.name+'.tmp')
with tmp.open('w',encoding='utf-8',newline='') as f: f.write(s); f.flush(); os.fsync(f.fileno())
os.chmod(tmp,stat.S_IMODE(st.st_mode)); os.chown(tmp,st.st_uid,st.st_gid); os.replace(tmp,p)
PY
chmod 600 "$CAND_DIR/bridge.env"
chmod 755 "$CAND_DIR/agillm43_http_bridge.py"
python3 -m py_compile "$CAND_DIR/agillm43_http_bridge.py"
printf 'volume_candidate=%s bytes=%s sha256=%s sidecar=%s\n' "$FINAL" "$(stat -c %s "$FINAL")" "$(sha256sum "$FINAL" | awk '{print $1}')" "$SIDECAR"
printf 'candidate_env_sha256=%s candidate_bridge_sha256=%s\n' "$(sha256sum "$CAND_DIR/bridge.env" | awk '{print $1}')" "$(sha256sum "$CAND_DIR/agillm43_http_bridge.py" | awk '{print $1}')"
grep -E '^(AGILLM_BRIDGE_(CKPT|PINNED_CKPT|REQUIRE_PIN|RUNTIME|CONCURRENCY)|AGILLM43_DECOMPRESS_CACHE|TOKENIZER_(DIR|ID)|HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE)=' "$CAND_DIR/bridge.env"
