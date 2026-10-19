#!/usr/bin/env python3
import hashlib,json,os,py_compile,shutil,time
from pathlib import Path
stamp="20260930T0519-fleet"
entries=[
("/root/agillm41_codex_monitor/agillm_ext.py","92c2732ff2ee9ce1b48765f2ac202a0683118282931da13a3e9b14157c16f86c"),
("/root/agillm41_codex_monitor/server.py","e310adbdf3d9ea149221c84b64af864c70d74ea26e96f8800320a9ff5dacdd59")]
payload=json.loads(Path("/root/agillm41_codex_monitor/fleet-append-20260930.json").read_text())
receipt=[]
for name,expected in entries:
    p=Path(name); actual=hashlib.sha256(p.read_bytes()).hexdigest()
    assert actual==expected,(name,"source changed",actual)
for name,expected in entries:
    p=Path(name); old=p.read_text()
    if p.name=="agillm_ext.py": new=old+payload["append"]
    else:
        a=old.index("def _live_payload():")
        b=old.index("\nPORT =",a)
        new=old[:a]+'def _live_payload():\n    import fleet_live\n    return fleet_live.live()\n'+old[b:]
    tmp=p.with_name(p.name+".candidate-fleet")
    tmp.write_text(new);py_compile.compile(str(tmp),doraise=True)
    backup=p.with_name(p.name+".bak-"+stamp);shutil.copy2(p,backup);os.replace(tmp,p)
    receipt.append({"path":str(p),"before":expected,"after":hashlib.sha256(p.read_bytes()).hexdigest(),"backup":str(backup)})
override=Path("/etc/systemd/system/agillm-rpv16-status.service.d/90-fleet.conf")
assert not override.exists(),"override already exists; inspect before retry"
override.write_text("[Unit]\nDescription=Read-only current Standard and RPV16 fleet telemetry\n[Service]\nExecStart=\nExecStart=/usr/bin/python3 /usr/local/libexec/agillm-fleet-status-collector-20260930.py\nExecStartPost=\nTimeoutStartSec=25\n")
receipt.append({"path":str(override),"after":hashlib.sha256(override.read_bytes()).hexdigest(),"rollback":"remove only this drop-in then daemon-reload"})
out=Path("/opt/agillm43_netserve/fleet-telemetry-deploy-receipt-20260930.json")
out.write_text(json.dumps({"applied_at":time.time(),"files":receipt,"monitor_restarted":False},indent=2))
print(out);print(json.dumps(receipt,indent=2))
