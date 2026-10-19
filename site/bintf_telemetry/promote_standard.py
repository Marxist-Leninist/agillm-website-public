"""Promote the verified Standard checkpoint with the bridge's existing safe reload."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/opt/agillm43_netserve')
receipt_dir=root/'receipts/latest-20261008-standard'
verified=json.loads((receipt_dir/'STANDARD_STAGE_VERIFIED.json').read_text())
assert verified['ok'] and verified['step']==3558989
checkpoint=verified['checkpoint']
assert Path(checkpoint).is_file()
env=root/'bridge.env'
original=env.read_bytes()
backup=receipt_dir/'bridge.before-promotion.env'
assert not backup.exists(), 'Promotion backup already exists; inspect before retrying'
backup.write_bytes(original)
backup.chmod(0o600)
updates={'AGILLM_BRIDGE_CKPT':checkpoint,'AGILLM_BRIDGE_PINNED_CKPT':checkpoint,'AGILLM_BRIDGE_PINNED_SHA256':verified['checkpoint_sha256']}
lines=original.decode().splitlines()
for key,value in updates.items():
    assert sum(line.startswith(key+'=') for line in lines)==1
    lines=[key+'='+value if line.startswith(key+'=') else line for line in lines]
candidate=receipt_dir/'bridge.candidate.env'
candidate.write_text('\n'.join(lines)+'\n')
candidate.chmod(0o600)
check=subprocess.run(['python3',str(root/'agillm43_safe_reload.py'),'--env',str(candidate),'--check-only'],capture_output=True,text=True,timeout=45)
print(check.stdout,flush=True)
if check.returncode:
    raise RuntimeError('Safe checkpoint preflight failed: '+check.stderr)
assert env.read_bytes()==original, 'Bridge configuration changed concurrently'
tmp=env.with_suffix('.promotion-tmp')
tmp.write_bytes(candidate.read_bytes())
tmp.chmod(env.stat().st_mode & 0o777)
os.replace(tmp,env)
reload=subprocess.run(['systemctl','reload','agillm43-bridge.service'],capture_output=True,text=True,timeout=45)
if reload.returncode:
    env.write_bytes(original)
    raise RuntimeError('Reload request failed; prior configuration restored')
result={'requested_at':time.time(),'target_step':verified['step'],'target_checkpoint':checkpoint,'checkpoint_sha256':verified['checkpoint_sha256'],'previous_env_backup':str(backup),'safe_preflight':True,'reload_requested':True,'method':'blue-green warm, switch, drain; no trainer restart'}
(receipt_dir/'STANDARD_PROMOTION_REQUESTED.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
