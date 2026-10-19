#!/usr/bin/env python3
"""Idle-gated production reliability deployment with rollback; weights stay untouched."""
import hashlib,json,os,shutil,subprocess,time,urllib.request
from pathlib import Path
D=Path(__file__).parent;R=D.parent
B=R/'agillm43_http_bridge.py';E=R/'bridge.env';OLD=R/'runtime_derived/runtime_standard_modes_20261001_minimum.py';NEW=R/'runtime_derived/runtime_standard_modes_20261002_cooperative_cancel.py'
manifest=json.loads((D/'manifest.json').read_text());tests=json.loads((D/'unit_tests.json').read_text())
assert tests['passed']==9
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(B)==manifest['original_bridge_sha256'],'bridge changed by another agent'
assert sha(OLD)==manifest['original_runtime_sha256'],'runtime changed by another agent'
assert sha(D/'bridge.patched.py')==manifest['bridge_sha256']
assert sha(D/'runtime.patched.py')==manifest['runtime_sha256']
with urllib.request.urlopen('http://127.0.0.1:8493/health',timeout=5) as f:health=json.load(f)
assert not any(health.get(k,0) for k in ('busy','active_request_count','pending_request_count')),'serving requests active: wait for idle'
config={}
for line in E.read_text().splitlines():
    if line.strip() and not line.lstrip().startswith('#') and '=' in line:
        k,v=line.split('=',1);config[k.strip()]=v.strip()
assert config['AGILLM_BRIDGE_RUNTIME']==str(OLD)
ckpt=Path(config['AGILLM_BRIDGE_CKPT']);I=Path(str(ckpt)+'.identity.json');identity=json.loads(I.read_text())
assert identity['runtime_sha256']==manifest['original_runtime_sha256']
assert Path(identity['runtime_file']).resolve()==OLD.resolve()
for src,name in [(B,'bridge.production.before.py'),(E,'bridge.env.before'),(I,'identity.before.json')]:
    dst=D/name;shutil.copy2(src,dst)
    if src==E:dst.chmod(0o600)
def atomic(path,data,mode=0o644):
    temp=path.with_name(path.name+'.inferencefix.tmp');temp.write_bytes(data);temp.chmod(mode);os.replace(temp,path)
atomic(NEW,(D/'runtime.patched.py').read_bytes())
newenv=E.read_text().replace('AGILLM_BRIDGE_RUNTIME='+str(OLD),'AGILLM_BRIDGE_RUNTIME='+str(NEW))
assert newenv!=E.read_text()
identity['runtime_file']=str(NEW);identity['runtime_sha256']=manifest['runtime_sha256']
identity['serving_reliability_patch']={'id':'cooperative-cancel-20261002','parent_runtime_sha256':manifest['original_runtime_sha256'],'receipt':str(D/'manifest.json'),'decoding_functions_changed':False,'checkpoint_weights_changed':False}
print('DEPLOY_START',json.dumps({'at':time.time(),'previous_health':{k:health.get(k) for k in ('ready','warming','loads','busy')},'checkpoint':ckpt.name,'runtime':NEW.name}),flush=True)
try:
    subprocess.run(['systemctl','stop','agillm43-bridge.service'],check=True,timeout=30)
    atomic(B,(D/'bridge.patched.py').read_bytes())
    atomic(E,newenv.encode(),0o600)
    atomic(I,json.dumps(identity,indent=2).encode())
    subprocess.run(['systemctl','start','agillm43-bridge.service'],check=True,timeout=20)
except Exception:
    for target,backup,mode in [(B,'bridge.production.before.py',0o644),(E,'bridge.env.before',0o600),(I,'identity.before.json',0o644)]:atomic(target,(D/backup).read_bytes(),mode)
    subprocess.run(['systemctl','start','agillm43-bridge.service'],timeout=20)
    raise
receipt={'status':'DEPLOYED_WARMING','at':time.time(),'service':'agillm43-bridge.service','bridge_sha256':sha(B),'runtime_sha256':sha(NEW),'checkpoint':str(ckpt),'checkpoint_sha256_unchanged':identity['model']['source_checkpoint_sha256'],'unit_tests_passed':tests['passed']}
(D/'deployment.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
