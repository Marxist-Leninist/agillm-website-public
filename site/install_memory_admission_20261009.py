#!/usr/bin/env python3
"""Install the tested load-admission repair and activate only when Standard has no user work."""
import ast,hashlib,importlib.util,json,os,pathlib,subprocess,tempfile,time,urllib.request
ROOT=pathlib.Path('/opt/agillm43_netserve');P=ROOT/'agillm43_http_bridge.py';UNIT='agillm43-bridge.service'
EXPECTED='179cf461a0624822cbae1cc6a326f36644be7906e45eb84a882b019ccf33626d'
OUT=ROOT/'recovery_20261009_0952';OUT.mkdir(exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def health():
 return json.load(urllib.request.urlopen('http://127.0.0.1:8493/health',timeout=5))

tests=subprocess.run(['/usr/bin/python3',str(ROOT/'bridge_memory_admission.py')],capture_output=True,text=True,timeout=10)
if tests.returncode:raise SystemExit(tests.stderr or tests.stdout)
test_record=json.loads(tests.stdout);assert test_record['ok']
(OUT/'unit-tests.json').write_text(json.dumps(test_record,indent=2)+'\n')
before=P.read_bytes()
if sha(before)!=EXPECTED:raise SystemExit('Concurrent bridge source edit; no mutation.')
h=health()
if any(h.get(key,0) for key in ('busy','active_request_count','pending_request_count')):raise SystemExit('Active Standard request; no source switch or restart.')
s=before.decode();old='import hashlib, json, os, re, signal, subprocess, threading, time\n'
assert s.count(old)==1;s=s.replace(old,old+'from bridge_memory_admission import require_capacity as _require_load_capacity, status as _memory_admission_status\n',1)
old='    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,\n'
assert s.count(old)==1;s=s.replace(old,'    _require_load_capacity()  # Do not allocate a model that would exhaust this service or host.\n'+old,1)
old='                             "live_slots": len(live), "busy": busy,\n'
assert s.count(old)==1;s=s.replace(old,old+'                             "memory_admission": _memory_admission_status(),\n',1)
compile(s,str(P),'exec');ast.parse(s)
assert '_desired_concurrency()' in s and s.count('ThreadingHTTPServer')==before.decode().count('ThreadingHTTPServer')
(OUT/'bridge.before.py').write_bytes(before)
spec=importlib.util.spec_from_file_location('sg_service_coop_recovery','/opt/goddess/sg_service_coop.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);coop=mod.ServiceCoop()
obs=coop.handle({'unit':UNIT,'operation':'status','agent_id':'astra-geth-recovery-20261009-0944'})
if not obs.get('ok'):raise SystemExit('Native service observation unavailable; no mutation.')
if sha(P.read_bytes())!=EXPECTED:raise SystemExit('Concurrent bridge edit; no mutation.')
h2=health()
if any(h2.get(key,0) for key in ('busy','active_request_count','pending_request_count')):raise SystemExit('New request arrived; no mutation.')
fd,tmp=tempfile.mkstemp(prefix='.memory-admission-',dir=ROOT)
with os.fdopen(fd,'w') as f:f.write(s);f.flush();os.fsync(f.fileno())
os.chmod(tmp,P.stat().st_mode&0o777);os.replace(tmp,P)
receipt={'source_installed':True,'source_before':EXPECTED,'source_after':sha(P.read_bytes()),'time':time.time(),'tests_passed':len(test_record['tests']),'existing_checkpoint':h.get('target_ckpt'),'concurrency_config_changed':False,'memory_limit_changed':False,'trainer_changed':False,'activation_complete':False}
(OUT/'deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
result=coop.handle({'unit':UNIT,'operation':'restart','agent_id':'astra-geth-recovery-20261009-0944',
 'reason':'Restore website uptime: tested model-load admission prevents repeated cgroup OOM; Standard has no active or queued requests; no trainer/checkpoint changes.',
 'request_id':'bridge-memory-admission-recovery-20261009-0952',
 'expected_revision':obs['revision'],'expected_observation_sha256':obs['observation_sha256']})
receipt['native_service_receipt']=result;receipt['activation_dispatched']=result.get('execution_success') is True
(OUT/'deployment.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if result.get('execution_success') is not True:raise SystemExit('Native restart not confirmed. Reconcile this same request receipt before any retry.')
