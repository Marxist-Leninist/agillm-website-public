#!/usr/bin/env python3
"""Sanitized read-only 16B telemetry, with atomic last-good publication."""
import fcntl,json,math,os,shlex,subprocess,time,urllib.request
from pathlib import Path
ROOT=Path('/opt/agillm43_netserve')
OUT=Path(os.getenv('BINTF16B_TELEMETRY_OUT',str(ROOT/'bintf16b_training_status.json')))
LOCK=Path(os.getenv('BINTF16B_TELEMETRY_LOCK',str(ROOT/'bintf_telemetry/collector16b.lock')))
FIELDS=('step','net','net_step','seen','ar','masked','rate','rate_avg','rate_e2e','gpu_gb','time','steps','target','last_save','started','skipped','flips_pct','ps','ids','eta_days','unit','loss_unit','pid')
REMOTE=r'''import json,time,hashlib
from pathlib import Path
r=Path('/workspace/agillm16b_run')
for attempt in range(3):
 try:
  s=json.loads((r/'status.json').read_text());break
 except ValueError:
  if attempt==2:raise
  time.sleep(.25)
t={k:s.get(k) for k in @FIELDS@}
pid=s.get('pid')
try:
 cmd=Path('/proc/'+str(int(pid))+'/cmdline').read_bytes()
 t['process_alive']=b'bintf2_16b.py' in cmd and b'/workspace/agillm16b_run' in cmd
except (OSError,TypeError,ValueError):t['process_alive']=False
t['eval']={str(k):{q:v.get(q) for q in ('ar','ar_tail','nat16','step','seen','ps','together','ff_zero','long')} for k,v in s.get('eval',{}).items() if isinstance(v,dict)}
t['model_desc']=s.get('model','')[:600]
saved={}
for name in ('ckpt.good','ckpt'):
 p=r/name/'state.json'
 try:
  raw=p.read_bytes();d=json.loads(raw)
  saved[name]={'step':sum(d.get('steps',[])),'steps':d.get('steps'),'seen':d.get('seen'),'updated_at':p.stat().st_mtime,'state_sha256':hashlib.sha256(raw).hexdigest()}
 except (OSError,ValueError,TypeError):saved[name]=None
history=[]
try:
 with (r/'metrics.jsonl').open('rb') as f:
  size=f.seek(0,2);f.seek(max(0,size-196608));lines=f.read().decode('utf-8','replace').splitlines()
  if size>196608:lines=lines[1:]
 for line in lines[-160:]:
  try:
   d=json.loads(line);history.append({k:d.get(k) for k in ('step','time','net','ar','masked','rate')})
  except (ValueError,AttributeError):pass
except OSError:pass
print(json.dumps({'training':t,'saved':saved,'history':history,'source_time':time.time()}))
'''.replace('@FIELDS@',repr(FIELDS))
def clean(v):
 if isinstance(v,float) and not math.isfinite(v):return None
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
 if isinstance(v,list):return [clean(x) for x in v]
 return v
def number(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
def atomic_write(value):
 tmp=OUT.with_name(OUT.name+'.tmp')
 with tmp.open('w') as f:
  json.dump(clean(value),f,allow_nan=False,separators=(',',':'));f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o644)
 os.replace(tmp,OUT)
def read_trainer():
 command=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile=/opt/agillm43_netserve/bintf_telemetry/known_hosts16b','-o','GlobalKnownHostsFile=/dev/null','-o','HostKeyAlgorithms=ssh-ed25519','-o','ConnectTimeout=6','-o','ConnectionAttempts=1','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=1','-i','/root/.ssh/id_ed25519','-p','23851','root@x.x.x.x','python3 -c '+shlex.quote(REMOTE)]
 proc=subprocess.run(command,capture_output=True,text=True,timeout=20,check=True)
 data=json.loads(next(x for x in reversed(proc.stdout.splitlines()) if x.startswith('{')))
 if not all(number(x) for x in (data.get('source_time'),data.get('training',{}).get('time'),data.get('training',{}).get('step'))):raise ValueError('invalid telemetry')
 return data
def read_serving():
 try:
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
  with opener.open('http://127.0.0.1:18623/health',timeout=3) as r:h=json.load(r)
  m=h.get('model') or {}
  return {'ready':bool(h.get('ready')),'step':m.get('step'),'revision':h.get('runtime_revision'),'exported_at':m.get('exported_at'),'val_nats':m.get('val_nats'),'parameters':m.get('parameters'),'warming':bool(h.get('warming')),'checked_at':time.time()}
 except Exception:return {'ready':False,'status':'unknown','error':'health unavailable','checked_at':time.time()}
def collect():
 now=time.time();result={'schema':'agillm.bintf.telemetry.v1','model':'bintf16b','attempted_at':now,'interval_s':30,'stale_after_s':180,'ok':False}
 try:
  data=read_trainer();received=time.time();result.update(data)
  result.update(ok=True,last_success_at=received,training_age_s=max(0,data['source_time']-data['training']['time']))
  result['stale']=result['training_age_s']>180
  result['source']={'vast_instance_id':55988326,'clock_offset_s':data['source_time']-received}
 except Exception:
  try:
   prev=json.loads(OUT.read_text())
   for k in ('training','saved','history','last_success_at','source_time','source'):
    if k in prev:result[k]=prev[k]
  except (OSError,ValueError):pass
  result.update(stale=True,error='training telemetry temporarily unavailable')
  last=result.get('last_success_at');src=result.get('source_time');stamp=(result.get('training') or {}).get('time')
  result['training_age_s']=max(0,src-stamp)+max(0,now-last) if all(number(x) for x in (src,stamp,last)) else None
 result['serving']=read_serving();result['published_at']=time.time();atomic_write(result)
 print(json.dumps({k:result.get(k) for k in ('ok','stale','training_age_s','last_success_at')}))
 return result
def main():
 with LOCK.open('w') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:return
  collect()
if __name__=='__main__':main()
