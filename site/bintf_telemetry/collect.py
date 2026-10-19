#!/usr/bin/env python3
"""Read-only Binary 4B telemetry. Never changes trainer or checkpoint state."""
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request

ROOT = Path('/opt/agillm43_netserve/bintf_telemetry')
OUT = ROOT / 'status.json'
CONFIG = Path('/opt/bintf4b_serve/refresh.json')
FIELDS = ('step','net','net_step','seen','ar','masked','rate','rate_avg','gpu_gb','time','steps','target','last_save','started','skipped','flips_pct','ps')


def clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    return v


def atomic_write(value):
    tmp = OUT.with_suffix('.tmp')
    tmp.write_text(json.dumps(clean(value), separators=(',', ':'), allow_nan=False))
    tmp.chmod(0o644)
    os.replace(tmp, OUT)


def main():
    ROOT.mkdir(exist_ok=True)
    lock = (ROOT/'collector.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    now = time.time()
    result = {'schema': 'agillm.bintf.telemetry.v1', 'model': 'bintf4b', 'attempted_at': now, 'interval_s': 30, 'ok': False}
    try:
        conf = json.loads(CONFIG.read_text())
        run = conf.get('run', '/workspace/bintf4b_run')
        remote = '''import json,os,time,subprocess
from pathlib import Path
r=Path(RUN)
s=json.loads((r/'status.json').read_text())
t={k:s.get(k) for k in FIELDS}
pid=s.get('pid');t['process_alive']=bool(pid and Path('/proc/'+str(int(pid))).exists())
t['eval']={str(k):{q:v.get(q) for q in ['ar','ar_tail','nat16','step','seen','ps']} for k,v in s.get('eval',{}).items() if isinstance(v,dict)}
saved={}
for name in ['ckpt.good','ckpt']:
 p=r/name/'state.json'
 try:
  d=json.loads(p.read_text());saved[name]={'step':sum(d.get('steps',[])),'seen':d.get('seen'),'updated_at':p.stat().st_mtime}
 except (OSError,ValueError,TypeError):saved[name]=None
history=[]
try:
 with (r/'metrics.jsonl').open('rb') as f:
  size=f.seek(0,2);f.seek(max(0,size-196608));lines=f.read().decode('utf-8','replace').splitlines()
  if size>196608:lines=lines[1:]
 for line in lines[-160:]:
  try:
   d=json.loads(line);history.append({k:d.get(k) for k in ['step','time','net','ar','masked','rate']})
  except ValueError:pass
except OSError:pass
print(json.dumps({'training':t,'saved':saved,'history':history,'source_time':time.time()}))
'''.replace('RUN', repr(run)).replace('FIELDS', repr(FIELDS))
        ssh = ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=1','-i',conf.get('key','/root/.ssh/id_ed25519'),'-p',str(conf['vast_port']),'root@'+conf['vast_host']]
        proc = subprocess.run(ssh+['python3 -c '+shlex.quote(remote)], capture_output=True, text=True, timeout=22)
        if proc.returncode:
            raise RuntimeError('trainer_ssh_failed')
        payload = next((x for x in reversed(proc.stdout.splitlines()) if x.startswith('{')), '')
        data = json.loads(payload)
        result.update(data)
        result['ok'] = True
        result['last_success_at'] = time.time()
        result['training_age_s'] = max(0, data['source_time']-float(data['training']['time']))
        result['stale'] = result['training_age_s'] > 180
    except Exception as exc:
        try:
            previous = json.loads(OUT.read_text())
            for k in ('training','saved','history','last_success_at','source_time'):
                if k in previous:result[k]=previous[k]
        except (OSError,ValueError):
            pass
        result['error'] = type(exc).__name__ + ': training telemetry temporarily unavailable'
        result['stale'] = True
    try:
        with urllib.request.urlopen('http://127.0.0.1:18621/health', timeout=4) as response:
            health = json.load(response)
        model = health.get('model', {})
        result['serving'] = {'ready':bool(health.get('ready')), 'step':model.get('step'), 'checkpoint':health.get('ckpt'), 'revision':health.get('runtime_revision'), 'exported_at':model.get('exported_at'), 'val_nats':model.get('val_nats'), 'parameters':model.get('parameters'), 'parscale':health.get('parscale'), 'checked_at':time.time()}
    except Exception:
        result['serving'] = {'ready':False, 'error':'health unavailable', 'checked_at':time.time()}
    atomic_write(result)
    print(json.dumps({'ok':result['ok'],'stale':result['stale'],'training_step':result.get('training',{}).get('step'),'saved':result.get('saved'),'serving_step':result.get('serving',{}).get('step')}))

if __name__ == '__main__':
    main()
