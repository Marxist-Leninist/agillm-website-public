#!/usr/bin/env python3
"""Read-only AGILLM Binary 8B telemetry. Never changes trainer or checkpoint state.

Fork of collect.py (Binary 4B). Same output schema, "agillm.bintf.telemetry.v1", with
"model":"bintf8b", written atomically (mode 0644) to status8b.json every 30 s by
agillm-bintf8b-telemetry.timer.

The trainer box is only ever read: status.json, ckpt.good/state.json, ckpt/state.json and
the tail of metrics.jsonl, through one tiny python one-liner over SSH. The primary SSH
endpoint and a fallback come from /opt/bintf8b_serve/ops/telemetry.json. Serving health is
read from the local 8B CPU worker; when that worker is not up the collector still succeeds
and reports serving.ready=false.
"""
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import urllib.request

ROOT = Path('/opt/agillm43_netserve/bintf_telemetry')
# The four overrides exist only so the collector can be exercised against a scratch
# directory; the systemd unit sets none of them.
OUT = Path(os.environ.get('BINTF8B_TELEMETRY_OUT') or ROOT / 'status8b.json')
LOCK = Path(os.environ.get('BINTF8B_TELEMETRY_LOCK') or ROOT / 'collector8b.lock')
CONFIG = Path(os.environ.get('BINTF8B_TELEMETRY_CONFIG') or '/opt/bintf8b_serve/ops/telemetry.json')
HEALTH_URL = os.environ.get('BINTF8B_TELEMETRY_HEALTH') or 'http://127.0.0.1:18622/health'

# Identical to the Binary 4B collector: these are the keys the website parses.
FIELDS = ('step', 'net', 'net_step', 'seen', 'ar', 'masked', 'rate', 'rate_avg', 'gpu_gb', 'time', 'steps', 'target',
          'last_save', 'started', 'skipped', 'flips_pct', 'ps')
# Extras the 8B trainer's status.json carries. Additive only; absent keys become null.
EXTRA = ('ids', 'lr_mult', 'scale', 'head_bad', 'rate_e2e', 'rate_e2e_tokens', 'rate_e2e_seconds', 'eta_days',
         'unit', 'loss_unit')
EVAL_KEYS = ('ar', 'ar_tail', 'nat16', 'step', 'seen', 'ps', 'together', 'ff_zero', 'long')
STALE_AFTER_S = 180
SSH_BUDGET_S = 21.0      # both endpoints together; the unit's TimeoutStartSec is 30
SSH_ATTEMPT_S = 10.0

REMOTE = r'''import json,time
from pathlib import Path
r=Path(@RUN@)
s=None
for _ in (0,1,2):
 try:
  s=json.loads((r/'status.json').read_text());break
 except ValueError:time.sleep(0.25)
t={k:s.get(k) for k in @FIELDS@}
pid=s.get('pid');t['process_alive']=bool(pid and Path('/proc/'+str(int(pid))).exists())
t['eval']={str(k):{q:v.get(q) for q in @EVALQ@} for k,v in s.get('eval',{}).items() if isinstance(v,dict)}
m=s.get('model');t['model_desc']=m[:600] if isinstance(m,str) else None
saved={}
for name in ['ckpt.good','ckpt']:
 p=r/name/'state.json'
 try:
  d=json.loads(p.read_text());saved[name]={'step':sum(d.get('steps',[])),'seen':d.get('seen'),'updated_at':p.stat().st_mtime}
 except (OSError,ValueError,TypeError,AttributeError):saved[name]=None
history=[]
try:
 with (r/'metrics.jsonl').open('rb') as f:
  size=f.seek(0,2);f.seek(max(0,size-196608));lines=f.read().decode('utf-8','replace').splitlines()
  if size>196608:lines=lines[1:]
 for line in lines[-160:]:
  try:
   d=json.loads(line);history.append({k:d.get(k) for k in ['step','time','net','ar','masked','rate']})
  except (ValueError,AttributeError):pass
except OSError:pass
print(json.dumps({'training':t,'saved':saved,'history':history,'source_time':time.time()}))
'''


def clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    return v


def atomic_write(value):
    text = json.dumps(clean(value), separators=(',', ':'), allow_nan=False)
    tmp = OUT.with_name(OUT.name + '.tmp')
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    with os.fdopen(fd, 'w') as handle:
        handle.write(text)
        handle.flush()
        os.fchmod(handle.fileno(), 0o644)
    os.replace(tmp, OUT)


def endpoints(conf):
    """Primary first, then the fallback; malformed entries are skipped."""
    found = []
    for label, host, port in (('primary', conf.get('vast_host'), conf.get('vast_port')),
                              ('fallback', conf.get('fallback_host'), conf.get('fallback_port'))):
        try:
            port = int(port)
        except (TypeError, ValueError):
            continue
        if isinstance(host, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.-]*', host) and 0 < port < 65536:
            found.append((label, host, port))
    return found


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def read_trainer(conf):
    run = conf.get('run', '/workspace/agillm8b_run')
    remote = (REMOTE.replace('@RUN@', repr(run)).replace('@FIELDS@', repr(FIELDS + EXTRA))
              .replace('@EVALQ@', repr(list(EVAL_KEYS))))
    command = 'python3 -c ' + shlex.quote(remote)
    key = conf.get('key', '/root/.ssh/id_ed25519')
    started = time.monotonic()
    for label, host, port in endpoints(conf):
        left = SSH_BUDGET_S - (time.monotonic() - started)
        if left < 3:
            break
        ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=6', '-o', 'ConnectionAttempts=1',
               '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=1', '-i', key, '-p', str(port), 'root@' + host]
        try:
            proc = subprocess.run(ssh + [command], capture_output=True, text=True, timeout=min(SSH_ATTEMPT_S, left))
        except subprocess.TimeoutExpired:
            continue
        if proc.returncode:
            continue
        payload = next((x for x in reversed(proc.stdout.splitlines()) if x.startswith('{')), '')
        try:
            data = json.loads(payload)
        except ValueError:
            continue
        training = data.get('training') if isinstance(data, dict) else None
        if isinstance(training, dict) and number(training.get('time')) and number(training.get('step')) \
                and number(data.get('source_time')):
            return data, label
    raise RuntimeError('trainer_ssh_failed')


def read_serving():
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(HEALTH_URL, timeout=3) as response:
            health = json.load(response)
        model = health.get('model')
        if not isinstance(model, dict):
            model = {}
        return {'ready': bool(health.get('ready')), 'step': model.get('step'), 'checkpoint': health.get('ckpt'),
                'revision': health.get('runtime_revision'), 'exported_at': model.get('exported_at'),
                'val_nats': model.get('val_nats'), 'parameters': model.get('parameters'),
                'parscale': health.get('parscale'), 'warming': bool(health.get('warming')), 'checked_at': time.time()}
    except Exception:
        return {'ready': False, 'error': 'health unavailable', 'checked_at': time.time()}


def main():
    lock = LOCK.open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return
    now = time.time()
    result = {'schema': 'agillm.bintf.telemetry.v1', 'model': 'bintf8b', 'attempted_at': now, 'interval_s': 30, 'ok': False}
    try:
        conf = json.loads(CONFIG.read_text())
        data, label = read_trainer(conf)
        received = time.time()
        result.update({k: data.get(k) for k in ('training', 'saved', 'history', 'source_time')})
        result['ok'] = True
        result['last_success_at'] = received
        result['training_age_s'] = max(0, data['source_time'] - float(data['training']['time']))
        result['stale'] = result['training_age_s'] > STALE_AFTER_S
        # Trainer-box clock minus GETH clock (includes the SSH round trip). The website ages
        # training.time against the visitor's clock, so a large offset here would matter.
        result['source'] = {'vast_instance_id': conf.get('vast_instance_id'), 'endpoint': label, 'run': conf.get('run'),
                            'clock_offset_s': data['source_time'] - received}
    except Exception as exc:
        try:
            previous = json.loads(OUT.read_text())
            for k in ('training', 'saved', 'history', 'last_success_at', 'source_time', 'source'):
                if k in previous:
                    result[k] = previous[k]
        except (OSError, ValueError):
            pass
        result['error'] = type(exc).__name__ + ': training telemetry temporarily unavailable'
        result['stale'] = True
    result['serving'] = read_serving()
    atomic_write(result)
    training = result.get('training') or {}
    print(json.dumps({'ok': result['ok'], 'stale': result['stale'], 'training_step': training.get('step'),
                      'saved': result.get('saved'), 'serving_ready': result['serving'].get('ready'),
                      'serving_step': result['serving'].get('step'),
                      'endpoint': (result.get('source') or {}).get('endpoint')}))


if __name__ == '__main__':
    main()
