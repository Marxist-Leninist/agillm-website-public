#!/usr/bin/env python3
"""Read-only Binary 32B training telemetry for the agillm-ui fleet card, with atomic last-good publication.

One short ssh per run to the 32B training box (cat/stat/tail only; nothing is written or signalled there).
The 32B "native" trainer differs from the 4B/8B/16B one, so the feed says what each figure is:
  training.loss                 mean next-token ("ar" mode) training loss over the most recent ar steps in train.log.
                                A training-batch figure on batches the trainer reuses across sub-layers: NOT a held-out score.
  training.fresh_tokens_seen    sampled input-token occurrences; can repeat corpus tokens, not unique coverage
  training.objective_tokens_seen  token-losses computed, counting reuse of the same batch (not comparable)
  training.fresh_tokens_per_second  delta of fresh tokens over wall time across the last ~10 minutes of observations
  training.heldout_*            only if someone publishes /workspace/agillm32b_run/heldout_latest.json
                                ({"step":..,"ar_nats":..,"copy_rate":..,"at":..,"protocol":".."}); otherwise null
"""
import fcntl, json, math, os, shlex, subprocess, time, urllib.request
from pathlib import Path
from telemetry_metadata import enrich_metadata

ROOT = Path('/opt/agillm43_netserve')
OUT = Path(os.getenv('BINTF32B_TELEMETRY_OUT', str(ROOT / 'bintf32b_training_status.json')))
LOCK = Path(os.getenv('BINTF32B_TELEMETRY_LOCK', str(ROOT / 'bintf_telemetry/collector32b.lock')))
KNOWN_HOSTS = os.getenv('BINTF32B_KNOWN_HOSTS', str(ROOT / 'bintf_telemetry/known_hosts32b'))
# Vast instance 55575647. The proxy host/port change if the instance is recreated: edit these two lines then.
SSH_HOST, SSH_PORT = os.getenv('BINTF32B_SSH_HOST', 'ssh9.vast.ai'), os.getenv('BINTF32B_SSH_PORT', '24734')
SSH_KEY = os.getenv('BINTF32B_SSH_KEY', '/root/.ssh/id_ed25519')
WINDOW_S, STALE_S = 600, 180

REMOTE = r'''import json,os,subprocess,time
from pathlib import Path
r=Path('/workspace/agillm32b_run')
for attempt in range(3):
 try:
  raw=(r/'status.json').read_text();s=json.loads(raw);break
 except (ValueError,OSError):
  if attempt==2:raise
  time.sleep(.25)
t={k:s.get(k) for k in ('step','mode','loss','fresh_tokens_seen','objective_tokens_seen','objective_tokens_per_second','objective_version','train_path','active_sublayer','group','head_gain','sequence','batch','runtime_revision','stride','nat_mask_kind','diffusion_path','curriculum_stage','stage_completed_batches','stage_target_batches','planned_sequences')}
t['time']=(r/'status.json').stat().st_mtime
alive=False
try:
 pid=int((r/'trainer.pid').read_text().strip())
 cmd=Path('/proc/%d/cmdline'%pid).read_bytes()
 alive=b'agillm32b' in cmd and b'/workspace/agillm32b_run' in cmd
except (OSError,ValueError):pass
t['process_alive']=alive
modes={};ar=[]
try:
 with (r/'train.log').open('rb') as f:
  size=f.seek(0,2);f.seek(max(0,size-393216));lines=f.read().decode('utf-8','replace').splitlines()
  if size>393216:lines=lines[1:]
 for line in lines[-500:]:
  try:d=json.loads(line)
  except ValueError:continue
  if d.get('event')!='native_step' or not isinstance(d.get('loss'),(int,float)):continue
  modes[str(d.get('mode'))]={'loss':d['loss'],'step':d.get('step')}
  if d.get('mode')=='ar':ar.append(d['loss'])
except OSError:pass
t['loss_latest']=t.get('loss');t['loss_by_mode']=modes
t['loss']=sum(ar[-60:])/len(ar[-60:]) if ar else None
t['loss_ar_steps']=len(ar[-60:])
saved=None
try:
 l=json.loads((r/'latest.json').read_text());p=Path(l['path'])
 saved={'step':l.get('step'),'sha256':l.get('sha256'),'bytes':l.get('bytes'),'updated_at':p.stat().st_mtime}
except (OSError,ValueError,KeyError,TypeError):pass
try:
 h=json.loads((r/'heldout_latest.json').read_text())
 t['heldout_nats']=h.get('ar_nats');t['heldout_step']=h.get('step');t['heldout_copy_rate']=h.get('copy_rate');t['heldout_at']=h.get('at');t['heldout_protocol']=str(h.get('protocol',''))[:80]
except (OSError,ValueError,AttributeError):pass
try:
 g=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=6).stdout.strip().split(',')
 t['gpu_gb']=float(g[0])/1024;t['gpu_util_pct']=float(g[1])
except Exception:pass
print(json.dumps({'training':t,'saved':{'ckpt':saved},'source_time':time.time()}))
'''


def clean(v):
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [clean(x) for x in v]
    return v


def number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def atomic_write(value):
    tmp = OUT.with_name(OUT.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(clean(value), f, allow_nan=False, separators=(',', ':'))
        f.flush(); os.fsync(f.fileno()); os.fchmod(f.fileno(), 0o644)
    os.replace(tmp, OUT)


def read_trainer():
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'UserKnownHostsFile=' + KNOWN_HOSTS,
               '-o', 'ConnectTimeout=6', '-o', 'ConnectionAttempts=1', '-o', 'ServerAliveInterval=5', '-o', 'ServerAliveCountMax=1',
               '-o', 'LogLevel=ERROR', '-i', SSH_KEY, '-p', SSH_PORT, 'root@' + SSH_HOST, 'python3 -c ' + shlex.quote(REMOTE)]
    proc = subprocess.run(command, capture_output=True, text=True, timeout=20, check=True)
    data = json.loads(next(x for x in reversed(proc.stdout.splitlines()) if x.startswith('{')))
    t = data.get('training', {})
    if not all(number(x) for x in (data.get('source_time'), t.get('time'), t.get('step'))):
        raise ValueError('invalid telemetry')
    return data


def rates(window, now_entry):
    """window: [[source_time, fresh, objective], ...] oldest first. Counters going backwards (trainer restart from an
    older checkpoint) start a new window."""
    if window and (now_entry[1] < window[-1][1] or now_entry[2] < window[-1][2]):
        window = []
    window = [w for w in window if now_entry[0] - w[0] <= WINDOW_S] + [now_entry]
    first = window[0]
    dt = now_entry[0] - first[0]
    if dt < 45:
        return window[-60:], None, None
    return window[-60:], (now_entry[1] - first[1]) / dt, (now_entry[2] - first[2]) / dt


def collect():
    now = time.time()
    result = {'schema': 'agillm.bintf.telemetry.v1', 'model': 'bintf32b', 'attempted_at': now, 'interval_s': 30, 'stale_after_s': STALE_S, 'ok': False}
    try:
        prev = json.loads(OUT.read_text())
    except (OSError, ValueError):
        prev = {}
    try:
        data = read_trainer(); received = time.time(); t = data['training']
        window = prev.get('rate_window') if isinstance(prev.get('rate_window'), list) else []
        fresh, objective = t.get('fresh_tokens_seen'), t.get('objective_tokens_seen')
        if number(fresh) and number(objective):
            window, t['fresh_tokens_per_second'], t['objective_tokens_per_second_window'] = rates(window, [data['source_time'], fresh, objective])
        else:
            t['fresh_tokens_per_second'] = None
        t['rate_window_seconds'] = round(window[-1][0] - window[0][0], 1) if len(window) > 1 else 0
        t['loss_definition'] = 'mean next-token (ar mode) training loss over the last %s ar steps; training batches are reused across sub-layers, so this is not a held-out score' % t.get('loss_ar_steps')
        result.update(data)
        result.update(ok=True, rate_window=window, last_success_at=received, training_age_s=max(0, data['source_time'] - t['time']))
        result['stale'] = result['training_age_s'] > STALE_S
        result['source'] = {'vast_instance_id': 55575647, 'clock_offset_s': data['source_time'] - received}
    except Exception as exc:
        for k in ('training', 'saved', 'rate_window', 'last_success_at', 'source_time', 'source'):
            if k in prev:
                result[k] = prev[k]
        result.update(stale=True, error='training telemetry temporarily unavailable', error_kind=type(exc).__name__)
        last, src, stamp = result.get('last_success_at'), result.get('source_time'), (result.get('training') or {}).get('time')
        result['training_age_s'] = max(0, src - stamp) + max(0, now - last) if all(number(x) for x in (src, stamp, last)) else None
    try:
        with urllib.request.urlopen('https://codex.opentransformers.online/agillm-bintf32b-health', timeout=3) as response:
            health = json.load(response)
        result['serving'] = {k: health.get(k) for k in ('ready', 'runtime_revision', 'ckpt', 'max_prompt_tokens', 'max_new_tokens', 'max_context_tokens', 'canonical_modes', 'context_policy')}
        result['serving']['step'] = (health.get('model') or {}).get('step')
        result['serving']['status'] = 'ready' if health.get('ready') else 'warming'
    except Exception:
        result['serving'] = {'ready': False, 'status': 'health unavailable'}
    result['published_at'] = time.time()
    enrich_metadata(result, result['published_at'])
    atomic_write(result)
    print(json.dumps({k: result.get(k) for k in ('ok', 'stale', 'training_age_s', 'last_success_at', 'error_kind')}))
    return result


def main():
    with LOCK.open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        collect()


if __name__ == '__main__':
    main()
