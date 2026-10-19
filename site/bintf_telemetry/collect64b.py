#!/usr/bin/env python3
"""Read-only Ternary 64B (terntf) training telemetry for the agillm-ui fleet card, with atomic last-good publication.

One short ssh per run to the 64B training box (status/stat only; nothing is written or signalled there).
Field meanings (trainer-reported, from /workspace/tern64b_run/status.json):
  training.loss           latest single-step next-token training loss (batch of 2048 ids); NOT a held-out score
  training.ema            trainer loss EMA
  training.fresh_tokens_seen  status tokens_seen (sampled input ids this resume segment lineage)
  training.fresh_tokens_per_second  delta of tokens_seen over wall time across recent observations
  serving                 always not-serving: this run has no inference worker; the card is training/checkpoint only
"""
import fcntl, json, math, os, shlex, subprocess, time
from pathlib import Path

ROOT = Path('/opt/agillm43_netserve')
OUT = Path(os.getenv('TERN64B_TELEMETRY_OUT', str(ROOT / 'tern64b_training_status.json')))
LOCK = Path(os.getenv('TERN64B_TELEMETRY_LOCK', str(ROOT / 'bintf_telemetry/collector64b.lock')))
KNOWN_HOSTS = os.getenv('TERN64B_KNOWN_HOSTS', str(ROOT / 'bintf_telemetry/known_hosts64b'))
SAMPLE_LINK = ROOT / 'tern64b_sample_link.json'
# Vast instance 54869075 (RTX 3060). The proxy host/port change if the instance is recreated: edit these two lines then.
SSH_HOST, SSH_PORT = os.getenv('TERN64B_SSH_HOST', 'ssh5.vast.ai'), os.getenv('TERN64B_SSH_PORT', '32150')
SSH_KEY = os.getenv('TERN64B_SSH_KEY', '/root/.ssh/id_ed25519')
WINDOW_S, STALE_S = 900, 300

REMOTE = r'''import glob,json,os,subprocess,time
from pathlib import Path
r=Path('/workspace/tern64b_run')
for attempt in range(3):
 try:
  raw=(r/'status.json').read_text();s=json.loads(raw);break
 except (ValueError,OSError):
  if attempt==2:raise
  time.sleep(.25)
t={k:s.get(k) for k in ('step','loss','ema','flips','flips_pct','gov','scale','skipped','s_per_step','gpu_gb','tokens_seen','last_save_step','code_sha256','resilience','data','paused','fine')}
t['time']=(r/'status.json').stat().st_mtime
try:
 sj=json.loads((r/'state.json').read_text()); sa=sj.get('args') or {}; sc=sj.get('cli') or {}
 t['weights']=sa.get('weights'); t['table_mode']=sa.get('table'); t['latent_bits']=sa.get('latent_bits'); t['hf_repo']=sc.get('hf_repo')
except (OSError,ValueError,AttributeError):pass
alive=False
try:
 pid=int(s.get('pid'))
 cmd=Path('/proc/%d/cmdline'%pid).read_bytes()
 alive=b'terntf.py' in cmd and b'/workspace/tern64b_run' in cmd
except (OSError,ValueError,TypeError):pass
t['process_alive']=alive
saved=None
try:
 ck=r/'ckpt.pt'; st=ck.stat()
 tabs=sorted(glob.glob(str(r/'table_*.i8')))
 tab=tabs[-1] if tabs else None
 saved={'step':s.get('last_save_step'),'sha256':None,'bytes':st.st_size,'updated_at':st.st_mtime,
        'table_file':os.path.basename(tab) if tab else None,
        'table_bytes':os.path.getsize(tab) if tab else None,
        'latents_bytes':(r/'latents.i8').stat().st_size}
except OSError:pass
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


def rate(window, entry):
    if window and entry[1] < window[-1][1]:
        window = []
    window = [w for w in window if entry[0] - w[0] <= WINDOW_S] + [entry]
    first = window[0]
    dt = entry[0] - first[0]
    if dt < 60:
        return window[-60:], None
    return window[-60:], (entry[1] - first[1]) / dt


def collect():
    now = time.time()
    result = {'schema': 'agillm.tern64b.telemetry.v1', 'model': 'tern64b', 'attempted_at': now,
              'interval_s': 30, 'stale_after_s': STALE_S, 'ok': False,
              'serving': {'ready': False, 'status': 'not serving',
                          'note': 'Training/checkpoint only; no inference worker for the 64B run yet.'},
              'artifact': {'hf_repo': 'MarxistLeninist/AGILLM-64B-tern',
                           'offbox_backup': 'no coherent off-box backup of the current binary (bin8, since step 311) run yet; needs a PAUSE-window export',
                           'onbox_run': 'vast:54869075:/workspace/tern64b_run'}}
    try:
        result['sample'] = json.loads(SAMPLE_LINK.read_text())
    except (OSError, ValueError):
        result['sample'] = None
    try:
        prev = json.loads(OUT.read_text())
    except (OSError, ValueError):
        prev = {}
    try:
        data = read_trainer(); received = time.time(); t = data['training']
        t['fresh_tokens_seen'] = t.get('tokens_seen')
        window = prev.get('rate_window') if isinstance(prev.get('rate_window'), list) else []
        if number(t.get('tokens_seen')):
            window, t['fresh_tokens_per_second'] = rate(window, [data['source_time'], t['tokens_seen']])
        else:
            t['fresh_tokens_per_second'] = None
        t['rate_window_seconds'] = round(window[-1][0] - window[0][0], 1) if len(window) > 1 else 0
        t['loss_definition'] = 'latest single-step next-token training loss on a 2048-id batch (ema = trainer EMA); not a held-out score'
        if isinstance(t.get('data'), str):
            t['data'] = t['data'][:160]
        result.update(data)
        result['artifact']['hf_repo'] = t.get('hf_repo') or 'none configured (trainer HF uploads off; old ternary snapshots only in MarxistLeninist/AGILLM-64B-tern)'
        result.update(ok=True, rate_window=window, last_success_at=received,
                      training_age_s=max(0, data['source_time'] - t['time']))
        result['stale'] = result['training_age_s'] > STALE_S
        result['source'] = {'vast_instance_id': 54869075, 'clock_offset_s': data['source_time'] - received}
    except Exception as exc:
        for k in ('training', 'saved', 'rate_window', 'last_success_at', 'source_time', 'source'):
            if k in prev:
                result[k] = prev[k]
        result.update(stale=True, error='training telemetry temporarily unavailable', error_kind=type(exc).__name__)
        last, src, stamp = result.get('last_success_at'), result.get('source_time'), (result.get('training') or {}).get('time')
        result['training_age_s'] = max(0, src - stamp) + max(0, now - last) if all(number(x) for x in (src, stamp, last)) else None
    result['published_at'] = time.time()
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
