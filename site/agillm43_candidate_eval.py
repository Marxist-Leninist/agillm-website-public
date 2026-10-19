#!/usr/bin/env python3
"""Off-public AGILLM4.3 checkpoint candidate evaluator.

Starts a temporary local bridge on 127.0.0.1:8494 using a temp bridge.env,
runs a small quality/speed smoke, writes JSON, and terminates the temp bridge.
It never edits the public bridge.env and never changes the pinned public model.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path('/opt/agillm43_netserve')
PUBLIC_ENV = ROOT / 'bridge.env'
BRIDGE = ROOT / 'agillm43_http_bridge.py'
OUT_DIR = ROOT / 'candidate_evals'
PORT = int(os.environ.get('AGILLM_CANDIDATE_PORT', '8494'))
BASE_URL = f'http://127.0.0.1:{PORT}'

QUALITY_CASES = [
    ('math', '2 + 2 =', ['4'], 8),
    ('fact', 'The capital of France is', ['paris'], 8),
    ('code', 'Python code to print hi:', ['print'], 16),
    ('language', 'A polite greeting is', ['hello', 'hi', 'good'], 12),
]

SCENARIOS = [
    ('fast nat', {
        'mode': 'nat', 'temperature': 0.25, 'top_k': 0, 'top_p': 1.0,
        'min_p': 0.0, 'greedy': True, 'ignore_eos': True, 'nat_passes': 1,
        'repetition_penalty': 2.0, 'presence_penalty': 0.8,
        'frequency_penalty': 1.2, 'penalty_last_n': 0,
    }),
    ('quality sat var', {
        'mode': 'sat', 'var': True, 'temperature': 0.65, 'top_k': 40,
        'top_p': 0.95, 'min_p': 0.01, 'greedy': False, 'ignore_eos': True,
        'nat_passes': 2, 'repetition_penalty': 2.0, 'presence_penalty': 0.6,
        'frequency_penalty': 1.0, 'penalty_last_n': 200,
    }),
    ('quality ar', {
        'mode': 'ar', 'temperature': 0.25, 'top_k': 0, 'top_p': 1.0,
        'min_p': 0.0, 'greedy': True, 'ignore_eos': True, 'nat_passes': 1,
        'repetition_penalty': 1.3, 'presence_penalty': 0.0,
        'frequency_penalty': 0.3, 'penalty_last_n': 128,
    }),
]

SPEED_SCENARIOS = [
    ('nat', {'mode': 'nat', 'repetition_penalty': 2.0, 'presence_penalty': 0.8, 'frequency_penalty': 1.2, 'penalty_last_n': 0}),
    ('sat fixed', {'mode': 'sat', 'var': False, 'repetition_penalty': 2.0, 'presence_penalty': 0.6, 'frequency_penalty': 1.0, 'penalty_last_n': 200}),
    ('sat var', {'mode': 'sat', 'var': True, 'repetition_penalty': 2.0, 'presence_penalty': 0.6, 'frequency_penalty': 1.0, 'penalty_last_n': 200}),
    ('ar', {'mode': 'ar', 'repetition_penalty': 1.3, 'presence_penalty': 0.0, 'frequency_penalty': 0.3, 'penalty_last_n': 128}),
]


def read_public_env() -> dict[str, str]:
    cfg = {}
    for raw in PUBLIC_ENV.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            cfg[k.strip()] = v.strip()
    return cfg


def request_json(path: str, payload: dict | None, token: str, timeout: float = 120.0) -> dict:
    headers = {'X-AGILLM-Token': token}
    data = None
    if payload is not None:
        headers['Content-Type'] = 'application/json'
        data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(BASE_URL + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode('utf-8', 'replace'))


def continuation(text: str, prompt: str) -> str:
    text = str(text or '').replace('\r', ' ').replace('\n', ' ')
    return text[len(prompt):].lstrip() if prompt and text.startswith(prompt) else text


def hit(text: str, needles: list[str]) -> bool:
    low = str(text or '').lower()
    return any(str(n or '').lower() in low for n in needles)


def median(values):
    return statistics.median([float(v) for v in values]) if values else 0.0


def wait_ready(token: str, max_s: float = 900.0) -> dict:
    deadline = time.time() + max_s
    last = None
    while time.time() < deadline:
        try:
            h = request_json('/health', None, token, timeout=3.0)
            last = h
            if h.get('ready'):
                return h
        except Exception as exc:
            last = {'error': str(exc)}
        time.sleep(5)
    raise TimeoutError(f'candidate bridge not ready: {last}')


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print('usage: agillm43_candidate_eval.py /path/to/checkpoint.pt', file=sys.stderr)
        return 2
    ckpt = Path(argv[1]).resolve()
    if not ckpt.exists():
        print(f'missing checkpoint: {ckpt}', file=sys.stderr)
        return 2
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = read_public_env()
    token = cfg.get('AGILLM_BRIDGE_TOKEN', '')
    run_id = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + ckpt.stem
    out_path = OUT_DIR / f'{run_id}.json'
    log_path = OUT_DIR / f'{run_id}.log'
    tmpdir = Path(tempfile.mkdtemp(prefix='agillm43_candidate_bridge_'))
    proc = None
    result = {'candidate': str(ckpt), 'started_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    try:
        shutil.copy2(BRIDGE, tmpdir / 'agillm43_http_bridge.py')
        (tmpdir / 'bridge.env').write_text('\n'.join([
            f'AGILLM_BRIDGE_TOKEN={token}',
            f'AGILLM_BRIDGE_CKPT={ckpt}',
            'AGILLM_BRIDGE_RUNTIME=' + cfg.get('AGILLM_BRIDGE_RUNTIME', '/workspace/agillm43_infer_demo/agillm41_current.py'),
            'AGILLM_BRIDGE_DTYPE=' + cfg.get('AGILLM_BRIDGE_DTYPE', 'fp32'),
            'AGILLM_BRIDGE_THREADS=' + cfg.get('AGILLM_BRIDGE_THREADS', '16'),
            'AGILLM_BRIDGE_INTEROP=' + cfg.get('AGILLM_BRIDGE_INTEROP', '0'),
            'AGILLM_BRIDGE_CONCURRENCY=' + cfg.get('AGILLM_BRIDGE_CONCURRENCY', '1'),
            'AGILLM_BRIDGE_OVERFLOW_THREADS=' + cfg.get('AGILLM_BRIDGE_OVERFLOW_THREADS', '4'),
            'TOKENIZER_DIR=' + cfg.get('TOKENIZER_DIR', '/workspace/tokenizers/deepseek-v4-pro'),
            'TOKENIZER_ID=' + cfg.get('TOKENIZER_ID', 'deepseek-ai/DeepSeek-V4-Pro'),
            'HF_HUB_OFFLINE=' + cfg.get('HF_HUB_OFFLINE', '1'),
            'TRANSFORMERS_OFFLINE=' + cfg.get('TRANSFORMERS_OFFLINE', '1'),
            'AGILLM_BRIDGE_REQUIRE_PIN=0',
            'AGILLM_KV_BUFFER=' + cfg.get('AGILLM_KV_BUFFER', '0'),
            'LD_PRELOAD=' + cfg.get('LD_PRELOAD', ''),
            '',
        ]))
        env = os.environ.copy()
        env.update({'AGILLM_BRIDGE_PORT': str(PORT)})
        if cfg.get('LD_PRELOAD'):
            env['LD_PRELOAD'] = cfg['LD_PRELOAD']
        log = open(log_path, 'w')
        proc = subprocess.Popen(
            ['python3', '-u', str(tmpdir / 'agillm43_http_bridge.py')],
            cwd=str(tmpdir), env=env, stdout=log, stderr=subprocess.STDOUT,
            text=True,
        )
        result['bridge_pid'] = proc.pid
        result['health'] = wait_ready(token)
        quality = {}
        for label, base in SCENARIOS:
            rows, rates, buttons, hits = [], [], [], 0
            for case_name, prompt, needles, max_new in QUALITY_CASES:
                req = {**base, 'prompt': prompt, 'max_new': max_new, 'min_new': 0, 'stream': False}
                t0 = time.time()
                out = request_json('/infer', req, token, timeout=180.0)
                wall = max(0.001, time.time() - t0)
                stats = out.get('stats') or {}
                text = continuation(out.get('text') or '', prompt)
                ok = hit(text, needles)
                hits += int(ok)
                tok = int(stats.get('tokens') or 0)
                gen = float(stats.get('seconds') or 0.0)
                gen_rate = float(stats.get('tok_s') or ((tok / gen) if tok and gen else 0.0))
                button = (tok / wall) if tok else 0.0
                rates.append(gen_rate); buttons.append(button)
                rows.append({'case': case_name, 'hit': ok, 'tokens': tok, 'gen_tok_s': gen_rate, 'button_tok_s': button, 'continuation': text[:240]})
            quality[label] = {'hits': hits, 'total': len(QUALITY_CASES), 'median_gen_tok_s': median(rates), 'median_button_tok_s': median(buttons), 'cases': rows}
        result['quality'] = quality
        speed = {}
        base_speed = {
            'prompt': 'The quick brown fox jumps over the lazy dog and then',
            'max_new': 32, 'min_new': 0, 'stream': False, 'temperature': 0.25,
            'top_k': 0, 'top_p': 1.0, 'min_p': 0.0, 'greedy': True,
            'ignore_eos': True, 'nat_passes': 1,
        }
        for label, extra in SPEED_SCENARIOS:
            rates, buttons = [], []
            for _ in range(2):
                req = {**base_speed, **extra}
                t0 = time.time()
                out = request_json('/infer', req, token, timeout=180.0)
                wall = max(0.001, time.time() - t0)
                stats = out.get('stats') or {}
                tok = int(stats.get('tokens') or 0)
                gen = float(stats.get('seconds') or 0.0)
                rates.append(float(stats.get('tok_s') or ((tok / gen) if tok and gen else 0.0)))
                buttons.append((tok / wall) if tok else 0.0)
            speed[label] = {'median_gen_tok_s': median(rates), 'median_button_tok_s': median(buttons)}
        result['speed'] = speed
        result['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        print(out_path)
        return 0
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        result['finished_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
        print(out_path)
        return 1
    finally:
        if proc and proc.poll() is None:
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
