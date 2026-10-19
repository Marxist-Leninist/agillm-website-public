#!/usr/bin/env python3
"""Promote complete AGILLM4.3 full checkpoints into the live FP32 pool.

The watcher requires a structurally complete, stable candidate. Promotion is
blue-green through SIGHUP. The old pool remains available if a replacement
cannot warm, and bridge.env is rolled back when promotion verification fails.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pathlib
import re
import signal
import sys
import time
import urllib.request
from dataclasses import dataclass
from typing import Any

ROOT = pathlib.Path('/workspace/agillm43_repretrain_v4_ckpts').resolve()
LIVE = pathlib.Path('/workspace/agillm43_live_infer').resolve()
DEFAULT_LATEST = ROOT / 'latest.json'
DEFAULT_ENV = LIVE / 'bridge.env'
BRIDGE_PID = LIVE / 'bridge.pid'
LOCK = LIVE / 'watch_latest.lock'
STATE = LIVE / 'watch_latest.state.json'
HEALTH_URL = 'http://127.0.0.1:8493/health'
STEP_RE = re.compile(r'^pretrain_step0*(\d+)(?:_.*)?[.]pt$')


@dataclass(frozen=True)
class Candidate:
    path: pathlib.Path
    step: int
    signature: str
    age_s: float
    size: int
    shard_count: int
    shard_bytes: int


def emit(event: str, **fields: Any) -> None:
    row = {'ts': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'event': event}
    row.update(fields)
    print(json.dumps(row, sort_keys=True, separators=(',', ':')), flush=True)


def read_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f'{path.name} is not a JSON object')
    return value


def inside_root(path: pathlib.Path) -> pathlib.Path:
    resolved = path.resolve(strict=True)
    resolved.relative_to(ROOT)
    return resolved


def regular_file(path: pathlib.Path, min_bytes: int = 1) -> os.stat_result:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'missing or unsafe file: {path.name}')
    st = path.stat()
    if st.st_size < min_bytes:
        raise ValueError(f'undersized file: {path.name} ({st.st_size} bytes)')
    return st


def inspect_candidate(latest_file: pathlib.Path) -> Candidate:
    latest = read_json(latest_file)
    raw_path = str(latest.get('path') or '')
    path = inside_root(pathlib.Path(raw_path))
    match = STEP_RE.match(path.name)
    if not match:
        raise ValueError(f'unexpected checkpoint name: {path.name}')
    filename_step = int(match.group(1))
    latest_step = int(latest.get('step') or 0)
    if latest_step != filename_step or latest_step <= 0:
        raise ValueError(f'latest step mismatch: metadata={latest_step} filename={filename_step}')

    ckpt_st = regular_file(path, 100_000_000)
    tokenizer = pathlib.Path(str(path) + '.tokenizer.json')
    tok_st = regular_file(tokenizer, 1_000_000)
    provenance = path.with_suffix('.provenance.json')
    prov_st = regular_file(provenance, 256)
    prov = read_json(provenance)
    if prov.get('checkpoint_type') != 'full':
        raise ValueError(f'not a full checkpoint: {prov.get("checkpoint_type")!r}')
    if int(prov.get('step') or 0) != latest_step:
        raise ValueError('provenance step does not match latest.json')

    shards = pathlib.Path(str(path) + '.shards')
    if shards.is_symlink() or not shards.is_dir():
        raise ValueError(f'missing or unsafe shard directory: {shards.name}')
    shard_files = sorted(q for q in shards.rglob('*') if q.is_file() and not q.is_symlink())
    if len(shard_files) < 4:
        raise ValueError(f'incomplete shard set: {len(shard_files)} file(s)')
    shard_stats = [q.stat() for q in shard_files]
    shard_bytes = sum(st.st_size for st in shard_stats)
    if shard_bytes < 100_000_000:
        raise ValueError(f'undersized shard set: {shard_bytes} bytes')

    latest_st = regular_file(latest_file, 64)
    newest_mtime = max(
        [latest_st.st_mtime, ckpt_st.st_mtime, tok_st.st_mtime, prov_st.st_mtime]
        + [st.st_mtime for st in shard_stats]
    )
    age_s = max(0.0, time.time() - newest_mtime)
    sig_payload = {
        'path': str(path), 'step': latest_step,
        'ckpt': [ckpt_st.st_size, ckpt_st.st_mtime_ns],
        'tokenizer': [tok_st.st_size, tok_st.st_mtime_ns],
        'provenance': [prov_st.st_size, prov_st.st_mtime_ns],
        'shards': [len(shard_files), shard_bytes, max(st.st_mtime_ns for st in shard_stats)],
    }
    signature = hashlib.sha256(json.dumps(sig_payload, sort_keys=True).encode()).hexdigest()
    return Candidate(path, latest_step, signature, age_s, ckpt_st.st_size,
                     len(shard_files), shard_bytes)


def env_values(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in text.splitlines():
        if not raw or raw.lstrip().startswith('#') or '=' not in raw:
            continue
        key, value = raw.split('=', 1)
        out[key.strip()] = value
    return out


def replace_env(text: str, updates: dict[str, str]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        if raw and not raw.lstrip().startswith('#') and '=' in raw:
            key = raw.split('=', 1)[0].strip()
            if key in updates:
                out.append(f'{key}={updates[key]}')
                seen.add(key)
                continue
        out.append(raw)
    for key, value in updates.items():
        if key not in seen:
            out.append(f'{key}={value}')
    return '\n'.join(out).rstrip() + '\n'


def atomic_write(path: pathlib.Path, text: str, mode: int = 0o600) -> None:
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    temp.write_text(text)
    temp.chmod(mode)
    temp.replace(path)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def bridge_pid() -> int:
    pid = int(BRIDGE_PID.read_text().strip())
    cmdline = pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode('utf-8', 'replace')
    if 'agillm43_http_bridge.py' not in cmdline:
        raise RuntimeError(f'bridge pid {pid} does not identify the live bridge')
    return pid


def health(timeout: float = 3.0) -> dict[str, Any]:
    with urllib.request.urlopen(HEALTH_URL, timeout=timeout) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError('bridge health is not an object')
    return value


def wait_for_candidate(name: str, timeout_s: float = 240.0) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    last: dict[str, Any] = {}
    while time.time() < deadline:
        try:
            last = health()
            desired = int(last.get('concurrency') or 1)
            if (last.get('ready') is True and last.get('warming') is False
                    and last.get('ckpt') == name
                    and int(last.get('live_slots') or 0) >= desired):
                return last
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f'bridge did not promote {name}; last health={last}')


def promote(candidate: Candidate, env_file: pathlib.Path) -> dict[str, Any]:
    original = env_file.read_text()
    current = env_values(original)
    if pathlib.Path(current.get('AGILLM_BRIDGE_CKPT', '')).name == candidate.path.name:
        return {'changed': False, 'reason': 'already_configured'}

    digest = sha256_file(candidate.path)
    updated = replace_env(original, {
        'AGILLM_BRIDGE_CKPT': str(candidate.path),
        'AGILLM_BRIDGE_PINNED_CKPT': str(candidate.path),
        'AGILLM_BRIDGE_PINNED_SHA256': digest,
        'AGILLM_BRIDGE_REQUIRE_PIN': '1',
    })
    atomic_write(env_file, updated)
    pid = bridge_pid()
    before = {}
    try:
        before = health()
    except Exception:
        pass
    try:
        os.kill(pid, signal.SIGHUP)
        after = wait_for_candidate(candidate.path.name)
        return {
            'changed': True, 'step': candidate.step, 'checkpoint': candidate.path.name,
            'loads_before': before.get('loads'), 'loads_after': after.get('loads'),
            'live_slots': after.get('live_slots'), 'host': after.get('host'),
        }
    except Exception:
        atomic_write(env_file, original)
        try:
            os.kill(bridge_pid(), signal.SIGHUP)
        except Exception:
            pass
        raise


def run(args: argparse.Namespace) -> int:
    latest_file = pathlib.Path(args.latest_file).resolve()
    env_file = pathlib.Path(args.env_file).resolve()
    stable_sig = ''
    stable_count = 0
    last_rejection = ''

    with LOCK.open('a+') as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            emit('watcher_already_running')
            return 0

        emit('watcher_started', poll_s=args.poll, min_age_s=args.min_age,
             confirmations=args.confirmations)
        while True:
            try:
                candidate = inspect_candidate(latest_file)
                if candidate.signature == stable_sig:
                    stable_count += 1
                else:
                    stable_sig = candidate.signature
                    stable_count = 1
                configured = pathlib.Path(env_values(env_file.read_text()).get('AGILLM_BRIDGE_CKPT', '')).name
                eligible = candidate.age_s >= args.min_age and stable_count >= args.confirmations
                if configured == candidate.path.name and not args.force:
                    emit('candidate_current', step=candidate.step, checkpoint=candidate.path.name,
                         age_s=round(candidate.age_s, 1), stable_count=stable_count)
                elif eligible:
                    emit('promotion_started', step=candidate.step, checkpoint=candidate.path.name,
                         age_s=round(candidate.age_s, 1), stable_count=stable_count,
                         size=candidate.size, shard_count=candidate.shard_count)
                    result = promote(candidate, env_file)
                    atomic_write(STATE, json.dumps({
                        'step': candidate.step, 'checkpoint': candidate.path.name,
                        'signature': candidate.signature,
                        'promoted_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                        'result': result,
                    }, indent=2, sort_keys=True) + '\n')
                    emit('promotion_complete', **result)
                    args.force = False
                else:
                    emit('candidate_waiting', step=candidate.step, checkpoint=candidate.path.name,
                         age_s=round(candidate.age_s, 1), stable_count=stable_count,
                         required_age_s=args.min_age, required_confirmations=args.confirmations)
                last_rejection = ''
            except Exception as exc:
                message = f'{type(exc).__name__}: {exc}'
                if message != last_rejection:
                    emit('candidate_rejected', error=message)
                    last_rejection = message
                stable_sig = ''
                stable_count = 0
                if args.once:
                    return 1
            if args.once:
                return 0
            time.sleep(max(5.0, args.poll))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--latest-file', default=str(DEFAULT_LATEST))
    parser.add_argument('--env-file', default=str(DEFAULT_ENV))
    parser.add_argument('--poll', type=float, default=30.0)
    parser.add_argument('--min-age', type=float, default=60.0)
    parser.add_argument('--confirmations', type=int, default=2)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    args.confirmations = max(1, args.confirmations)
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
