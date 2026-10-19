#!/usr/bin/env python3
"""Transactionally promote a Hugging Face-verified AGILLM4.3 checkpoint on GETH.

The checkpoint bytes may arrive through any transport. Hugging Face's immutable
SHA256SUMS and HF_VERIFIED receipt are the authority. The bridge changes only
after every serving file hashes correctly and the fail-closed reload gate passes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request

NETSERVE = pathlib.Path('/opt/agillm43_netserve')
BRIDGE_ENV = NETSERVE / 'bridge.env'
RELOAD_GATE = NETSERVE / 'agillm43_safe_reload.py'
PATCHER = NETSERVE / 'agillm43_patch_runtime_inspector.py'
DERIVED_RUNTIME_DIR = NETSERVE / 'runtime_derived'
STATUS_JSON = NETSERVE / 'agillm_hf_status.json'
SERVE_ROOT = pathlib.Path('/mnt/geth-vol1/agillm43/ckpts/artifacts/full/hf_verified')
BRIDGE_SERVICE = 'agillm43-bridge.service'
SERVE_KEEP = 2
PACKAGE_DIR_RE = re.compile(r'^step([0-9]+)-([0-9a-f]{16})$')
SHA_RE = re.compile(r'^[0-9a-f]{64}$')
STEP_RE = re.compile(r'pretrain_step0*([0-9]+).*\.pt$')
TRUSTED_VAST_BUILDER_SHA = 'c5d3fbeacfcf15b42117b3599d2456c89c960491f57cf787afb8c6b255041371'
TRUSTED_VAST_RUNTIME_SHA = '64f2ac73bd7c3aff489ff3caf795ca5daffdefc28932d94a2e71805a6461bc57'
EXPECTED_VAST_LABEL = 'agillm43-retrain-134B'


class PromotionError(RuntimeError):
    pass


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open('rb', buffering=0) as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_text(path: pathlib.Path, text: str, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + f'.tmp.{os.getpid()}')
    tmp.write_text(text, encoding='utf-8')
    os.chmod(tmp, mode)
    with tmp.open('rb') as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: pathlib.Path, value: object, mode: int = 0o644) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + '\n', mode)


def install_bridge_env(source: pathlib.Path) -> None:
    """Replace the intentionally immutable bridge.env, then restore immutability."""
    subprocess.run(['chattr', '-i', str(BRIDGE_ENV)], check=False)
    tmp = BRIDGE_ENV.with_name('.' + BRIDGE_ENV.name + f'.install.{os.getpid()}')
    try:
        shutil.copy2(source, tmp)
        os.chmod(tmp, 0o600)
        with tmp.open('rb') as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, BRIDGE_ENV)
        subprocess.run(['chattr', '+i', str(BRIDGE_ENV)], check=True)
    except Exception:
        tmp.unlink(missing_ok=True)
        # Do not leave the live configuration casually mutable after an error.
        subprocess.run(['chattr', '+i', str(BRIDGE_ENV)], check=False)
        raise


def load_env(path: pathlib.Path) -> tuple[list[str], dict[str, str]]:
    lines = path.read_text(encoding='utf-8').splitlines()
    values: dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            values[k] = v
    return lines, values


def update_env(lines: list[str], updates: dict[str, str]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for raw in lines:
        if '=' in raw and not raw.lstrip().startswith('#'):
            key = raw.split('=', 1)[0]
            if key in updates:
                out.append(f'{key}={updates[key]}')
                seen.add(key)
                continue
        out.append(raw)
    for key, value in updates.items():
        if key not in seen:
            out.append(f'{key}={value}')
    return '\n'.join(out).rstrip() + '\n'


def parse_sums(path: pathlib.Path) -> dict[str, str]:
    sums: dict[str, str] = {}
    for number, raw in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not raw.strip():
            continue
        try:
            digest, rel = raw.split('  ', 1)
        except ValueError as exc:
            raise PromotionError(f'invalid SHA256SUMS line {number}') from exc
        if not SHA_RE.fullmatch(digest) or rel.startswith('/') or '..' in pathlib.PurePosixPath(rel).parts:
            raise PromotionError(f'unsafe SHA256SUMS line {number}')
        sums[rel] = digest
    return sums


def package_file(source: pathlib.Path, rel: str) -> pathlib.Path:
    direct = source / rel
    if direct.exists():
        return direct
    if rel.startswith('checkpoints/'):
        stripped = source / rel.removeprefix('checkpoints/')
        if stripped.exists():
            return stripped
    return direct


def verify_source(source: pathlib.Path, metadata: pathlib.Path) -> dict[str, object]:
    meta = json.loads((metadata / 'metadata.json').read_text(encoding='utf-8'))
    source_kind = str(meta.get('source') or 'huggingface')
    if source_kind == 'huggingface':
        receipt_path = metadata / 'HF_VERIFIED.json'
        integrity = 'hf-sha256-verified'
    elif source_kind == 'vast-direct':
        receipt_path = metadata / 'VAST_VERIFIED.json'
        integrity = 'vast-source-and-destination-sha256-verified'
    else:
        raise PromotionError(f'unsupported verification source: {source_kind!r}')

    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    export = json.loads((metadata / 'recovery/export_manifest.json').read_text(encoding='utf-8'))
    if receipt.get('ok') is not True or int(receipt.get('step', -1)) != int(meta.get('step', -2)):
        raise PromotionError('verification receipt is not valid for metadata step')
    if receipt.get('content_commit') != meta.get('content_commit'):
        raise PromotionError('verification receipt and serving metadata disagree on content identity')
    if export.get('status') != 'validated_complete' or int(export.get('step', -1)) != int(receipt['step']):
        raise PromotionError('export manifest is not a validated complete checkpoint')

    sums_path = metadata / 'SHA256SUMS'
    sums = parse_sums(sums_path)
    checkpoint_entries = {rel: digest for rel, digest in sums.items() if rel.startswith('checkpoints/')}
    if not checkpoint_entries:
        raise PromotionError('SHA256SUMS has no checkpoint package entries')
    checked = 0
    total = 0
    for rel, expected in sorted(checkpoint_entries.items()):
        path = package_file(source, rel)
        if path.is_symlink() or not path.is_file():
            raise PromotionError(f'missing or unsafe checkpoint member: {rel}')
        actual = sha256_file(path)
        if actual != expected:
            raise PromotionError(f'SHA256 mismatch for {rel}: {actual} != {expected}')
        checked += 1
        total += path.stat().st_size

    runtime = metadata / 'runtime/agillm41.py'
    if runtime.is_symlink() or not runtime.is_file():
        raise PromotionError('verified inference runtime is missing or unsafe')
    runtime_sha = sha256_file(runtime)
    if runtime_sha != sums.get('runtime/agillm41.py'):
        raise PromotionError('runtime hash does not match SHA256SUMS')
    checkpoint_name = str(receipt['checkpoint'])
    checkpoint_sha = str(receipt['checkpoint_sha256'])
    if pathlib.PurePath(checkpoint_name).name != checkpoint_name or not checkpoint_name.endswith('.pt'):
        raise PromotionError('receipt checkpoint filename is unsafe')
    if checkpoint_entries.get('checkpoints/' + checkpoint_name) != checkpoint_sha:
        raise PromotionError('receipt checkpoint hash is absent from SHA256SUMS')

    if source_kind == 'vast-direct':
        checks = {
            'receipt_schema': receipt.get('schema') == 'agillm43.vast-direct-verification.v1',
            'metadata_schema': meta.get('schema') == 'agillm43.vast-serving-metadata.v1',
            'receipt_source': receipt.get('source') == 'vast-direct',
            'metadata_source': meta.get('source') == 'vast-direct',
            'route_label': receipt.get('route_label') == EXPECTED_VAST_LABEL,
            'builder_sha256': receipt.get('builder_sha256') == TRUSTED_VAST_BUILDER_SHA,
            'runtime_sha256': receipt.get('runtime_sha256') == TRUSTED_VAST_RUNTIME_SHA == runtime_sha,
            'source_verified': receipt.get('source_sha256_verified') is True,
            'destination_verified': receipt.get('destination_sha256_verified') is True,
            'sums_sha256': receipt.get('sha256sums_sha256') == sha256_file(sums_path),
            'receipt_commit': receipt.get('receipt_commit') == meta.get('receipt_commit'),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise PromotionError('Vast direct receipt failed: ' + ','.join(failed))
    else:
        schema = receipt.get('schema')
        if schema not in (None, 'agillm43.hf-verification.v2'):
            raise PromotionError('unexpected Hugging Face receipt schema')

    return {
        'receipt': receipt,
        'metadata': meta,
        'export': export,
        'sums': sums,
        'checkpoint_name': checkpoint_name,
        'checkpoint_sha256': checkpoint_sha,
        'runtime_sha256': runtime_sha,
        'checkpoint_files': checked,
        'checkpoint_bytes': total,
        'source_kind': source_kind,
        'integrity': integrity,
        'receipt_file': receipt_path.name,
    }


def fsync_tree(root: pathlib.Path) -> None:
    for path in sorted(root.rglob('*')):
        if path.is_file():
            with path.open('rb') as f:
                os.fsync(f.fileno())
    for path in sorted((p for p in root.rglob('*') if p.is_dir()), reverse=True):
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def stage_package(source: pathlib.Path, metadata: pathlib.Path, info: dict[str, object], move_source: bool) -> pathlib.Path:
    step = int(info['receipt']['step'])  # type: ignore[index]
    checkpoint_sha = str(info['checkpoint_sha256'])
    final = SERVE_ROOT / f'step{step}-{checkpoint_sha[:16]}'
    if final.is_dir():
        # Re-verify an existing final package before trusting it.
        verify_source(final, metadata)
        return final
    SERVE_ROOT.mkdir(parents=True, exist_ok=True)
    staging = SERVE_ROOT / f'.incoming-step{step}-{checkpoint_sha[:16]}.{os.getpid()}'
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(mode=0o700)
    checkpoints = staging / 'checkpoints'
    if move_source:
        if source.stat().st_dev != SERVE_ROOT.stat().st_dev:
            raise PromotionError('move-source requested across filesystems')
        os.replace(source, checkpoints)
    else:
        shutil.copytree(source, checkpoints, copy_function=shutil.copy2)
    shutil.copytree(metadata, staging / 'metadata')
    runtime_dir = staging / 'runtime'
    runtime_dir.mkdir()
    shutil.copy2(metadata / 'runtime/agillm41.py', runtime_dir / 'agillm41.py')
    atomic_json(staging / 'SERVING_PACKAGE.json', {
        'schema': 'agillm43.hf-serving-package.v1',
        'authoritative_source': info['source_kind'],
        'integrity': info['integrity'],
        'step': step,
        'checkpoint': info['checkpoint_name'],
        'checkpoint_sha256': checkpoint_sha,
        'runtime_sha256': info['runtime_sha256'],
        'repo': info['metadata']['repo'],  # type: ignore[index]
        'prefix': info['metadata']['prefix'],  # type: ignore[index]
        'content_commit': info['metadata']['content_commit'],  # type: ignore[index]
        'receipt_commit': info['metadata']['receipt_commit'],  # type: ignore[index]
        'verified_at': info['receipt']['verified_at'],  # type: ignore[index]
        'staged_at': dt.datetime.now(dt.timezone.utc).isoformat(),
    })
    fsync_tree(staging)
    os.replace(staging, final)
    fd = os.open(SERVE_ROOT, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    verify_source(final, metadata)
    return final


def package_size(root: pathlib.Path) -> int:
    total = 0
    for item in root.rglob('*'):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except FileNotFoundError:
            pass
    return total


def serving_package_identity(path: pathlib.Path) -> tuple[int, str] | None:
    """Validate enough immutable identity before a package may be retention-pruned."""
    try:
        if path.is_symlink() or not path.is_dir() or os.path.ismount(path):
            return None
        if path.parent.resolve() != SERVE_ROOT.resolve():
            return None
        match = PACKAGE_DIR_RE.fullmatch(path.name)
        if not match:
            return None
        marker = json.loads((path / 'SERVING_PACKAGE.json').read_text(encoding='utf-8'))
        step = int(marker.get('step', -1))
        digest = str(marker.get('checkpoint_sha256') or '')
        checkpoint = str(marker.get('checkpoint') or '')
        if marker.get('schema') != 'agillm43.hf-serving-package.v1':
            return None
        if step != int(match.group(1)) or not SHA_RE.fullmatch(digest) or not digest.startswith(match.group(2)):
            return None
        if pathlib.PurePath(checkpoint).name != checkpoint or not checkpoint.endswith('.pt'):
            return None
        if not (path / 'checkpoints' / checkpoint).is_file():
            return None
        if not (path / 'runtime' / 'agillm41.py').is_file():
            return None
        return step, digest
    except Exception:
        return None


def prune_old_serving_packages(current: pathlib.Path, keep: int = SERVE_KEEP) -> dict[str, object]:
    """Keep the current HF package and one verified rollback, never a rolling mirror."""
    if keep < 1:
        raise ValueError('serving package retention must keep at least one package')
    current = current.resolve()
    if current.parent != SERVE_ROOT.resolve() or serving_package_identity(current) is None:
        raise PromotionError('current serving package failed retention identity checks')
    packages: list[tuple[int, pathlib.Path]] = []
    if SERVE_ROOT.exists():
        for path in SERVE_ROOT.iterdir():
            identity = serving_package_identity(path)
            if identity is not None:
                packages.append((identity[0], path.resolve()))
    packages.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    ordered = [current] + [path for _, path in packages if path != current]
    keep_paths = set(ordered[:keep])
    removed: list[dict[str, object]] = []
    for _, path in sorted(packages, key=lambda item: (item[0], item[1].name)):
        if path in keep_paths:
            continue
        size = package_size(path)
        retiring = SERVE_ROOT / f'.retiring-{path.name}.{os.getpid()}'
        if retiring.exists():
            shutil.rmtree(retiring)
        os.replace(path, retiring)
        shutil.rmtree(retiring)
        removed.append({'path': str(path), 'bytes': size})
    if removed:
        fd = os.open(SERVE_ROOT, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return {
        'policy': f'keep-{keep}-verified-serving-packages',
        'kept': [str(path) for path in ordered[:keep] if path.exists()],
        'removed': removed,
    }


def health() -> dict[str, object]:
    try:
        with urllib.request.urlopen('http://127.0.0.1:8493/health', timeout=10) as r:
            return json.loads(r.read())
    except Exception:
        return {}


def wait_idle(timeout: int = 300) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        h = health()
        if h.get('ready') is True and int(h.get('busy', 0)) == 0 and not h.get('warming'):
            return
        time.sleep(2)
    raise PromotionError('bridge did not become idle before promotion')


def wait_checkpoint(name: str, timeout: int = 900) -> dict[str, object]:
    deadline = time.time() + timeout
    last: dict[str, object] = {}
    while time.time() < deadline:
        last = health()
        if last.get('ready') is True and last.get('ckpt') == name and int(last.get('busy', 0)) == 0:
            return last
        time.sleep(3)
    raise PromotionError(f'bridge failed to serve {name}; last health={last}')


def request_json(path: str, token: str, payload: dict[str, object], timeout: int = 180) -> dict[str, object]:
    body = json.dumps(payload).encode()
    req = urllib.request.Request('http://127.0.0.1:8493' + path, data=body, method='POST', headers={
        'Content-Type': 'application/json',
        'X-AGILLM-Token': token,
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as exc:
        raise PromotionError(f'{path} HTTP {exc.code}: {exc.read()[:500]!r}') from exc


def smoke_stream_logits(token: str) -> dict[str, object]:
    results: dict[str, object] = {}
    for mode, extra in [('ar', {}), ('nat', {'nat_passes': 1}), ('sat', {'var': False})]:
        payload: dict[str, object] = {
            'prompt': 'The',
            'mode': mode,
            'max_new': 2,
            'min_new': 2,
            'temperature': 0.25,
            'top_p': 1.0,
            'greedy': True,
            'ignore_eos': True,
        }
        payload.update(extra)
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            'http://127.0.0.1:8493/stream', data=body, method='POST', headers={
                'Content-Type': 'application/json',
                'X-AGILLM-Token': token,
            },
        )
        token_events = 0
        top_events = 0
        done = False
        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                for raw in response:
                    if not raw.strip():
                        continue
                    event = json.loads(raw.decode('utf-8', 'replace'))
                    if event.get('event') == 'error':
                        raise PromotionError(f'{mode} stream error: {event.get("error")}')
                    if event.get('event') == 'stream_tok':
                        token_events += 1
                        data = event.get('data') if isinstance(event.get('data'), dict) else {}
                        top = data.get('top')
                        if isinstance(top, list) and top:
                            for item in top:
                                probability = float(item.get('prob'))
                                if probability < 0.0 or probability > 1.0:
                                    raise PromotionError(f'{mode} invalid logit probability: {probability}')
                                if not isinstance(item.get('token'), str):
                                    raise PromotionError(f'{mode} logit token is not text')
                            top_events += 1
                    elif event.get('event') == 'done':
                        done = True
        except urllib.error.HTTPError as exc:
            raise PromotionError(f'{mode} stream HTTP {exc.code}: {exc.read()[:500]!r}') from exc
        if not done or token_events < 1 or top_events < 1:
            raise PromotionError(
                f'{mode} logit stream smoke failed: done={done} '
                f'token_events={token_events} top_events={top_events}'
            )
        results[mode] = {
            'ok': True,
            'token_events': token_events,
            'top_events': top_events,
        }
    return results


def smoke(token: str) -> dict[str, object]:
    results: dict[str, object] = {}
    for mode, extra in [('nat', {'nat_passes': 1}), ('ar', {}), ('sat', {'var': False})]:
        payload: dict[str, object] = {
            'prompt': 'The scientific method requires',
            'mode': mode,
            'max_new': 8,
            'min_new': 1,
            'temperature': 0.25,
            'top_p': 1.0,
            'greedy': True,
            'ignore_eos': True,
        }
        payload.update(extra)
        result = request_json('/infer', token, payload)
        if result.get('ok') is not True:
            raise PromotionError(f'{mode} smoke failed: {result}')
        results[mode] = {
            'ok': True,
            'tokens': result.get('tokens'),
            'tok_s': result.get('tok_s'),
            'wall_s': result.get('wall_s'),
        }
    results['stream_logits'] = smoke_stream_logits(token)
    return results


def run(args: argparse.Namespace) -> dict[str, object]:
    source = args.source.resolve(strict=True)
    metadata = args.metadata.resolve(strict=True)
    info = verify_source(source, metadata)
    final = stage_package(source, metadata, info, args.move_source)
    checkpoint = final / 'checkpoints' / str(info['checkpoint_name'])
    runtime_sha = str(info['runtime_sha256'])
    source_runtime = NETSERVE / f'agillm41_runtime_{runtime_sha[:12]}.py'
    if not source_runtime.exists():
        shutil.copy2(metadata / 'runtime/agillm41.py', source_runtime)
        os.chmod(source_runtime, 0o644)
    if sha256_file(source_runtime) != runtime_sha:
        raise PromotionError('installed source inference runtime hash mismatch')
    subprocess.run([sys.executable, '-m', 'py_compile', str(source_runtime)], check=True)

    DERIVED_RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    runtime = DERIVED_RUNTIME_DIR / f'agillm41_runtime_{runtime_sha[:12]}_logits_v1.py'
    patch = subprocess.run([
        sys.executable, str(PATCHER), '--input', str(source_runtime), '--output', str(runtime),
    ], text=True, capture_output=True)
    if patch.returncode != 0:
        raise PromotionError(
            f'logit runtime patch rejected candidate: {(patch.stderr or patch.stdout).strip()[:1000]}'
        )
    try:
        patch_info = json.loads(patch.stdout.strip().splitlines()[-1])
    except Exception as exc:
        raise PromotionError('logit runtime patch returned invalid JSON') from exc
    if patch_info.get('source_sha256') != runtime_sha:
        raise PromotionError('logit runtime patch source hash mismatch')
    serving_runtime_sha = sha256_file(runtime)
    if patch_info.get('output_sha256') != serving_runtime_sha:
        raise PromotionError('logit runtime patch output hash mismatch')
    subprocess.run([sys.executable, '-m', 'py_compile', str(runtime)], check=True)

    old_lines, old_values = load_env(BRIDGE_ENV)
    token = old_values.get('AGILLM_BRIDGE_TOKEN', '')
    if not token:
        raise PromotionError('bridge token is missing')
    candidate = BRIDGE_ENV.with_name(f'.bridge.env.hf-step{info["receipt"]["step"]}.{os.getpid()}')  # type: ignore[index]
    candidate_text = update_env(old_lines, {
        'AGILLM_BRIDGE_CKPT': str(checkpoint),
        'AGILLM_BRIDGE_PINNED_CKPT': str(checkpoint),
        'AGILLM_BRIDGE_PINNED_SHA256': str(info['checkpoint_sha256']),
        'AGILLM_BRIDGE_RUNTIME': str(runtime),
        'AGILLM_BRIDGE_REQUIRE_PIN': '1',
    })
    atomic_text(candidate, candidate_text)
    gate = subprocess.run([sys.executable, str(RELOAD_GATE), '--env', str(candidate), '--check-only'], text=True, capture_output=True)
    if gate.returncode != 0:
        raise PromotionError(f'reload gate rejected candidate: {gate.stderr.strip()}')
    wait_idle()

    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = BRIDGE_ENV.with_name(f'bridge.env.backup-hf-step{info["receipt"]["step"]}-{stamp}')  # type: ignore[index]
    shutil.copy2(BRIDGE_ENV, backup)
    old_status = STATUS_JSON.read_bytes() if STATUS_JSON.exists() else None
    try:
        subprocess.run(['systemctl', 'stop', BRIDGE_SERVICE], check=True)
        install_bridge_env(candidate)
        subprocess.run(['systemctl', 'start', BRIDGE_SERVICE], check=True)
        bridge_health = wait_checkpoint(str(info['checkpoint_name']))
        smokes = smoke(token)
        deployed_at = dt.datetime.now(dt.timezone.utc).isoformat()
        status = {
            'schema': ('agillm43.hf-serving-status.v1' if info['source_kind'] == 'huggingface'
                       else 'agillm43.serving-status.v2'),
            'status': 'serving_verified',
            'serving': True,
            'authoritative_source': info['source_kind'],
            'source': info['source_kind'],
            'integrity': info['integrity'],
            'repo': info['metadata']['repo'],  # type: ignore[index]
            'prefix': info['metadata']['prefix'],  # type: ignore[index]
            'content_commit': info['metadata']['content_commit'],  # type: ignore[index]
            'receipt_commit': info['metadata']['receipt_commit'],  # type: ignore[index]
            'verified_at': info['receipt']['verified_at'],  # type: ignore[index]
            'deployed_at': deployed_at,
            'step': int(info['receipt']['step']),  # type: ignore[index]
            'file': info['checkpoint_name'],
            'checkpoint_sha256': info['checkpoint_sha256'],
            'runtime_sha256': runtime_sha,
            'source_runtime_sha256': runtime_sha,
            'serving_runtime_sha256': serving_runtime_sha,
            'serving_runtime': str(runtime),
            'runtime_patch': patch_info,
            'verified_files': info['receipt']['files_verified'],  # type: ignore[index]
            'verified_bytes': info['receipt']['verified_bytes'],  # type: ignore[index]
            'package_path': str(final),
            'bridge': bridge_health,
            'smoke': smokes,
        }
        try:
            status['retention'] = prune_old_serving_packages(final)
        except Exception as exc:
            status['retention_warning'] = f'{type(exc).__name__}: {exc}'[:500]
        atomic_json(STATUS_JSON, status)
        return status
    except Exception:
        subprocess.run(['systemctl', 'stop', BRIDGE_SERVICE], check=False)
        install_bridge_env(backup)
        subprocess.run(['systemctl', 'start', BRIDGE_SERVICE], check=False)
        old_name = pathlib.Path(old_values.get('AGILLM_BRIDGE_CKPT', '')).name
        if old_name:
            try:
                wait_checkpoint(old_name, timeout=600)
            except Exception:
                pass
        if old_status is None:
            STATUS_JSON.unlink(missing_ok=True)
        else:
            tmp = STATUS_JSON.with_name('.' + STATUS_JSON.name + f'.restore.{os.getpid()}')
            tmp.write_bytes(old_status)
            os.replace(tmp, STATUS_JSON)
        raise
    finally:
        candidate.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', type=pathlib.Path, required=True, help='verified checkpoint package directory')
    ap.add_argument('--metadata', type=pathlib.Path, required=True, help='HF receipt/SHA/runtime directory')
    ap.add_argument('--move-source', action='store_true', help='rename source into serving storage on same filesystem')
    args = ap.parse_args()
    try:
        result = run(args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(json.dumps({'ok': False, 'error': type(exc).__name__, 'message': str(exc)}, indent=2), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
