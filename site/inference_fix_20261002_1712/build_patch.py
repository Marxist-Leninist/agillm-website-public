#!/usr/bin/env python3
"""Build non-decoding inference reliability patch; production deployment is separate."""
import ast, hashlib, json
from pathlib import Path
ROOT=Path('/opt/agillm43_netserve'); OUT=Path(__file__).parent
BP=ROOT/'agillm43_http_bridge.py'; RP=ROOT/'runtime_derived/runtime_standard_modes_20261001_minimum.py'
b=BP.read_text(); r=RP.read_text()
def replace_once(text, old, new):
    assert text.count(old)==1, (old[:90],text.count(old))
    return text.replace(old,new,1)
def replace_function(text,name,new):
    node=next(n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)
    lines=text.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+new.rstrip()+'\n'+''.join(lines[node.end_lineno:])
manifest={'original_bridge_sha256':hashlib.sha256(b.encode()).hexdigest(),'original_runtime_sha256':hashlib.sha256(r.encode()).hexdigest(),'checkpoint_weights_changed':False,'decoding_functions_changed':False}
(OUT/'bridge.before.py').write_text(b);(OUT/'runtime.before.py').write_text(r)
# Request-scoped markers avoid signal races and never cancel another request.
b=replace_once(b,'def cancel_request(request_id=None):', '''def _cooperative_cancel_path(pid, request_id):
    root = "/run/agillm43-cancel"
    os.makedirs(root, mode=0o700, exist_ok=True)
    digest = hashlib.sha256(str(request_id).encode("utf-8")).hexdigest()
    return os.path.join(root, "%s-%s.cancel" % (int(pid), digest))


def cancel_request(request_id=None):''')
anchor='''    try:
        if proc is not None and proc.poll() is None:
            proc.kill()
    except Exception as exc:
        return 500, {"ok": False, "error": "failed to stop request: %s" % exc}'''
cooperative='''    # 20261002: a cooperative child finishes its current forward, then aborts
    # only this request. Keep resident weights; do not cold-load after Stop.
    if proc is not None and proc.poll() is None and getattr(proc, "agillm_cooperative_cancel", False):
        try:
            marker = _cooperative_cancel_path(proc.pid, target_id)
            fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            os.close(fd)
        except OSError as exc:
            print("[bridge] cooperative cancel marker failed; using hard fallback: %s" % exc, flush=True)
        else:
            def hard_fallback():
                # Never kill a worker reused by a later request.
                with request_lock:
                    still_target = active_requests.get(target_id) is slot
                if still_target and proc.poll() is None:
                    print("[bridge] cooperative cancel deadline exceeded; retiring slot=%s" % slot_id, flush=True)
                    proc.kill()
                    _drop_slot(slot)
                    _schedule_pool_repair()
                try:
                    os.unlink(marker)
                except FileNotFoundError:
                    pass
            fallback = threading.Timer(15.0, hard_fallback)
            fallback.daemon = True
            fallback.start()
            return 200, {"ok": True, "canceling": True, "queued": False,
                         "request_id": target_id, "slot": slot_id,
                         "warming": False, "cooperative": True,
                         "note": "request canceled without unloading the model"}

'''+anchor
b=replace_once(b,anchor,cooperative)
b=replace_once(b,'            proc.agillm_model_identity = model_identity if _identity_inputs_unchanged(model_identity) else None','            proc.agillm_cooperative_cancel = "cooperative_cancel=1" in line\n            proc.agillm_model_identity = model_identity if _identity_inputs_unchanged(model_identity) else None')
assert b.count('        payload["_dtype"] = active_dtype')==2
b=b.replace('        payload["_dtype"] = active_dtype','        payload["_dtype"] = active_dtype\n        payload["_bridge_request_id"] = request_id')
b=replace_function(b,'_schedule_pool_repair','''def _schedule_pool_repair():
    # Deduplicate background repair instead of blocking queued HTTP handlers on
    # warm_lock or creating a thread for every queue poll.
    if not repair_schedule_lock.acquire(blocking=False):
        return
    def repair():
        try:
            ensure_pool()
        except Exception as exc:
            print("[bridge] pool repair failed: %s" % exc, flush=True)
        finally:
            repair_schedule_lock.release()
    threading.Thread(target=repair, name="agillm-pool-repair", daemon=True).start()
''')
b=replace_once(b,'reload_lock = threading.Lock()','repair_schedule_lock = threading.Lock()\nreload_lock = threading.Lock()')
b=replace_function(b,'acquire_slot','''def acquire_slot(max_wait_s=None, request_id=None):
    """Bound queue time INCLUDING model warm-up; honor queued cancellation."""
    if max_wait_s is None:
        max_wait_s = _queue_timeout_s()
    max_wait_s = max(0.1, float(max_wait_s))
    t0 = time.monotonic()
    deadline = t0 + max_wait_s
    repair_at = 0.0
    while time.monotonic() < deadline:
        if request_id:
            with request_lock:
                _prune_cancel_tombstones_locked()
                if request_id in cancelled_requests:
                    return None, time.monotonic() - t0
        with pool_lock:
            live = _planned_slots([s for s in slots if _slot_alive(s)])
            _refresh_ready_locked()
            ordered = sorted(live, key=lambda s: (-int(s.get("threads") or 0), int(s.get("id") or 0)))
        for slot in ordered:
            if slot["lock"].acquire(blocking=False):
                if _slot_alive(slot):
                    return slot, time.monotonic() - t0
                slot["lock"].release()
                _drop_slot(slot)
        now = time.monotonic()
        if not live and now >= repair_at:
            _schedule_pool_repair()
            repair_at = now + 1.0
        time.sleep(min(0.05, max(0.0, deadline - now)))
    return None, time.monotonic() - t0
''')
assert b.count('slot, queue_wait = acquire_slot()')==2
b=b.replace('slot, queue_wait = acquire_slot()','slot, queue_wait = acquire_slot(request_id=request_id)')
b=replace_once(b,'"elapsed_s": round(time.time() - heartbeat_started, 1),','''"elapsed_s": round(time.time() - heartbeat_started, 1),
                        "phase": ("generating" if stream_request_id in active_requests else
                                  "loading" if warming or not ready else "queued"),''')
b=replace_once(b,'"slot_dtypes": [s.get("dtype") for s in live],','''"slot_dtypes": [s.get("dtype") for s in live],
                             "cancel_preserves_worker": bool(live) and all(getattr(s["proc"], "agillm_cooperative_cancel", False) for s in live),
                             "target_ckpt": display_ckpt(bridge_env().get("AGILLM_BRIDGE_CKPT", "")),
                             "reliability_patch": "cooperative-cancel-20261002",''')
# Install lightweight checks on actual model/head forward boundaries, not on
# every submodule. Return None when not canceled: normal results are unchanged.
node=next(n for n in ast.parse(r).body if isinstance(n,ast.FunctionDef) and n.name=='infer_server')
ls=r.splitlines(keepends=True); fn=''.join(ls[node.lineno-1:node.end_lineno])
fn=replace_once(fn,'    print("[INFER_SERVER_READY]", flush=True)','''    import hashlib as _cancel_hashlib
    import os as _cancel_os
    import pathlib as _cancel_pathlib
    _cancel_current = {"path": None}
    def _cooperative_check(_module, _inputs):
        path = _cancel_current["path"]
        if path is not None and path.exists():
            raise RuntimeError("request canceled")
    _cancel_hooks = []
    for _name in ("core", "ar_h", "sat_h", "nat_h"):
        _module = inst.get(_name)
        if _module is not None:
            _cancel_hooks.append(_module.register_forward_pre_hook(_cooperative_check))
    print("[INFER_SERVER_READY] cooperative_cancel=1", flush=True)''')
fn=replace_once(fn,'            req = json.loads(raw_line)','''            req = json.loads(raw_line)
            _request_id = str(req.get("_bridge_request_id", ""))
            _cancel_current["path"] = (
                _cancel_pathlib.Path("/run/agillm43-cancel") /
                (str(_cancel_os.getpid()) + "-" + _cancel_hashlib.sha256(_request_id.encode("utf-8")).hexdigest() + ".cancel")
            ) if _request_id else None''')
fn=replace_once(fn,'            with torch.inference_mode():\n                _agillm43_generate_from_instance(inst, run_args)','            with torch.inference_mode():\n                _cooperative_check(None, None)\n                _agillm43_generate_from_instance(inst, run_args)')
assert fn.rstrip().endswith('print("[INFER_SERVER_RESULT_END]", flush=True)')
fn=fn.rstrip()+'''
        finally:
            _path = _cancel_current["path"]
            _cancel_current["path"] = None
            if _path is not None:
                try:
                    _path.unlink(missing_ok=True)
                except OSError:
                    pass
'''
r=''.join(ls[:node.lineno-1])+fn+''.join(ls[node.end_lineno:])
ast.parse(b);ast.parse(r)
# Strong invariant: only infer_server changed in the 35k-line runtime.
def top_functions(src):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(src).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
before=top_functions((OUT/'runtime.before.py').read_text());after=top_functions(r)
changed=[k for k in before if before[k]!=after.get(k)]
assert changed==['infer_server'],changed
(OUT/'bridge.patched.py').write_text(b);(OUT/'runtime.patched.py').write_text(r)
manifest.update(bridge_sha256=hashlib.sha256(b.encode()).hexdigest(),runtime_sha256=hashlib.sha256(r.encode()).hexdigest(),changed_runtime_functions=changed)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
