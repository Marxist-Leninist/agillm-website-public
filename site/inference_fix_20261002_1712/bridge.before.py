#!/usr/bin/env python3
"""HTTP bridge for the AGILLM4.3 warm --server stdin loop. v2: zero-downtime reload.

POST /infer   {"prompt": ..., "max_new": 64, "mode": "nat", ...}
GET  /health  {"ok", "ready", "warming", "ckpt", "uptime_s", "loads"}
POST /reload  (token-authed) warm a replacement child from a fresh read of
              bridge.env, swap atomically when READY, retire the old child.
              The old child keeps serving during the warm-up. SIGHUP does the
              same (systemd: `systemctl reload agillm43-bridge`).

NEVER restart this service just to change ckpt/dtype/threads: edit bridge.env
then `systemctl reload agillm43-bridge` for a zero-downtime swap.
"""
import hashlib, json, os, re, signal, subprocess, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_FILE = os.path.join(BASE_DIR, "bridge.env")
# Keep the inference runtime selectable in bridge.env.  Checkpoint formats can
# evolve independently of the HTTP bridge, and a versioned path makes an
# upgrade or rollback explicit instead of silently replacing the old runtime.
DEFAULT_RUNTIME = "/workspace/agillm43_infer_demo/agillm41_current.py"
DEFAULT_CKPT = "/opt/agillm43_offload/ckpts/pretrain_step00002127_from00243186_20260701T0647Z.pt"
PORT = int(os.environ.get("AGILLM_BRIDGE_PORT", "8493"))
TOKEN = (os.environ.get("AGILLM_BRIDGE_TOKEN") or "").strip()
HOST_LABEL = (os.environ.get("AGILLM_BRIDGE_HOST_LABEL") or os.uname().nodename or "inference").strip()
DEVICE_LABEL = (os.environ.get("AGILLM_BRIDGE_DEVICE_LABEL") or "cpu").strip()
STREAM_HEARTBEAT_S = max(1.0, float(os.environ.get("AGILLM_STREAM_HEARTBEAT_S", "10")))

def _rewrite_legacy_sat_var_gate(payload):
    """Undo the 20260821 SAT-var profile that forced almost-all stride-1."""
    try:
        temp = float(payload.get("sat_gate_temperature", 1.0) or 1.0)
    except (TypeError, ValueError):
        temp = 1.0
    try:
        bias = float(payload.get("sat_gate_stride2_bias", 0.0) or 0.0)
    except (TypeError, ValueError):
        bias = 0.0
    greedy_raw = payload.get("sat_gate_greedy", True)
    greedy = str(greedy_raw).strip().lower() in {"1", "true", "yes", "on"}
    if temp >= 50.0 and bias <= -10.0 and not greedy:
        payload["sat_gate_greedy"] = True
        payload["sat_gate_temperature"] = 1.0
        payload["sat_gate_stride2_bias"] = 0.0
        payload["sat_gate_stride1_threshold"] = 0.05
        payload["_sat_var_gate_rewritten"] = "legacy-t100-bias-14"



def _env_seconds(name, default, minimum=1.0, maximum=3600.0):
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = float(default)
    return max(float(minimum), min(float(maximum), value))


# A child can occasionally stop producing stdout while remaining alive. A plain
# readline() has no Python-level timeout, so enforce the deadline by retiring the
# child process; EOF then unblocks the request thread and the pool can replace it.
REQUEST_TIMEOUT_S = _env_seconds("AGILLM_BRIDGE_REQUEST_TIMEOUT_S", 300.0, 10.0, 1800.0)
QUEUE_TIMEOUT_S = _env_seconds("AGILLM_BRIDGE_QUEUE_TIMEOUT_S", 60.0, 1.0, 600.0)
STAT_RE = re.compile(r"\[(?P<sec>[0-9.]+)s \| (?P<tok>[0-9]+) tokens \| (?P<tps>[0-9.]+) tok/s\]")

REQUEST_FIELDS = {"prompt", "mode", "max_new", "min_new", "temperature", "top_k",
                  "top_p", "greedy", "ignore_eos", "nat_passes", "nat_greedy", "var",
                  "sat_trace", "sat_symbol_penalty", "sat_function_penalty", "sat_numeric_penalty", "sat_ar_first_weight", "sat_ar_second_weight",
                  "sat_gate_greedy", "sat_gate_temperature", "sat_gate_stride2_bias",
                  "sat_gate_stride1_threshold", "sat_prompt_align_greedy",
                  "sat_realign_greedy", "sat_second_greedy", "sat_second_temperature_scale",
                  "repetition_penalty", "presence_penalty", "frequency_penalty", "penalty_last_n",
                  "min_p", "nat_conf_threshold", "nat_chunk_size", "nat_stream_top_every",
                  "nat_ar_verify", "nat_ar_top_k", "nat_ar_verify_mode", "nat_ar_nat_weight",
                  "nat_ar_row_batch", "nat_anti_collapse", "ar_draft", "ar_draft_max",
                  "sampler", "euler_steps", "euler_start_sigma", "dblock_blocks",
                  "dblock_infer_blocks", "output_row_norm_clip",
                  "dtype", "infer_dtype", "_dtype"}

DTYPE_ALIASES = {
    "fp32": "fp32", "float32": "fp32", "float": "fp32",
    "fp16": "fp16", "float16": "fp16", "half": "fp16",
    "bf16": "bf16", "bfloat16": "bf16",
}

pool_lock = threading.RLock()     # protects the warm child pool metadata
warm_lock = threading.RLock()    # serializes repair and blue-green pool changes
reload_lock = threading.Lock()   # one cancel/reload at a time
slots = []                       # each slot owns one warm stdin/stdout child
next_slot_id = 0
next_slot_pick = 0
cancel_epoch = 0                 # invalidates handlers that were waiting before /cancel
ready = False
warming = False
started_at = time.time()
loads = 0
# Each slot has its own pipe_needs_resync flag. A single child pipe cannot serve
# two requests concurrently, so concurrency means multiple warm children.
# Request IDs remain bridge-local: the model runtime never sees them.
request_lock = threading.RLock()
pending_requests = set()
cancelled_requests = {}  # request_id -> monotonic expiry, covers cancel-before-register races
active_requests = {}
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
CANCEL_TOMBSTONE_S = 120.0
MAX_CANCEL_TOMBSTONES = 1024


def _request_identity(req):
    raw = str((req or {}).get("request_id") or "").strip()
    if raw:
        if not REQUEST_ID_RE.fullmatch(raw):
            raise ValueError("request_id must be 1-128 characters: letters, digits, dot, underscore, colon or dash")
        return raw
    return "legacy-%d-%d" % (time.time_ns(), threading.get_ident())


def _prune_cancel_tombstones_locked(now=None):
    now = time.monotonic() if now is None else float(now)
    for request_id, expiry in list(cancelled_requests.items()):
        if expiry <= now:
            cancelled_requests.pop(request_id, None)
    if len(cancelled_requests) > MAX_CANCEL_TOMBSTONES:
        for request_id, _ in sorted(cancelled_requests.items(), key=lambda item: item[1])[
                :len(cancelled_requests) - MAX_CANCEL_TOMBSTONES]:
            cancelled_requests.pop(request_id, None)


def _remember_cancel_locked(request_id):
    _prune_cancel_tombstones_locked()
    cancelled_requests[request_id] = time.monotonic() + CANCEL_TOMBSTONE_S
    _prune_cancel_tombstones_locked()


def _register_pending_request(request_id):
    with request_lock:
        _prune_cancel_tombstones_locked()
        if request_id in cancelled_requests:
            cancelled_requests.pop(request_id, None)
            return "canceled"
        if request_id in pending_requests or request_id in active_requests:
            return "duplicate"
        pending_requests.add(request_id)
        return "ok"


def _activate_request(request_id, slot):
    with request_lock:
        _prune_cancel_tombstones_locked()
        pending_requests.discard(request_id)
        if request_id in cancelled_requests:
            cancelled_requests.pop(request_id, None)
            return False
        active_requests[request_id] = slot
        return True


def _finish_request(request_id, slot=None):
    if not request_id:
        return
    with request_lock:
        if slot is None or active_requests.get(request_id) is slot:
            active_requests.pop(request_id, None)
        pending_requests.discard(request_id)
        cancelled_requests.pop(request_id, None)


def _request_counts():
    with request_lock:
        _prune_cancel_tombstones_locked()
        return len(active_requests), len(pending_requests)


def cancel_request(request_id=None):
    """Cancel one queued/active request without disturbing any other slot."""
    raw = str(request_id or "").strip()
    if raw and not REQUEST_ID_RE.fullmatch(raw):
        return 400, {"ok": False, "error": "invalid request_id"}
    with request_lock:
        if raw:
            target_id = raw
        else:
            candidates = list(active_requests) + list(pending_requests)
            candidates = list(dict.fromkeys(candidates))
            if not candidates:
                return 404, {"ok": False, "error": "no active request"}
            if len(candidates) != 1:
                return 409, {"ok": False, "error": "request_id required while multiple requests are active"}
            target_id = candidates[0]

        if target_id in pending_requests:
            # A queued request has not been handed to a model child yet. Remove
            # it from the visible pending set immediately; the tombstone still
            # makes the waiting handler fail closed if it later reaches activate.
            pending_requests.discard(target_id)
            _remember_cancel_locked(target_id)
            return 202, {"ok": True, "canceling": True, "queued": True,
                         "request_id": target_id, "note": "queued request canceled"}
        slot = active_requests.get(target_id)
        if slot is None:
            # Stop and generation are separate HTTP requests and can arrive out
            # of order. Remember the stop briefly so a later registration with
            # this ID is rejected before it can acquire/write to a child.
            _remember_cancel_locked(target_id)
            return 202, {"ok": True, "canceling": True, "queued": True,
                         "request_id": target_id, "note": "cancel remembered before request registration"}
        _remember_cancel_locked(target_id)
        proc = slot.get("proc")
        slot["cancel_request_id"] = target_id
        slot_id = slot.get("id")

    try:
        if proc is not None and proc.poll() is None:
            proc.kill()
    except Exception as exc:
        return 500, {"ok": False, "error": "failed to stop request: %s" % exc}
    # stop-readiness 20260924: retire the killed slot before answering. Until the
    # stream handler noticed EOF the dead child still counted as a live slot, so
    # /health kept reporting ready=True for a moment after Stop. The UI's
    # readiness wait saw two ready polls, re-enabled Continue, and the next
    # request sat queued (heartbeats only) behind a 1-2 minute replacement
    # warm-up. Dropping the slot here makes /health report ready=False and
    # warming=True immediately and starts the replacement warm right away.
    try:
        if proc is not None:
            proc.wait(timeout=5.0)
    except Exception:
        pass
    slot["pipe_needs_resync"] = True
    _drop_slot(slot)
    _schedule_pool_repair()
    with pool_lock:
        live_slots = len(_planned_slots([s for s in slots if _slot_alive(s)]))
    return 200, {"ok": True, "canceling": True, "queued": False,
                 "request_id": target_id, "slot": slot_id,
                 "live_slots": live_slots, "warming": True,
                 "note": "target slot retired; replacement warms without touching other requests"}


def bridge_env():
    """Process env overlaid with a FRESH read of bridge.env (so reload picks up edits).

    AGILLM_BRIDGE_REQUIRE_PIN=1 protects the public GUI endpoint from
    accidentally drifting to an unvalidated latest delta. To test a candidate,
    either set REQUIRE_PIN=0 temporarily or update AGILLM_BRIDGE_PINNED_CKPT
    after the candidate passes the quality gate.
    """
    cfg = dict(os.environ)
    try:
        with open(ENV_FILE) as f:
            for raw in f:
                line = raw.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip()
    except OSError:
        pass
    pin = (cfg.get("AGILLM_BRIDGE_PINNED_CKPT") or "").strip()
    require_pin = (cfg.get("AGILLM_BRIDGE_REQUIRE_PIN") or "0").strip().lower()
    if pin and require_pin not in {"0", "false", "no", "off"}:
        requested = (cfg.get("AGILLM_BRIDGE_CKPT") or "").strip()
        if requested and requested != pin:
            print(
                "[bridge] quality pin overriding AGILLM_BRIDGE_CKPT "
                + os.path.basename(requested) + " -> " + os.path.basename(pin),
                flush=True,
            )
        cfg["AGILLM_BRIDGE_CKPT"] = pin
    return cfg


def normalize_dtype(value, default="fp32"):
    raw = str(value or "").strip().lower().replace("torch.", "")
    if not raw:
        return default
    return DTYPE_ALIASES.get(raw, default)


def requested_dtype(req):
    if not isinstance(req, dict):
        return None
    for key in ("infer_dtype", "dtype", "_dtype"):
        if req.get(key):
            return normalize_dtype(req.get(key), None)
    return None


def _truthy(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def effective_bridge_dtype(dtype, cfg=None):
    cfg = cfg or bridge_env()
    dtype = normalize_dtype(dtype, "fp32")
    # This public GETH bridge is CPU-backed. fp16/bf16 are useful for explicit
    # experiments, but they destroy the normal NAT button-to-done speed here.
    if dtype in {"fp16", "bf16"} and not _truthy(cfg.get("AGILLM_BRIDGE_ALLOW_CPU_HALF")):
        return "fp32"
    return dtype


def effective_requested_dtype(req):
    dtype = requested_dtype(req)
    if not dtype:
        return None
    effective = effective_bridge_dtype(dtype)
    if effective != dtype:
        print("[bridge] CPU dtype request " + dtype + " clamped to " + effective + " (set AGILLM_BRIDGE_ALLOW_CPU_HALF=1 to allow)", flush=True)
    return effective


def write_bridge_dtype(dtype):
    dtype = normalize_dtype(dtype, "fp32")
    lines = []
    found = False
    try:
        with open(ENV_FILE) as f:
            lines = f.read().splitlines()
    except OSError:
        lines = []
    out = []
    for line in lines:
        if line.startswith("AGILLM_BRIDGE_DTYPE="):
            out.append("AGILLM_BRIDGE_DTYPE=" + dtype)
            found = True
        else:
            out.append(line)
    if not found:
        out.append("AGILLM_BRIDGE_DTYPE=" + dtype)
    tmp = ENV_FILE + ".tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(out).rstrip() + "\n")
    os.replace(tmp, ENV_FILE)
    os.environ["AGILLM_BRIDGE_DTYPE"] = dtype
    return dtype


def ensure_request_dtype(req):
    dtype = effective_requested_dtype(req)
    current = effective_bridge_dtype(bridge_env().get("AGILLM_BRIDGE_DTYPE"))
    if not dtype:
        return current
    with pool_lock:
        live = [s for s in slots if _slot_alive(s)]
        mismatch = any(normalize_dtype(s.get("dtype"), current) != dtype for s in live)
    if current != dtype:
        print("[bridge] dtype switch requested " + current + " -> " + dtype, flush=True)
        write_bridge_dtype(dtype)
        mismatch = True
    if mismatch:
        result = reload_child()
        if not result.get("ok"):
            raise RuntimeError("dtype reload failed: " + str(result.get("error") or result))
    return dtype



def _identity_file_stamp(path):
    st = os.stat(path)
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def _verified_model_identity(ckpt, runtime):
    """Validate trusted local identity metadata against the exact worker inputs.

    No torch load and no inference. Missing/invalid metadata never invents a model
    identity or prevents legacy checkpoints from serving.
    """
    try:
        with open(ckpt + ".identity.json") as f:
            meta = json.load(f)
        model = meta["model"]
        if meta.get("schema_version") != 1:
            return None
        if not isinstance(model["name"], str) or not model["name"].strip():
            return None
        if type(model["step"]) is not int or model["step"] < 0:
            return None
        if type(model["seen_tokens"]) is not int or model["seen_tokens"] < 0:
            return None
        if meta["canonical_modes"] != ["ar", "sat_fixed", "sat_variable", "nat"]:
            return None
        inputs = [
            (ckpt, meta["checkpoint_file"], model["source_checkpoint_sha256"]),
            (runtime, meta["runtime_file"], meta["runtime_sha256"]),
            (ckpt + ".tokenizer.json", meta["tokenizer_file"], model["tokenizer_sha256"]),
        ]
        # Sharded checkpoints need the weight bytes bound as well as the manifest.
        bundle = meta.get("checkpoint_files", [])
        shard_root = os.path.realpath(ckpt + ".shards")
        shard_paths = set()
        for item in bundle:
            declared = item["path"]
            resolved = os.path.realpath(declared)
            if not resolved.startswith(shard_root + os.sep) or resolved in shard_paths:
                return None
            shard_paths.add(resolved)
            inputs.append((declared, declared, item["sha256"]))
        actual_shards = {os.path.realpath(os.path.join(root, name))
                         for root, _, names in os.walk(shard_root) for name in names}
        if shard_paths != actual_shards:
            return None
        stamps = []
        for actual, declared, expected in inputs:
            if not os.path.isabs(declared) or os.path.realpath(actual) != os.path.realpath(declared):
                return None
            if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
                return None
            before = _identity_file_stamp(actual)
            digest = hashlib.sha256()
            with open(actual, "rb") as f:
                for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected or _identity_file_stamp(actual) != before:
                return None
            stamps.append((actual, before))
        return {
            "model": {key: model[key] for key in
                      ("name", "step", "seen_tokens", "source_checkpoint_sha256", "tokenizer_sha256")},
            "canonical_modes": list(meta["canonical_modes"]),
            "_files": stamps,
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _identity_inputs_unchanged(identity):
    try:
        return bool(identity) and all(
            _identity_file_stamp(path) == stamp for path, stamp in identity["_files"])
    except (OSError, KeyError, TypeError):
        return False


def _health_model_identity(live):
    identities = [slot.get("model_identity") for slot in live]
    if not identities or any(identity is None for identity in identities):
        return {"model_identity_verified": False}
    first = identities[0]
    if any(identity["model"] != first["model"] or
           identity["canonical_modes"] != first["canonical_modes"]
           for identity in identities[1:]):
        return {"model_identity_verified": False}
    return {"model_identity_verified": True, "model": first["model"],
            "checkpoint_step": first["model"]["step"], "step": first["model"]["step"],
            "canonical_modes": first["canonical_modes"]}


def spawn_child(threads=None, ckpt_override=None):
    """Start a fresh warm child; block until READY.

    Returns (proc, ckpt, threads) or (None, ckpt, threads). The primary slot
    can use the full CPU budget while overflow slots keep a second request from
    waiting behind a slow AR run.
    """
    cfg = bridge_env()
    runtime = (cfg.get("AGILLM_BRIDGE_RUNTIME") or DEFAULT_RUNTIME).strip()
    ckpt = ckpt_override or cfg.get("AGILLM_BRIDGE_CKPT", DEFAULT_CKPT)
    if threads is None:
        threads = _worker_thread_plan(cfg)[0]
    try:
        threads = str(max(1, int(threads)))
    except Exception:
        threads = "1"
    dtype = effective_bridge_dtype(cfg.get("AGILLM_BRIDGE_DTYPE"), cfg)
    interop = cfg.get("AGILLM_BRIDGE_INTEROP", "0")
    model_identity = _verified_model_identity(ckpt, runtime)
    env = dict(cfg)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = threads
    env["MKL_NUM_THREADS"] = threads
    cmd = ["python3", "-u", runtime, "infer", "--server", "--device", "cpu",
           "--cpu_threads", threads, "--ckpt", ckpt, "--mode", "nat",
           "--nat_passes", "1", "--max_new", "64", "--min_new", "0",
           "--temperature", "0.0", "--top_p", "1.0", "--greedy", "--ignore_eos",
           "--plain-output", "--repetition_penalty", "1.05",
           "--presence_penalty", "0.0", "--frequency_penalty", "0.0",
           "--penalty_last_n", "128", "--infer_dtype", dtype,
           "--cpu_interop_threads", interop]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, env=env, text=True, bufsize=1)
    # Keep a small, redacted tail so pre-READY child failures are diagnosable.
    startup_tail = []
    deadline = time.time() + 600
    while time.time() < deadline:
        line = proc.stdout.readline()
        if line:
            safe_line = re.sub(r"(?i)(token|secret|key)(\s*[=:]\s*)\S+", r"\1\2<redacted>", line.strip())[:512]
            startup_tail.append(safe_line)
            del startup_tail[:-8]
        if not line:
            if proc.poll() is not None:
                print("[bridge] warm child exited before READY rc=%s tail=%s" % (proc.returncode, json.dumps(startup_tail, ensure_ascii=True)[:4096]), flush=True)
                return None, ckpt, threads, dtype
            continue
        if "[INFER_SERVER_READY]" in line:
            proc.agillm_model_identity = model_identity if _identity_inputs_unchanged(model_identity) else None
            return proc, ckpt, threads, dtype
    print("[bridge] warm child timed out before READY tail=%s" % json.dumps(startup_tail, ensure_ascii=True)[:4096], flush=True)
    try:
        proc.kill()
    except Exception:
        pass
    return None, ckpt, threads, dtype


def display_ckpt(path):
    name = os.path.basename(path or "")
    if name.endswith(".raw.pt"):
        meta = path + ".json"
        try:
            with open(meta) as f:
                source_name = (json.load(f).get("source_name") or "").strip()
            if source_name:
                return source_name
        except Exception:
            pass
    return name


def _retire(proc):
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.stdin.write('{"cmd":"quit"}\n')
        proc.stdin.flush()
    except Exception:
        pass
    try:
        proc.terminate()
    except Exception:
        pass


def _desired_concurrency():
    cfg = bridge_env()
    raw = (cfg.get("AGILLM_BRIDGE_CONCURRENCY") or os.environ.get("AGILLM_BRIDGE_CONCURRENCY") or "2").strip()
    try:
        n = int(raw)
    except Exception:
        n = 2
    return max(1, min(4, n))


def _int_cfg(cfg, name, default):
    try:
        return max(1, int((cfg.get(name) or "").strip()))
    except Exception:
        return default


def _worker_thread_plan(cfg=None):
    cfg = cfg or bridge_env()
    desired = _desired_concurrency()
    total = _int_cfg(cfg, "AGILLM_BRIDGE_THREADS", 14)
    explicit = (cfg.get("AGILLM_BRIDGE_THREADS_PER_CHILD") or "").strip()
    if explicit:
        per_child = _int_cfg(cfg, "AGILLM_BRIDGE_THREADS_PER_CHILD", max(1, total // desired))
        return [per_child] * desired
    primary = _int_cfg(cfg, "AGILLM_BRIDGE_PRIMARY_THREADS", total)
    overflow = _int_cfg(cfg, "AGILLM_BRIDGE_OVERFLOW_THREADS", max(1, total // desired))
    return [primary] + [overflow] * max(0, desired - 1)


def _planned_slots(live=None, plan=None):
    """Select exactly the configured thread multiset; surplus slots stay ineligible."""
    plan = list(plan or _worker_thread_plan())
    remaining = list(live if live is not None else slots)
    chosen = []
    for threads in plan:
        match = next((i for i, slot in enumerate(remaining)
                      if _slot_alive(slot) and int(slot.get("threads") or 0) == int(threads)), None)
        if match is not None:
            chosen.append(remaining.pop(match))
    return chosen


def _slot_alive(slot):
    proc = slot.get("proc") if slot else None
    return proc is not None and proc.poll() is None


def _cancel_epoch():
    with pool_lock:
        return cancel_epoch


def _request_canceled(epoch):
    with pool_lock:
        return epoch != cancel_epoch


def _release_canceled_slot(slot):
    try:
        slot["lock"].release()
    except Exception:
        pass


def _refresh_ready_locked():
    global ready
    ready = any(_slot_alive(s) for s in slots)
    return ready


def _make_slot(proc, ckpt, threads, dtype):
    global next_slot_id
    next_slot_id += 1
    return {"id": next_slot_id, "proc": proc, "ckpt": ckpt, "threads": int(threads),
            "model_identity": getattr(proc, "agillm_model_identity", None),
            "dtype": normalize_dtype(dtype, "fp32"),
            "lock": threading.Lock(), "pipe_needs_resync": False}


def _drop_slot(slot):
    global ready
    try:
        _retire(slot.get("proc"))
    except Exception:
        pass
    with pool_lock:
        if slot in slots:
            slots.remove(slot)
        _refresh_ready_locked()


def ensure_pool(ckpt_override=None):
    """Keep the primary full-thread slot plus overflow slots alive."""
    global loads, slots, warming
    with warm_lock:
        plan = _worker_thread_plan()
        with pool_lock:
            slots[:] = [s for s in slots if _slot_alive(s)]
            _refresh_ready_locked()
            available = {}
            for slot in slots:
                available[int(slot.get("threads") or 0)] = available.get(int(slot.get("threads") or 0), 0) + 1
            missing = []
            for threads in plan:
                count = available.get(int(threads), 0)
                if count:
                    available[int(threads)] = count - 1
                else:
                    missing.append(int(threads))
        # stop-readiness 20260924: a pool repair (after Stop or a dead child) is a
        # warm-up too. Advertise it on /health so clients wait for the replacement
        # instead of queueing blind behind a 1-2 minute model load.
        was_warming = warming
        if missing:
            warming = True
        try:
            for threads in missing:
                proc, ckpt, actual_threads, actual_dtype = spawn_child(threads, ckpt_override)
                if proc is None:
                    print("[bridge] warm slot FAILED threads=%s" % actual_threads, flush=True)
                    break
                with pool_lock:
                    slot = _make_slot(proc, ckpt, actual_threads, actual_dtype)
                    slots.append(slot)
                    loads += 1
                    _refresh_ready_locked()
                    print("[bridge] warmed slot=%s threads=%s dtype=%s ckpt=%s" % (slot["id"], slot["threads"], slot["dtype"], display_ckpt(ckpt)), flush=True)
        finally:
            if missing:
                warming = was_warming
        with pool_lock:
            return _refresh_ready_locked()


def _schedule_pool_repair():
    threading.Thread(
        target=ensure_pool,
        name="agillm-pool-repair",
        daemon=True,
    ).start()


def _queue_timeout_s():
    """Queue timeout, re-read from bridge.env on every call.

    A blue-green reload can leave the pool warming for longer than the old
    hard-coded 30s, which made every request that landed mid-reload fail with a
    queue timeout instead of simply waiting for the new slots. Reading it live
    keeps this knob consistent with the file contract above: bridge.env edits
    must not need a service restart.
    """
    try:
        raw = bridge_env().get("AGILLM_BRIDGE_QUEUE_TIMEOUT_S")
    except Exception:
        return QUEUE_TIMEOUT_S
    if raw is None or not str(raw).strip():
        return QUEUE_TIMEOUT_S
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return QUEUE_TIMEOUT_S
    return max(1.0, min(600.0, value))


def acquire_slot(max_wait_s=None):
    """Return (slot, queue_wait_s) after reserving one warm child.

    Prefer the full-thread primary slot when it is idle. Overflow slots are for
    concurrency, not the normal fast path.
    """
    if max_wait_s is None:
        max_wait_s = _queue_timeout_s()
    max_wait_s = max(0.1, float(max_wait_s))
    t0 = time.time()
    if not ensure_pool():
        return None, time.time() - t0
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        with pool_lock:
            live = _planned_slots([s for s in slots if _slot_alive(s)])
            _refresh_ready_locked()
            ordered = sorted(live, key=lambda s: (-int(s.get("threads") or 0), int(s.get("id") or 0)))
        for slot in ordered:
            if slot["lock"].acquire(blocking=False):
                if _slot_alive(slot):
                    return slot, time.time() - t0
                slot["lock"].release()
                _drop_slot(slot)
        time.sleep(0.05)
        ensure_pool()
    return None, time.time() - t0


SYNC_SENTINEL = "@@BRIDGE_SYNC@@"


def resync_child(proc, max_s=120.0):
    """Discard stale child output so the next response read belongs to the
    next request written.

    The child answers an invalid-JSON line with a JSONDecodeError error block;
    since it processes stdin strictly in order, everything before that block
    is a leftover from an abandoned/errored earlier request."""
    try:
        proc.stdin.write(SYNC_SENTINEL + "\n")
        proc.stdin.flush()
    except Exception:
        return False
    deadline = time.time() + max_s
    stale = 0
    seen_sentinel = False
    while time.time() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                return False
            continue
        s = line.rstrip("\n")
        if "[INFER_SERVER_ERROR]" in s and "JSONDecodeError" in s:
            seen_sentinel = True
            continue
        if seen_sentinel and "[INFER_SERVER_RESULT_END]" in s:
            if stale:
                print("[bridge] resync: discarded %d stale line(s) from child pipe" % stale, flush=True)
            return True
        stale += 1
    print("[bridge] resync TIMED OUT after %.0fs (%d lines)" % (max_s, stale), flush=True)
    return False


def _retire_slot_when_idle(slot, timeout_s=3600.0):
    """Drain an old blue-green slot without interrupting an accepted request."""
    lock = slot.get("lock") if slot else None
    proc = slot.get("proc") if slot else None
    if lock is None:
        _retire(proc)
        return
    acquired = False
    try:
        acquired = lock.acquire(timeout=max(0.0, float(timeout_s)))
        _retire(proc)
    finally:
        if acquired:
            try:
                lock.release()
            except Exception:
                pass


def _drain_old_pool(old):
    for slot in old:
        threading.Thread(
            target=_retire_slot_when_idle,
            args=(slot,),
            name="agillm-blue-green-drain-%s" % slot.get("id"),
            daemon=True,
        ).start()


def reload_child(cancel=False, started_event=None, start_state=None):
    """Reload the pool.

    Normal reloads are blue-green: warm the complete replacement pool first,
    atomically switch new requests to it, then drain old in-flight requests.
    Cancel deliberately remains eager because Stop is expected to interrupt work.
    """
    global slots, ready, loads, warming, cancel_epoch
    if not reload_lock.acquire(blocking=False):
        if start_state is not None:
            start_state.update({"accepted": False, "error": "reload already in progress"})
        if started_event is not None:
            started_event.set()
        return {"ok": False, "error": "reload already in progress"}
    warm_lock.acquire()
    warming = True
    try:
        with pool_lock:
            old = [slot for slot in slots if _slot_alive(slot)]
            active_ckpt = next((slot.get("ckpt") for slot in old if slot.get("ckpt")), None)

        if cancel:
            print("[bridge] cancel: retiring active pool and warming replacement", flush=True)
            with pool_lock:
                cancel_epoch += 1
                slots = []
                ready = False
                if start_state is not None:
                    start_state.update({
                        "accepted": True,
                        "checkpoint": display_ckpt(active_ckpt) if active_ckpt else "",
                    })
            if started_event is not None:
                started_event.set()
            for slot in old:
                _retire(slot.get("proc"))
            ok = ensure_pool(ckpt_override=active_ckpt)
            with pool_lock:
                live = [slot for slot in slots if _slot_alive(slot)]
                ckpt = display_ckpt(live[0]["ckpt"]) if live else ""
                concurrency = _desired_concurrency()
            if not ok or not live:
                print("[bridge] cancel FAILED; no replacement slots ready", flush=True)
                return {"ok": False, "error": "replacement pool failed to start"}
            print("[bridge] cancel complete ckpt=%s slots=%d/%d" % (
                ckpt, len(live), concurrency), flush=True)
            return {"ok": True, "ckpt": ckpt, "live_slots": len(live),
                    "concurrency": concurrency}

        if start_state is not None:
            start_state.update({"accepted": True, "checkpoint": "blue-green"})
        if started_event is not None:
            started_event.set()

        plan = _worker_thread_plan()
        replacement = []
        print("[bridge] reload: blue-green warming %d replacement slot(s)" % len(plan), flush=True)
        try:
            for threads in plan:
                proc, ckpt, actual_threads, actual_dtype = spawn_child(threads, None)
                if proc is None:
                    raise RuntimeError("replacement slot failed to warm")
                replacement.append(_make_slot(proc, ckpt, actual_threads, actual_dtype))
                print("[bridge] reload warmed candidate slot=%s threads=%s dtype=%s ckpt=%s" % (
                    replacement[-1]["id"], replacement[-1]["threads"],
                    replacement[-1]["dtype"], display_ckpt(ckpt)), flush=True)
        except Exception as exc:
            for slot in replacement:
                _retire(slot.get("proc"))
            print("[bridge] reload FAILED; keeping old pool: %s" % exc, flush=True)
            return {"ok": False, "error": "replacement pool failed to start"}

        with pool_lock:
            slots = replacement
            loads += len(replacement)
            _refresh_ready_locked()
            live = [slot for slot in slots if _slot_alive(slot)]
            ckpt = display_ckpt(live[0]["ckpt"]) if live else ""
            concurrency = _desired_concurrency()

        _drain_old_pool(old)
        print("[bridge] reload blue-green switch complete ckpt=%s slots=%d/%d; draining_old=%d" % (
            ckpt, len(live), concurrency, len(old)), flush=True)
        return {"ok": True, "ckpt": ckpt, "live_slots": len(live),
                "concurrency": concurrency, "draining_old": len(old)}
    finally:
        if started_event is not None:
            started_event.set()
        warming = False
        warm_lock.release()
        reload_lock.release()

def _sighup(_signum, _frame):
    threading.Thread(target=reload_child, daemon=True).start()


def resync_if_needed(slot):
    """Pay the sentinel cost only when this slot's previous read may have
    left child output behind."""
    if not slot.get("pipe_needs_resync"):
        return True
    if not resync_child(slot.get("proc")):
        _drop_slot(slot)
        return False
    slot["pipe_needs_resync"] = False
    return True


def _arm_request_watchdog(proc, slot_id):
    expired = threading.Event()
    finished = threading.Event()

    def expire():
        if finished.is_set() or proc.poll() is not None:
            return
        expired.set()
        print("[bridge] request watchdog expired slot=%s timeout_s=%.1f; retiring child" % (
            slot_id, REQUEST_TIMEOUT_S), flush=True)
        try:
            proc.kill()
        except Exception:
            pass

    timer = threading.Timer(REQUEST_TIMEOUT_S, expire)
    timer.daemon = True
    timer.start()
    return timer, expired, finished


def do_infer(req):
    request_epoch = _cancel_epoch()
    try:
        active_dtype = ensure_request_dtype(req)
        request_id = _request_identity(req)
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    registration = _register_pending_request(request_id)
    if registration != "ok":
        return {"ok": False,
                "error": "request canceled" if registration == "canceled" else "duplicate request_id",
                "request_id": request_id}
    slot, queue_wait = acquire_slot()
    if slot is None:
        _finish_request(request_id)
        return {"ok": False, "error": "no warm child slot available", "request_id": request_id,
                "queue_wait_s": round(queue_wait, 3)}
    if _request_canceled(request_epoch) or not _activate_request(request_id, slot):
        _release_canceled_slot(slot)
        _finish_request(request_id, slot)
        return {"ok": False, "error": "request canceled", "request_id": request_id,
                "queue_wait_s": round(queue_wait, 3)}
    request_watchdog = None
    request_expired = None
    request_finished = None
    try:
        proc = slot.get("proc")
        if not resync_if_needed(slot):
            return {"ok": False, "error": "child pipe resync failed", "slot": slot.get("id"), "queue_wait_s": round(queue_wait, 3)}
        payload = {k: v for k, v in req.items() if k in REQUEST_FIELDS}
        payload["dtype"] = active_dtype
        payload["infer_dtype"] = active_dtype
        payload["_dtype"] = active_dtype
        payload.setdefault("max_new", 64)
        _req_temp = payload.get("temperature", None)
        try:
            _req_temp_f = float(_req_temp) if _req_temp is not None else 0.0
        except (TypeError, ValueError):
            _req_temp_f = 0.0
        _wants_sample = (payload.get("greedy") is False) or (_req_temp_f > 0.0)
        if _wants_sample and "mode" not in payload:
            payload["mode"] = "ar"
        payload.setdefault("mode", "ar")
        _mode = str(payload.get("mode", "ar")).strip().lower()
        if _mode == "sat" and "greedy" not in payload and _req_temp is None:
            _wants_sample = True
        if _mode == "ar":
            payload.setdefault("sampler", "dblock")
            payload.setdefault("dblock_blocks", 14)
            payload.setdefault("dblock_infer_blocks", "0")
        if _mode == "nat":
            payload.setdefault("nat_passes", 1)
        if _mode == "sat":
            _sat_var = str(payload.get("var", False)).strip().lower() in {"1", "true", "yes", "on"}
            payload.setdefault("sampler", "dblock")
            payload.setdefault("dblock_blocks", 14)
            payload.setdefault("dblock_infer_blocks", "0")
            payload.setdefault("temperature", 1.1)
            payload.setdefault("top_p", 0.98)
            payload.setdefault("top_k", 512)
            payload.setdefault("min_p", 0.001)
            payload.setdefault("repetition_penalty", 1.2)
            payload.setdefault("presence_penalty", 0.2)
            payload.setdefault("frequency_penalty", 0.2)
            payload.setdefault("penalty_last_n", 192)
            payload.setdefault("sat_trace", True)
            payload.setdefault("sat_symbol_penalty", 4.0)
            payload.setdefault("sat_function_penalty", 1.0)
            payload.setdefault("sat_numeric_penalty", 0.5)
            payload.setdefault("sat_prompt_align_greedy", True)
            payload.setdefault("sat_realign_greedy", True)
            payload.setdefault("sat_second_greedy", False)
            payload.setdefault("sat_second_temperature_scale", 0.9)
            payload.setdefault("sat_ar_second_weight", 0.0)
            # h[:, -1] here is SAT-masked state, not an independent causal AR
            # verifier. Blending it adds a vocabulary projection and cross-couples
            # the two SAT positions, so production keeps the false blend disabled.
            payload["sat_ar_first_weight"] = payload["sat_ar_second_weight"] = 0.0
            if _sat_var:
                payload.setdefault("sat_ar_first_weight", 0.5)
                payload.setdefault("sat_gate_greedy", True)
                payload.setdefault("sat_gate_temperature", 1.0)
                payload.setdefault("sat_gate_stride2_bias", 0.0)
                payload.setdefault("sat_gate_stride1_threshold", 0.05)
                _rewrite_legacy_sat_var_gate(payload)
            else:
                payload.setdefault("sat_ar_first_weight", 0.0)
                payload.setdefault("sat_gate_greedy", True)
                payload.setdefault("sat_gate_temperature", 1.0)
                payload.setdefault("sat_gate_stride2_bias", 0.0)
                payload.setdefault("sat_gate_stride1_threshold", 0.05)
        payload.setdefault("output_row_norm_clip", 10.0)
        payload.setdefault("min_new", 1)
        payload.setdefault("greedy", not _wants_sample)
        if _mode == "nat":
            payload.setdefault("nat_greedy", payload.get("greedy") is not False)
        if _wants_sample:
            payload.setdefault("temperature", 0.7)
            payload.setdefault("top_p", 0.95)
            payload.setdefault("top_k", 40)
        payload.setdefault("repetition_penalty", 1.05)
        payload.setdefault("presence_penalty", 0.0)
        payload.setdefault("frequency_penalty", 0.0)
        payload.setdefault("penalty_last_n", 128)
        payload.setdefault("ignore_eos", False)
        t0 = time.time()
        request_watchdog, request_expired, request_finished = _arm_request_watchdog(proc, slot.get("id"))
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        lines, stat, nat_trace, sat_trace, started, completed = [], None, None, None, False, False
        deadline = time.time() + REQUEST_TIMEOUT_S
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    timed_out = bool(request_expired and request_expired.is_set())
                    slot["pipe_needs_resync"] = True
                    _drop_slot(slot)
                    _schedule_pool_repair()
                    return {"ok": False,
                            "error": "child request timed out" if timed_out else "child died mid-request",
                            "slot": slot.get("id"), "queue_wait_s": round(queue_wait, 3)}
                continue
            s = line.rstrip("\n")
            if "[INFER_SERVER_RESULT_START]" in s:
                started = True
                continue
            if "[INFER_SERVER_RESULT_END]" in s:
                completed = True
                if request_finished is not None:
                    request_finished.set()
                break
            if "[INFER_SERVER_ERROR]" in s:
                slot["pipe_needs_resync"] = True
                return {"ok": False, "error": s, "slot": slot.get("id"), "queue_wait_s": round(queue_wait, 3)}
            if started:
                if s.startswith("[nat-trace] "):
                    try:
                        nat_trace = json.loads(s[len("[nat-trace] "):])
                    except Exception:
                        nat_trace = None
                    continue
                if s.startswith("[sat-trace] "):
                    try:
                        sat_trace = json.loads(s[len("[sat-trace] "):])
                    except Exception:
                        sat_trace = None
                    continue
                m = STAT_RE.search(s)
                if m:
                    stat = {"seconds": float(m.group("sec")), "tokens": int(m.group("tok")),
                            "tok_s": float(m.group("tps"))}
                    if nat_trace is not None:
                        stat["nat_trace"] = nat_trace
                    if sat_trace is not None:
                        stat["sat_trace"] = sat_trace
                    continue
                if s.startswith("[infer]") or s.startswith("Generating"):
                    continue
                lines.append(s)
        if not completed:
            if request_expired is not None:
                request_expired.set()
            try:
                proc.kill()
            except Exception:
                pass
            slot["pipe_needs_resync"] = True
            _drop_slot(slot)
            _schedule_pool_repair()
            return {"ok": False, "error": "child request timed out", "slot": slot.get("id"), "queue_wait_s": round(queue_wait, 3)}
        slot["pipe_needs_resync"] = False
        wall = time.time() - t0
        return {"ok": True, "text": "\n".join(lines).strip(), "stats": stat,
                "wall_s": round(wall, 3),
                "button_tok_s": round((stat["tokens"] / wall), 2) if stat and wall > 0 else None,
                "ckpt": display_ckpt(slot.get("ckpt")), "host": HOST_LABEL,
                "dtype": normalize_dtype(slot.get("dtype"), active_dtype),
                "slot": slot.get("id"), "threads": slot.get("threads"),
                "request_id": request_id,
                "queue_wait_s": round(queue_wait, 3)}
    finally:
        _finish_request(request_id, slot)
        if request_finished is not None:
            request_finished.set()
        if request_watchdog is not None:
            request_watchdog.cancel()
        try:
            slot["lock"].release()
        except Exception:
            pass


def do_stream(req, emit):
    """Run one request and emit newline-JSON events as tokens/slots commit."""
    request_epoch = _cancel_epoch()
    try:
        active_dtype = ensure_request_dtype(req)
        request_id = _request_identity(req)
    except Exception as exc:
        emit({"event": "error", "ok": False, "error": str(exc)})
        return
    registration = _register_pending_request(request_id)
    if registration != "ok":
        emit({"event": "error", "ok": False,
              "error": "request canceled" if registration == "canceled" else "duplicate request_id",
              "request_id": request_id})
        return
    slot, queue_wait = acquire_slot()
    if slot is None:
        _finish_request(request_id)
        emit({"event": "error", "ok": False, "error": "no warm child slot available",
              "request_id": request_id, "queue_wait_s": round(queue_wait, 3)})
        return
    if _request_canceled(request_epoch) or not _activate_request(request_id, slot):
        _release_canceled_slot(slot)
        _finish_request(request_id, slot)
        emit({"event": "error", "ok": False, "error": "request canceled",
              "request_id": request_id, "queue_wait_s": round(queue_wait, 3)})
        return
    request_watchdog = None
    request_expired = None
    request_finished = None
    try:
        proc = slot.get("proc")
        if not resync_if_needed(slot):
            emit({"event": "error", "ok": False, "error": "child pipe resync failed", "slot": slot.get("id"), "queue_wait_s": round(queue_wait, 3)})
            return
        payload = {k: v for k, v in req.items() if k in REQUEST_FIELDS}
        payload["dtype"] = active_dtype
        payload["infer_dtype"] = active_dtype
        payload["_dtype"] = active_dtype
        payload.setdefault("max_new", 64)
        _req_temp = payload.get("temperature", None)
        try:
            _req_temp_f = float(_req_temp) if _req_temp is not None else 0.0
        except (TypeError, ValueError):
            _req_temp_f = 0.0
        _wants_sample = (payload.get("greedy") is False) or (_req_temp_f > 0.0)
        if _wants_sample and "mode" not in payload:
            payload["mode"] = "ar"
        payload.setdefault("mode", "ar")
        _mode = str(payload.get("mode", "ar")).strip().lower()
        if _mode == "sat" and "greedy" not in payload and _req_temp is None:
            _wants_sample = True
        if _mode == "ar":
            payload.setdefault("sampler", "dblock")
            payload.setdefault("dblock_blocks", 14)
            payload.setdefault("dblock_infer_blocks", "0")
        if _mode == "nat":
            payload.setdefault("nat_passes", 1)
        if _mode == "sat":
            _sat_var = str(payload.get("var", False)).strip().lower() in {"1", "true", "yes", "on"}
            payload.setdefault("sampler", "dblock")
            payload.setdefault("dblock_blocks", 14)
            payload.setdefault("dblock_infer_blocks", "0")
            payload.setdefault("temperature", 1.1)
            payload.setdefault("top_p", 0.98)
            payload.setdefault("top_k", 512)
            payload.setdefault("min_p", 0.001)
            payload.setdefault("repetition_penalty", 1.2)
            payload.setdefault("presence_penalty", 0.2)
            payload.setdefault("frequency_penalty", 0.2)
            payload.setdefault("penalty_last_n", 192)
            payload.setdefault("sat_trace", True)
            payload.setdefault("sat_symbol_penalty", 4.0)
            payload.setdefault("sat_function_penalty", 1.0)
            payload.setdefault("sat_numeric_penalty", 0.5)
            payload.setdefault("sat_prompt_align_greedy", True)
            payload.setdefault("sat_realign_greedy", True)
            payload.setdefault("sat_second_greedy", False)
            payload.setdefault("sat_second_temperature_scale", 0.9)
            payload.setdefault("sat_ar_second_weight", 0.0)
            # h[:, -1] here is SAT-masked state, not an independent causal AR
            # verifier. Blending it adds a vocabulary projection and cross-couples
            # the two SAT positions, so production keeps the false blend disabled.
            payload["sat_ar_first_weight"] = payload["sat_ar_second_weight"] = 0.0
            if _sat_var:
                payload.setdefault("sat_ar_first_weight", 0.5)
                payload.setdefault("sat_gate_greedy", True)
                payload.setdefault("sat_gate_temperature", 1.0)
                payload.setdefault("sat_gate_stride2_bias", 0.0)
                payload.setdefault("sat_gate_stride1_threshold", 0.05)
                _rewrite_legacy_sat_var_gate(payload)
            else:
                payload.setdefault("sat_ar_first_weight", 0.0)
                payload.setdefault("sat_gate_greedy", True)
                payload.setdefault("sat_gate_temperature", 1.0)
                payload.setdefault("sat_gate_stride2_bias", 0.0)
                payload.setdefault("sat_gate_stride1_threshold", 0.05)
        payload.setdefault("output_row_norm_clip", 10.0)
        payload.setdefault("min_new", 1)
        payload.setdefault("greedy", not _wants_sample)
        if _mode == "nat":
            payload.setdefault("nat_greedy", payload.get("greedy") is not False)
        if _wants_sample:
            payload.setdefault("temperature", 0.7)
            payload.setdefault("top_p", 0.95)
            payload.setdefault("top_k", 40)
        payload.setdefault("repetition_penalty", 1.05)
        payload.setdefault("presence_penalty", 0.0)
        payload.setdefault("frequency_penalty", 0.0)
        payload.setdefault("penalty_last_n", 128)
        payload.setdefault("ignore_eos", False)
        payload["stream"] = True
        emit({"event": "start", "ok": True, "ckpt": display_ckpt(slot.get("ckpt")), "host": HOST_LABEL, "slot": slot.get("id"),
              "threads": slot.get("threads"), "dtype": normalize_dtype(slot.get("dtype"), active_dtype),
              "request_id": request_id, "queue_wait_s": round(queue_wait, 3)})
        t0 = time.time()
        request_watchdog, request_expired, request_finished = _arm_request_watchdog(proc, slot.get("id"))
        proc.stdin.write(json.dumps(payload) + "\n")
        proc.stdin.flush()
        lines, stat, nat_trace, sat_trace, started, completed = [], None, None, None, False, False
        deadline = time.time() + REQUEST_TIMEOUT_S
        while time.time() < deadline:
            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    timed_out = bool(request_expired and request_expired.is_set())
                    slot["pipe_needs_resync"] = True
                    _drop_slot(slot)
                    _schedule_pool_repair()
                    emit({"event": "error", "ok": False,
                          "error": "child request timed out" if timed_out else "child died mid-request",
                          "slot": slot.get("id"), "queue_wait_s": round(queue_wait, 3)})
                    return
                continue
            s = line.rstrip("\n")
            if "[INFER_SERVER_RESULT_START]" in s:
                started = True
                continue
            if "[INFER_SERVER_RESULT_END]" in s:
                completed = True
                if request_finished is not None:
                    request_finished.set()
                break
            if "[INFER_SERVER_ERROR]" in s:
                slot["pipe_needs_resync"] = True
                emit({"event": "error", "ok": False, "error": s, "slot": slot.get("id"), "queue_wait_s": round(queue_wait, 3)})
                return
            if not started:
                continue
            if s.startswith("[STREAM_BEGIN] "):
                try:
                    emit({"event": "stream_begin", "data": json.loads(s[len("[STREAM_BEGIN] ") :])})
                except Exception:
                    pass
                continue
            if s.startswith("[STREAM_NAT] ") or s.startswith("[STREAM_AR] ") or s.startswith("[STREAM_SAT] "):
                try:
                    emit({"event": "stream_tok", "data": json.loads(s.split("] ", 1)[1])})
                except Exception:
                    pass
                continue
            if s.startswith("[nat-trace] "):
                try:
                    nat_trace = json.loads(s[len("[nat-trace] "):])
                    emit({"event": "nat_trace", "data": nat_trace})
                except Exception:
                    nat_trace = None
                continue
            if s.startswith("[sat-trace] "):
                try:
                    sat_trace = json.loads(s[len("[sat-trace] "):])
                    emit({"event": "sat_trace", "data": sat_trace})
                except Exception:
                    sat_trace = None
                continue
            m = STAT_RE.search(s)
            if m:
                stat = {"seconds": float(m.group("sec")), "tokens": int(m.group("tok")),
                        "tok_s": float(m.group("tps"))}
                if nat_trace is not None:
                    stat["nat_trace"] = nat_trace
                if sat_trace is not None:
                    stat["sat_trace"] = sat_trace
                emit({"event": "stats", "stats": stat})
                continue
            if s.startswith("[infer]") or s.startswith("Generating") or s.startswith("["):
                continue
            lines.append(s)
        if not completed:
            if request_expired is not None:
                request_expired.set()
            try:
                proc.kill()
            except Exception:
                pass
            slot["pipe_needs_resync"] = True
            _drop_slot(slot)
            _schedule_pool_repair()
            emit({"event": "error", "ok": False, "error": "child request timed out", "slot": slot.get("id"), "queue_wait_s": round(queue_wait, 3)})
            return
        slot["pipe_needs_resync"] = False
        wall = time.time() - t0
        done = {"event": "done", "ok": True, "text": "\n".join(lines).strip(), "stats": stat,
                "wall_s": round(wall, 3),
                "button_tok_s": round((stat["tokens"] / wall), 2) if stat and wall > 0 else None,
                "ckpt": display_ckpt(slot.get("ckpt")), "host": HOST_LABEL,
                "dtype": normalize_dtype(slot.get("dtype"), active_dtype),
                "slot": slot.get("id"), "threads": slot.get("threads"),
                "request_id": request_id,
                "queue_wait_s": round(queue_wait, 3)}
        emit(done)
    finally:
        _finish_request(request_id, slot)
        if request_finished is not None:
            request_finished.set()
        if request_watchdog is not None:
            request_watchdog.cancel()
        try:
            slot["lock"].release()
        except Exception:
            pass


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Client timed out or pressed Stop; keep the bridge process healthy.
            return

    def _authed(self):
        return not TOKEN or (self.headers.get("X-AGILLM-Token") or "").strip() == TOKEN

    def do_GET(self):
        if self.path == "/health":
            with pool_lock:
                all_live = [s for s in slots if _slot_alive(s)]
                live = _planned_slots(all_live)
                draining = [s for s in all_live if s not in live]
                busy = sum(1 for s in live if s["lock"].locked())
                ckpt = display_ckpt(live[0]["ckpt"]) if live else ""
                is_ready = ready and bool(live)
            active_count, pending_count = _request_counts()
            self._send(200, {"ok": True, "ready": is_ready, "warming": warming,
                             "ckpt": ckpt, "uptime_s": round(time.time() - started_at),
                             **_health_model_identity(live),
                             "loads": loads, "concurrency": _desired_concurrency(),
                             "request_timeout_s": REQUEST_TIMEOUT_S,
                             "queue_timeout_s": _queue_timeout_s(),
                             "dtype": effective_bridge_dtype(bridge_env().get("AGILLM_BRIDGE_DTYPE")),
                             "live_slots": len(live), "busy": busy,
                             "draining_slots": len(draining),
                             "active_request_count": active_count,
                             "pending_request_count": pending_count,
                             "slot_threads": [s.get("threads") for s in live],
                             "slot_dtypes": [s.get("dtype") for s in live],
                             "host": HOST_LABEL, "device": DEVICE_LABEL})
        else:
            self._send(404, {"ok": False})

    def do_POST(self):
        if self.path in {"/reload", "/cancel"}:
            if not self._authed():
                self._send(401, {"ok": False, "error": "bad or missing X-AGILLM-Token"})
                return
            if self.path == "/cancel":
                try:
                    n = int(self.headers.get("Content-Length", 0) or 0)
                    if n < 0 or n > 4096:
                        raise ValueError("cancel body too large")
                    body = json.loads(self.rfile.read(n) or b"{}")
                    request_id = body.get("request_id") if isinstance(body, dict) else None
                except Exception as exc:
                    self._send(400, {"ok": False, "error": str(exc)})
                    return
                code, result = cancel_request(request_id)
                self._send(code, result)
                return
            started = threading.Event()
            start_state = {}
            threading.Thread(
                target=reload_child,
                args=(False, started, start_state),
                daemon=True,
            ).start()
            if not started.wait(timeout=2.0):
                self._send(503, {"ok": False, "error": "reload worker did not start"})
                return
            if not start_state.get("accepted"):
                self._send(409, {"ok": False, "error": start_state.get("error", "reload already in progress")})
                return
            self._send(200, {"ok": True, "canceling": False, "reloading": True,
                             "checkpoint": start_state.get("checkpoint", ""),
                             "note": "poll /health: warming goes false and live_slots returns when the replacement pool is ready"})
            return
        if self.path not in {"/infer", "/stream"}:
            self._send(404, {"ok": False})
            return
        if not self._authed():
            self._send(401, {"ok": False, "error": "bad or missing X-AGILLM-Token"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as exc:
            self._send(400, {"ok": False, "error": str(exc)})
            return
        if self.path == "/stream":
            # Canonicalise the request id before any response bytes are sent.  If
            # this client disappears, the bridge can cancel exactly this request
            # instead of leaving a long generation monopolising the only warm slot.
            try:
                stream_request_id = _request_identity(req)
                req["request_id"] = stream_request_id
            except Exception as exc:
                self._send(400, {"ok": False, "error": str(exc)})
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            client_gone = [False]
            write_lock = threading.Lock()

            def emit(obj):
                if client_gone[0]:
                    return
                try:
                    payload = json.dumps(obj).encode() + b"\n"
                    with write_lock:
                        if client_gone[0]:
                            return
                        self.wfile.write(payload)
                        self.wfile.flush()
                except Exception:
                    # The old behaviour kept consuming the full generation to keep
                    # the child pipe in sync.  With concurrency=1 that made a phone
                    # reload leave a ghost 2048-token job blocking every new request.
                    # Retire only this request/slot instead; cancel_request kills its
                    # child and do_stream's existing cleanup drops/resyncs the slot.
                    client_gone[0] = True
                    code, result = cancel_request(stream_request_id)
                    print("[bridge] stream client disconnected; request=%s cancel=%s note=%s" %
                          (stream_request_id, code, result.get("note", result.get("error", ""))),
                          flush=True)

            # Some NAT passes can spend a long time in one model forward before
            # committing positions. Keep the NDJSON response active across proxies
            # without manufacturing tokens or changing the final result.
            heartbeat_stop = threading.Event()
            heartbeat_started = time.time()

            def heartbeat():
                while not heartbeat_stop.wait(STREAM_HEARTBEAT_S):
                    emit({
                        "event": "heartbeat",
                        "ok": True,
                        "elapsed_s": round(time.time() - heartbeat_started, 1),
                    })

            heartbeat_thread = threading.Thread(
                target=heartbeat,
                name="agillm-stream-heartbeat",
                daemon=True,
            )
            heartbeat_thread.start()
            try:
                do_stream(req, emit)
            finally:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=1.0)
            return
        out = do_infer(req)
        self._send(200 if out.get("ok") else 500, out)


def _warm_initial_pool():
    global warming
    warming = True
    print("[bridge] v4 warming initial pool in background", flush=True)
    try:
        ensure_pool()
        with pool_lock:
            live = [s for s in slots if _slot_alive(s)]
            ckpt = display_ckpt(live[0]["ckpt"]) if live else ""
        print("[bridge] ready=" + str(ready) + " ckpt=" + ckpt + "; slots=" + str(len(live)) + "/" + str(_desired_concurrency()) + "; threads=" + ",".join(str(s.get("threads")) for s in live) + "; dtypes=" + ",".join(str(s.get("dtype")) for s in live), flush=True)
    finally:
        warming = False


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, _sighup)
    print("[bridge] v4 listening on 0.0.0.0:" + str(PORT) + " while pool warms", flush=True)
    threading.Thread(target=_warm_initial_pool, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
