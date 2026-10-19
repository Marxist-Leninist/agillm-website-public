"""Live GB10 telemetry for the AGILLM UI (GETH side).

Added 2026-09-23 (claude-opus55-3dff5aa4) for Scott's "more live telemetry".  Every INTERVAL
seconds a daemon thread streams live_probe_remote.py to the GB10 box over the verified SSH route
(stdin; nothing is written there) and caches one compact document: GPU, host memory and disk,
both trainers' processes, the Standard heartbeat tail and LR override, RPV16 checkpoint/resume
events and the autopromote held-out series.  Never runs on a request path.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROBE = HERE / "live_probe_remote.py"
INTERVAL = 20
TIMEOUT = 45
TAIL_ROWS = 900

_LOCK = threading.Lock()
_STATE: dict = {"doc": None, "ok_at": 0.0, "attempted_at": 0.0, "error": None, "running": False,
                "std_prev": None, "std_rate": None}


def _worker(ssh_argv) -> None:
    err, doc = None, None
    try:
        argv = ssh_argv()
        if not argv:
            err = "no verified GB10 SSH route"
        else:
            r = subprocess.run(argv + ["nice -n 15 timeout -k 5 30 python3 -"], input=PROBE.read_bytes(),
                               capture_output=True, timeout=TIMEOUT)
            out = r.stdout.decode("utf-8", "replace").strip()
            if r.returncode != 0 or not out:
                err = "probe rc=%s %s" % (r.returncode, r.stderr.decode("utf-8", "replace").strip()[-200:])
            else:
                doc = json.loads(out.splitlines()[-1])
    except subprocess.TimeoutExpired:
        err = "probe timed out"
    except Exception as exc:
        err = "%s: %s" % (type(exc).__name__, exc)
    now = time.time()
    with _LOCK:
        _STATE["running"] = False
        if doc is None:
            _STATE["error"] = err
            return
        rows = ((doc.get("standard") or {}).get("rows") or [])
        if rows:
            seen = rows[-1][1]
            prev = _STATE.get("std_prev")
            if prev and seen > prev[0] and now > prev[1]:
                inst = (seen - prev[0]) / (now - prev[1])
                old = _STATE.get("std_rate")
                _STATE["std_rate"] = inst if old is None else 0.7 * old + 0.3 * inst
            if not prev or seen != prev[0]:
                _STATE["std_prev"] = (seen, now)
        _STATE["doc"] = doc
        _STATE["ok_at"] = now
        _STATE["error"] = None


def maybe_refresh(ssh_argv) -> bool:
    now = time.time()
    with _LOCK:
        if _STATE["running"] or now - _STATE["attempted_at"] < INTERVAL:
            return False
        _STATE["running"] = True
        _STATE["attempted_at"] = now
    try:
        threading.Thread(target=_worker, args=(ssh_argv,), daemon=True, name="agillm-live-telemetry").start()
    except Exception as exc:
        with _LOCK:
            _STATE["running"] = False
            _STATE["error"] = "thread start failed: %s" % exc
        return False
    return True


def _med(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(statistics.median(vals), 4) if vals else None


def snapshot() -> dict:
    with _LOCK:
        doc = _STATE["doc"]
        ok_at, err, rate = _STATE["ok_at"], _STATE["error"], _STATE["std_rate"]
    out = {"schema": "agillm.gb10.live-telemetry.v1", "interval_s": INTERVAL,
           "age_s": int(time.time() - ok_at) if ok_at else None, "error": err}
    if not isinstance(doc, dict):
        out["ok"] = False
        return out
    out["ok"] = True
    out["probe_ts"] = doc.get("ts")
    out["gpu"] = doc.get("gpu")
    out["host"] = doc.get("host")
    procs = doc.get("procs") or {}
    std = doc.get("standard") or {}
    rows = std.get("rows") or []
    tail = rows[-TAIL_ROWS:]
    last = rows[-1] if rows else None
    sp = (procs.get("standard") or [{}])[0]
    out["standard"] = {
        "pid": sp.get("pid"), "uptime_s": sp.get("uptime_s"), "source": sp.get("source"),
        "step": last[0] if last else None, "seen_tok": last[1] if last else None,
        "loss_last": last[2] if last else None, "lr": last[3] if last else None,
        "loss_median_last_50": _med([r[2] for r in rows[-50:]]),
        "loss_median_last_200": _med([r[2] for r in rows[-200:]]),
        "raw_tok_s_median_last_50": _med([r[4] for r in rows[-50:]]),
        "completed_tok_s": round(rate, 1) if rate else None,
        "log_age_s": int(time.time() - std["log_mtime"]) if std.get("log_mtime") else None,
        "tail": {"cols": std.get("cols"), "rows": tail},
        "val": {"cols": ["step", "ce"], "rows": std.get("val") or [],
                "meaning": "held-out CE from the trainer's own [val] pass (surviving lineage)"},
    }
    rp = (procs.get("rpv16") or [{}])[0]
    rv = doc.get("rpv16") or {}
    out["rpv16"] = {"pid": rp.get("pid"), "uptime_s": rp.get("uptime_s"), "source": rp.get("source"),
                    "last_checkpoint": rv.get("last_checkpoint"), "last_resume": rv.get("last_resume"),
                    "log_age_s": int(time.time() - rv["log_mtime"]) if rv.get("log_mtime") else None}
    ovr = doc.get("lr_override")
    lro = None
    if isinstance(ovr, dict) and ovr.get("lr"):
        seen = last[1] if last else None
        since, ramp = float(ovr.get("since_seen_tok") or 0), float(ovr.get("ramp_tokens") or 0)
        prog = min(1.0, max(0.0, (seen - since) / ramp)) if (seen and ramp > 0) else (1.0 if seen else None)
        lro = {"target_lr": ovr.get("lr"), "since_seen_tok": ovr.get("since_seen_tok"), "ramp_tokens": ovr.get("ramp_tokens"),
               "progress": round(prog, 4) if prog is not None else None, "effective_lr": last[3] if last else None,
               "by": ovr.get("by"), "for": ovr.get("for"), "created_utc": ovr.get("created_utc"), "revert": ovr.get("revert"),
               "anneal": ovr.get("anneal") if isinstance(ovr.get("anneal"), dict) else None, "anneal_by": ovr.get("anneal_by")}
    out["lr_override"] = lro
    pr = doc.get("promote") or {}
    out["heldout_live"] = {"source": "GETH autopromote (eval on each new RPV16 checkpoint, held-out CE, lower is better)",
                           "history": pr.get("history") or [], "served": pr.get("served"),
                           "last_evaluated_step": pr.get("last_evaluated_step")}
    return out
