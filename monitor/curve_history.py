"""Full-lineage loss-curve history for the AGILLM UI (GETH side).

Added 2026-09-23 (claude-opus55-3dff5aa4, Scott's "fix this graph improve it").
The live /api/training feed only tails the last ~1,500 heartbeats, so the chart
could never show the RPV16 run as a whole.  Every INTERVAL seconds a daemon
thread streams curve_history_reduce.py to the GB10 box over the existing
verified SSH route (stdin, nothing is written there).  The reducer reads the
trainer logs once (~5 s, niced), keeps only the surviving branch and returns
bucketed per-objective summaries.  The result is cached on disk so a monitor
restart serves history immediately, and it is never fetched on a request path.

Read-only with respect to training: one `nice -n 15 python3 -` per refresh.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REDUCER = HERE / "curve_history_reduce.py"
CACHE = HERE / "state" / "curve_history_cache.json"
REFERENCE = Path("/opt/agillm43_netserve/rpv16_curve_reference.json")
MANUAL_HELDOUT = Path("/opt/agillm43_netserve/rpv16_heldout_points.json")
INTERVAL = int(os.environ.get("AGILLM_CURVE_HISTORY_INTERVAL", "300"))
RETRY = 120
TIMEOUT = 180

CONFIG = {
    "jobs": [
        {"name": "rpv16", "path": "/workspace/rpv16-keepalive-trainer.log", "format": "jsonl",
         "lineage": "GB10-51049010/rpv16-targetfix-v6", "label": "RPV16 1PF 2.0B"},
        {"name": "standard", "path": "/workspace/dual_training_20260920/standard_trainer.log",
         "format": "heartbeat", "lineage": "GB10-51049010/standard-1.1b", "label": "Standard 1.1B"},
    ],
    "max_rows": 1200,
    "min_bucket": 64,
    "recent_bucket": 64,
    "recent_span": 16384,
    "heldout_globs": ["/workspace/cowork_loss_diag_*/eval_*.json", "/workspace/rpv16_promote/eval_step*.json"],
}

_LOCK = threading.Lock()
_STATE: dict = {"doc": None, "attempted_at": 0.0, "ok_at": 0.0, "error": None,
                "running": False, "loaded_cache": False, "duration_s": None}


def _load_cache() -> None:
    if _STATE["loaded_cache"]:
        return
    _STATE["loaded_cache"] = True
    try:
        doc = json.loads(CACHE.read_text())
        if isinstance(doc, dict) and isinstance(doc.get("jobs"), dict):
            _STATE["doc"] = doc
            _STATE["ok_at"] = float(doc.get("fetched_at") or 0)
    except Exception:
        pass


def _write_cache(doc: dict) -> None:
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, separators=(",", ":")))
        os.replace(tmp, CACHE)
    except Exception:
        pass


def _validate(doc) -> str | None:
    if not isinstance(doc, dict) or doc.get("ok") is not True or not isinstance(doc.get("jobs"), dict):
        return "reducer returned no document"
    r = doc["jobs"].get("rpv16") or {}
    if r.get("ok") is not True:
        return "rpv16 reduction failed: %s" % (r.get("reason") or "unknown")
    cols = r.get("cols") or []
    rows = r.get("rows") or []
    if "s0" not in cols or not rows or any(len(x) != len(cols) for x in rows):
        return "rpv16 reduction has malformed rows"
    return None


def _refresh_worker(ssh_argv) -> None:
    t0 = time.time()
    err = None
    doc = None
    try:
        argv = ssh_argv()
        if not argv:
            err = "no verified GB10 SSH route"
        else:
            # Bound the run on the GB10 side too (remote limit < local TIMEOUT), so a
            # timed-out reduction never keeps running or overlaps the next one.
            remote = ("timeout -k 5 %d nice -n 15 python3 - " % (TIMEOUT - 30)) + shlex.quote(
                json.dumps(CONFIG, separators=(",", ":")))
            r = subprocess.run(argv + [remote], input=REDUCER.read_bytes(), capture_output=True,
                               timeout=TIMEOUT)
            out = r.stdout.decode("utf-8", "replace").strip()
            if r.returncode != 0 or not out:
                err = "reducer rc=%s %s" % (r.returncode, r.stderr.decode("utf-8", "replace").strip()[-300:])
            else:
                doc = json.loads(out.splitlines()[-1])
                err = _validate(doc)
    except subprocess.TimeoutExpired:
        err = "reducer timed out after %ss" % TIMEOUT
    except Exception as exc:
        err = "%s: %s" % (type(exc).__name__, exc)
    with _LOCK:
        _STATE["running"] = False
        _STATE["duration_s"] = round(time.time() - t0, 2)
        if err is None and doc is not None:
            doc["fetched_at"] = int(time.time())
            _STATE["doc"] = doc
            _STATE["ok_at"] = time.time()
            _STATE["error"] = None
        else:
            _STATE["error"] = err
    if err is None and doc is not None:
        _write_cache(doc)


def maybe_refresh(ssh_argv, force: bool = False) -> bool:
    """Start a background reduction when due.  Never blocks the caller."""
    now = time.time()
    with _LOCK:
        _load_cache()
        if _STATE["running"]:
            return False
        due = INTERVAL if (_STATE["error"] is None and _STATE["doc"] is not None) else RETRY
        if not force and _STATE["attempted_at"] and now - _STATE["attempted_at"] < due:
            return False
        _STATE["running"] = True
        _STATE["attempted_at"] = now
    try:
        threading.Thread(target=_refresh_worker, args=(ssh_argv,), daemon=True,
                         name="agillm-curve-history").start()
    except Exception as exc:
        with _LOCK:
            _STATE["running"] = False
            _STATE["error"] = "thread start failed: %s" % exc
        return False
    return True


def _json_file(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _heldout(doc: dict) -> list:
    pts = list(doc.get("heldout") or [])
    manual = _json_file(MANUAL_HELDOUT)
    if isinstance(manual, dict):
        manual = manual.get("points")
    if isinstance(manual, list):
        for p in manual:
            if isinstance(p, dict) and isinstance(p.get("step"), int) and isinstance(p.get("ce"), (int, float)):
                pts.append(dict(p, source_file=p.get("source_file") or str(MANUAL_HELDOUT)))
    seen, out = set(), []
    for p in sorted(pts, key=lambda p: (p["step"], p.get("mtime") or 0)):
        key = (p["step"], round(float(p["ce"]), 4), p.get("protocol"))
        if key in seen:
            continue
        seen.add(key)
        p = dict({k: v for k, v in p.items() if k != "source_file"}, source=os.path.basename(str(p.get("source_file") or "")))
        out.append(p)
    return out


def _tag_lineage(points: list, h: dict) -> list:
    """Mark each held-out point as on the current lineage or on an abandoned branch.

    A checkpoint evaluated before a rollback, at a step inside the abandoned range,
    belongs to that branch even if the current lineage later passes the same step.
    """
    last = (h.get("summary") or {}).get("last_step") or 0
    out = []
    for p in points:
        tag = "current" if p["step"] <= last else "unknown"
        for b in h.get("branches") or []:
            cut_ts = b.get("cut_ts")
            if b.get("from_step", 0) <= p["step"] <= b.get("to_step", -1) and (
                    cut_ts is None or not p.get("mtime") or p["mtime"] <= cut_ts + 60):
                tag = "abandoned"
                break
        out.append(dict(p, lineage=tag))
    return out


def snapshot() -> dict:
    """{'history': rpv16 doc, 'standard_history': standard doc} ready to attach, or {}."""
    with _LOCK:
        _load_cache()
        doc = _STATE["doc"]
        state = {"refreshed_at": int(_STATE["ok_at"]) or None, "attempted_at": int(_STATE["attempted_at"]) or None,
                 "error": _STATE["error"], "running": _STATE["running"], "duration_s": _STATE["duration_s"],
                 "interval_s": INTERVAL}
    if not isinstance(doc, dict):
        return {"history_state": state}
    now = time.time()
    state["age_s"] = int(now - state["refreshed_at"]) if state["refreshed_at"] else None
    out: dict = {"history_state": state}
    jobs = doc.get("jobs") or {}
    rp = jobs.get("rpv16")
    if isinstance(rp, dict) and rp.get("ok"):
        h = dict(rp)
        h["heldout"] = _tag_lineage(_heldout(doc), rp)
        ref = _json_file(REFERENCE)
        if isinstance(ref, dict):
            h["reference"] = ref
        h["state"] = state
        out["history"] = h
    sd = jobs.get("standard")
    if isinstance(sd, dict) and sd.get("ok"):
        s = dict(sd)
        s["state"] = state
        out["standard_history"] = s
    return out


def _col(h: dict, name: str):
    return (h.get("cols") or []).index(name)


def text_digest(h: dict, marks: int = 12) -> list:
    """Plain-text lines for the MCP agillm_curves tool."""
    if not isinstance(h, dict) or not h.get("rows"):
        return []
    C = {n: i for i, n in enumerate(h.get("cols") or [])}
    rows = h["rows"]
    sm = h.get("summary") or {}
    best = h.get("best") or {}
    ref = h.get("reference") or {}
    lines = ["RPV16 full-lineage training history (%s; %s buckets of %s steps; steps %s-%s; %s rows kept, %s abandoned by %s resumes):" % (
        h.get("lineage"), len(rows), h.get("bucket_steps"), sm.get("first_step"), sm.get("last_step"),
        sm.get("rows_surviving"), sm.get("rows_abandoned"), len(h.get("resumes") or []))]
    if ref.get("unigram_nats") is not None:
        lines.append("  reference: context-free unigram entropy %.4f nats (%s). AR CE above this is worse than ignoring context." % (
            ref["unigram_nats"], ref.get("dataset") or "trainer stream"))
    ba, bc = best.get("ar") or {}, best.get("anc") or {}
    if ba:
        lines.append("  best AR bucket %.4f @%s-%s; best full-stack anchor %s @%s-%s" % (
            ba["value"], ba["s0"], ba["s1"], bc.get("value"), bc.get("s0"), bc.get("s1")))
    lines.append("  AR mean last 10k steps %s (n=%s); 50k steps earlier %s; AR slope over last 100k %s nats/100k steps; anchor slope %s" % (
        sm.get("ar_mean_last_10k"), sm.get("ar_n_last_10k"), sm.get("ar_mean_50k_earlier"),
        sm.get("ar_slope_per_100k_last_100k"), sm.get("anc_slope_per_100k_last_100k")))
    step = max(1, len(rows) // marks)
    lines.append("  step range | AR | SAT | NAT | anchor | mixed p10-p90 | lr")
    for r in rows[::step] + ([rows[-1]] if (len(rows) - 1) % step else []):
        lines.append("  %s-%s | %s | %s | %s | %s | %s-%s | %s" % (
            r[C["s0"]], r[C["s1"]], r[C["ar"]], r[C["sat"]], r[C["nat"]], r[C["anc"]],
            r[C["lo"]], r[C["hi"]], r[C["lr"]]))
    for b in h.get("branches") or []:
        lines.append("  %s: AR last 10k %s; best AR %s" % (b.get("label"), b.get("ar_mean_last_10k"),
                     ((b.get("best") or {}).get("ar") or {}).get("value")))
    for p in h.get("heldout") or []:
        lines.append("  held-out CE %.4f @%s [%s lineage] (%s, %s tokens, %s)" % (p["ce"], p["step"], p.get("lineage"), p.get("protocol"), p.get("tokens"), p.get("ckpt") or p.get("source")))
    st = h.get("state") or {}
    lines.append("  history refreshed %ss ago (every %ss)%s" % (st.get("age_s"), st.get("interval_s"),
                 ("; last refresh error: %s" % st["error"]) if st.get("error") else ""))
    return lines
