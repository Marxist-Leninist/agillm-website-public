"""Live GB10 RPV16-1PF trainer feed for the Codex monitor (added 2026-09-20).

Stdlib only. Read-only with respect to the Vast box: tails the live JSONL
trainer log + checkpoint dir listing + GPU util over the caller's SSH
function (dependency-injected: no import of agillm_ext, no cycles).
Exposes a small cached snapshot the monitor attaches to /api/training
(gb10_live) and /api/checkpoint (rpv16_live). Never raises to callers.
"""
from __future__ import annotations

import json
import statistics
import threading
import time

KEEPALIVE_LOG = "/workspace/rpv16-keepalive-trainer.log"
CKPT_DIR = "/workspace/agillm-gb10-1pf-targetfix-active"
TAIL_LINES = 120
MIN_INTERVAL_S = 45
STALL_STEP_S = 3.0  # steps slower than this are checkpoint/anchor stalls: drop from medians
N_PARAMS = 1990038784  # served RPV16 parameter count (CPU /health)
PFLOP_TARGET = 1e15

_LOCK = threading.Lock()
_STATE = {"ok": False, "reason": "not polled yet", "polled_at": 0}
_LAST_ATTEMPT = 0.0

_REMOTE = (
    "tail -n %d -- %s 2>/dev/null; echo '<<S>>'; "
    "ls -la --time-style=+%%s %s/*.pt 2>/dev/null | head -n 8; echo '<<S>>'; "
    "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null"
) % (TAIL_LINES, KEEPALIVE_LOG, CKPT_DIR)


def _parse_pt_line(line: str):
    parts = line.split()
    if len(parts) < 6 or parts[0].startswith("total"):
        return None
    try:
        size = int(parts[4])
    except ValueError:
        return None
    return {"name": parts[-1].rsplit("/", 1)[-1], "size_bytes": size}


def poll(ssh_fn) -> dict:
    """One live fetch. ssh_fn(remote_cmd, timeout) -> stdout string."""
    out = {"ok": False, "polled_at": int(time.time())}
    try:
        raw = ssh_fn(_REMOTE, timeout=40)
    except Exception as exc:  # noqa: BLE001 - never break the monitor
        out["reason"] = "%s: %s" % (type(exc).__name__, exc)
        return out
    if not raw or not raw.strip():
        out["reason"] = "vast unreachable"
        return out
    parts = raw.split("<<S>>")
    trains, ckpts = [], []
    for line in (parts[0] if parts else "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("event") == "train":
            trains.append(row)
        elif row.get("event") in ("checkpoint", "checkpoint_async_started"):
            ckpts.append(row)
    if not trains:
        out["reason"] = "no train events in live log tail"
        return out
    fresh = [r for r in trains if (r.get("step_s") or 0) < STALL_STEP_S]
    pool = fresh or trains
    med_tok_s = statistics.median(r.get("tok_s") or 0 for r in pool)
    med_step_s = statistics.median(r.get("step_s") or 0 for r in pool)
    latest = trains[-1]
    # 6N dense-equivalent training FLOP/s estimate (standard convention).
    tflops = med_tok_s * 6.0 * N_PARAMS / 1e12
    pct = med_tok_s * 6.0 * N_PARAMS / PFLOP_TARGET * 100.0
    pts = []
    if len(parts) > 1:
        for line in parts[1].splitlines():
            p = _parse_pt_line(line)
            if p:
                pts.append(p)
    gpu_util = None
    if len(parts) > 2:
        try:
            gpu_util = float(parts[2].strip().splitlines()[0])
        except (ValueError, IndexError):
            gpu_util = None
    last_ckpt = ckpts[-1] if ckpts else {}
    out.update({
        "ok": True,
        "source": "gb10-rpv16-keepalive-live",
        "step": latest.get("step"),
        "seen_tokens": latest.get("seen_tokens"),
        "loss": latest.get("loss"),
        "objective": latest.get("objective"),
        "median_tok_s": round(med_tok_s, 1),
        "median_step_s": round(med_step_s, 4),
        "tail_rows": len(trains),
        "fresh_rows": len(fresh),
        "log_ts": latest.get("ts"),
        "gpu_util_pct": gpu_util,
        "checkpoints": pts[:8],
        "last_checkpoint_event": {k: last_ckpt.get(k) for k in (
            "event", "step", "stall_s", "save_s", "async", "path")},
        # PFLOP sidecar, same shape the UI already paints (paintPflopTarget).
        "compute_status": {
            "schema": "agillm.gb10.rpv16-pflop.v1",
            "generated_unix": time.time(),
            "target": {"petaflop_s": 1.0},
            "whole_step_useful_math": {
                "percent_of_1pflop_target": round(pct, 3),
                "tflops_s_est": round(tflops, 1),
            },
            "accounting": "6N dense-equivalent (N=1990038784); median tok/s over live tail, checkpoint-stall steps excluded; estimate, not GPU utilization",
        },
    })
    return out


def refresh(ssh_fn, force: bool = False):
    global _LAST_ATTEMPT
    now = time.time()
    with _LOCK:
        if not force and now - _LAST_ATTEMPT < MIN_INTERVAL_S:
            return dict(_STATE)
        _LAST_ATTEMPT = now
    try:
        snap = poll(ssh_fn)
    except Exception as exc:  # noqa: BLE001
        snap = {"ok": False, "reason": "%s: %s" % (type(exc).__name__, exc),
                "polled_at": int(time.time())}
    with _LOCK:
        if snap.get("ok") or not _STATE.get("ok"):
            _STATE.clear()
            _STATE.update(snap)
        else:
            _STATE["last_error"] = snap.get("reason")
            _STATE["last_error_at"] = snap.get("polled_at")
    return dict(_STATE)


def snapshot() -> dict:
    with _LOCK:
        return dict(_STATE)


def checkpoint_view() -> dict:
    with _LOCK:
        s = dict(_STATE)
    if not s.get("ok"):
        return {"ok": False, "reason": s.get("reason"), "polled_at": s.get("polled_at")}
    pts = s.get("checkpoints") or []
    latest_pt = next((p for p in pts if p["name"] == "latest.pt"), (pts[0] if pts else {}))
    return {
        "ok": True,
        "polled_at": s.get("polled_at"),
        "trainer_step": s.get("step"),
        "latest_pt": latest_pt.get("name"),
        "latest_pt_bytes": latest_pt.get("size_bytes"),
        "last_checkpoint_event": s.get("last_checkpoint_event"),
        "checkpoints_on_box": len(pts),
    }
