"""Live Standard-1.1B trainer feed for the Codex monitor (added 2026-09-23, cowork).

Symmetric to gb10_live.py but for the Standard 1.1B DBlock trainer, whose log is
heartbeat TEXT lines (not JSONL). Stdlib only, read-only w.r.t. the Vast box: one
combined SSH tail of the trainer log + the hot-LR override + the safety-guard status +
nvidia-smi. Exposes a cached snapshot the monitor attaches to /api/training as
`standard_live`. Never raises to callers.
"""
from __future__ import annotations

import json
import re
import statistics
import threading
import time

LOG = "/workspace/dual_training_20260920/standard_trainer.log"
OVERRIDE = "/workspace/agillm43_lr_override.json"
GUARD_STATUS = "/workspace/std_lr_safety_guard_20260923/status.json"
TAIL_LINES = 200
MIN_INTERVAL_S = 45
N_PARAMS = 1_210_000_000  # agillm4_floor d1280 x 28L x 20h ~= 1.21B
PFLOP_TARGET = 1e15

_LOCK = threading.Lock()
_STATE = {"ok": False, "reason": "not polled yet", "polled_at": 0}
_LAST_ATTEMPT = 0.0

_REMOTE = (
    "tail -n %d -- %s 2>/dev/null; echo '<<S>>'; "
    "cat %s 2>/dev/null; echo '<<S>>'; "
    "cat %s 2>/dev/null; echo '<<S>>'; "
    "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null; echo '<<S>>'; "
    "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null; echo '<<S>>'; "
    "grep -a '^\\[val\\] step=' %s 2>/dev/null | tail -4"
) % (TAIL_LINES, LOG, OVERRIDE, GUARD_STATUS, LOG)

_HB = re.compile(
    r"\[heartbeat\] phase=(?P<phase>\S+) pid=(?P<pid>\d+) step=(?P<step>\d+) "
    r"seen_tok=(?P<seen>\d+) loss=(?P<loss>[-\d.eE]+) lr=(?P<lr>[-\d.eE+]+) "
    r"B=(?P<B>\d+) L=(?P<L>\d+) raw_tok_s=(?P<raw>[-\d.eE+]+) "
    r"fullstack_equiv_tok_s=(?P<fs>[-\d.eE+]+) layer_updates_s=(?P<lu>[-\d.eE+]+)"
)
_HB_GPU = re.compile(r"gpu_alloc=(?P<alloc>[\d.]+)GB gpu_reserved=(?P<res>[\d.]+)GB gpu_peak=(?P<peak>[\d.]+)GB")
_VAL = re.compile(r"\[val\] step=(?P<step>\d+) tokens=\d+ ce=(?P<ce>[\d.]+) ppl=(?P<ppl>[\d.]+)")


def _f(x):
    try:
        v = float(x)
        return v if v == v and v not in (float("inf"), float("-inf")) else None
    except (TypeError, ValueError):
        return None


def poll(ssh_fn) -> dict:
    out = {"ok": False, "polled_at": int(time.time())}
    try:
        raw = ssh_fn(_REMOTE, timeout=40)
    except Exception as exc:  # noqa: BLE001 - never break the monitor
        out["reason"] = "%s: %s" % (type(exc).__name__, exc)
        return out
    if not raw or not raw.strip():
        out["reason"] = "vast unreachable"
        return out
    seg = raw.split("<<S>>")
    logtxt = seg[0] if len(seg) > 0 else ""
    ov_txt = seg[1] if len(seg) > 1 else ""
    guard_txt = seg[2] if len(seg) > 2 else ""
    apps_txt = seg[3] if len(seg) > 3 else ""
    gpu_txt = seg[4] if len(seg) > 4 else ""
    val_txt = seg[5] if len(seg) > 5 else ""

    hbs, vals = [], []
    for line in logtxt.splitlines():
        m = _HB.search(line)
        if m:
            row = {k: m.group(k) for k in ("phase", "pid", "step", "seen", "loss", "lr", "B", "L", "raw", "fs", "lu")}
            g = _HB_GPU.search(line)
            if g:
                row["alloc"], row["res"], row["peak"] = g.group("alloc"), g.group("res"), g.group("peak")
            hbs.append(row)
            continue
        v = _VAL.search(line)
        if v:
            vals.append((int(v.group("step")), _f(v.group("ce")), _f(v.group("ppl"))))
    # dedicated [val] grep segment (heldout lines are sparse in the raw tail)
    for line in val_txt.splitlines():
        v = _VAL.search(line)
        if v:
            vals.append((int(v.group("step")), _f(v.group("ce")), _f(v.group("ppl"))))
    if vals:
        seen_steps = {}
        for s, c, p in vals:
            seen_steps[s] = (s, c, p)
        vals = [seen_steps[s] for s in sorted(seen_steps)]
    if not hbs:
        out["reason"] = "no heartbeat lines in standard log tail"
        return out
    latest = hbs[-1]
    raws = [_f(h.get("raw")) for h in hbs if _f(h.get("raw"))]
    fss = [_f(h.get("fs")) for h in hbs if _f(h.get("fs"))]
    lus = [_f(h.get("lu")) for h in hbs if _f(h.get("lu"))]
    med_raw = statistics.median(raws) if raws else None
    med_fs = statistics.median(fss) if fss else None
    med_lu = statistics.median(lus) if lus else None

    last_val = vals[-1] if vals else None
    # heldout CE min over the tail (best recent)
    ce_vals = [c for _s, c, _p in vals if c is not None]

    override = None
    if ov_txt.strip():
        try:
            override = json.loads(ov_txt)
        except ValueError:
            override = {"raw": ov_txt.strip()[:200]}
    guard = None
    if guard_txt.strip():
        try:
            guard = json.loads(guard_txt)
        except ValueError:
            guard = None

    # per-process GPU memory split (attribute this trainer's pid)
    apps = {}
    for line in apps_txt.splitlines():
        p = line.split(",")
        if len(p) == 2:
            try:
                apps[int(p[0].strip())] = int(p[1].strip())
            except ValueError:
                pass
    gpu_util = gpu_mem_used = gpu_mem_total = None
    if gpu_txt.strip():
        first = gpu_txt.strip().splitlines()[0].split(",")
        if len(first) >= 3:
            gpu_util, gpu_mem_used, gpu_mem_total = (_f(first[0]), _f(first[1]), _f(first[2]))

    pid = None
    try:
        pid = int(latest.get("pid"))
    except (TypeError, ValueError):
        pid = None
    this_mem = apps.get(pid) if pid is not None else None

    # 6N dense-equivalent full-model TFLOP/s from the trainer's own full-stack-equiv tok/s.
    tflops = pct = None
    if med_fs:
        tflops = med_fs * 6.0 * N_PARAMS / 1e12
        pct = med_fs * 6.0 * N_PARAMS / PFLOP_TARGET * 100.0

    lr_now = _f(latest.get("lr"))
    out.update({
        "ok": True,
        "source": "standard-1p1b-dblock-live",
        "pid": pid,
        "phase": latest.get("phase"),
        "step": int(latest["step"]),
        "seen_tokens": int(latest["seen"]),
        "loss": _f(latest.get("loss")),
        "lr": lr_now,
        "batch": int(latest["B"]),
        "block": int(latest["L"]),
        "raw_tok_s": round(med_raw, 1) if med_raw else None,
        "fullstack_equiv_tok_s": round(med_fs, 1) if med_fs else None,
        "layer_updates_s": round(med_lu, 1) if med_lu else None,
        "tail_rows": len(hbs),
        "gpu_alloc_gb": _f(latest.get("alloc")),
        "gpu_reserved_gb": _f(latest.get("res")),
        "gpu_peak_gb": _f(latest.get("peak")),
        "gpu_util_pct": gpu_util,
        "gpu_mem_used_mb": gpu_mem_used,
        "gpu_mem_total_mb": gpu_mem_total,
        "gpu_mem_this_trainer_mb": this_mem,
        "gpu_procs": len(apps),
        "heldout_ce": (last_val[1] if last_val else None),
        "heldout_ppl": (last_val[2] if last_val else None),
        "heldout_step": (last_val[0] if last_val else None),
        "heldout_ce_best_tail": (min(ce_vals) if ce_vals else None),
        "lr_override": override,
        "lr_guard": ({"state": ("REVERTED" if (guard or {}).get("reverted") else "watching"),
                      "override_present": (guard or {}).get("override_present"),
                      "base_val": (guard or {}).get("base_val"),
                      "last_decision": ((guard or {}).get("last_decision") or {}).get("kind"),
                      "utc": (guard or {}).get("utc")} if guard else None),
        "compute_status": {
            "schema": "agillm.standard.1p1b-flops.v1",
            "generated_unix": time.time(),
            "whole_step_useful_math": {
                "percent_of_1pflop_target": (round(pct, 3) if pct else None),
                "tflops_s_est": (round(tflops, 1) if tflops else None),
            },
            "accounting": "6N dense-equivalent (N=1.21B) from full-stack-equiv tok/s; estimate, not GPU utilization; shares the GPU with RPV16",
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
        snap = {"ok": False, "reason": "%s: %s" % (type(exc).__name__, exc), "polled_at": int(time.time())}
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
