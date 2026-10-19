#!/usr/bin/env python3
"""Training-curve + health feed for the AGILLM UI / MCP (codex.opentransformers.online).

Prints one JSON document (and optionally renders a PNG) with the run's loss/CE history on the
CURRENT lineage plus a heuristic health assessment per area (whole model, AR, SAT fixed,
SAT variable, NAT, DBlock local training, detachable-50M student, trainer/box).
  probes    frozen heldout_v3 CE from the 20-min CPU probe loop (w257 / w2048 / by position)
  val       the trainer's scheduled [val] on the same held-out set, incl. SAT shift-2 and NAT suffix CE
  train     heartbeat batch loss (EMA + bucket mean), effective LR, throughput (downsampled)
  markers   runtime hops / resumes; governor LR changes
  multimode full held-out AR / SAT shift-2 / NAT suffix-16/32/64 CE over ALL 48 windows (multimode_*.json):
            written every ~20 min by multimode_monitor.py on the same checkpoint as the AR probe (plus the
            hop-time points). This is the SAT/NAT quality signal the health heuristics alarm on; the trainer's
            own [val] SAT/NAT numbers are single-window crops (126 / 16 / 32 / 64 tokens) and are informational.
  timeline  (step, unix time) pairs from checkpoint receipts
  health    {overall, areas{...}, flags[...]} heuristics (see health())
Read-only; incremental parse cache so each call only reads the new log tail. Rollback forks are
cut on 'Resumed from step S' (points with step > S are dropped), so series are monotone in step.
"""
import json, os, re, sys, time, glob, math, statistics
from datetime import datetime, timezone
from trainer_progress_health_v1 import collect_progress
from dashboard_evidence_v2 import enrich as enrich_dashboard, VERSION as DASHBOARD_EVIDENCE_VERSION

LOG = "/workspace/agillm43_production_v14/production_quality_v18_hard_spike_gates/logs/singlefile_intelligence_v25_satvar_pretrain.log"
SAVE = "/workspace/agillm43_production_v14/production_quality_v18_hard_spike_gates/checkpoints_continue400B_from200B"
PROBE = "/workspace/agillm43_serving_probe"
CACHE = os.path.join(PROBE, "training_curves_cache.json")
OUT = os.path.join(PROBE, "training_curves.json")
PNG = os.path.join(PROBE, "training_curves.png")
LR_OVERRIDE = os.path.join("/workspace", "agillm43_lr_override" + ".json")   # read only
FIX_STEP = 2785394   # 2026-09-06 12:08Z rollback + fused-CE gradient fix

HB = re.compile(rb"^\[heartbeat\] phase=(\w+) pid=(\d+) step=(\d+) seen_tok=(\d+) loss=([0-9.eE+-]+) lr=([0-9.eE+-]+) .*?raw_tok_s=([0-9.]+).*?gpu_peak=([0-9.]+)GB")
ANCH = re.compile(rb"^\[dblock-anchor\] step=(\d+) raw_ce=([0-9.eE+-]+).*?baseline=([0-9.eE+-]+)")
VAL = re.compile(rb"^\[val\] step=(\d+) tokens=(\d+) ce=([0-9.]+) ppl=[0-9.]+ ?(\{.*\})?")
RES = re.compile(rb"^Resumed from step (\d+)")
LAUNCH = re.compile(rb'^\[singlefile-supervisor\] launch (\{.*\})')
TAG = re.compile(r"pretrain_step0*(\d+)_(.+?)_[0-9a-f]{16}\.pt$")
ROLL = 400
TRAIN_EMA_ALPHA = 0.03
TRAIN_SERIES_MAX_POINTS = 1500

def _cut(points, step, key="step"):
    while points and points[-1][key] > step:
        points.pop()

def fresh_cache():
    return {"schema": 5, "offset": 0, "inode": None, "hb": [], "val": [], "markers": [], "tail": "",
            "anchors": [], "receipts": [], "localdiag": [], "satvar": [], "since_resume": {"step": None, "tracebacks": 0, "spikes": 0,
            "det50_errors": 0, "det50_error_examples": []}, "det50_window": None, "det50_last": None}

def load_cache():
    try:
        with open(CACHE) as f:
            c = json.load(f)
        if c.get("schema") == 5:
            return c
    except Exception:
        pass
    return fresh_cache()

SATVAR_VB_TAGS = (b"[dblock-satvar-anchor] {", b"[dblock-satvar] {")
FRONTIER_LATEST = "/workspace/satvar_vb_20260912/frontier_latest.json"


def _satvar_vb_record(line, global_step):
    """One [dblock-satvar-anchor] / [dblock-satvar] JSON line -> compact telemetry record (or None)."""
    kind = "anchor" if line.startswith(b"[dblock-satvar-anchor]") else "local"
    d = json.loads(line[line.index(b"{"):].decode("utf-8", "replace"))
    rg = d.get("regret") or {}
    ce = d.get("raw") if kind == "anchor" else d.get("ce")
    return {"kind": kind, "step": int(global_step or 0), "dstep": int(d.get("step") or 0),
            "ce": (round(float(ce), 4) if ce is not None else None), "finite": bool(d.get("finite", True)),
            "ce_by_d": rg.get("ce_by_d") or {}, "pred_ce_by_d": rg.get("pred_ce_by_d") or {}, "mae": rg.get("regret_mae_log"),
            "d_hist": d.get("d_hist") or {}, "sizes": d.get("sizes") or {}, "updates": int(d.get("regret_updates") or 0),
            "regret_loss": d.get("regret_loss"), "targets": int((d.get("tokens") if kind == "anchor" else d.get("targets")) or 0)}


def _satvar_vb_backfill(path, upto):
    """First run after the v28 patch: scan the already-consumed log prefix for variable-block lines."""
    out = []; step = 0
    try:
        with open(path, "rb") as f:
            while f.tell() < upto:
                line = f.readline()
                if not line:
                    break
                if line.startswith(b"[heartbeat]"):
                    m = HB.match(line)
                    if m and m.group(1) == b"pretrain":
                        step = int(m.group(3))
                elif line.startswith(SATVAR_VB_TAGS):
                    try:
                        out.append(_satvar_vb_record(line.rstrip(b"\n"), step))
                    except Exception:
                        pass
    except OSError:
        pass
    return out[-400:]


def parse_log(c):
    st = os.stat(LOG)
    if c.get("inode") != st.st_ino or c.get("offset", 0) > st.st_size:
        c.update(fresh_cache()); c["inode"] = st.st_ino
    hb, val, markers = c["hb"], c["val"], c["markers"]
    receipts, localdiag, satvar = c["receipts"], c["localdiag"], c["satvar"]
    anchors = c.setdefault("anchors", [])
    sr = c["since_resume"]
    current_global_step = hb[-1]["step"] if hb else None
    if "satvar_vb" not in c:
        c["satvar_vb"] = _satvar_vb_backfill(LOG, int(c.get("offset", 0)))
    pending = c.get("tail", "").encode("utf-8", "surrogateescape")
    with open(LOG, "rb") as f:
        f.seek(c["offset"])
        while True:
            chunk = f.read(16 << 20)
            if not chunk:
                break
            data = pending + chunk
            lines = data.split(b"\n")
            pending = lines.pop()
            for line in lines:
                if line.startswith(b"[heartbeat]"):
                    m = HB.match(line)
                    if m and m.group(1) == b"pretrain":
                        current_global_step = int(m.group(3))
                        hb.append({"step": current_global_step, "tok": int(m.group(4)), "loss": float(m.group(5)), "lr": float(m.group(6)),
                                   "tps": float(m.group(7)), "peak": float(m.group(8))})
                elif line.startswith(b"[dblock-anchor]"):
                    m = ANCH.match(line)
                    if m:
                        anchors.append({"step": int(current_global_step or 0), "dblock_step": int(m.group(1)), "ce": float(m.group(2)), "baseline": float(m.group(3))})
                        if len(anchors) > 3000: del anchors[:len(anchors) - 3000]
                elif line.startswith(b"[dblock-target-route-receipt]"):
                    try:
                        d = json.loads(line[len(b"[dblock-target-route-receipt] "):].decode("utf-8", "replace"))
                        an = d.get("anchors") or {}
                        rec = {"step": int(d.get("committed_step_after") or 0), "committed": bool(d.get("committed")), "outcome": str(d.get("outcome") or "")}
                        for fam in ("ar", "sat", "nat"):
                            a = an.get(fam) or {}
                            ran_ = bool(a.get("ran"))
                            rec[fam] = [int(ran_), int(bool(a.get("accepted"))), int(bool(a.get("spike"))), int(ran_ and not a.get("finite", True))]
                            if a.get("spike"):
                                sr["spikes"] += 1
                        receipts.append(rec)
                        if len(receipts) > ROLL: del receipts[:len(receipts) - ROLL]
                    except Exception:
                        pass
                elif line.startswith(b"[dblock-local-diag]"):
                    try:
                        d = json.loads(line[len(b"[dblock-local-diag] "):].decode("utf-8", "replace"))
                        localdiag.append({"step": int(d.get("step") or 0), "block": int(d.get("block", -1)), "obj": str(d.get("objective") or ""), "ce": float(d.get("raw_average_ce") or 0.0)})
                        if len(localdiag) > ROLL: del localdiag[:len(localdiag) - ROLL]
                    except Exception:
                        pass
                elif line.startswith(b"[sat-variable-grad]"):
                    try:
                        d = json.loads(line[len(b"[sat-variable-grad] "):].decode("utf-8", "replace"))
                        satvar.append({"step": int(d.get("step") or 0), "gate": float(d.get("gate_preclip_norm") or 0.0), "main": float(d.get("main_preclip_norm") or 0.0)})
                        if len(satvar) > 300: del satvar[:len(satvar) - 300]
                    except Exception:
                        pass
                elif line.startswith(SATVAR_VB_TAGS):
                    try:
                        svb = c.setdefault("satvar_vb", [])
                        svb.append(_satvar_vb_record(line, current_global_step))
                        if len(svb) > 400: del svb[:len(svb) - 400]
                    except Exception:
                        pass
                elif line.startswith(b"[val] step="):
                    m = VAL.match(line)
                    if m:
                        rec = {"step": int(m.group(1)), "ce": float(m.group(3))}
                        if m.group(4):
                            try:
                                d = json.loads(m.group(4).decode("utf-8", "replace"))
                                for k in ("sat_shift2_ce", "sat_shift2_top1", "nat_suffix16_ce", "nat_suffix32_ce", "nat_suffix64_ce", "nat_suffix16_top1", "nat_suffix32_top1", "nat_suffix64_top1", "nat_clean_suffix_ce"):
                                    if k in d: rec[k] = round(float(d[k]), 4)
                            except Exception:
                                pass
                        val.append(rec)
                elif line.startswith(b"[detachable-50m] {"):
                    if line.startswith(b'[detachable-50m] {"error"'):
                        sr["det50_errors"] += 1
                        try:
                            d = json.loads(line[len(b"[detachable-50m] "):].decode("utf-8", "replace"))
                            ex = {"step": d.get("parent_step"), "mode": d.get("popup_mode"), "error": str(d.get("error"))[:120], "dblock_step_mod3": int(d.get("parent_dblock_step", -1)) % 3}
                            sr["det50_error_examples"] = (sr.get("det50_error_examples") or [])[-4:] + [ex]
                        except Exception:
                            pass
                    elif line.startswith(b'[detachable-50m] {"elapsed_ms"'):
                        try:
                            d = json.loads(line[len(b"[detachable-50m] "):].decode("utf-8", "replace"))
                            w = ((d.get("post_fix_qualification") or {}).get("window") or {})
                            c["det50_window"] = {"attempts": w.get("mode_attempts"), "commits": w.get("mode_commits"), "error_skips": w.get("mode_error_skips"), "oom_skips": w.get("mode_oom_skips"), "end_parent_step": w.get("end_parent_step")}
                            c["det50_last"] = {"step": d.get("parent_step"), "mode": d.get("popup_mode"), "outcome": d.get("outcome"), "student_ce": d.get("student_ce"), "top1_agreement": d.get("teacher_student_top1_agreement"), "kl": d.get("teacher_student_kl"),
                                                   "sat_gate": ((d.get("sat_gate_v27_13") or {}).get("online_summary") or {})}
                        except Exception:
                            pass
                elif line.startswith(b"Traceback (most recent call last)"):
                    sr["tracebacks"] += 1
                elif line.startswith(b"Resumed from step "):
                    m = RES.match(line)
                    if m:
                        s = int(m.group(1))
                        _cut(hb, s); _cut(val, s); _cut(markers, s); _cut(anchors, s)
                        current_global_step = s
                        c["satvar_vb"] = [r for r in c.get("satvar_vb", []) if int(r.get("step") or 0) <= s]
                        if not (markers and markers[-1]["step"] == s and markers[-1].get("kind") == "launch"):
                            markers.append({"step": s, "kind": "resume", "label": "resume"})
                        c["since_resume"] = sr = {"step": s, "tracebacks": 0, "spikes": 0, "det50_errors": 0, "det50_error_examples": []}
                elif line.startswith(b"[singlefile-supervisor] launch {"):
                    m = LAUNCH.match(line)
                    if m:
                        try:
                            d = json.loads(m.group(1).decode("utf-8", "replace"))
                            ck = os.path.basename(str(d.get("checkpoint", "")))
                            s = int(d.get("step", 0))
                            t = TAG.match(ck)
                            _cut(hb, s); _cut(val, s); _cut(markers, s); _cut(anchors, s)
                            current_global_step = s
                            c["satvar_vb"] = [r for r in c.get("satvar_vb", []) if int(r.get("step") or 0) <= s]
                            markers.append({"step": s, "kind": "launch", "label": t.group(2) if t else "resume", "checkpoint": ck, "run_id": d.get("run_id")})
                        except Exception:
                            pass
            c["offset"] = f.tell() - len(pending)
    c["tail"] = pending.decode("utf-8", "surrogateescape")
    return c

def ema_series(hb, alpha=TRAIN_EMA_ALPHA, max_points=TRAIN_SERIES_MAX_POINTS):
    if not hb: return []
    out = []; e = None
    for p in hb:
        e = p["loss"] if e is None else (1 - alpha) * e + alpha * p["loss"]
        out.append((p["step"], e, p["loss"], p["lr"], p["tps"], p["peak"], p["tok"]))
    n = len(out)
    if n <= max_points:
        return [{"step": s, "ema": round(e, 4), "loss": round(l, 4), "lr": lr, "tps": round(t, 1), "peak": pk, "tok": tk} for s, e, l, lr, t, pk, tk in out]
    size = n / max_points; res = []; i = 0.0
    while i < n:
        j = min(n, int(i + size)); seg = out[int(i):max(int(i) + 1, j)]
        res.append({"step": seg[-1][0], "ema": round(seg[-1][1], 4), "loss": round(sum(x[2] for x in seg) / len(seg), 4),
                    "lr": seg[-1][3], "tps": round(sum(x[4] for x in seg) / len(seg), 1), "peak": max(x[5] for x in seg), "tok": seg[-1][6]})
        i += size
    return res

def ts_epoch(s):
    try: return int(datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp())
    except Exception: return None

def jload(p, default=None):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return default

def probes():
    rows = []
    try:
        with open(os.path.join(PROBE, "heldout_monitor.jsonl")) as f:
            for line in f:
                try: d = json.loads(line)
                except Exception: continue
                if "w257" not in d: continue
                bp = d.get("w2048_by_pos") or {}
                rows.append({"step": int(d["step"]), "ts": d.get("ts"), "t": ts_epoch(d.get("ts", "")), "w257": d["w257"], "w2048": d.get("w2048"),
                             "p1": bp.get("p1-256"), "p2": bp.get("p257-1024"), "p3": bp.get("p1025-2047"), "checkpoint": d.get("checkpoint")})
    except FileNotFoundError:
        pass
    out = []
    for r in rows:
        while out and out[-1]["step"] >= r["step"]: out.pop()
        out.append(r)
    return out

def governor_events():
    ev = []
    try:
        with open(os.path.join(PROBE, "lr_governor_events.jsonl")) as f:
            for line in f:
                try: d = json.loads(line)
                except Exception: continue
                if d.get("kind") == "lr_change":
                    ev.append({"ts": d.get("ts"), "t": ts_epoch(d.get("ts", "")), "old_lr": d.get("old_lr"), "new_lr": d.get("new_lr"), "reason": d.get("reason")})
                elif d.get("kind") == "probe":
                    ev.append({"ts": d.get("ts"), "t": ts_epoch(d.get("ts", "")), "probe": True, "step": d.get("step"), "action": d.get("action"), "cooldown": d.get("cooldown"), "lr_target": d.get("lr_target"), "lr_cap": d.get("lr_cap"), "cap_blocked_streak": d.get("cap_blocked_streak")})
    except FileNotFoundError:
        pass
    return ev

def multimode(current=False):
    """Full held-out (48 x 257-token windows) AR / SAT shift-2 / NAT suffix CE points, chronological, lineage-cut
    like probes() (a later point with a lower-or-equal step pops the earlier ones). nat_pooled = token-weighted
    mean of the three suffix lengths (16+32+64 = 112 tokens per window, 5,376 tokens over 48 windows)."""
    rows = []
    for p in glob.glob(os.path.join(PROBE, "multimode_*.json")):
        if "mask1" in p: continue
        d = jload(p)
        if not d: continue
        is_current = d.get("comparable_to_legacy") is False
        if is_current != current: continue
        m = re.search(r"pretrain_step0*(\d+)_", os.path.basename(str(d.get("checkpoint", ""))))
        if not m: continue
        try: mt = os.stat(p).st_mtime
        except Exception: mt = 0.0
        t = ts_epoch(d.get("ts", "")) if d.get("ts") else mt
        n16, n32, n64 = d.get("nat_suffix16_ce"), d.get("nat_suffix32_ce"), d.get("nat_suffix64_ce")
        pooled = round((16.0 * n16 + 32.0 * n32 + 64.0 * n64) / 112.0, 4) if None not in (n16, n32, n64) else None
        rows.append({"step": int(m.group(1)), "t": t, "ts": d.get("ts") or datetime.fromtimestamp(mt, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                     "source": d.get("source") or "hop", "lineage_from": d.get("lineage_from"), "windows": d.get("windows"),
                     "eval_protocol": d.get("eval_protocol"), "protocol_sha256": d.get("protocol_sha256"),
                     "comparable_to_legacy": d.get("comparable_to_legacy"), "governor_may_consume": d.get("governor_may_consume"),
                     "snapshot_verification": d.get("snapshot_verification"),
                     "snapshot_verified": d.get("snapshot_verified"), "checkpoint_sha256": d.get("checkpoint_sha256"),
                     "ar": d.get("ar_ce"), "ar_top1": d.get("ar_top1"), "sat": d.get("sat_shift2_ce"), "sat_top1": d.get("sat_shift2_top1"),
                     "nat16": n16, "nat32": n32, "nat64": n64, "nat_pooled": pooled,
                     "nat16_top1": d.get("nat_suffix16_top1"), "nat32_top1": d.get("nat_suffix32_top1"), "nat64_top1": d.get("nat_suffix64_top1")})
    rows.sort(key=lambda r: (r["t"], r["step"])); out = []
    for r in rows:
        while out and out[-1]["step"] >= r["step"]: out.pop()
        out.append(r)
    return out

def timeline():
    rows = []
    for p in glob.glob(os.path.join(SAVE, "training_pointer_receipts", "step*.json")):
        d = jload(p)
        if not d: continue
        t = ts_epoch(str(d.get("saved_at", "")))
        if t: rows.append((t, int(d.get("step", 0))))
    rows.sort(); out = []
    for t, s in rows:
        while out and out[-1][0] >= s: out.pop()
        out.append([s, t])
    return out

# ----------------------------------------------------------------------------------------------- health
def _lvl(*levels):
    order = {"ok": 0, "maintenance": 0.5, "warn": 1, "bad": 2}
    return max(levels, key=lambda l: order.get(l, 0)) if levels else "ok"


def _mode_quality_evidence_v1(areas, now, flag):
    """Separate local update health from age-limited held-out evidence; no model calls."""
    stale = []
    for name, metric in (("ar", "ar_ce"), ("sat_fixed", "sat_shift2_ce"), ("nat", "pooled")):
        area = areas[name]
        heldout = area.get("full_heldout") or {}
        stamp = ts_epoch(str(heldout.get("ts") or ""))
        age = now - stamp if stamp else None
        value = heldout.get(metric)
        try:
            valid_value = value is not None and math.isfinite(float(value))
        except (ValueError, TypeError, OverflowError):
            valid_value = False
        present = bool(stamp and valid_value)
        fresh = bool(present and age is not None and 0 <= age <= 3000)
        status = ("fresh_full_heldout" if fresh else
                  "invalid_probe_time" if present and age < 0 else
                  "stale_last_known" if present else "missing_full_heldout")
        area["quality_fresh"] = fresh
        area["quality_evidence_status"] = status
        area["quality_age_min"] = round(age / 60.0, 1) if age is not None else None
        area["quality_level"] = heldout.get("level", "ok") if fresh else "maintenance"
        area["quality_note"] = ("Local anchor acceptance and gradient stability do not establish fresh held-out quality. "
                                "Coverage warnings do not assert model regression; freshness limit is 50 minutes.")
        if heldout:
            heldout["fresh"] = fresh
            heldout["evidence_status"] = status
            heldout["age_min"] = area["quality_age_min"]
            if not fresh:
                heldout["alarm_applied"] = False
        # Freshness is evidence coverage, not local training health. Keep the mode's
        # operational badge driven by its anchors/gradients; expose stale held-out
        # evidence separately via quality_level/quality_evidence_status.
        if not fresh:
            stale.append(name + (" %.0f min" % (age / 60.0) if present and age >= 0 else " " + status))
    if stale:
        flag("model", "maintenance", "Full held-out quality evidence is not fresh: " + "; ".join(stale) +
             ". Local training health is shown separately; this is stale coverage, not a regression verdict.")


def _satvar_clock_evidence_v1(area, svg, receipts, global_step):
    """Grad and receipt counters share DBlock state, not the global heartbeat clock."""
    grad_step = int(svg[-1]["step"]) if svg else None
    receipt_step = int(receipts[-1]["step"]) if receipts else None
    lag = receipt_step - grad_step if grad_step is not None and receipt_step is not None else None
    area.update({"last_dblock_step": grad_step,
                 "step_domain": "dblock_optimizer_state",
                 "latest_dblock_committed_step": receipt_step,
                 "gradient_lag_dblock_steps": lag,
                 "current_global_trainer_step": global_step,
                 "gradient_evidence_status": ("unavailable" if lag is None else
                      "counter_mismatch" if lag < 0 else "stale_dblock_updates" if lag > ROLL else "recent_dblock_updates"),
                 "counter_note": "last_step is retained as a legacy alias of last_dblock_step; both use the DBlock optimizer-state counter, NOT the global trainer step. Lag compares only the same counter domain.",
                 "evidence_scope": "Gradient stability and local gate diagnostics; not a held-out test of adaptive SAT decoding quality."})

def _satvar_variable_block_v28(areas, c, flag, global_step):
    """v28 variable-block SAT: the objective that makes SAT-variable CHOOSE its stride.

    Evidence: [dblock-satvar-anchor] (full-stack anchor every 3 DBlock steps, audited every 25 steps)
    and [dblock-satvar] (local SAT steps) JSON lines carrying per-distance slot CE, the regret head's
    predicted CE per distance and its update counter, plus the latest held-out speed/quality frontier
    written by satvar_frontier_eval.py.  Quality is the frontier (speedup vs CE), not a stride-2 threshold.
    """
    area = areas.setdefault("sat_variable", {"level": "ok"})
    recs = c.get("satvar_vb") or []
    anchors = [r for r in recs if r.get("kind") == "anchor"]
    loc = [r for r in recs if r.get("kind") == "local"]
    vb = {"schema": "agillm44.satvar.variable-block.v1", "anchors_seen": len(anchors), "local_audits_seen": len(loc)}
    if not recs:
        vb["status"] = "no [dblock-satvar*] lines in the parsed log (pre-v28 runtime or objective disabled)"
        vb["level"] = "warn"
        area["variable_block"] = vb
        area["level"] = _lvl(area.get("level", "ok"), "warn")
        flag("sat_variable", "warn", "variable-block SAT objective (v28) not observed in the parsed log")
        return
    last = recs[-1]
    lag = (int(global_step) - int(last["step"])) if (global_step is not None and last.get("step")) else None
    vb.update({"last_step": last.get("step"), "last_dblock_step": last.get("dstep"), "lag_steps": lag, "regret_updates": last.get("updates")})
    win = anchors[-20:]
    def med_by_d(key):
        out = {}
        for d in ("1", "2", "3", "4", "5", "6", "7", "8"):
            vals = [float(r[key][d]) for r in win if isinstance(r.get(key), dict) and d in r[key]]
            if vals: out[d] = round(statistics.median(vals), 3)
        return out
    maes = [float(r["mae"]) for r in win if r.get("mae") is not None]
    vb["anchor_window"] = {"n": len(win), "ce_by_d_median": med_by_d("ce_by_d"), "pred_ce_by_d_median": med_by_d("pred_ce_by_d"),
                           "regret_mae_log_median": (round(statistics.median(maes), 3) if maes else None),
                           "note": "median over the last %d audited anchors (256-token crops, 2 rows, random block sizes 1..4); d = tokens beyond the block" % len(win)}
    cbd = vb["anchor_window"]["ce_by_d_median"]; pbd = vb["anchor_window"]["pred_ce_by_d_median"]
    if cbd and pbd:
        vb["anchor_window"]["pred_minus_obs_by_d"] = {d: round(pbd[d] - cbd[d], 3) for d in cbd if d in pbd}
    if anchors:
        la = anchors[-1]
        vb["last_anchor"] = {k: la.get(k) for k in ("step", "ce", "ce_by_d", "pred_ce_by_d", "mae", "d_hist", "sizes", "targets", "finite")}
    if loc:
        ll = loc[-1]
        vb["last_local_sat_step"] = {k: ll.get(k) for k in ("step", "ce", "ce_by_d", "pred_ce_by_d", "d_hist", "sizes", "targets")}
    lv = "ok"
    if any(not r.get("finite", True) for r in recs[-20:]):
        lv = "bad"; flag("sat_variable", "bad", "variable-block SAT anchor produced a non-finite loss")
    elif lag is not None and lag > 1500:
        lv = "warn"; flag("sat_variable", "warn", "no variable-block SAT anchor/audit for %d trainer steps (expected every ~75)" % lag)
    fr = jload(FRONTIER_LATEST)
    if isinstance(fr, dict) and fr.get("per_lambda"):
        rows = []
        for r in fr["per_lambda"]:
            rows.append({"lambda": r.get("lambda"), "tokens_per_forward": r.get("tokens_per_forward"), "mean_stride": r.get("mean_stride"),
                         "mean_ce": r.get("mean_ce"), "stride_hist": r.get("stride_hist"),
                         "ce_by_d": {d: (v or {}).get("ce") for d, v in (r.get("ce_by_d") or {}).items()}})
        base = next((r for r in rows if not (r.get("lambda") or 0)), None)
        vb["frontier"] = {"checkpoint": os.path.basename(str(fr.get("checkpoint") or "")), "policy": fr.get("policy"),
                          "regret_updates_at_eval": fr.get("regret_updates"), "windows": fr.get("windows"), "seq": fr.get("seq"), "kmax": fr.get("kmax"),
                          "ar_baseline_ce": (base or {}).get("mean_ce"), "rows": rows,
                          "note": "teacher-forced held-out walk: the policy picks stride m* per position (lambda = nats traded per saved forward pass); tokens_per_forward is the speedup over AR, mean_ce its cost"}
        try: vb["frontier"]["age_min"] = round((time.time() - os.path.getmtime(FRONTIER_LATEST)) / 60.0, 1)
        except OSError: pass
    else:
        vb["frontier"] = {"status": "not yet measured (satvar_frontier_eval.py writes %s)" % FRONTIER_LATEST}
    vb["quality_note"] = ("v28 variable-block SAT: each SAT training step samples block sizes 1..4 and slot j predicts token j+1 beyond its block; "
                          "a detached regret head learns E[CE] per distance and serving picks the stride maximising lambda*(m-1) - regret. "
                          "Quality is the frontier (speedup vs CE), not a stride-2 threshold; the legacy gate fields are the satgate_v3 diagnostic.")
    vb["level"] = lv
    area["variable_block"] = vb
    area["level"] = _lvl(area.get("level", "ok"), lv)


def heldout_monitor_health(now, status=None, proc_root="/proc"):
    """Distinguish a running CPU probe from dead or deliberately paused telemetry."""
    ev = dict(status if status is not None else (jload(os.path.join(PROBE, "heldout_monitor_status.json"), {}) or {}))
    phase = str(ev.get("phase") or "unknown")
    pause_requested = os.path.exists(os.path.join(PROBE, "PAUSE_FOR_80K_TRAIN.txt"))
    try:
        pid = int(ev.get("pid") or 0)
        if pid <= 0: raise ValueError("missing PID")
        with open(os.path.join(proc_root, str(pid), "cmdline"), "rb") as f:
            args = f.read().split(b"\0")
        if not any(os.path.basename(arg.decode("utf-8", "replace")) == "heldout_monitor_v2.py" for arg in args):
            raise ValueError("PID identity mismatch")
        with open(os.path.join(proc_root, str(pid), "stat")) as f:
            state = f.read().rsplit(") ", 1)[1].split()[0]
        ev["process_state"] = state
        if state in ("T", "t"): phase = "suspended"
        elif state in ("Z", "X"): phase = "offline"
        elif phase == "running":
            started = ts_epoch(ev.get("started_utc", ""))
            if not started or now - started > 3660: phase = "stalled"
        elif phase == "paused" and pause_requested:
            pass
        elif phase != "unknown":
            updated = ts_epoch(ev.get("updated_utc", ""))
            if not updated or now - updated > 180: phase = "status_stale"
        ev["process_verified"] = state not in ("Z", "X")
    except (OSError, ValueError, IndexError, TypeError):
        phase = "paused" if pause_requested else ("offline" if ev.get("pid") else "unknown")
        ev["process_verified"] = False
    if pause_requested and phase in ("offline", "unknown", "status_stale", "suspended"):
        ev["reported_phase"] = phase
        phase = "paused"
    ev["pause_requested"] = bool(pause_requested)
    ev["phase"] = phase
    ev["checked_at"] = int(now)
    ev["cadence_note"] = "20-minute target; evaluates completed checkpoints; save fallback never sooner than 30 minutes"
    return ev

def multimode_monitor_health(proc_root="/proc"):
    """Observe existing CPU monitor/worker identities without signalling or spawning them."""
    monitors = []; workers = []
    try: entries = list(os.scandir(proc_root))
    except OSError: entries = []
    for entry in entries:
        if not entry.name.isdigit(): continue
        try:
            with open(os.path.join(entry.path, "cmdline"), "rb") as f:
                args = [x.decode("utf-8", "replace") for x in f.read().split(b"\0") if x]
            names = [os.path.basename(x) for x in args]
            role = "monitor" if "multimode_monitor.py" in names else "worker" if "multimode_probe.py" in names else None
            if role is None: continue
            with open(os.path.join(entry.path, "stat")) as f:
                fields = f.read().rsplit(") ", 1)[1].split()
            rec = {"pid": int(entry.name), "state": fields[0], "ppid": int(fields[1])}
            if rec["state"] in ("Z", "X"): continue
            if role == "worker":
                match = next((re.search(r"pretrain_step0*(\d+)_", os.path.basename(x)) for x in args if re.search(r"pretrain_step0*(\d+)_", os.path.basename(x))), None)
                rec["checkpoint_step"] = int(match.group(1)) if match else None
                workers.append(rec)
            else: monitors.append(rec)
        except (OSError, ValueError, IndexError): continue
    phase = "offline"
    if monitors:
        phase = "waiting_checkpoint"
        if len(monitors) > 1: phase = "duplicate_monitors"
        elif monitors[0]["state"] in ("T", "t"): phase = "suspended"
        elif workers:
            active = [x for x in workers if x["ppid"] == monitors[0]["pid"]]
            if active: phase = "suspended" if any(x["state"] in ("T", "t") for x in active) else "running"
    return {"phase": phase, "device": "cpu", "cpu_threads": 2,
            "monitor_pids": [x["pid"] for x in monitors], "worker_pids": [x["pid"] for x in workers],
            "checkpoint_step": next((x["checkpoint_step"] for x in workers if x.get("checkpoint_step") is not None), None)}

def health(c, P, V, hb, tl):
    now = time.time()
    rs = jload(os.path.join(SAVE, "run_state.json"), {}) or {}
    gov = jload(os.path.join(PROBE, "lr_governor_state.json"), {}) or {}
    ovr = jload(LR_OVERRIDE, {}) or {}
    flags = []; areas = {}
    def flag(area, level, msg, **kv):
        flags.append(dict(area=area, level=level, message=msg, **kv))
    try: log_age = now - os.stat(LOG).st_mtime
    except Exception: log_age = 1e9
    last = hb[-1] if hb else {}
    sr = c.get("since_resume") or {}
    runtime_progress = collect_progress(c, SAVE, rs, now, sr.get("step"))
    lv = runtime_progress["level"]
    flags.extend(runtime_progress["alerts"])
    if sr.get("tracebacks", 0) > 0: lv = "bad"; flag("trainer", "bad", "%d Python traceback(s) since the last resume" % sr["tracebacks"])
    try:
        stv = os.statvfs("/workspace"); free_gb = stv.f_bavail * stv.f_frsize / 1e9
    except Exception: free_gb = None
    if free_gb is not None:
        if free_gb < 14: lv = _lvl(lv, "bad"); flag("trainer", "bad", "disk %.1f GiB free (12 GiB is the trainer cleanup/pruning floor, not an automatic pause)" % free_gb)
        elif free_gb < 20: lv = _lvl(lv, "warn"); flag("trainer", "warn", "disk %.1f GB free" % free_gb)
    if last.get("peak", 0) > 22: lv = _lvl(lv, "warn"); flag("trainer", "warn", "GPU peak %.1f GB of 24" % last["peak"])
    # Current-run progress is independent of interrupted checkpoint intervals.
    areas["trainer"] = {"level": lv, "step": runtime_progress.get("step", last.get("step")),
        "pid": runtime_progress.get("pid"), "state": runtime_progress["state"],
        "lr": last.get("lr"), "raw_tok_s": last.get("tps"), "gpu_peak_gb": last.get("peak"),
        "log_age_s": int(log_age), "disk_free_gb": None if free_gb is None else round(free_gb, 1),
        "tracebacks_since_resume": sr.get("tracebacks", 0), "resume_step": sr.get("step"),
        "pace_last_interval": None,
        **{k: v for k, v in runtime_progress.items() if k not in ("alerts", "level", "step")}}
    lv = "ok"; best = min((p["w257"] for p in P), default=None); lastp = P[-1] if P else None
    evaluation = heldout_monitor_health(now)
    probe_age = max(0.0, now - lastp["t"]) if lastp and lastp.get("t") else None
    probe_fresh = probe_age is not None and probe_age <= 2400
    trend = None
    if len(P) >= 3:
        trend = round(P[-1]["w257"] - P[-3]["w257"], 4)
    # Model quality uses the same robust sustained-shift statistic as SAT/NAT below.
    # Comparing every fresh point with the single all-time minimum is a best-of-N bias: as N grows,
    # the minimum drifts downward even when the underlying model quality is stationary.  Alarm on the
    # median of the latest 3 probes versus the preceding 12-probe median, scaled by MAD-sigma.
    # Keep the historical minimum visible as a descriptive best, never as the alarm baseline.
    vals257 = [p["w257"] for p in P]
    over_streak = 0  # retained as descriptive telemetry for backward-compatible UI/details
    if best is not None:
        for p in reversed(P):
            if p["w257"] > best + 0.03: over_streak += 1
            else: break
    robust_model = None
    if len(vals257) >= 15:
        ref_window = vals257[-15:-3]
        robust_ref = statistics.median(ref_window)
        robust_sigma = 1.4826 * statistics.median([abs(v - robust_ref) for v in ref_window])
        robust_med3 = statistics.median(vals257[-3:])
        robust_delta = robust_med3 - robust_ref
        robust_warn_at = max(2.5 * robust_sigma, 0.05)
        robust_bad_at = max(5.0 * robust_sigma, 0.15)
        robust_level = "bad" if robust_delta > robust_bad_at else ("warn" if robust_delta > robust_warn_at else "ok")
        robust_model = {"median_last3": round(robust_med3, 4), "trailing_median": round(robust_ref, 4),
                        "sigma_mad": round(robust_sigma, 4), "delta_vs_trailing_median": round(robust_delta, 4),
                        "warn_at": round(robust_warn_at, 4), "bad_at": round(robust_bad_at, 4),
                        "level": robust_level, "window": "median(last 3) vs preceding 12",
                        "alarm_applied": bool(probe_fresh and robust_level != "ok")}
        if probe_fresh and robust_level != "ok":
            lv = _lvl(lv, robust_level)
            flag("model", robust_level,
                 "held-out CE sustained shift: median last 3 %.4f is +%.3f above trailing median %.4f "
                 "(MAD-sigma %.4f; %s threshold %.3f)" %
                 (robust_med3, robust_delta, robust_ref, robust_sigma, robust_level,
                  robust_bad_at if robust_level == "bad" else robust_warn_at))
    if lastp:
        if probe_age is not None and not probe_fresh:
            phase = evaluation.get("phase", "unknown")
            coverage_level = "maintenance" if phase in ("running", "waiting_checkpoint", "waiting_interval", "paused") else "warn"
            lv = _lvl(lv, coverage_level)
            if phase == "running":
                detail = "CPU evaluation running on checkpoint {:,} ({} cores)".format(int(evaluation.get("checkpoint_step") or 0), int(evaluation.get("cpu_threads") or 0))
            else:
                detail = {"waiting_checkpoint": "waiting for the next completed checkpoint", "waiting_interval": "CPU monitor active; awaiting next interval", "suspended": "quality monitor is suspended", "offline": "quality monitor is offline", "paused": "quality monitor explicitly paused", "failed": "latest quality evaluation failed", "stalled": "quality evaluation exceeded its deadline", "status_stale": "quality monitor status is stale"}.get(phase, "quality monitor status unavailable")
            flag("model", coverage_level, "last held-out probe is %d min old; %s" % (probe_age // 60, detail))
    ga = None
    for e in reversed(governor_events()):
        if e.get("probe"): ga = e; break
    areas["model"] = {"level": lv, "evaluation": evaluation, "fresh": probe_fresh, "evidence_status": "fresh" if probe_fresh else "stale_last_known", "probe_age_s": None if probe_age is None else round(probe_age, 1), "heldout_w257": lastp and lastp["w257"], "heldout_w2048": lastp and lastp.get("w2048"), "best_w257": best, "best_step": next((p["step"] for p in P if p["w257"] == best), None) if best is not None else None,
                      "last_probe_step": lastp and lastp["step"], "last_probe_ts": lastp and lastp.get("ts"), "trend_3probes": trend,
                      "probes_above_best": over_streak, "vs_best": (None if (lastp is None or best is None) else round(lastp["w257"] - best, 4)),
                      "robust_baseline": robust_model,
                      "noise_note": "alarm uses median(last 3) vs preceding 12 median with MAD-sigma; all-time best is display-only to avoid best-of-N bias", "trainer_val_ce": V[-1]["ce"] if V else None, "trainer_val_step": V[-1]["step"] if V else None,
                      "governor": {"lr": ovr.get("lr"), "cap": (ga and ga.get("lr_cap")) or (ovr.get("governor") or {}).get("cap"), "cap_source": "live probe event" if (ga and ga.get("lr_cap")) else "override file (may lag a cap change)", "cap_blocked_streak": ga and ga.get("cap_blocked_streak"), "cooldown": gov.get("cooldown"), "best": gov.get("best"), "last_action": ga and ga.get("action")}}
    rec = c.get("receipts") or []
    fam_last = {"ar": rs.get("fullstack_anchor_last") or {}, "sat": rs.get("fullstack_sat_anchor_last") or {}, "nat": rs.get("fullstack_nat_anchor_last") or {}}
    spike_consec = rs.get("fullstack_anchor_spike_consecutive_by_family") or {}
    hot_path = str((rs.get("dataset_provenance") or {}).get("hot_config_path") or "")
    hot_cfg = jload(hot_path, {}) if hot_path else {}
    if not isinstance(hot_cfg, dict): hot_cfg = {}
    def effective_anchor_every(fam):
        if fam == "sat" and not bool(rs.get("repair_mode", False)):
            return 3, "runtime_v25_sat_contract"
        key = "dblock_fullstack_%s_every" % fam
        raw = hot_cfg.get(key, rs.get(key, 3))
        try: every = max(0, int(raw))
        except (TypeError, ValueError): every = 3
        return every, ("hot_config" if key in hot_cfg else "run_state")
    def fam_stats(fam):
        ran = sum(r[fam][0] for r in rec); acc = sum(r[fam][1] for r in rec); spk = sum(r[fam][2] for r in rec); nonf = sum(r[fam][3] for r in rec)
        return ran, acc, spk, nonf
    for fam, name in (("ar", "ar"), ("sat", "sat_fixed"), ("nat", "nat")):
        ran, acc, spk, nonf = fam_stats(fam)
        lv = "ok"; n = len(rec) or 1
        expected_every, cadence_source = effective_anchor_every(fam)
        exp = (n / expected_every) if expected_every > 0 else 0.0
        fl = fam_last[fam]
        if rec:
            if expected_every > 0 and ran < 0.7 * exp: lv = "warn"; flag(name, "warn", "%s composition anchor ran on %d of the last %d steps (expected ~%d at every=%d)" % (fam.upper(), ran, n, int(exp), expected_every))
            if ran and acc < 0.9 * ran: lv = _lvl(lv, "warn"); flag(name, "warn", "%s anchor accepted %d of %d runs" % (fam.upper(), acc, ran))
            if spk: lv = _lvl(lv, "warn"); flag(name, "warn", "%d %s anchor spike(s) in the last %d steps" % (spk, fam.upper(), n))
            if nonf: lv = "bad"; flag(name, "bad", "%d non-finite %s anchor loss(es) in the last %d steps" % (nonf, fam.upper(), n))
        if int(spike_consec.get(fam, 0) or 0) >= 2: lv = "bad"; flag(name, "bad", "%s anchor spiking on consecutive steps (%d)" % (fam.upper(), spike_consec[fam]))
        if fl.get("spike"): lv = _lvl(lv, "warn"); flag(name, "warn", "last %s anchor flagged spike: %s" % (fam.upper(), fl.get("spike_reason")))
        areas[name] = {"level": lv, "anchor_training_level": lv, "anchor_ran": ran, "anchor_accepted": acc, "anchor_spikes": spk, "window_steps": len(rec), "anchor_expected_every": expected_every, "anchor_expected_runs": round(exp, 2), "anchor_cadence_source": cadence_source, "last_raw_ce": fl.get("raw_ce"), "spike_baseline": fl.get("spike_baseline"), "targets": fl.get("tokens"), "rows": fl.get("rows"), "spike_consecutive": spike_consec.get(fam)}
    ld = c.get("localdiag") or []
    for fam, name in (("ar", "ar"), ("sat", "sat_fixed"), ("nat", "nat")):
        vals = [x["ce"] for x in ld if x["obj"] == fam]
        if vals: areas[name]["local_objective_ce_median"] = round(statistics.median(vals[-60:]), 4)
    # ---- SAT fixed / NAT quality. Primary (alarm) signal: full held-out multimode probes, 48 windows every ~20 min.
    #      SAT shift-2 = 12,240 tokens, NAT pooled = 5,376 tokens per point (std err ~0.03-0.05 nats), so +0.10 is real.
    #      The trainer's own [val] SAT/NAT are single-window crops (126 / 16 / 32 / 64 tokens, std err ~0.3-0.6) -> informational,
    #      used as a fallback alarm (rolling median of 3) only while the robust signal is missing or stale.
    mm = multimode(); mm_age = ((now - mm[-1]["t"]) / 60.0) if mm else None
    robust_fresh = bool(mm) and mm_age is not None and mm_age <= 50
    lastp_age = ((now - P[-1]["t"]) / 60.0) if P else None
    def robust_check(name, key, label, tokens, warn_k=2.5, bad_k=5.0, warn_floor=0.05, bad_floor=0.15):
        """Flag a SUSTAINED level shift, measured against a robust reference.

        The earlier version compared the latest point to the MIN of the recent window. That reference is
        biased low by roughly the noise amplitude: measured over 18 probes since the v27.16.8 hop, a typical
        NAT point sits +0.104 above the min of its trailing 12 (max +0.168) while sitting 0.030 BELOW the
        median - so a +0.10 warn threshold fired about half the time, and "two in a row" then escalated a
        perfectly healthy series to bad (04:12Z 2026-09-07). Reference is now the median of the trailing
        window, scale is its MAD-sigma (NAT ~0.048, SAT ~0.032), and the statistic is the median of the last
        three probes, so one outlier - high or low - can neither raise an alarm nor move the baseline."""
        ser = [x for x in mm if x.get(key) is not None]
        if len(ser) < 6: return None
        vals = [x[key] for x in ser]
        window = vals[-15:-3] if len(vals) >= 15 else vals[:-3]
        if len(window) < 4: return None
        ref = statistics.median(window)
        sigma = 1.4826 * statistics.median([abs(v - ref) for v in window])
        med3 = statistics.median(vals[-3:])
        warn_at = max(warn_k * sigma, warn_floor); bad_at = max(bad_k * sigma, bad_floor)
        delta = med3 - ref
        lv = "bad" if delta > bad_at else ("warn" if delta > warn_at else "ok")
        # A historical shift is not a current alarm once the multimode telemetry is stale.
        # Keep the last-known statistics visible for diagnosis, but only let fresh full-heldout
        # probes affect the live health level. Otherwise a paused probe loop can leave a yellow/red
        # SAT badge stuck for hours after the model, LR, or checkpoint has moved on.
        if lv != "ok" and robust_fresh:
            areas[name]["level"] = _lvl(areas[name]["level"], lv)
            flag(name, lv, "full held-out %s CE has shifted up: median of the last 3 probes %.4f is %.3f above the trailing median %.4f (%.0f x sigma %.4f; 48 windows, %s tokens)" % (label, med3, delta, ref, (delta / sigma if sigma else 0), sigma, "{:,}".format(tokens)))
        return {"step": ser[-1]["step"], "ts": ser[-1].get("ts"), "age_min": round((now - ser[-1]["t"]) / 60.0, 1), "value": vals[-1],
                "median_last3": round(med3, 4), "trailing_median": round(ref, 4), "sigma_mad": round(sigma, 4),
                "delta_vs_trailing_median": round(delta, 4), "warn_at": round(warn_at, 3), "bad_at": round(bad_at, 3),
                "best_seen": round(min(vals), 4), "n_probes": len(ser), "tokens": tokens, "level": lv,
                "fresh": robust_fresh, "alarm_applied": bool(robust_fresh and lv != "ok"),
                "note": ("fresh alarm compares the median of the last 3 probes to the trailing median +/- MAD sigma; "
                         "when telemetry is stale the same statistics are last-known/display-only and do not affect live health") }
    r_sat = robust_check("sat_fixed", "sat", "SAT shift-2", 12240)
    r_nat = robust_check("nat", "nat_pooled", "NAT suffix (pooled 16/32/64)", 5376)
    if mm:
        m = mm[-1]
        w257_same = next((p["w257"] for p in P if p["step"] == m["step"]), None)
        areas["ar"]["full_heldout"] = {"step": m["step"], "ts": m.get("ts"), "ar_ce": m["ar"], "ar_top1": m["ar_top1"], "w257_probe_same_step": w257_same,
                                       "consistency_delta": (None if (w257_same is None or m["ar"] is None) else round(m["ar"] - w257_same, 4))}
        areas["sat_fixed"]["full_heldout"] = dict(r_sat or {}, sat_shift2_ce=m["sat"], top1=m["sat_top1"])
        areas["nat"]["full_heldout"] = dict(r_nat or {}, suffix16=m["nat16"], suffix32=m["nat32"], suffix64=m["nat64"], pooled=m.get("nat_pooled"))
        if (w257_same is not None and m["ar"] is not None and abs(m["ar"] - w257_same) > 0.02):
            areas["model"]["level"] = _lvl(areas["model"]["level"], "warn"); flag("model", "warn", "eval consistency: multimode AR CE %.4f vs ctxlen w257 %.4f at step %d differ by %.3f (same 48 windows; check the probe runtime)" % (m["ar"], w257_same, m["step"], m["ar"] - w257_same))
    mm_evaluation = multimode_monitor_health()
    for name in ("sat_fixed", "nat"):
        areas[name]["evaluation"] = mm_evaluation
    if mm and not robust_fresh and lastp_age is not None and lastp_age <= 45:
        for name in ("sat_fixed", "nat"):
            areas[name]["level"] = _lvl(areas[name]["level"], "warn")
        phase = mm_evaluation["phase"]
        if phase == "running":
            detail = "CPU evaluation running on checkpoint {:,} (2 cores)".format(int(mm_evaluation.get("checkpoint_step") or 0))
        else:
            detail = {"waiting_checkpoint": "monitor active, waiting for an eligible checkpoint", "suspended": "quality monitor or worker suspended", "offline": "quality monitor offline", "duplicate_monitors": "duplicate quality monitors detected"}.get(phase, "quality status unavailable")
        flag("nat", "warn", "full held-out SAT/NAT probe is stale (%d min old); %s" % (int(mm_age), detail))
    def rolling_fallback(name, series, key, label, tokens, thresh):
        vals = [v[key] for v in series]
        if len(vals) < 5: return
        meds = [statistics.median(vals[i - 2:i + 1]) for i in range(2, len(vals))]
        delta = meds[-1] - min(meds[:-1])
        if delta > thresh:
            areas[name]["level"] = _lvl(areas[name]["level"], "warn")
            flag(name, "warn", "fallback (full held-out probe unavailable): trainer-val %s rolling-median-of-3 %.3f is %.2f above its recent best (%d-token single-window crops)" % (label, meds[-1], delta, tokens))
    sv = [v for v in V if "sat_shift2_ce" in v][-12:]
    if sv:
        areas["sat_fixed"]["trainer_val_sat_shift2_ce"] = sv[-1]["sat_shift2_ce"]; areas["sat_fixed"]["trainer_val_step"] = sv[-1]["step"]
        areas["sat_fixed"]["trainer_val_note"] = "single 126-token window per [val]; informational only - full_heldout is the alarm signal"
        if not robust_fresh: rolling_fallback("sat_fixed", sv, "sat_shift2_ce", "SAT shift-2", 126, 0.4)
    nv = [v for v in V if "nat_suffix32_ce" in v][-12:]
    if nv:
        areas["nat"]["trainer_val_nat_suffix_ce"] = {"16": nv[-1].get("nat_suffix16_ce"), "32": nv[-1].get("nat_suffix32_ce"), "64": nv[-1].get("nat_suffix64_ce")}; areas["nat"]["trainer_val_step"] = nv[-1]["step"]
        areas["nat"]["trainer_val_note"] = "single 16/32/64-token windows per [val]; informational only - full_heldout is the alarm signal"
        pooled = [dict(v, nat_pooled=(16.0 * v["nat_suffix16_ce"] + 32.0 * v["nat_suffix32_ce"] + 64.0 * v["nat_suffix64_ce"]) / 112.0) for v in nv if None not in (v.get("nat_suffix16_ce"), v.get("nat_suffix32_ce"), v.get("nat_suffix64_ce"))]
        if not robust_fresh: rolling_fallback("nat", pooled, "nat_pooled", "NAT suffix (pooled 16/32/64)", 112, 0.5)
    nvc = int(rs.get("fullstack_nat_anchor_no_valid_crop_consecutive") or 0)
    if nvc >= 4: areas["nat"]["level"] = _lvl(areas["nat"]["level"], "warn"); flag("nat", "warn", "NAT anchor found no valid crop on %d consecutive steps" % nvc)
    areas["nat"]["mask_id"] = fam_last["nat"].get("mask_id")
    svg = c.get("satvar") or []; lv = "ok"
    if svg:
        gates = [x["gate"] for x in svg[-100:]]; mains = [x["main"] for x in svg[-100:]]
        gmed = statistics.median(gates); mmed = statistics.median(mains)
        # Pre-clip grad norms are heavy-tailed and BOTH are clipped (main 1.0, gate 0.25), so an isolated
        # spike changes no update: recent main is median ~13, p99 ~43, with ~0.15% of steps over 200, which
        # made "max > 200" fire on ~3% of health checks for nothing. Alarm on the sustained level, or on a
        # burst of extremes, not on one tail sample.
        n_main_hi = sum(1 for m in mains if m > 200)
        if any((not math.isfinite(g)) for g in gates) or any((not math.isfinite(m)) for m in mains):
            lv = "bad"; flag("sat_variable", "bad", "non-finite SAT-variable gradient")
        elif max(gates) > 50: lv = "warn"; flag("sat_variable", "warn", "SAT-variable gate pre-clip grad norm peaked at %.1f (clip 0.25)" % max(gates))
        if gmed > 25: lv = _lvl(lv, "warn"); flag("sat_variable", "warn", "SAT-variable gate pre-clip grad norm median %.1f is high for a sustained stretch (clip 0.25, normal ~8)" % gmed)
        if mmed > 60: lv = _lvl(lv, "warn"); flag("sat_variable", "warn", "main pre-clip grad norm median %.0f on SAT-variable steps is high for a sustained stretch (clip 1.0, normal ~13)" % mmed)
        elif n_main_hi >= 5: lv = _lvl(lv, "warn"); flag("sat_variable", "warn", "%d of the last %d SAT-variable steps had main pre-clip grad norm > 200 (normal ~0.15%%; clip 1.0)" % (n_main_hi, len(mains)))
        areas["sat_variable"] = {"level": lv, "gate_grad_norm_median": round(gmed, 4), "gate_grad_norm_max": round(max(gates), 3), "main_grad_norm_median": round(mmed, 2),
                                 "main_grad_norm_max": round(max(mains), 1), "main_over_200_in_window": n_main_hi, "samples": len(gates), "last_step": svg[-1]["step"],
                                 "noise_note": "both norms are clipped (main 1.0, gate 0.25) and heavy-tailed; single spikes are absorbed - the alarm is on the median or a burst of 5+ over 200"}
    else:
        areas["sat_variable"] = {"level": "warn", "note": "no [sat-variable-grad] lines seen"}; flag("sat_variable", "warn", "no SAT-variable gate updates observed in the parsed log")
    sgs = ((fam_last["sat"].get("sat_variable_gate") or {}) if isinstance(fam_last["sat"].get("sat_variable_gate"), dict) else {})
    if sgs:
        gate_keys = ("applied", "enabled", "required", "blocks", "stride2_labels", "predicted_stride2",
                     "confusion", "correct", "loss", "mean_probability_stride2",
                     "max_probability_stride2", "serving_stride2_threshold")
        areas["sat_variable"]["last_anchor_gate"] = {k: sgs.get(k) for k in gate_keys if k in sgs}
        areas["sat_variable"]["routing_scope"] = "native_training_gate_diagnostic_only"
        try:
            n_blocks = int(sgs.get("blocks") or 0)
            n_stride2 = int(sgs.get("stride2_labels") or 0)
            n_pred2 = int(sgs.get("predicted_stride2") or 0)
            threshold = float(sgs.get("serving_stride2_threshold")) if sgs.get("serving_stride2_threshold") is not None else None
            max_p2 = float(sgs.get("max_probability_stride2")) if sgs.get("max_probability_stride2") is not None else None
            if n_stride2 > 0 and n_pred2 == 0:
                # The checkpoint's native gate stays a training diagnostic. The deployed SAT-V
                # serving path (runtime_satgate_v3) is gate-decided at its own serving threshold with
                # no verifier; dashboard_evidence_v2 reports it from its live receipt below.
                areas["sat_variable"]["level"] = _lvl(areas["sat_variable"]["level"], "maintenance")
                areas["sat_variable"]["routing_readiness"] = "native_training_gate_fail_closed"
                areas["sat_variable"]["routing_scope"] = "native_training_gate_diagnostic_only"
                areas["sat_variable"]["routing_level"] = "maintenance"
                areas["sat_variable"]["native_training_gate_note"] = (
                     "Legacy checkpoint-native gate: %d stride-2 labels across %d blocks, 0 predictions at threshold %s%s. Diagnostic only; the deployed SAT-V serving path (runtime_satgate_v3) is gate-decided at its own serving threshold with no verifier and is reported separately." %
                     (n_stride2, n_blocks, ("%.3f" % threshold if threshold is not None else "unknown"),
                      ("; max gate score=%.3f" % max_p2 if max_p2 is not None else "")))
            elif n_stride2 > 0 and n_pred2 > 0:
                areas["sat_variable"]["routing_readiness"] = "two_class_predictions_seen"
                areas["sat_variable"]["routing_level"] = "ok"
            else:
                areas["sat_variable"]["routing_readiness"] = "native_training_gate_no_stride2_prediction_latest_anchor" if n_pred2 == 0 else "native_training_gate_latest_anchor_has_no_stride2_labels"
                areas["sat_variable"]["routing_level"] = "maintenance" if n_pred2 == 0 else "ok"
        except (TypeError, ValueError, OverflowError):
            areas["sat_variable"]["routing_readiness"] = "gate_diagnostics_unparseable"
    _satvar_clock_evidence_v1(areas["sat_variable"], svg, rec, areas["trainer"].get("step"))
    _satvar_variable_block_v28(areas, c, flag, areas["trainer"].get("step"))
    if areas["sat_variable"]["gradient_evidence_status"] in ("counter_mismatch", "stale_dblock_updates"):
        areas["sat_variable"]["level"] = _lvl(areas["sat_variable"]["level"], "warn")
        flag("sat_variable", "warn", "SAT-variable gradient counter evidence: %s (lag %s DBlock steps)" % (areas["sat_variable"]["gradient_evidence_status"], areas["sat_variable"]["gradient_lag_dblock_steps"]))
    lv = "ok"; lsp = rs.get("local_spike_consecutive_by_objective_block") or {}
    worst = 0
    for fam, arr in lsp.items():
        for b, v in enumerate(arr or []):
            if v and v > worst: worst = v
            if v >= 3: lv = "bad"; flag("dblock", "bad", "block %d %s local objective spiking on %d consecutive attempts" % (b, fam, v))
            elif v >= 1: lv = _lvl(lv, "warn"); flag("dblock", "warn", "block %d %s local objective spiked (%d)" % (b, fam, v))
    if rs.get("dblock_attempt_rejected"): lv = _lvl(lv, "warn"); flag("dblock", "warn", "last DBlock attempt was rejected")
    cb = (rs.get("committed_objective_steps_by_block") or {}).get("ar") or []
    skew = None
    if cb and min(cb) > 0:
        skew = max(cb) / min(cb)
        if skew > 1.35: lv = _lvl(lv, "warn"); flag("dblock", "warn", "block update-count skew %.2f (policy max 1.35)" % skew)
    per_block = {}
    for x in ld:
        if x["obj"] == "ar": per_block.setdefault(x["block"], []).append(x["ce"])
    pb = {str(b): round(statistics.median(v[-8:]), 3) for b, v in sorted(per_block.items())}
    if len(pb) >= 6:
        med = statistics.median(pb.values())
        lag = [b for b, v in pb.items() if v > 1.5 * med]
        if lag: lv = _lvl(lv, "warn"); flag("dblock", "warn", "block(s) %s local AR CE > 1.5x the median (%.2f)" % (",".join(lag), med))
    committed_rate = (sum(1 for r in rec if r["committed"]) / len(rec)) if rec else None
    if committed_rate is not None and committed_rate < 0.95: lv = _lvl(lv, "warn"); flag("dblock", "warn", "only %.0f%% of the last %d DBlock steps committed" % (100 * committed_rate, len(rec)))
    areas["dblock"] = {"level": lv, "committed_rate": committed_rate and round(committed_rate, 3), "update_count_skew_ar": skew and round(skew, 3), "local_ar_ce_by_block": pb, "last_train": rs.get("dblock_last_train_sublayers"), "objective_mode": "committed_9_1_1", "worst_local_spike_streak": worst, "clean_context": "e2e"}
    lv = "ok"; w = c.get("det50_window") or {}; errs = sr.get("det50_errors", 0)
    att = (w.get("attempts") or {}); com = (w.get("commits") or {})
    if errs:
        lv = "warn"; flag("det50", "warn", "%d student error-skip(s) since the last resume (e.g. %s)" % (errs, (sr.get("det50_error_examples") or [{}])[-1].get("error", "?")))
    if att and com:
        rate = sum(com.values()) / max(1, sum(att.values()))
        if rate < 0.95: lv = _lvl(lv, "warn"); flag("det50", "warn", "student lifetime commit rate %.1f%%" % (100 * rate))
    if w.get("end_parent_step") and last.get("step") and last["step"] - int(w["end_parent_step"]) > 60: lv = _lvl(lv, "warn"); flag("det50", "warn", "no student commit logged for %d steps" % (last["step"] - int(w["end_parent_step"])))
    detlast = c.get("det50_last") or {}
    detgate = detlast.get("sat_gate") or {}
    detrouting = "unknown"
    detrouting_level = "maintenance"
    if detgate:
        if detgate.get("online_sanity_passed") is False or detgate.get("heldout_qualification") is False:
            detrouting = str(detgate.get("routing_release_state") or "not-qualified")
            detrouting_level = "warn"
        else:
            detrouting = str(detgate.get("routing_release_state") or "qualified")
            detrouting_level = "ok"
    areas["det50"] = {"level": lv, "errors_since_resume": errs, "window": w, "last": detlast,
                      "sat_gate_routing_status": detrouting, "sat_gate_routing_level": detrouting_level,
                      "sat_gate_online": detgate}
    _mode_quality_evidence_v1(areas, now, flag)
    enrich_dashboard(areas, c, rs, V, hb, now, PROBE, LOG, flag)
    from live_health_repair import enrich as enrich_offhost_live
    enrich_offhost_live(areas, flags, now, PROBE, flag)
    overall = _lvl(*[a["level"] for a in areas.values()])
    order = {"bad": 0, "warn": 1, "maintenance": 2, "ok": 3}
    flags.sort(key=lambda f: order.get(f["level"], 3))
    summary = ("ALL FINE" if overall == "ok" else ("BENCHMARK" if overall == "maintenance" else ("WARNING" if overall == "warn" else "PROBLEM"))) + " - " + (
        "; ".join(f["message"] for f in flags[:4]) if flags else "held-out %.4f (best %.4f), trainer alive, anchors accepted, DBlock committing, student committing" % (lastp["w257"] if lastp else float("nan"), best if best is not None else float("nan")))
    return {"overall": overall, "summary": summary, "areas": areas, "flags": flags, "checked_at": int(now), "evidence_version": "mode-quality-and-clock-v1-20260911", "dashboard_evidence_version": DASHBOARD_EVIDENCE_VERSION}

# ----------------------------------------------------------------------------------------------- PNG
def render_png(doc, path):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return False
    W, H = 1080, 2340
    im = Image.new("RGB", (W, H), (11, 14, 20)); dr = ImageDraw.Draw(im)
    def font(sz):
        for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",):
            if os.path.exists(p):
                try: return ImageFont.truetype(p, sz)
                except Exception: pass
        try: return ImageFont.load_default(size=sz)
        except Exception: return ImageFont.load_default()
    F, Fs, Fb, Fh = font(22), font(19), font(28), font(34)
    h = doc.get("health") or {}; ov = h.get("overall", "?")
    col = {"ok": (63, 185, 80), "warn": (210, 153, 34), "bad": (248, 81, 73)}.get(ov, (139, 152, 171))
    dr.text((40, 34), "AGILLM-4.4 training curves", font=Fh, fill=(230, 237, 243))
    dr.text((40, 82), datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC") + "  -  heldout_v3 CE (lower is better)", font=Fs, fill=(139, 152, 171))
    import textwrap
    summ = (h.get("summary") or "")
    _wl = textwrap.wrap(summ, 84) or [""]
    l1, l2 = _wl[0], (" ".join(_wl[1:])[:84] if len(_wl) > 1 else "")
    dr.rounded_rectangle((40, 120, 1040, 196 if l2 else 170), radius=12, fill=(17, 22, 36), outline=col, width=2)
    dr.text((58, 132), ("STATUS: " + l1), font=F, fill=col)
    if l2: dr.text((58, 160), l2, font=F, fill=col)
    y = (206 if l2 else 182)
    # per-area chips
    names = [("trainer", "trainer"), ("model", "model"), ("ar", "AR"), ("sat_fixed", "SAT fixed"), ("sat_variable", "SAT var"), ("nat", "NAT"), ("dblock", "DBlock"), ("det50", "50M student")]
    cx = 40
    for key, label in names:
        lvl = ((h.get("areas") or {}).get(key) or {}).get("level", "?")
        cc = {"ok": (63, 185, 80), "warn": (210, 153, 34), "bad": (248, 81, 73)}.get(lvl, (139, 152, 171))
        w = 18 + 11 * len(label) + 30
        dr.rounded_rectangle((cx, y, cx + w, y + 34), radius=17, fill=(17, 22, 36), outline=cc, width=2)
        dr.ellipse((cx + 12, y + 11, cx + 24, y + 23), fill=cc)
        dr.text((cx + 32, y + 6), label, font=Fs, fill=(230, 237, 243))
        cx += w + 10
    y += 46
    for f in (h.get("flags") or [])[:4]:
        c2 = {"warn": (210, 153, 34), "bad": (248, 81, 73)}.get(f["level"], (139, 152, 171))
        dr.text((58, y), ("- %s: %s" % (f["area"], f["message"]))[:105], font=Fs, fill=c2); y += 26
    def panel(x0, y0, x1, y1, title, series, logy=False, xlab=True):
        dr.rectangle((x0, y0, x1, y1), fill=(8, 11, 17), outline=(42, 51, 70))
        dr.text((x0 + 12, y0 + 8), title, font=Fb, fill=(230, 237, 243))
        pl, pt, pr, pb = x0 + 90, y0 + 52, x1 - 20, y1 - 74
        pts_all = [(x, v) for s in series for x, v in s["pts"] if v is not None and (not logy or v > 0)]
        if len(pts_all) < 2: dr.text((pl, pt), "no data", font=F, fill=(139, 152, 171)); return
        xs = [p[0] for p in pts_all]; x0v, x1v = min(xs), max(xs)
        ys = []
        for s in series:
            vv = sorted((math.log10(v) if logy else v) for x, v in s["pts"] if v is not None and (not logy or v > 0))
            if not vv: continue
            if s.get("robust") and len(vv) > 20: ys += [vv[int(len(vv) * 0.02)], vv[int(len(vv) * 0.98)]]
            else: ys += [vv[0], vv[-1]]
        ymin, ymax = min(ys), max(ys)
        if ymax - ymin < 1e-6: ymin -= 0.1; ymax += 0.1
        pad = (ymax - ymin) * 0.08; ymin -= pad; ymax += pad
        X = lambda s: pl + (s - x0v) / max(1, (x1v - x0v)) * (pr - pl); Y = lambda v: pt + (ymax - v) / (ymax - ymin) * (pb - pt)
        for i in range(5):
            v = ymin + (ymax - ymin) * i / 4; yy = Y(v)
            dr.line((pl, yy, pr, yy), fill=(29, 37, 54)); dr.text((x0 + 10, yy - 11), ("%.1e" % (10 ** v) if logy else "%.2f" % v), font=Fs, fill=(139, 152, 171))
        for i in range(4):
            s = x0v + (x1v - x0v) * i / 3; xx = X(s)
            dr.line((xx, pt, xx, pb), fill=(29, 37, 54))
            if xlab: dr.text((min(xx, pr - 80), pb + 8), "{:,}".format(int(s)), font=Fs, fill=(139, 152, 171))
        for s in series:
            pts = [(X(x), max(pt, min(pb, Y(math.log10(v) if logy else v)))) for x, v in s["pts"] if v is not None and (not logy or v > 0)]
            if s.get("break_before"):
                segments = []; seg = []
                for i, (x, v) in enumerate(s["pts"]):
                    if i in s["break_before"] and seg: segments.append(seg); seg = []
                    if v is not None and (not logy or v > 0):
                        seg.append((X(x), max(pt, min(pb, Y(math.log10(v) if logy else v)))))
                if seg: segments.append(seg)
                for seg in segments:
                    if len(seg) >= 2: dr.line(seg, fill=s["color"], width=s.get("width", 3), joint="curve")
            elif len(pts) >= 2: dr.line(pts, fill=s["color"], width=s.get("width", 3), joint="curve")
            if s.get("dots"):
                for px, py in pts: dr.ellipse((px - 4, py - 4, px + 4, py + 4), fill=s["color"])
        lx = pl
        for s in series:
            dr.rectangle((lx, y1 - 22, lx + 26, y1 - 16), fill=s["color"]); dr.text((lx + 34, y1 - 32), s["name"], font=Fs, fill=(201, 209, 217)); lx += 34 + 12 * len(s["name"]) + 24
    P = doc.get("probes") or []; V = doc.get("val") or []; T = doc.get("train") or []; M = doc.get("multimode") or []
    MC = doc.get("multimode_current") or []; MOD = doc.get("modes") or []
    MC_BREAKS = [i for i in range(1,len(MC)) if MC[i].get("protocol_sha256") != MC[i-1].get("protocol_sha256")]
    Pf = [p for p in P if p["step"] >= FIX_STEP]; Vf = [v for v in V if v["step"] >= FIX_STEP]; Tf = [t for t in T if t["step"] >= FIX_STEP]; Mf = [m for m in M if m["step"] >= FIX_STEP]
    avail = (H - 30) - (y + 14) - 4 * 20
    h1, h2, h3, h4 = int(avail * 0.28), int(avail * 0.20), int(avail * 0.19), int(avail * 0.14)
    y1 = y + 14
    panel(40, y1, 1040, y1 + h1, "Since the 2026-09-06 fix (step %s+)" % "{:,}".format(FIX_STEP), [
        {"name": "held-out 257", "pts": [(p["step"], p["w257"]) for p in Pf], "color": (88, 166, 255), "width": 4, "dots": True},
        {"name": "held-out 2048", "pts": [(p["step"], p.get("w2048")) for p in Pf], "color": (165, 214, 255), "width": 2},
        {"name": "trainer val", "pts": [(v["step"], v["ce"]) for v in Vf], "color": (63, 185, 80), "width": 3, "dots": True},
        {"name": "train loss EMA", "pts": [(t["step"], t["ema"]) for t in Tf], "color": (154, 164, 178), "width": 2, "robust": True}])
    y2 = y1 + h1 + 20
    panel(40, y2, 1040, y2 + h2, "All time (since Aug 25 on this lineage)", [
        {"name": "trainer val", "pts": [(v["step"], v["ce"]) for v in V], "color": (63, 185, 80), "width": 3},
        {"name": "train loss EMA", "pts": [(t["step"], t["ema"]) for t in T], "color": (154, 164, 178), "width": 2, "robust": True},
        {"name": "held-out 257", "pts": [(p["step"], p["w257"]) for p in P], "color": (88, 166, 255), "width": 3}])
    y3 = y2 + h2 + 20
    panel(40, y3, 1040, y3 + h3, "Modes: current CPU heldout; SAT-var training anchor", [
        {"name": "SAT fixed CPU v2", "break_before": MC_BREAKS, "pts": [(m["step"], m.get("sat")) for m in MC], "color": (210,168,255), "width": 3},
        {"name": "SAT var train", "pts": [(m["step"],m.get("sat_var_ce")) for m in MOD if m.get("source")=="training_anchor"], "color": (86,212,221), "width": 2},
        {"name": "NAT CPU v2", "break_before": MC_BREAKS, "pts": [(m["step"],m.get("nat_pooled")) for m in MC], "color": (255,123,114), "width": 3}])
    ylegacy = y3 + h3 + 20
    panel(40, ylegacy, 1040, ylegacy + h3, "Legacy SAT / NAT full held-out CE, 48 windows", [
        {"name": "SAT shift-2", "pts": [(m["step"], m.get("sat")) for m in Mf], "color": (210, 168, 255), "width": 3, "dots": True},
        {"name": "NAT 16", "pts": [(m["step"], m.get("nat16")) for m in Mf], "color": (255, 123, 114), "width": 2},
        {"name": "NAT 32", "pts": [(m["step"], m.get("nat32")) for m in Mf], "color": (255, 159, 151), "width": 2},
        {"name": "NAT 64", "pts": [(m["step"], m.get("nat64")) for m in Mf], "color": (255, 193, 187), "width": 2},
        {"name": "NAT pooled", "pts": [(m["step"], m.get("nat_pooled")) for m in Mf], "color": (255, 70, 60), "width": 3, "dots": True}])
    y4 = ylegacy + h3 + 20
    panel(40, y4, 1040, min(H - 30, y4 + h4), "Learning rate since the fix (log)", [
        {"name": "effective LR", "pts": [(t["step"], t["lr"]) for t in Tf], "color": (210, 153, 34), "width": 3}], logy=True)
    im.save(path + ".tmp", "PNG", optimize=True); os.replace(path + ".tmp", path)
    return True


# MODE_TELEMETRY_V1_20260912: measured sources stay separate.
def mode_telemetry(c, latest):
    def number(value):
        try:
            v = float(value)
            return v if math.isfinite(v) else None
        except (TypeError, ValueError):
            return None
    rows = []
    current = max(int(latest.get("step") or 0), max((int(v.get("step") or 0) for v in c.get("val", [])), default=0))
    for v in c.get("val", [])[-TRAIN_SERIES_MAX_POINTS:]:
        r = {"step": int(v["step"]), "source": "trainer_val", "step_basis": "exact_validation"}
        for src, dst in (("sat_shift2_ce", "sat_fixed_ce"), ("nat_suffix16_ce", "nat16"), ("nat_suffix32_ce", "nat32"), ("nat_suffix64_ce", "nat64")):
            n = number(v.get(src))
            if n is not None: r[dst] = n
        ns = [r.get("nat16"), r.get("nat32"), r.get("nat64")]
        if all(n is not None for n in ns):
            r["nat_ce"] = round(sum(w*n for w,n in zip((16,32,64), ns))/112, 4)
        else:
            n = number(v.get("nat_clean_suffix_ce"))
            if n is not None: r["nat_ce"] = n
        if len(r) > 3: rows.append(r)
    for v in c.get("satvar_vb", [])[-TRAIN_SERIES_MAX_POINTS:]:
        step = int(v.get("step") or 0)
        ce = number(v.get("ce"))
        if step > current or not v.get("finite", True) or ce is None: continue
        r = {"step": step, "dblock_step": v.get("dstep"),
             "source": "training_anchor" if v.get("kind") == "anchor" else "training_local",
             "step_basis": "preceding_heartbeat", "sat_var_ce": ce,
             "sat_var_regret_updates": v.get("updates"), "targets": v.get("targets"),
             "sat_var_ce_by_distance": v.get("ce_by_d") or {},
             "span_basis": "sampled_training_blocks_not_generated_tokens"}
        sizes = v.get("sizes") or {}
        counts = {str(k): number(n) for k,n in sizes.items()}
        counts = {k:n for k,n in counts.items() if n is not None and n >= 0 and k.isdigit()}
        total = sum(counts.values())
        if total > 0:
            r["sat_var_train_mean_span"] = round(sum(int(k)*n for k,n in counts.items())/total, 6)
            r["sat_var_train_p1"] = round(counts.get("1", 0)/total, 6)
            r["sat_var_train_p2"] = round(counts.get("2", 0)/total, 6)
            r["sat_var_train_span_counts"] = counts
        for src,dst in (("mae","sat_var_regret_mae_log"),("regret_loss","sat_var_regret_loss")):
            n = number(v.get(src))
            if n is not None: r[dst] = n
        rows.append(r)
    rows.sort(key=lambda r:(r["step"],r["source"]))
    try: log_age = max(0, int(time.time()-os.stat(LOG).st_mtime))
    except OSError: log_age = None
    meta = {"schema":"agillm44.mode-telemetry.v1", "log_age_seconds":log_age,
            "comparison_note":"Validation crops, full held-out probes, full-stack training anchors and local training objectives are separate measurements. Training span counts are sampled block sizes, not emitted-token choices."}
    for name,key in (("sat_fixed","sat_fixed_ce"),("sat_var","sat_var_ce"),("nat","nat_ce")):
        candidates = [r for r in rows if key in r]
        last = candidates[-1] if candidates else None
        lag = max(0,current-last["step"]) if last else None
        meta[name] = {"status":("missing" if last is None else "stale" if lag > 5000 or log_age is None or log_age > 600 else "recent"),
                      "current_step":current, "last_step":last["step"] if last else None,
                      "step_lag":lag, "source":last["source"] if last else None,
                      "note":("SAT-var CE comes from training audits; sampled training spans do not measure generated choices." if name=="sat_var" else
                              "Trainer validation: one 126-token SAT crop and 16/32/64-token NAT crops; full held-out probes remain separate."),
                      "step_basis":last.get("step_basis") if last else None}
    return rows, meta


def main():
    t0 = time.time(); c = load_cache(); parse_log(c)
    hb = c["hb"]; latest = hb[-1] if hb else {}
    P = probes(); V = c["val"]; A = c.get("anchors") or []; tl = timeline()
    train = ema_series(hb)
    doc = {
        "schema": "agillm43.training-curves.v2", "generated_at": int(time.time()), "log_bytes": os.path.getsize(LOG), "log_mtime": int(os.stat(LOG).st_mtime),
        "parse_seconds": round(time.time() - t0, 2), "heartbeats": len(hb), "fix_step": FIX_STEP,
        "latest": {"step": latest.get("step"), "seen_tok": latest.get("tok"), "loss": latest.get("loss"), "ema": train[-1]["ema"] if train else None, "lr": latest.get("lr"), "raw_tok_s": latest.get("tps"), "gpu_peak_gb": latest.get("peak")},
        "train": train, "anchors": A[-TRAIN_SERIES_MAX_POINTS:], "val": V, "probes": P, "governor": [e for e in governor_events() if not e.get("probe")], "markers": c["markers"][-60:], "multimode": multimode(), "timeline": tl,
        "anchor_meta": {"signal": "full-stack AR anchor CE on fixed-size trainer anchor crops", "comparable_across_dblock_objective_routes": True, "purpose": "primary local training trend; raw mixed-objective heartbeat loss is diagnostic noise"},
        "train_meta": {"signal": "current-step mixed local-objective loss sampled at each trainer heartbeat",
                       "comparable_to_frozen_heldout": False,
                       "ema_alpha_per_heartbeat": TRAIN_EMA_ALPHA,
                       "ema_bias_corrected": False,
                       "ema_half_life_samples": round(math.log(0.5) / math.log(1.0 - TRAIN_EMA_ALPHA), 2),
                       "nominal_heartbeat_seconds": 30,
                       "ema_half_life_minutes_at_nominal_cadence": round(math.log(0.5) / math.log(1.0 - TRAIN_EMA_ALPHA) * 0.5, 2),
                       "max_chart_points": TRAIN_SERIES_MAX_POINTS,
                       "downsample": "last EMA/step/LR/tokens per bucket; arithmetic mean raw loss/throughput; max peak"},
        "heldout": {"name": "heldout_v3", "tokens": 12404, "sha256": "513b30203ffd18167655deb2d9cd610541df87d1ea0b8a0f9241c58e68703d21"},
    }
    doc["modes"], doc["mode_meta"] = mode_telemetry(c, latest)
    doc["multimode_current"] = multimode(current=True)
    try: doc["health"] = health(c, P, V, hb, tl)
    except Exception as exc:
        doc["health"] = {"overall": "warn", "summary": "health check failed: %s: %s" % (type(exc).__name__, exc), "areas": {}, "flags": [], "checked_at": int(time.time())}
    with open(CACHE + ".tmp", "w") as f: json.dump(c, f)
    os.replace(CACHE + ".tmp", CACHE)
    doc["health_seconds"] = round(time.time() - t0 - doc["parse_seconds"], 2)
    if "--png" in sys.argv:
        try: doc["png"] = PNG if render_png(doc, PNG) else None
        except Exception as exc: doc["png"] = None; doc["png_error"] = "%s: %s" % (type(exc).__name__, exc)
    js = json.dumps(doc, separators=(",", ":"))
    with open(OUT + ".tmp", "w") as f: f.write(js)
    os.replace(OUT + ".tmp", OUT)
    if "--quiet" not in sys.argv: sys.stdout.write(js)

if __name__ == "__main__":
    main()
