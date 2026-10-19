#!/usr/bin/env python3
"""Read-only full-lineage loss-curve reduction for the AGILLM UI.

Runs on the GB10 box, fed over SSH stdin by GETH's curve_history.py:
    python3 - '<json config>'  < curve_history_reduce.py
It streams each trainer log once, keeps only rows on the surviving branch and
prints bucketed summaries as one compact JSON document on stdout.  It never
writes files and never touches the trainer.

Surviving branch: a resume from checkpoint step R abandons every earlier row
with step > R (those weights were never saved).  Scanning the file backwards,
a row survives only if no later row, and no later "resumed" event, has an
equal or smaller step.
"""
import glob
import json
import math
import os
import platform
import re
import sys
import time
from array import array

OBJ_CODE = {"ar": 0, "sat": 1, "nat": 2}
NAN = float("nan")


def _f(v, default=NAN):
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


class Rows:
    __slots__ = ("step", "loss", "obj", "stage", "anc", "ts", "lr", "tps", "gn", "tok", "legacy", "events")

    def __init__(self):
        self.step = array("q"); self.loss = array("d"); self.obj = array("b")
        self.stage = array("b"); self.anc = array("b"); self.ts = array("d")
        self.lr = array("d"); self.tps = array("d"); self.gn = array("d")
        self.tok = array("q"); self.legacy = array("b")
        self.events = []  # (row_index_when_seen, kind, step, ts, extra)

    def add(self, step, loss, obj, stage, anc, ts, lr, tps, gn, tok, legacy):
        self.step.append(step); self.loss.append(loss); self.obj.append(obj)
        self.stage.append(stage); self.anc.append(anc); self.ts.append(ts)
        self.lr.append(lr); self.tps.append(tps); self.gn.append(gn)
        self.tok.append(tok); self.legacy.append(legacy)


def parse_jsonl(path):
    """RPV16 target-fix trainer: one JSON object per line."""
    rows = Rows()
    with open(path, "rb") as fh:
        for line in fh:
            if not line.startswith(b"{"):
                continue
            if line.startswith(b'{"event": "train"'):
                try:
                    d = json.loads(line)
                    step = int(d["step"])
                except Exception:
                    continue
                loss = _f(d.get("loss"))
                if not math.isfinite(loss):
                    continue
                o = d.get("objective")
                legacy = 1 if o is None else 0
                obj = OBJ_CODE.get(o if o is not None else "ar", 3)
                st = d.get("train_stage")
                # -1 is a real stage (joint all-stage update); -2 marks "no stage logged" (e.g. global-anchor rows)
                stage = int(st) if isinstance(st, int) and -128 < st < 128 else -2
                rows.add(step, loss, obj, stage, 1 if d.get("global_anchor") else 0,
                         _f(d.get("ts")), _f(d.get("lr")), _f(d.get("tok_s")),
                         _f(d.get("grad_norm")), int(_f(d.get("seen_tokens"), 0.0)), legacy)
            elif line.startswith(b'{"event": "hf_source_wrap"'):
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                s = d.get("step")
                s = int(s) if isinstance(s, (int, float)) else (rows.step[-1] if len(rows.step) else 0)
                rows.events.append((len(rows.step), "wrap", s, _f(d.get("ts")), str(d.get("source") or "")))
            elif line.startswith(b'{"event": "resumed"') or line.startswith(b'{"event": "checkpoint"'):
                try:
                    d = json.loads(line)
                    step = int(d["step"])
                except Exception:
                    continue
                kind = "resume" if d.get("event") == "resumed" else "checkpoint"
                rows.events.append((len(rows.step), kind, step, _f(d.get("ts")),
                                    os.path.basename(str(d.get("path") or ""))))
    return rows


_HB = re.compile(rb"^\[heartbeat\].*?\bstep=(\d+)\s+seen_tok=(\d+)\s+loss=([-\d.eE+naif]+)\s+lr=([-\d.eE+]+)(?:.*?\braw_tok_s=([\d.]+))?")
_DIAG = re.compile(rb'^\[dblock-local-diag\] (\{.*\})\s*$')


def parse_heartbeat(path):
    """Standard 1.1B trainer: text '[heartbeat] ... step=.. loss=.. lr=..' lines.

    Per-objective CE comes from '[dblock-local-diag]' JSON lines (objective,
    raw_average_ce, step); heartbeat loss is the mixed local-batch loss.
    """
    rows = Rows()
    diag = []  # (row_index_when_seen, step, obj, ce)
    with open(path, "rb") as fh:
        for line in fh:
            if line.startswith(b"[heartbeat]"):
                m = _HB.match(line)
                if not m:
                    continue
                loss = _f(m.group(3).decode())
                if not math.isfinite(loss):
                    continue
                rows.add(int(m.group(1)), loss, 3, -2, 0, NAN, _f(m.group(4).decode()),
                         _f((m.group(5) or b"nan").decode()), NAN, int(m.group(2)), 0)
            elif line.startswith(b"[dblock-local-diag]"):
                m = _DIAG.match(line)
                if not m:
                    continue
                try:
                    d = json.loads(m.group(1))
                    diag.append((len(rows.step), int(d["step"]), OBJ_CODE.get(d.get("objective"), 3),
                                 _f(d.get("raw_average_ce"))))
                except Exception:
                    continue
    return rows, diag


def surviving_mask(rows, upto=None):
    """Mask of rows on the surviving branch of rows[:upto] (events before upto only)."""
    n = len(rows.step) if upto is None else upto
    keep = bytearray(len(rows.step))
    ev = sorted((e for e in rows.events if e[1] == "resume" and (upto is None or e[0] < upto)), key=lambda e: e[0])
    ei = len(ev) - 1
    min_after = math.inf
    for i in range(n - 1, -1, -1):
        # resume events recorded after row i (seen when len(rows)==i+1 or later)
        while ei >= 0 and ev[ei][0] > i:
            min_after = min(min_after, ev[ei][2] + 1)
            ei -= 1
        s = rows.step[i]
        if s < min_after:
            keep[i] = 1
            min_after = s
    return keep


def _pct(sorted_vals, q):
    if not sorted_vals:
        return None
    k = min(len(sorted_vals) - 1, max(0, int(math.ceil(q * len(sorted_vals))) - 1))
    return sorted_vals[k]


def _mean(vals):
    return (sum(vals) / len(vals)) if vals else None


def _med(vals):
    if not vals:
        return None
    v = sorted(vals)
    m = len(v) // 2
    return v[m] if len(v) % 2 else 0.5 * (v[m - 1] + v[m])


COLS = ["s0", "s1", "t0", "t1", "n", "loss", "lo", "hi", "ar", "ar_n", "sat", "sat_n",
        "nat", "nat_n", "anc", "anc_n", "lr", "tps", "gn", "tok", "ar_deep", "ar_p10", "ar_p90",
        "ar_full15", "ar_full15_n", "full", "full_n"]


def _r4(v):
    return None if v is None or not math.isfinite(v) else round(v, 4)


def _g4(v):
    return None if v is None or not math.isfinite(v) else float("%.4g" % v)


def bucketize(idx, rows, bucket, lo_step=None, diag_by_row=None):
    """idx: surviving row indices sorted by step.  Returns list of row lists."""
    out = []
    cur = None
    acc = None

    def flush():
        if acc is None or not acc["loss"]:
            return
        L = sorted(acc["loss"])
        ts = [t for t in acc["ts"] if math.isfinite(t)]
        ar, sat, nat = acc["ar"], acc["sat"], acc["nat"]
        lr = acc["lr"]
        tps = [x for x in acc["tps"] if math.isfinite(x) and x > 0]
        gn = [x for x in acc["gn"] if math.isfinite(x)]
        out.append([
            acc["s0"], acc["s1"],
            int(min(ts)) if ts else None, int(max(ts)) if ts else None,
            len(L), _r4(_mean(L)), _r4(_pct(L, 0.10)), _r4(_pct(L, 0.90)),
            _r4(_mean(ar)), len(ar), _r4(_mean(sat)), len(sat), _r4(_mean(nat)), len(nat),
            _r4(_mean(acc["anc"])), len(acc["anc"]),
            _g4(lr), (int(round(_med(tps))) if tps else None), _g4(_med(gn)) if gn else None,
            acc["tok"] if acc["tok"] > 0 else None, _r4(_mean(acc["ar_deep"])),
            _r4(_pct(sorted(ar), 0.10)), _r4(_pct(sorted(ar), 0.90)),
            _r4(_mean(acc["ar_full"])), len(acc["ar_full"]),
            _r4(_mean(acc["full"])), len(acc["full"]),
        ])

    for i in idx:
        s = rows.step[i]
        if lo_step is not None and s < lo_step:
            continue
        b = s // bucket
        if b != cur:
            flush()
            cur = b
            acc = {"s0": s, "s1": s, "loss": [], "ts": [], "ar": [], "sat": [], "nat": [],
                   "anc": [], "ar_deep": [], "ar_full": [], "full": [], "lr": NAN, "tps": [], "gn": [], "tok": 0}
        acc["s1"] = s
        v = rows.loss[i]
        acc["loss"].append(v)
        acc["ts"].append(rows.ts[i])
        if diag_by_row is not None:
            for (o, ce) in diag_by_row.get(i, ()):
                if math.isfinite(ce):
                    (acc["ar"] if o == 0 else acc["sat"] if o == 1 else acc["nat"] if o == 2 else []).append(ce)
        if rows.anc[i]:
            acc["anc"].append(v)
            acc["full"].append(v)
        else:
            o = rows.obj[i]
            if o == 0:
                acc["ar"].append(v)
                if rows.stage[i] >= 10:
                    acc["ar_deep"].append(v)
                if rows.stage[i] in (15, -1):
                    acc["ar_full"].append(v)
                    acc["full"].append(v)
            elif o == 1:
                acc["sat"].append(v)
            elif o == 2:
                acc["nat"].append(v)
        if math.isfinite(rows.lr[i]):
            acc["lr"] = rows.lr[i]
        acc["tps"].append(rows.tps[i]); acc["gn"].append(rows.gn[i])
        if rows.tok[i] > 0:
            acc["tok"] = rows.tok[i]
    flush()
    return out


def best_of(buckets, col, min_n):
    ci, ni = COLS.index(col), COLS.index(col + "_n")
    best = None
    for r in buckets:
        v, n = r[ci], r[ni]
        if v is None or n < min_n:
            continue
        if best is None or v < best["value"]:
            best = {"value": v, "s0": r[0], "s1": r[1], "step": (r[0] + r[1]) // 2, "n": n}
    return best


def window_mean(idx, rows, lo, hi, obj=0):
    vals = [rows.loss[i] for i in idx if lo <= rows.step[i] <= hi and not rows.anc[i] and rows.obj[i] == obj]
    return (_r4(_mean(vals)), len(vals))


def slope_per_100k(buckets, col, span_steps):
    """Least-squares slope of a bucket-mean column over the last span_steps, per 100k steps."""
    ci, ni = COLS.index(col), COLS.index(col + "_n")
    if not buckets:
        return None
    last = buckets[-1][1]
    pts = [((r[0] + r[1]) / 2.0, r[ci], r[ni]) for r in buckets if r[1] >= last - span_steps and r[ci] is not None and r[ni] >= 8]
    if len(pts) < 4:
        return None
    w = [p[2] for p in pts]
    sw = sum(w)
    mx = sum(p[0] * p[2] for p in pts) / sw
    my = sum(p[1] * p[2] for p in pts) / sw
    sxx = sum(p[2] * (p[0] - mx) ** 2 for p in pts)
    if sxx <= 0:
        return None
    sxy = sum(p[2] * (p[0] - mx) * (p[1] - my) for p in pts)
    return round(sxy / sxx * 100000.0, 4)


def resume_depths(rows):
    """[(event_index, pos, step, ts, depth)] for every resume.

    depth = newest step of the lineage that was live just before the resume minus
    the resume step.  Rows already abandoned by an earlier resume do not count, so a
    clean restart after a rollback has depth ~0.  A negative depth means the run
    resumed ahead of its last logged row.
    """
    out = []
    stack = []  # surviving lineage steps so far, strictly increasing
    pos_done = 0
    for k, e in enumerate(rows.events):
        if e[1] != "resume":
            continue
        pos = e[0]
        for i in range(pos_done, pos):
            s = rows.step[i]
            while stack and stack[-1] >= s:
                stack.pop()
            stack.append(s)
        pos_done = max(pos_done, pos)
        R = e[2]
        out.append((k, pos, R, e[3], (stack[-1] - R) if stack else 0))
        while stack and stack[-1] > R:
            stack.pop()
    return out


def later_min_resume(rows):
    """{event_index: min resume step among resumes logged after that event} (inf if none)."""
    out = {}
    m = math.inf
    for k in range(len(rows.events) - 1, -1, -1):
        out[k] = m
        e = rows.events[k]
        if e[1] == "resume":
            m = min(m, e[2])
    return out


def window_mean_where(idx, rows, lo, hi, pred):
    vals = [rows.loss[i] for i in idx if lo <= rows.step[i] <= hi and pred(i)]
    return (_r4(_mean(vals)), len(vals))


def reduce_job(job, cfg):
    t0 = time.time()
    path = job["path"]
    st = os.stat(path)
    if job.get("format") == "heartbeat":
        rows, diag = parse_heartbeat(path)
    else:
        rows, diag = parse_jsonl(path), None
    n = len(rows.step)
    keep = surviving_mask(rows)
    idx = [i for i in range(n) if keep[i]]
    idx.sort(key=lambda i: rows.step[i])
    diag_by_row = None
    if diag is not None:
        # dblock-local-diag uses its own step counter, so attribute each line to
        # the heartbeat row logged just before it; the surviving mask then applies.
        diag_by_row = {}
        for (pos, _s, o, ce) in diag:
            if pos > 0 and keep[pos - 1]:
                diag_by_row.setdefault(pos - 1, []).append((o, ce))
    if not idx:
        return {"ok": False, "reason": "no train rows", "source_log": path}
    first, last = rows.step[idx[0]], rows.step[idx[-1]]
    # Major rollbacks: a resume far behind the newest logged step abandons a whole
    # branch.  Keep the most recent ones as separate, explicitly labelled series.
    depths = resume_depths(rows)
    min_depth = int(cfg.get("rollback_min_steps", 4096))
    branch_idx = []
    for (_k, pos, R, ts, depth) in reversed([d for d in depths if d[4] >= min_depth]):
        if len(branch_idx) >= int(cfg.get("max_branches", 2)):
            break
        m = surviving_mask(rows, upto=pos)
        bi = [i for i in range(pos) if m[i] and rows.step[i] > R]
        bi.sort(key=lambda i: rows.step[i])
        if bi:
            branch_idx.append((R, ts, depth, bi))
    branch_idx.reverse()
    span_hi = max([last] + [rows.step[b[3][-1]] for b in branch_idx])
    max_rows = int(cfg.get("max_rows", 1200))
    bucket = int(cfg.get("min_bucket", 64))
    while (span_hi - first + 1) / bucket > max_rows:
        bucket *= 2
    main = bucketize(idx, rows, bucket, diag_by_row=diag_by_row)
    branches = []
    for (R, ts, depth, bi) in branch_idx:
        brows = bucketize(bi, rows, bucket)
        b_last = rows.step[bi[-1]]
        b_ar = window_mean(bi, rows, b_last - 9999, b_last)
        branches.append({
            "kind": "abandoned",
            "cut_step": R, "cut_ts": int(ts) if math.isfinite(ts) else None, "depth_steps": depth,
            "from_step": rows.step[bi[0]], "to_step": b_last,
            "label": "abandoned branch %s-%s (rolled back to %s)" % (rows.step[bi[0]], b_last, R),
            "ar_mean_last_10k": b_ar[0], "best": {"ar": best_of(brows, "ar", 32), "anc": best_of(brows, "anc", 8)},
            "rows": brows,
        })
    rb, rspan = int(cfg.get("recent_bucket", 64)), int(cfg.get("recent_span", 16384))
    recent = bucketize(idx, rows, rb, lo_step=max(first, last - rspan + 1), diag_by_row=diag_by_row)
    legacy_idx = [i for i in idx if rows.legacy[i]]
    lmin = later_min_resume(rows)

    def on_current(k, s):
        # an event at step s survives unless a later resume went back below it
        return s <= last and s <= lmin[k]

    resumes = {}
    for (k, _pos, s, ts, depth) in depths:
        lin = "current" if on_current(k, s) else "abandoned"
        r = resumes.setdefault((s, lin), {"step": s, "lineage": lin, "ts": None, "count": 0, "max_depth": depth})
        r["count"] += 1
        r["max_depth"] = max(r["max_depth"], depth)
        if math.isfinite(ts):
            r["ts"] = int(ts)
    last_ck = None
    for k, e in enumerate(rows.events):
        if e[1] == "checkpoint" and on_current(k, e[2]):
            last_ck = e
    markers = []
    jfirst = next((rows.step[i] for i in idx if rows.stage[i] == -1), None)
    if jfirst is not None:
        markers.append({"kind": "objective_change", "step": jfirst, "ts": None, "lineage": "current",
                        "label": "joint all-stage update: plain AR rows are full-model from here"})
    for k, e in enumerate(rows.events):
        if e[1] == "wrap":
            markers.append({"kind": "data_wrap", "step": e[2], "ts": int(e[3]) if math.isfinite(e[3]) else None,
                            "lineage": "current" if on_current(k, e[2]) else "abandoned",
                            "label": "dataset wrapped (repeated data from here)"})
    ar_now = window_mean(idx, rows, last - 9999, last)
    ar_prev = window_mean(idx, rows, last - 59999, last - 50000)
    anc_now = window_mean_where(idx, rows, last - 9999, last, lambda i: rows.anc[i])
    full_now = window_mean_where(idx, rows, last - 9999, last,
                                 lambda i: (not rows.anc[i]) and rows.obj[i] == 0 and rows.stage[i] in (15, -1))
    fullm_now = window_mean_where(idx, rows, last - 9999, last,
                                  lambda i: rows.anc[i] or (rows.obj[i] == 0 and rows.stage[i] in (15, -1)))
    fullm_2k = window_mean_where(idx, rows, last - 1999, last,
                                 lambda i: rows.anc[i] or (rows.obj[i] == 0 and rows.stage[i] in (15, -1)))
    joint_n = sum(1 for i in idx[-4096:] if rows.stage[i] == -1)
    mix_now = window_mean_where(idx, rows, last - 9999, last, lambda i: True)
    doc = {
        "ok": True,
        "schema": "agillm.rpv16.curve-history.v1",
        "lineage": job["lineage"],
        "label": job.get("label"),
        "cols": COLS,
        "bucket_steps": bucket,
        "rows": main,
        "recent": {"bucket_steps": rb, "span_steps": rspan, "rows": recent},
        "resumes": sorted(resumes.values(), key=lambda r: (r["step"], r["lineage"])),
        "markers": markers,
        "branches": branches,
        "rollback_min_steps": min_depth,
        "best": {"ar": best_of(main, "ar", 32), "anc": best_of(main, "anc", 8),
                 "ar_full15": best_of(main, "ar_full15", 8), "full": best_of(main, "full", 8)},
        "summary": {
            "first_step": first, "last_step": last,
            "rows_total": n, "rows_surviving": len(idx), "rows_abandoned": n - len(idx),
            "ar_mean_last_10k": ar_now[0], "ar_n_last_10k": ar_now[1],
            "ar_mean_50k_earlier": ar_prev[0], "ar_n_50k_earlier": ar_prev[1],
            "anc_mean_last_10k": anc_now[0], "anc_n_last_10k": anc_now[1],
            "ar_full15_mean_last_10k": full_now[0], "ar_full15_n_last_10k": full_now[1],
            "full_mean_last_10k": fullm_now[0], "full_n_last_10k": fullm_now[1],
            "full_mean_last_2k": fullm_2k[0], "full_n_last_2k": fullm_2k[1],
            "full_slope_per_100k_last_100k": slope_per_100k(main, "full", 100000),
            "objective_mode": "joint" if joint_n else "stage-local",
            "loss_mean_last_10k": mix_now[0],
            "ar_slope_per_100k_last_100k": slope_per_100k(main, "ar", 100000),
            "anc_slope_per_100k_last_100k": slope_per_100k(main, "anc", 100000),
            "last_checkpoint": ({"step": last_ck[2], "ts": int(last_ck[3]) if math.isfinite(last_ck[3]) else None}
                                if last_ck else None),
        },
        "meta": {
            "built_at": int(time.time()),
            "reduce_s": round(time.time() - t0, 2),
            "source_log": path,
            "log_bytes": st.st_size,
            "log_mtime": int(st.st_mtime),
            "format": job.get("format", "jsonl"),
            "legacy_ar_until_step": (rows.step[legacy_idx[-1]] if legacy_idx else None),
            "legacy_note": ("rows up to legacy_ar_until_step predate the objective field; that phase trained AR only, so they are counted as ar"
                            if legacy_idx else None),
            "loss_note": ("ar/sat/nat are per-objective means of training-batch loss excluding global-anchor rows; "
                          "ar_p10/ar_p90 bound the AR rows; ar_deep = AR mean over DBlock stages 10-15; "
                          "ar_full15 = full-depth AR rows (stage 15 in stage-local mode, stage -1 in joint mode); "
                          "full = full-model CE in either regime (global-anchor rows + full-depth AR rows); "
                          "in joint mode (from 2026-09-23 ~05:03Z) plain ar rows are themselves full-model; "
                          "the plain ar column averages 16 early-exit depths (stage-local objective), so the "
                          "full model is best read from anc or ar_full15; "
                          "anc is the global-anchor (full-stack) loss; loss/lo/hi mix all objectives and stages"),
        },
    }
    if job.get("format") == "heartbeat":
        doc["meta"]["loss_note"] = ("loss/lo/hi = mixed local-batch heartbeat loss; ar/sat/nat = per-objective "
                                    "raw_average_ce from [dblock-local-diag] lines in the bucket; no global-anchor rows")
    return doc


def collect_heldout(patterns):
    pts = []
    seen = set()
    for pat in patterns or []:
        for f in sorted(glob.glob(pat)):
            if f in seen:
                continue
            seen.add(f)
            try:
                if os.path.getsize(f) > 4 * 1024 * 1024:
                    continue
                d = json.load(open(f))
            except Exception:
                continue
            if not isinstance(d, dict) or not isinstance(d.get("step"), int):
                continue
            ce = d.get("ce")
            val = ce.get("exact") if isinstance(ce, dict) else ce
            val = _f(val)
            if not math.isfinite(val):
                continue
            hm = d.get("heldout_meta") or {}
            proto = "%s %sx%s" % (hm.get("source") or "held-out", hm.get("rows") or "?", hm.get("seq") or "?")
            pts.append({
                "step": d["step"], "ce": round(val, 4), "variant": "exact" if isinstance(ce, dict) else None,
                "tokens": d.get("tokens"), "protocol": proto,
                "tokenizer_sha256_prefix": str(hm.get("tokenizer_sha256") or "")[:12] or None,
                "ckpt": os.path.basename(str(d.get("ckpt") or "")) or None,
                "source_file": f, "mtime": int(os.path.getmtime(f)),
            })
    pts.sort(key=lambda p: (p["step"], p["mtime"]))
    return pts


def main():
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    out = {"ok": True, "host": platform.node(), "generated_at": int(time.time()), "jobs": {}}
    for job in cfg.get("jobs", []):
        try:
            out["jobs"][job["name"]] = reduce_job(job, cfg)
        except Exception as exc:
            out["jobs"][job["name"]] = {"ok": False, "reason": "%s: %s" % (type(exc).__name__, exc),
                                        "source_log": job.get("path")}
    try:
        out["heldout"] = collect_heldout(cfg.get("heldout_globs"))
    except Exception as exc:
        out["heldout"] = []
        out["heldout_error"] = "%s: %s" % (type(exc).__name__, exc)
    sys.stdout.write(json.dumps(out, separators=(",", ":"), allow_nan=False))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
