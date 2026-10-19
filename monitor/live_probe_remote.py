#!/usr/bin/env python3
"""Read-only live telemetry probe for the AGILLM UI (runs on the GB10 box, fed over SSH stdin).

One compact JSON document: GPU, host memory, disk, both trainers' processes, the Standard
heartbeat tail, the Standard LR override, recent checkpoint events and the autopromote held-out
history.  Never writes anything.
"""
import json, os, re, shutil, subprocess, sys, time

STD_LOG = "/workspace/dual_training_20260920/standard_trainer.log"
RPV_LOG = "/workspace/rpv16-keepalive-trainer.log"
OVR = "/workspace/agillm43_lr_override.json"
PROMOTE = "/workspace/rpv16_promote/state.json"
HB = re.compile(r"^\[heartbeat\].*?\bstep=(\d+)\s+seen_tok=(\d+)\s+loss=(\S+)\s+lr=(\S+)(?:.*?\braw_tok_s=([\d.]+))?")
VAL = re.compile(r"^\[val\] step=(\d+)\s+tokens=\d+\s+ce=(\S+)")
GOV = re.compile(r"^\[lr-governor\] effective override lr=(\S+) at seen_tok=(\d+)")


def fnum(x):
    try:
        v = float(x)
        return v if v == v and abs(v) != float("inf") else None
    except Exception:
        return None


def tail_lines(path, max_bytes):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
        lines = data.decode("utf-8", "replace").splitlines()
        return lines[1:] if size > max_bytes else lines, size, os.path.getmtime(path)
    except Exception:
        return [], None, None


def gpu():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,temperature.gpu,clocks.sm",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        p = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"util_pct": fnum(p[0]), "power_w": fnum(p[1]), "temp_c": fnum(p[2]), "sm_clock_mhz": fnum(p[3])}
    except Exception as exc:
        return {"error": str(exc)[:120]}


def host():
    mem = {}
    try:
        for ln in open("/proc/meminfo"):
            k, v = ln.split(":", 1)
            mem[k] = int(v.split()[0]) * 1024
    except Exception:
        pass
    du = shutil.disk_usage("/workspace")
    g = 1024 ** 3
    out = {"mem_total_gib": round(mem.get("MemTotal", 0) / g, 2), "mem_available_gib": round(mem.get("MemAvailable", 0) / g, 2),
           "disk_total_gib": round(du.total / g, 2), "disk_free_gib": round(du.free / g, 2)}
    try:
        out["load1"] = round(os.getloadavg()[0], 2)
    except Exception:
        pass
    return out


def procs():
    hz = os.sysconf("SC_CLK_TCK")
    boot = None
    try:
        for ln in open("/proc/stat"):
            if ln.startswith("btime"):
                boot = int(ln.split()[1])
    except Exception:
        pass
    found = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = open("/proc/%s/cmdline" % pid, "rb").read().replace(b"\0", b" ").decode("utf-8", "replace")
        except Exception:
            continue
        argv = cmd.split()
        if "train" not in argv or not argv or "python" not in os.path.basename(argv[0]):
            continue
        script = next((os.path.basename(t) for t in argv[1:4] if t.endswith(".py")), "")
        kind = "rpv16" if script.startswith("agillm_gb10_1pf") else "standard" if script.startswith("agillm44_retention4") else None
        if not kind:
            continue
        try:
            st = open("/proc/%s/stat" % pid).read().rsplit(")", 1)[1].split()
            start = boot + int(st[19]) / hz if boot else None
        except Exception:
            start = None
        src = next((t for t in cmd.split() if t.endswith(".py")), None)
        found.setdefault(kind, []).append({"pid": int(pid), "started": int(start) if start else None,
                                           "uptime_s": int(time.time() - start) if start else None,
                                           "source": os.path.basename(src) if src else None})
    return found


def standard_val(n=160):
    """Held-out [val] rows of the surviving lineage (a later resume supersedes rows at or past its step)."""
    try:
        r = subprocess.run(["grep", "-a", r"^\[val\] step=", STD_LOG], capture_output=True, timeout=10,
                           env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"})
        rows = []
        for ln in r.stdout.decode("utf-8", "replace").splitlines():
            m = VAL.match(ln)
            if m:
                rows.append([int(m.group(1)), fnum(m.group(2))])
    except Exception:
        return []
    keep, floor = [], None
    for st, ce in reversed(rows):
        if floor is None or st < floor:
            keep.append([st, ce])
            floor = st
    return keep[::-1][-n:]


def standard_tail(n=900):
    lines, size, mtime = tail_lines(STD_LOG, 14_000_000)
    rows, gov = [], None
    for ln in lines:
        if ln.startswith("[heartbeat]"):
            m = HB.match(ln)
            if m:
                rows.append([int(m.group(1)), int(m.group(2)), fnum(m.group(3)), fnum(m.group(4)), fnum(m.group(5))])
        elif ln.startswith("[lr-governor]"):
            g = GOV.match(ln)
            gov = {"effective_lr": fnum(g.group(1)), "seen_tok": int(g.group(2))} if g else {"line": ln[:200]}
    rows = rows[-n:]
    return {"cols": ["step", "seen_tok", "loss", "lr", "raw_tok_s"], "rows": rows, "log_bytes": size,
            "log_mtime": int(mtime) if mtime else None, "last_governor": gov, "val": standard_val()}


def rpv16_events():
    lines, size, mtime = tail_lines(RPV_LOG, 3_000_000)
    ck, res, last = None, None, None
    for ln in lines:
        if not ln.startswith("{"):
            continue
        if ln.startswith('{"event": "checkpoint"'):
            try:
                d = json.loads(ln)
                ck = {k: d.get(k) for k in ("step", "ts", "save_s", "stall_s", "async", "reason")}
            except Exception:
                pass
        elif ln.startswith('{"event": "resumed"'):
            try:
                d = json.loads(ln)
                res = {k: d.get(k) for k in ("step", "ts", "resume_total_s")}
            except Exception:
                pass
    return {"last_checkpoint": ck, "last_resume": res, "log_mtime": int(mtime) if mtime else None}


def override():
    try:
        return json.load(open(OVR))
    except FileNotFoundError:
        return None
    except Exception as exc:
        return {"error": str(exc)[:120]}


def promote():
    try:
        d = json.load(open(PROMOTE))
        return {"history": d.get("history") or [], "served": d.get("served"),
                "served_history": d.get("served_history") or [], "last_evaluated_step": d.get("last_evaluated_step")}
    except Exception as exc:
        return {"error": str(exc)[:120]}


doc = {"ok": True, "ts": int(time.time()), "gpu": gpu(), "host": host(), "procs": procs(),
       "standard": standard_tail(), "rpv16": rpv16_events(), "lr_override": override(), "promote": promote()}
sys.stdout.write(json.dumps(doc, separators=(",", ":")) + "\n")
