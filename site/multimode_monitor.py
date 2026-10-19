"""multimode_monitor: after every held-out AR probe (heldout_monitor.jsonl) run multimode_probe.py on the SAME
checkpoint on CPU (nice 10) so SAT shift-2 and NAT suffix-16/32/64 CE are measured over all 48 held-out windows
(12,240 SAT tokens; 768 / 1,536 / 3,072 NAT tokens) instead of the trainer's single-window [val] crops
(126 / 16 / 32 / 64 tokens), which are too small to alarm on.
Writes multimode_<step>.json (same schema as the hop-time probes + ts/step/lineage_from/source/wall_seconds),
refreshes multimode_latest.json and appends one JSON line per run to multimode_monitor.jsonl.
Context-safe: one JSON line per run, no progress bars. Never touches the GPU (CUDA_VISIBLE_DEVICES="")."""
import json, os, re, subprocess, sys, time, pathlib
PROBE = pathlib.Path("/workspace/agillm43_serving_probe")
SAVE = pathlib.Path("/workspace/agillm43_production_v14/production_quality_v18_hard_spike_gates/checkpoints_continue400B_from200B")
JL = PROBE / "heldout_monitor.jsonl"; OUTL = PROBE / "multimode_monitor.jsonl"; LATEST = PROBE / "multimode_latest.json"
POLL = 20; TIMEOUT = 1500
def now_iso(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def atomic_json(p, obj):
    t = p.with_name(p.name + ".tmp")
    with t.open("w") as f: json.dump(obj, f, indent=1); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)
def last_probe():
    try:
        with JL.open("rb") as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 16000)); tail = f.read().decode("utf-8", "replace")
    except FileNotFoundError:
        return None
    for line in reversed(tail.splitlines()):
        try: d = json.loads(line)
        except Exception: continue
        if "w257" in d and d.get("checkpoint") and d.get("step") is not None: return d
    return None
def find_ckpt(name):
    p = SAVE / name
    if p.is_file(): return p
    try:
        q = pathlib.Path(json.loads((SAVE / "training_latest.json").read_text()).get("checkpoint_path", ""))
        if q.name == name and q.is_file(): return q
    except Exception: pass
    return None
def run(step, path):
    tmp = PROBE / (".multimode_%d.partial.json" % step); final = PROBE / ("multimode_%d.json" % step)
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = ""; t0 = time.time()
    rec = {"ts": now_iso(), "step": step, "checkpoint": path.name}
    try:
        p = subprocess.run(["nice", "-n", "10", "python3", str(PROBE / "multimode_probe.py"), str(path), str(tmp)], env=env, capture_output=True, text=True, timeout=TIMEOUT)
        rec["rc"] = p.returncode
        d = json.load(open(tmp)); os.remove(tmp)
        m = re.search(r"_from0*(\d+)_", path.name)
        d.update({"ts": rec["ts"], "step": step, "lineage_from": int(m.group(1)) if m else None, "source": "multimode_monitor", "wall_seconds": round(time.time() - t0, 1)})
        atomic_json(final, d); atomic_json(LATEST, d)
        rec.update({k: d.get(k) for k in ("ar_ce", "sat_shift2_ce", "nat_suffix16_ce", "nat_suffix32_ce", "nat_suffix64_ce", "seconds")})
    except Exception as exc:
        tail = ""
        try: tail = p.stderr[-300:]
        except Exception: pass
        rec["error"] = "probe_failed %r tail=%s" % (exc, tail)
    rec["wall_seconds"] = round(time.time() - t0, 1)
    OUTL.open("a").write(json.dumps(rec) + "\n"); print(json.dumps(rec), flush=True)
done = set()
for p in PROBE.glob("multimode_*.json"):
    try: done.add(json.loads(p.read_text()).get("checkpoint", "").split("/")[-1])
    except Exception: pass
print(json.dumps({"ts": now_iso(), "start": True, "pid": os.getpid(), "already_done": len(done)}), flush=True)
while True:
    try:
        lp = last_probe()
        if lp and lp["checkpoint"] not in done:
            path = find_ckpt(lp["checkpoint"])
            if path is not None:
                done.add(lp["checkpoint"]); run(int(lp["step"]), path)
            else:
                done.add(lp["checkpoint"]); rec = {"ts": now_iso(), "step": lp["step"], "checkpoint": lp["checkpoint"], "skipped": "checkpoint no longer on disk"}
                OUTL.open("a").write(json.dumps(rec) + "\n"); print(json.dumps(rec), flush=True)
    except Exception as exc:
        print(json.dumps({"ts": now_iso(), "error": repr(exc)[:300]}), flush=True)
    time.sleep(POLL)
