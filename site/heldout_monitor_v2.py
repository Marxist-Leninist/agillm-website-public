"""Held-out monitor v2 = v1 probe loop + a held-out-guarded LR governor.
Every INTERVAL s: FLUSH_NOW -> wait for training_latest.json -> ctxlen_probe.py (plain full stack, CPU) ->
append to heldout_monitor.jsonl. Then the governor (state in lr_governor_state.json, events in
lr_governor_events.jsonl) adjusts /workspace/agillm43_lr_override.json:
  * backoff  : w257 > best + BACKOFF_DELTA  -> lr_target /= 2 (floor LR_FLOOR), applied immediately, cooldown
  * emergency: w257 > best + EMERGENCY_DELTA -> lr_target /= 4
  * escalate : last 3 probes net-improving (w257[-1] <= w257[-3] - IMPROVE_DELTA), at/near best, no cooldown,
               lr_target < LR_CAP -> lr_target *= ESCALATE (continuous ramp from the current effective LR)
The runtime applies max(schedule, lr * clamp((seen_tok - since)/ramp, 0, 1)), so a change is encoded with
since_seen_tok chosen to continue from the current effective value (no dip to zero).
Context-safe: one JSON line per probe / per governor event; no progress bars."""
import json, os, subprocess, sys, time, pathlib
SAVE = pathlib.Path("/workspace/agillm43_production_v14/production_quality_v18_hard_spike_gates/checkpoints_continue400B_from200B")
OUT = pathlib.Path("/workspace/agillm43_serving_probe/heldout_monitor.jsonl")
GOV_STATE = pathlib.Path("/workspace/agillm43_serving_probe/lr_governor_state.json")
GOV_EVENTS = pathlib.Path("/workspace/agillm43_serving_probe/lr_governor_events.jsonl")
OVERRIDE = pathlib.Path("/workspace/agillm43_lr_override.json")
INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
GOVERN = (sys.argv[2] if len(sys.argv) > 2 else "on") == "on"
LR_CAP = float(os.environ.get("GOV_LR_CAP", "2e-5")); LR_FLOOR = 1e-6; ESCALATE = 1.5
BACKOFF_DELTA = 0.03; FAST_BACKOFF_DELTA = 0.10; EMERGENCY_DELTA = 0.30; IMPROVE_DELTA = 0.005; NEAR_BEST = 0.008
BACKOFF_CONFIRM_PROBES = 2   # probe noise is ~0.03-0.07 nats; one elevated probe is not evidence
COOLDOWN_PROBES = 3; RAMP_TOKENS = 10_000_000
TOK_PER_STEP = 60 * 2048
def now_iso(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
def pointer():
    try: return json.loads((SAVE / "training_latest.json").read_text())
    except Exception: return {}
def atomic_write(p, obj):
    t = p.with_name(p.name + ".tmp")
    with t.open("w") as f: json.dump(obj, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(t, p)
def gov_load():
    try: return json.loads(GOV_STATE.read_text())
    except Exception: return {"hist": [], "best": None, "cooldown": 0, "lr_target": None}
def event(**kw):
    kw["ts"] = now_iso(); GOV_EVENTS.open("a").write(json.dumps(kw) + "\n"); print(json.dumps(kw), flush=True)
LOG = pathlib.Path("/workspace/agillm43_production_v14/production_quality_v18_hard_spike_gates/logs/singlefile_intelligence_v25_satvar_pretrain.log")
def live_seen_tok(fallback):
    """Latest seen_tok the trainer printed (heartbeat / lr-governor lines); the checkpoint pointer lags by a probe."""
    try:
        import re
        with LOG.open("rb") as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 400_000)); tail = f.read().decode("utf-8", "replace")
        m = re.findall(r"seen_tok=(\d+)", tail)
        return float(m[-1]) if m else float(fallback)
    except Exception:
        return float(fallback)
def current_override():
    try: return json.loads(OVERRIDE.read_text())
    except Exception: return {}
def effective_lr(ov, seen):
    lr = float(ov.get("lr") or 0.0); since = float(ov.get("since_seen_tok") or 0.0); ramp = float(ov.get("ramp_tokens") or 0.0)
    if ramp > 0: lr *= min(1.0, max(0.0, (seen - since) / ramp))
    return lr
def set_lr(new_lr, seen, reason, immediate):
    ov = current_override(); old_eff = effective_lr(ov, seen)
    new = dict(ov); new["lr"] = float(new_lr)
    if immediate or old_eff <= 0 or new_lr <= old_eff:
        new["ramp_tokens"] = RAMP_TOKENS; new["since_seen_tok"] = int(seen - RAMP_TOKENS)   # fully ramped now
    else:
        # continue from old_eff: frac0 = old_eff/new_lr at seen -> since = seen - frac0*ramp
        new["ramp_tokens"] = RAMP_TOKENS; new["since_seen_tok"] = int(seen - (old_eff / new_lr) * RAMP_TOKENS)
    new["governor"] = {"schema": "heldout-guarded-lr-governor.v1", "reason": reason, "prev_lr": ov.get("lr"), "prev_effective": old_eff,
                       "cap": LR_CAP, "floor": LR_FLOOR, "updated_utc": now_iso(), "by": "heldout_monitor_v2 (claude-fable-cowork-lrfix-20260906)"}
    new["updated_utc"] = now_iso()
    atomic_write(OVERRIDE, new)
    event(kind="lr_change", reason=reason, old_lr=ov.get("lr"), old_effective=old_eff, new_lr=float(new_lr), since_seen_tok=new["since_seen_tok"], ramp_tokens=new["ramp_tokens"])
def govern(rec):
    st = gov_load(); w = rec.get("w257")
    if w is None: return
    st["hist"] = (st.get("hist") or [])[-24:] + [{"ts": rec["ts"], "step": rec["step"], "w257": w, "w2048": rec.get("w2048")}]
    best_prev = st.get("best"); st["best"] = w if best_prev is None else min(best_prev, w)
    ov = current_override(); seen = live_seen_tok(pointer().get("seen_tok") or 0.0) + TOK_PER_STEP
    if st.get("lr_target") is None: st["lr_target"] = float(ov.get("lr") or 0.0)
    lr_t = float(st["lr_target"]); eff = effective_lr(ov, seen); action = "hold"
    over = best_prev is not None and w > best_prev + BACKOFF_DELTA
    st["over_streak"] = (int(st.get("over_streak", 0) or 0) + 1) if over else 0
    fast = best_prev is not None and w > best_prev + FAST_BACKOFF_DELTA
    pend = st.get("pending_rescind") or None
    # during a cooldown, "still getting worse" must clear the noise floor too, not IMPROVE_DELTA (0.005)
    worse_than_last = len(st["hist"]) >= 2 and w > float(st["hist"][-2]["w257"]) + BACKOFF_DELTA
    if best_prev is not None and w > best_prev + EMERGENCY_DELTA:
        lr_t = max(LR_FLOOR, lr_t / 4.0); set_lr(lr_t, seen, f"EMERGENCY held-out {w:.4f} > best {best_prev:.4f} + {EMERGENCY_DELTA}", True); st["cooldown"] = COOLDOWN_PROBES * 2; action = "emergency_backoff"
        st["pending_rescind"] = None
    elif over and (fast or st["over_streak"] >= BACKOFF_CONFIRM_PROBES) and (st.get("cooldown", 0) == 0 or worse_than_last or fast):
        # confirmed regression vs best: halve. One elevated probe is inside probe noise, so a slow
        # backoff needs two in a row; a single >+0.10 jump still reacts immediately. During a backoff
        # cooldown only halve again if still getting worse probe-to-probe.
        prev_target = lr_t
        lr_t = max(LR_FLOOR, lr_t / 2.0)
        set_lr(lr_t, seen, f"backoff: held-out {w:.4f} > best {best_prev:.4f} + {BACKOFF_DELTA} on {st['over_streak']} consecutive probe(s)" + (" (fast: > +%.2f)" % FAST_BACKOFF_DELTA if fast else ""), True)
        st["cooldown"] = COOLDOWN_PROBES; action = "backoff"
        st["pending_rescind"] = {"restore_lr": float(prev_target), "best_prev": float(best_prev), "step": rec.get("step"), "trigger_w257": w, "ts": now_iso()}
    else:
        h = [x["w257"] for x in st["hist"]]
        ramped = eff >= 0.98 * float(ov.get("lr") or 0.0)
        if pend and best_prev is not None and w <= float(pend.get("best_prev", w)):
            # the probe that triggered the last backoff was noise: held-out is back at/below the
            # pre-backoff best, so restore the LR the governor had (continuous ramp, no dip).
            lr_t = min(LR_CAP, float(pend["restore_lr"]))
            set_lr(lr_t, seen, f"rescind backoff: held-out {w:.4f} <= pre-backoff best {float(pend['best_prev']):.4f} (trigger {pend.get('trigger_w257')} was noise)", False)
            st["cooldown"] = 0; st["pending_rescind"] = None; action = "rescind_backoff"
        elif st.get("cooldown", 0) > 0: st["cooldown"] -= 1; action = "hold_cooldown"
        elif len(h) >= 3 and ramped and h[-1] <= h[-3] - IMPROVE_DELTA and h[-1] <= st["best"] + NEAR_BEST and lr_t < LR_CAP:
            new_t = min(LR_CAP, lr_t * ESCALATE); set_lr(new_t, seen, f"escalate: improving {h[-3]:.4f} -> {h[-1]:.4f} over 3 probes at lr {eff:.2e}", False); lr_t = new_t; st["cooldown"] = 2; action = "escalate"
        elif not ramped: action = "hold_ramping"
    st["lr_target"] = lr_t; st["last_action"] = action; st["last_effective_lr"] = eff; st["updated_utc"] = now_iso()
    atomic_write(GOV_STATE, st)
    event(kind="probe", step=rec["step"], w257=w, w2048=rec.get("w2048"), best=st["best"], effective_lr=eff, lr_target=lr_t, action=action, cooldown=st.get("cooldown", 0), over_streak=st.get("over_streak", 0))
last_probed = None
while True:
    t0 = time.time(); before = pointer(); flag = SAVE / "FLUSH_NOW"
    if not flag.exists():
        try: flag.touch(exist_ok=False)
        except FileExistsError: pass
    ckpt = None
    for _ in range(240):
        time.sleep(5); now = pointer()
        if now.get("record_sha256") and now.get("record_sha256") != before.get("record_sha256"):
            ckpt = now; break
    if ckpt is None:
        OUT.open("a").write(json.dumps({"ts": now_iso(), "error": "flush_timeout", "pointer_step": before.get("step")}) + "\n")
    else:
        path = ckpt["checkpoint_path"]; step = ckpt["step"]
        if step != last_probed and pathlib.Path(path).is_file():
            res_path = f"/workspace/agillm43_serving_probe/ctxlen_{step}.json"
            env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = ""
            p = subprocess.run(["nice", "-n", "10", "python3", "/workspace/agillm43_serving_probe/ctxlen_probe.py", path, res_path], env=env, capture_output=True, text=True, timeout=1500)
            rec = {"ts": now_iso(), "step": step, "checkpoint": pathlib.Path(path).name}
            try:
                r = json.load(open(res_path)); rec.update({"w257": r["w257"]["ce"], "w2048": r["w2048"]["ce"], "w2048_by_pos": r["w2048"]["by_position"], "seconds": r["seconds"]})
            except Exception as exc:
                rec["error"] = f"probe_failed rc={p.returncode} {exc!r} tail={p.stderr[-300:]}"
            OUT.open("a").write(json.dumps(rec) + "\n"); last_probed = step
            if GOVERN and "w257" in rec:
                try: govern(rec)
                except Exception as exc: event(kind="governor_error", error=repr(exc)[:300])
    time.sleep(max(60, INTERVAL - (time.time() - t0)))
