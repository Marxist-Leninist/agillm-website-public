"""AGILLM UI + checkpoint feed + MCP server, bolted onto the Codex monitor.

Added 2026-09-04. Stdlib only. Everything here is read-only with respect to the
training run: it reads the newest checkpoint provenance sidecar over SSH and
proxies generation to the existing local bridge tunnel. Nothing writes to the
vast box.
"""
from __future__ import annotations

import json, math, os, re, subprocess, threading, time
from pathlib import Path
from urllib.request import urlopen, Request

import gb10_live  # GB10 RPV16 live feed (2026-09-20)
import standard_live  # Standard 1.1B live feed (2026-09-23, cowork)
import curve_history  # full-lineage loss-curve history (2026-09-23)
import live_telemetry  # live GB10 telemetry block (2026-09-23)

HERE = Path(__file__).resolve().parent
VAST_ROUTE = Path("/var/lib/agillm43-vast-route/gb10-verified-direct.json")
VAST_DIRECT_ROUTE = VAST_ROUTE
GB10_KNOWN_HOSTS = "/var/lib/agillm43-vast-route/gb10_known_hosts"
GB10_HOST_ALIAS = "agillm44-gb10-51049010"
LIVE_STATUS = Path("/opt/agillm43_netserve/agillm_vast_live_status.json")
SSH_KEY = "/root/.ssh/id_ed25519"
CKPT_DIR = "/workspace/agillm-gb10-1pf-targetfix-active"
TRAIN_LOG = "/workspace/rpv16-keepalive-trainer.log"
BRIDGE = os.environ.get("AGILLM_BRIDGE_URL", "http://127.0.0.1:8494")
_TOKEN_FILE = HERE / ".bridge_token"

_LOCK = threading.Lock()
_CKPT: dict = {"ok": False, "reason": "not polled yet", "polled_at": 0}

# Fast MCP status tail cache. Status requests must never block on the Vast SSH link:
# the SG gateway has a bounded backend read window, while a transient SSH stall can
# legitimately take tens of seconds. Keep the last good tail and refresh it in a
# single daemon thread instead of doing live SSH on the request path.
_TAIL_CACHE: dict = {
    "lines": [], "polled_at": 0, "attempted_at": 0,
    "refreshing": False, "last_error": "not polled yet",
}
_TAIL_MIN_INTERVAL = max(5, int(os.environ.get("AGILLM_TAIL_INTERVAL", "20")))
_TAIL_FETCH_LINES = 400  # retain the tool's full documented 400-line maximum



def bridge_token() -> str:
    try:
        return _TOKEN_FILE.read_text().strip()
    except Exception:
        return os.environ.get("AGILLM_BRIDGE_TOKEN", "")


def vast_target():
    """Prefer the separately verified direct route; fall back to provider SSH metadata."""
    for route_path in (VAST_DIRECT_ROUTE, VAST_ROUTE):
        try:
            d = json.loads(route_path.read_text())
            h, p = d.get("ssh_host"), d.get("ssh_port")
            if route_path == VAST_DIRECT_ROUTE:
                if d.get("instance_id") != 51049010:
                    continue
                verified_at = float(d.get("verified_at_epoch") or 0)
                if time.time() - verified_at > 3600:
                    continue
            if h and p:
                return str(h), str(p), d
        except Exception:
            continue
    return None, None, {}


def ssh_vast(remote_cmd: str, timeout: int = 45) -> str:
    host, port, _ = vast_target()
    if not host:
        return ""
    try:
        r = subprocess.run(
            ["ssh", "-F", "/dev/null", "-i", SSH_KEY, "-p", port, "-o", "BatchMode=yes",
             "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=12",
             "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + GB10_KNOWN_HOSTS,
             "-o", "GlobalKnownHostsFile=/dev/null", "-o", "HostKeyAlias=" + GB10_HOST_ALIAS,
             "-o", "LogLevel=ERROR", f"root@{host}", remote_cmd],
            capture_output=True, text=True, timeout=timeout, errors="replace")
        return r.stdout
    except Exception:
        return ""


def ssh_vast_argv():
    """SSH argv prefix (without remote command) for the verified GB10 route, or None."""
    host, port, _ = vast_target()
    if not host:
        return None
    return ["ssh", "-F", "/dev/null", "-i", SSH_KEY, "-p", str(port), "-o", "BatchMode=yes",
            "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=12",
            "-o", "StrictHostKeyChecking=yes", "-o", "UserKnownHostsFile=" + GB10_KNOWN_HOSTS,
            "-o", "GlobalKnownHostsFile=/dev/null", "-o", "HostKeyAlias=" + GB10_HOST_ALIAS,
            "-o", "LogLevel=ERROR", f"root@{host}"]


def bridge_get(path: str, timeout: float = 8.0):
    try:
        req = Request(BRIDGE + path, headers={"X-AGILLM-Token": bridge_token()})
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def bridge_post(path: str, payload: dict, timeout: float = 300.0):
    try:
        body = json.dumps(payload).encode()
        req = Request(BRIDGE + path, data=body, method="POST",
                      headers={"Content-Type": "application/json",
                               "X-AGILLM-Token": bridge_token()})
        with urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------- checkpoint
_REMOTE = (
    "D=" + CKPT_DIR + "; "
    "P=$(ls -1t \"$D\"/*.pt 2>/dev/null | head -1); "
    "S=$(ls -1t \"$D\"/*.provenance.json 2>/dev/null | head -1); "
    "if [ -z \"$S\" ] && [ -s \"$D/training_latest.json\" ]; then S=\"$D/training_latest.json\"; fi; "
    "echo \"PT=$P\"; "
    "if [ -n \"$P\" ]; then echo \"SIZE=$(stat -c %s \"$P\")\"; "
    "echo \"MTIME=$(stat -c %Y \"$P\")\"; fi; "
    "echo \"COUNT=$(ls -1 \"$D\"/*.pt 2>/dev/null | wc -l)\"; "
    "echo \"DIRBYTES=$(du -sb \"$D\" 2>/dev/null | cut -f1)\"; "
    "echo '<<PROV>>'; if [ -n \"$S\" ]; then cat \"$S\"; fi"
)


def _i(v):
    try:
        return int(str(v).strip())
    except Exception:
        return 0


def _status_health() -> dict:
    # Prefer fresh resilient multi-route live-status bridge health.
    try:
        age = time.time() - LIVE_STATUS.stat().st_mtime
        if age <= 900:
            d = json.loads(LIVE_STATUS.read_text())
            h = d.get("bridge")
            if isinstance(h, dict) and h.get("ok") is True and h.get("ready") is True:
                out = dict(h)
                out.setdefault("health_source", "verified_local_live_status")
                out["status_cache_age_s"] = round(age, 3)
                return out
    except Exception:
        pass
    return bridge_get("/health")


def _checkpoint_from_local_status() -> dict:
    try:
        d = json.loads(LIVE_STATUS.read_text())
        name = str(d.get("last_saved_file") or d.get("file") or "")
        step = d.get("last_saved_step") if d.get("last_saved_step") is not None else d.get("step")
        health = _status_health()
        if not name or step is None:
            return {}
        return {
            "ok": True, "polled_at": int(time.time()),
            "checkpoint_name": name, "checkpoint_path": "",
            "size_bytes": 0, "saved_at_unix": 0, "checkpoint_count": 0, "dir_bytes": 0,
            "step": step, "effective_global_step": None,
            "seen_tok": d.get("seen_tok"), "effective_seen_tok": d.get("effective_seen_tok"),
            "loss": d.get("loss"), "loss_ar": None, "loss_sat": None, "loss_nat": None,
            "batch_size": None, "block_size": None,
            "created_at_iso": d.get("checkpoint_saved_at"),
            "warmstart_source": None, "train_script": None, "inference_samples": [],
            "hostname": (health or {}).get("host"), "checkpoint_type": "full",
            "health": health, "source": "verified_local_live_status_fallback",
        }
    except Exception:
        return {}


def poll_checkpoint() -> dict:
    """Read the active RPV16 target-fix-v6 lineage without loading the 11GB checkpoint."""
    import shlex
    qd = shlex.quote(CKPT_DIR)
    qf = shlex.quote(TRAIN_LOG)
    remote = (
        "set -e; D=" + qd + "; F=" + qf + "; P=\"$D/latest.pt\"; "
        "echo \"PT=$P\"; "
        "if [ -s \"$P\" ]; then echo \"SIZE=$(stat -c %s \"$P\")\"; "
        "echo \"MTIME=$(stat -c %Y \"$P\")\"; fi; "
        "echo \"DIRBYTES=$(du -sb \"$D\" 2>/dev/null | cut -f1)\"; "
        "echo '<<CHECKPOINT>>'; grep -F '\"event\": \"checkpoint\"' \"$F\" | tail -1 || true; "
        "echo '<<TRAIN>>'; grep -F '\"event\": \"train\"' \"$F\" | tail -1 || true; "
        "echo '<<RESUME>>'; grep -F '\"event\": \"resumed\"' \"$F\" | tail -1 || true"
    )
    raw = ssh_vast(remote, timeout=35)
    out = {"ok": False, "polled_at": int(time.time()), "branch": "rpv16-targetfix-v6"}
    if not raw:
        cached = _checkpoint_from_local_status()
        if cached:
            cached["branch"] = "legacy-status-fallback"
            return cached
        out["reason"] = "GB10 target-fix route unreachable"
        out["health"] = _status_health()
        return out

    meta, _, rest = raw.partition("<<CHECKPOINT>>")
    ck_raw, _, rest = rest.partition("<<TRAIN>>")
    train_raw, _, resume_raw = rest.partition("<<RESUME>>")
    for line in meta.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            out[k.strip().lower()] = v.strip()

    def jload(blob):
        try:
            line = [x for x in blob.splitlines() if x.strip()][-1]
            v = json.loads(line)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}

    ck = jload(ck_raw)
    live = jload(train_raw)
    resumed = jload(resume_raw)
    path = out.get("pt", "")
    ts = float(ck.get("ts") or out.get("mtime") or 0)
    health = _status_health()

    # The dedicated RPV16 CPU service is deliberately reported separately from
    # the legacy public bridge. During checkpoint promotion these can differ.
    try:
        with urlopen("http://127.0.0.1:18620/health", timeout=1.5) as r:
            rpv_health = json.loads(r.read().decode("utf-8", "replace"))
    except Exception as exc:
        rpv_health = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    out.update({
        "ok": bool(path and ck.get("step") is not None),
        "checkpoint_name": Path(path).name if path else "",
        "checkpoint_path": path,
        "size_bytes": _i(out.get("size")),
        "saved_at_unix": int(ts or 0),
        "checkpoint_count": 1 if path else 0,
        "dir_bytes": _i(out.get("dirbytes")),
        "step": ck.get("step"),
        "effective_global_step": ck.get("optimizer_update_clock"),
        "seen_tok": ck.get("seen_tokens"),
        "effective_seen_tok": ck.get("seen_tokens"),
        "loss": live.get("loss"),
        "loss_ar": live.get("loss") if live.get("objective") == "ar" else None,
        "loss_sat": live.get("loss") if live.get("objective") == "sat" else None,
        "loss_nat": live.get("loss") if live.get("objective") == "nat" else None,
        "batch_size": 24,
        "block_size": 2048,
        "created_at_iso": (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))
                           if ts else None),
        "warmstart_source": resumed.get("path"),
        "train_script": "agillm_gb10_1pf.rpv16_pf2_topology_satpartner_fullnll_moeresidual_dataretry_20260920.py",
        "inference_samples": [],
        "hostname": "GB10-51049010",
        "checkpoint_type": "RPV16 recovery v6",
        "checkpoint_schema": ck.get("schema"),
        "target_alignment": ck.get("target_alignment"),
        "checkpoint_stall_s": ck.get("stall_s"),
        "checkpoint_async": ck.get("async"),
        "live_step": live.get("step"),
        "live_seen_tok": live.get("seen_tokens"),
        "live_loss": live.get("loss"),
        "live_tok_s": live.get("tok_s"),
        "live_pipeline_tok_s": live.get("pipeline_tok_s"),
        "live_objective": live.get("objective"),
        "health": health,
        "serving": {
            "public_bridge": health,
            "rpv16_cpu": rpv_health,
            "note": "training and serving lineages are shown separately until CPU promotion completes",
        },
        "instance": {"id": 51049010, "label": "agillm44-gb10-benchmark-recovery-20260914",
                     "status": "running", "observed_at": int(time.time())},
        "source": "gb10-targetfix-v6-json-log",
    })
    return out

# ------------------------------------------------------------ training curves
TRAINING_CMD = "python3 /workspace/agillm43_serving_probe/training_curves.py --png"
TRAINING_PNG_CMD = "base64 -w0 /workspace/agillm43_serving_probe/training_curves.png 2>/dev/null"
_TRAIN_PNG: dict = {"bytes": b"", "at": 0}
TRAINING_MIN_INTERVAL = int(os.environ.get("AGILLM_CURVES_INTERVAL", "20"))
RECOVERED_TRAINING_HISTORY = "/opt/agillm43_netserve/recovered_training_history.json"
_TRAIN: dict = {"ok": False, "reason": "not polled yet", "polled_at": 0}
_TRAIN_LAST = 0.0

def _training_fallback_from_local_status(reason: str) -> dict:
    """Return current GB10 checkpoint status when the historical curve collector is absent.

    This keeps the UI operational and honest: checkpoint freshness is live, while
    curve history remains explicitly unavailable until its collector is restored.
    """
    now = time.time()
    try:
        snap = checkpoint_snapshot()
        polled = float(snap.get("polled_at") or 0)
        step = int(snap.get("step") or 0)
        if snap.get("ok") is True and step > 0 and polled > 0 and now - polled <= 180:
            health = {
                "overall": "maintenance",
                "summary": "GB10 checkpoint feed is live; full training-curve history is temporarily unavailable",
                "source": "gb10-checkpoint-live",
                "degraded": True,
                "curves_available": False,
                "trainer_live_step": step,
                "checkpoint_age_s": round(max(0.0, now - polled), 3),
                "last_error": reason,
                "areas": {
                    "trainer": {"state": "maintenance", "level": "maintenance", "step": step},
                    "model": {"level": "maintenance", "fresh": False},
                },
                "flags": [{"level": "maintenance", "area": "trainer",
                           "message": "current GB10 checkpoint is visible; historical curve collector is reconnecting"}],
            }
            return {
                "ok": True, "degraded": True, "source": "gb10-checkpoint-live",
                "reason": reason, "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "polled_at": int(now),
                "latest": {"step": step, "loss": snap.get("loss"),
                           "checkpoint_name": snap.get("checkpoint_name"),
                           "seen_tok": snap.get("seen_tok")},
                "fix_step": None, "health": health, "curves_available": False,
                "series": [], "markers": [], "since_fix": {}, "all_time": {},
            }
    except Exception:
        pass
    # Final fallback: the independently maintained serving/trainer mirror.
    try:
        age = now - LIVE_STATUS.stat().st_mtime
        if age > 900:
            return {}
        d = json.loads(LIVE_STATUS.read_text())
        bridge = d.get("bridge") or {}
        step = d.get("trainer_live_step")
        if not (isinstance(step, int) and step > 0 and bridge.get("ok") is True and bridge.get("ready") is True):
            return {}
        verified_at = d.get("verified_at") or d.get("status_updated_at")
        health = {
            "overall": "maintenance", "summary": "using last verified local status mirror",
            "source": "verified-local-status", "degraded": True, "curves_available": False,
            "trainer_live_step": step, "serving_step": d.get("serving_step"),
            "bridge_ready": True, "mirror_age_s": round(age, 3), "last_error": reason,
        }
        return {
            "ok": True, "degraded": True, "source": "verified-local-status",
            "reason": reason, "generated_at": verified_at, "polled_at": int(now),
            "latest": {"step": step, "serving_step": d.get("serving_step"),
                       "serving_file": d.get("serving_file"), "loss": d.get("loss")},
            "fix_step": None, "health": health, "curves_available": False,
            "series": [], "markers": [], "since_fix": {}, "all_time": {},
        }
    except Exception:
        return {}


def poll_training() -> dict:
    """Parse recent JSONL telemetry from the active target-fix-v6 trainer."""
    import shlex
    qf = shlex.quote(TRAIN_LOG)
    remote = (
        "set -e; F=" + qf + "; "
        "stat -c 'AGILLM_CURVE_META %s %Y' -- \"$F\"; "
        "tail -n 3500 -- \"$F\""
    )
    raw = ssh_vast(remote, timeout=35)
    if not raw.strip():
        fallback = _training_fallback_from_local_status("GB10 target-fix training log unreachable")
        return fallback or {"ok": False, "reason": "GB10 target-fix training log unreachable",
                            "polled_at": int(time.time())}

    log_bytes = log_mtime = None
    points, checkpoints, resumes, configs = [], [], [], []
    for line in raw.splitlines():
        if line.startswith("AGILLM_CURVE_META "):
            try:
                _, n, mt = line.split(None, 2)
                log_bytes, log_mtime = int(n), int(float(mt))
            except Exception:
                pass
            continue
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        ev = d.get("event")
        if ev == "train" and d.get("step") is not None:
            wall = float(d.get("step_wall_s") or d.get("step_s") or 0.0)
            p = {
                "pid": None,
                "step": int(d["step"]),
                "tok": int(d.get("seen_tokens") or 0),
                "loss": float(d.get("loss") or 0.0),
                "lr": float(d.get("lr") or 0.0),
                "tps": float(d.get("tok_s") or 0.0),
                "peak": round(float(d.get("cuda_reserved") or 0) / (1024**3), 3),
                "step_wall_s": wall,
                "objective": d.get("objective"),
                "train_stage": d.get("train_stage"),
                "global_anchor": bool(d.get("global_anchor")),
                "pipeline_tok_s": float(d.get("pipeline_tok_s") or 0.0),
                "pipeline_target_s": float(d.get("pipeline_target_s") or 0.0),
                "objective_counts": d.get("objective_counts") or {},
                "fresh_input_tokens_total": int(d.get("fresh_input_tokens_total") or 0),
                "optimizer_exposure_tokens_total": int(d.get("optimizer_exposure_tokens_total") or d.get("seen_tokens") or 0),
                "descending_sweep_shared_batch": bool(d.get("descending_sweep_shared_batch")),
                "descending_sweep_cache": bool(d.get("descending_sweep_cache")),
                "descending_sweep_safe_cache": bool(d.get("descending_sweep_safe_cache")),
                "sweep_objective": d.get("sweep_objective"),
                "sparse": d.get("sparse") or {},
                "ts": float(d.get("ts") or 0.0),
            }
            # A restart replays only the unsaved tail. Trim abandoned future
            # rows so the chart and throughput window follow the active branch.
            if points and p["step"] <= points[-1]["step"]:
                points = [x for x in points if x["step"] < p["step"]]
            points.append(p)
        elif ev == "checkpoint":
            checkpoints.append(d)
        elif ev == "resumed":
            resumes.append(d)
        elif ev in ("fullstack_config", "stagefwd_config", "model_ready"):
            configs.append(d)

    # Throughput and quality windows must belong to the currently running branch.
    # A restart from an older durable checkpoint can leave numerically later rows
    # from a rejected branch in the tailed log; cut them at the latest resume step.
    if resumes:
        active_resume_step = int(resumes[-1].get("step") or -1)
        active_resume_ts = float(resumes[-1].get("ts") or 0.0)
        points = [x for x in points if x["step"] > active_resume_step and x.get("ts", 0.0) >= active_resume_ts]

    if not points:
        return {"ok": False, "reason": "no parseable target-fix train events after latest resume",
                "polled_at": int(time.time())}

    alpha, ema = 0.03, None
    enriched = []
    for p in points[-1500:]:
        ema = p["loss"] if ema is None else (1.0-alpha)*ema + alpha*p["loss"]
        q = dict(p)
        q["ema"] = round(ema, 4)
        q["loss"] = round(q["loss"], 4)
        q["tps"] = round(q["tps"], 1)
        enriched.append(q)

    latest = points[-1]
    now = int(time.time())
    age = max(0, now - int(log_mtime or now))
    level = "ok" if age <= 90 else "maintenance"

    def window_metrics(rows):
        rows = [x for x in rows if float(x.get("step_wall_s") or 0) > 0]
        if not rows:
            return {"steps": 0, "tok_s": None}
        # One optimizer-facing exposure is batch(24)*seq(2048)=49152 tokens.
        # Descending-sweep reuse means this is not unique-input throughput.
        wall = sum(float(x["step_wall_s"]) for x in rows)
        fresh_delta = rows[-1].get("fresh_input_tokens_total", 0) - rows[0].get("fresh_input_tokens_total", 0)
        shared_batch = any(bool(x.get("descending_sweep_shared_batch")) for x in rows)
        exposure_rate = round(49152.0 * len(rows) / wall, 1) if wall > 0 else None
        if shared_batch:
            fresh_rate = round(fresh_delta / wall, 1) if wall > 0 and fresh_delta > 0 else None
            semantics = "optimizer exposure; fresh stream reported separately under sweep reuse"
        else:
            fresh_rate = exposure_rate
            semantics = "independent-batch production; fresh stream equals optimizer exposure"
        return {
            "steps": len(rows),
            "tok_s": exposure_rate,
            "optimizer_exposure_tok_s": exposure_rate,
            "fresh_stream_tok_s": fresh_rate,
            "fresh_rate_inferred_from_independent_batch": bool(not shared_batch and exposure_rate is not None),
            "wall_s": round(wall, 3),
            "semantics": semantics,
        }

    w64 = window_metrics(points[-64:])
    w256 = window_metrics(points[-256:])
    objectives = {x.get("objective") for x in points[-256:] if x.get("objective")}
    # after a rollback the tailed log can still hold the abandoned branch's last save
    ck = next((c for c in reversed(checkpoints) if int(c.get("step") or 0) <= latest["step"]), {})
    cfg = next((x for x in reversed(configs) if x.get("event") == "fullstack_config"), {})
    areas = {
        "trainer": {"level": level, "state": "training" if level == "ok" else "maintenance",
                    "step": latest["step"], "pid": None,
                    "heartbeat_age_s": age, "log_age_seconds": age},
        "model": {"level": "maintenance", "fresh": True,
                  "note": "target-fix-v6 trainer is live; CPU serving qualification is tracked separately"},
        "ar": {"level": "ok" if "ar" in objectives else "maintenance"},
        "sat_fixed": {"level": "maintenance"},
        "sat_variable": {"level": "ok" if "sat" in objectives else "maintenance"},
        "nat": {"level": "ok" if "nat" in objectives else "maintenance"},
        "dblock": {"level": level},
        "det50": {"level": "maintenance"},
    }
    flags = []
    if level != "ok":
        flags.append({"level": "maintenance", "area": "trainer",
                      "message": f"target-fix log age is {age}s"})
    flags.append({"level": "maintenance", "area": "serving",
                  "message": "training checkpoint and CPU serving checkpoint are intentionally separate until promotion passes"})

    return {
        "schema": "agillm.gb10.targetfix.training.v2",
        "ok": True,
        "source": "gb10-production-log-live",
        "branch": "rpv16-targetfix-v6",
        "curves_available": True,
        "generated_at": now, "polled_at": now,
        "log_bytes": log_bytes, "log_mtime": log_mtime,
        "heartbeats": len(points), "fix_step": None,
        "latest": {
            "step": latest["step"], "seen_tok": latest["tok"],
            "loss": latest["loss"], "ema": enriched[-1]["ema"],
            "lr": latest["lr"], "raw_tok_s": latest["tps"],
            "fullstack_tok_s_64": w64["tok_s"],
            "fullstack_tok_s_256": w256["tok_s"],
            "pipeline_tok_s": latest["pipeline_tok_s"],
            "pipeline_target_s": latest["pipeline_target_s"],
            "objective": latest["objective"],
            "train_stage": latest["train_stage"],
            "global_anchor": latest["global_anchor"],
            "fresh_input_tokens_total": latest["fresh_input_tokens_total"],
            "optimizer_exposure_tokens_total": latest["optimizer_exposure_tokens_total"],
            "descending_sweep_shared_batch": latest["descending_sweep_shared_batch"],
            "descending_sweep_cache": latest["descending_sweep_cache"],
            "descending_sweep_safe_cache": latest["descending_sweep_safe_cache"],
            "sweep_objective": latest["sweep_objective"],
            "gpu_peak_gb": latest["peak"], "pid": None,
        },
        "checkpoint": {
            "step": ck.get("step"), "seen_tok": ck.get("seen_tokens"),
            "schema": ck.get("schema"), "stall_s": ck.get("stall_s"),
            "async": ck.get("async"), "reason": ck.get("reason"),
            "ts": ck.get("ts"),
        },
        "fullstack": {
            "rolling64": w64, "rolling256": w256,
            "async_checkpoint": bool(cfg.get("async_checkpoint")) if cfg else False,
            "prefetch": cfg.get("prefetch"),
            "prefetch_depth": cfg.get("prefetch_depth"),
            "optimizer_exposure_tokens_per_update": 49152,
            "token_accounting": "optimizer exposures and fresh stream positions are separate under sweep reuse",
            "descending_sweep_safe_cache": bool(cfg.get("descending_sweep_safe_cache")) if cfg else False,
            "descending_sweep_cache_policy": cfg.get("descending_sweep_cache_policy"),
            "checkpoint_policy": cfg.get("checkpoint_policy"),
        },
        "optimization": {
            "verified_sparse_microkernel_tflops": 969.794609,
            "kernel_target_pflops": 1.022,
            "kernel_target_basis": "packed NVFP4 mxf4nvf4 + mma.sp + hardware 2:4 sparsity",
            "kernel_vs_trainer_note": "verified microkernel TFLOP/s and 1.022-PF target are not end-to-end trainer FLOP/s",
            "descending_cache_candidate": {
                "state": "rejected quality gate; rolled back 2026-09-20",
                "policy_tested": "AR/SAT cached; NAT/head exact uncached",
                "canary_sweeps": {"ar": 16, "sat": 2, "nat": 2},
                "baseline_exposure_tok_s": 95851.9,
                "cold_inclusive_exposure_tok_s": 124598.9,
                "warmed_exposure_tok_s": 149420.9,
                "warmed_speedup_x": 1.5589,
                "matched_median_local_speedup_x": 2.211784802927254,
                "rollback_source_sha256": "70cc853b6b438b339fae28f0b17b6f2d6e4cbbfb4ecead92d650f505ad037c78",
            },
            "matched_quality_gate": {
                "common_start_step": 204303,
                "updates_each": 340,
                "median_local_speedup_x": 2.211784802927254,
                "fixed_text_nll_delta_safe_minus_baseline": 0.017353057861328125,
                "training_exclusion_verified": False,
                "accepted": False,
                "status": "rejected; production rolled back to independent-batch source",
            },
            "masked_quant_candidate": {
                "state": "rejected correctness",
                "speedup_pct": 1.096, "max_loss_delta": 0.000815,
            },
            "async_checkpoint_candidate": {
                "state": "rejected canary; production remains synchronous",
                "production_async_checkpoint": bool(cfg.get("async_checkpoint")) if cfg else False,
            },
        },
        "train": [dict({k: p[k] for k in ("step","ema","loss","lr","tps","peak","tok")},
                       objective=p.get("objective"), global_anchor=bool(p.get("global_anchor")),
                       train_stage=p.get("train_stage"), ts=(round(p["ts"], 1) if p.get("ts") else None))
                  for p in enriched],
        "anchors": [], "val": [], "probes": [], "governor": [],
        "markers": [{"step": int(x.get("step") or 0), "kind": "launch", "label": "resume",
                     "checkpoint": Path(str(x.get("path") or "")).name,
                     "run_id": "rpv16-targetfix-v6"} for x in resumes[-60:]],
        "multimode": [], "multimode_current": [], "modes": [],
        "mode_meta": {
            "log_age_seconds": age,
            "sat_fixed": {"status": "not-production-objective", "step_lag": None},
            "nat": {"status": "live" if "nat" in objectives else "missing", "step_lag": 0},
            "sat_var": {"status": "live" if "sat" in objectives else "missing", "step_lag": 0},
        },
        "timeline": [],
        "train_meta": {
            "signal": "GB10 target-fix-v6 JSONL mixed-objective trainer",
            "ema_alpha_per_heartbeat": alpha,
            "max_chart_points": 1500,
            "source_log": TRAIN_LOG,
            "read_only_remote_collector": True,
        },
        "health": {"overall": level, "summary": "Live GB10 target-fix-v6 full-stack trainer",
                   "areas": areas, "flags": flags, "checked_at": now},
    }

def _render_gb10_tail_png(doc: dict) -> bytes:
    """Render a compact loss/EMA PNG locally on GETH from the live JSON feed."""
    try:
        from io import BytesIO
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return b""
    rows = [p for p in (doc.get("train") or [])
            if isinstance(p, dict) and p.get("step") is not None and p.get("loss") is not None]
    if len(rows) < 2:
        return b""
    W, H = 1200, 700
    im = Image.new("RGB", (W, H), (11, 15, 20))
    dr = ImageDraw.Draw(im)
    def ft(size, bold=False):
        names = (["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else []) + ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
        for name in names:
            try: return ImageFont.truetype(name, size)
            except Exception: pass
        return ImageFont.load_default()
    titlef, bodyf, smallf = ft(30, True), ft(20), ft(16)
    latest = doc.get("latest") or {}
    dr.text((50, 28), "AGILLM GB10 live training loss", font=titlef, fill=(230, 237, 243))
    subtitle = "step {:,}  loss {:.3f}  EMA {:.3f}  LR {:.2e}  {:,.0f} raw tok/s".format(
        int(latest.get("step") or 0), float(latest.get("loss") or 0), float(latest.get("ema") or 0),
        float(latest.get("lr") or 0), float(latest.get("raw_tok_s") or 0))
    dr.text((50, 72), subtitle, font=bodyf, fill=(139, 152, 171))
    x0, y0, x1, y1 = 88, 130, 1155, 615
    dr.rectangle((x0, y0, x1, y1), fill=(8, 11, 17), outline=(42, 51, 70), width=2)
    steps = [int(p["step"]) for p in rows]
    raw = [float(p["loss"]) for p in rows]
    ema = [float(p.get("ema", p["loss"])) for p in rows]
    ranked = sorted(raw + ema)
    lo = ranked[max(0, int(len(ranked) * .01) - 1)]
    hi = ranked[min(len(ranked)-1, int(len(ranked) * .99))]
    if hi <= lo: hi = lo + 1.0
    pad = max(.05, (hi-lo)*.08); lo -= pad; hi += pad
    s0, s1 = min(steps), max(steps)
    def X(s): return x0 + 18 + (s-s0)/max(1,s1-s0)*(x1-x0-36)
    def Y(v):
        v=max(lo,min(hi,v)); return y0 + 22 + (hi-v)/(hi-lo)*(y1-y0-52)
    for i in range(6):
        frac=i/5; yy=y0+22+frac*(y1-y0-52); val=hi-frac*(hi-lo)
        dr.line((x0+18,yy,x1-18,yy),fill=(29,37,54),width=1)
        dr.text((8,yy-9),f"{val:.2f}",font=smallf,fill=(139,152,171))
    rawpts=[(X(s),Y(v)) for s,v in zip(steps,raw)]
    emapts=[(X(s),Y(v)) for s,v in zip(steps,ema)]
    if len(rawpts)>1: dr.line(rawpts,fill=(112,122,138),width=1)
    if len(emapts)>1: dr.line(emapts,fill=(224,168,82),width=4)
    dr.text((x0+22,y1+20),f"{s0:,}",font=smallf,fill=(139,152,171))
    endtxt=f"{s1:,}"; tw=dr.textbbox((0,0),endtxt,font=smallf)[2]
    dr.text((x1-22-tw,y1+20),endtxt,font=smallf,fill=(139,152,171))
    dr.line((830,74,875,74),fill=(112,122,138),width=2); dr.text((884,62),"raw",font=smallf,fill=(139,152,171))
    dr.line((960,74,1005,74),fill=(224,168,82),width=4); dr.text((1014,62),"EMA",font=smallf,fill=(230,237,243))
    dr.text((50, 660), "Source: active GB10 RPV16 target-fix-v6 JSONL trainer · refreshed by GETH read-only collector", font=smallf, fill=(139,152,171))
    out=BytesIO(); im.save(out, format="PNG", optimize=True); return out.getvalue()

def _png_font(size, bold=False):
    from PIL import ImageFont
    names = (["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"] if bold else []) + ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _render_history_png(doc: dict, hist: dict) -> bytes:
    """Full-lineage RPV16 chart: per-objective bucket means, abandoned branches, held-out CE, unigram reference."""
    from io import BytesIO
    from PIL import Image, ImageDraw
    C = {n: i for i, n in enumerate(hist.get("cols") or [])}
    rows = hist.get("rows") or []
    if len(rows) < 2 or "s0" not in C:
        return b""
    W, H = 1400, 800
    im = Image.new("RGB", (W, H), (11, 15, 20))
    dr = ImageDraw.Draw(im)
    titlef, bodyf, smallf = _png_font(28, True), _png_font(18), _png_font(15)
    sm = hist.get("summary") or {}
    best = (hist.get("best") or {}).get("full") or (hist.get("best") or {}).get("anc") or (hist.get("best") or {}).get("ar") or {}
    ref = hist.get("reference") or {}
    held = hist.get("heldout") or []
    branches = hist.get("branches") or []
    latest = doc.get("latest") or {}

    def mid(r):
        return (r[C["s0"]] + r[C["s1"]]) / 2.0

    # anc (global-anchor rows, decoded through the full 16-stage stack) is the model's own
    # training CE; the plain "ar" column averages 16 early-exit depths (stage-local objective).
    head = "full" if "full" in C else "anc"
    series = [(head, (88, 166, 255), 4), ("ar", (126, 190, 255), 1), ("sat", (210, 168, 255), 2), ("nat", (255, 166, 87), 2)]
    ys = []
    for r in rows:
        for k, _c, _w in series:
            if r[C[k]] is not None and r[C[k + "_n"]] >= 8:
                ys.append(r[C[k]])
    for b in branches:
        for r in b.get("rows") or []:
            if r[C["ar"]] is not None and r[C["ar_n"]] >= 8:
                ys.append(r[C["ar"]])
    ys += [p["ce"] for p in held]
    if ref.get("unigram_nats") is not None:
        ys.append(float(ref["unigram_nats"]))
    if ref.get("heldout_unigram_ce") is not None:
        ys.append(float(ref["heldout_unigram_ce"]))
    if not ys:
        return b""
    ys.sort()
    lo = ys[max(0, int(len(ys) * 0.005))]
    hi = ys[min(len(ys) - 1, int(len(ys) * 0.995))]
    if hi - lo < 0.3:
        c = (hi + lo) / 2; lo, hi = c - 0.15, c + 0.15
    pad = (hi - lo) * 0.08; lo -= pad; hi += pad
    s0 = rows[0][C["s0"]]
    s1 = max([rows[-1][C["s1"]]] + [b["to_step"] for b in branches if b.get("to_step")])
    x0, y0, x1, y1 = 90, 150, W - 40, H - 90

    def X(s):
        return x0 + (s - s0) / max(1, s1 - s0) * (x1 - x0)

    def Y(v):
        v = max(lo, min(hi, v))
        return y1 - (v - lo) / (hi - lo) * (y1 - y0)

    dr.rectangle((x0, y0, x1, y1), fill=(8, 11, 17), outline=(42, 51, 70), width=2)
    # y grid
    span = hi - lo
    stp = 0.1 if span < 1.2 else 0.25 if span < 3 else 0.5 if span < 6 else 1.0
    v = math.ceil(lo / stp) * stp
    while v <= hi:
        yy = Y(v)
        dr.line((x0, yy, x1, yy), fill=(29, 37, 54), width=1)
        dr.text((x0 - 10, yy), "%.2f" % v, font=smallf, fill=(139, 152, 171), anchor="rm")
        v += stp
    # x grid
    xs = max(1, s1 - s0) / 6.0
    mag = 10 ** math.floor(math.log10(xs))
    xs = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= xs)
    v = math.ceil(s0 / xs) * xs
    while v <= s1:
        xx = X(v)
        dr.line((xx, y0, xx, y1), fill=(29, 37, 54), width=1)
        dr.text((xx, y1 + 8), "{:,}".format(int(v)), font=smallf, fill=(139, 152, 171), anchor="mt")
        v += xs
    dr.text(((x0 + x1) / 2, y1 + 30), "training step", font=smallf, fill=(110, 118, 129), anchor="mt")
    dr.text((20, (y0 + y1) / 2), "CE (nats)", font=smallf, fill=(110, 118, 129), anchor="lm")

    def dashed(p0, p1, fill, width=1, dash=8, gap=6):
        (ax, ay), (bx, by) = p0, p1
        L = math.hypot(bx - ax, by - ay)
        if L <= 0:
            return
        n = int(L // (dash + gap)) + 1
        for i in range(n):
            t0 = i * (dash + gap) / L
            t1 = min(1.0, (i * (dash + gap) + dash) / L)
            if t0 >= 1:
                break
            dr.line((ax + (bx - ax) * t0, ay + (by - ay) * t0, ax + (bx - ax) * t1, ay + (by - ay) * t1), fill=fill, width=width)

    def poly(rs, key, fill, width, min_n=8, dash=False):
        seg = []
        prev = None
        for r in rs:
            val = r[C[key]]
            ok = val is not None and r[C[key + "_n"]] >= min_n
            gap = prev is not None and r[C["s0"]] - prev > 4 * max(1, hist.get("bucket_steps") or 1)
            if not ok or gap:
                if len(seg) > 1:
                    (dashed_poly if dash else solid_poly)(seg, fill, width)
                seg = []
            if ok:
                seg.append((X(mid(r)), Y(val)))
            prev = r[C["s1"]]
        if len(seg) > 1:
            (dashed_poly if dash else solid_poly)(seg, fill, width)

    def solid_poly(seg, fill, width):
        dr.line(seg, fill=fill, width=width, joint="curve")

    def dashed_poly(seg, fill, width):
        for a, b in zip(seg, seg[1:]):
            dashed(a, b, fill, width, dash=5, gap=4)

    # AR p10-p90 band (current lineage); split at gaps so it never bridges missing data
    hk, lk = ("ar_p90", "ar_p10") if "ar_p90" in C else ("hi", "lo")
    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    seg, prev = [], None
    for r in rows + [None]:
        ok = r is not None and r[C[hk]] is not None and r[C[lk]] is not None and r[C["ar_n"]] >= 8
        gap = r is not None and prev is not None and r[C["s0"]] - prev > 4 * max(1, hist.get("bucket_steps") or 1)
        if (not ok or gap) and len(seg) > 1:
            od.polygon([(x, t) for x, t, _b in seg] + [(x, b) for x, _t, b in seg[::-1]], fill=(88, 166, 255, 40))
        if not ok or gap:
            seg = []
        if ok:
            seg.append((X(mid(r)), Y(r[C[hk]]), Y(r[C[lk]])))
        if r is not None:
            prev = r[C["s1"]]
    im.paste(overlay, (0, 0), overlay)
    dr = ImageDraw.Draw(im)
    # abandoned branches (faded)
    clipped = []
    for b in branches:
        bkey = next((k for k in ("full", "anc") if k in C and any(r[C[k]] is not None and r[C[k + "_n"]] >= 8 for r in (b.get("rows") or []))), "ar")
        poly(b.get("rows") or [], bkey, (70, 96, 130), 3)
        for r in b.get("rows") or []:
            v = r[C[bkey]]
            if v is not None and r[C[bkey + "_n"]] >= 8 and v > hi:
                clipped.append((v, mid(r)))
        cut = b.get("cut_step")
        if cut is not None:
            dashed((X(cut), y0), (X(cut), y1), (248, 81, 73), 2)
            dr.text((X(cut) + 6, y0 + 8), "rolled back to {:,}".format(cut), font=smallf, fill=(248, 129, 120))
        if b.get("to_step") and b.get("from_step"):
            bx = (X(b["from_step"]) + X(b["to_step"])) / 2
            dr.text((bx, y0 + 30), "abandoned branch {:,}-{:,}".format(b["from_step"], b["to_step"]), font=smallf,
                    fill=(120, 140, 170), anchor="mt")
    for mk in hist.get("markers") or []:
        if mk.get("kind") != "data_wrap" or mk.get("step") is None or not (s0 <= mk["step"] <= s1):
            continue
        mx = X(mk["step"])
        dashed((mx, y0), (mx, y1), (110, 118, 129), 1, dash=2, gap=6)
        dr.text((mx + 4, y1 - 6), "data wrap" + (" (abandoned)" if mk.get("lineage") == "abandoned" else ""),
                font=smallf, fill=(139, 152, 171), anchor="lb")
    if clipped:
        cv, cs = max(clipped)
        dr.text((x1 - 8, y0 + 8), "abandoned-branch peaks clipped: max %.2f @%s" % (cv, "{:,}".format(int(cs))),
                font=smallf, fill=(120, 140, 170), anchor="rt")
    for key, col, w in series[::-1]:
        poly(rows, key, col, w)
    # unigram reference
    if ref.get("unigram_nats") is not None:
        yy = Y(float(ref["unigram_nats"]))
        dashed((x0, yy), (x1, yy), (227, 179, 65), 2, dash=3, gap=5)
        dr.text((x1 - 8, yy - 4), "unigram, training stream (no context) %.2f" % float(ref["unigram_nats"]), font=smallf, fill=(227, 179, 65), anchor="rb")
    if ref.get("heldout_unigram_ce") is not None:
        yh = Y(float(ref["heldout_unigram_ce"]))
        dashed((x0, yh), (x1, yh), (190, 150, 60), 1, dash=2, gap=6)
        dr.text((x1 - 8, yh - 4), "unigram on held-out set %.2f" % float(ref["heldout_unigram_ce"]), font=smallf, fill=(190, 150, 60), anchor="rb")
    # best AR marker
    if best:
        bx, by = X(best["step"]), Y(best["value"])
        dr.ellipse((bx - 5, by - 5, bx + 5, by + 5), outline=(88, 166, 255), width=2)
        dr.text((bx, by + 10), "best full model %.3f" % best["value"], font=smallf, fill=(88, 166, 255), anchor="mt")
    # held-out CE points
    for p in held:
        px, py = X(p["step"]), Y(p["ce"])
        on_branch = p.get("lineage") == "abandoned"
        dr.ellipse((px - 6, py - 6, px + 6, py + 6), fill=(150, 160, 175) if on_branch else (230, 237, 243),
                   outline=(11, 15, 20), width=2)
        label = "held-out %.3f%s" % (p["ce"], " (abandoned)" if on_branch else "")
        right = px + 9 + dr.textlength(label, font=smallf) > x1 - 4
        dr.text((px - 9, py) if right else (px + 9, py), label, font=smallf,
                fill=(170, 180, 195) if on_branch else (230, 237, 243), anchor="rm" if right else "lm")
    # title + legend
    dr.text((40, 24), "AGILLM RPV16 2.0B (GB10) - full training history", font=titlef, fill=(230, 237, 243))
    ar10 = next((sm.get(k) for k in ("full_mean_last_10k", "anc_mean_last_10k", "ar_mean_last_10k") if sm.get(k) is not None), None)
    sub = "step {:,}  ·  full-model train CE last 10k {}  ·  best {} @{:,}".format(
        int(latest.get("step") or sm.get("last_step") or 0), ("%.3f" % ar10) if ar10 is not None else "—",
        ("%.3f" % best["value"]) if best else "—", int(best.get("step") or 0) if best else 0)
    cur_held = [q for q in held if q.get("lineage") != "abandoned"]
    if cur_held:
        sub += "  ·  held-out %.3f @%s" % (cur_held[-1]["ce"], "{:,}".format(cur_held[-1]["step"]))
    if ref.get("unigram_nats") is not None:
        sub += "  ·  unigram %.3f" % float(ref["unigram_nats"])
    dr.text((40, 64), sub, font=bodyf, fill=(139, 152, 171))
    lx = 40
    for label, col in (("full model", (88, 166, 255)), ("AR (16-exit mean until joint mode)", (126, 190, 255)), ("SAT", (210, 168, 255)),
                       ("NAT", (255, 166, 87)), ("AR p10-p90", (52, 84, 120)), ("abandoned branch", (70, 96, 130)),
                       ("held-out CE", (230, 237, 243))):
        dr.rectangle((lx, 108, lx + 22, 116), fill=col)
        dr.text((lx + 28, 112), label, font=smallf, fill=(200, 208, 218), anchor="lm")
        lx += 40 + dr.textlength(label, font=smallf)
    dr.text((40, H - 34), "Training-batch CE per objective, %s-step bucket means; abandoned rows excluded from the main lines; "
            "held-out = fixed fineweb-edu set; source GB10 trainer log." % (hist.get("bucket_steps") or "?"),
            font=smallf, fill=(110, 118, 129))
    out = BytesIO()
    im.save(out, format="PNG", optimize=True)
    return out.getvalue()


def _render_gb10_training_png(doc: dict) -> bytes:
    """Prefer the full-lineage history chart; fall back to the live-tail loss/EMA chart."""
    try:
        hist = curve_history.snapshot().get("history")
        if isinstance(hist, dict):
            png = _render_history_png(doc, hist)
            if png:
                return png
    except Exception:
        pass
    return _render_gb10_tail_png(doc)


def refresh_training(force: bool = False):
    global _TRAIN_LAST
    try:
        gb10_live.refresh(ssh_vast)
    except Exception:
        pass
    try:
        standard_live.refresh(ssh_vast)
    except Exception:
        pass
    try:
        curve_history.maybe_refresh(ssh_vast_argv)
    except Exception:
        pass
    try:
        live_telemetry.maybe_refresh(ssh_vast_argv)
    except Exception:
        pass
    now = time.time()
    if not force and now - _TRAIN_LAST < TRAINING_MIN_INTERVAL:
        return None
    _TRAIN_LAST = now
    try:
        doc = poll_training()
    except Exception as exc:
        doc = {"ok": False, "reason": f"{type(exc).__name__}: {exc}", "polled_at": int(time.time())}
    png = b""
    if doc.get("ok") and doc.get("source") == "gb10-production-log-live":
        try:
            png = _render_gb10_training_png(doc)
        except Exception:
            png = b""
    elif doc.get("ok") and doc.get("png"):
        try:
            import base64 as _b64
            raw = ssh_vast(TRAINING_PNG_CMD, timeout=60).strip()
            if raw:
                png = _b64.b64decode(raw)
                if not png.startswith(b"\x89PNG"):
                    png = b""
        except Exception:
            png = b""
    with _LOCK:
        if doc.get("ok") or not _TRAIN.get("ok"):
            _TRAIN.clear(); _TRAIN.update(doc)
        else:
            _TRAIN["last_error"] = doc.get("reason"); _TRAIN["last_error_at"] = doc.get("polled_at")
        if png:
            _TRAIN_PNG["bytes"] = png; _TRAIN_PNG["at"] = int(now)
    return doc


def training_png() -> bytes:
    with _LOCK:
        return _TRAIN_PNG["bytes"]


def health_snapshot() -> dict:
    with _LOCK:
        d = dict(_TRAIN)
    if not d.get("ok"):
        fallback = _training_fallback_from_local_status(str(d.get("reason") or "curve collector unavailable"))
        if fallback:
            d = fallback
    h = dict(d.get("health") or {})
    h["ok"] = bool(d.get("ok"))
    h["generated_at"] = d.get("generated_at"); h["polled_at"] = d.get("polled_at")
    h["latest"] = d.get("latest"); h["fix_step"] = d.get("fix_step")
    h["png_url"] = "https://codex.opentransformers.online/api/training.png"
    h["page_url"] = "https://codex.opentransformers.online/agillm-curves"
    return h


def curves_text(rng: str = "fix", points: int = 24) -> str:
    """Plain-text digest for agents without vision: health + recent series."""
    with _LOCK:
        d = dict(_TRAIN)
    if not d.get("ok"):
        return "training curves unavailable: %s" % d.get("reason")
    h = d.get("health") or {}
    L = d.get("latest") or {}
    lines = []
    lines.append("AGILLM-4.4 training — %s" % (h.get("summary") or "no health summary"))
    lines.append("areas: " + ", ".join("%s=%s" % (k, v.get("level")) for k, v in (h.get("areas") or {}).items()))
    for f in (h.get("flags") or [])[:8]:
        lines.append("  [%s] %s: %s" % (f.get("level"), f.get("area"), f.get("message")))
    lines.append("latest heartbeat: step %s, batch loss %s, LR %s, %s tok/s, GPU peak %s GB" % (L.get("step"), L.get("loss"), L.get("lr"), L.get("raw_tok_s"), L.get("gpu_peak_gb")))
    try:
        lines.extend(curve_history.text_digest(curve_history.snapshot().get("history") or {}))
    except Exception as exc:
        lines.append("full-lineage history unavailable: %s" % exc)
    m = (h.get("areas") or {}).get("model") or {}
    lines.append("held-out (heldout_v3, frozen, CPU fp32): w257 %s / w2048 %s @%s (%s); best %s @%s; trainer val %s @%s; governor lr %s cap %s cooldown %s last %s" % (
        m.get("heldout_w257"), m.get("heldout_w2048"), m.get("last_probe_step"), m.get("last_probe_ts"), m.get("best_w257"), m.get("best_step"), m.get("trainer_val_ce"), m.get("trainer_val_step"),
        (m.get("governor") or {}).get("lr"), (m.get("governor") or {}).get("cap"), (m.get("governor") or {}).get("cooldown"), (m.get("governor") or {}).get("last_action")))
    # w2048 by position. heldout_v3 is EOS-JOINED DOCS (see its MANIFEST): a 2048-token window
    # splices ~7 document/window boundaries.
    #
    # CORRECTED 2026-09-08 by a controlled A/B (same checkpoint 2802435, same buckets, identical
    # runtime_sha256 f2ea9a12c526, fp32, 10 threads; only the token source differed) against
    # heldout_longctx_v1 = 16 x 2048 CONTIGUOUS single-document windows, zero EOS inside:
    #
    #                  spliced(v3)   contiguous     the transition
    #   p1-256            6.3372       5.3814       -0.9558   <- v3 is 0.96 nats harder here
    #   p257-1024         5.5346       5.4073       -0.1273
    #   p1025-2047        5.5853       5.4964       -0.0889
    #   p1->p2            -0.8026      +0.0259      <- the "improvement" is NOT context
    #   p2->p3            +0.0507      +0.0891      <- degradation is WORSE without splicing
    #
    # This overturns what this block used to say. The big p1-256 -> p257-1024 drop on v3 is COLD
    # START, not context paying off: an EOS-joined window begins at an arbitrary splice with no
    # usable context, so that bucket is anomalously hard. Run the same model over clean contiguous
    # text and the same transition is +0.0259 - no improvement at all. So there is NO bucket in the
    # spliced set that honestly measures context exploitation; p257-1024 was not it.
    #
    # Nor is splicing the cause of the post-1024 rise: the contiguous arm has no splices and rises
    # MORE (+0.0891 vs +0.0507). At 2.8M steps this model is not exploiting context past a few
    # hundred tokens; past ~1024 it is mildly worse. Caveat carried deliberately: +0.0891 is 0.29x
    # the between-document stdev at n=16 docs, so "consistent with mild degradation", NOT settled.
    #
    # w2048 stays a legitimate frozen TRACKING number. It is not a long-context capability measure
    # and no bucket of it should be reported as one.
    _lp = (d.get("probes") or [])
    _lp = _lp[-1] if _lp else {}
    if _lp.get("p1") is not None and _lp.get("p2") is not None and _lp.get("p3") is not None:
        _d21 = _lp["p2"] - _lp["p1"]; _d32 = _lp["p3"] - _lp["p2"]
        lines.append("held-out w2048 by position @%s: p1-256 %.4f -> p257-1024 %.4f (%+.4f, COLD-START recovery not context) -> p1025-2047 %.4f (%+.4f vs p257-1024%s)" % (
            _lp.get("step"), _lp["p1"], _lp["p2"], _d21, _lp["p3"], _d32, ", i.e. WORSE with depth" if _d32 > 0 else ""))
        lines.append("  heldout_v3 is EOS-joined (~7 boundaries per 2048 window), so p1-256 starts at a splice with no context and is anomalously hard - the big p1->p2 drop is recovery from that, NOT the model using context. Controlled A/B on contiguous single-document text (heldout_longctx_v1, same ckpt/runtime, 2026-09-08) gave p1->p2 of only +0.0259 and p2->p3 of +0.0891, i.e. no improvement and slightly WORSE with depth. w2048 is a valid frozen TRACKING number; no bucket of it measures long-context capability.")
    for k in ("ar", "sat_fixed", "sat_variable", "nat", "dblock", "det50"):
        a = (h.get("areas") or {}).get(k) or {}
        a2 = {kk: vv for kk, vv in a.items() if kk not in ("level", "local_ar_ce_by_block", "last_train", "window", "last", "last_anchor_gate")}
        lines.append("%s [%s]: %s" % (k, a.get("level"), json.dumps(a2, separators=(",", ":"))[:900]))
    fix = int(d.get("fix_step") or 0)
    P = d.get("probes") or []; V = d.get("val") or []
    if rng == "fix":
        P = [p for p in P if p["step"] >= fix]; V = [v for v in V if v["step"] >= fix]
    lines.append("held-out probes (step w257 w2048 ts), last %d:" % min(points, len(P)))
    for p in P[-points:]:
        lines.append("  %s %s %s %s" % (p["step"], p["w257"], p.get("w2048"), (p.get("ts") or "")[11:16]))
    lines.append("trainer val (step ce | sat_shift2 nat16/32/64 = single-window crops of 126/16/32/64 tokens, informational only), last %d:" % min(points, len(V)))
    for v in V[-points:]:
        lines.append("  %s %s | %s %s/%s/%s" % (v["step"], v["ce"], v.get("sat_shift2_ce"), v.get("nat_suffix16_ce"), v.get("nat_suffix32_ce"), v.get("nat_suffix64_ce")))
    mm = d.get("multimode") or []
    if rng == "fix":
        mm = [x for x in mm if x["step"] >= fix]
    if mm:
        lines.append("full held-out SAT/NAT (48 windows x 257 tok, every ~20 min on the probe checkpoint; THE alarm signal - the median of the last 3 probes vs the TRAILING MEDIAN, warn max(2.5 sigma, 0.05) / bad max(5 sigma, 0.15); never vs best/min, which is biased low by the noise), last %d (step sat_shift2 nat16/32/64 pooled ts src):" % min(points, len(mm)))
        for x in mm[-points:]:
            lines.append("  %s %s %s/%s/%s %s %s %s" % (x["step"], x.get("sat"), x.get("nat16"), x.get("nat32"), x.get("nat64"), x.get("nat_pooled"), (x.get("ts") or "")[11:16], x.get("source")))
        if len(mm) > points:
            lines.append("  earlier full held-out points (step sat/nat_pooled): " + "; ".join("%s: %s/%s" % (x["step"], x.get("sat"), x.get("nat_pooled")) for x in mm[:-points][-8:]))
    lines.append("hops: " + ", ".join("%s@%s" % (x.get("label"), x.get("step")) for x in (d.get("markers") or []) if x.get("label") != "resume")[-400:])
    lines.append("chart PNG: https://codex.opentransformers.online/api/training.png · page: https://codex.opentransformers.online/agillm-curves · json: /api/training")
    return "\n".join(lines)



def _attach_recovered_history(doc: dict) -> dict:
    """Attach read-only pre-GB10 quality history recovered from SG archives.

    The old 3090 was deleted, so the every-heartbeat parse cache is unavailable.
    This file contains only exact numeric held-out points found in durable SG
    memory/coordination records; it is deliberately a separate lineage.
    """
    try:
        with open(RECOVERED_TRAINING_HISTORY, "r", encoding="utf-8") as f:
            hist = json.load(f)
        points = hist.get("points") if isinstance(hist, dict) else None
        if isinstance(points, list):
            doc["recovered_probes"] = points
            doc["recovered_history_meta"] = {
                "schema": hist.get("schema"),
                "lineage": hist.get("lineage"),
                "provenance": hist.get("provenance"),
                "protocol": hist.get("protocol"),
                "dense_heartbeat_history_recovered": bool(hist.get("dense_heartbeat_history_recovered")),
                "point_count": len(points),
                "first_step": points[0].get("step") if points else None,
                "last_step": points[-1].get("step") if points else None,
            }
    except Exception as exc:
        doc["recovered_history_meta"] = {"error": f"{type(exc).__name__}: {exc}"}
    # Current GB10 branch metadata stays explicit because its step counter
    # overlaps the older 3090 lineage after the branch resumed from an older ckpt.
    train = doc.get("train") if isinstance(doc.get("train"), list) else []
    doc["branch_meta"] = {
        "current": "GB10-51049010",
        "current_first_step": train[0].get("step") if train else None,
        "current_latest_step": (doc.get("latest") or {}).get("step"),
        "history": "RTX3090-47086549",
        "step_axis_overlaps": True,
        "note": "GB10 resumed from an older checkpoint; branch lines must not be connected.",
    }
    return doc


def training_snapshot(query: str = "") -> dict:
    with _LOCK:
        d = dict(_TRAIN)
    if not d.get("ok"):
        fallback = _training_fallback_from_local_status(str(d.get("reason") or "curve collector unavailable"))
        if fallback:
            d = fallback
    d = _attach_recovered_history(d)
    # Full-lineage history (curve_history.py).  ?history=0 returns only its state and
    # summary so 60 s polls stay small; clients refetch rows when refreshed_at changes.
    try:
        from urllib.parse import parse_qs
        q = parse_qs(query or "")
        want = (q.get("history") or ["1"])[-1] not in ("0", "false", "no")
        hs = curve_history.snapshot()
        d["history_state"] = hs.get("history_state")
        try:
            d["live"] = live_telemetry.snapshot()
        except Exception as _exc:
            d["live"] = {"ok": False, "error": str(_exc)[:200]}
        h = hs.get("history")
        if isinstance(h, dict):
            sm = h.get("summary") or {}
            if isinstance(d.get("branch_meta"), dict):
                rp = [p.get("step") for p in (d.get("recovered_probes") or [])
                      if isinstance(p, dict) and isinstance(p.get("step"), int)]
                fs, ls = sm.get("first_step"), sm.get("last_step")
                overl = bool(rp and fs is not None and ls is not None and min(rp) <= ls and max(rp) >= fs)
                d["branch_meta"] = dict(d["branch_meta"], current_first_step=fs, lineage_first_step=fs,
                                        tail_first_step=d["branch_meta"].get("current_first_step"),
                                        step_axis_overlaps=overl,
                                        note=("current = RPV16 2.0B lineage (first step from history.summary); "
                                              "history = old RTX 3090 Standard 1.1B probes, a different model; "
                                              "never plot them on one axis"))
            if isinstance(d.get("recovered_history_meta"), dict):
                d["recovered_history_meta"] = dict(d["recovered_history_meta"],
                                                   model="Standard 1.1B (RTX 3090 lineage), not RPV16")
            # Honest model-quality flag (service liveness stays in health.overall).
            q = []
            # full = regime-independent full-model CE (anchor rows + full-depth AR; joint mode logs no anchors)
            anc = sm.get("full_mean_last_10k") if sm.get("full_mean_last_10k") is not None else sm.get("anc_mean_last_10k")
            banc = (((h.get("best") or {}).get("full") or (h.get("best") or {}).get("anc")) or {}).get("value")
            uni = (h.get("reference") or {}).get("unigram_nats")
            if anc is not None and uni is not None and anc >= uni:
                q.append("full-model training CE (last 10k steps) %.3f is not below the no-context unigram baseline %.3f"
                         % (anc, uni))
            # drift vs best uses a short recent window, so a fresh best is not reported as a regression
            rec = sm.get("full_mean_last_2k") if (sm.get("full_n_last_2k") or 0) >= 50 else anc
            hcur = sorted([p for p in (h.get("heldout") or []) if p.get("lineage") == "current"
                           and isinstance(p.get("ce"), (int, float)) and isinstance(p.get("step"), int)],
                          key=lambda p: p["step"])
            # held-out CE is the ground truth; training CE is objective-mix dependent (stage-local vs joint)
            ho_improving = len(hcur) >= 2 and hcur[-1]["ce"] <= hcur[-2]["ce"] + 0.005
            qnotes = []
            if rec is not None and banc is not None and rec - banc > 0.10:
                if ho_improving:
                    qnotes.append("training CE %.3f is %.3f above its best %.3f, but held-out CE on the current lineage "
                                  "is improving (%.4f @%s -> %.4f @%s): objective-mix drift, not a regression"
                                  % (rec, rec - banc, banc, hcur[-2]["ce"], hcur[-2]["step"], hcur[-1]["ce"], hcur[-1]["step"]))
                else:
                    q.append("recent full-model training CE %.3f is %.3f above its best %.3f" % (rec, rec - banc, banc))
            if len(hcur) >= 2 and hcur[-1]["ce"] > hcur[-2]["ce"] + 0.02:
                q.append("held-out CE rose %.3f -> %.3f (@%s -> @%s)"
                         % (hcur[-2]["ce"], hcur[-1]["ce"], hcur[-2]["step"], hcur[-1]["step"]))
            if q:
                hh = dict(d.get("health") or {})
                hh["flags"] = [{"level": "warn", "area": "quality", "message": "; ".join(q)}] + list(hh.get("flags") or [])
                hh["quality"] = {"level": "warn", "reasons": q, "notes": qnotes,
                                 "source": "history.summary / best / reference / heldout"}
                d["health"] = hh
            elif qnotes or hcur:
                hh = dict(d.get("health") or {})
                hh["quality"] = {"level": "ok", "reasons": [], "notes": qnotes,
                                 "heldout_latest": hcur[-1] if hcur else None,
                                 "source": "history.summary / best / reference / heldout"}
                d["health"] = hh
            if want:
                d["history"] = h
                if isinstance(hs.get("standard_history"), dict):
                    d["standard_history"] = hs["standard_history"]
            else:
                d["history_summary"] = {k: h.get(k) for k in ("schema", "lineage", "summary", "best", "reference", "heldout", "state")}
                d["history_summary"]["built_at"] = (h.get("meta") or {}).get("built_at")
    except Exception as exc:
        d["history_state"] = {"error": "%s: %s" % (type(exc).__name__, exc)}
    # AGI44_PFLOP_SIDECAR_20260915: expose the independently calculated
    # AGILLM4.4 useful-model FLOP telemetry already mirrored into LIVE_STATUS.
    # This is an execution-graph estimate, never a GPU-utilisation claim.
    try:
        live = json.loads(LIVE_STATUS.read_text())
        compute = live.get("compute_status")
        generated = float((compute or {}).get("generated_unix") or 0.0)
        if isinstance(compute, dict) and generated > 0.0 and time.time() - generated <= 180.0:
            d["compute_status"] = compute
    except Exception:
        pass
    try:
        _g = gb10_live.snapshot()
    except Exception:
        _g = {}
    if isinstance(_g, dict) and _g.get("ok"):
        d["gb10_live"] = _g
        if isinstance(_g.get("compute_status"), dict):
            d["compute_status"] = _g["compute_status"]
    try:
        _s = standard_live.snapshot()
    except Exception:
        _s = {}
    if isinstance(_s, dict) and _s.get("ok"):
        d["standard_live"] = _s
    return d


def checkpoint_snapshot() -> dict:
    with _LOCK:
        snap = dict(_CKPT)
    try:
        _r = gb10_live.checkpoint_view()
    except Exception:
        _r = {}
    if snap.get("polled_at"):
        if _r:
            snap["rpv16_live"] = _r
        return snap
    cached = _checkpoint_from_local_status()
    out = cached or snap
    if _r and isinstance(out, dict):
        out["rpv16_live"] = _r
    return out


def refresh_checkpoint():
    """Called from the existing poller loop -> the UI stays current on its own."""
    try:
        snap = poll_checkpoint()
    except Exception as exc:
        snap = {"ok": False, "reason": f"{type(exc).__name__}: {exc}",
                "polled_at": int(time.time())}
    with _LOCK:
        _CKPT.clear(); _CKPT.update(snap)
    return snap


def _refresh_training_tail_worker(fetch_lines: int) -> None:
    # SG_STATUS_CACHE_HARDENED_20260912
    # Bound the collector too; report failed SSH separately from a genuinely
    # empty log. Requests never wait for this worker.
    import shlex
    n = max(5, min(400, int(fetch_lines)))
    try:
        # AGI44_OBJECTIVE_TELEMETRY_20261002: the old GB10 route (instance 51049010) and RPV16 log path are gone, so this
        # always failed with "Vast SSH route unavailable". Use the Standard V100 log tail that the
        # read-only fleet collector (agillm-rpv16-status.service) fetches every 30 s; no extra SSH.
        _fleet = json.loads(Path("/opt/agillm43_netserve/fleet_training_status.json").read_text())
        _sm = (_fleet.get("models") or {}).get("standard") or {}
        _obs = _sm.get("observed_at")
        if not _sm.get("ok") or not isinstance(_obs, (int, float)) or time.time() - _obs > 180:
            raise RuntimeError("Standard V100 fleet telemetry stale or unavailable (agillm-rpv16-status.service collector)")
        fresh = ["[standard-v100-52631126] " + str(x) for x in (_sm.get("log_tail") or [])][-n:]
        if not fresh:
            raise RuntimeError("Standard V100 fleet telemetry has no log_tail yet")
        err = None
    except Exception as exc:
        fresh = []
        err = f"{type(exc).__name__}: {exc}"[:500]
        # If the source-training SSH route is temporarily unavailable, expose
        # the separately verified local live-status mirror instead of an empty
        # status response. Keep last_error populated so callers can distinguish
        # this degraded-but-useful fallback from a successful raw-log tail.
        try:
            age = time.time() - LIVE_STATUS.stat().st_mtime
            if age <= 900:
                live = json.loads(LIVE_STATUS.read_text())
                summary = {k: live.get(k) for k in (
                    "trainer_live_step", "last_saved_step", "last_saved_file",
                    "seen_tok", "loss", "serving", "serving_step",
                    "serving_file", "status", "verified_at"
                )}
                summary["mirror_age_sec"] = round(age, 3)
                fresh = ["[verified-local-status-fallback] " + json.dumps(summary, sort_keys=True)]
        except Exception:
            fresh = []
    now = int(time.time())
    with _LOCK:
        _TAIL_CACHE["refreshing"] = False
        if err is None:
            _TAIL_CACHE["lines"] = fresh[-400:]
            _TAIL_CACHE["polled_at"] = now
            _TAIL_CACHE["last_error"] = None
            _TAIL_CACHE["last_error_at"] = None
        else:
            if fresh:
                _TAIL_CACHE["lines"] = fresh[-400:]
                _TAIL_CACHE["polled_at"] = now
            _TAIL_CACHE["last_error"] = err
            _TAIL_CACHE["last_error_at"] = now


def _schedule_training_tail_refresh(fetch_lines: int = _TAIL_FETCH_LINES) -> bool:
    now = int(time.time())
    with _LOCK:
        if _TAIL_CACHE.get("refreshing"):
            return False
        last = int(_TAIL_CACHE.get("attempted_at") or 0)
        if last and now - last < _TAIL_MIN_INTERVAL:
            return False
        _TAIL_CACHE["refreshing"] = True
        _TAIL_CACHE["attempted_at"] = now
    try:
        t = threading.Thread(
            target=_refresh_training_tail_worker,
            args=(min(400, max(_TAIL_FETCH_LINES, int(fetch_lines))),),
            name="agillm-tail-refresh", daemon=True)
        t.start()
    except Exception as exc:
        with _LOCK:
            _TAIL_CACHE["refreshing"] = False
            _TAIL_CACHE["last_error"] = f"{type(exc).__name__}: {exc}"[:500]
            _TAIL_CACHE["last_error_at"] = int(time.time())
        return False
    return True


def training_tail(lines: int = 60) -> list:
    n = max(5, min(400, int(lines)))
    _schedule_training_tail_refresh(max(n, _TAIL_FETCH_LINES))
    with _LOCK:
        cached = list(_TAIL_CACHE.get("lines") or [])
    if cached:
        return cached[-n:]

    # A raw-log refresh is asynchronous by design. On a cold cache, seed the
    # current request immediately from the separately verified local status
    # mirror instead of returning an empty log and only fixing the next call.
    try:
        age = time.time() - LIVE_STATUS.stat().st_mtime
        if age <= 900:
            live = json.loads(LIVE_STATUS.read_text())
            summary = {k: live.get(k) for k in (
                "trainer_live_step", "last_saved_step", "last_saved_file",
                "seen_tok", "loss", "serving", "serving_step",
                "serving_file", "status", "verified_at"
            )}
            summary["mirror_age_sec"] = round(age, 3)
            line = "[verified-local-status-fallback] " + json.dumps(summary, sort_keys=True)
            now = int(time.time())
            with _LOCK:
                if not (_TAIL_CACHE.get("lines") or []):
                    _TAIL_CACHE["lines"] = [line]
                    _TAIL_CACHE["polled_at"] = now
                seeded = list(_TAIL_CACHE.get("lines") or [])
            return seeded[-n:]
    except Exception:
        pass
    return []


def training_tail_meta() -> dict:
    with _LOCK:
        d = dict(_TAIL_CACHE)
    polled = int(d.get("polled_at") or 0)
    age = max(0, int(time.time()) - polled) if polled else None
    fallback = bool((d.get("lines") or []) and str((d.get("lines") or [""])[-1]).startswith("[verified-local-status-fallback] "))
    degraded = bool(d.get("last_error"))
    stale_after = 900 if fallback else 60
    return {
        "cached_lines": len(d.get("lines") or []),
        "polled_at": polled or None,
        "age_s": age,
        "available": bool(polled),
        "source": "verified-local-status" if fallback else ("raw-training-log" if polled else None),
        "degraded": degraded,
        "stale": not polled or age > stale_after,
        "stale_after_s": stale_after,
        "refreshing": bool(d.get("refreshing")),
        "last_attempt_at": d.get("attempted_at") or None,
        "last_error": d.get("last_error"),
        "last_error_at": d.get("last_error_at"),
        "refresh_timeout_s": 20,
    }


# AR_SAMPLING_PASSTHROUGH 20260911: forward the truncation and repetition
# knobs instead of sending only temperature/greedy. Omitted keys are left out
# of the payload entirely so the bridge's per-mode defaults still apply.
_SAMPLING_KEYS = ("top_k", "top_p", "min_p", "repetition_penalty",
                  "presence_penalty", "frequency_penalty", "penalty_last_n",
                  "sampler", "dblock_infer_blocks", "mode", "var", "sat_gate_stride2_bias",
                  "sat_var_stride2_rate", "sat_var_fast_floor")


def generate(prompt: str, max_new_tokens: int = 48, temperature: float = 0.7,
             greedy: bool = False, **sampling) -> dict:
    payload = {
        "prompt": str(prompt), "max_new": int(max_new_tokens),
        "temperature": float(temperature), "greedy": bool(greedy),
    }
    # SAT-variable speed dial (confidence13 policy, 2026-09-12): target stride-2 rate 0..0.5.
    # Rides on the existing float passthrough sat_gate_stride2_bias; the trace reports the
    # realised rate. 0 = fast path off (lossless verified-tree behaviour only).
    rate = sampling.get("sat_var_stride2_rate", None)
    if rate is not None:
        rate = max(0.0, min(0.5, float(rate)))
        sampling["sat_var_stride2_rate"] = rate            # first-class field (bridge + child allowlists, 2026-09-12)
        sampling["sat_gate_stride2_bias"] = rate           # fallback carrier for older bridges
        sampling.setdefault("mode", "sat"); sampling.setdefault("var", True)
    for key in _SAMPLING_KEYS:
        value = sampling.get(key)
        if value is not None:
            payload[key] = value
    return bridge_post("/infer", payload)


# --------------------------------------------------------------------- MCP
MCP_PROTOCOL = "2025-06-18"
MCP_TOOLS = [
    {"name": "agillm_checkpoint",
     "description": "Latest AGILLM-4.3 training checkpoint: step, tokens seen, "
                    "loss breakdown, save time, file size, and which checkpoint "
                    "the serving bridge currently has loaded.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "agillm_status",
     "description": "Live training status for the AGILLM-4.3 run: heartbeat step, "
                    "throughput, learning rate, GPU, and the vast instance route.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "agillm_generate",
     "description": "Generate text from the live AGILLM-4.3 model through the "
                    "serving bridge. CPU-served, so expect tens of seconds.",
     "inputSchema": {"type": "object",
                     "properties": {
                         "prompt": {"type": "string"},
                         "max_new_tokens": {"type": "integer", "default": 48},
                         "temperature": {"type": "number", "default": 0.7},
                         "greedy": {"type": "boolean", "default": False},
                         "mode": {"type": "string",
                                  "description": "ar (default), sat or nat."},
                         "top_k": {"type": "integer",
                                   "description": "Keep only the k most likely tokens. 0 disables."},
                         "top_p": {"type": "number",
                                   "description": "Nucleus mass. 1.0 disables."},
                         "min_p": {"type": "number",
                                   "description": "Keep tokens with prob >= min_p * max_prob. Best single knob for this checkpoint."},
                         "repetition_penalty": {"type": "number"},
                         "presence_penalty": {"type": "number"},
                         "frequency_penalty": {"type": "number"},
                         "penalty_last_n": {"type": "integer",
                                            "description": "Repetition lookback window in tokens."},
                         "sampler": {"type": "string",
                                     "description": "ar = full 28-layer serial KV decode (best quality). dblock = independent DBlock pair (faster, much weaker)."},
                         "var": {"type": "boolean", "description": "With mode=sat: variable stride (SAT-V)."},
                         "sat_var_stride2_rate": {"type": "number",
                                                  "description": "SAT-V speed dial: target fraction of blocks committed as 2 tokens from one forward (0..0.5). Implies mode=sat var=true. 0 = lossless verified path only. Held-out cost vs SAT stride-1: 0.1->+0.06 nats/tok 1.05x, 0.2->+0.12 1.11x, 0.3->+0.20 1.18x, 0.4->+0.28 1.25x."}},
                     "required": ["prompt"]}},
    {"name": "agillm_training_log",
     "description": "Tail the live AGILLM-4.3 training log.",
     "inputSchema": {"type": "object",
                     "properties": {"lines": {"type": "integer", "default": 60}}}},
    {"name": "agillm_health",
     "description": "Heuristic health check of the live AGILLM-4.4 run: overall ok/warn/bad plus per-area "
                    "verdicts (trainer/box, whole model held-out, AR, SAT fixed, SAT variable, NAT, DBlock "
                    "local training, detachable-50M student) with the flags that fired. Text + JSON.",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "agillm_curves",
     "description": "Training curves as text for any agent: health verdict, latest heartbeat, frozen held-out "
                    "CE probes (20-min), trainer hourly val (incl. SAT/NAT), full held-out multimode points, "
                    "runtime hops. Set with_image=true to also attach the chart PNG (agents with vision).",
     "inputSchema": {"type": "object",
                     "properties": {"range": {"type": "string", "enum": ["fix", "all"], "default": "fix"},
                                    "points": {"type": "integer", "default": 24},
                                    "with_image": {"type": "boolean", "default": False}}}},
    {"name": "agillm_curves_image",
     "description": "The training-curve chart as a PNG image (1080x1920 portrait: status, since-fix panel, "
                    "all-time panel, LR). For agents with vision; others should call agillm_curves.",
     "inputSchema": {"type": "object", "properties": {}}},
]


def _text(obj):
    return {"content": [{"type": "text",
                         "text": obj if isinstance(obj, str)
                         else json.dumps(obj, indent=2, ensure_ascii=False)}]}


def mcp_call_tool(name: str, args: dict) -> dict:
    args = args or {}
    if name == "agillm_checkpoint":
        return _text(checkpoint_snapshot())
    if name == "agillm_status":
        snap = checkpoint_snapshot()
        return _text({"health": snap.get("health"), "instance": snap.get("instance"),
                      "latest_checkpoint": snap.get("checkpoint_name"),
                      "step": snap.get("step"), "seen_tok": snap.get("seen_tok"),
                      "loss": snap.get("loss"), "polled_at": snap.get("polled_at"),
                      "recent_log": training_tail(15),
                      "recent_log_meta": training_tail_meta()})
    if name == "agillm_generate":
        if not args.get("prompt"):
            return {"content": [{"type": "text", "text": "prompt is required"}],
                    "isError": True}
        return _text(generate(
            args["prompt"], args.get("max_new_tokens", 48),
            args.get("temperature", 0.7), args.get("greedy", False),
            **{k: args[k] for k in _SAMPLING_KEYS if k in args}))
    if name == "agillm_training_log":
        return _text("\n".join(training_tail(args.get("lines", 60))) + "\n\n[training-log metadata] " + json.dumps(training_tail_meta(), separators=(",", ":")))
    if name == "agillm_health":
        h = health_snapshot()
        head = "%s — %s" % ((h.get("overall") or "?").upper(), h.get("summary") or "")
        return {"content": [{"type": "text", "text": head + "\n" + json.dumps(h, indent=1, ensure_ascii=False)}]}
    if name == "agillm_curves":
        txt = curves_text(str(args.get("range") or "fix"), int(args.get("points") or 24))
        content = [{"type": "text", "text": txt}]
        if args.get("with_image"):
            png = training_png()
            if png:
                import base64 as _b64
                content.append({"type": "image", "data": _b64.b64encode(png).decode("ascii"), "mimeType": "image/png"})
            else:
                content.append({"type": "text", "text": "(chart image not available yet)"})
        return {"content": content}
    if name == "agillm_curves_image":
        png = training_png()
        if not png:
            return {"content": [{"type": "text", "text": "chart image not available yet"}], "isError": True}
        import base64 as _b64
        h = health_snapshot()
        return {"content": [{"type": "text", "text": "AGILLM-4.4 training curves — %s (%s)" % ((h.get("overall") or "?").upper(), h.get("summary") or "")},
                            {"type": "image", "data": _b64.b64encode(png).decode("ascii"), "mimeType": "image/png"}]}
    return {"content": [{"type": "text", "text": f"unknown tool {name!r}"}],
            "isError": True}


def mcp_handle(msg: dict):
    """One JSON-RPC message in, one response dict out (None = notification)."""
    mid = msg.get("id")
    method = msg.get("method") or ""
    params = msg.get("params") or {}

    def ok(result):
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def err(code, message):
        return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}

    if method == "initialize":
        return ok({"protocolVersion": MCP_PROTOCOL,
                   "capabilities": {"tools": {"listChanged": False}},
                   "serverInfo": {"name": "agillm-codex", "version": "1.0.0"},
                   "instructions": "AGILLM-4.3 live model and training telemetry. "
                                   "agillm_checkpoint for the newest checkpoint, "
                                   "agillm_generate to sample the live model."})
    if method in ("notifications/initialized", "initialized", "notifications/cancelled"):
        return None
    if method == "ping":
        return ok({})
    if method == "tools/list":
        return ok({"tools": MCP_TOOLS})
    if method == "tools/call":
        try:
            return ok(mcp_call_tool(params.get("name", ""), params.get("arguments") or {}))
        except Exception as exc:
            return err(-32603, f"{type(exc).__name__}: {exc}")
    if method in ("resources/list", "prompts/list"):
        return ok({"resources": []} if method.startswith("resources") else {"prompts": []})
    return err(-32601, f"method not found: {method}")


# Current fleet telemetry (2026-09-30). Old curve files remain historical evidence.
import fleet_live as _fleet_live
_legacy_training_snapshot = training_snapshot

def training_snapshot(query: str = "") -> dict:
    return _fleet_live.training(_legacy_training_snapshot(query))

def checkpoint_snapshot() -> dict:
    return _fleet_live.checkpoint({})

def health_snapshot() -> dict:
    return _fleet_live.health()

def refresh_checkpoint():
    return checkpoint_snapshot()

def refresh_training(force: bool = False):
    # Current fleet collection is the bounded 30-second systemd timer. Never
    # re-poll the retired GB10 host or stamp cached evidence with the current time.
    return _fleet_live.snapshot()


def training_png() -> bytes:
    return _fleet_live.png()
