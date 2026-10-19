#!/usr/bin/env python3
"""AGILLM-4.3 Federation dashboard - Codex UI, GETH-hosted, always-on.

Serves the Codex frontend (index.html + static/) and produces /api/snapshot in
the schema the Codex UI expects, from REAL federation data reachable from geth:
the vast 4090 master over SSH (training log + run_state + nvidia-smi), the local
public-join accepted-contribution spool (validated update norms + leases), and
geth's worker-update queue. Stdlib only. Binds 127.0.0.1:PORT; nginx terminates
TLS for https://monitor.opentransformers.online .
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from urllib.request import urlopen, Request

import agillm_ext


def _live_payload():
    import fleet_live
    return fleet_live.live()

PORT = int(os.environ.get("CODEX_MON_PORT", "8793"))
HERE = Path(__file__).resolve().parent
POLL_SEC = int(os.environ.get("CODEX_MON_POLL", "20"))

VAST = {
    "host": os.environ.get("AGILLM_MASTER_HOST", "x.x.x.x"),
    "port": os.environ.get("AGILLM_MASTER_PORT", "57290"),
    "key": os.environ.get("AGILLM_MASTER_KEY", "/root/.ssh/goddess_vast"),
    "log": "/workspace/agillm41_master_train.log",
    "run_state": "/workspace/agillm4_4090_ckpts_active/run_state.json",
}
COORD = os.environ.get("AGILLM_COORD_URL", "http://127.0.0.1:8787").rstrip("/")
GETH_UPDATES = Path(os.environ.get("AGILLM_GETH_UPDATES", "/root/agillm41_worker/updates"))
ACCEPTED_DIR = Path("/root/agillm41_public_join/spool/accepted")
COORD_LOG = Path("/root/agillm41_public_join/logs/coordinator.log")
AGILLM_SERVICES = [
    "agillm41-public-join.service",
    "agillm41-public-join-validate.timer",
    "agillm41-public-join-promote.timer",
]

DBLOCK = re.compile(
    r"\[dblock\]\s+step=(?P<step>\d+)\s+block=(?P<block>\d+)\s+obj=(?P<obj>\w+)"
    r".*?\bloss=(?P<loss>-?\d+(?:\.\d+)?)"
    r"(?:.*?\bweighted=(?P<weighted>-?\d+(?:\.\d+)?))?"
)

_LOCK = threading.Lock()
_SNAP: dict = {
    "timestamp": 0,
    "join_root": "connecting to federation...",
    "workers": [],
    "hardware": {},
    "scheduler": {},
    "processes": {},
    "coordinator": {},
    "leases": [],
    "batches": [],
    "norm_series": [],
    "loss_series": [],
    "summary": {},
}
_LOGS: dict[str, list[str]] = {"cpu": [], "cuda": [], "cuda_err": []}


def num(value):
    try:
        f = float(value)
        return int(f) if f.is_integer() else f
    except (TypeError, ValueError):
        return None


def ssh_master(remote_cmd: str, timeout: int = 30) -> str:
    try:
        result = subprocess.run(
            [
                "ssh", "-i", VAST["key"], "-p", str(VAST["port"]),
                "-o", "BatchMode=yes", "-o", "ConnectTimeout=12",
                "-o", "StrictHostKeyChecking=no", "-o", "LogLevel=ERROR",
                f"root@{VAST['host']}", remote_cmd,
            ],
            capture_output=True, text=True, timeout=timeout, errors="replace",
        )
        return result.stdout
    except Exception:
        return ""


def poll_master():
    combined = ssh_master(
        "tail -n 500 %s 2>/dev/null; echo '<<S>>'; cat %s 2>/dev/null; echo '<<S>>'; "
        "nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu "
        "--format=csv,noheader,nounits 2>/dev/null"
        % (VAST["log"], VAST["run_state"])
    )
    parts = combined.split("<<S>>")
    log_text = parts[0] if parts else ""
    run_state = {}
    if len(parts) > 1:
        try:
            run_state = json.loads(parts[1].strip())
        except Exception:
            run_state = {}
    gpu = {}
    if len(parts) > 2 and parts[2].strip():
        row = [x.strip() for x in parts[2].strip().splitlines()[0].split(",")]
        if len(row) >= 5:
            gpu = {"name": row[0], "driver": row[1], "mem_total": num(row[2]),
                   "mem_used": num(row[3]), "util": num(row[4])}
    return log_text, run_state, gpu


def parse_dblocks(log_text: str):
    batches = []
    for line in log_text.splitlines():
        m = DBLOCK.search(line)
        if not m:
            continue
        batches.append({
            "worker": "vast-4090-master", "device": "cuda",
            "step": int(m.group("step")), "block": int(m.group("block")),
            "objective": m.group("obj"), "loss": float(m.group("loss")),
            "weighted": float(m.group("weighted")) if m.group("weighted") else None,
            "peak_alloc": None, "peak_reserved": None,
        })
    return batches


def coord_get(path: str, timeout: float = 3.0):
    try:
        with urlopen(Request(COORD + path), timeout=timeout) as response:
            return json.loads(response.read(2_000_000).decode("utf-8", errors="replace"))
    except Exception:
        return None


def read_accepted_spool():
    """Validated public-join contributions: update norms + leases + worker nodes."""
    norm_series, leases, nodes = [], [], {}
    try:
        files = sorted(ACCEPTED_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:90]
    except Exception:
        files = []
    now = time.time()
    rows = []
    for jf in files:
        try:
            rec = json.loads(jf.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        validation = rec.get("validation") if isinstance(rec.get("validation"), dict) else {}
        norm = validation.get("update_norm")
        node_id = rec.get("node_id") or "unknown"
        meta = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
        device = (rec.get("capabilities") or {}).get("device") or meta.get("tier") or "cpu"
        pkg = rec.get("package") if isinstance(rec.get("package"), dict) else {}
        expires = rec.get("expires_at")
        ts = rec.get("validated_at") or rec.get("submitted_at") or rec.get("leased_at") or 0
        rows.append((ts, node_id, device, norm, meta, pkg, expires, rec))
    rows.sort(key=lambda r: r[0])  # oldest -> newest for the chart
    for ts, node_id, device, norm, meta, pkg, expires, rec in rows:
        if isinstance(norm, (int, float)):
            norm_series.append({"norm": float(norm), "worker": node_id, "device": device, "source": "accepted"})
        leases.append({
            "lane": device,
            "source_cycle": (meta.get("source_cycle") or "")[:24] or meta.get("network"),
            "network": meta.get("network"),
            "result_exists": bool(rec.get("result_file") or rec.get("submitted_at")),
            "result_mb": round((pkg.get("bytes") or 0) / 1_000_000, 1),
            "expires_in_sec": int(expires - now) if isinstance(expires, (int, float)) else None,
        })
        entry = nodes.setdefault(node_id, {
            "name": node_id, "device": device, "status": rec.get("state") or "validated",
            "steps": 0, "latest_loss": None, "submissions": 0, "log_age_sec": None, "error": None,
        })
        entry["submissions"] += 1
    leases.reverse()  # newest first for the table
    return norm_series, leases, nodes


def host_cpu():
    name, cores = "geth (Hetzner)", None
    try:
        cores = num(subprocess.run(["nproc"], capture_output=True, text=True, timeout=2).stdout.strip())
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    return {"name": name, "cores": cores, "logical_processors": cores, "max_clock_mhz": None}


def service_status():
    active, detail_parts = 0, []
    for svc in AGILLM_SERVICES:
        try:
            state = subprocess.run(["systemctl", "is-active", svc], capture_output=True, text=True, timeout=2).stdout.strip()
        except Exception:
            state = "unknown"
        if state == "active":
            active += 1
        detail_parts.append(f"{svc.replace('agillm41-', '').replace('.service', '').replace('.timer', '')}={state}")
    status = "active" if active == len(AGILLM_SERVICES) else ("waiting" if active else "unavailable")
    return {"task": "federation services", "status": status, "detail": " | ".join(detail_parts), "last_run": None}


def python_count():
    try:
        return num(subprocess.run(["pgrep", "-c", "python3"], capture_output=True, text=True, timeout=2).stdout.strip()) or 0
    except Exception:
        return 0


def tail_local(path: Path, lines: int = 200):
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 400_000))
            data = handle.read().decode("utf-8", errors="replace")
        return data.splitlines()[-lines:]
    except Exception:
        return []


def build_snapshot():
    log_text, run_state, gpu = poll_master()
    batches = parse_dblocks(log_text)
    master_online = bool(log_text.strip())

    log_lines = log_text.splitlines()
    _LOGS["cpu"] = log_lines[-200:]
    _LOGS["cuda"] = [ln for ln in log_lines if re.search(r"side|merge|async_side|lease|heartbeat", ln, re.I)][-200:]
    _LOGS["cuda_err"] = tail_local(COORD_LOG, 200)

    master_step = batches[-1]["step"] if batches else 0
    for key in ("global_step", "step", "master_step", "total_steps"):
        if isinstance(run_state.get(key), (int, float)):
            master_step = int(run_state[key])
            break
    master_loss = next((b["loss"] for b in reversed(batches) if b.get("loss") is not None), None)

    workers = [{
        "name": "vast-4090-master", "device": "cuda",
        "status": "training" if master_online else "missing",
        "steps": master_step, "latest_loss": master_loss, "submissions": 0,
        "log_age_sec": 0 if master_online else None,
        "error": None if master_online else "master log unreachable over SSH",
    }]

    health = coord_get("/health")
    norm_series, leases, nodes = read_accepted_spool()
    workers += sorted(nodes.values(), key=lambda w: -w["submissions"])[:14]

    try:
        updates = sorted(GETH_UPDATES.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        updates = []
    queue_recent = len([u for u in updates if (time.time() - u.stat().st_mtime) < 3600]) if updates else 0

    cuda_runtime = {
        "available": bool(gpu), "name": gpu.get("name"), "driver": gpu.get("driver"),
        "memory_total_mb": gpu.get("mem_total"), "memory_used_mb": gpu.get("mem_used"),
        "utilization_pct": gpu.get("util"),
        "detail": "vast 4090 master" if gpu else "master nvidia-smi unavailable",
    }
    hardware = {
        "cpu": host_cpu(), "cuda_runtime": cuda_runtime,
        "gpus": ([{"kind": "cuda", "name": gpu.get("name") or "RTX 4090",
                   "adapter_ram_mb": gpu.get("mem_total"), "driver": gpu.get("driver")}] if gpu else []),
    }

    loss_series = [{"worker": b["worker"], "device": b["device"], "step": b["step"],
                    "loss": b["loss"], "weighted": b["weighted"], "objective": b["objective"]}
                   for b in batches[-80:]]
    norm_vals = [n["norm"] for n in norm_series if isinstance(n.get("norm"), (int, float))]
    loss_vals = [b["loss"] for b in batches if isinstance(b.get("loss"), (int, float))]
    active = 1 if master_online else 0
    active += sum(1 for w in workers[1:] if w["status"] in ("training", "submitted", "waiting", "leased", "active", "validated"))
    errors = [] if master_online else ["master log unreachable over SSH"]

    return {
        "timestamp": time.time(),
        "join_root": "geth - vast-4090 master (step %s) + %d federation nodes - queue %d/h"
                     % (f"{master_step:,}", len(nodes), queue_recent),
        "workers": workers,
        "hardware": hardware,
        "scheduler": service_status(),
        "processes": {"python_count": python_count()},
        "coordinator": {"ok": bool(health), "url": COORD, "health": health},
        "leases": leases[:40],
        "batches": batches[-100:],
        "norm_series": norm_series[-60:],
        "loss_series": loss_series,
        "summary": {
            "active_workers": active,
            "total_steps": master_step,
            "total_submissions": sum(w.get("submissions") or 0 for w in workers),
            "latest_norm": norm_series[-1]["norm"] if norm_series else None,
            "avg_norm": round(sum(norm_vals) / len(norm_vals), 2) if norm_vals else None,
            "peak_norm": max(norm_vals) if norm_vals else None,
            "latest_loss": loss_vals[-1] if loss_vals else None,
            "avg_loss": round(sum(loss_vals[-20:]) / len(loss_vals[-20:]), 4) if loss_vals else None,
            "no_lease_events": 0,
            "errors": errors,
        },
    }


def poller():
    while True:
        try:
            snap = build_snapshot()
            with _LOCK:
                _SNAP.clear()
                _SNAP.update(snap)
            agillm_ext.refresh_checkpoint()
            agillm_ext.refresh_training()
        except Exception as exc:
            with _LOCK:
                _SNAP["summary"] = {"errors": [f"poller error: {exc}"]}
        time.sleep(POLL_SEC)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def log_message(self, *args):
        pass

    def _json(self, value):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        # /api/training now carries the full-lineage curve history (~0.5 MB). The nginx
        # gzip filter does not compress these proxied responses, so compress here.
        if len(body) > 4096 and "gzip" in (self.headers.get("Accept-Encoding") or "").lower():
            import gzip
            body = gzip.compress(body, 5)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Vary", "Accept-Encoding")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/agillm-ui", "/agillm-ui/"):
            try:
                body = (HERE / "agillm-ui.html").read_bytes()
            except Exception:
                self.send_error(HTTPStatus.NOT_FOUND); return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if parsed.path == "/api/live":
            self._json(_live_payload()); return
        if parsed.path == "/agillm-live.js":
            try:
                body = (HERE / "static" / "agillm_live_panel.js").read_bytes()
            except Exception:
                self.send_error(HTTPStatus.NOT_FOUND); return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/javascript; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if parsed.path == "/api/checkpoint":
            self._json(agillm_ext.checkpoint_snapshot()); return
        if parsed.path == "/api/training":
            self._json(agillm_ext.training_snapshot(parsed.query)); return
        if parsed.path == "/api/health":
            self._json(agillm_ext.health_snapshot()); return
        if parsed.path == "/api/training.png":
            png = agillm_ext.training_png()
            if not png:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "chart not rendered yet"); return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers(); self.wfile.write(png); return
        if parsed.path in ("/agillm-curves", "/agillm-curves/"):
            try:
                body = Path("/opt/agillm43_netserve/agillm_curves_page.html").read_bytes()
            except Exception:
                self.send_error(HTTPStatus.NOT_FOUND); return
            # live telemetry panel, injected at serve time (page file untouched)
            tag = b'<script src="/agillm-live.js" defer></script>'
            if tag not in body:
                i = body.rfind(b"</body>")
                body = body[:i] + tag + body[i:] if i >= 0 else body + tag
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if parsed.path == "/mcp":
            self._json({"server": "agillm-codex", "transport": "http-jsonrpc",
                        "protocolVersion": agillm_ext.MCP_PROTOCOL,
                        "endpoint": "https://codex.opentransformers.online/mcp",
                        "usage": "POST JSON-RPC 2.0 here (initialize, tools/list, tools/call)",
                        "tools": [t["name"] for t in agillm_ext.MCP_TOOLS]}); return
        if parsed.path == "/api/snapshot":
            with _LOCK:
                snap = dict(_SNAP)
            self._json(snap)
            return
        if parsed.path == "/api/log":
            query = parse_qs(parsed.query)
            lane = (query.get("lane") or ["cpu"])[0]
            count = min(300, max(20, int((query.get("lines") or ["120"])[0])))
            label = {"cpu": "vast master training log", "cuda": "master merge / lease feed",
                     "cuda_err": str(COORD_LOG)}.get(lane, lane)
            with _LOCK:
                lines = list(_LOGS.get(lane, []))[-count:]
            self._json({"lane": lane, "path": label, "exists": bool(lines), "lines": lines})
            return
        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, f"bad json: {exc}"); return

        if parsed.path == "/mcp":
            if isinstance(payload, list):
                out = [r for r in (agillm_ext.mcp_handle(m) for m in payload) if r]
                if not out:
                    self.send_response(HTTPStatus.ACCEPTED); self.end_headers(); return
                self._json(out); return
            resp = agillm_ext.mcp_handle(payload)
            if resp is None:
                self.send_response(HTTPStatus.ACCEPTED); self.end_headers(); return
            self._json(resp); return

        if parsed.path == "/api/generate":
            # 2026-09-12: forward the sampling knobs the UI now sends (top_p 0.95,
            # repetition_penalty 1.05); greedy stays opt-in. Measured on ckpt 2875945:
            # greedy -> rep_frac_8 0.29-1.0 (loops), temp 0.7/top_p 0.95 -> 0.0.
            _extra = {k: payload[k] for k in getattr(agillm_ext, "_SAMPLING_KEYS", ()) if k in payload}
            self._json(agillm_ext.generate(
                payload.get("prompt", ""),
                payload.get("max_new_tokens", 48),
                payload.get("temperature", 0.7),
                payload.get("greedy", False), **_extra)); return

        self.send_error(HTTPStatus.NOT_FOUND)


def main():
    threading.Thread(target=poller, daemon=True).start()
    print(f"Codex federation dashboard on http://127.0.0.1:{PORT} (serving {HERE})")
    # Also listen on the private WireGuard/LAN address so the SG1/SG2 MCP nodes can
    # register this server as a backend without going through the public edge.
    for extra in [a for a in os.environ.get("CODEX_MON_EXTRA_BIND", "x.x.x.x").split(",") if a.strip()]:
        try:
            srv = ThreadingHTTPServer((extra.strip(), PORT), Handler)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            print(f"  also on http://{extra.strip()}:{PORT}")
        except Exception as exc:
            print(f"  extra bind {extra} failed: {exc}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
