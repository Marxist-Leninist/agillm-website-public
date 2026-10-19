"""Patch agillm_ext.py + server.py: training PNG + health + MCP tools + private bind (2026-09-06)."""
import re, ast
p = '/root/agillm41_codex_monitor/agillm_ext.py'; s = open(p, encoding='utf-8').read()
def rep(old, new, n=1):
    global s
    c = s.count(old); assert c == n, (old[:60], c); s = s.replace(old, new)

rep('''TRAINING_CMD = "python3 /workspace/agillm43_serving_probe/training_curves.py"''',
    '''TRAINING_CMD = "python3 /workspace/agillm43_serving_probe/training_curves.py --png"
TRAINING_PNG_CMD = "base64 -w0 /workspace/agillm43_serving_probe/training_curves.png 2>/dev/null"
_TRAIN_PNG: dict = {"bytes": b"", "at": 0}''')

rep('''def refresh_training(force: bool = False):
    global _TRAIN_LAST
    now = time.time()
    if not force and now - _TRAIN_LAST < TRAINING_MIN_INTERVAL:
        return None
    _TRAIN_LAST = now
    try:
        doc = poll_training()
    except Exception as exc:
        doc = {"ok": False, "reason": f"{type(exc).__name__}: {exc}", "polled_at": int(time.time())}
    with _LOCK:
        if doc.get("ok") or not _TRAIN.get("ok"):
            _TRAIN.clear(); _TRAIN.update(doc)
        else:
            _TRAIN["last_error"] = doc.get("reason"); _TRAIN["last_error_at"] = doc.get("polled_at")
    return doc
''', '''def refresh_training(force: bool = False):
    global _TRAIN_LAST
    now = time.time()
    if not force and now - _TRAIN_LAST < TRAINING_MIN_INTERVAL:
        return None
    _TRAIN_LAST = now
    try:
        doc = poll_training()
    except Exception as exc:
        doc = {"ok": False, "reason": f"{type(exc).__name__}: {exc}", "polled_at": int(time.time())}
    png = b""
    if doc.get("ok") and doc.get("png"):
        try:
            import base64 as _b64
            raw = ssh_vast(TRAINING_PNG_CMD, timeout=60).strip()
            if raw:
                png = _b64.b64decode(raw)
                if not png.startswith(b"\\x89PNG"):
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
    m = (h.get("areas") or {}).get("model") or {}
    lines.append("held-out (heldout_v3, frozen, CPU fp32): w257 %s / w2048 %s @%s (%s); best %s @%s; trainer val %s @%s; governor lr %s cap %s cooldown %s last %s" % (
        m.get("heldout_w257"), m.get("heldout_w2048"), m.get("last_probe_step"), m.get("last_probe_ts"), m.get("best_w257"), m.get("best_step"), m.get("trainer_val_ce"), m.get("trainer_val_step"),
        (m.get("governor") or {}).get("lr"), (m.get("governor") or {}).get("cap"), (m.get("governor") or {}).get("cooldown"), (m.get("governor") or {}).get("last_action")))
    for k in ("ar", "sat_fixed", "sat_variable", "nat", "dblock", "det50"):
        a = (h.get("areas") or {}).get(k) or {}
        a2 = {kk: vv for kk, vv in a.items() if kk not in ("level", "local_ar_ce_by_block", "last_train", "window", "last", "last_anchor_gate")}
        lines.append("%s [%s]: %s" % (k, a.get("level"), json.dumps(a2, separators=(",", ":"))[:600]))
    fix = int(d.get("fix_step") or 0)
    P = d.get("probes") or []; V = d.get("val") or []
    if rng == "fix":
        P = [p for p in P if p["step"] >= fix]; V = [v for v in V if v["step"] >= fix]
    lines.append("held-out probes (step w257 w2048 ts), last %d:" % min(points, len(P)))
    for p in P[-points:]:
        lines.append("  %s %s %s %s" % (p["step"], p["w257"], p.get("w2048"), (p.get("ts") or "")[11:16]))
    lines.append("trainer val (step ce sat_shift2 nat32), last %d:" % min(points, len(V)))
    for v in V[-points:]:
        lines.append("  %s %s %s %s" % (v["step"], v["ce"], v.get("sat_shift2_ce"), v.get("nat_suffix32_ce")))
    mm = d.get("multimode") or []
    if mm:
        lines.append("full held-out (48 windows) AR/SAT/NAT16/32/64 CE by step: " + "; ".join("%s: %s/%s/%s/%s/%s" % (x["step"], x.get("ar"), x.get("sat"), x.get("nat16"), x.get("nat32"), x.get("nat64")) for x in mm))
    lines.append("hops: " + ", ".join("%s@%s" % (x.get("label"), x.get("step")) for x in (d.get("markers") or []) if x.get("label") != "resume")[-400:])
    lines.append("chart PNG: https://codex.opentransformers.online/api/training.png · page: https://codex.opentransformers.online/agillm-curves · json: /api/training")
    return "\\n".join(lines)

''')

# MCP tools
rep('''    {"name": "agillm_training_log",
     "description": "Tail the live AGILLM-4.3 training log.",
     "inputSchema": {"type": "object",
                     "properties": {"lines": {"type": "integer", "default": 60}}}},
]''', '''    {"name": "agillm_training_log",
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
]''')

rep('''    if name == "agillm_training_log":
        return _text("\\n".join(training_tail(args.get("lines", 60))))
''', '''    if name == "agillm_training_log":
        return _text("\\n".join(training_tail(args.get("lines", 60))))
    if name == "agillm_health":
        h = health_snapshot()
        head = "%s — %s" % ((h.get("overall") or "?").upper(), h.get("summary") or "")
        return {"content": [{"type": "text", "text": head + "\\n" + json.dumps(h, indent=1, ensure_ascii=False)}]}
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
''')
open(p, 'w', encoding='utf-8').write(s); ast.parse(s); print('agillm_ext patched')

# ---------------- server.py
q = '/root/agillm41_codex_monitor/server.py'; t = open(q, encoding='utf-8').read()
def rep2(old, new, n=1):
    global t
    c = t.count(old); assert c == n, (old[:60], c); t = t.replace(old, new)
rep2('''        if parsed.path == "/api/training":
            self._json(agillm_ext.training_snapshot()); return
''', '''        if parsed.path == "/api/training":
            self._json(agillm_ext.training_snapshot()); return
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
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
''')
rep2('''def main():
    threading.Thread(target=poller, daemon=True).start()
    print(f"Codex federation dashboard on http://127.0.0.1:{PORT} (serving {HERE})")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
''', '''def main():
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
''')
open(q, 'w', encoding='utf-8').write(t); ast.parse(t); print('server patched')
