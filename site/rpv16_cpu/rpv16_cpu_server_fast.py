#!/usr/bin/env python3
"""RPV16 GETH CPU server entry point with the decode fast path (fable-cowork 2026-10-03).

Loads the unchanged rpv16_cpu_server.py (which installs rpv16_multimode_v1), then installs a decode fast path on top
and serves the same HTTP API.

RPV16_FASTPATH_VERSION=2 (default): rpv16_fastpath_v2 - exact int8 kernels, native float path, one C++ call per
    decode step. The engine proves the whole model bitwise against the v1 path while it loads; a failed proof leaves
    the v1 path in place (see /health -> fastpath and the receipt it names).
RPV16_FASTPATH_VERSION=1: rpv16_fastpath_v1 (KV-cached SAT/NAT on the stock quantised Linears).
If v2 cannot be imported or installed, v1 is installed on a freshly loaded server module.
Rollback: RPV16_FASTPATH_VERSION=1 in the unit, or start rpv16_cpu_server.py again.
"""
import importlib
import importlib.util
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def load(version):
    spec = importlib.util.spec_from_file_location("rpv16_cpu_server", os.path.join(HERE, "rpv16_cpu_server.py"))
    srv = importlib.util.module_from_spec(spec)
    sys.modules["rpv16_cpu_server"] = srv
    spec.loader.exec_module(srv)
    fp = importlib.import_module("rpv16_fastpath_v%d" % version)
    fp.install(srv.__dict__)
    return srv, fp


WANT = 1 if os.environ.get("RPV16_FASTPATH_VERSION", "2").strip() == "1" else 2
INSTALL_ERROR = None
try:
    srv, fastpath = load(WANT)
except Exception as exc:
    if WANT == 1:
        raise
    INSTALL_ERROR = "%s: %s" % (type(exc).__name__, exc)
    traceback.print_exc()
    print("[rpv16-fast] fast path v2 not installed (%s); serving with v1" % INSTALL_ERROR, flush=True)
    srv, fastpath = load(1)

ns = srv.__dict__
_multimode_health = ns["health"]


def health():
    d = _multimode_health()
    rev = ns.get("FASTPATH_REVISION")
    d["fastpath_revision"] = rev
    if rev not in (None, "disabled"):
        d["runtime_revision"] = str(d.get("runtime_revision")) + "+" + rev
        for cap in (d.get("capabilities") or {}).values():
            sem = cap.get("semantics") or {}
            if sem.get("canonical_mode") in ("sat_fixed", "sat_variable", "nat", "ar"):
                sem["kv_cache"] = True
    state = ns.get("FASTPATH_STATE")
    if isinstance(state, dict):
        summary = state.get("summary") or {}
        d["fastpath"] = {"revision": state.get("revision"), "int8_kernels": bool(state.get("int8_kernels")),
                         "native_float": state.get("native"), "topk": bool(state.get("topk")), "sampler": bool(state.get("sampler")),
                         "proven_bitwise_at_load": bool(summary.get("ok")), "error": state.get("error"), "receipt": state.get("receipt"),
                         "kernel": (summary.get("kernel") or {}).get("tag"), "session_rows": summary.get("session_rows"),
                         "extras": summary.get("extras"), "probe_speedup": summary.get("speedup"),
                         "upgrade_seconds": summary.get("upgrade_seconds"), "rss_mb": summary.get("rss_mb_final")}
    if INSTALL_ERROR:
        d["fastpath_install_error"] = INSTALL_ERROR
    return d


app = ns["app"]
app.router.routes[:] = [r for r in app.router.routes if getattr(r, "path", None) != "/health"]
app.add_api_route("/health", health, methods=["GET"])
ns["health"] = health

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("RPV16_CPU_PORT", "18620")), log_level="info")
