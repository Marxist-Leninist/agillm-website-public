#!/usr/bin/env python3
import datetime as dt
import json
import pathlib
import threading
import time
import urllib.error
import urllib.request
import uuid

ORIGIN = "https://codex.opentransformers.online"
HEALTH_URL = ORIGIN + "/agillm43-infer/health"
STREAM_URL = ORIGIN + "/agillm-stream"
CANCEL_URL = ORIGIN + "/agillm-cancel"
ROOT = pathlib.Path("/opt/agillm43_netserve/receipts")
STAMP = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RUN = ROOT / f"deployment_acceptance_{STAMP}"
RUN.mkdir(parents=True, exist_ok=False)

PROFILE = {
    "mode": "nat",
    "nat_passes": 1,
    "nat_chunk_size": 0,
    "nat_ar_verify": False,
    "nat_anti_collapse": True,
    "nat_stream_top_every": 64,
    "greedy": True,
    "temperature": 0,
    "top_p": 1.0,
    "top_k": 32,
    "ignore_eos": True,
    "repetition_penalty": 1.05,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "penalty_last_n": 128,
    "infer_dtype": "fp32",
}


def request_json(url, method="GET", payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as res:
        body = res.read().decode("utf-8", "replace")
        return res.status, json.loads(body)


def health():
    return request_json(HEALTH_URL, timeout=10)[1]


def make_payload(name, tokens, **overrides):
    out = dict(PROFILE)
    out.update({
        "request_id": f"accept-{name}-{uuid.uuid4().hex}",
        "prompt": "The year is 2050 and artificial intelligence systems",
        "max_new": int(tokens),
        "min_new": int(tokens),
    })
    out.update(overrides)
    return out


def parse_result(events, elapsed, payload, raw_name, error=None):
    done = next((e for e in reversed(events) if e.get("event") == "done"), {})
    err_event = next((e for e in reversed(events) if e.get("event") == "error"), {})
    stats = done.get("stats") or {}
    trace = stats.get("nat_trace") or next(
        (e.get("data") for e in events if e.get("event") == "nat_trace"), {}
    ) or {}
    start = next((e for e in events if e.get("event") == "start"), {})
    return {
        "request_id": payload["request_id"],
        "tokens_requested": payload["max_new"],
        "passes": payload.get("nat_passes"),
        "ok": done.get("ok") is True,
        "done_tokens": stats.get("tokens"),
        "stream_token_events": sum(1 for e in events if e.get("event") == "stream_tok"),
        "server_seconds": stats.get("seconds"),
        "generation_tok_s": stats.get("tok_s"),
        "button_to_done_seconds": round(elapsed, 4),
        "button_to_done_tok_s": round((stats.get("tokens") or 0) / max(elapsed, 1e-9), 2),
        "slot": done.get("slot", start.get("slot")),
        "threads": done.get("threads", start.get("threads")),
        "queue_wait_s": done.get("queue_wait_s", start.get("queue_wait_s")),
        "host": done.get("host", start.get("host")),
        "request_id_echoed": done.get("request_id", start.get("request_id")),
        "nat_trace": trace,
        "error_event": err_event or None,
        "transport_error": error,
        "text_preview": (done.get("text") or "")[:240].replace("\n", " "),
        "raw_file": raw_name,
    }


def stream(payload, raw_name, timeout=600):
    req = urllib.request.Request(
        STREAM_URL,
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    events = []
    raw = []
    error = None
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as res:
            for bline in res:
                line = bline.decode("utf-8", "replace").rstrip("\n")
                raw.append(line)
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - start
    (RUN / raw_name).write_text("\n".join(raw) + ("\n" if raw else ""))
    return parse_result(events, elapsed, payload, raw_name, error)


def threaded_stream(payload, raw_name, sink, key):
    sink[key] = stream(payload, raw_name)


def wait_health(predicate, timeout=30, interval=0.05):
    deadline = time.monotonic() + timeout
    samples = []
    while time.monotonic() < deadline:
        try:
            h = health()
            samples.append(h)
            if predicate(h):
                return h, samples
        except Exception:
            pass
        time.sleep(interval)
    raise TimeoutError(f"health predicate not met in {timeout}s; last={samples[-1] if samples else None}")


manifest = {
    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    "origin": ORIGIN,
    "health_url": HEALTH_URL,
    "stream_url": STREAM_URL,
    "cancel_url": CANCEL_URL,
    "profile": PROFILE,
    "initial_health": health(),
    "tests": {},
}

# 1. Two long requests must occupy both remote slots and both finish exactly.
con_results = {}
p1 = make_payload("con-a", 1024)
p2 = make_payload("con-b", 1024)
t1 = threading.Thread(target=threaded_stream, args=(p1, "concurrent_a.ndjson", con_results, "a"), daemon=True)
t2 = threading.Thread(target=threaded_stream, args=(p2, "concurrent_b.ndjson", con_results, "b"), daemon=True)
t1.start(); t2.start()
active2, active_samples = wait_health(
    lambda h: h.get("host") == "vast-47086549" and int(h.get("busy") or 0) >= 2
              and int(h.get("active_request_count") or 0) >= 2,
    timeout=30,
)
t1.join(600); t2.join(600)
if t1.is_alive() or t2.is_alive():
    raise RuntimeError("concurrency streams did not finish")
manifest["tests"]["concurrency"] = {
    "active_two_sample": active2,
    "sample_count": len(active_samples),
    "a": con_results.get("a"),
    "b": con_results.get("b"),
}
ca, cb = con_results["a"], con_results["b"]
assert ca["done_tokens"] == 1024 and cb["done_tokens"] == 1024, (ca, cb)
assert ca["slot"] != cb["slot"], (ca["slot"], cb["slot"])
assert float(ca["queue_wait_s"] or 0) < 1.0 and float(cb["queue_wait_s"] or 0) < 1.0

# 2. Cancel A by ID while B remains alive and completes exactly.
cancel_results = {}
pa = make_payload("cancel-a", 1024)
pb = make_payload("cancel-b", 1024)
ta = threading.Thread(target=threaded_stream, args=(pa, "cancel_a.ndjson", cancel_results, "a"), daemon=True)
tb = threading.Thread(target=threaded_stream, args=(pb, "cancel_b.ndjson", cancel_results, "b"), daemon=True)
ta.start(); tb.start()
cancel_active, _ = wait_health(lambda h: int(h.get("active_request_count") or 0) >= 2, timeout=30)
status, cancel_reply = request_json(CANCEL_URL, method="POST", payload={"request_id": pa["request_id"]}, timeout=15)
ta.join(180); tb.join(600)
if ta.is_alive() or tb.is_alive():
    raise RuntimeError("cancel isolation streams did not terminate")
repaired, repair_samples = wait_health(
    lambda h: h.get("ready") is True and h.get("warming") is not True
              and int(h.get("live_slots") or 0) >= int(h.get("concurrency") or 1),
    timeout=180, interval=0.25,
)
manifest["tests"]["targeted_cancel"] = {
    "active_two_sample": cancel_active,
    "cancel_http": status,
    "cancel_reply": cancel_reply,
    "a": cancel_results.get("a"),
    "b": cancel_results.get("b"),
    "repaired_health": repaired,
    "repair_sample_count": len(repair_samples),
}
assert status in (200, 202) and cancel_reply.get("ok") is True
assert cancel_results["b"]["done_tokens"] == 1024, cancel_results["b"]
assert cancel_results["a"]["done_tokens"] != 1024, cancel_results["a"]

# 3. Cancel-before-register must be consumed before a slot is acquired.
pre_id = f"accept-pre-cancel-{uuid.uuid4().hex}"
pre_status, pre_reply = request_json(CANCEL_URL, method="POST", payload={"request_id": pre_id}, timeout=15)
pre_payload = make_payload("pre-cancel", 8)
pre_payload["request_id"] = pre_id
pre_result = stream(pre_payload, "cancel_before_register.ndjson", timeout=30)
manifest["tests"]["cancel_before_register"] = {
    "cancel_http": pre_status,
    "cancel_reply": pre_reply,
    "stream": pre_result,
}
assert pre_status == 202 and pre_reply.get("ok") is True
assert pre_result["done_tokens"] is None and (pre_result["error_event"] or {}).get("error") == "request canceled"

# 4. Pass selector must map to exact model-forward count.
p4 = make_payload("pass4", 48, nat_passes=4)
pass4 = stream(p4, "pass4.ndjson", timeout=180)
manifest["tests"]["pass4"] = pass4
assert pass4["done_tokens"] == 48, pass4
assert int((pass4["nat_trace"] or {}).get("passes") or 0) == 4, pass4
assert int((pass4["nat_trace"] or {}).get("core_forwards") or 0) == 4, pass4

# 5. Full UI ceiling: exact 2048 generated tokens, eight progressive 256-token forwards.
p2048 = make_payload("exact2048", 2048, nat_passes=1)
exact2048 = stream(p2048, "exact2048.ndjson", timeout=900)
manifest["tests"]["exact_2048"] = exact2048
assert exact2048["done_tokens"] == 2048, exact2048
assert exact2048["stream_token_events"] == 2048, exact2048
trace2048 = exact2048["nat_trace"] or {}
assert int(trace2048.get("passes") or 0) == 1, trace2048
assert int(trace2048.get("core_forwards") or 0) == 8, trace2048

manifest["final_health"] = health()
manifest["checks"] = {
    "public_is_vast": manifest["final_health"].get("host") == "vast-47086549",
    "full_pool_ready": int(manifest["final_health"].get("live_slots") or 0) == 2,
    "concurrency_distinct_slots": ca["slot"] != cb["slot"],
    "cancel_isolated": cancel_results["b"]["done_tokens"] == 1024 and cancel_results["a"]["done_tokens"] != 1024,
    "exact_2048": exact2048["done_tokens"] == 2048 and exact2048["stream_token_events"] == 2048,
    "pass4_exact": int((pass4["nat_trace"] or {}).get("core_forwards") or 0) == 4,
}
assert all(manifest["checks"].values()), manifest["checks"]
manifest_path = RUN / "manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
(ROOT / "latest_deployment_acceptance.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
print(json.dumps({
    "ok": True,
    "receipt_dir": str(RUN),
    "manifest": str(manifest_path),
    "checks": manifest["checks"],
    "concurrency_slots": [ca["slot"], cb["slot"]],
    "cancel_b_tokens": cancel_results["b"]["done_tokens"],
    "pass4_tok_s": pass4["button_to_done_tok_s"],
    "exact2048_tok_s": exact2048["button_to_done_tok_s"],
    "exact2048_seconds": exact2048["button_to_done_seconds"],
}, ensure_ascii=False, sort_keys=True))
