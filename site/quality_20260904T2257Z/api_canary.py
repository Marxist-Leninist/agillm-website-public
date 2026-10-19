"""Exercise the configured primary bridge; print results, never credentials."""
import json
import os
import urllib.request

base = "http://127.0.0.1:8493"
headers = {"Content-Type": "application/json",
           "X-AGILLM-Token": os.environ["AGILLM_BRIDGE_TOKEN"]}

def request(route, body):
    req = urllib.request.Request(base + route, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.read().decode()

for mode in ("ar", "sat", "nat"):
    for exact in (False, True):
        body = dict(prompt="The capital of France is", mode=mode, max_new=16,
                    min_new=16 if exact else 0, ignore_eos=exact,
                    length_mode="exact" if exact else "natural", repetition_guard=not exact,
                    greedy=True, temperature=0, top_p=1, top_k=0,
                    nat_passes=1, nat_ar_verify=False, var=False,
                    sat_ar_first_weight=0, sat_ar_second_weight=0)
        result = json.loads(request("/infer", body))
        assert result.get("ok"), result
        stats = result["stats"]
        assert stats["tokens"] == 16 if exact else 0 <= stats["tokens"] <= 16
        assert stats["repetition_guard_enabled"] == (not exact), stats
        assert result["finish_reason"] in ("eos", "length", "repetition_loop"), result
        if mode == "nat":
            assert stats["nat_trace"]["core_forwards"] == 1, stats
            assert stats["nat_trace"]["ar_verify"] is False, stats
        print(json.dumps({"case": mode + ("_exact" if exact else "_natural"),
                          "tokens": stats["tokens"], "finish_reason": result["finish_reason"],
                          "ok": True}), flush=True)

stream = request("/stream", dict(prompt="The capital of France is", mode="ar", max_new=256,
    min_new=0, ignore_eos=False, length_mode="natural", repetition_guard=True, greedy=True,
    temperature=0, top_p=1, top_k=0, min_p=0, repetition_penalty=1.05,
    presence_penalty=0, frequency_penalty=0, penalty_last_n=128, stream=True))
events = [json.loads(line) for line in stream.splitlines() if line.strip()]
names = [event.get("event") for event in events]
assert "error" not in names, events
assert "stream_tok" in names and "generation_finish" in names and "stats" in names and "done" in names, names
assert names.index("generation_finish") < names.index("stats") < names.index("done"), names
finish = next(event["data"] for event in events if event.get("event") == "generation_finish")
print(json.dumps({"case": "ar_stream_long", "finish": finish, "stream_events": len(events), "ok": True}), flush=True)
guard = json.loads(request("/infer", dict(prompt="Hello", mode="ar", max_new=256,
    min_new=0, ignore_eos=True, length_mode="natural", repetition_guard=True, greedy=True,
    temperature=0, top_p=1, top_k=0, min_p=0, repetition_penalty=1.0,
    presence_penalty=0, frequency_penalty=0, penalty_last_n=128)))
assert guard.get("ok") and guard["stats"]["repetition_guard_enabled"], guard
print(json.dumps({"case": "ar_loop_probe", "tokens": guard["stats"]["tokens"],
                  "finish_reason": guard["finish_reason"], "ok": True}), flush=True)
print("API_CANARY_ALL_OK", flush=True)
