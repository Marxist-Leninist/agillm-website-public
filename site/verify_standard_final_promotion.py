"""Verify only the pinned Standard final cutover against current public/tunnel/runtime identity."""
from __future__ import annotations
import hashlib,json,re,subprocess,time
from pathlib import Path
from urllib.request import urlopen
BASE=Path("/opt/agillm43_netserve/receipts/standard-final-20260930")
STEP=3133615
CKPT="pretrain_final.pt"
HOST="vast-v100-52631126"
SHA="53aba5977671b7d588227f5de3206bef9822d659d59bf87c77d086bcda17a1c4"
TOKENIZER="REDACTED"
RUNTIME="790cd4719a8d2837b5be45bf72432ac953dbb2688896dae5f3c1881631c4d9aa"
BRIDGE="d6c9c5c1d3d3785ce15d1ae67f696555bc3e081ecc5c9d6dd4f7d8993a922f5d"
UNIT_SHA="132d0a490cc7d4350fb2444539e202daa954b578c506c2ab7d6e0666a98c80c4"
UPSTREAM_SHA="1a2b15895e7e73aa29ef8d264d557038be3fc831014be2a66ad210dcab1b9d56"
FILES={
 "promotion.json":"d3434fbd3439b2caf1624487f497c738a74549d7698370e2a3dcc72a1b8fa56a",
 "public_stream_verification.json":"f86eaf78040f5d144f379803f5160cee35f9edd51f561980103212be0b83106e",
 "checkpoint_hash_verification.json":"c7e5b51716332f37c413df3732e90872479edf7b2b87632d6e023e705f551283",
 "smoke_verified.json":"d439f798e026ed06f41fd6478a3f0703561474079e9a0034e7a22c58512df381",
 "launch.json":"625e21f635304a9ce5fd7e6aaf7b561e1d4b7ea8c7be0415739fcd4da902de99",
}
MODES={"ar","sat_fixed","sat_variable","nat"}
def require(condition,reason):
    if not condition: raise ValueError(reason)
def health_matches(h):
    m=h.get("model") if isinstance(h,dict) and isinstance(h.get("model"),dict) else {}
    return bool(isinstance(h,dict) and h.get("ok") is True and h.get("ready") is True and not h.get("warming")
                and h.get("ckpt")==CKPT and h.get("step")==m.get("step")==STEP and h.get("host")==HOST
                and h.get("device")=="cuda" and m.get("source_checkpoint_sha256")==SHA
                and m.get("tokenizer_sha256")==TOKENIZER and set(h.get("canonical_modes") or [])==MODES)
def validate_documents(docs,health,now):
    require(health_matches(health),"public_health_identity_mismatch")
    require(isinstance(health.get("observed_at"),(int,float)) and -10<=now-health["observed_at"]<=100,"public_health_stale")
    p=docs["promotion.json"];s=docs["public_stream_verification.json"];h=docs["checkpoint_hash_verification.json"];sm=docs["smoke_verified.json"];l=docs["launch.json"]
    require(p.get("ok") is True and p.get("step")==STEP and p.get("verified_utc"),"cutover_receipt_incomplete")
    require(p.get("reload",{}).get("rc")==p.get("nginx_test",{}).get("rc")==0,"cutover_reload_not_verified")
    require(health_matches(p.get("public_health")) and health_matches(p.get("candidate_health")),"receipt_health_identity_mismatch")
    require(s.get("ok") is True and health_matches(s.get("health")),"stream_health_identity_mismatch")
    rows=s.get("rows") or []
    require(len(rows)==4 and {r.get("mode") for r in rows}==MODES,"four_mode_stream_proof_missing")
    for row in rows:
        events=row.get("events") or []
        done=[e for e in events if e.get("event")=="done"]
        require(len(done)==1 and done[0].get("ok") is True and done[0].get("ckpt")==CKPT and done[0].get("host")==HOST and bool(done[0].get("text")),"stream_completion_mismatch")
        require(not any(e.get("event")=="error" for e in events),"stream_error_recorded")
    require(sm.get("ok") is True and all(sm.get("checks",{}).get(k) is True for k in ("ar_output_parity","sat_fixed_all_emissions_stride2","sat_variable_regret_active","nat_output_parity")),"mode_semantics_checks_incomplete")
    require(h.get("step")==STEP and h.get("checkpoint_main_sha256")==SHA and h.get("tokenizer_sha256")==TOKENIZER,"checkpoint_hash_proof_identity_mismatch")
    shards=h.get("shards") or []
    require(h.get("all_sha256_match") is True and h.get("count")==len(shards)==62 and len({x.get("name") for x in shards})==62 and all(x.get("ok") is True and x.get("size_ok") is True for x in shards),"shard_verification_incomplete")
    require(l.get("instance_id")==52631126 and l.get("port")==18593 and l.get("runtime_sha256")==RUNTIME and l.get("bridge_sha256")==BRIDGE and l.get("checkpoint_sha256")==SHA,"launch_identity_mismatch")
    return True
REMOTE=r'''
import json,hashlib,time
from pathlib import Path
b=Path("/workspace/agillm-ui-serving-20260930")
l=json.loads((b/"launch.json").read_text());pid=int(l["pid"])
bridge=b/"agillm43_http_bridge_cuda.py";runtime=b/"runtime_standard_modes_20260930.py"
ckpt=Path("/workspace/gb10_nvfp4_openai_20260915/live_canary/pretrain_final.pt")
args=(Path("/proc")/str(pid)/"cmdline").read_bytes().split(b"\0")
assert str(bridge).encode() in args
workers=[]
for p in Path("/proc").iterdir():
 if not p.name.isdigit():continue
 try:
  status=(p/"status").read_text()
  parent=int(next(x.split()[1] for x in status.splitlines() if x.startswith("PPid:")))
  a=(p/"cmdline").read_bytes().split(b"\0")
  if parent==pid and str(runtime).encode() in a and b"infer" in a and str(ckpt).encode() in a and b"cuda" in a:workers.append(int(p.name))
 except (OSError,ValueError,StopIteration):pass
assert len(workers)==1
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps({"observed_at":time.time(),"bridge_pid":pid,"worker_pid":workers[0],"runtime_sha256":sha(runtime),"bridge_sha256":sha(bridge),"checkpoint_sha256":sha(ckpt),"tokenizer_sha256":sha(Path(str(ckpt)+".tokenizer.json"))}))
'''
def runtime_attestation():
    r=subprocess.run(["ssh","-F","/dev/null","-i","/root/.ssh/goddess_vast","-p","43062","-o","BatchMode=yes","-o","IdentitiesOnly=yes","-o","StrictHostKeyChecking=yes","-o","ConnectTimeout=5","root@x.x.x.x","python3 -"],input=REMOTE,text=True,capture_output=True,timeout=12,check=True)
    a=json.loads(r.stdout)
    require(a.get("runtime_sha256")==RUNTIME and a.get("bridge_sha256")==BRIDGE and a.get("checkpoint_sha256")==SHA and a.get("tokenizer_sha256")==TOKENIZER,"live_runtime_or_checkpoint_hash_mismatch")
    return a
def verify(health):
    try:
        docs={}
        for name,expected in FILES.items():
            raw=(BASE/name).read_bytes()
            require(hashlib.sha256(raw).hexdigest()==expected,"pinned_receipt_changed:"+name)
            docs[name]=json.loads(raw)
        validate_documents(docs,health,time.time())
        unit=Path("/etc/systemd/system/agillm-standard-v100-tunnel.service")
        require(hashlib.sha256(unit.read_bytes()).hexdigest()==UNIT_SHA,"tunnel_unit_changed")
        active=subprocess.run(["systemctl","is-active","--quiet","agillm-standard-v100-tunnel.service"],timeout=3)
        require(active.returncode==0,"tunnel_not_active")
        config=Path("/opt/agillm43_netserve/agillm43-infer-edge.nginx.conf").read_text()
        match=re.search(r"upstream\s+agillm43_infer_backend\s*\{(.*?)\}",config,re.S)
        require(match is not None,"standard_upstream_missing")
        block="\n".join(x.strip() for x in match[1].splitlines() if x.strip() and not x.strip().startswith("#"))
        require(hashlib.sha256(block.encode()).hexdigest()==UPSTREAM_SHA,"standard_upstream_changed")
        with urlopen("http://127.0.0.1:18593/health",timeout=3) as response: tunnel=json.load(response)
        require(health_matches(tunnel),"tunnel_health_identity_mismatch")
        attestation=runtime_attestation()
        p=docs["promotion.json"]
        normalized={"schema":"agillm.standard.final-promotion.v1","status":"complete","step":STEP,
                    "completed_at":p["verified_utc"],"serving_checkpoint":CKPT,"primary_health":p["public_health"],
                    "sha256_total":64,"sha256_verified":64,"smoke":{m:{"ok":True,"source":"pinned public_stream_verification.json"} for m in MODES},
                    "verification":{"scope":"actual_serving_checkpoint","public_health_observed_at":health["observed_at"],
                                    "runtime_attestation":attestation,"pinned_artifacts":FILES,"tunnel_unit_sha256":UNIT_SHA,"standard_upstream_sha256":UPSTREAM_SHA}}
        return normalized,str(BASE/"promotion.json"),FILES["promotion.json"],"matching_standard_final_receipt_and_current_public_tunnel_runtime_identity"
    except Exception as error:
        return None,None,None,"standard_final_verification_failed:"+str(error)[:180]
