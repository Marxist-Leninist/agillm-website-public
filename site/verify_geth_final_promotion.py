"""Verify the pinned GETH CPU Standard final deployment without touching trainers."""
from __future__ import annotations
import hashlib,json,os,re,subprocess,time
from pathlib import Path
from urllib.request import urlopen

BASE=Path("/opt/agillm43_netserve/receipts/geth-primary-20260930")
NET=Path("/opt/agillm43_netserve")
CKPT=Path("/opt/agillm43_offload/ckpts/step3133615_20260930/pretrain_final.pt")
RUNTIME_PATH=NET/"runtime_derived/runtime_standard_modes_20260930.py"
BRIDGE_PATH=NET/"agillm43_http_bridge.py"
STEP=3133615
SHA="53aba5977671b7d588227f5de3206bef9822d659d59bf87c77d086bcda17a1c4"
TOKENIZER="REDACTED"
RUNTIME="790cd4719a8d2837b5be45bf72432ac953dbb2688896dae5f3c1881631c4d9aa"
BRIDGE="27859ba1a3c5a7afdce3b3c77880113c6424ab6e440ee54af84d4ccc08b09b35"
MODES={"ar","sat_fixed","sat_variable","nat"}
# Filled from completed immutable receipts by finalize_geth_status_candidate.py.
PINS = {'CPU_READY.json': '5a1dcbb0627c81e3eec06e3c1121085cef2803fa155ef4cc410e6dc874458c77', 'CPU_SMOKE.json': '7d73ea73a3d1d67628d5e7aab33e360f1f8f9d23b71c7701463b9a7290bc745e', 'PUBLIC_GETH_VERIFIED.json': '2836778c22b408c2beab7a3970285f416e6a7e5e593581638dcdbc3d21eeb482', 'STAGE_VERIFIED.json': '7dcca4929640ede7a0882f5d81dabdb9cf1bcec199e841cbe7d5adcef820ade5'}
def require(condition,reason):
    if not condition: raise ValueError(reason)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()
def health_matches(h):
    m=h.get("model") if isinstance(h,dict) and isinstance(h.get("model"),dict) else {}
    return bool(isinstance(h,dict) and h.get("ok") is True and h.get("ready") is True
                and not h.get("warming") and h.get("model_identity_verified") is True
                and h.get("ckpt")==CKPT.name and h.get("host")=="geth-cpu" and h.get("device")=="cpu"
                and (h.get("checkpoint_step") or h.get("step"))==m.get("step")==STEP
                and m.get("source_checkpoint_sha256")==SHA and m.get("tokenizer_sha256")==TOKENIZER
                and set(h.get("canonical_modes") or [])==MODES)
def completion_matches(result,tokens):
    return bool(isinstance(result,dict) and result.get("ok") is True
                and result.get("host")=="geth-cpu" and result.get("ckpt")==CKPT.name
                and result.get("stats",{}).get("tokens")==tokens and bool(result.get("text")))
def validate_documents(docs,health,now,document_hashes):
    require(health_matches(health),"public_health_identity_mismatch")
    require(isinstance(health.get("observed_at"),(int,float)) and -10<=now-health["observed_at"]<=100,"public_health_stale")
    ready=docs["CPU_READY.json"];smoke=docs["CPU_SMOKE.json"];public=docs["PUBLIC_GETH_VERIFIED.json"];stage=docs["STAGE_VERIFIED.json"]
    require(ready.get("ok") is True and health_matches(ready.get("health")),"cpu_readiness_unverified")
    require(ready.get("bridge_sha256")==BRIDGE and ready.get("runtime_sha256")==RUNTIME,"cpu_readiness_source_mismatch")
    require(smoke.get("ok") is True and health_matches(smoke.get("health_before")) and health_matches(smoke.get("health_after")),"cpu_smoke_unverified")
    rows=smoke.get("modes") or []
    require(len(rows)==4 and {r.get("mode") for r in rows}==MODES,"four_mode_proof_missing")
    for row in rows:
        require(completion_matches(row.get("result"),8),"mode_completion_mismatch:"+str(row.get("mode")))
        trace=row["result"].get("stats",{}).get("sat_trace",{})
        if row["mode"]=="sat_fixed":
            require(trace.get("mode")=="sat-fixed" and trace.get("stride_hist",{}).get("2")==4,"sat_fixed_stride_proof_missing")
        if row["mode"]=="sat_variable":
            require(trace.get("mode")=="sat-var" and trace.get("policy")=="regret","sat_variable_regret_proof_missing")
    stream=smoke.get("stream") or {}
    require(stream.get("events",0)>0 and completion_matches(stream.get("done"),16),"cpu_stream_unverified")
    require(stream["done"].get("event")=="done","cpu_stream_final_event_missing")
    cancel=smoke.get("cancel") or {};response=cancel.get("response") or {}
    require(cancel.get("received_tokens_before_stop",0)>0 and response.get("queued") is False
            and response.get("request_id")=="geth-primary-20260930-cancel","cpu_stop_unverified")
    require(stage.get("ok") is True and stage.get("step")==STEP and stage.get("checkpoint")==str(CKPT)
            and stage.get("checkpoint_sha256")==SHA and stage.get("runtime_sha256")==RUNTIME
            and stage.get("tokenizer_sha256")==TOKENIZER,"staging_identity_mismatch")
    files=stage.get("files") or []
    require(len(files)==65 and len({r.get("rel") for r in files})==65 and all(r.get("ok") is True for r in files),"staging_files_unverified")
    shards=[r for r in files if str(r.get("rel","")).startswith(CKPT.name+".shards/")]
    require(len(shards)==62,"staging_shard_count_mismatch")
    require(public.get("ok") is True and health_matches(public.get("health")) and public.get("verified_utc"),"public_cutover_unverified")
    require(public.get("upstream")=="127.0.0.1:8493" and public.get("checkpoint_sha256")==SHA
            and public.get("bridge_sha256")==BRIDGE and public.get("runtime_sha256")==RUNTIME,"public_cutover_identity_mismatch")
    require(public.get("cpu_smoke_sha256")==document_hashes["CPU_SMOKE.json"],"public_smoke_pin_mismatch")
    return True
def runtime_attestation():
    state=subprocess.check_output(["systemctl","show","agillm43-bridge.service","-p","MainPID","-p","ActiveState"],text=True,timeout=5)
    props=dict(line.split("=",1) for line in state.splitlines() if "=" in line)
    require(props.get("ActiveState")=="active","cpu_service_not_active")
    pid=int(props.get("MainPID") or 0);require(pid>0,"cpu_service_pid_missing")
    proc=Path("/proc")/str(pid);args=(proc/"cmdline").read_bytes().split(b"\0")
    require(str(BRIDGE_PATH).encode() in args,"cpu_bridge_process_mismatch")
    require("/system.slice/agillm43-bridge.service" in (proc/"cgroup").read_text(),"cpu_bridge_cgroup_mismatch")
    workers=[]
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():continue
        try:
            parent=int(next(x.split()[1] for x in (p/"status").read_text().splitlines() if x.startswith("PPid:")))
            args=(p/"cmdline").read_bytes().split(b"\0")
            if parent==pid and str(RUNTIME_PATH).encode() in args and b"infer" in args and str(CKPT).encode() in args:
                i=args.index(b"--device");require(args[i+1]==b"cpu","cpu_worker_device_mismatch")
                workers.append(int(p.name))
        except (OSError,StopIteration,IndexError):continue
    require(len(workers)==1,"cpu_worker_count_mismatch")
    env={}
    for line in (NET/"bridge.env").read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k,v=line.split("=",1);env[k]=v
    require(env.get("AGILLM_BRIDGE_CKPT")==env.get("AGILLM_BRIDGE_PINNED_CKPT")==str(CKPT)
            and env.get("AGILLM_BRIDGE_PINNED_SHA256")==SHA and env.get("AGILLM_BRIDGE_REQUIRE_PIN")=="1"
            and env.get("AGILLM_BRIDGE_RUNTIME")==str(RUNTIME_PATH),"cpu_config_identity_mismatch")
    a={"observed_at":time.time(),"host":"geth-cpu","device":"cpu","bridge_pid":pid,"worker_pid":workers[0],
       "runtime_sha256":sha(RUNTIME_PATH),"bridge_sha256":sha(BRIDGE_PATH),"checkpoint_sha256":sha(CKPT),
       "tokenizer_sha256":sha(str(CKPT)+".tokenizer.json"),"env_sha256":sha(NET/"bridge.env")}
    require(a["runtime_sha256"]==RUNTIME and a["bridge_sha256"]==BRIDGE and a["checkpoint_sha256"]==SHA
            and a["tokenizer_sha256"]==TOKENIZER,"cpu_runtime_or_checkpoint_hash_mismatch")
    # Hashing source on disk is meaningful only if it predates the processes that loaded it.
    btime=int(next(x.split()[1] for x in Path("/proc/stat").read_text().splitlines() if x.startswith("btime ")))
    hz=os.sysconf("SC_CLK_TCK")
    def started(p):
        fields=(Path("/proc")/str(p)/"stat").read_text().rsplit(") ",1)[1].split()
        return btime+int(fields[19])/hz
    require(BRIDGE_PATH.stat().st_mtime<=started(pid)+1,"bridge_changed_after_process_start")
    require(RUNTIME_PATH.stat().st_mtime<=started(workers[0])+1,"runtime_changed_after_worker_start")
    require((proc/"cmdline").read_bytes().split(b"\0").count(str(BRIDGE_PATH).encode())==1,"cpu_bridge_replaced_during_attestation")
    return a
def verify(health):
    try:
        names={"CPU_READY.json","CPU_SMOKE.json","PUBLIC_GETH_VERIFIED.json","STAGE_VERIFIED.json"}
        require(set(PINS)==names,"geth_receipt_pins_missing")
        docs={};hashes={}
        for name,expected in PINS.items():
            raw=(BASE/name).read_bytes();digest=hashlib.sha256(raw).hexdigest()
            require(digest==expected,"pinned_receipt_changed:"+name)
            docs[name]=json.loads(raw);hashes[name]=digest
        validate_documents(docs,health,time.time(),hashes)
        config=NET/"agillm43-infer-edge.nginx.conf"
        require(sha(config)==docs["PUBLIC_GETH_VERIFIED.json"]["nginx_sha256"],"geth_nginx_config_changed")
        match=re.search(r"upstream\s+agillm43_infer_backend\s*\{(.*?)\}",config.read_text(),re.S)
        require(match is not None,"standard_upstream_missing")
        servers=re.findall(r"^\s*server\s+([^;]+);",match[1],re.M)
        primaries=[s.split()[0] for s in servers if "backup" not in s.split()[1:] and "down" not in s.split()[1:]]
        require(primaries==["127.0.0.1:8493"],"geth_primary_upstream_mismatch")
        with urlopen("http://127.0.0.1:8493/health",timeout=5) as response:local=json.load(response)
        require(health_matches(local),"local_cpu_health_identity_mismatch")
        attestation=runtime_attestation()
        require(attestation["env_sha256"]==docs["CPU_READY.json"]["env_sha256"],"cpu_env_changed_after_readiness")
        p=docs["PUBLIC_GETH_VERIFIED.json"]
        normalized={"schema":"agillm.standard.geth-final-promotion.v1","status":"complete","step":STEP,
                    "completed_at":p["verified_utc"],"serving_checkpoint":CKPT.name,"primary_health":p["health"],
                    "sha256_total":65,"sha256_verified":65,
                    "smoke":{m:{"ok":True,"source":"pinned CPU_SMOKE.json"} for m in MODES},
                    "verification":{"scope":"actual_serving_checkpoint","host":"geth-cpu","device":"cpu",
                                    "public_health_observed_at":health["observed_at"],"runtime_attestation":attestation,
                                    "pinned_artifacts":dict(PINS),"nginx_sha256":p["nginx_sha256"]}}
        return normalized,str(BASE/"PUBLIC_GETH_VERIFIED.json"),PINS["PUBLIC_GETH_VERIFIED.json"],"matching_geth_final_receipt_and_current_public_local_runtime_identity"
    except Exception as error:
        return None,None,None,"geth_final_verification_failed:"+str(error)[:180]
