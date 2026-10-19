import json,hashlib,py_compile
from pathlib import Path
p=Path("/opt/agillm43_netserve/update_vast_live_status.py")
expected="6b4acb931e22bb9611a3b731e32637368a566a8145bd9846204ce7784754a4f1"
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,"updater source changed"
s=p.read_text()
changes={
'saved_step = step_of(saved_ckpt) or step_of(serving_ckpt)':'saved_step = train.get("checkpoint_step") or step_of(saved_ckpt) or train.get("step")',
'serving_step = step_of(serving_ckpt)':'serving_step = health.get("serving_step") or health.get("step") or step_of(serving_ckpt)',
'"mirror_age_sec": 0,':'"mirror_age_sec": max(0, time.time()-train["observed_at"]) if isinstance(train.get("observed_at"),(int,float)) else None,',
'"verified_at": now,':'"verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime(health["observed_at"])) if isinstance(health.get("observed_at"),(int,float)) else None,',
'status["trainer_live_step"] = live_step':'status["trainer_live_step"] = live_step if train.get("trainer_status") == "training" and train.get("trainer_fresh") else None\n    status["trainer_status"] = train.get("trainer_status")\n    status["trainer_fresh"] = train.get("trainer_fresh")\n    status["training_observed_at"] = train.get("observed_at")\n    status["training_source_observed_at"] = train.get("source_observed_at")',
}
for a,b in changes.items():
 assert s.count(a)==1,(a,s.count(a));s=s.replace(a,b)
extra=json.loads(Path("/opt/agillm43_netserve/fleet-updater-extra-20260930.json").read_text())["extra"]
s=s.replace('\nif __name__ == "__main__":',extra+'\nif __name__ == "__main__":')
candidate=p.with_name("update_vast_live_status.fleet-candidate.py");candidate.write_text(s);py_compile.compile(str(candidate),doraise=True)
print(json.dumps({"candidate":str(candidate),"before":expected,"after":hashlib.sha256(candidate.read_bytes()).hexdigest()}))
