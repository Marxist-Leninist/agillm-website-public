"""Publish only the values needed by the three-model live monitor."""
from pathlib import Path
import json,os,time
root=Path('/opt/agillm43_netserve')
keys=('model_id','model','status','running','completed','fresh','step','seen_tokens','target_tokens','source_observed_at','observed_at','collected_at','loss','lr','raw_tok_s','median_tok_s','wall_tok_s','throughput_definition','heldout_ce','heldout_step','validation_loss','last_checkpoint_event')
try:
    source=json.loads((root/'fleet_training_status.json').read_text())
    out={'schema':'agillm.fleet.compact.v1','generated_at':time.time(),'interval_s':30}
    for key in ('standard','rpv16'):
        model=source.get('models',{}).get(key,{})
        entry={k:model[k] for k in keys if k in model and model[k] is not None}
        gpu=model.get('gpu')
        if isinstance(gpu,dict):entry['gpu']={k:gpu[k] for k in ('mem_used_mb','util_pct','temp_c') if k in gpu}
        out[key+'_live']=entry
    target=root/'bintf_telemetry/fleet-compact.json'
    tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(out,separators=(',',':'),allow_nan=False));tmp.chmod(0o644);os.replace(tmp,target)
    print('fleet_compact_bytes='+str(target.stat().st_size))
except Exception as exc:
    # Preserve prior samples and their original timestamps; the UI marks them stale.
    print('fleet_compact_unavailable:'+type(exc).__name__)
    raise SystemExit(1)
