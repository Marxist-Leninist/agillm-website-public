"""Bounded, end-to-end smoke tests for the actually served Standard checkpoint."""
import json,time,urllib.request,uuid
from pathlib import Path
root='https://codex.opentransformers.online'
out=Path('/opt/agillm43_netserve/bintf_telemetry/standard-live-acceptance.json')
def get_health():
    return json.load(urllib.request.urlopen(root+'/agillm43-infer/health',timeout=6))
report={'started_at':time.time(),'expected_step':3558989,'checks':[]}
try:
    health=get_health()
    assert health.get('ready') and health.get('step')==3558989 and health.get('model_identity_verified'), 'Expected checkpoint is not ready yet'
    report['health_before']=health
    for label,mode,variable in [('AR','ar',False),('SAT fixed','sat',False),('SAT variable','sat',True),('NAT','nat',False)]:
        request_id='site-acceptance-'+uuid.uuid4().hex
        body={'prompt':'The capital of France is','max_new':8,'min_new':8,'ignore_eos':True,'mode':mode,'var':variable,'greedy':True,'temperature':0,'sampler':'ar','request_id':request_id,'nat_passes':1,'nat_greedy':True,'sat_trace':True,'sat_gate_greedy':True,'repetition_penalty':1.05,'output_row_norm_clip':0}
        req=urllib.request.Request(root+'/agillm43-infer/infer',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
        started=time.perf_counter()
        response=json.load(urllib.request.urlopen(req,timeout=90))
        result={'mode':label,'wall_s':round(time.perf_counter()-started,3),'response':response}
        report['checks'].append(result)
        out.write_text(json.dumps(report,indent=2))
        print(json.dumps(result),flush=True)
        assert response.get('ok'), 'Generation failed for '+label
    report['health_after']=get_health()
    report['ok']=report['health_after'].get('ready') and report['health_after'].get('step')==3558989
except Exception as exc:
    report['ok']=False
    report['error']=type(exc).__name__+': '+str(exc)
finally:
    report['finished_at']=time.time()
    out.write_text(json.dumps(report,indent=2))
print(json.dumps({'ok':report['ok'],'checks':len(report['checks']),'error':report.get('error'),'path':str(out)}))
