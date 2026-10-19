#!/usr/bin/env python3
import json,os,subprocess,time,urllib.request
from pathlib import Path
from probe import call,C,BASE
D=Path(__file__).parent
report={'modes':[],'cancellation':[],'started_at':time.time()}
def health():return call('/health',timeout=5)[0]
def wait_idle(limit=500):
    end=time.monotonic()+limit;last=0
    while time.monotonic()<end:
        h=health()
        if h.get('ready') and not h.get('warming') and not any(h.get(k,0) for k in ('busy','active_request_count','pending_request_count')):return h
        if time.monotonic()-last>15:print('WAIT',json.dumps({k:h.get(k) for k in ('ready','warming','busy','loads','pending_request_count')}),flush=True);last=time.monotonic()
        time.sleep(.3)
    raise TimeoutError('worker did not become idle')
def worker():
    pid=subprocess.check_output(['systemctl','show','agillm43-bridge.service','-p','MainPID','--value'],text=True).strip()
    return subprocess.check_output(['pgrep','-P',pid],text=True).strip().splitlines()
def payload(mode='ar',var=False,max_new=8):
    return {'request_id':'repair-live-'+str(time.time_ns()),'prompt':'Python code:\ndef add(a, b):','mode':mode,'var':var,'max_new':max_new,'min_new':0,'greedy':True,'ignore_eos':True,'temperature':0.0,'nat_passes':1,'sat_trace':True}
def infer(body):
    try:
        out,wall=call('/infer',body,timeout=90)
        assert out.get('ok'),out
        assert (out.get('stats') or {}).get('tokens')==body['max_new'],out
        return out,wall
    except Exception:
        call('/cancel',{'request_id':body['request_id']},timeout=5)
        raise
try:
    h=wait_idle();assert h.get('cancel_preserves_worker') and h.get('model_identity_verified'),h
    report['initial_health']=h;initial_worker=worker();report['worker_pids']=initial_worker
    baseline=None
    for mode,var in [('ar',False),('sat',False),('sat',True),('nat',False)]:
        wait_idle(60)
        out,wall=infer(payload(mode,var))
        if mode=='ar':baseline=out['text']
        row={'mode':mode,'variable':var,'wall_s':wall,'tokens':out['stats']['tokens'],'queue_s':out.get('queue_wait_s'),'text':out['text'],'stats':out['stats']}
        report['modes'].append(row);print('MODE_PASS',json.dumps(row),flush=True)
    for kind in ['explicit_stop','stream_disconnect']:
        before=wait_idle(60);old_worker=worker();body=payload(max_new=128)
        req=urllib.request.Request(BASE+'/stream',data=json.dumps(body).encode(),headers={'Content-Type':'application/json','X-AGILLM-Token':C.get('AGILLM_BRIDGE_TOKEN','')})
        start=time.monotonic();response=urllib.request.urlopen(req,timeout=90);cancel_reply=None;observed=False
        try:
            for raw in response:
                ev=json.loads(raw)
                if ev.get('event')=='error':raise RuntimeError(ev)
                if ev.get('event')=='stream_tok':
                    observed=True;stop_at=time.monotonic()
                    if kind=='explicit_stop':
                        cancel_reply,_=call('/cancel',{'request_id':body['request_id']},timeout=5)
                        assert cancel_reply.get('cooperative'),cancel_reply
                    break
            assert observed,'no real token received before cancellation'
        finally:
            response.close()
        after=wait_idle(20);recovered=time.monotonic()-stop_at;new_worker=worker()
        assert after['loads']==before['loads'] and new_worker==old_worker,(before,after,old_worker,new_worker)
        follow,wall=infer(payload())
        assert follow['text']==baseline,'greedy AR output changed after cancellation'
        row={'kind':kind,'stop_to_idle_s':round(recovered,3),'next_8_tokens_s':wall,'same_worker':True,'same_load_count':after['loads'],'next_output_identical':True,'reply':cancel_reply}
        report['cancellation'].append(row);print('CANCEL_PASS',json.dumps(row),flush=True)
    report['final_health']=health();report['status']='PASS';report['finished_at']=time.time()
except Exception as e:
    report['status']='FAIL';report['error']=repr(e);report['finished_at']=time.time();print('LIVE_TEST_FAILURE',repr(e),flush=True)
finally:
    (D/'live_tests.json').write_text(json.dumps(report,indent=2));print('FINAL',json.dumps({k:v for k,v in report.items() if k not in ('modes','initial_health','final_health')}),flush=True)
if report['status']!='PASS':raise SystemExit(1)
