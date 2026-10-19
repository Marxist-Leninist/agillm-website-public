#!/usr/bin/env python3
import json, os, time, urllib.request, sys
from pathlib import Path
ROOT=Path('/opt/agillm43_netserve')
def config():
    out={}
    for line in (ROOT/'bridge.env').read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#') and '=' in line:
            k,v=line.split('=',1);out[k.strip()]=v.strip()
    return out
C=config(); BASE='http://127.0.0.1:'+os.environ.get('PROBE_PORT','8493')
def call(path,payload=None,timeout=75):
    headers={'Content-Type':'application/json','X-AGILLM-Token':C.get('AGILLM_BRIDGE_TOKEN','')}
    req=urllib.request.Request(BASE+path,data=json.dumps(payload).encode() if payload is not None else None,headers=headers)
    t=time.monotonic()
    with urllib.request.urlopen(req,timeout=timeout) as r: out=json.load(r)
    return out,round(time.monotonic()-t,3)
if __name__=='__main__':
    results=[]
    for mode,var in [('ar',False),('sat',False),('sat',True),('nat',False)]:
        h,_=call('/health');print('health',json.dumps({k:h.get(k) for k in ('ready','warming','busy','loads','checkpoint_step')}),flush=True)
        if not h.get('ready') or h.get('busy') or h.get('pending_request_count'): break
        body={'request_id':'repair-probe-'+str(time.time_ns()),'prompt':'Python code:\ndef add(a, b):','mode':mode,'var':var,'max_new':8,'min_new':0,'greedy':True,'ignore_eos':True,'temperature':0.0,'nat_passes':1,'sat_trace':True}
        try:
            out,seconds=call('/infer',body)
            row={'mode':mode,'var':var,'wall_s':seconds,**out}
            results.append(row);print(json.dumps(row),flush=True)
        except Exception as e: print(type(e).__name__,str(e),flush=True);break
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(results,indent=2))
