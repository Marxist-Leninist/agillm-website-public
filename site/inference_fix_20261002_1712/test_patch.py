#!/usr/bin/env python3
import ast, contextlib, hashlib, importlib.util, io, json, os, sys, threading, time, types
from pathlib import Path
D=Path(__file__).parent
spec=importlib.util.spec_from_file_location('test_bridge',D/'bridge.patched.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
checks=[]
def ok(name):checks.append(name);print('PASS',name,flush=True)
class Proc:
    def __init__(self):self.pid=900000000+os.getpid();self.kills=0;self.agillm_cooperative_cancel=True
    def poll(self):return None
    def kill(self):self.kills+=1
p=Proc();slot={'id':1,'proc':p,'threads':8,'dtype':'fp32','lock':threading.Lock(),'ckpt':'test'}
b._planned_slots=lambda live,*a,**kw:live
b.slots[:]=[slot];b.active_requests['unit-active']=slot
code,out=b.cancel_request('unit-active');marker=Path(b._cooperative_cancel_path(p.pid,'unit-active'))
assert code==200 and out['cooperative'] and marker.exists() and p.kills==0
ok('active cancellation preserves worker and creates request-specific marker')
b.active_requests.clear();marker.unlink(missing_ok=True)
b.pending_requests.add('unit-queued');code,out=b.cancel_request('unit-queued')
assert code==202 and out['queued'] and p.kills==0 and not b.pending_requests
ok('queued cancellation never touches resident worker')
b.active_requests['other']=slot;code,out=b.cancel_request('absent-before-register')
assert code==202 and b.active_requests['other'] is slot and p.kills==0
assert b._register_pending_request('absent-before-register')=='canceled'
ok('early cancel tombstone rejects reordered request and isolates other requests')
b.active_requests.clear();b.cancelled_requests.clear();b.slots.clear()
repairs=[];b._schedule_pool_repair=lambda:repairs.append(True)
t=time.monotonic();result,wait=b.acquire_slot(max_wait_s=.12)
assert result is None and .10<=time.monotonic()-t<.5 and repairs
ok('queue deadline includes unavailable/warming worker, without synchronous load')
b._remember_cancel_locked('unit-wait-cancel');t=time.monotonic();result,wait=b.acquire_slot(.3,'unit-wait-cancel')
assert result is None and time.monotonic()-t<.08
ok('queued cancellation exits immediately')
b.cancelled_requests.clear();b.slots[:]=[slot];slot['lock'].acquire();repairs.clear()
t=time.monotonic();result,wait=b.acquire_slot(.12,'unit-busy')
assert result is None and time.monotonic()-t<.5 and not repairs
slot['lock'].release()
ok('busy worker timeout neither reloads nor duplicates worker')
result,wait=b.acquire_slot(.2,'unit-idle');assert result is slot and wait<.05;slot['lock'].release()
ok('idle worker acquired immediately')
# Execute only infer_server's AST with fake models; do not import the full trainer
# or load weights. Verify pre-cancel, in-forward cancel, pipe resync, and reuse.
class Module:
    def __init__(self):self.hooks=[]
    def register_forward_pre_hook(self,h):self.hooks.append(h);return h
    def forward(self):
        for h in self.hooks:assert h(self,()) is None

def exercise(source,requests):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='infer_server');node.decorator_list=[]
    tree=ast.Module(body=[node],type_ignores=[]);ast.fix_missing_locations(tree)
    inst={k:Module() for k in ('core','ar_h','sat_h','nat_h')};inst['repair_execution']={}
    fake_sys=types.SimpleNamespace(stdin=io.StringIO('\n'.join(requests)))
    def prepare(args):return inst
    def generate(inst,args):
        inst['core'].forward()
        if args.prompt=='cancel-during':
            rid='during-request';path=Path('/run/agillm43-cancel')/(str(os.getpid())+'-'+hashlib.sha256(rid.encode()).hexdigest()+'.cancel');path.touch(mode=0o600)
        inst['ar_h'].forward();print('OUTPUT:'+args.prompt)
    import copy
    ns={'sys':fake_sys,'json':json,'copy':copy,'torch':types.SimpleNamespace(inference_mode=contextlib.nullcontext),'_agillm43_prepare_infer_instance':prepare,'_agillm43_generate_from_instance':generate,'_repair_validate_inference_request':lambda *args:None}
    exec(compile(tree,'<infer-server-unit>','exec'),ns)
    cap=io.StringIO();args=types.SimpleNamespace(mode='ar')
    with contextlib.redirect_stdout(cap):ns['infer_server'](args)
    return cap.getvalue()
root=Path('/run/agillm43-cancel');root.mkdir(mode=0o700,exist_ok=True)
pre=root/(str(os.getpid())+'-'+hashlib.sha256(b'pre-request').hexdigest()+'.cancel');pre.touch(mode=0o600)
requests=[json.dumps({'prompt':'pre-cancel','_bridge_request_id':'pre-request'}),json.dumps({'prompt':'cancel-during','_bridge_request_id':'during-request'}),'@@BRIDGE_SYNC@@',json.dumps({'prompt':'next-good','_bridge_request_id':'good-request'})]
out=exercise((D/'runtime.patched.py').read_text(),requests)
assert 'cooperative_cancel=1' in out and out.count('RuntimeError: request canceled')==2
assert 'OUTPUT:pre-cancel' not in out and 'OUTPUT:cancel-during' not in out and 'OUTPUT:next-good' in out
assert 'JSONDecodeError' in out and out.count('[INFER_SERVER_RESULT_END]')==4 and not pre.exists()
ok('runtime cancellation before and during forward, pipe resync, next request succeeds')
requests=[json.dumps({'prompt':'normal-'+mode,'mode':mode,'var':var,'_bridge_request_id':'normal-'+str(i)}) for i,(mode,var) in enumerate([('ar',False),('sat',False),('sat',True),('nat',False)])]
a=exercise((D/'runtime.before.py').read_text(),requests).splitlines()[1:]
c=exercise((D/'runtime.patched.py').read_text(),requests).splitlines()[1:]
assert a==c
ok('unchanged uncanceled AR/SAT-fixed/SAT-variable/NAT server output in harness')
report={'passed':len(checks),'checks':checks,'scope':'mocked lifecycle/queue tests; actual model validation still required'}
(D/'unit_tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
