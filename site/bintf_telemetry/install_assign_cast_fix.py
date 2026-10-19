"""Convert only mismatched tensor dtypes, never duplicate the entire state dict."""
import ast,hashlib,json,os,pathlib,tempfile,time
from pathlib import Path
import torch
root=Path('/opt/agillm43_netserve')
old=root/'runtime_derived/runtime_standard_modes_20261008_mmap_v45.py'
new=root/'runtime_derived/runtime_standard_modes_20261008_mmap_v46.py'
assert not new.exists()
text=old.read_text()
helper='''def _agillm43_assign_inference_state(module, state):
    """Match ordinary copy/cast semantics while retaining compatible mapped tensors."""
    state = dict(state)
    expected = module.state_dict()
    conversions = []
    for name, value in state.items():
        target = expected.get(name)
        if torch.is_tensor(value) and torch.is_tensor(target):
            if value.shape != target.shape:
                continue  # load_state_dict retains its normal strict shape error.
            if value.dtype != target.dtype or value.device != target.device:
                conversions.append([name, str(value.dtype), str(target.dtype), value.numel()])
                state[name] = value.to(device=target.device, dtype=target.dtype)
    print("[load-profile] assigned_tensor_casts=" + json.dumps(conversions), flush=True)
    result = module.load_state_dict(state, assign=True)
    print("[load-profile] exact_storage_assignment_complete", flush=True)
    return result


'''
a=text.index('def _agillm43_assign_inference_state(');b=text.index('def _agillm43_cpu_file_load_mmap(',a)
text=text[:a]+helper+text[b:];ast.parse(text)
ns={'torch':torch,'json':json};exec(helper,ns)
for target_dtype in [torch.float32,torch.float64]:
    reference=torch.nn.Linear(8,4).to(dtype=target_dtype)
    actual=torch.nn.Linear(8,4).to(dtype=target_dtype)
    state={'weight':torch.randn(4,8,dtype=torch.float32),'bias':torch.randn(4,dtype=torch.float64)}
    reference.load_state_dict(state,assign=False)
    ns['_agillm43_assign_inference_state'](actual,state)
    x=torch.randn(2,8,dtype=target_dtype)
    assert torch.equal(reference(x),actual(x))
    assert all(p.dtype==target_dtype for p in actual.parameters())
    if target_dtype==torch.float32:assert actual.weight.data_ptr()==state['weight'].data_ptr()
new.write_text(text);new.chmod(0o644)
sha=hashlib.sha256(new.read_bytes()).hexdigest()
checkpoint=Path('/opt/agillm43_offload/ckpts/step3558989_20261008/pretrain_step03558989_from03514490_20261008T1246Z.pt')
p=Path(str(checkpoint)+'.identity.json');meta=json.loads(p.read_text());meta['runtime_file']=str(new);meta['runtime_sha256']=sha;p.write_text(json.dumps(meta,indent=2)+'\n')
env=root/'bridge.env';s=env.read_text();needle='AGILLM_BRIDGE_RUNTIME='+str(old);assert s.count(needle)==1;env.write_text(s.replace(needle,'AGILLM_BRIDGE_RUNTIME='+str(new)))
r={'time':time.time(),'runtime':str(new),'runtime_sha256':sha,'tests':['mixed source dtype vs target float32 bitwise output parity','mixed source dtype vs target float64 bitwise output parity','unchanged compatible storage pointers'],'checkpoint_unchanged':True}
(root/'bintf_telemetry/assign-cast-receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
