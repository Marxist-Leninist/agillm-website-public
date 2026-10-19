"""Restore identical FP32 inference parameters without duplicating mapped storage."""
import ast,hashlib,json,os,tempfile
from pathlib import Path
import torch
root=Path('/opt/agillm43_netserve')
old=root/'runtime_derived/runtime_standard_modes_20261008_mmap_v44.py'
new=root/'runtime_derived/runtime_standard_modes_20261008_mmap_v45.py'
assert not new.exists()
source=old.read_text()
helper='''def _agillm43_assign_inference_state(module, state):
    """Adopt checkpoint storage only when it exactly matches the existing contract."""
    state = dict(state)
    expected = module.state_dict()
    for name, value in state.items():
        target = expected.get(name)
        if torch.is_tensor(value) and torch.is_tensor(target):
            if value.shape != target.shape or value.dtype != target.dtype or value.device != target.device:
                # Preserve the existing loader's conversion behavior for other formats.
                return module.load_state_dict(state, assign=False)
    return module.load_state_dict(state, assign=True)


'''
a=source.index('def _agillm43_cpu_file_load_mmap(')
source=source[:a]+helper+source[a:]
oldline='    core.load_state_dict(_prepare_core_state_dict_for_load(core, sd["core"]))'
assert source.count(oldline)==1
source=source.replace(oldline,'    _agillm43_assign_inference_state(core, _prepare_core_state_dict_for_load(core, sd["core"]))')
needle='    print(f"[load-profile] head_state_load={time.perf_counter() - _t_stage:.1f}s", flush=True)'
assert source.count(needle)==1
replacement='''    if tie_weights:
        for _name, _head in (("ar", ar_h), ("sat", sat_h), ("nat", nat_h)):
            if _head is None:
                continue
            _saved = sd.get(_name, {}).get("proj.weight")
            if torch.is_tensor(_saved):
                if _saved.data_ptr() != core.emb.weight.data_ptr() and not torch.equal(_saved, core.emb.weight):
                    raise ValueError("Tied " + _name + " projection does not match core embedding")
            _head.proj.weight = core.emb.weight
        import gc as _gc
        _gc.collect()
'''+needle
source=source.replace(needle,replacement)
ast.parse(source)
ns={'torch':torch};exec(helper,ns)
class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.emb=torch.nn.Embedding(16,8);self.block=torch.nn.Linear(8,8)
    def forward(self,x):return self.block(self.emb(x))
torch.manual_seed(7)
with tempfile.TemporaryDirectory() as temp:
    original=Tiny();state=original.state_dict();path=Path(temp)/'state.pt';torch.save(state,path)
    mapped=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    copied=Tiny();copied.load_state_dict(mapped)
    assigned=Tiny();ns['_agillm43_assign_inference_state'](assigned,mapped)
    ids=torch.tensor([[1,3,5]])
    assert torch.equal(copied(ids),assigned(ids))
    assert assigned.emb.weight.data_ptr()==mapped['emb.weight'].data_ptr()
    head=torch.nn.Linear(8,16,bias=False);head.weight=assigned.emb.weight
    assert head.weight is assigned.emb.weight
    assert torch.equal(torch.nn.functional.linear(copied(ids),copied.emb.weight),head(assigned(ids)))
    different=Tiny().double();ns['_agillm43_assign_inference_state'](different,mapped)
    assert different.emb.weight.dtype==torch.float64
new.write_text(source);new.chmod(0o644)
sha=hashlib.sha256(new.read_bytes()).hexdigest()
checkpoint=Path('/opt/agillm43_offload/ckpts/step3558989_20261008/pretrain_step03558989_from03514490_20261008T1246Z.pt')
p=Path(str(checkpoint)+'.identity.json');meta=json.loads(p.read_text());meta['runtime_file']=str(new);meta['runtime_sha256']=sha;p.write_text(json.dumps(meta,indent=2)+'\n')
env=root/'bridge.env';s=env.read_text();needle='AGILLM_BRIDGE_RUNTIME='+str(old);assert s.count(needle)==1;env.write_text(s.replace(needle,'AGILLM_BRIDGE_RUNTIME='+str(new)))
receipt={'runtime':str(new),'sha256':sha,'tests':['copy-versus-mapped output exact equality','mapped data pointers','tied output projection identity and output equality','dtype mismatch preserves old copy/cast behavior'],'source_checkpoint_unchanged':True}
(root/'bintf_telemetry/assign-loader-receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
