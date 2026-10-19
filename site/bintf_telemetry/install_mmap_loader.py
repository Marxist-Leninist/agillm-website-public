"""Install an inference-only, exact-weight mapped loader and a bounded worker."""
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import torch

root=Path('/opt/agillm43_netserve')
old=root/'runtime_derived/runtime_standard_modes_20261003_fastpath_v42.py'
new=root/'runtime_derived/runtime_standard_modes_20261008_mmap_v43.py'
assert not new.exists(), 'Runtime candidate already exists; inspect before repeating'
text=old.read_text()
helper='''def _agillm43_cpu_file_load_mmap(path, map_location="cpu", weights_only=False):
    """Map CPU checkpoint storage privately; preserve values, dtype and aliases."""
    use_mmap = str(map_location) == "cpu"
    try:
        return torch.load(path, map_location=map_location,
                          weights_only=weights_only, mmap=use_mmap)
    except RuntimeError as exc:
        # Legacy non-ZIP serialization cannot be mapped. Do not hide other failures.
        if use_mmap and "mmap can only be used with files saved with" in str(exc):
            return torch.load(path, map_location=map_location, weights_only=weights_only)
        raise


'''
start=text.index('def _agillm43_load_pt(path,')
end=text.index('\ndef _do_delta_save(',start)
loader=text[start:end]
for value in ['cached','path']:
    original='torch.load('+value+', map_location=map_location, weights_only=weights_only)'
    assert original in loader
    loader=loader.replace(original,'_agillm43_cpu_file_load_mmap('+value+', map_location=map_location, weights_only=weights_only)')
text=text[:start]+helper+loader+text[end:]
original='cache_dir = path.parent / ".agillm43_decompressed_cache"'
replacement='cache_dir = pathlib.Path(os.environ.get("AGILLM43_DECOMPRESS_CACHE_DIR", "/opt/agillm43_netserve/weight_cache")) / path.parent.name'
assert text.count(original)==1
text=text.replace(original,replacement)
ast.parse(text)
# Verify exact values, dtypes and tied aliases without loading any large model.
ns={'torch':torch};exec(compile(helper,'mapped_loader_test','exec'),ns)
with tempfile.TemporaryDirectory() as tmp:
    file=Path(tmp)/'sample.pt'
    weight=torch.arange(128,dtype=torch.float32).reshape(16,8)
    torch.save({'weight':weight,'alias':weight,'bias':torch.arange(8,dtype=torch.float16)},file)
    expected=torch.load(file,map_location='cpu',weights_only=True)
    actual=ns['_agillm43_cpu_file_load_mmap'](file,weights_only=True)
    for key in expected:
        assert torch.equal(expected[key],actual[key]) and expected[key].dtype==actual[key].dtype
    assert actual['weight'].data_ptr()==actual['alias'].data_ptr()
    actual['weight'][0,0]=-1
    assert torch.load(file,weights_only=True)['weight'][0,0]==0, 'Mapping must be private'
new.write_text(text);new.chmod(0o644)
sha=hashlib.sha256(new.read_bytes()).hexdigest()
checkpoint=Path('/opt/agillm43_offload/ckpts/step3558989_20261008/pretrain_step03558989_from03514490_20261008T1246Z.pt')
identity=Path(str(checkpoint)+'.identity.json')
backup=identity.with_suffix('.json.before-mmap')
backup.write_bytes(identity.read_bytes());backup.chmod(0o600)
meta=json.loads(identity.read_text());meta['runtime_file']=str(new);meta['runtime_sha256']=sha
identity.write_text(json.dumps(meta,indent=2)+'\n')
env=root/'bridge.env'
updates={'AGILLM_BRIDGE_RUNTIME':str(new),'AGILLM43_DECOMPRESS_CACHE':'1','AGILLM43_DECOMPRESS_CACHE_DIR':str(root/'weight_cache')}
lines=env.read_text().splitlines()
for key,value in updates.items():
    if any(line.startswith(key+'=') for line in lines):
        lines=[key+'='+value if line.startswith(key+'=') else line for line in lines]
    else:lines.append(key+'='+value)
tmp=env.with_suffix('.mmap-tmp');tmp.write_text('\n'.join(lines)+'\n');tmp.chmod(env.stat().st_mode & 0o777);os.replace(tmp,env)
drop=Path('/etc/systemd/system/agillm43-bridge.service.d/memory-budget.conf')
drop.write_text('[Service]\nMemoryHigh=7G\nMemoryMax=8G\nMemorySwapMax=0\nOOMPolicy=stop\n')
receipt={'at':time.time(),'runtime':str(new),'runtime_sha256':sha,'previous_runtime':str(old),'tests':['bitwise values','dtypes','tied aliases','private mapping does not modify saved weights'],'external_cache':str(root/'weight_cache'),'memory_high_gib':7,'memory_max_gib':8,'checkpoint_step':3558989}
(root/'bintf_telemetry/mmap-loader-receipt.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
