"""Convert legacy serving caches to mapped ZIP storage without changing checkpoints."""
import ast,hashlib,json,os,tempfile
from pathlib import Path
import torch
root=Path('/opt/agillm43_netserve')
old=root/'runtime_derived/runtime_standard_modes_20261008_mmap_v43.py'
new=root/'runtime_derived/runtime_standard_modes_20261008_mmap_v44.py'
assert not new.exists()
source=old.read_text()
helper='''def _agillm43_cpu_file_load_mmap(path, map_location="cpu", weights_only=False):
    """Privately map exact checkpoint tensors, converting legacy cache files once."""
    if str(map_location) != "cpu":
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    try:
        return torch.load(path, map_location=map_location,
                          weights_only=weights_only, mmap=True)
    except RuntimeError as exc:
        if "mmap can only be used with files saved with" not in str(exc):
            raise
    import fcntl, gc, hashlib, json
    original = pathlib.Path(path).resolve()
    stat = original.stat()
    identity = str(original) + ":" + str(stat.st_size) + ":" + str(stat.st_mtime_ns)
    key = REDACTED
    cache_root = pathlib.Path(os.environ.get("AGILLM43_MMAP_CACHE_DIR", "/opt/agillm43_netserve/mmap_cache"))
    cache_root.mkdir(parents=True, exist_ok=True)
    mapped = cache_root / (key + ".pt")
    with (cache_root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not mapped.is_file():
            state = torch.load(original, map_location="cpu", weights_only=weights_only)
            temporary = mapped.with_suffix(".tmp")
            try:
                torch.save(state, temporary, _use_new_zipfile_serialization=True)
                temporary.chmod(0o600)
                os.replace(temporary, mapped)
            finally:
                del state
                gc.collect()
                if temporary.exists():
                    temporary.unlink()
            print("[ckpt-mmap] converted legacy cache " + original.name, flush=True)
    return torch.load(mapped, map_location="cpu", weights_only=weights_only, mmap=True)


'''
a=source.index('def _agillm43_cpu_file_load_mmap(')
b=source.index('def _agillm43_load_pt(',a)
updated=source[:a]+helper+source[b:]
ast.parse(updated)
ns={'torch':torch,'pathlib':__import__('pathlib'),'os':os}
exec(helper,ns)
with tempfile.TemporaryDirectory() as tmp:
    os.environ['AGILLM43_MMAP_CACHE_DIR']=tmp+'/mapped'
    weight=torch.arange(128,dtype=torch.float32).reshape(16,8)
    payload={'weight':weight,'alias':weight,'bias':torch.arange(8,dtype=torch.float16)}
    for legacy in (True,False):
        path=Path(tmp)/('legacy.pt' if legacy else 'modern.pt')
        torch.save(payload,path,_use_new_zipfile_serialization=not legacy)
        expected=hashlib.sha256(path.read_bytes()).hexdigest()
        actual=ns['_agillm43_cpu_file_load_mmap'](path,weights_only=True)
        for key in payload:
            assert torch.equal(payload[key],actual[key]) and payload[key].dtype==actual[key].dtype
        assert actual['weight'].data_ptr()==actual['alias'].data_ptr()
        actual['weight'][0,0]=-1
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected
        del actual
        second=ns['_agillm43_cpu_file_load_mmap'](path,weights_only=True)
        assert torch.equal(second['weight'],payload['weight'])
    del os.environ['AGILLM43_MMAP_CACHE_DIR']
new.write_text(updated);new.chmod(0o644)
sha=hashlib.sha256(new.read_bytes()).hexdigest()
checkpoint=Path('/opt/agillm43_offload/ckpts/step3558989_20261008/pretrain_step03558989_from03514490_20261008T1246Z.pt')
identity=Path(str(checkpoint)+'.identity.json')
meta=json.loads(identity.read_text());meta['runtime_file']=str(new);meta['runtime_sha256']=sha
identity.write_text(json.dumps(meta,indent=2)+'\n')
env=root/'bridge.env';text=env.read_text();needle='AGILLM_BRIDGE_RUNTIME='+str(old);assert text.count(needle)==1
text=text.replace(needle,'AGILLM_BRIDGE_RUNTIME='+str(new));env.write_text(text)
print(json.dumps({'runtime':str(new),'sha256':sha,'tests':['legacy and modern tensor equality','dtype equality','tied aliases','private writes','unchanged source SHA256','cache reuse'],'checkpoint_original_unchanged':True}))
