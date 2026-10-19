"""CPU-only, bounded-memory export of an immutable RPV checkpoint to a tar stream.
No GPU access and no multi-gigabyte temporary archive on the training host.
"""
from pathlib import Path
import argparse, gc, hashlib, importlib.util, io, json, os, sys, tarfile, time
import torch

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--converter',default=str(Path(__file__).with_name('convert_checkpoint.py')));ap.add_argument('--tokenizer',required=True);a=ap.parse_args()
    torch.set_num_threads(1)
    source_path=Path(a.source).resolve(); source_fd=os.open(source_path,os.O_RDONLY)
    source=Path(f"/proc/self/fd/{source_fd}");identity=os.fstat(source_fd)
    spec=importlib.util.spec_from_file_location('rpv_export',a.converter)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    ck=torch.load(source,map_location='cpu',mmap=True,weights_only=False)
    ah=m.allheads_inference_payload(ck)
    assert ah is not None, 'An intact learned multimode checkpoint is required'
    state=ck['model'];source_sha=m.sha256_file(source)
    meta={'schema':'agillm.rpv16.cpu-int8-cache.v1','source_schema':ck['schema'],
          'export_source_identity':{'path':str(source_path),'inode':identity.st_ino,'size':identity.st_size,'stability':'open-fd-pre-post-fstat'},
          'source_sha256':source_sha,'step':int(ck['step']),'seen_tokens':int(ck['seen_tokens']),
          'target_alignment':ck.get('target_alignment'),
          'parameter_count':sum(v.numel() for v in state.values() if torch.is_tensor(v)),
          'quantization':'dynamic-int8-per-tensor-linear; fp32 activations; bf16 embedding; fp32 norms/router/mix',
          'allheads_export':{'present':True,'policy_weights_retained':True,
              'regret_updates':int(ah['aux']['regret_updates']),'objective_counts':ah['objective_counts'],
              'policy_quantization':'none; exact-fp32'},
          'export_method':'cpu-streaming-no-training-host-archive-v1'}
    shared={'meta':meta,'allheads':ah}
    for name in ('embed.weight','norm.weight','mix_head.weight'):
        shared[name]=state[name].detach().cpu().to(torch.bfloat16 if name=='embed.weight' else torch.float32).contiguous()
    for name in ('in_proj.weight','out_proj.weight','factor_head.weight'):
        shared[name]=m.quantize_weight(state[name])
    if 'tied_bias' in state:
        shared['tied_bias']=state['tied_bias'].detach().cpu().float().contiguous()
        shared['tied_logit_scale']=state['tied_logit_scale'].detach().cpu().float().reshape(()).contiguous()
        meta['classifier']='tied_full_vocab'
    sums=[]
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tf:
        def add_bytes(name,data):
            info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600;info.mtime=int(time.time())
            tf.addfile(info,io.BytesIO(data));sums.append((name,hashlib.sha256(data).hexdigest(),len(data)))
        def add_tensor(name,obj):
            buff=io.BytesIO();torch.save(obj,buff);data=buff.getvalue();add_bytes(name,data)
            print(json.dumps({'event':'exported','file':name,'bytes':len(data)}),file=sys.stderr,flush=True)
        add_tensor('shared.pt',shared);del shared;gc.collect()
        for si in range(16):
            pre=f'stages.{si}.';part={'meta':meta,'stage':si}
            for name,tensor in state.items():
                if name.startswith(pre):
                    part[name[len(pre):]]=m.quantize_weight(tensor) if m.is_stage_quant(name) else tensor.detach().cpu().float().contiguous()
            add_tensor(f'stage{si:02d}.pt',part);del part;gc.collect()
        tok=Path(a.tokenizer).read_bytes()
        add_bytes('tokenizer_bundle.json',tok)
        meta['tokenizer_sha256']=hashlib.sha256(tok).hexdigest()
        add_bytes('manifest.json',(json.dumps(meta,indent=2)+'\n').encode())
        add_bytes('SHA256SUMS',(''.join(sha+'  '+name+'\n' for name,sha,size in sums)).encode())
        end=os.fstat(source_fd);assert (end.st_ino,end.st_size,end.st_mtime_ns)==(identity.st_ino,identity.st_size,identity.st_mtime_ns),'source changed'
        add_bytes('CONVERSION_COMPLETE',b'ok\n')
    print(json.dumps({'event':'complete','meta':meta,'source_bytes':identity.st_size}),file=sys.stderr,flush=True)

if __name__=='__main__':main()
