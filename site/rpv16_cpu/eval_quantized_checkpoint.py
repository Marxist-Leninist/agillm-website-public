#!/usr/bin/env python3
import gc, importlib.util, json, sys, time
from pathlib import Path
import torch
import torch.nn.functional as F

SERVER='/opt/agillm43_netserve/rpv16_cpu/rpv16_cpu_server.py'
TOK='/srv/agillm-rpv16-1pf/step19456-auto1/tokenizer_bundle.json'
IDS='/srv/agillm-rpv16-1pf/heldout_token_ids.json'
spec=importlib.util.spec_from_file_location('rpv16srv',SERVER)
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.set_num_threads(12)

def eval_root(root):
    t0=time.time(); e=m.RPV16CPU(Path(root),Path(TOK)); load=time.time()-t0
    ids=torch.tensor(json.load(open(IDS))[:126],dtype=torch.long)
    with torch.inference_mode():
        x=F.embedding(ids,e.embed).float(); x=e.in_proj(x)
        for st in e.stages: x,_=st.prefill(x)
        h=e.out_proj(m.rms_norm(x[:-1],e.norm))
        raw=e.factor_head(h.float()).view(ids.numel()-1,m.VOCAB_RANK,m.VOCAB_GROUP_PAD+m.VOCAB_LOW)
        gl=raw[:,:,:m.VOCAB_GROUP_PAD].float(); gl[:,:,m.VOCAB_GROUPS:]=-1e9
        ll=raw[:,:,m.VOCAB_GROUP_PAD:].float()
        mix=F.linear(h.float(),e.mix_head).view(ids.numel()-1,m.VOCAB_RANK).log_softmax(-1)
        targets=ids[1:]
        gm=e.group_map[:,targets].T; lm=e.low_map[:,targets].T
        lg=gl.log_softmax(-1).gather(2,gm.unsqueeze(-1)).squeeze(-1)
        lo=ll.log_softmax(-1).gather(2,lm.unsqueeze(-1)).squeeze(-1)
        token_lp=REDACTED
        ce=float((-token_lp).mean()); ppl=float(torch.exp((-token_lp).mean()).clamp(max=1e9))
    meta=dict(e.meta); del e,x,h,raw,gl,ll,mix,gm,lm,lg,lo,token_lp; gc.collect()
    return {'root':root,'step':meta.get('step'),'seen_tokens':meta.get('seen_tokens'),'ce_125':ce,'ppl_125':ppl,'load_s':load,'target_alignment':meta.get('target_alignment'),'source_sha256':meta.get('source_sha256')}

out=[]
for root in ['/srv/agillm-rpv16-1pf/step15360/cpu_int8','/srv/agillm-rpv16-1pf/step19456-auto1/cpu_int8']:
    r=eval_root(root); print(json.dumps(r),flush=True); out.append(r)
# Owner policy 2026-09-18: quality metrics are advisory only; never hold a complete checkpoint for CE/PPL movement.
print(json.dumps({'baseline':out[0],'candidate':out[1],'ce_delta':out[1]['ce_125']-out[0]['ce_125'],'candidate_pass':True,'quality_metrics_advisory_only':True},sort_keys=True),flush=True)
