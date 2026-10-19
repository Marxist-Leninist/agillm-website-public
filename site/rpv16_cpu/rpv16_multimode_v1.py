"""RPV16 CPU multimode inference with checkpoint-native SAT and NAT contracts.

The optional module installs into the existing HTTP server. It uses the saved
slot embeddings and learned CE-regret policy, not repeated AR labelled as SAT.
NAT is same-position masked reconstruction with the checkpoint's causal mask.
Single-sequence routing is token-local: balanced cross-position routing would
couple future prompt positions and make an incremental KV cache inconsistent.
"""
from __future__ import annotations
import asyncio
import json
import math
import socket
import time
import types
import uuid
from pathlib import Path
import torch
import torch.nn.functional as F
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

REVISION = "rpv16-native-multimode-v2-natbatched-20260925"
CANONICAL = ("ar", "sat_fixed", "sat_variable", "nat")


def mode_key(body):
    mode = str(body.get("mode") or "ar").lower().replace("-", "_")
    if mode == "sat":
        return "sat_variable" if bool(body.get("var", False)) else "sat_fixed"
    if mode in ("sat_var", "sat_v"):
        return "sat_variable"
    if mode not in CANONICAL:
        raise ValueError("Unsupported RPV16 mode: " + mode)
    return mode


def attention_mask(length, window, first2=None):
    pos = torch.arange(length)
    end = pos.clone()
    if first2 is not None:
        if len(first2) != length:
            raise ValueError("SAT partition length differs from token sequence")
        pair = torch.as_tensor(first2, dtype=torch.bool)
        if bool(pair[-1]) or bool((pair[:-1] & pair[1:]).any()):
            raise ValueError("Overlapping or incomplete SAT pair")
        end += pair.long()
    keep = pos[None, :] <= end[:, None]
    if window >= 0:
        keep &= pos[None, :] >= pos[:, None] - int(window)
    return keep[None, None, :, :]


def select_stride(predicted_log_ce, lam, merge_cost=0.0):
    pred = predicted_log_ce.detach().float()
    if pred.numel() != 2 or not bool(torch.isfinite(pred).all()):
        raise ValueError("Invalid learned SAT regret prediction")
    ce = pred.reshape(2).clamp(-20, 20).exp()
    gap = max(0.0, float(ce[1] - ce[0]))
    return (2 if float(lam) - gap - float(merge_cost) > 0 else 1), gap


def check_aux(aux):
    shapes = {"shift_emb": (2,256), "regret_updates": (),
              "stride_regret.0.weight": (256,256), "stride_regret.0.bias": (256,),
              "stride_regret.2.weight": (2,256), "stride_regret.2.bias": (2,)}
    if not isinstance(aux, dict) or set(aux) != set(shapes):
        raise ValueError("Checkpoint does not contain the complete learned SAT policy")
    for key, shape in shapes.items():
        if tuple(aux[key].shape) != shape or not bool(torch.isfinite(aux[key]).all()):
            raise ValueError("Invalid auxiliary tensor " + key)


def install(ns):
    Base = ns["RPV16CPU"]
    D, QH, KH, HD = ns["D"], ns["Q_HEADS"], ns["KV_HEADS"], ns["HEAD_DIM"]
    rms = ns["rms_norm"]
    app = ns["app"]
    legacy_health = ns["health"]

    class MultimodeCPU(Base):
        def __init__(self, root, tokenizer_path):
            super().__init__(root, tokenizer_path)
            payload = torch.load(root / "shared.pt", map_location="cpu", mmap=True, weights_only=False)
            ah = payload.get("allheads")
            self.aux = None
            self.mode_error = "No learned all-heads state in checkpoint"
            self.objective_counts = {}
            self.aux_config = {}
            if ah is not None:
                check_aux(ah.get("aux"))
                self.aux = {k:v.detach().clone().contiguous() for k,v in ah["aux"].items()}
                self.aux_config = ah["config"]
                self.objective_counts = ah["objective_counts"]
                eff = self.aux_config["effective"]
                self.policy_lambda = float(eff["lam"])
                self.policy_merge_cost = float(eff["mu"])
                if not (math.isfinite(self.policy_lambda) and self.policy_lambda > 0 and
                        math.isfinite(self.policy_merge_cost) and self.policy_merge_cost >= 0):
                    raise ValueError("Invalid checkpoint SAT speed/quality policy")
                self.mode_error = ""
            del payload
            def route(stage, hidden):
                return F.linear(hidden.reshape(-1,D).float(), stage.router).argmax(-1)
            # A request is one independent sequence, never a six-sequence router group.
            for stage in self.stages:
                stage._route = types.MethodType(route, stage)
            self.runtime_revision = REVISION

        def available_modes(self):
            modes = ["ar"]
            if self.aux is not None:
                if self.objective_counts.get("sat",0) > 0:
                    modes.append("sat_fixed")
                    if int(self.aux["regret_updates"]) > 0:
                        modes.append("sat_variable")
                if (self.objective_counts.get("nat",0) > 0 and
                        self.aux_config.get("nat_attn") == "causal" and
                        self.aux_config.get("nat_mask_id") == 2):
                    modes.append("nat")
            return modes

        @torch.inference_mode()
        def all_hidden(self, ids, first2=None):
            token_ids = torch.tensor(ids, dtype=torch.long)
            x = self.in_proj(F.embedding(token_ids, self.embed).float())
            length = len(ids)
            for stage in self.stages:
                h = rms(x, stage.n1)
                q = stage.q(h).reshape(length,QH,HD).permute(1,0,2)[None]
                k = stage.k(h).reshape(length,KH,HD).permute(1,0,2)
                v = stage.v(h).reshape(length,KH,HD).permute(1,0,2)
                kr = k.repeat_interleave(QH//KH,dim=0)[None]
                vr = v.repeat_interleave(QH//KH,dim=0)[None]
                mask = attention_mask(length, stage.window, first2)
                a = F.scaled_dot_product_attention(q,kr,vr,attn_mask=mask,dropout_p=0.0)
                a = a[0].permute(1,0,2).reshape(length,D)
                x = x.float() + stage.o(a)
                x = x + stage._ffn(rms(x,stage.n2))
            return self.out_proj(rms(x,self.norm))

        @torch.inference_mode()
        def prefill(self, ids):
            return super().prefill(ids)

        @torch.inference_mode()
        def step(self, token_id, caches):
            return super().step(token_id,caches)

        @torch.inference_mode()
        def head(self, hidden):
            return self._hidden_to_logp(hidden)

        @torch.inference_mode()
        def sample(self, hidden, history, body, forbid_mask=False):
            scores = self._hidden_to_logp(hidden)
            if forbid_mask:
                scores = scores.clone(); scores[2] = float("-inf")
            return self._sample(scores,history,body)

        @torch.inference_mode()
        def policy(self, hidden):
            a = self.aux
            z = F.gelu(F.linear(hidden.detach().float(),a["stride_regret.0.weight"],a["stride_regret.0.bias"]))
            z = F.linear(z,a["stride_regret.2.weight"],a["stride_regret.2.bias"])
            return select_stride(z,self.policy_lambda,self.policy_merge_cost)

        def events(self, body, request_id, entered):
            mode = mode_key(body)
            if mode not in self.available_modes():
                raise ValueError("RPV16 mode unavailable for this checkpoint: " + mode)
            prompt = str(body.get("prompt") or "")
            original_ids = list(self.tokenizer.encode(prompt).ids) or [1]
            ids = original_ids[-ns["MAX_PROMPT"]:]
            limit = min(ns["MAX_NEW"],256) if mode == "nat" else ns["MAX_NEW"]
            requested = int(body.get("max_new",body.get("max_new_tokens",32)) or 32)
            budget = max(1,min(limit,requested))
            minimum = max(0,min(budget,int(body.get("min_new",0) or 0)))
            ignore_eos = bool(body.get("ignore_eos",False))
            ckpt = f"rpv16_1pf_step{self.meta.get('step',0)}_cpu_int8.pt"
            yield {"event":"start","host":socket.gethostname(),"slot":0,"threads":ns["THREADS"],
                   "queue_wait_s":0.0,"ckpt":ckpt,"request_id":request_id}
            yield {"event":"stream_begin","data":{"mode":("sat" if mode.startswith("sat_") else mode),
                   "var":mode=="sat_variable","canonical_mode":mode,"sampler":("sat" if mode.startswith("sat_") else mode),"runtime":REVISION,
                   "step":self.meta.get("step"),"source_checkpoint_sha256":ns["SOURCE_CHECKPOINT_SHA"]}}
            start = time.perf_counter(); generated=[]; previous=""; forwards=0; prefill_s=0.0
            strides={"1":0,"2":0}; policy_strides={"1":0,"2":0}; passes_done=0; first_token_s=None
            def token_event(token, top):
                nonlocal previous,first_token_s
                generated.append(int(token))
                decoded = self.tokenizer.decode(generated,skip_special_tokens=False)
                piece = decoded[len(previous):] if decoded.startswith(previous) else self.tokenizer.decode([int(token)],skip_special_tokens=False)
                previous=decoded
                if first_token_s is None:first_token_s=time.perf_counter()-entered
                return {"event":"stream_tok","data":{"text":piece,"i":len(generated)-1,"top":top}}
            if mode == "ar":
                h,caches=self.prefill(ids);forwards+=1;prefill_s=time.perf_counter()-start
                for i in range(budget):
                    if self.cancel.is_set():break
                    token,top=self.sample(h[0],ids+generated,body)
                    if token in self.eos_ids and i>=minimum and not ignore_eos:break
                    yield token_event(token,top);strides["1"]+=1
                    if i+1<budget and not self.cancel.is_set():h=self.step(token,caches);forwards+=1
            elif mode.startswith("sat_"):
                ended=False
                while len(generated)<budget and not ended and not self.cancel.is_set():
                    context=ids+generated
                    stride=1;pair_h=None
                    if len(context)>=2 and budget-len(generated)>=2:
                        first=[False]*len(context);first[-2]=True
                        pair_h=self.all_hidden(context,first);forwards+=1
                        if mode=="sat_fixed":stride=2
                        else:
                            stride,gap=self.policy(pair_h[-1]);policy_strides[str(stride)]+=1
                    if stride==2:
                        hidden=(pair_h[-2]+self.aux["shift_emb"][0],pair_h[-1]+self.aux["shift_emb"][1])
                    else:
                        h=self.all_hidden(context);forwards+=1
                        hidden=(h[-1]+self.aux["shift_emb"][0],)
                    if not prefill_s:prefill_s=time.perf_counter()-start
                    committed=0
                    for h in hidden:
                        token,top=self.sample(h,ids+generated,body)
                        if token in self.eos_ids and len(generated)>=minimum and not ignore_eos:
                            ended=True;break
                        if self.cancel.is_set():ended=True;break
                        yield token_event(token,top);committed+=1
                    if committed:strides[str(committed)]+=1
            else:
                passes=max(1,min(8,int(body.get("nat_passes",4) or 1)))
                masked=set(range(budget));output=[2]*budget;confidence=[0.0]*budget;tops=[[] for _ in range(budget)]
                local_body=dict(body)
                if body.get("nat_greedy",True):local_body.update(greedy=True,temperature=0.0)
                for p in range(passes):
                    if self.cancel.is_set():break
                    sequence=ids+[2 if i in masked else output[i] for i in range(budget)]
                    assert sequence[:len(ids)]==ids
                    hidden=self.all_hidden(sequence);forwards+=1
                    if not prefill_s:prefill_s=time.perf_counter()-start
                    # Batch only the fp32 tied classifier; keep sequential sampling,
                    # repetition penalties, confidence ranking and refinement unchanged.
                    positions=sorted(masked)
                    head_batch=32 if getattr(self,"tied",False) else 1
                    for offset in range(0,len(positions),head_batch):
                        chunk=positions[offset:offset+head_batch]
                        scores_batch=None
                        if getattr(self,"tied",False):
                            selected=hidden[[len(ids)+i for i in chunk]].float()
                            scores_batch=(F.linear(selected,self.tied_w)*self.tied_scale+self.tied_bias).log_softmax(-1)
                        for row,i in enumerate(chunk):
                            if scores_batch is None:
                                token,top=self.sample(hidden[len(ids)+i],ids+output[:i],local_body,forbid_mask=True)
                            else:
                                scores=scores_batch[row].clone()
                                scores[2]=float("-inf")
                                token,top=self._sample(scores,ids+output[:i],local_body)
                            output[i]=token;tops[i]=top
                            confidence[i]=next((entry["prob"] for entry in top if entry["id"]==token),0.0)
                    passes_done+=1
                    remaining=math.ceil(budget*(1-(p+1)/passes))
                    masked=set(sorted(range(budget),key=lambda i:(confidence[i],i))[:remaining])
                    yield {"event":"progress","data":{"canonical_mode":"nat","pass":passes_done,"passes":passes,
                                                        "masked_remaining":len(masked),"emitted_tokens":0}}
                if not self.cancel.is_set():
                    for i,token in enumerate(output):
                        if token in self.eos_ids and i>=minimum and not ignore_eos:break
                        yield token_event(token,tops[i])
            elapsed=max(1e-9,time.perf_counter()-start)
            wall=max(1e-9,time.perf_counter()-entered)
            completion=self.tokenizer.decode(generated,skip_special_tokens=False)
            stats={"tokens":len(generated),"elapsed_s":elapsed,"tok_s":len(generated)/elapsed,
                   "request_to_done_s":wall,"request_to_done_tok_s":len(generated)/wall,
                   "prefill_s":prefill_s,"time_to_first_token_s":first_token_s,
                   "prompt_tokens":len(ids),"prompt_truncated":len(original_ids)>len(ids),
                   "original_prompt_tokens":len(original_ids),"max_prompt_tokens":ns["MAX_PROMPT"],
                   "requested_tokens":requested,"max_new_tokens_effective":budget,
                   "mode":("sat" if mode.startswith("sat_") else mode),"canonical_mode":mode,
                   "var":mode=="sat_variable","model_forwards":forwards,"emitted_stride_hist":strides,
                   "policy_stride_hist":policy_strides,"nat_passes":passes_done,
                   "policy_basis":("saved_learned_ce_regret_plus_speed_reward_minus_merge_cost" if mode=="sat_variable" else None),
                   "routing":"single_sequence_token_local_argmax","runtime_revision":REVISION,
                   "model":"AGILLM-GB10-1PF RPV16","device":"GETH CPU","threads":ns["THREADS"],
                   "quantization":self.meta.get("quantization"),"step":self.meta.get("step"),
                   "seen_tokens":self.meta.get("seen_tokens"),"source_checkpoint_sha256":ns["SOURCE_CHECKPOINT_SHA"],
                   "serving_snapshot_sha256":self.meta.get("source_sha256"),"cancelled":self.cancel.is_set()}
            yield {"event":"stats","stats":stats}
            yield {"event":"done","ok":True,"text":prompt+completion,"completion":completion,
                   "stats":stats,"host":socket.gethostname(),"slot":0,"threads":ns["THREADS"],
                   "queue_wait_s":0.0,"ckpt":ckpt,"request_id":request_id}

    ns["RPV16CPU"]=MultimodeCPU
    def enriched_health():
        d=legacy_health();e=ns["engine"]();modes=e.available_modes()
        d.update(canonical_modes=modes,runtime_revision=REVISION,busy=e.lock.locked(),
                 active_request_id=e.active_request_id,mode_limits={"nat":{"max_new_tokens":256,"max_passes":8}})
        d["capabilities"]={}
        for mode in CANONICAL:
            available=mode in modes
            d["capabilities"][mode]={"trained":available,"backend_configured":True,"worker_ready":available,
                "available":available,"requestable":available,"reason":"" if available else e.mode_error or "Saved objective/policy evidence missing",
                "semantics":{"runtime_mode":"sat" if mode.startswith("sat_") else mode,"runtime_sampler":("sat" if mode.startswith("sat_") else mode),
                    "variable":mode=="sat_variable","canonical_mode":mode,
                    "commit_policy":{"ar":"one_token_kv_cache","sat_fixed":"native_block_causal_two_slot",
                      "sat_variable":"learned_ce_regret_dynamic_one_or_two","nat":"causal_masked_suffix_parallel_refinement"}[mode]}}
        d["model"]["objective_counts"]=e.objective_counts
        d["model"]["regret_updates"]=int(e.aux["regret_updates"]) if e.aux is not None else 0
        return d

    def iterator(body,entered):
        e=ns["engine"]()
        if not e.lock.acquire(blocking=False):
            yield {"event":"error","error":"RPV16 CPU worker busy","status":429};return
        e.cancel.clear();req_id=str(body.get("request_id") or uuid.uuid4());e.active_request_id=req_id
        try:
            yield from e.events(body,req_id,entered)
        except Exception as exc:
            yield {"event":"error","error":f"{type(exc).__name__}: {exc}"}
        finally:
            e.active_request_id=None;e.lock.release()

    def validate(body):
        if not isinstance(body,dict):raise ValueError("Request must be a JSON object")
        key=mode_key(body)
        if key not in ns["engine"]().available_modes():raise ValueError("Mode unavailable for loaded checkpoint: "+key)
        for field in ("max_new","max_new_tokens","min_new","nat_passes"):
            if field in body:int(body[field])

    async def infer(request:Request):
        entered=time.perf_counter()
        try:body=await request.json();validate(body)
        except (ValueError,TypeError) as exc:return JSONResponse({"ok":False,"error":str(exc)},status_code=400)
        def run():
            last=None
            for item in iterator(body,entered):
                if item["event"]=="error":return item
                if item["event"]=="done":last=item
            return last or {"event":"error","error":"No completion"}
        result=await asyncio.to_thread(run)
        if result.get("event")=="error":return JSONResponse({"ok":False,**result},status_code=result.get("status",500))
        return result

    async def stream(request:Request):
        entered=time.perf_counter()
        try:body=await request.json();validate(body)
        except (ValueError,TypeError) as exc:return JSONResponse({"ok":False,"error":str(exc)},status_code=400)
        def lines():
            for item in iterator(body,entered):yield json.dumps(item,allow_nan=False)+"\n"
        return StreamingResponse(lines(),media_type="application/x-ndjson",headers={"Cache-Control":"no-store","X-Accel-Buffering":"no"})

    app.router.routes[:]=[r for r in app.router.routes if getattr(r,"path",None) not in ("/health","/infer","/stream")]
    app.add_api_route('/health',enriched_health,methods=['GET'])
    app.add_api_route('/infer',infer,methods=['POST'])
    app.add_api_route('/stream',stream,methods=['POST'])
    ns['health']=enriched_health
    ns['MULTIMODE_REVISION']=REVISION
