#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import math
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from tokenizers import Tokenizer
from torch.ao.nn.quantized.dynamic import Linear as DynamicQLinear

ROOT = Path(os.environ.get("RPV16_CPU_ROOT", "/srv/agillm-rpv16-1pf/step15360/cpu_int8"))
TOKENIZER_PATH = Path(os.environ.get("RPV16_TOKENIZER", "/srv/agillm-rpv16-1pf/step15360/tokenizer_bundle.json"))
SOURCE_CHECKPOINT_SHA = os.environ.get("RPV16_SOURCE_CHECKPOINT_SHA", "c44bc186dad53042626578658d5c96dc81cf1b9eedef6547838e0b47dc081ac2")
THREADS = max(1, int(os.environ.get("RPV16_CPU_THREADS", "12")))
MAX_PROMPT = max(8, int(os.environ.get("RPV16_MAX_PROMPT", "128")))
MAX_NEW = max(1, int(os.environ.get("RPV16_MAX_NEW", "1024")))
VOCAB = 129_280
TOKEN_DIM = 256
D = 1_280
STAGES = 16
EXPERTS = 6
FFN = 5_120
Q_HEADS = 20
KV_HEADS = 5
HEAD_DIM = 64
VOCAB_GROUPS = 505
VOCAB_LOW = 256
VOCAB_RANK = 16
VOCAB_GROUP_PAD = 512
AFFINE_A = (1,3,7,9,11,13,17,19,21,23,27,29,31,33,37,39)
AFFINE_B = tuple((i * 7919) % VOCAB for i in range(VOCAB_RANK))
WINDOWS = (256,256,256,256,512,512,512,512,1024,1024,1024,1024,-1,-1,-1,-1)

# GETH's Skylake CPU has strong INT8 kernels but no native BF16 dot-product ISA.
torch.set_num_threads(THREADS)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
torch.backends.quantized.engine = "x86"


def load_tokenizer(path: Path) -> Tokenizer:
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and isinstance(obj.get("tokenizer_bundle"), dict):
        raw = obj["tokenizer_bundle"].get("tokenizer.json")
        if isinstance(raw, str):
            return Tokenizer.from_str(raw)
    if isinstance(obj, dict) and isinstance(obj.get("tokenizer.json"), str):
        return Tokenizer.from_str(obj["tokenizer.json"])
    return Tokenizer.from_str(path.read_text())


def pack_linear(qweight: torch.Tensor) -> DynamicQLinear:
    if not qweight.is_quantized:
        raise TypeError("expected quantized linear weight")
    out_features, in_features = qweight.shape
    layer = DynamicQLinear(in_features, out_features, bias_=False, dtype=torch.qint8)
    layer.set_weight_bias(qweight, None)
    layer.eval()
    return layer


def rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + 1e-5)
    return y * weight


PERMS = torch.tensor(
    [[(a*j + b) % EXPERTS for j in range(EXPERTS)] for a in (1, EXPERTS - 1) for b in range(EXPERTS)],
    dtype=torch.long,
)


class StageCPU:
    def __init__(self, payload: dict[str, Any], window: int):
        self.window = int(window)
        self.n1 = payload["n1.weight"].float()
        self.n2 = payload["n2.weight"].float()
        self.router = payload["router.weight"].float()
        self.q = pack_linear(payload["attn.q.weight"])
        self.k = pack_linear(payload["attn.k.weight"])
        self.v = pack_linear(payload["attn.v.weight"])
        self.o = pack_linear(payload["attn.o.weight"])
        self.experts = []
        for i in range(EXPERTS):
            self.experts.append((
                pack_linear(payload[f"experts.{i}.gate.weight"]),
                pack_linear(payload[f"experts.{i}.up.weight"]),
                pack_linear(payload[f"experts.{i}.down.weight"]),
            ))

    def _route(self, h: torch.Tensor) -> torch.Tensor:
        flat = h.reshape(-1, D).float()
        logits = F.linear(flat, self.router)
        if flat.shape[0] % EXPERTS == 0 and flat.shape[0] >= EXPERTS:
            groups = flat.shape[0] // EXPERTS
            gl = logits.view(EXPERTS, groups, EXPERTS).permute(1, 0, 2).contiguous()
            pick = PERMS.view(1, PERMS.shape[0], EXPERTS, 1).expand(groups, -1, -1, 1)
            scores = gl.unsqueeze(1).expand(-1, PERMS.shape[0], -1, -1).gather(3, pick).squeeze(3).sum(2)
            best = scores.argmax(1)
            return PERMS.index_select(0, best).transpose(0, 1).reshape(-1)
        return logits.argmax(-1)

    def _ffn(self, h: torch.Tensor) -> torch.Tensor:
        flat = h.reshape(-1, D).float()
        route = self._route(flat)
        y = torch.empty_like(flat)
        for i, (gate, up, down) in enumerate(self.experts):
            idx = (route == i).nonzero(as_tuple=False).flatten()
            if idx.numel():
                z = flat.index_select(0, idx)
                yi = down(F.silu(gate(z)) * up(z))
                y.index_copy_(0, idx, yi)
        return y.view_as(h)

    def prefill(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: [S,D]. Match incremental decoding at every prompt length:
        # each stage retains its trained window, including the current token.
        h = rms_norm(x, self.n1)
        s = h.shape[0]
        q = self.q(h).view(s, Q_HEADS, HEAD_DIM).permute(1, 0, 2).unsqueeze(0)
        k0 = self.k(h).view(s, KV_HEADS, HEAD_DIM).permute(1, 0, 2).contiguous()
        v0 = self.v(h).view(s, KV_HEADS, HEAD_DIM).permute(1, 0, 2).contiguous()
        kr = k0.repeat_interleave(Q_HEADS // KV_HEADS, dim=0).unsqueeze(0)
        vr = v0.repeat_interleave(Q_HEADS // KV_HEADS, dim=0).unsqueeze(0)
        positions = torch.arange(s, device=h.device)
        distance = positions[:, None] - positions[None, :]
        mask = distance >= 0
        if self.window >= 0:
            mask &= distance <= self.window
        mask = mask.view(1, 1, s, s)
        a = F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask, dropout_p=0.0, is_causal=False)
        a = a.squeeze(0).permute(1, 0, 2).reshape(s, D)
        x = x.float() + self.o(a)
        x = x + self._ffn(rms_norm(x, self.n2))
        return x, (k0, v0)

    def step(self, x: torch.Tensor, cache: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: [1,D]
        h = rms_norm(x, self.n1)
        q = self.q(h).view(1, Q_HEADS, HEAD_DIM).permute(1, 0, 2)
        kn = self.k(h).view(1, KV_HEADS, HEAD_DIM).permute(1, 0, 2)
        vn = self.v(h).view(1, KV_HEADS, HEAD_DIM).permute(1, 0, 2)
        k0 = torch.cat((cache[0], kn), dim=1)
        v0 = torch.cat((cache[1], vn), dim=1)
        if self.window >= 0 and k0.shape[1] > self.window + 1:
            k0 = k0[:, -(self.window + 1):]
            v0 = v0[:, -(self.window + 1):]
        kr = k0.repeat_interleave(Q_HEADS // KV_HEADS, dim=0)
        vr = v0.repeat_interleave(Q_HEADS // KV_HEADS, dim=0)
        scores = torch.matmul(q, kr.transpose(-1, -2)) / math.sqrt(HEAD_DIM)
        probs = scores.softmax(-1)
        a = torch.matmul(probs, vr).permute(1, 0, 2).reshape(1, D)
        x = x.float() + self.o(a)
        x = x + self._ffn(rms_norm(x, self.n2))
        return x, (k0, v0)


class RPV16CPU:
    def __init__(self, root: Path, tokenizer_path: Path):
        t0 = time.time()
        self.root = root
        self.tokenizer = load_tokenizer(tokenizer_path)
        shared = torch.load(root / "shared.pt", map_location="cpu", weights_only=False)
        self.meta = dict(shared["meta"])
        self.embed = shared["embed.weight"]
        self.in_proj = pack_linear(shared["in_proj.weight"])
        self.norm = shared["norm.weight"].float()
        self.out_proj = pack_linear(shared["out_proj.weight"])
        # 2026-09-23 cowork: tied full-vocab classifier (trainer RPV16_TIED_HEAD): logits = (h @ E^T) * scale + bias
        # with E = the token embedding. Exports without tied_bias keep the product-vocabulary path unchanged.
        self.tied = "tied_bias" in shared
        if self.tied:
            self.tied_w = self.embed.float().contiguous()
            self.tied_bias = shared["tied_bias"].float().contiguous()
            self.tied_scale = float(shared["tied_logit_scale"])
            self.factor_head = None
            self.mix_head = None
        else:
            self.factor_head = pack_linear(shared["factor_head.weight"])
            self.mix_head = shared["mix_head.weight"].float()
        del shared
        self.stages = []
        for i in range(STAGES):
            payload = torch.load(root / f"stage{i:02d}.pt", map_location="cpu", weights_only=False)
            self.stages.append(StageCPU(payload, WINDOWS[i]))
            del payload
        ids = torch.arange(VOCAB, dtype=torch.long)
        aa = torch.tensor(AFFINE_A, dtype=torch.long)[:, None]
        bb = torch.tensor(AFFINE_B, dtype=torch.long)[:, None]
        code = torch.remainder(aa * ids[None, :] + bb, VOCAB)
        self.group_map = torch.div(code, VOCAB_LOW, rounding_mode="floor")
        self.low_map = torch.remainder(code, VOCAB_LOW)
        self.eos_ids = set()
        for text in ("<|endoftext|>", "<|eot_id|>", "</s>", "<eos>"):
            tid = self.tokenizer.token_to_id(text)
            if tid is not None:
                self.eos_ids.add(int(tid))
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.active_request_id: str | None = None
        self.loaded_s = time.time() - t0

    def _hidden_to_logp(self, h: torch.Tensor) -> torch.Tensor:
        # Dynamic quantized Linear requires a batch dimension.
        if h.ndim == 1:
            h = h.unsqueeze(0)
        if self.tied:
            z = F.linear(h.float(), self.tied_w)[0] * self.tied_scale + self.tied_bias
            return z.log_softmax(-1)
        # Exact RPV16 product-vocabulary probability factorisation, with INT8
        # serving-cache approximation only in the linear weights.
        raw = self.factor_head(h.float())[0].view(VOCAB_RANK, VOCAB_GROUP_PAD + VOCAB_LOW)
        gl = raw[:, :VOCAB_GROUP_PAD].float()
        gl[:, VOCAB_GROUPS:] = -1.0e9
        ll = raw[:, VOCAB_GROUP_PAD:].float()
        lg = gl.log_softmax(-1).gather(1, self.group_map)
        lo = ll.log_softmax(-1).gather(1, self.low_map)
        lm = F.linear(h.float(), self.mix_head)[0].view(VOCAB_RANK).log_softmax(-1)
        return torch.logsumexp(lm[:, None] + lg + lo, dim=0)

    @staticmethod
    def _apply_penalties(scores: torch.Tensor, history: list[int], body: dict[str, Any]) -> torch.Tensor:
        last_n = max(0, int(body.get("penalty_last_n", 128) or 0))
        hist = history[-last_n:] if last_n else []
        if not hist:
            return scores
        counts: dict[int, int] = {}
        for tid in hist:
            counts[tid] = counts.get(tid, 0) + 1
        rep = float(body.get("repetition_penalty", 1.0) or 1.0)
        presence = float(body.get("presence_penalty", 0.0) or 0.0)
        frequency = float(body.get("frequency_penalty", 0.0) or 0.0)
        for tid, count in counts.items():
            if rep != 1.0:
                scores[tid] = scores[tid] / rep if scores[tid] > 0 else scores[tid] * rep
            scores[tid] -= presence + frequency * count
        return scores

    @staticmethod
    def _block_repeated_ngrams(scores: torch.Tensor, history: list[int], body: dict[str, Any]) -> torch.Tensor:
        # RPV16 is still young and can fall into short deterministic phrase loops.
        # Block only exact repeated token n-grams; this is decode-time only and
        # leaves model probabilities/training untouched. Four is conservative
        # enough to preserve normal repeated words while breaking phrase cycles.
        n = max(0, min(8, int(body.get("no_repeat_ngram_size", 4) or 0)))
        if n < 2 or len(history) < n - 1:
            return scores
        prefix = history[-(n - 1):]
        banned: set[int] = set()
        stop = len(history) - n + 1
        for i in range(max(0, stop)):
            if history[i:i + n - 1] == prefix:
                banned.add(history[i + n - 1])
        if banned:
            idx = torch.tensor(sorted(banned), dtype=torch.long, device=scores.device)
            scores.index_fill_(0, idx, float("-inf"))
        return scores

    def _sample(self, logp: torch.Tensor, history: list[int], body: dict[str, Any]) -> tuple[int, list[dict[str, Any]]]:
        scores = self._apply_penalties(logp.clone(), history, body)
        scores = self._block_repeated_ngrams(scores, history, body)
        greedy = bool(body.get("greedy", False)) or float(body.get("temperature", 0.0) or 0.0) <= 0
        temperature = max(1e-5, float(body.get("temperature", 1.0) or 1.0))
        probs = (scores if greedy else scores / temperature).softmax(-1)
        top_prob, top_id = torch.topk(probs, k=20)
        top = []
        for p, tid in zip(top_prob.tolist(), top_id.tolist()):
            top.append({"token": self.tokenizer.decode([int(tid)], skip_special_tokens=False), "prob": float(p), "id": int(tid)})
        if greedy:
            return int(top_id[0]), top
        top_k = int(body.get("top_k", 0) or 0)
        min_p = float(body.get("min_p", 0.0) or 0.0)
        top_p = float(body.get("top_p", 1.0) or 1.0)
        work = probs.clone()
        if min_p > 0:
            work[work < work.max() * min_p] = 0
        if top_k > 0 and top_k < work.numel():
            kth = torch.topk(work, k=top_k).values[-1]
            work[work < kth] = 0
        if 0 < top_p < 1:
            vals, idx = torch.sort(work, descending=True)
            # Nucleus probability is relative to the candidates remaining
            # after min_p/top_k. Retain the first token crossing top_p.
            cum = vals.cumsum(0) / vals.sum().clamp_min(torch.finfo(vals.dtype).tiny)
            drop = cum > top_p
            drop[1:] = drop[:-1].clone()
            drop[0] = False
            work[idx[drop]] = 0
        total = work.sum()
        if not torch.isfinite(total) or total <= 0:
            return int(top_id[0]), top
        return int(torch.multinomial(work / total, 1).item()), top

    def prefill(self, ids: list[int]) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        t = torch.tensor(ids, dtype=torch.long)
        x = F.embedding(t, self.embed).float()
        x = self.in_proj(x)
        caches = []
        for stage in self.stages:
            x, cache = stage.prefill(x)
            caches.append(cache)
        h = self.out_proj(rms_norm(x[-1:], self.norm))
        return h, caches

    def step(self, token_id: int, caches: list[tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        t = torch.tensor([token_id], dtype=torch.long)
        x = self.in_proj(F.embedding(t, self.embed).float())
        for i, stage in enumerate(self.stages):
            x, caches[i] = stage.step(x, caches[i])
        return self.out_proj(rms_norm(x, self.norm))

    def generate(self, body: dict[str, Any], emit=None) -> dict[str, Any]:
        prompt = str(body.get("prompt") or "")
        enc = self.tokenizer.encode(prompt)
        prompt_ids = list(enc.ids)
        if not prompt_ids:
            prompt_ids = [1]
        truncated = len(prompt_ids) > MAX_PROMPT
        prompt_ids = prompt_ids[-MAX_PROMPT:]
        max_new = max(1, min(MAX_NEW, int(body.get("max_new", 32) or 32)))
        min_new = max(0, min(max_new, int(body.get("min_new", 0) or 0)))
        ignore_eos = bool(body.get("ignore_eos", False))
        generated: list[int] = []
        prev_decoded = ""
        t0 = time.perf_counter()
        h, caches = self.prefill(prompt_ids)
        prefill_s = time.perf_counter() - t0
        for i in range(max_new):
            if self.cancel.is_set():
                break
            logp = self._hidden_to_logp(h[0])
            token_id, top = self._sample(logp, prompt_ids + generated, body)
            if token_id in self.eos_ids and i >= min_new and not ignore_eos:
                break
            generated.append(token_id)
            decoded = self.tokenizer.decode(generated, skip_special_tokens=False)
            piece = decoded[len(prev_decoded):] if decoded.startswith(prev_decoded) else self.tokenizer.decode([token_id], skip_special_tokens=False)
            prev_decoded = decoded
            if emit is not None:
                emit(piece, i, top)
            # No next-token distribution is needed after the requested final token.
            if i + 1 < max_new and not self.cancel.is_set():
                h = self.step(token_id, caches)
        elapsed = max(1e-9, time.perf_counter() - t0)
        completion = self.tokenizer.decode(generated, skip_special_tokens=False)
        text = prompt + completion
        stats = {
            "tokens": len(generated), "elapsed_s": elapsed, "tok_s": len(generated) / elapsed,
            "prefill_s": prefill_s, "prompt_tokens": len(prompt_ids), "prompt_truncated": truncated,
            "model": "AGILLM-GB10-1PF RPV16", "device": "GETH CPU", "threads": THREADS,
            "quantization": self.meta.get("quantization"), "step": self.meta.get("step"),
            "seen_tokens": self.meta.get("seen_tokens"), "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA, "serving_snapshot_sha256": self.meta.get("source_sha256"),
        }
        return {"text": text, "completion": completion, "stats": stats}


ENGINE: RPV16CPU | None = None
STARTED = time.time()
app = FastAPI(title="AGILLM RPV16 1PF CPU inference", docs_url=None, redoc_url=None)


def engine() -> RPV16CPU:
    global ENGINE
    if ENGINE is None:
        ENGINE = RPV16CPU(ROOT, TOKENIZER_PATH)
    return ENGINE


@app.on_event("startup")
def startup() -> None:
    engine()


@app.get("/health")
def health() -> dict[str, Any]:
    e = engine()
    return {
        "ok": True, "ready": True, "host": socket.gethostname(), "device": "CPU", "dtype": "dynamic-int8",
        "threads": THREADS, "live_slots": 1, "max_prompt_tokens": MAX_PROMPT, "max_new_tokens": MAX_NEW,
        "ckpt": f"rpv16_1pf_step{e.meta.get('step', 0)}_cpu_int8.pt",
        "model": {
            "name": "AGILLM-GB10-1PF RPV16", "parameters": e.meta.get("parameter_count"),
            "step": e.meta.get("step"), "seen_tokens": e.meta.get("seen_tokens"),
            "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA, "serving_snapshot_sha256": e.meta.get("source_sha256"), "source_schema": e.meta.get("source_schema"),
            "target_alignment": e.meta.get("target_alignment"), "loaded_s": e.loaded_s,
            "classifier": ("tied_full_vocab" if getattr(e, "tied", False) else "rpv16_rank16_product"),
        },
        "canonical_modes": ["ar"],
        "capabilities": {"ar": {"trained": True, "backend_configured": True, "worker_ready": True, "available": True,
            "reason": "", "semantics": {"runtime_mode": "ar", "runtime_sampler": "ar", "commit_policy": "exactly_1_token_per_decode_step"}}},
        "uptime_s": time.time() - STARTED,
    }


@app.post("/infer")
async def infer(request: Request):
    body = await request.json()
    e = engine()
    if str(body.get("mode") or "ar").lower() != "ar":
        return JSONResponse({"ok": False, "error": "RPV16 GETH CPU lane is AR-only"}, status_code=400)
    acquired = e.lock.acquire(blocking=False)
    if not acquired:
        return JSONResponse({"ok": False, "error": "RPV16 CPU worker busy"}, status_code=429)
    req_id = str(body.get("request_id") or uuid.uuid4())
    e.active_request_id = req_id
    e.cancel.clear()
    try:
        result = await asyncio.to_thread(e.generate, body)
        return {"ok": True, **result, "ckpt": f"rpv16_1pf_step{e.meta.get('step', 0)}_cpu_int8.pt",
                "host": socket.gethostname(), "slot": 0, "threads": THREADS, "queue_wait_s": 0.0}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, status_code=500)
    finally:
        e.active_request_id = None
        e.lock.release()


@app.post("/stream")
async def stream(request: Request):
    body = await request.json()
    e = engine()
    if str(body.get("mode") or "ar").lower() != "ar":
        return JSONResponse({"ok": False, "error": "RPV16 GETH CPU lane is AR-only"}, status_code=400)
    req_id = str(body.get("request_id") or uuid.uuid4())

    def iterator():
        acquired = e.lock.acquire(blocking=False)
        if not acquired:
            yield json.dumps({"event":"error", "error":"RPV16 CPU worker busy"}) + "\n"
            return
        e.active_request_id = req_id
        e.cancel.clear()
        try:
            yield json.dumps({"event":"start", "host":socket.gethostname(), "slot":0, "threads":THREADS,
                              "queue_wait_s":0.0, "ckpt":f"rpv16_1pf_step{e.meta.get('step', 0)}_cpu_int8.pt"}) + "\n"
            yield json.dumps({"event":"stream_begin", "data":{"mode":"ar", "sampler":"ar", "runtime":"rpv16_cpu_int8", "step":e.meta.get("step"), "source_checkpoint_sha256":SOURCE_CHECKPOINT_SHA}}) + "\n"
            def emit(piece, i, top):
                events.append(json.dumps({"event":"stream_tok", "data":{"text":piece, "i":i, "top":top}}) + "\n")
            # A sync StreamingResponse iterator runs in Starlette's worker pool.
            # `events` is flushed after each model token through the generator below.
            prompt = str(body.get("prompt") or "")
            enc = e.tokenizer.encode(prompt)
            prompt_ids = list(enc.ids) or [1]
            truncated = len(prompt_ids) > MAX_PROMPT
            prompt_ids = prompt_ids[-MAX_PROMPT:]
            max_new = max(1, min(MAX_NEW, int(body.get("max_new", 32) or 32)))
            min_new = max(0, min(max_new, int(body.get("min_new", 0) or 0)))
            ignore_eos = bool(body.get("ignore_eos", False))
            generated = []
            prev_decoded = ""
            started = time.perf_counter()
            h, caches = e.prefill(prompt_ids)
            prefill_s = time.perf_counter() - started
            for i in range(max_new):
                if e.cancel.is_set():
                    break
                logp = e._hidden_to_logp(h[0])
                token_id, top = e._sample(logp, prompt_ids + generated, body)
                if token_id in e.eos_ids and i >= min_new and not ignore_eos:
                    break
                generated.append(token_id)
                decoded = e.tokenizer.decode(generated, skip_special_tokens=False)
                piece = decoded[len(prev_decoded):] if decoded.startswith(prev_decoded) else e.tokenizer.decode([token_id], skip_special_tokens=False)
                prev_decoded = decoded
                yield json.dumps({"event":"stream_tok", "data":{"text":piece, "i":i, "top":top}}) + "\n"
                # Avoid a discarded full-stack forward after the final emission.
                if i + 1 < max_new and not e.cancel.is_set():
                    h = e.step(token_id, caches)
            elapsed = max(1e-9, time.perf_counter() - started)
            completion = e.tokenizer.decode(generated, skip_special_tokens=False)
            stats = {"tokens":len(generated), "elapsed_s":elapsed, "tok_s":len(generated)/elapsed,
                     "prefill_s":prefill_s, "prompt_tokens":len(prompt_ids), "prompt_truncated":truncated,
                     "model":"AGILLM-GB10-1PF RPV16", "device":"GETH CPU", "threads":THREADS,
                     "step":e.meta.get("step"), "seen_tokens":e.meta.get("seen_tokens"), "source_checkpoint_sha256":SOURCE_CHECKPOINT_SHA, "serving_snapshot_sha256":e.meta.get("source_sha256")}
            yield json.dumps({"event":"stats", "stats":stats}) + "\n"
            yield json.dumps({"event":"done", "text":prompt + completion, "stats":stats,
                              "host":socket.gethostname(), "slot":0, "threads":THREADS, "queue_wait_s":0.0,
                              "ckpt":f"rpv16_1pf_step{e.meta.get('step', 0)}_cpu_int8.pt"}) + "\n"
        except Exception as exc:
            yield json.dumps({"event":"error", "error":f"{type(exc).__name__}: {exc}"}) + "\n"
        finally:
            e.active_request_id = None
            e.lock.release()

    return StreamingResponse(iterator(), media_type="application/x-ndjson", headers={"Cache-Control":"no-store"})


@app.post("/cancel")
def cancel() -> dict[str, Any]:
    e = engine()
    e.cancel.set()
    return {"ok": True, "request_id": e.active_request_id}



import rpv16_multimode_v1
rpv16_multimode_v1.install(globals())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("RPV16_CPU_PORT", "18620")), log_level="info")
