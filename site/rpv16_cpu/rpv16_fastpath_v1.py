"""RPV16 GETH CPU decode fast path (fable-cowork 2026-10-03). Same model, same decoding semantics.

Installed AFTER rpv16_multimode_v1 by rpv16_cpu_server_fast.py. What changes:
  * AR: per-stage KV cache is a preallocated buffer; grouped-query attention by head-group broadcasting
    (the old path re-concatenated the whole cache and repeat_interleave'd K/V for every token, every stage).
  * SAT fixed / SAT variable: KV-cached. The old path re-ran the full prompt+output through all 16 stages for
    every 2-token step (and twice for a stride-1 step). Now committed tokens live in a causal cache and each
    step runs only the 2-token "pair" block (exact pair mask) plus a <=2-token causal cache extension.
    Committed positions are causal in both paths, so the cache holds exactly what the recompute produced.
  * NAT: the prompt is causal and never sees the masked suffix, so its K/V are computed once per request
    and every refinement pass runs only the suffix block against them.
  * Sampling: vectorised repetition/presence/frequency penalties; nucleus sort restricted to the tokens that
    survive min_p/top_k (identical kept set); single-token decode strings cached.
Set RPV16_FASTPATH=0 to disable everything at start-up (the wrapper then serves the original paths).
"""
from __future__ import annotations

import math
import os
import time
import types

import torch
import torch.nn.functional as F

REVISION = "rpv16-fastpath-v1-20261003"


class KV:
    """Causal K/V for absolute positions [0, n) of one stage. k/v: [KH, cap, HD]."""
    __slots__ = ("k", "v", "n")

    def __init__(self, k, v, n):
        self.k, self.v, self.n = k, v, n

    def ensure(self, need):
        cap = self.k.shape[1]
        if need <= cap:
            return
        new_cap = max(need, cap * 2)
        k = torch.empty(self.k.shape[0], new_cap, self.k.shape[2]); v = torch.empty_like(k)
        k[:, :self.n] = self.k[:, :self.n]; v[:, :self.n] = self.v[:, :self.n]
        self.k, self.v = k, v


def install(ns):
    if os.environ.get("RPV16_FASTPATH", "1") == "0":
        ns["FASTPATH_REVISION"] = "disabled"
        return
    Stage = ns["StageCPU"]
    Engine = ns["RPV16CPU"]                     # MultimodeCPU (multimode already installed)
    D, QH, KH, HD = ns["D"], ns["Q_HEADS"], ns["KV_HEADS"], ns["HEAD_DIM"]
    G = QH // KH
    rms = ns["rms_norm"]
    SQ = math.sqrt(HD)
    CAP_EXTRA = int(ns["MAX_NEW"]) + 8
    legacy_prefill = Stage.prefill
    legacy_step = Stage.step

    # ------------------------------------------------------------------ attention core
    def attend(self, h_block, cache, c, extra_mask, write):
        """Block of B new rows (absolute positions c..c+B-1) attending to cache[0:c] plus the block itself.
        write=True: the block's K/V are stored at [c, c+B) first and attention reads a zero-copy view.
        write=False (SAT pair scratch): the cache is never touched; K/V are concatenated."""
        B = h_block.shape[0]
        q = self.q(h_block).view(B, KH, G, HD).permute(1, 2, 0, 3)          # [KH, G, B, HD]
        kn = self.k(h_block).view(B, KH, HD).permute(1, 0, 2)               # [KH, B, HD]
        vn = self.v(h_block).view(B, KH, HD).permute(1, 0, 2)
        w = self.window
        lo = 0 if w < 0 else max(0, c - w)                                   # earliest cache row any query may see
        if write:
            cache.ensure(c + B)
            cache.k[:, c:c + B] = kn; cache.v[:, c:c + B] = vn
            cache.n = c + B
            K = cache.k[:, lo:c + B]; V = cache.v[:, lo:c + B]
        else:
            K = torch.cat((cache.k[:, lo:c], kn), 1) if c > lo else kn
            V = torch.cat((cache.v[:, lo:c], vn), 1) if c > lo else vn
        L = K.shape[1]
        if B == 1 and (w < 0 or L <= w + 1):
            vis = None                                                        # single causal row: everything visible
        else:
            pos_q = torch.arange(c, c + B)
            pos_k = torch.arange(lo, c + B)
            vis = torch.ones(B, L, dtype=torch.bool)
            vis[:, L - B:] = extra_mask(B)
            if w >= 0:
                vis &= pos_k[None, :] >= (pos_q[:, None] - w)
        scores = torch.matmul(q, K.transpose(-1, -2).unsqueeze(1)) / SQ      # [KH, G, B, L]
        if vis is not None:
            scores = scores.masked_fill(~vis, float("-inf"))
        a = torch.matmul(scores.softmax(-1), V.unsqueeze(1))                # [KH, G, B, HD]
        return a.permute(2, 0, 1, 3).reshape(B, D)

    causal_block = lambda B: torch.ones(B, B, dtype=torch.bool).tril()

    def pair_block(B):
        m = torch.ones(B, B, dtype=torch.bool).tril()
        m[B - 2, B - 1] = True                                              # first slot of the pair sees the second
        return m

    def block_forward(self, x, cache, c, mask_fn, write):
        h = rms(x, self.n1)
        a = attend(self, h, cache, c, mask_fn, write)
        x = x.float() + self.o(a)
        return x + self._ffn(rms(x, self.n2))

    # ------------------------------------------------------------------ AR prefill / step
    def prefill(self, x):
        x2, (k0, v0) = legacy_prefill(self, x)
        s = k0.shape[1]
        kb = torch.empty(KH, s + CAP_EXTRA, HD); vb = torch.empty_like(kb)
        kb[:, :s] = k0; vb[:, :s] = v0
        return x2, KV(kb, vb, s)

    def step(self, x, cache):
        if not isinstance(cache, KV):
            return legacy_step(self, x, cache)
        return block_forward(self, x, cache, cache.n, causal_block, True), cache

    Stage.prefill = prefill
    Stage.step = step

    # ------------------------------------------------------------------ cached SAT / NAT helpers on the engine
    def embed_rows(self, ids):
        t = torch.tensor(ids, dtype=torch.long)
        return self.in_proj(F.embedding(t, self.embed).float())

    @torch.inference_mode()
    def causal_extend(self, caches, ids, c0):
        """Append causal K/V for absolute positions c0..len(ids)-1; returns final hidden of those rows."""
        x = embed_rows(self, ids[c0:])
        for st, cache in zip(self.stages, caches):
            x = block_forward(st, x, cache, c0, causal_block, True)
        return self.out_proj(rms(x, self.norm))

    @torch.inference_mode()
    def pair_hidden(self, caches, ids):
        """Hidden of the last two positions under the SAT pair mask, cache covering [0, n-2)."""
        n = len(ids)
        x = embed_rows(self, ids[n - 2:])
        for st, cache in zip(self.stages, caches):
            x = block_forward(st, x, cache, n - 2, pair_block, False)
        return self.out_proj(rms(x, self.norm))

    def new_caches(self, cap):
        return [KV(torch.empty(KH, cap, HD), torch.empty(KH, cap, HD), 0) for _ in self.stages]

    Engine.causal_extend = causal_extend
    Engine.pair_hidden = pair_hidden
    Engine.new_caches = new_caches

    @torch.inference_mode()
    def sat_forward(self, caches, context):
        """ONE block forward per SAT step: causal rows for positions [c, n) (written to the cache) plus the two
        scratch pair rows for positions n-2, n-1 under the SAT pair mask. Returns (pair_h [2, d] or None,
        causal_last_h [d]). Equivalent to the original all_hidden(context, first2) + all_hidden(context)."""
        n = len(context)
        c = caches[0].n
        assert c < n, (c, n)
        want_pair = n >= 2
        xc = embed_rows(self, context[c:])
        xp = embed_rows(self, context[n - 2:]) if want_pair else None
        Bc = xc.shape[0]
        x = torch.cat((xc, xp), 0) if want_pair else xc
        B = x.shape[0]
        for st, cache in zip(self.stages, caches):
            h = rms(x, st.n1)
            q = st.q(h).view(B, KH, G, HD).permute(1, 2, 0, 3)
            k = st.k(h).view(B, KH, HD).permute(1, 0, 2)
            v = st.v(h).view(B, KH, HD).permute(1, 0, 2)
            cache.ensure(n)
            cache.k[:, c:n] = k[:, :Bc]; cache.v[:, c:n] = v[:, :Bc]; cache.n = n
            w = st.window
            first = min(c, n - 2) if want_pair else c
            lo = 0 if w < 0 else max(0, first - w)
            if want_pair:
                K = torch.cat((cache.k[:, lo:n], k[:, Bc:]), 1); V = torch.cat((cache.v[:, lo:n], v[:, Bc:]), 1)
                pos_q = torch.cat((torch.arange(c, n), torch.arange(n - 2, n)))
                pos_k = torch.cat((torch.arange(lo, n), torch.arange(n - 2, n)))
                L = K.shape[1]
                vis = torch.zeros(B, L, dtype=torch.bool)
                cache_cols = n - lo
                vis[:Bc, :cache_cols] = pos_k[None, :cache_cols] <= pos_q[:Bc, None]          # causal rows
                vis[Bc:, :cache_cols] = pos_k[None, :cache_cols] < (n - 2)                     # pair rows: strictly before the pair
                vis[Bc:, cache_cols:] = True                                                   # pair rows see both pair slots
            else:
                K = cache.k[:, lo:n]; V = cache.v[:, lo:n]
                pos_q = torch.arange(c, n); pos_k = torch.arange(lo, n)
                vis = pos_k[None, :] <= pos_q[:, None]
            if w >= 0:
                vis = vis & (pos_k[None, :] >= (pos_q[:, None] - w))
            scores = torch.matmul(q, K.transpose(-1, -2).unsqueeze(1)) / SQ
            scores = scores.masked_fill(~vis, float("-inf"))
            a = torch.matmul(scores.softmax(-1), V.unsqueeze(1)).permute(2, 0, 1, 3).reshape(B, D)
            x = x.float() + st.o(a)
            x = x + st._ffn(rms(x, st.n2))
        rows = x[Bc - 1:] if want_pair else x[Bc - 1:Bc]
        out = self.out_proj(rms(rows, self.norm))
        return (out[1:] if want_pair else None), out[0]

    Engine.sat_forward = sat_forward

    # ------------------------------------------------------------------ sampling
    tiny = torch.finfo(torch.float32).tiny

    def fast_penalties(scores, history, body):
        last_n = max(0, int(body.get("penalty_last_n", 128) or 0))
        hist = history[-last_n:] if last_n else []
        if not hist:
            return scores
        rep = float(body.get("repetition_penalty", 1.0) or 1.0)
        presence = float(body.get("presence_penalty", 0.0) or 0.0)
        frequency = float(body.get("frequency_penalty", 0.0) or 0.0)
        ids, counts = torch.unique(torch.tensor(hist, dtype=torch.long), return_counts=True)
        vals = scores.index_select(0, ids)
        if rep != 1.0:
            vals = torch.where(vals > 0, vals / rep, vals * rep)
        vals = vals - (presence + frequency * counts.to(vals.dtype))
        scores.index_copy_(0, ids, vals)
        return scores

    piece_cache = {}

    def piece(self, tid):
        s = piece_cache.get(tid)
        if s is None:
            s = self.tokenizer.decode([int(tid)], skip_special_tokens=False)
            if len(piece_cache) < 200000:
                piece_cache[tid] = s
        return s

    def fast_sample(self, logp, history, body):
        scores = fast_penalties(logp.clone(), history, body)
        scores = self._block_repeated_ngrams(scores, history, body)
        greedy = bool(body.get("greedy", False)) or float(body.get("temperature", 0.0) or 0.0) <= 0
        temperature = max(1e-5, float(body.get("temperature", 1.0) or 1.0))
        probs = (scores if greedy else scores / temperature).softmax(-1)
        top_prob, top_id = torch.topk(probs, k=20)
        top = [{"token": piece(self, t), "prob": float(p), "id": int(t)} for p, t in zip(top_prob.tolist(), top_id.tolist())]
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
            cand = work.nonzero().flatten()
            if cand.numel() > 0:
                vals, order = torch.sort(work.index_select(0, cand), descending=True)
                idx = cand.index_select(0, order)
                cum = vals.cumsum(0) / vals.sum().clamp_min(tiny)
                drop = cum > top_p
                drop[1:] = drop[:-1].clone()
                drop[0] = False
                work[idx[drop]] = 0
        total = work.sum()
        if not torch.isfinite(total) or total <= 0:
            return int(top_id[0]), top
        return int(torch.multinomial(work / total, 1).item()), top

    Engine._sample = fast_sample

    # ------------------------------------------------------------------ cached SAT + NAT event paths
    base_events = Engine.events

    def events(self, body, request_id, entered):
        mode = ns["mode_key_fast"](body)
        if mode == "ar":
            yield from base_events(self, body, request_id, entered)
            return
        if os.environ.get("RPV16_FASTPATH_MULTIMODE", "1") == "0":
            yield from base_events(self, body, request_id, entered)
            return
        yield from fast_multimode_events(self, body, request_id, entered, mode)

    import rpv16_multimode_v1 as mm
    ns["mode_key_fast"] = mm.mode_key

    def fast_multimode_events(self, body, request_id, entered, mode):
        import socket
        if mode not in self.available_modes():
            raise ValueError("RPV16 mode unavailable for this checkpoint: " + mode)
        prompt = str(body.get("prompt") or "")
        original_ids = list(self.tokenizer.encode(prompt).ids) or [1]
        ids = original_ids[-ns["MAX_PROMPT"]:]
        limit = min(ns["MAX_NEW"], 256) if mode == "nat" else ns["MAX_NEW"]
        requested = int(body.get("max_new", body.get("max_new_tokens", 32)) or 32)
        budget = max(1, min(limit, requested))
        minimum = max(0, min(budget, int(body.get("min_new", 0) or 0)))
        ignore_eos = bool(body.get("ignore_eos", False))
        ckpt = f"rpv16_1pf_step{self.meta.get('step', 0)}_cpu_int8.pt"
        yield {"event": "start", "host": socket.gethostname(), "slot": 0, "threads": ns["THREADS"],
               "queue_wait_s": 0.0, "ckpt": ckpt, "request_id": request_id}
        yield {"event": "stream_begin", "data": {"mode": ("sat" if mode.startswith("sat_") else mode),
               "var": mode == "sat_variable", "canonical_mode": mode, "sampler": ("sat" if mode.startswith("sat_") else mode),
               "runtime": mm.REVISION + "+" + REVISION, "step": self.meta.get("step"),
               "source_checkpoint_sha256": ns["SOURCE_CHECKPOINT_SHA"]}}
        start = time.perf_counter(); generated = []; previous = ""; forwards = 0; prefill_s = 0.0
        strides = {"1": 0, "2": 0}; policy_strides = {"1": 0, "2": 0}; passes_done = 0; first_token_s = None

        def token_event(token, top):
            nonlocal previous, first_token_s
            generated.append(int(token))
            decoded = self.tokenizer.decode(generated, skip_special_tokens=False)
            pc = decoded[len(previous):] if decoded.startswith(previous) else self.tokenizer.decode([int(token)], skip_special_tokens=False)
            previous = decoded
            if first_token_s is None:
                first_token_s = time.perf_counter() - entered
            return {"event": "stream_tok", "data": {"text": pc, "i": len(generated) - 1, "top": top}}

        caches = self.new_caches(len(ids) + budget + 8)
        if mode.startswith("sat_"):
            shift = self.aux["shift_emb"]
            ended = False
            while len(generated) < budget and not ended and not self.cancel.is_set():
                context = ids + generated
                n = len(context)
                use_pair = n >= 2 and budget - len(generated) >= 2
                if not use_pair and caches[0].n >= n:
                    caches_n_reset = n - 1
                    for cc in caches:
                        cc.n = min(cc.n, caches_n_reset)
                pair_h, last_h = self.sat_forward(caches, context) if use_pair else (None, self.causal_extend(caches, context, min(caches[0].n, n - 1))[-1])
                forwards += 1
                stride = 1
                if use_pair:
                    if mode == "sat_fixed":
                        stride = 2
                    else:
                        stride, gap = self.policy(pair_h[-1]); policy_strides[str(stride)] += 1
                if stride == 2:
                    hidden = (pair_h[-2] + shift[0], pair_h[-1] + shift[1])
                else:
                    hidden = (last_h + shift[0],)
                if not prefill_s:
                    prefill_s = time.perf_counter() - start
                committed = 0
                for hh in hidden:
                    token, top = self.sample(hh, ids + generated, body)
                    if token in self.eos_ids and len(generated) >= minimum and not ignore_eos:
                        ended = True; break
                    if self.cancel.is_set():
                        ended = True; break
                    yield token_event(token, top); committed += 1
                if committed:
                    strides[str(committed)] += 1
        else:
            passes = max(1, min(8, int(body.get("nat_passes", 4) or 1)))
            masked = set(range(budget)); output = [2] * budget; confidence = [0.0] * budget; tops = [[] for _ in range(budget)]
            local_body = dict(body)
            if body.get("nat_greedy", True):
                local_body.update(greedy=True, temperature=0.0)
            # prompt K/V once (causal; the suffix never changes them)
            self.causal_extend(caches, ids, 0); forwards += 1
            P = len(ids)
            for p in range(passes):
                if self.cancel.is_set():
                    break
                suffix = [2 if i in masked else output[i] for i in range(budget)]
                x = embed_rows(self, suffix)
                for st, cache in zip(self.stages, caches):
                    x = block_forward(st, x, cache, P, causal_block, False)
                hidden = self.out_proj(rms(x, self.norm)); forwards += 1
                if not prefill_s:
                    prefill_s = time.perf_counter() - start
                positions = sorted(masked)
                head_batch = 32 if getattr(self, "tied", False) else 1
                for offset in range(0, len(positions), head_batch):
                    chunk = positions[offset:offset + head_batch]
                    scores_batch = None
                    if getattr(self, "tied", False):
                        selected = hidden[chunk].float()
                        scores_batch = (F.linear(selected, self.tied_w) * self.tied_scale + self.tied_bias).log_softmax(-1)
                    for row, i in enumerate(chunk):
                        if scores_batch is None:
                            token, top = self.sample(hidden[i], ids + output[:i], local_body, forbid_mask=True)
                        else:
                            scores = scores_batch[row].clone()
                            scores[2] = float("-inf")
                            token, top = self._sample(scores, ids + output[:i], local_body)
                        output[i] = token; tops[i] = top
                        confidence[i] = next((entry["prob"] for entry in top if entry["id"] == token), 0.0)
                passes_done += 1
                remaining = math.ceil(budget * (1 - (p + 1) / passes))
                masked = set(sorted(range(budget), key=lambda i: (confidence[i], i))[:remaining])
                yield {"event": "progress", "data": {"canonical_mode": "nat", "pass": passes_done, "passes": passes,
                                                    "masked_remaining": len(masked), "emitted_tokens": 0}}
            if not self.cancel.is_set():
                for i, token in enumerate(output):
                    if token in self.eos_ids and i >= minimum and not ignore_eos:
                        break
                    yield token_event(token, tops[i])
        elapsed = max(1e-9, time.perf_counter() - start)
        wall = max(1e-9, time.perf_counter() - entered)
        completion = self.tokenizer.decode(generated, skip_special_tokens=False)
        stats = {"tokens": len(generated), "elapsed_s": elapsed, "tok_s": len(generated) / elapsed,
                 "request_to_done_s": wall, "request_to_done_tok_s": len(generated) / wall,
                 "prefill_s": prefill_s, "time_to_first_token_s": first_token_s,
                 "prompt_tokens": len(ids), "prompt_truncated": len(original_ids) > len(ids),
                 "original_prompt_tokens": len(original_ids), "max_prompt_tokens": ns["MAX_PROMPT"],
                 "requested_tokens": requested, "max_new_tokens_effective": budget,
                 "mode": ("sat" if mode.startswith("sat_") else mode), "canonical_mode": mode,
                 "var": mode == "sat_variable", "model_forwards": forwards, "emitted_stride_hist": strides,
                 "policy_stride_hist": policy_strides, "nat_passes": passes_done,
                 "policy_basis": ("saved_learned_ce_regret_plus_speed_reward_minus_merge_cost" if mode == "sat_variable" else None),
                 "routing": "single_sequence_token_local_argmax", "runtime_revision": mm.REVISION + "+" + REVISION,
                 "kv_cache": True,
                 "model": "AGILLM-GB10-1PF RPV16", "device": "GETH CPU", "threads": ns["THREADS"],
                 "quantization": self.meta.get("quantization"), "step": self.meta.get("step"),
                 "seen_tokens": self.meta.get("seen_tokens"), "source_checkpoint_sha256": ns["SOURCE_CHECKPOINT_SHA"],
                 "serving_snapshot_sha256": self.meta.get("source_sha256"), "cancelled": self.cancel.is_set()}
        yield {"event": "stats", "stats": stats}
        yield {"event": "done", "ok": True, "text": prompt + completion, "completion": completion,
               "stats": stats, "host": socket.gethostname(), "slot": 0, "threads": ns["THREADS"],
               "queue_wait_s": 0.0, "ckpt": ckpt, "request_id": request_id}

    Engine.events = events
    ns["FASTPATH_REVISION"] = REVISION
