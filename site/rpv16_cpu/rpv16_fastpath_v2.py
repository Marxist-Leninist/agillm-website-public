"""RPV16 GETH CPU decode fast path v2 (fable-cowork 2026-10-03). Same model, same decoding semantics, same numbers.

Installed AFTER rpv16_multimode_v1 by rpv16_cpu_server_fast.py. Everything v1 did is kept (and stays the fallback):
  * AR: preallocated per-stage KV cache; SAT fixed / variable and NAT KV-cached; vectorised penalties.
v2 adds, without changing a single bit of the hidden states and logits:
  * int8 kernels (fastpath_v2/src/rpv16_kernels_v2.cpp) in place of the 354 stock dynamic-quantised Linears. The
    integer sums are order free and the quantisation / rescale arithmetic is fbgemm's, so outputs are bitwise identical;
    q/k/v and gate/up share one quantisation, the routed FFN is one call, and the second FFN Linear reads only the
    columns whose quantised input is non-zero (~16% on the trained model).
  * One C++ call per decode step for all 16 stages, on one thread team. Its float operators are either the same
    libtorch operators or "native" pieces: RMS norm, softmax and the SiLU gate in torch's own arithmetic order (with
    torch's own vector exp), the two attention products and the router by the very BLAS entry points torch.matmul /
    F.linear call (cblas_sgemm_batch, sgemm_) straight on the caches (a transposed K cache makes that copy-free).
    Every native piece is switched on only after it was compared bitwise with the libtorch operator in this process.
  * Sampler: top-20 in one AVX-512 pass (falls back to torch.topk when leaders tie); the token is drawn by inverse
    CDF from the kept candidates only - same kept set, same weights, same distribution as torch.multinomial over the
    whole vocabulary (the random stream differs; requests are not seeded).
Every worker start proves the switch: each Linear against its stock module on probe inputs, then a teacher-forced
walk through AR / SAT / NAT with the stock modules against the kernels - hidden states, caches and log-probs must be
bitwise equal - and writes a receipt. Any failure leaves the v1 path with the stock modules in place.
Switches: RPV16_FASTPATH=0 everything off; RPV16_FASTPATH_INT8=0 v1 behaviour (stock Linears);
RPV16_FASTPATH_NATIVE=0 kernels with libtorch float operators; RPV16_FASTPATH_SAMPLER=0 v1 sampler;
RPV16_FASTPATH_SESSION_ROWS (default 4; 0 = one thread session per Linear); RPV16_FASTPATH_KEEP_STOCK=1 keep the
stock modules in memory after the switch.

v2.1 (same day), again without changing a number:
  * Attention of a decode step: one head per claim on the thread team (the same BLAS call per head) instead of one
    sequential batch call - enabled per block class (1-4 rows / bigger) only after a bitwise comparison.
  * Output head: the logits come straight from the bf16 embedding table (66 MB instead of the 132 MB fp32 copy),
    every row summed in the order of the sgemm call behind F.linear(h[1, 256], table.float()) - an order read off MKL
    with cancellation probes and compared bitwise on the real table (random and real hidden states) at every start.
  * Decode loop: AR runs in this module's loop too; the text piece of a token comes from a frozen decoded prefix plus
    the decoded tail (split points verified when taken) instead of decoding the whole output after every token, and
    the repeated-n-gram filter reads an index kept up to date per token instead of scanning the history.
  * The presence / frequency penalty constant is computed in double precision exactly as the original does.
Switches: RPV16_FASTPATH_ATTN_SPLIT=0, RPV16_FASTPATH_HEAD=0, RPV16_FASTPATH_LOOP=0.

v2.2: the two attention products (scores = q . K^T, output = probabilities . V) are computed by the kernel itself
in the summation orders of the cblas_sgemm_batch call behind torch.matmul - a single row and a block of two or more
rows are summed differently by MKL; both orders were read off with cancellation probes - with the heads spread over
the thread team. Enabled per block class only after a bitwise comparison with torch.matmul at load: one row at EVERY
key count the service can reach, 2-4 rows (SAT steps) at every key count. A class that fails keeps the BLAS call.
Bigger blocks (prefill, NAT passes) keep the BLAS call by default: the native order is bitwise identical there too
(checked on a grid of shapes when asked for) but not faster on this machine.
Switch: RPV16_FASTPATH_ATTN_NATIVE = 4 (default) | all | 1 | 0. The per-head BLAS split of v2.1 is superseded and
off (RPV16_FASTPATH_ATTN_SPLIT=1 brings its load-time check back).
"""
from __future__ import annotations

import ctypes
import gc
import hashlib
import json
import math
import os
import time

import torch
import torch.nn.functional as F

REVISION = "rpv16-fastpath-v2.2-20261003"
HERE = os.path.dirname(os.path.abspath(__file__))
KERNEL_SRC = os.path.join(HERE, "fastpath_v2", "src", "rpv16_kernels_v2.cpp")
KERNEL_SHA256 = "10d2147d5fe218b4d7fe5952b61942e68f877ce7269d0b0c99c861ee8512358c"
BUILD_ROOT = os.environ.get("RPV16_FASTPATH_BUILD", os.path.join(HERE, "fastpath_v2", "build"))
RECEIPTS = os.environ.get("RPV16_FASTPATH_RECEIPTS", "/opt/agillm43_netserve/receipts/rpv16_fastpath_v2")
CFLAGS = ["-O3", "-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512dq", "-mfma", "-fopenmp", "-ffp-contract=off"]
NATIVE_OFF = (False, False, False, False, False, 16, 16, True, 0, 16)   # set_native: every float piece by libtorch
STATE = {"revision": REVISION, "int8_kernels": False, "native": None, "topk": False, "sampler": False, "error": None,
         "receipt": None, "K": None, "summary": None, "head": False, "pieces": False, "loop": True, "attention_split": None, "attention_native": None}
PROBE_TEXT = ("The steam engine was developed over many decades. Early devices pumped water out of mines, and later "
              "designs turned wheels in mills and factories. Railways followed, and with them towns grew along the lines; "
              "goods that once took weeks by canal arrived in a day. By the end of the century the same principle drove "
              "ships across the oceans, and engineers were already asking what would replace coal.")


def load_kernel():
    src = open(KERNEL_SRC, encoding="utf-8").read()
    sha = hashlib.sha256(src.encode()).hexdigest()
    if sha != KERNEL_SHA256:
        raise RuntimeError("kernel source hash %s differs from the reviewed one" % sha)
    tag = hashlib.sha256("\n".join([src, " ".join(CFLAGS), torch.__version__]).encode()).hexdigest()[:16]
    build = os.path.join(BUILD_ROOT, "build_" + tag)
    os.makedirs(build, exist_ok=True)
    from torch.utils.cpp_extension import load_inline
    t0 = time.time()
    K = load_inline(name="rpv16_kernels_" + tag, cpp_sources=[src], functions=None, extra_cflags=CFLAGS,
                    extra_ldflags=["-fopenmp"], build_directory=build, verbose=False)
    return K, {"sha256": sha, "tag": tag, "build_dir": build, "load_s": round(time.time() - t0, 2)}


def rss_mb():
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return None


def trim():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def adversarial(m, k, gen):
    """rows whose scaled values sit on / next to the rounding boundaries of the 7-bit activation quantiser"""
    x = torch.randn(m, k, generator=gen) * float(torch.empty(1).uniform_(0.05, 30, generator=gen))
    mn, mx = float(x.min()), float(x.max())
    scale = (max(mx, 0.0) - min(mn, 0.0)) / 127.0
    half = (torch.randint(-120, 120, (m, k), generator=gen).float() + 0.5) * scale
    pick = torch.rand(m, k, generator=gen) < 0.5
    y = torch.where(pick, half * (1 + (torch.randint(-2, 3, (m, k), generator=gen).float() * 1.2e-7)), x)
    y[0, 0] = mn
    y[-1, -1] = mx
    return y.clamp(mn, mx).contiguous()


def gated(m, k, gen):
    """like the FFN hidden: mostly tiny values, a few large ones"""
    x = F.silu(torch.randn(m, k, generator=gen) * 2) * torch.randn(m, k, generator=gen)
    big = torch.rand(m, k, generator=gen) < 0.01
    return torch.where(big, x * 40, x).contiguous()


class KV:
    """Causal K/V for absolute positions [0, n) of one stage. k/v: [KH, cap, HD]; kt: K transposed, [KH, HD, cap]
    (only with the kernel model, which keeps it in step with k)."""
    __slots__ = ("k", "v", "n", "kt")

    def __init__(self, k, v, n, kt=None):
        self.k, self.v, self.n, self.kt = k, v, n, kt

    def ensure(self, need):
        cap = self.k.shape[1]
        if need <= cap:
            return
        new_cap = max(need, cap * 2)
        k = torch.empty(self.k.shape[0], new_cap, self.k.shape[2]); v = torch.empty_like(k)
        k[:, :self.n] = self.k[:, :self.n]; v[:, :self.n] = self.v[:, :self.n]
        if self.kt is not None:
            kt = torch.empty(self.k.shape[0], self.k.shape[2], new_cap)
            kt[:, :, :self.n] = self.kt[:, :, :self.n]
            self.kt = kt
        self.k, self.v = k, v


class QLin:
    """a kernel Linear with the call interface of the stock module"""
    __slots__ = ("k", "in_features", "out_features")

    def __init__(self, k):
        self.k = k
        self.in_features = int(k.in_features())
        self.out_features = int(k.out_features())

    def __call__(self, x):
        y = self.k.call(x)
        return y if x.dim() == 2 else y.view(*x.shape[:-1], self.out_features)


def install(ns):
    if os.environ.get("RPV16_FASTPATH", "1") == "0":
        ns["FASTPATH_REVISION"] = "disabled"
        return
    import rpv16_multimode_v1 as mm
    Stage = ns["StageCPU"]
    Engine = ns["RPV16CPU"]                     # MultimodeCPU (multimode already installed)
    D, QH, KH, HD = ns["D"], ns["Q_HEADS"], ns["KV_HEADS"], ns["HEAD_DIM"]
    G = QH // KH
    rms = ns["rms_norm"]
    SQ = math.sqrt(HD)
    CAP_EXTRA = int(ns["MAX_NEW"]) + 8
    MAX_KEYS = int(ns["MAX_PROMPT"]) + int(ns["MAX_NEW"]) + 8
    legacy_prefill = Stage.prefill
    legacy_ffn = Stage._ffn
    WANT_INT8 = os.environ.get("RPV16_FASTPATH_INT8", "1") != "0"
    WANT_NATIVE = os.environ.get("RPV16_FASTPATH_NATIVE", "1") != "0"
    WANT_SAMPLER = os.environ.get("RPV16_FASTPATH_SAMPLER", "1") != "0"
    KEEP_STOCK = os.environ.get("RPV16_FASTPATH_KEEP_STOCK", "0") == "1"
    SESSION_ROWS = max(0, int(os.environ.get("RPV16_FASTPATH_SESSION_ROWS", "4") or 0))
    WANT_SPLIT = os.environ.get("RPV16_FASTPATH_ATTN_SPLIT", "0") == "1"      # superseded by the native products
    _rows = os.environ.get("RPV16_FASTPATH_ATTN_NATIVE", "4").strip().lower()
    WANT_ATTN_ROWS = (1 << 30) if _rows in ("all", "") else max(0, int(_rows))
    WANT_HEAD = os.environ.get("RPV16_FASTPATH_HEAD", "1") != "0"
    STATE["loop"] = os.environ.get("RPV16_FASTPATH_LOOP", "1") != "0"
    TD = int(ns["TOKEN_DIM"])
    FFH = int(ns["FFN"])
    ns["FASTPATH_STATE"] = STATE

    # ------------------------------------------------------------------ attention core (v1, the reference path)
    def attend(self, h_block, cache, c, extra_mask, write):
        """Block of B new rows (absolute positions c..c+B-1) attending to cache[0:c] plus the block itself.
        write=True: the block's K/V are stored at [c, c+B) first and attention reads a zero-copy view.
        write=False: the cache is never touched; K/V are concatenated."""
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

    def block_forward(self, x, cache, c, mask_fn, write):
        h = rms(x, self.n1)
        a = attend(self, h, cache, c, mask_fn, write)
        x = x.float() + self.o(a)
        return x + self._ffn(rms(x, self.n2))

    # ------------------------------------------------------------------ stage level: prefill / step / FFN
    def prefill(self, x):
        x2, (k0, v0) = legacy_prefill(self, x)
        s = k0.shape[1]
        cap = s + CAP_EXTRA
        kb = torch.empty(KH, cap, HD); vb = torch.empty_like(kb)
        kb[:, :s] = k0; vb[:, :s] = v0
        kt = None
        if getattr(self, "_ffn_k", None) is not None:
            kt = torch.empty(KH, HD, cap)
            kt[:, :, :s] = k0.transpose(1, 2)
        return x2, KV(kb, vb, s, kt)

    def stage_step(self, x, cache):
        return block_forward(self, x, cache, cache.n, causal_block, True), cache

    def _ffn(self, h):
        fk = getattr(self, "_ffn_k", None)
        if fk is None:
            return legacy_ffn(self, h)
        flat = h.reshape(-1, D).float()
        return fk.call(flat, self._route(flat)).view_as(h)

    Stage.prefill = prefill
    Stage.step = stage_step
    Stage._ffn = _ffn

    # ------------------------------------------------------------------ engine level: cached AR / SAT / NAT blocks
    def kv_lists(caches, need):
        for cc in caches:
            cc.ensure(need)
            if cc.kt is None:
                cc.kt = torch.empty(KH, HD, cc.k.shape[1])
                cc.kt[:, :, :cc.n] = cc.k[:, :cc.n].transpose(1, 2)
        return [cc.k for cc in caches], [cc.v for cc in caches], [cc.kt for cc in caches]

    def set_n(caches, n):
        for cc in caches:
            cc.n = n

    def embed_rows(self, ids):
        t = torch.tensor(ids, dtype=torch.long)
        return self.in_proj(F.embedding(t, self.embed).float())

    base_step = Engine.step

    @torch.inference_mode()
    def step(self, token_id, caches):
        km = getattr(self, "_km", None)
        if km is None or not caches or not isinstance(caches[0], KV):
            return base_step(self, token_id, caches)
        x = embed_rows(self, [token_id])
        c = caches[0].n
        ks, vs, kts = kv_lists(caches, c + 1)
        x = km.forward(x, ks, vs, kts, c, 0, 0)
        set_n(caches, c + 1)
        return self.out_proj(rms(x, self.norm))

    @torch.inference_mode()
    def causal_extend(self, caches, ids, c0):
        """Append causal K/V for absolute positions c0..len(ids)-1; returns final hidden of those rows."""
        x = embed_rows(self, ids[c0:])
        km = getattr(self, "_km", None)
        if km is not None:
            ks, vs, kts = kv_lists(caches, len(ids))
            x = km.forward(x, ks, vs, kts, c0, 0, 0)
            set_n(caches, len(ids))
        else:
            for st, cache in zip(self.stages, caches):
                x = block_forward(st, x, cache, c0, causal_block, True)
        return self.out_proj(rms(x, self.norm))

    @torch.inference_mode()
    def suffix_forward(self, caches, x, P):
        """NAT: a causal block of rows for positions P.. over the prompt cache [0, P); the committed cache is not changed."""
        km = getattr(self, "_km", None)
        if km is not None:
            ks, vs, kts = kv_lists(caches, P + x.shape[0])
            return km.forward(x, ks, vs, kts, P, 1, 0)
        for st, cache in zip(self.stages, caches):
            x = block_forward(st, x, cache, P, causal_block, False)
        return x

    def new_caches(self, cap):
        on = getattr(self, "_km", None) is not None
        return [KV(torch.empty(KH, cap, HD), torch.empty(KH, cap, HD), 0, torch.empty(KH, HD, cap) if on else None)
                for _ in self.stages]

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
        km = getattr(self, "_km", None)
        if km is not None:
            ks, vs, kts = kv_lists(caches, n + 2)
            x = km.forward(x, ks, vs, kts, c, 2, n)
            set_n(caches, n)
        else:
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

    Engine.step = step
    Engine.causal_extend = causal_extend
    Engine.new_caches = new_caches
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
        vals = vals - torch.tensor([presence + frequency * c for c in counts.tolist()], dtype=torch.float64).to(vals.dtype)
        scores.index_copy_(0, ids, vals)
        return scores

    class History(list):
        """the token history of one request (a plain list to its readers) that also keeps, per n, which tokens followed
        each (n-1)-gram - what _block_repeated_ngrams finds by scanning the whole history after every token"""

        def __init__(self, tokens):
            super().__init__(int(t) for t in tokens)
            self.tables = {}

        def push(self, token):
            self.append(int(token))
            for n, table in self.tables.items():
                if len(self) >= n:
                    table.setdefault(tuple(self[-n:-1]), set()).add(self[-1])

        def banned(self, n):
            table = self.tables.get(n)
            if table is None:
                table = {}
                for i in range(len(self) - n + 1):
                    table.setdefault(tuple(self[i:i + n - 1]), set()).add(self[i + n - 1])
                self.tables[n] = table
            return table.get(tuple(self[-(n - 1):]), ())

    def block_ngrams(self, scores, history, body):
        if not isinstance(history, History):
            return self._block_repeated_ngrams(scores, history, body)
        n = max(0, min(8, int(body.get("no_repeat_ngram_size", 4) or 0)))
        if n < 2 or len(history) < n - 1:
            return scores
        banned = history.banned(n)
        if banned:
            idx = torch.tensor(sorted(banned), dtype=torch.long, device=scores.device)
            scores.index_fill_(0, idx, float("-inf"))
        return scores

    class Pieces:
        """The stream_tok text of every token of a growing list: exactly what decoding the whole list after each token
        yields, computed from a frozen decoded prefix plus the decoded tail. A prefix is frozen only where
        decode(left) + decode(right) == decode(left + right) holds with KEEP tokens of context to the right."""
        KEEP = 24
        TRY = 48

        def __init__(self, tok, whole=False):
            self.tok = tok; self.ids = []; self.base = 0; self.base_text = ""; self.previous = ""; self.whole = whole

        def push(self, token):
            ids = self.ids; tok = self.tok
            ids.append(int(token))
            if self.whole:
                tail = None
                decoded = tok.decode(ids, skip_special_tokens=False)
            else:
                tail = tok.decode(ids[self.base:], skip_special_tokens=False)
                decoded = self.base_text + tail
            prev = self.previous
            pc = decoded[len(prev):] if decoded.startswith(prev) else tok.decode([int(token)], skip_special_tokens=False)
            self.previous = decoded
            if tail is not None and len(ids) - self.base >= self.TRY:
                cut = len(ids) - self.KEEP
                left = tok.decode(ids[self.base:cut], skip_special_tokens=False)
                right = tok.decode(ids[cut:], skip_special_tokens=False)
                if left + right == tail and not left.endswith("\ufffd") and not right.startswith("\ufffd"):
                    self.base_text += left
                    self.base = cut
            return pc

    def pieces_selfcheck(tok):
        """incremental pieces against whole-list decoding on text, mixed scripts and random token ids"""
        g = torch.Generator().manual_seed(17)
        V = int(ns["VOCAB"])
        seqs = [list(tok.encode((PROBE_TEXT + " ") * 3).ids)[:400]]
        mixed = "na\u00efve caf\u00e9 \u2014 \u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8 \U0001f642\U0001f680 \u0645\u0631\u062d\u0628\u0627 \u043f\u0440\u0438\u0432\u0435\u0442\n\tx = [1, 2];  "
        seqs.append(list(tok.encode(mixed * 12).ids)[:400])
        for _ in range(3):
            seqs.append(torch.randint(0, V, (300,), generator=g).tolist())
        low = torch.randint(0, min(V, 2000), (300,), generator=g).tolist()      # special / byte-level tokens
        seqs.append(low)
        bad = 0; tokens = 0; frozen = 0
        for seq in seqs:
            a = Pieces(tok, whole=True); b = Pieces(tok)
            for t in seq:
                tokens += 1
                if a.push(t) != b.push(t):
                    bad += 1
            if a.previous != b.previous:
                bad += 1
            frozen += b.base
        return {"ok": bad == 0 and frozen > 0, "sequences": len(seqs), "tokens": tokens, "mismatches": bad, "frozen_tokens": frozen}

    piece_cache = {}

    def piece(self, tid):
        s = piece_cache.get(tid)
        if s is None:
            s = self.tokenizer.decode([int(tid)], skip_special_tokens=False)
            if len(piece_cache) < 200000:
                piece_cache[tid] = s
        return s

    def reference_work(probs, top_k, min_p, top_p):
        """the v1 / original filter chain: probabilities of the kept tokens, zero elsewhere"""
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
        return work

    def kept_candidates(kk, probs, lead, top_k, min_p, top_p):
        """The same kept set and weights without touching the whole vocabulary again. lead: the strictly descending
        top values of probs. Returns (ids or None for 'all tokens', weights) or None when the reference chain must decide."""
        n = probs.numel()
        thr = None
        if min_p > 0:
            thr = lead[0] * min_p                               # == work.max() * min_p
        if 0 < top_k < n:
            if top_k <= lead.numel():
                kth = lead[top_k - 1]
            elif top_k <= 65536:
                kth = kk.topk(probs, top_k, -1)[0][-1]
            else:
                return None
            thr = kth if thr is None else torch.maximum(thr, kth)
        if thr is not None:
            if not float(thr) > 0:
                return None
            ids = (probs >= thr).nonzero().flatten()            # ascending, like work.nonzero()
            w = probs.index_select(0, ids)
            if 0 < top_p < 1:
                vals, order = torch.sort(w, descending=True)
                ids = ids.index_select(0, order)
                cum = vals.cumsum(0) / vals.sum().clamp_min(tiny)
                drop = cum > top_p
                drop[1:] = drop[:-1].clone()
                drop[0] = False
                keep = ~drop
                ids, w = ids[keep], vals[keep]
            return ids, w
        if 0 < top_p < 1:
            # nucleus over the whole vocabulary: the sorted prefix is enough when the cut is clear of rounding and ties
            m = min(n, 1024)
            pv, pi, nans = kk.topk(probs, m, -1)
            if nans:
                return None
            ratio = pv.cumsum(0).double() / float(probs.sum())
            over = (ratio > top_p).nonzero()
            if over.numel() == 0:
                return None
            j = int(over[0])
            if j + 1 >= m or float(pv[j]) == float(pv[j + 1]) or bool(((ratio[:j + 2] - top_p).abs() < 4e-6).any()):
                return None
            return pi[:j + 1], pv[:j + 1]
        return None, probs

    def draw(ids, w):
        c = w.double().cumsum(0)
        total = float(c[-1])
        if not math.isfinite(total) or total <= 0:
            return None
        j = int(torch.searchsorted(c, torch.rand((), dtype=torch.float64) * total, right=True))
        j = min(j, c.numel() - 1)
        return int(ids[j]) if ids is not None else j

    def fast_sample(self, logp, history, body):
        scores = fast_penalties(logp.clone(), history, body)
        scores = block_ngrams(self, scores, history, body)
        greedy = bool(body.get("greedy", False)) or float(body.get("temperature", 0.0) or 0.0) <= 0
        temperature = max(1e-5, float(body.get("temperature", 1.0) or 1.0))
        probs = (scores if greedy else scores / temperature).softmax(-1)
        kk = STATE["K"] if STATE["topk"] else None
        lead = None
        if kk is not None:
            lead, lead_id, nans = kk.topk(probs, 21, -1)
            if nans or not bool((lead[1:] < lead[:-1]).all()):
                lead = None                                   # NaN or equal leaders: the reference call decides the order
        if lead is not None:
            top_prob, top_id = lead[:20], lead_id[:20]
        else:
            top_prob, top_id = torch.topk(probs, k=20)
        top = [{"token": piece(self, t), "prob": float(p), "id": int(t)} for p, t in zip(top_prob.tolist(), top_id.tolist())]
        if greedy:
            return int(top_id[0]), top
        top_k = int(body.get("top_k", 0) or 0)
        min_p = float(body.get("min_p", 0.0) or 0.0)
        top_p = float(body.get("top_p", 1.0) or 1.0)
        if lead is not None and STATE["sampler"]:
            kept = kept_candidates(kk, probs, lead, top_k, min_p, top_p)
            if kept is not None:
                token = draw(kept[0], kept[1])
                return (int(top_id[0]) if token is None else token), top
        work = reference_work(probs, top_k, min_p, top_p)
        total = work.sum()
        if not torch.isfinite(total) or total <= 0:
            return int(top_id[0]), top
        return int(torch.multinomial(work / total, 1).item()), top

    Engine._sample = fast_sample

    # ------------------------------------------------------------------ output head
    base_logp = Engine._hidden_to_logp

    def fast_logp(self, h):
        if getattr(self, "_head_on", False) and h.numel() == TD:
            z = STATE["K"].head_logits(self.embed, h.float()) * self.tied_scale + self.tied_bias
            return z.log_softmax(-1)
        return base_logp(self, h)

    Engine._hidden_to_logp = fast_logp

    def head_selfcheck(self, K, real_rows):
        """the bf16-table head against F.linear on the fp32 copy: every row, random and real hidden states"""
        info = {"ok": False}
        emb = getattr(self, "embed", None)
        if not getattr(self, "tied", False) or emb is None:
            info["reason"] = "no tied full-vocabulary head"
            return info
        if emb.dtype != torch.bfloat16 or emb.dim() != 2 or emb.shape[1] != 256 or TD != 256 or not emb.is_contiguous():
            info["reason"] = "embedding table is not a contiguous bfloat16 [V, 256] tensor"
            return info
        if not torch.equal(emb.float(), self.tied_w):
            info["reason"] = "the fp32 head is not the embedding table"
            return info
        g = torch.Generator().manual_seed(123)
        vecs = [torch.randn(TD, generator=g) * sc for sc in (1.0, 0.05, 12.0, 1.0, 3.0)]
        vecs[3][::2] = 0.0
        vecs[4] = vecs[4] * torch.pow(10.0, torch.empty(TD).uniform_(-3, 3, generator=g))
        vecs += [r.float().contiguous() for r in real_rows]
        differing = 0
        t_ref = t_new = 1e9
        for hv in vecs:
            t0 = time.perf_counter()
            ref = F.linear(hv[None], self.tied_w)[0]
            t1 = time.perf_counter()
            got = K.head_logits(emb, hv)
            t2 = time.perf_counter()
            t_ref = min(t_ref, t1 - t0); t_new = min(t_new, t2 - t1)
            differing += int((ref != got).sum())
        info.update(ok=differing == 0, vectors=len(vecs), real_hidden_states=len(real_rows), rows=int(emb.shape[0]),
                    rows_differing=differing, product_ms_fp32_sgemm=round(t_ref * 1e3, 3), product_ms_bf16_kernel=round(t_new * 1e3, 3))
        return info

    def sampler_selfcheck(kk):
        """kept set and weights of the candidate path against the reference chain, on peaked and flat distributions"""
        g = torch.Generator().manual_seed(31)
        V = int(ns["VOCAB"])
        cases = 0; bad = []; decided = 0
        for temp in (0.6, 1.0, 2.5):
            logits = torch.randn(V, generator=g) * 3.0
            logits[torch.randint(0, V, (40,), generator=g)] += torch.rand(40, generator=g) * 9
            logits[torch.randint(0, V, (7,), generator=g)] = float("-inf")
            probs = (logits / temp).softmax(-1)
            lead, lead_id, nans = kk.topk(probs, 21, -1)
            if nans or not bool((lead[1:] < lead[:-1]).all()):
                continue
            ref_top = torch.topk(probs, k=20)
            if not (torch.equal(ref_top.values, lead[:20]) and torch.equal(ref_top.indices, lead_id[:20])):
                bad.append(("top20", temp))
            for top_k, min_p, top_p in ((0, 0.0, 1.0), (40, 0.0, 1.0), (40, 0.0, 0.9), (5, 0.05, 0.95), (0, 0.02, 1.0), (0, 0.02, 0.8),
                                        (0, 0.0, 0.9), (0, 0.0, 0.5), (200, 0.0, 0.99), (1, 0.0, 1.0), (3, 0.9, 0.3)):
                cases += 1
                kept = kept_candidates(kk, probs, lead, top_k, min_p, top_p)
                if kept is None:
                    continue
                decided += 1
                mine = torch.zeros_like(probs)
                if kept[0] is None:
                    mine = kept[1].clone()
                else:
                    mine[kept[0]] = kept[1]
                if not torch.equal(mine, reference_work(probs, top_k, min_p, top_p)):
                    bad.append((temp, top_k, min_p, top_p))
        # the draw itself: frequencies of a small kept set
        w = torch.tensor([0.5, 0.25, 0.125, 0.125])
        ids = torch.tensor([7, 3, 11, 2])
        torch_state = torch.random.get_rng_state()
        torch.manual_seed(5)
        counts = {}
        for _ in range(4000):
            t = draw(ids, w); counts[t] = counts.get(t, 0) + 1
        torch.random.set_rng_state(torch_state)
        freq = [counts.get(int(i), 0) / 4000 for i in ids]
        draw_ok = all(abs(f - float(p)) < 0.03 for f, p in zip(freq, w))
        return {"ok": not bad and draw_ok and decided > 0, "cases": cases, "decided_by_candidates": decided, "failed": bad[:6],
                "draw_frequencies": [round(f, 3) for f in freq]}

    def topk_selfcheck(kk):
        g = torch.Generator().manual_seed(8)
        bad = 0
        for trial in range(40):
            v = torch.randn(int(ns["VOCAB"]) if trial < 6 else 4000, generator=g)
            if trial % 3 == 0:
                v = (v * 8).round() / 8
            k = 21 if trial < 6 else int(torch.randint(1, 60, (1,), generator=g))
            vals, idx, nans = kk.topk(v, k, -1)
            ref = torch.sort(v, descending=True, stable=True)
            if nans or not (torch.equal(vals, ref.values[:k]) and torch.equal(idx, ref.indices[:k])):
                bad += 1
        return {"ok": bad == 0, "bad": bad, "cases": 40}

    # ------------------------------------------------------------------ the int8 kernel switch
    def to_kernel(K, mod, layout):
        w = mod.weight()
        if w.qscheme() != torch.per_tensor_affine or w.dtype != torch.qint8:
            raise RuntimeError("unsupported weight quantisation %s / %s" % (w.qscheme(), w.dtype))
        b = mod.bias()
        packed = K.qpack(w.int_repr(), layout)
        return K.QLinear(packed, layout, float(w.q_scale()), int(w.q_zero_point()), None if b is None else b.detach().float().contiguous())

    def native_selfchecks(K, router, norm_w):
        """each native float piece against the libtorch operator it replaces; returns the set_native arguments"""
        g = torch.Generator().manual_seed(77)
        info = {}
        found = K.set_native(*NATIVE_OFF)[5:]
        info["entry_points"] = {"cblas_sgemm_batch": bool(found[0]), "sgemm_": bool(found[1]), "Sleef_expf16_u10": bool(found[2]),
                                "Sleef_expf8_u10": bool(found[3])}
        rows = [torch.randn(b, D, generator=g) * sc for b in (1, 2, 5, 40) for sc in (1.0, 1e-3, 37.0)]
        lanes = None
        for cand in (16, 8):
            K.set_native(False, False, False, False, False, cand, 16, True, 0, 16)
            if all(torch.equal(K.rms_native(x, norm_w), rms(x, norm_w)) for x in rows):
                lanes = cand
                break
        info["rms"] = {"ok": lanes is not None, "sum_lanes": lanes, "cases": len(rows)}
        sm = [torch.randn(KH, G, b, L, generator=g) * sc for b in (1, 3) for L in (1, 2, 7, 15, 16, 17, 31, 32, 33, 100, 257, 1000, 1537)
              for sc in (1.0, 6.0)]
        for t in sm[::3]:
            if t.shape[-1] > 1:
                t[..., 0] = float("-inf")                       # masked keys
        soft = None
        for sm_lanes, exp_kind, tree in [(w_, e, t) for w_ in (16, 8) for e in (0, 1) for t in (True, False)]:
            if exp_kind == 0 and not found[2 if sm_lanes == 16 else 3]:
                continue
            K.set_native(False, False, False, False, False, lanes or 16, sm_lanes, tree, exp_kind, 16)
            if all(torch.equal(K.softmax_native(x), x.softmax(-1)) for x in sm):
                soft = (sm_lanes, tree, exp_kind)
                break
        info["softmax"] = {"ok": soft is not None, "lanes_tree_exp": soft, "cases": len(sm)}
        silu_lanes = None
        gates = [(torch.randn(b, FFH, generator=g) * sc, torch.randn(b, FFH, generator=g)) for b in (1, 2, 3, 6) for sc in (1.0, 4.0, 30.0)
                 if b * FFH < 32768 and (b * FFH) % 32 == 0]
        for cand in (16, 8):
            if not gates or not found[2 if cand == 16 else 3]:
                continue
            K.set_native(False, False, False, False, True, 16, 16, True, 0, cand)
            if all(torch.equal(K.silu_native(a, b), F.silu(a) * b) for a, b in gates):
                silu_lanes = cand
                break
        info["silu_gate"] = {"ok": silu_lanes is not None, "lanes": silu_lanes, "cases": len(gates)}
        router_ok = bool(found[1]) and all(torch.equal(K.router_native(x, router), F.linear(x, router)) for x in rows)
        info["router"] = {"ok": router_ok, "cases": len(rows)}
        K.set_native(False, soft is not None, False, False, False, lanes or 16, soft[0] if soft else 16, soft[1] if soft else True,
                     soft[2] if soft else 0, 16)
        attn_ok = bool(found[0])
        bad = []
        cases = 0
        if attn_ok:
            cap = MAX_KEYS + 80
            kt = torch.zeros(KH, HD, cap); vv = torch.zeros(KH, cap, HD)
            lengths = sorted(set(list(range(1, 258)) + list(range(258, MAX_KEYS, 13)) + [511, 512, 513, 1023, 1024, 1025, MAX_KEYS - 1, MAX_KEYS]))
            for L in lengths:
                blocks = (1, 2, 3, 4) if (L < 24 or L in (256, 257, 513)) else (1,)
                if L in (64, 200):
                    blocks = blocks + (66,)
                for B in blocks:
                    lo = int(torch.randint(0, 70, (1,), generator=g))
                    q = torch.randn(B, D, generator=g)
                    Kc = torch.randn(KH, L, HD, generator=g); Vc = torch.randn(KH, L, HD, generator=g)
                    kt[:, :, lo:lo + L] = Kc.transpose(1, 2); vv[:, lo:lo + L] = Vc
                    a_n, sc_n = K.attend_native(q, kt, vv, lo, L, QH, B <= SESSION_ROWS)
                    qv = q.view(B, KH, G, HD).permute(1, 2, 0, 3)
                    sc_r = torch.matmul(qv, Kc.transpose(-1, -2).unsqueeze(1)) / SQ
                    a_r = torch.matmul(sc_r.softmax(-1), Vc.unsqueeze(1)).permute(2, 0, 1, 3).reshape(B, D)
                    cases += 1
                    if not (torch.equal(sc_n, sc_r) and torch.equal(a_n, a_r)):
                        bad.append((L, B, lo))
            attn_ok = not bad
        info["attention"] = {"ok": attn_ok, "cases": cases, "failed_L_B_lo": bad[:8]}
        split = [False, False]
        if attn_ok and WANT_SPLIT and hasattr(K, "set_attention_split"):
            def split_cases(case_list, sessions):
                failed = []
                for L, B in case_list:
                    lo = int(torch.randint(0, 70, (1,), generator=g))
                    q = torch.randn(B, D, generator=g)
                    Kc = torch.randn(KH, L, HD, generator=g); Vc = torch.randn(KH, L, HD, generator=g)
                    kt[:, :, lo:lo + L] = Kc.transpose(1, 2); vv[:, lo:lo + L] = Vc
                    qv = q.view(B, KH, G, HD).permute(1, 2, 0, 3)
                    sc_r = torch.matmul(qv, Kc.transpose(-1, -2).unsqueeze(1)) / SQ
                    a_r = torch.matmul(sc_r.softmax(-1), Vc.unsqueeze(1)).permute(2, 0, 1, 3).reshape(B, D)
                    for inside in sessions:
                        a_n, sc_n = K.attend_native(q, kt, vv, lo, L, QH, inside)
                        if not (torch.equal(sc_n, sc_r) and torch.equal(a_n, a_r)):
                            failed.append((L, B, lo, inside))
                return failed
            small = [(L, B) for L in list(range(1, 66)) + [100, 127, 128, 129, 255, 256, 257, 400, 511, 512, 513, 777, 1023, 1024, 1025,
                                                             MAX_KEYS - 1, MAX_KEYS] for B in (1, 2, 3, 4)]
            K.set_attention_split(True, False)
            bad_small = split_cases(small, (True, False))
            split[0] = not bad_small
            big = [(L, B) for B in (5, 8, 16, 33, 66, 130, 244) for L in (1, 7, 64, 200, 257, 513, 1025, MAX_KEYS)]
            K.set_attention_split(split[0], True)
            bad_big = split_cases(big, (False, True))
            split[1] = not bad_big
            info["attention_split"] = {"blocks_1_to_4_rows": split[0], "bigger_blocks": split[1], "cases": 2 * (len(small) + len(big)),
                                       "failed_small": bad_small[:4], "failed_big": bad_big[:4]}
        if hasattr(K, "set_attention_split"):
            K.set_attention_split(split[0], split[1])
        STATE["attention_split"] = {"blocks_1_to_4_rows": split[0], "bigger_blocks": split[1]}
        rows_native = [False, 0]
        if attn_ok and WANT_ATTN_ROWS and hasattr(K, "set_attention_native"):
            t1 = time.time()
            Kfull = torch.randn(KH, cap, HD, generator=g); Vfull = torch.randn(KH, cap, HD, generator=g)
            kt.copy_(Kfull.transpose(1, 2)); vv.copy_(Vfull)

            def exact(B, L, inside):
                lo = int(torch.randint(0, 70, (1,), generator=g))
                q = torch.randn(B, D, generator=g)
                Kc = Kfull[:, lo:lo + L]; Vc = Vfull[:, lo:lo + L]
                qv = q.view(B, KH, G, HD).permute(1, 2, 0, 3)
                sc_r = torch.matmul(qv, Kc.transpose(-1, -2).unsqueeze(1)) / SQ
                a_r = torch.matmul(sc_r.softmax(-1), Vc.unsqueeze(1)).permute(2, 0, 1, 3).reshape(B, D)
                a_n, sc_n = K.attend_native(q, kt, vv, lo, L, QH, inside)
                return torch.equal(sc_n, sc_r) and torch.equal(a_n, a_r)

            K.set_attention_native(True, 0)
            bad1 = [L for L in range(1, MAX_KEYS + 1) if not exact(1, L, L % 2 == 0)]
            rows_native[0] = not bad1
            an = {"single_row": rows_native[0], "key_counts": MAX_KEYS, "failed_single_row": bad1[:6]}
            if WANT_ATTN_ROWS >= 4:
                K.set_attention_native(rows_native[0], 4)
                bad4 = [(B, L) for B in (2, 3, 4) for L in range(1, MAX_KEYS + 1) if not exact(B, L, (L + B) % 2 == 0)]
                an["rows_2_to_4"] = not bad4; an["failed_rows_2_to_4"] = bad4[:6]
                if not bad4:
                    rows_native[1] = 4
                    if WANT_ATTN_ROWS > 4:
                        K.set_attention_native(rows_native[0], 1 << 30)
                        shapes = sorted(set((B, L) for B in (5, 6, 7, 8, 12, 16, 24, 33, 48, 64, 100, 130, 200, 244, 256, 300, 400, 512)
                                            for L in (B, B + 1, B + 63, 2 * B + 1, 255 + B, 383, 384, 385, 767, 768, 769, 1151, 1152, 1153, MAX_KEYS)
                                            if B <= L <= MAX_KEYS))
                        badb = [(B, L) for i, (B, L) in enumerate(shapes) if not exact(B, L, i % 2 == 0)]
                        an["bigger_blocks"] = not badb; an["bigger_block_shapes"] = len(shapes); an["failed_bigger_blocks"] = badb[:6]
                        if not badb:
                            rows_native[1] = 1 << 30
            an["seconds"] = round(time.time() - t1, 2)
            info["attention_native"] = an
        if hasattr(K, "set_attention_native"):
            K.set_attention_native(rows_native[0], rows_native[1])
        STATE["attention_native"] = {"single_row": rows_native[0], "rows_2_to_4": rows_native[1] >= 4, "bigger_blocks": rows_native[1] > 4}
        few = [torch.randn(b, D, generator=g) * sc for b in (1, 2, 3, 4) for sc in (1.0, 37.0)]
        info["router_in_session"] = bool(router_ok and hasattr(K, "router_native_session") and
                                         all(torch.equal(K.router_native_session(x, router), F.linear(x, router)) for x in few))
        args = (lanes is not None, soft is not None, attn_ok, router_ok, silu_lanes is not None, lanes or 16, soft[0] if soft else 16,
                soft[1] if soft else True, soft[2] if soft else 0, silu_lanes or 16)
        return args, info

    def probe(self):
        """a teacher-forced walk through every decode path: hidden states, caches, log-probs (and how long it took)"""
        out = {}; tm = {}
        ids_all = list(self.tokenizer.encode(PROBE_TEXT).ids)
        while len(ids_all) < 76:
            ids_all = ids_all + ids_all
        ids, cont = ids_all[:40], ids_all[40:76]
        with torch.inference_mode():
            t0 = time.perf_counter()
            h, caches = self.prefill(ids)
            tm["ar_prefill_ms"] = (time.perf_counter() - t0) * 1e3
            hs = [h]
            t0 = time.perf_counter()
            for t in cont[:20]:
                h = self.step(t, caches); hs.append(h)
            tm["ar_step_ms"] = (time.perf_counter() - t0) / 20 * 1e3
            out["ar_hidden"] = torch.cat(hs, 0)
            out["ar_k"] = torch.stack([c.k[:, :c.n] for c in caches]); out["ar_v"] = torch.stack([c.v[:, :c.n] for c in caches])
            t0 = time.perf_counter()
            out["logp"] = self._hidden_to_logp(h[0])
            tm["head_ms"] = (time.perf_counter() - t0) * 1e3
            caches = self.new_caches(len(ids) + 48)
            context = list(ids); rows = []; pos = 0
            t_first = None
            t0 = time.perf_counter()
            for i, stride in enumerate((2, 2, 1, 2, 1, 2, 2, 2)):
                pair_h, last_h = self.sat_forward(caches, context)
                if i == 0:
                    t_first = time.perf_counter()
                rows.append(last_h.reshape(1, -1)); rows.append(pair_h)
                context = context + cont[pos:pos + stride]; pos += stride
            tm["sat_first_ms"] = (t_first - t0) * 1e3
            tm["sat_step_ms"] = (time.perf_counter() - t_first) / 7 * 1e3
            n = len(context)
            for cc in caches:
                cc.n = min(cc.n, n - 1)
            rows.append(self.causal_extend(caches, context, min(caches[0].n, n - 1))[-1:])
            out["sat_hidden"] = torch.cat(rows, 0)
            out["sat_k"] = torch.stack([c.k[:, :c.n] for c in caches]); out["sat_v"] = torch.stack([c.v[:, :c.n] for c in caches])
            caches = self.new_caches(len(ids) + 48)
            out["nat_prompt_hidden"] = self.causal_extend(caches, ids, 0)
            suffix = [2] * 28 + cont[:4]
            t0 = time.perf_counter()
            x = suffix_forward(self, caches, embed_rows(self, suffix), len(ids))
            out["nat_hidden"] = self.out_proj(rms(x, self.norm))
            tm["nat_pass_ms"] = (time.perf_counter() - t0) * 1e3
            x = suffix_forward(self, caches, embed_rows(self, suffix[::-1]), len(ids))
            out["nat_hidden_2"] = self.out_proj(rms(x, self.norm))
        return out, {k: round(v, 3) for k, v in tm.items()}

    def switch(self, on):
        kit = self._kit
        for st, ks in zip(self.stages, kit["stages"]):
            src = ks["kernel"] if on else ks["stock"]
            st.q, st.k, st.v, st.o = src["q"], src["k"], src["v"], src["o"]
            st.experts = src["experts"]
            if on:
                st._ffn_k = ks["ffn"]
            elif hasattr(st, "_ffn_k"):
                del st._ffn_k
        src = kit["kernel"] if on else kit["stock"]
        self.in_proj, self.out_proj = src["in_proj"], src["out_proj"]
        self._km = kit["model"] if on else None
        self._head_on = bool(on and STATE["head"])

    def upgrade(self):
        rec = {"revision": REVISION, "pid": os.getpid(), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "torch": torch.__version__, "threads": torch.get_num_threads(), "step": self.meta.get("step"),
               "rss_mb_before": rss_mb(), "int8_kernels": False, "ok": False, "error": None}
        STATE["summary"] = rec
        t_all = time.time()
        self._km = None
        K, rec["kernel"] = load_kernel()
        STATE["K"] = K
        rec["topk_check"] = topk_selfcheck(K)
        STATE["topk"] = bool(rec["topk_check"]["ok"])
        if STATE["topk"] and WANT_SAMPLER:
            rec["sampler_check"] = sampler_selfcheck(K)
            STATE["sampler"] = bool(rec["sampler_check"]["ok"])
        try:
            rec["pieces_check"] = pieces_selfcheck(self.tokenizer)
        except Exception as exc:
            rec["pieces_check"] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
        STATE["pieces"] = bool(rec["pieces_check"]["ok"])
        if not WANT_INT8:
            rec["error"] = "int8 kernels disabled by RPV16_FASTPATH_INT8=0"
            return rec
        # 1. which rounding formula does this torch / fbgemm build quantise activations with?
        g = torch.Generator().manual_seed(9)
        probe_mod = self.stages[0].q
        probe_k = to_kernel(K, probe_mod, 4)
        score = {}
        for variant in (0, 1, 2):
            K.set_quant_variant(variant)
            g2 = torch.Generator().manual_seed(9)
            score[variant] = sum(int(torch.equal(probe_mod(x), probe_k.call(x))) for x in (adversarial(1 + t % 3, D, g2) for t in range(40)))
        best = max(score, key=score.get)
        K.set_quant_variant(best)
        rec["quantisation_variant"] = {"chosen": best, "agreement_of_40": score}
        if score[best] != 40:
            raise RuntimeError("no activation rounding formula reproduces the stock Linear: %s" % score)
        # 2. every Linear: pack, compare with its stock module on probe inputs
        t0 = time.time()
        checked = 0; failed = []

        def conv(mod, layout, name, sparse_input=False):
            nonlocal checked
            kl = to_kernel(K, mod, layout)
            k_in = int(kl.in_features())
            for x in ((gated(1, k_in, g), gated(3, k_in, g)) if sparse_input else (adversarial(1, k_in, g), torch.randn(3, k_in, generator=g) * 0.7)):
                checked += 1
                if not torch.equal(mod(x), kl.call(x)):
                    failed.append(name)
            return kl

        kit = {"stages": [], "stock": {"in_proj": self.in_proj, "out_proj": self.out_proj}}
        model = K.Model(D, QH, KH, HD)
        kin, kout = conv(self.in_proj, 4, "in_proj"), conv(self.out_proj, 4, "out_proj")
        kit["kernel"] = {"in_proj": QLin(kin), "out_proj": QLin(kout)}
        for i, st in enumerate(self.stages):
            kq, kk_, kv, ko = (conv(getattr(st, nm), 4, "stage%d.%s" % (i, nm)) for nm in ("q", "k", "v", "o"))
            gates, ups, downs = [], [], []
            for e, (gate, up, down) in enumerate(st.experts):
                gates.append(conv(gate, 4, "stage%d.expert%d.gate" % (i, e)))
                ups.append(conv(up, 4, "stage%d.expert%d.up" % (i, e)))
                downs.append(conv(down, 1, "stage%d.expert%d.down" % (i, e), True))
            ffn = K.FFN(gates, ups, downs)
            model.add_stage(st.n1.contiguous(), st.n2.contiguous(), st.router.contiguous(), kq, kk_, kv, ko, ffn, int(st.window))
            kit["stages"].append({
                "stock": {"q": st.q, "k": st.k, "v": st.v, "o": st.o, "experts": st.experts},
                "kernel": {"q": QLin(kq), "k": QLin(kk_), "v": QLin(kv), "o": QLin(ko),
                           "experts": [(QLin(a), QLin(b), QLin(c)) for a, b, c in zip(gates, ups, downs)]},
                "ffn": ffn})
        kit["model"] = model
        self._kit = kit
        rec["linear_check"] = {"linears": 2 + len(self.stages) * (4 + 3 * len(self.stages[0].experts)), "comparisons": checked,
                               "failed": failed[:12], "seconds": round(time.time() - t0, 2)}
        rec["rss_mb_packed"] = rss_mb()
        if failed:
            raise RuntimeError("%d kernel Linears differ from their stock modules" % len(failed))
        # 3. native float pieces (each compared with the libtorch operator)
        K.set_session_rows(SESSION_ROWS)
        rec["session_rows"] = SESSION_ROWS
        nat_args = NATIVE_OFF
        if WANT_NATIVE:
            t0 = time.time()
            nat_args, rec["native_check"] = native_selfchecks(K, self.stages[0].router.contiguous(), self.stages[0].n1.contiguous())
            rec["native_check"]["seconds"] = round(time.time() - t0, 2)
            if SESSION_ROWS and not rec["native_check"].get("router_in_session", False):
                K.set_session_rows(0)                     # the router product then runs outside any session, as F.linear does
                rec["session_rows"] = 0
        # 4. the whole model: stock modules (v1 path) against the kernels, bitwise
        K.set_native(*NATIVE_OFF)
        self._head_on = False
        ref, t_ref = probe(self)
        rec["probe_ms_stock"] = t_ref
        STATE["head"] = False
        if WANT_HEAD and hasattr(K, "head_logits"):
            try:
                real = [ref["ar_hidden"][i] for i in range(0, ref["ar_hidden"].shape[0], 2)] + [ref["sat_hidden"][i] for i in range(0, ref["sat_hidden"].shape[0], 3)]
                rec["head_check"] = head_selfcheck(self, K, real)
            except Exception as exc:
                rec["head_check"] = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc)}
            STATE["head"] = bool(rec["head_check"]["ok"])
        results = {}
        for label, args in (("libtorch_float", NATIVE_OFF), ("native_float", nat_args)):
            if label == "native_float" and not any(args[:5]):
                continue
            on = K.set_native(*args)
            switch(self, True)
            K.stats(True); K.profile(True)
            got, t_k = probe(self)
            st_ = K.stats(True); pr = K.profile(True)
            switch(self, False)
            diff = [k for k in ref if not (ref[k].shape == got[k].shape and torch.equal(ref[k], got[k]))]
            results[label] = {"bitwise_identical": not diff, "differs": diff, "probe_ms": t_k, "native_on": [int(v) for v in on[:5]],
                              "active_down_columns": round(st_[3] / max(st_[4], 1), 4), "wide_calls": int(st_[6]),
                              "profile_ms_per_forward": [round(v / max(st_[7], 1) * 1e3, 4) for v in pr]}
        rec["end_to_end"] = results
        rec["probe_tensors"] = {k: list(v.shape) for k, v in ref.items()}
        if not results["libtorch_float"]["bitwise_identical"]:
            raise RuntimeError("kernel model differs from the stock path: %s" % results["libtorch_float"]["differs"])
        final_args = nat_args if results.get("native_float", {}).get("bitwise_identical") else NATIVE_OFF
        on = K.set_native(*final_args)
        STATE["native"] = {"rms": bool(on[0]), "softmax": bool(on[1]), "attention": bool(on[2]), "router": bool(on[3]), "silu_gate": bool(on[4])}
        rec["native"] = STATE["native"]
        rec["extras"] = {"attention_native": STATE["attention_native"], "attention_split": STATE["attention_split"], "bf16_head": bool(STATE["head"]),
                         "incremental_pieces": bool(STATE["pieces"]), "own_loop_with_ngram_index": bool(STATE["loop"])}
        switch(self, True)
        rec["int8_kernels"] = True
        STATE["int8_kernels"] = True
        chosen = results["native_float"] if any(on[:5]) else results["libtorch_float"]
        rec["speedup"] = {k: round(t_ref[k] / max(chosen["probe_ms"][k], 1e-9), 2) for k in t_ref}
        if not KEEP_STOCK:
            kit.pop("stock"); [ks.pop("stock") for ks in kit["stages"]]
            del ref, got
            trim()
        rec["stock_modules_dropped"] = not KEEP_STOCK
        rec["rss_mb_after"] = rss_mb()
        rec["upgrade_seconds"] = round(time.time() - t_all, 2)
        rec["ok"] = True
        return rec

    base_init = Engine.__init__

    def init(self, root, tokenizer_path):
        base_init(self, root, tokenizer_path)
        try:
            rec = upgrade(self)
        except Exception as exc:                                     # the v1 path with the stock modules stays in place
            rec = STATE.get("summary") or {"revision": REVISION}
            rec["error"] = "%s: %s" % (type(exc).__name__, exc)
            try:
                if getattr(self, "_kit", None) is not None and "stock" in self._kit:
                    switch(self, False)
            except Exception as exc2:
                rec["restore_error"] = "%s: %s" % (type(exc2).__name__, exc2)
            self._km = None
            self._head_on = False
            STATE["int8_kernels"] = False
            STATE["head"] = False
        trim()
        rec["rss_mb_final"] = rss_mb()
        STATE["error"] = rec.get("error")
        STATE["summary"] = rec
        try:
            os.makedirs(RECEIPTS, exist_ok=True)
            path = os.path.join(RECEIPTS, "worker_%d_%d.json" % (os.getpid(), int(time.time())))
            with open(path, "w") as f:
                json.dump(rec, f, indent=1, default=str)
            STATE["receipt"] = path
        except Exception as exc:
            STATE["receipt"] = "not written: %s" % exc
        print("[rpv16-fastpath-v2] " + json.dumps({k: rec.get(k) for k in ("ok", "int8_kernels", "native", "extras", "error", "speedup",
                                                                              "rss_mb_before", "rss_mb_final", "upgrade_seconds")}, default=str), flush=True)

    Engine.__init__ = init
    ns["FASTPATH_SWITCH"] = switch

    # ------------------------------------------------------------------ cached SAT + NAT event paths
    base_events = Engine.events

    def events(self, body, request_id, entered):
        mode = mm.mode_key(body)
        if (mode == "ar" and not STATE["loop"]) or os.environ.get("RPV16_FASTPATH_MULTIMODE", "1") == "0":
            for ev in base_events(self, body, request_id, entered):
                if ev.get("event") in ("stats", "done") and isinstance(ev.get("stats"), dict):
                    ev["stats"]["fastpath_revision"] = REVISION
                    ev["stats"]["kv_cache"] = True
                    ev["stats"]["int8_kernels"] = bool(STATE["int8_kernels"])
                yield ev
            return
        yield from fast_multimode_events(self, body, request_id, entered, mode)

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
        start = time.perf_counter(); generated = []; forwards = 0; prefill_s = 0.0
        strides = {"1": 0, "2": 0}; policy_strides = {"1": 0, "2": 0}; passes_done = 0; first_token_s = None

        fast_loop = bool(STATE["loop"])
        pieces = Pieces(self.tokenizer, whole=not (fast_loop and STATE["pieces"]))
        history = History(ids) if fast_loop and mode != "nat" else None      # what the sampler sees as ids + generated

        def token_event(token, top):
            nonlocal first_token_s
            generated.append(int(token))
            if history is not None:
                history.push(token)
            pc = pieces.push(token)
            if first_token_s is None:
                first_token_s = time.perf_counter() - entered
            return {"event": "stream_tok", "data": {"text": pc, "i": len(generated) - 1, "top": top}}

        if mode == "ar":
            h, caches = self.prefill(ids); forwards += 1; prefill_s = time.perf_counter() - start
            for i in range(budget):
                if self.cancel.is_set():
                    break
                token, top = self.sample(h[0], history if history is not None else ids + generated, body)
                if token in self.eos_ids and i >= minimum and not ignore_eos:
                    break
                yield token_event(token, top); strides["1"] += 1
                if i + 1 < budget and not self.cancel.is_set():
                    h = self.step(token, caches); forwards += 1
        elif mode.startswith("sat_"):
            caches = self.new_caches(len(ids) + budget + 8)
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
                    token, top = self.sample(hh, history if history is not None else ids + generated, body)
                    if token in self.eos_ids and len(generated) >= minimum and not ignore_eos:
                        ended = True; break
                    if self.cancel.is_set():
                        ended = True; break
                    yield token_event(token, top); committed += 1
                if committed:
                    strides[str(committed)] += 1
        else:
            caches = self.new_caches(len(ids) + budget + 8)
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
                x = suffix_forward(self, caches, embed_rows(self, suffix), P)
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
                 "fastpath_revision": REVISION, "kv_cache": True, "int8_kernels": bool(STATE["int8_kernels"]),
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
    ns["FASTPATH_TEST"] = {"History": History, "Pieces": Pieces, "block_ngrams": block_ngrams, "fast_penalties": fast_penalties}
