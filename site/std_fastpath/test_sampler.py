# Distribution-level equivalence of the top-k-first sampler vs the runtime's _sample (copied verbatim).
import math, torch, time, json
import torch.nn.functional as F
torch.set_num_threads(8)
NAT_MASK_ID = 5; EOS = 2
def _decode_guard_logits(logits):
    logits = logits.clone(); logits[..., NAT_MASK_ID] = float("-inf"); return logits

def orig_probs(logits, T, top_k, top_p, min_p):   # verbatim math of runtime _sample up to safe_probs
    guarded = _decode_guard_logits(logits).float()
    fallback = guarded.argmax(-1, keepdim=True)
    temperature = float(T)
    scores = guarded / max(temperature, 1e-8)
    k = int(top_k or 0)
    if 0 < k < scores.size(-1):
        cutoff = torch.topk(scores, k, dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < cutoff, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    relative_min_p = max(0.0, min(1.0, float(min_p or 0.0)))
    if relative_min_p > 0.0:
        row_floor = probs.amax(dim=-1, keepdim=True) * relative_min_p
        probs = probs.masked_fill(probs < row_floor, 0.0)
    nucleus_p = max(0.0, min(1.0, float(top_p)))
    if nucleus_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        remove = torch.cumsum(sorted_probs, dim=-1) > nucleus_p
        shifted = remove.clone(); shifted[..., 1:] = remove[..., :-1]; shifted[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(shifted, 0.0)
        probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    row_sum = probs.sum(dim=-1, keepdim=True)
    valid = torch.isfinite(row_sum) & (row_sum > 0.0)
    normalized = probs / row_sum.clamp_min(torch.finfo(probs.dtype).tiny)
    fallback_probs = F.one_hot(fallback.squeeze(-1), num_classes=guarded.size(-1)).to(dtype=normalized.dtype)
    return torch.where(valid, normalized, fallback_probs)

def fast_probs(logits, T, top_k, top_p, min_p):   # the overlay's math, scattered back to vocab for comparison
    V = logits.size(-1)
    k = int(top_k or 0)
    guarded = _decode_guard_logits(logits).float()
    rows = guarded.reshape(-1, V)
    scores = rows / max(float(T), 1e-8)
    vals, idx = torch.topk(scores, k, dim=-1)
    if bool((scores >= vals[:, -1:]).sum(dim=-1).ne(k).any()) or not bool(torch.isfinite(vals).all()):
        return None
    probs = torch.softmax(vals, dim=-1)
    mp = max(0.0, min(1.0, float(min_p or 0.0)))
    if mp > 0.0:
        probs = probs.masked_fill(probs < probs[:, :1] * mp, 0.0)
    p = max(0.0, min(1.0, float(top_p)))
    if p < 1.0:
        remove = torch.cumsum(probs, dim=-1) > p
        shifted = remove.clone(); shifted[:, 1:] = remove[:, :-1]; shifted[:, 0] = False
        probs = probs.masked_fill(shifted, 0.0)
    row_sum = probs.sum(-1, keepdim=True)
    if not bool((torch.isfinite(row_sum) & (row_sum > 0)).all()):
        return None
    full = torch.zeros_like(rows).scatter_(-1, idx, probs / row_sum)
    return full.reshape(guarded.shape)

V = 129280
worst = 0.0; n = 0; skipped = 0
cases = [(0.5, 32, 0.92, 0.005), (0.55, 48, 0.94, 0.0), (0.7, 40, 0.95, 0.0), (1.1, 512, 0.98, 0.001), (1.0, 8, 1.0, 0.0), (0.3, 100, 0.5, 0.05)]
for scale in (1.0, 3.0, 8.0):
    for (T, k, p, mp) in cases:
        for rows in (1, 2):
            logits = torch.randn(rows, V) * scale
            a = orig_probs(logits, T, k, p, mp); b = fast_probs(logits, T, k, p, mp)
            if b is None: skipped += 1; continue
            d = float((a - b).abs().max()); worst = max(worst, d); n += 1
print(json.dumps({'cases': n, 'skipped_ties': skipped, 'max_abs_prob_diff': worst}))
# timing on one row
logits = torch.randn(1, V) * 3
for name, fn in (('orig', orig_probs), ('fast', fast_probs)):
    fn(logits, 0.5, 32, 0.92, 0.005); t = time.perf_counter()
    for _ in range(50): fn(logits, 0.5, 32, 0.92, 0.005)
    print(name, 'ms', round((time.perf_counter() - t) / 50 * 1e3, 3))
