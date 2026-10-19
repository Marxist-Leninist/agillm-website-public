import torch, gc, math, time
torch.set_num_threads(8)
def append(cached, new):
    B, H, T, D = cached.shape; n = new.size(2)
    base = cached._base
    if (base is not None and getattr(base, "_sfp_kv_fill", -1) == T and cached.data_ptr() == base.data_ptr()
            and cached.stride() == base.stride() and T + n <= base.size(2)):
        base[:, :, T:T + n].copy_(new); base._sfp_kv_fill = T + n
        return base[:, :, :T + n], 'tip'
    cap = ((max(2 * (T + n), T + n + 128) + 63) // 64) * 64
    buf = torch.empty((B, H, cap, D), dtype=new.dtype)
    buf[:, :, :T].copy_(cached); buf[:, :, T:T + n].copy_(new); buf._sfp_kv_fill = T + n
    return buf[:, :, :T + n], 'alloc'
k = torch.randn(1, 20, 40, 64)
stats = {}
ref = k.clone()
for i in range(600):
    new = torch.randn(1, 20, 1, 64)
    k, how = append(k, new); stats[how] = stats.get(how, 0) + 1
    ref = torch.cat([ref, new], 2)
    gc.collect()
print('append stats', stats, 'equal', torch.equal(k, ref))
# branch safety: two children from the same parent
p = k
c1, h1 = append(p, torch.ones(1, 20, 1, 64))
c2, h2 = append(p, torch.zeros(1, 20, 1, 64))
print('branch', h1, h2, 'c1 last ones', bool((c1[:, :, -1] == 1).all()), 'c2 last zeros', bool((c2[:, :, -1] == 0).all()), 'parent intact', torch.equal(p, ref))
# attention numerics: strided view vs contiguous
q = torch.randn(1, 20, 1, 64)
for T in (100, 600):
    kv = k[:, :, :T]; kc = kv.contiguous()
    a1 = ((q @ kv.transpose(-1, -2)) / 8.0).softmax(-1) @ kv
    a2 = ((q @ kc.transpose(-1, -2)) / 8.0).softmax(-1) @ kc
    t0 = time.perf_counter()
    for _ in range(200): ((q @ kv.transpose(-1, -2)) / 8.0).softmax(-1) @ kv
    t1 = time.perf_counter()
    for _ in range(200): ((q @ kc.transpose(-1, -2)) / 8.0).softmax(-1) @ kc
    t2 = time.perf_counter()
    for _ in range(200): torch.cat([kc, q], 2)
    t3 = time.perf_counter()
    print('T', T, 'bitwise equal', torch.equal(a1, a2), 'maxdiff', float((a1 - a2).abs().max()), 'strided us', round((t1 - t0) / 200 * 1e6, 1), 'contig us', round((t2 - t1) / 200 * 1e6, 1), 'cat us', round((t3 - t2) / 200 * 1e6, 1))
