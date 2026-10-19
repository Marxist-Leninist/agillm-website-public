import torch, time
torch.set_num_threads(8)
q = torch.randn(1, 20, 1, 64)
for T in (200, 600, 1200, 2000):
    buf = torch.randn(1, 20, 4096, 64)
    kv = buf[:, :, :T]; kc = kv.contiguous(); new = torch.randn(1, 20, 1, 64)
    def attn(k): return ((q @ k.transpose(-1, -2)) / 8.0).softmax(-1) @ k
    for f in (lambda: attn(kv), lambda: attn(kc), lambda: torch.cat([kc, new], 2), lambda: buf[:, :, T:T+1].copy_(new)):
        f()
    res = {}
    for name, f in (('attn_strided', lambda: attn(kv)), ('attn_contig', lambda: attn(kc)),
                    ('cat_one', lambda: torch.cat([kc, new], 2)), ('append_inplace', lambda: buf[:, :, T:T+1].copy_(new))):
        ts = []
        for rep in range(5):
            t = time.perf_counter()
            for _ in range(100): f()
            ts.append((time.perf_counter() - t) / 100 * 1e6)
        res[name] = round(sorted(ts)[2], 1)
    old = res['attn_contig'] + 2 * res['cat_one']; newc = res['attn_strided'] + res['append_inplace']
    print('T', T, res, 'per-layer old(cat k,v + attn) us', round(old, 1), 'new(tied append + strided attn) us', round(newc, 1), 'x28 saving ms', round((old - newc) * 28 / 1000, 2), flush=True)
