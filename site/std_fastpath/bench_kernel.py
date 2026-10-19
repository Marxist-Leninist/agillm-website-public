import os, time, json, statistics, gc, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch.ao.quantization import quantize_dynamic
from torch.utils.cpp_extension import load
avail = int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) * 1024
assert avail > 5e9, f'not enough memory available: {avail/1e9:.1f} GB'
HERE = '/opt/agillm43_netserve/std_fastpath'
t0 = time.time()
ext = load(name='agillm_f16gemv', sources=[f'{HERE}/f16gemv.cpp'], build_directory=f'{HERE}/build',
           extra_cflags=['-O3', '-mavx512f', '-mf16c', '-mfma', '-fopenmp'], extra_ldflags=['-fopenmp'], verbose=False)
print('build+load s', round(time.time() - t0, 1), flush=True)
torch.set_num_threads(int(os.environ.get('THREADS', '8')))
torch.manual_seed(0)
L = 28
packed, mine = [], []
with torch.inference_mode():
    for i in range(L):
        ex = nn.Sequential(nn.Linear(1280, 5120), nn.ReLU(), nn.Linear(5120, 1280))
        sh = nn.Sequential(nn.Linear(1280, 2560), nn.ReLU(), nn.Linear(2560, 1280))
        for m in (ex, sh):
            for p in m.parameters(): p.mul_(0.5)
        mine.append(tuple(t.contiguous() for t in (ex[0].weight.half(), ex[0].bias.float(), ex[2].weight.half(), ex[2].bias.float(),
                                                    sh[0].weight.half(), sh[0].bias.float(), sh[2].weight.half(), sh[2].bias.float())))
        packed.append((quantize_dynamic(ex, {nn.Linear}, dtype=torch.float16), quantize_dynamic(sh, {nn.Linear}, dtype=torch.float16)))
        del ex, sh
    gc.collect()
    x = torch.randn(1, 1280)
    def run_fbgemm():
        out = []
        for e, s in packed:
            out.append(e(x) + s(x))
        return out
    def run_mine():
        return [ext.mlp2(*w, x) for w in mine]
    def run_mine_sep():
        return [ext.mlp(w[0], w[1], w[2], w[3], x) + ext.mlp(w[4], w[5], w[6], w[7], x) for w in mine]
    a = run_fbgemm(); b = run_mine(); c = run_mine_sep()
    rel = max(float((p - q).abs().max() / p.abs().max()) for p, q in zip(a, b))
    rel2 = max(float((p - q).abs().max() / p.abs().max()) for p, q in zip(b, c))
    print('max rel diff fbgemm vs mine', rel, 'mlp2 vs mlp+mlp', rel2, flush=True)
    res = {'fbgemm': [], 'mine': [], 'mine_sep': []}
    for rep in range(12):
        for name, fn in (('fbgemm', run_fbgemm), ('mine', run_mine), ('mine_sep', run_mine_sep)):
            t = time.perf_counter(); fn(); res[name].append((time.perf_counter() - t) * 1e3)
    print(json.dumps({k: round(statistics.median(v), 2) for k, v in res.items()}), 'ms per 28-layer token (routed+shared)', flush=True)
    # head: fp32 MKL vs fp16 kernel
    del packed; gc.collect()
    W = torch.randn(129280, 1280) * 0.02
    W16 = W.half().contiguous()
    h = torch.randn(1, 1280)
    hr = {'fp32_mkl': [], 'fp16_mine': []}
    for rep in range(10):
        t = time.perf_counter(); z1 = F.linear(h, W); hr['fp32_mkl'].append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter(); z2 = ext.gemv(W16, h, None, False); hr['fp16_mine'].append((time.perf_counter() - t) * 1e3)
    print('head', json.dumps({k: round(statistics.median(v), 2) for k, v in hr.items()}), 'maxabs', float((z1 - z2).abs().max()), flush=True)
    # attention-sized fp32 vs fp16 (qkv 2560x1280 + proj 1280x1280) x 28, cache-cold rotation
    A32 = [(torch.randn(2560, 1280) * 0.02, torch.randn(1280, 1280) * 0.02) for _ in range(L)]
    A16 = [(q.half().contiguous(), p.half().contiguous()) for q, p in A32]
    ar = {'fp32_mkl': [], 'fp16_mine': []}
    for rep in range(10):
        t = time.perf_counter()
        for q, p in A32: F.linear(h, q); F.linear(h, p)
        ar['fp32_mkl'].append((time.perf_counter() - t) * 1e3)
        t = time.perf_counter()
        for q, p in A16: ext.gemv(q, h, None, False); ext.gemv(p, h, None, False)
        ar['fp16_mine'].append((time.perf_counter() - t) * 1e3)
    print('attn28', json.dumps({k: round(statistics.median(v), 2) for k, v in ar.items()}), flush=True)
