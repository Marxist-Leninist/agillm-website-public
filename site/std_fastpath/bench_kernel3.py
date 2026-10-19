import os, time, json, statistics, gc
import torch, torch.nn as nn, torch.nn.functional as F
from torch.ao.quantization import quantize_dynamic
from torch.utils.cpp_extension import load
import warnings; warnings.simplefilter('ignore')
avail = int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) * 1024
assert avail > 4e9, f'not enough memory available: {avail/1e9:.1f} GB'
HERE = '/opt/agillm43_netserve/std_fastpath'
os.makedirs(f'{HERE}/build_dev2', exist_ok=True)
t0 = time.time()
ext = load(name='agillm_std_kernels_dev2', sources=[f'{HERE}/std_kernels2.cpp'], build_directory=f'{HERE}/build_dev2',
           extra_cflags=['-O3', '-mavx512f', '-mf16c', '-mfma', '-fopenmp'], extra_ldflags=['-fopenmp'], verbose=False)
print('build+load s', round(time.time() - t0, 1), flush=True)
torch.set_num_threads(8)
torch.manual_seed(0)
L = 12
with torch.inference_mode():
    packed, mine = [], []
    for i in range(L):
        ex = nn.Sequential(nn.Linear(1280, 5120), nn.ReLU(), nn.Linear(5120, 1280))
        for p in ex.parameters(): p.mul_(0.5)
        assert torch.equal(ext.unpack(ext.pack(ex[0].weight)), ex[0].weight.half())
        mine.append((ext.pack(ex[0].weight), ex[0].bias.float().contiguous(), ext.pack(ex[2].weight), ex[2].bias.float().contiguous()))
        packed.append(quantize_dynamic(ex, {nn.Linear}, dtype=torch.float16))
        del ex
    gc.collect()
    for M in (1, 2, 3, 64):
        x = torch.randn(M, 1280)
        a = [p(x) for p in packed]; b = [ext.mlp(*w, x) for w in mine]
        rel = max(float((p - q).abs().max() / p.abs().max()) for p, q in zip(a, b))
        tf, tm = [], []
        reps = 6 if M <= 16 else 3
        for rep in range(reps):
            t = time.perf_counter(); [p(x) for p in packed]; tf.append(time.perf_counter() - t)
            t = time.perf_counter(); [ext.mlp(*w, x) for w in mine]; tm.append(time.perf_counter() - t)
        print(M, json.dumps({'fbgemm_ms_per_mlp': round(statistics.median(tf) / L * 1e3, 3), 'v2_ms_per_mlp': round(statistics.median(tm) / L * 1e3, 3), 'rel_diff': rel}), flush=True)
    # routed+shared single-row fused, 28 layers
    sh = []
    for i in range(L):
        s = nn.Sequential(nn.Linear(1280, 2560), nn.ReLU(), nn.Linear(2560, 1280))
        sh.append((quantize_dynamic(s, {nn.Linear}, dtype=torch.float16), (ext.pack(s[0].weight), s[0].bias.float().contiguous(), ext.pack(s[2].weight), s[2].bias.float().contiguous())))
    x = torch.randn(1, 1280)
    ref = [packed[i](x) + sh[i][0](x) for i in range(L)]
    got = [ext.mlp2(*mine[i], *sh[i][1], x) for i in range(L)]
    rel = max(float((p - q).abs().max() / p.abs().max()) for p, q in zip(ref, got))
    tf, tm = [], []
    for rep in range(8):
        t = time.perf_counter(); [packed[i](x) + sh[i][0](x) for i in range(L)]; tf.append(time.perf_counter() - t)
        t = time.perf_counter(); [ext.mlp2(*mine[i], *sh[i][1], x) for i in range(L)]; tm.append(time.perf_counter() - t)
    print('routed+shared per layer', json.dumps({'fbgemm_ms': round(statistics.median(tf) / L * 1e3, 3), 'v2_mlp2_ms': round(statistics.median(tm) / L * 1e3, 3), 'rel_diff': rel}), flush=True)
    # head 129280 x 1280
    del packed, sh; gc.collect()
    W = torch.randn(129280, 1280) * 0.02
    Pk = ext.pack(W); h = torch.randn(1, 1280)
    z1 = F.linear(h, W); z2 = ext.linear(Pk, h, None, False)
    th, tp = [], []
    for rep in range(8):
        t = time.perf_counter(); F.linear(h, W); th.append(time.perf_counter() - t)
        t = time.perf_counter(); ext.linear(Pk, h, None, False); tp.append(time.perf_counter() - t)
    print('head', json.dumps({'fp32_mkl_ms': round(statistics.median(th) * 1e3, 2), 'v2_fp16_ms': round(statistics.median(tp) * 1e3, 2), 'maxabs': float((z1 - z2).abs().max())}), flush=True)
