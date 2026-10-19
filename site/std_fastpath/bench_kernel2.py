import os, time, json, statistics, gc
import torch, torch.nn as nn, torch.nn.functional as F
from torch.ao.quantization import quantize_dynamic
from torch.utils.cpp_extension import load
import warnings; warnings.simplefilter('ignore')
avail = int(open('/proc/meminfo').read().split('MemAvailable:')[1].split()[0]) * 1024
assert avail > 4e9, f'not enough memory available: {avail/1e9:.1f} GB'
HERE = '/opt/agillm43_netserve/std_fastpath'
t0 = time.time()
ext = load(name='agillm_std_kernels_dev', sources=[f'{HERE}/std_kernels.cpp'], build_directory=f'{HERE}/build_dev',
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
        mine.append((ex[0].weight.half().contiguous(), ex[0].bias.float().contiguous(), ex[2].weight.half().contiguous(), ex[2].bias.float().contiguous()))
        packed.append(quantize_dynamic(ex, {nn.Linear}, dtype=torch.float16))
        del ex
    gc.collect()
    out = {}
    for M in (1, 2, 8, 16, 32, 64, 256):
        x = torch.randn(M, 1280)
        a = [p(x) for p in packed]; b = [ext.mlp(*w, x) for w in mine]
        rel = max(float((p - q).abs().max() / p.abs().max()) for p, q in zip(a, b))
        tf, tm, tc = [], [], []
        reps = 6 if M <= 16 else 3
        for rep in range(reps):
            t = time.perf_counter(); [p(x) for p in packed]; tf.append(time.perf_counter() - t)
            t = time.perf_counter(); [ext.mlp(*w, x) for w in mine]; tm.append(time.perf_counter() - t)
            if M >= 16:
                t = time.perf_counter()
                for w in mine:
                    h = F.relu(F.linear(x, w[0].float(), w[1])); F.linear(h, w[2].float(), w[3])
                tc.append(time.perf_counter() - t)
        r = {'fbgemm_ms_per_mlp': round(statistics.median(tf) / L * 1e3, 3), 'mine_ms_per_mlp': round(statistics.median(tm) / L * 1e3, 3), 'rel_diff': rel}
        if tc: r['convert_mkl_ms_per_mlp'] = round(statistics.median(tc) / L * 1e3, 3)
        out[M] = r
        print(M, json.dumps(r), flush=True)
