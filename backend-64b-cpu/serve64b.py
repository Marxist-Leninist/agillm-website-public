"""serve64b.py (claude-chat-64b-improve-20261019): CPU inference + streaming HTTP for the bin8 64B snapshot.
Model maths = tern64b_sample.TernModel (claude-code-64b-sample's mirror of terntf), with every body linear and the tied
table readout replaced by libbgemm (1-bit weights x int8 activations, exact vs the int8 reference). Weights are kept
packed: 8.05 GB of body signs + 0.2 GB of table signs. GET /health, POST /generate {prompt, max_new, temperature,
top_p, top_k} -> NDJSON stream {"t": text} ... {"done": true}. One generation at a time; listens on 127.0.0.1 only."""
import ctypes, importlib.util, json, os, sys, threading, time
import numpy as np, torch
C = '/root/claude64b_cpuinfer'
SNAP = os.environ.get('SNAP64', '/workspace/claude64b_infer/snap')
BITS = os.environ.get('BITS64', '/workspace/claude64b_infer/bits')
PORT = int(os.environ.get('PORT64', '8770'))
spec = importlib.util.spec_from_file_location('tsamp', f'{C}/tern64b_sample.py'); ts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ts)
lib = ctypes.CDLL(f'{C}/libbgemm.so')
lib.bgemm.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
lib.bgemm.restype = None
torch.set_num_threads(int(os.environ.get('TORCH_THREADS64', '4')))


def bmm(packed, rowpop, x):
    shp = x.shape; X = np.ascontiguousarray(x.reshape(-1, shp[-1]).float().numpy(), dtype=np.float32)
    N, n = X.shape; rows = packed.shape[0]; Y = np.empty((N, rows), np.float32)
    lib.bgemm(packed.ctypes.data, rowpop.ctypes.data, rows, n, X.ctypes.data, N, Y.ctypes.data)
    return torch.from_numpy(Y).view(*shp[:-1], rows)


class Fast(ts.TernModel):
    def __init__(self, snap_dir, log=print):
        super().__init__(snap_dir, 'cpu', check=True, log=log)
        os.makedirs(BITS, exist_ok=True)
        names = [n for n, _ in sorted(self.off.items(), key=lambda kv: kv[1][0])]
        idx_p, bin_p = os.path.join(BITS, 'index.json'), os.path.join(BITS, 'bits.bin')
        if not os.path.exists(idx_p):
            t0, off, index = time.time(), 0, {}
            with open(bin_p + '.tmp', 'wb') as f:
                for k, name in enumerate(names + ['table']):
                    w = self.table if name == 'table' else self.w(name)
                    b = (w > 0).numpy()
                    pk = np.packbits(b, axis=1, bitorder='little')
                    rp = b.sum(1, dtype=np.int64)
                    f.write(pk.tobytes()); f.write(rp.tobytes())
                    index[name] = dict(off=off, rows=int(b.shape[0]), n=int(b.shape[1]))
                    off += pk.nbytes + rp.nbytes
                    if k % 40 == 0: log(f'[pack] {k}/{len(names) + 1} {name} {time.time() - t0:.0f}s')
            os.replace(bin_p + '.tmp', bin_p)
            json.dump(dict(step=self.meta.get('step'), index=index), open(idx_p + '.tmp', 'w')); os.replace(idx_p + '.tmp', idx_p)
            log(f'[pack] done in {time.time() - t0:.0f}s')
        meta = json.load(open(idx_p)); mm = np.memmap(bin_p, dtype=np.uint8, mode='r')
        self.bits = {}
        for name, e in meta['index'].items():
            nb = e['rows'] * e['n'] // 8
            pk = np.ascontiguousarray(mm[e['off']:e['off'] + nb]).view(np.uint64).reshape(e['rows'], e['n'] // 64)
            rp = np.ascontiguousarray(mm[e['off'] + nb:e['off'] + nb + 8 * e['rows']]).view(np.int64)
            self.bits[name] = (pk, rp)
        del mm
        self.step = meta.get('step')
        log(f'[bits] {len(self.bits)} packed matrices in RAM, step {self.step}')

    def lin(self, x, name, scale):
        pk, rp = self.bits[name]
        return bmm(pk, rp, x) * scale

    def logits(self, h):
        pk, rp = self.bits['table']
        return bmm(pk, rp, h) * self.tscale


def stream(M, enc, prompt, max_new=48, temp=0.8, top_p=0.95, top_k=0, eos=None, ban=None, seed=None):
    ids = enc.encode(prompt, disallowed_special=()) or [eos]
    ids = ids[-512:]
    Lm, Tmax, hd = len(ids), len(ids) + max_new, M.d // M.H
    kv = [(torch.empty(1, M.H, Tmax, hd), torch.empty(1, M.H, Tmax, hd)) for _ in range(M.L)]
    g = torch.Generator().manual_seed(seed if seed is not None else int(time.time()))
    t = torch.arange(Lm)
    allowed = (t[None, :, None] >= t[None, None, :])
    out, prev = [], ''
    with torch.no_grad():
        h = M.forward(torch.tensor([ids]), t[None], allowed[:, None], kv, 0)
        used, last, cur = Lm, h[:, -1], Lm
        for step in range(max_new):
            l = M.logits(last)[0]
            if ban is not None: l = l.masked_fill(ban, -float('inf'))
            if temp <= 0: nxt = int(l.argmax())
            else:
                l = l / temp
                if top_k: v = torch.topk(l, top_k); l = torch.full_like(l, -float('inf')).scatter_(0, v.indices, v.values)
                pr = torch.softmax(l.float(), 0)
                if top_p < 1.0:
                    sp, si = torch.sort(pr, descending=True); sp[(sp.cumsum(0) - sp) > top_p] = 0
                    pr = torch.zeros_like(pr).scatter_(0, si, sp); pr = pr / pr.sum()
                nxt = int(torch.multinomial(pr, 1, generator=g))
            if eos is not None and nxt == eos: break
            out.append(nxt)
            txt = ts.safe_decode(enc, out)
            if len(txt) > len(prev): yield txt[len(prev):]; prev = txt
            if step == max_new - 1: break
            m = torch.ones(1, 1, 1, used + 1, dtype=torch.bool)
            h = M.forward(torch.tensor([[nxt]]), torch.tensor([[cur]]), m, kv, used)
            used, cur, last = used + 1, cur + 1, h[:, -1]


def main():
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    t0 = time.time(); M = Fast(SNAP); tk = ts.tokenizer('o200k_base')
    enc = tk[0] if isinstance(tk, tuple) else tk
    eos = tk[1] if isinstance(tk, tuple) and len(tk) > 1 and isinstance(tk[1], int) else getattr(enc, 'eot_token', None)
    ban = ts.banned_ids(enc, eos, M.V)
    ban = ban if isinstance(ban, torch.Tensor) else None
    lock = threading.Lock(); state = dict(busy=False, served=0, last_tok_s=None, ready_s=round(time.time() - t0, 1))
    print(f'[serve64b] ready in {state["ready_s"]}s on 127.0.0.1:{PORT}, step {M.step}', flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a): pass
        def _json(self, code, obj):
            b = json.dumps(obj).encode(); self.send_response(code); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
        def do_GET(self):
            if self.path.rstrip('/').endswith('/health'):
                return self._json(200, dict(ok=True, model='AGILLM-64B bin8', step=M.step, **state))
            self._json(404, dict(error='not found'))
        def do_POST(self):
            if not self.path.rstrip('/').endswith('/generate'): return self._json(404, dict(error='not found'))
            try: req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0') or 0)) or b'{}')
            except Exception: return self._json(400, dict(error='bad json'))
            if not lock.acquire(blocking=False): return self._json(503, dict(error='busy: one 64B generation at a time', busy=True))
            try:
                state['busy'] = True
                self.send_response(200); self.send_header('Content-Type', 'application/x-ndjson'); self.send_header('Cache-Control', 'no-cache')
                self.send_header('X-Accel-Buffering', 'no'); self.end_headers()
                n, ts0 = 0, time.time()
                mx = max(1, min(int(req.get('max_new', 48)), 160))
                for piece in stream(M, enc, str(req.get('prompt', ''))[:4000], mx, float(req.get('temperature', 0.8)),
                                    float(req.get('top_p', 0.95)), int(req.get('top_k', 0)), eos, ban):
                    n += 1; self.wfile.write((json.dumps(dict(t=piece)) + '\n').encode()); self.wfile.flush()
                dt = time.time() - ts0; state['served'] += 1; state['last_tok_s'] = round(n / max(dt, 1e-9), 2)
                self.wfile.write((json.dumps(dict(done=True, pieces=n, seconds=round(dt, 1), step=M.step)) + '\n').encode()); self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError): pass
            finally: state['busy'] = False; lock.release()
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()


if __name__ == '__main__':
    main()
