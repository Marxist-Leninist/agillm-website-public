"""Bounded CPU-only frozen-checkpoint inference canary; never touches trainer."""
import argparse
import json
import os
import queue
import subprocess
import threading
import time

p = argparse.ArgumentParser()
p.add_argument("--runtime", required=True)
p.add_argument("--checkpoint", required=True)
a = p.parse_args()
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
           OPENBLAS_NUM_THREADS="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
cmd = ["/opt/conda/bin/python", "-u", a.runtime, "infer", "--server", "--device", "cpu",
       "--cpu_threads", "4", "--cpu_interop_threads", "1", "--ckpt", a.checkpoint,
       "--mode", "ar", "--sampler", "dblock", "--dblock_blocks", "14",
       "--dblock_infer_blocks", "0", "--output_row_norm_clip", "10", "--max_new", "32",
       "--min_new", "0", "--temperature", "0.7", "--top_p", "1.0", "--greedy",
       "--plain-output", "--repetition_penalty", "1.05", "--presence_penalty", "0",
       "--frequency_penalty", "0", "--penalty_last_n", "128", "--infer_dtype", "fp32",
       "--ar_head_pack", "fp16", "--ar_head_rerank_k", "32"]
child = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
q = queue.Queue()
def reader():
    for line in child.stdout:
        q.put(line.rstrip("\n"))
    q.put(None)
threading.Thread(target=reader, daemon=True).start()

def until(marker, timeout):
    deadline = time.monotonic() + timeout
    lines = []
    while True:
        line = q.get(timeout=max(.01, deadline - time.monotonic()))
        if line is None:
            raise RuntimeError("Canary process exited: " + str(child.poll()))
        print(line, flush=True)
        lines.append(line)
        if marker in line:
            return lines
        if time.monotonic() >= deadline:
            raise TimeoutError(marker)

try:
    until("[INFER_SERVER_READY]", 240)
    cases = [
        ("ar_exact", {"mode": "ar", "sampler": "dblock", "max_new": 24,
                      "min_new": 24, "ignore_eos": True, "length_mode": "exact"}),
        ("ar_natural_long", {"mode": "ar", "sampler": "dblock", "max_new": 256}),
        ("ar_fullstack", {"mode": "ar", "sampler": "ar", "max_new": 24}),
        ("sat_fixed", {"mode": "sat", "sampler": "dblock", "var": False, "max_new": 16}),
        ("sat_variable", {"mode": "sat", "sampler": "dblock", "var": True, "max_new": 16}),
        ("nat", {"mode": "nat", "sampler": "ar", "nat_passes": 2,
                 "nat_ar_verify": False, "max_new": 16}),
    ]
    for name, options in cases:
        req = dict(prompt="The capital of France is", greedy=True, min_new=0,
                   ignore_eos=False, length_mode="natural", repetition_guard=True,
                   dblock_infer_blocks="0", stream=False)
        req.update(options)
        print("CANARY_CASE " + json.dumps({"case": name, "request": req}), flush=True)
        child.stdin.write(json.dumps(req) + "\n")
        child.stdin.flush()
        lines = until("[INFER_SERVER_RESULT_END]", 90)
        if any("[INFER_SERVER_ERROR]" in line for line in lines):
            raise RuntimeError("Generation failed: " + name)
        print("CANARY_CASE_OK " + name, flush=True)
    print("CANARY_ALL_OK", flush=True)
finally:
    if child.poll() is None:
        try:
            child.stdin.write('{"cmd":"quit"}\n')
            child.stdin.flush()
            child.wait(timeout=10)
        except (BrokenPipeError, subprocess.TimeoutExpired):
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
