#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import torch

STAGES = 16
LINEAR_SUFFIXES = (
    ".attn.q.weight", ".attn.k.weight", ".attn.v.weight", ".attn.o.weight",
    ".experts.",
)
SHARED_QUANT = {"in_proj.weight", "out_proj.weight", "factor_head.weight"}


def sha256_file(path: Path, chunk: int = 16 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def quantize_weight(t: torch.Tensor) -> torch.Tensor:
    x = t.detach().cpu().float().contiguous()
    maxabs = float(x.abs().max())
    scale = max(maxabs / 127.0, 1e-8)
    q = torch.quantize_per_tensor(x, scale=scale, zero_point=0, dtype=torch.qint8)
    del x
    return q


def is_stage_quant(name: str) -> bool:
    return any(s in name for s in LINEAR_SUFFIXES)


def allheads_inference_payload(checkpoint: dict) -> dict | None:
    """Validate and retain the learned policy; never reconstruct it from defaults.

    Optimizer tensors belong to recovery checkpoints, not inference exports.
    Learned auxiliary weights, update counters and their saved policy config
    remain bitwise identical in the unquantized CPU inference payload.
    """
    import copy
    import math
    blob = checkpoint.get("allheads")
    if blob is None:
        if checkpoint.get("schema") == "agillm.gb10.1pf.recovery.v6":
            raise ValueError("v6 checkpoint is missing its learned allheads state")
        return None
    if not isinstance(blob, dict) or blob.get("schema") != "agillm.gb10.1pf.allheads.v3":
        raise ValueError("unsupported or malformed allheads export schema")
    aux = blob.get("aux")
    cfg = blob.get("config")
    shapes = {"shift_emb": (2, 256), "regret_updates": (),
              "stride_regret.0.weight": (256, 256), "stride_regret.0.bias": (256,),
              "stride_regret.2.weight": (2, 256), "stride_regret.2.bias": (2,)}
    if not isinstance(aux, dict) or set(aux) != set(shapes):
        raise ValueError("learned SAT-var tensor keys missing or unexpected")
    for key, shape in shapes.items():
        tensor = aux[key]
        if not torch.is_tensor(tensor) or tuple(tensor.shape) != shape:
            raise ValueError("learned SAT-var tensor geometry mismatch: " + key)
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("nonfinite learned SAT-var tensor: " + key)
        expected_dtype = torch.int64 if key == "regret_updates" else torch.float32
        if tensor.dtype != expected_dtype:
            raise ValueError("learned SAT-var tensor dtype mismatch: " + key)
    if int(aux["regret_updates"]) < 0:
        raise ValueError("negative learned SAT-var update count")
    if not isinstance(cfg, dict) or cfg.get("satvar_kmax") != 2 or cfg.get("nat_mask_id") != 2:
        raise ValueError("missing or inconsistent SAT-var/NAT inference contract")
    if cfg.get("nat_attn") not in ("causal", "bidir"):
        raise ValueError("invalid saved NAT attention mode")
    effective = cfg.get("effective")
    if not isinstance(effective, dict):
        raise ValueError("missing saved speed/quality decision configuration")
    lam, mu = effective.get("lam"), effective.get("mu")
    if (isinstance(lam, bool) or not isinstance(lam, (int, float))
            or not math.isfinite(lam) or lam <= 0
            or isinstance(mu, bool) or not isinstance(mu, (int, float))
            or not math.isfinite(mu) or mu < 0):
        raise ValueError("invalid saved speed reward or merge cost")
    counts = blob.get("objective_counts")
    if (not isinstance(counts, dict) or set(counts) != {"ar", "sat", "nat"}
            or any(type(v) is not int or v < 0 for v in counts.values())):
        raise ValueError("invalid saved objective update counts")
    if type(blob.get("sat_steps")) is not int or blob["sat_steps"] < 0:
        raise ValueError("invalid saved SAT step counter")
    json.dumps(cfg, allow_nan=False)
    return {"schema": blob["schema"],
            "aux": {k: v.detach().cpu().contiguous().clone() for k, v in aux.items()},
            "config": copy.deepcopy(cfg), "objective_counts": dict(counts),
            "sat_steps": blob["sat_steps"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source-sha256", default="")
    args = ap.parse_args()

    src = Path(args.source).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    source_sha = args.source_sha256.strip() or sha256_file(src)

    print(json.dumps({"event":"load_source", "path":str(src), "sha256":source_sha}), flush=True)
    ck = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    state = ck["model"]
    allheads = allheads_inference_payload(ck)
    meta = {
        "schema": "agillm.rpv16.cpu-int8-cache.v1",
        "source_schema": ck.get("schema"),
        "source_sha256": source_sha,
        "step": int(ck.get("step", 0)),
        "seen_tokens": int(ck.get("seen_tokens", 0)),
        "target_alignment": ck.get("target_alignment"),
        "parameter_count": int(sum(v.numel() for v in state.values() if torch.is_tensor(v))),
        "quantization": "dynamic-int8-per-tensor-linear; fp32 activations; bf16 embedding; fp32 norms/router/mix",
    }

    meta["allheads_export"] = {
        "present": allheads is not None,
        "policy_weights_retained": allheads is not None,
        "regret_updates": 0 if allheads is None else int(allheads["aux"]["regret_updates"]),
        "objective_counts": {} if allheads is None else allheads["objective_counts"],
        "policy_quantization": None if allheads is None else "none; exact-fp32",
        "runtime_availability": "export metadata is not proof of serving-mode availability",
    }
    shared = {"meta": meta}
    if allheads is not None:
        shared["allheads"] = allheads
    for name in ("embed.weight", "norm.weight", "mix_head.weight"):
        t = state[name]
        shared[name] = t.detach().cpu().to(torch.bfloat16 if name == "embed.weight" else torch.float32).contiguous()
    for name in ("in_proj.weight", "out_proj.weight", "factor_head.weight"):
        print(json.dumps({"event":"quantize", "name":name, "shape":list(state[name].shape)}), flush=True)
        shared[name] = quantize_weight(state[name])
    # Corrected-target checkpoints use a trained tied full-vocabulary head.
    if "tied_bias" in state:
        shared["tied_bias"] = state["tied_bias"].detach().cpu().float().contiguous()
        shared["tied_logit_scale"] = state["tied_logit_scale"].detach().cpu().float().reshape(()).contiguous()
        meta["classifier"] = "tied_full_vocab"
    torch.save(shared, out / "shared.pt")
    del shared
    gc.collect()

    for si in range(STAGES):
        prefix = f"stages.{si}."
        payload = {"meta": meta, "stage": si}
        names = [k for k in state.keys() if k.startswith(prefix)]
        for name in names:
            local = name[len(prefix):]
            t = state[name]
            if is_stage_quant(name):
                payload[local] = quantize_weight(t)
            else:
                payload[local] = t.detach().cpu().float().contiguous()
        dst = out / f"stage{si:02d}.pt"
        torch.save(payload, dst)
        print(json.dumps({"event":"stage_written", "stage":si, "path":str(dst), "bytes":dst.stat().st_size}), flush=True)
        del payload
        gc.collect()

    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    sums = []
    for p in [out / "shared.pt"] + [out / f"stage{i:02d}.pt" for i in range(STAGES)]:
        sums.append(f"{sha256_file(p)}  {p.name}")
    (out / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    (out / "CONVERSION_COMPLETE").write_text("ok\n")
    print(json.dumps({"event":"done", **meta}), flush=True)


if __name__ == "__main__":
    main()
