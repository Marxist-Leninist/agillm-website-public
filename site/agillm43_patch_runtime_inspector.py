#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, hashlib, json, os, re, tempfile
from pathlib import Path

PATCH_VERSION = "agillm43-logit-inspector-v1"
MARKER = "# AGILLM43_LOGIT_INSPECTOR_PATCH_V1"

HELPERS = r"""
# AGILLM43_LOGIT_INSPECTOR_PATCH_V1
def _inspector_token_text(tok, token_id):
    token_id = int(token_id)
    try:
        text = tok.decode([token_id], skip_special_tokens=True)
        if not text:
            text = tok.decode([token_id])
        return text or f"<id:{token_id}>"
    except Exception:
        return f"<id:{token_id}>"


def _inspector_top(logits, tok, selected_id, top_n=5):
    try:
        with torch.no_grad():
            flat = logits.reshape(-1).float()
            count = min(max(1, int(top_n)), int(flat.numel()))
            values, indices = torch.topk(flat, count)
            probs = torch.exp(values - torch.logsumexp(flat, dim=-1))
            return [
                {
                    "id": int(token_id),
                    "token": _inspector_token_text(tok, token_id),
                    "prob": float(prob),
                    "selected": int(token_id) == int(selected_id),
                }
                for prob, token_id in zip(probs.tolist(), indices.tolist())
            ]
    except Exception:
        return []


def _stream_token_payload(index_key, index, text, logits, selected_id, tok,
                          ordinal, total, every):
    payload = {str(index_key): int(index), "text": text}
    ordinal, total, every = int(ordinal), int(total), max(1, int(every))
    if ordinal == 1 or ordinal % every == 0 or ordinal >= total:
        payload["top"] = _inspector_top(logits, tok, selected_id, 5)
    return payload


"""

class PatchError(RuntimeError):
    pass

def exact(text, old, new, expected, label):
    count = text.count(old)
    if count != expected:
        raise PatchError(f"{label}: expected {expected}, found {count}")
    return text.replace(old, new)

def patch(text):
    if MARKER in text:
        ast.parse(text)
        return text, {"already_patched": 1}
    counts = {}
    anchor = "\ndef _sample(logits, T, top_k, top_p, min_p, greedy):"
    text = exact(text, anchor, "\n" + HELPERS + anchor, 1, "helper anchor")
    counts["helpers"] = 1

    text = exact(text,
"""            logits = _suppress_eos(logits, args)
            return _sample(logits, args.temperature, args.top_k, args.top_p, args.min_p, args.greedy)
""",
"""            logits = _suppress_eos(logits, args)
            nxt = _sample(logits, args.temperature, args.top_k, args.top_p, args.min_p, args.greedy)
            return nxt, logits
""", 1, "AR return")
    counts["ar_return"] = 1

    text, n = re.subn(
        r"(?m)^(\s*)expected = _ar_next_from_hidden\(([^\n]+)\)$",
        r"\1expected, _ = _ar_next_from_hidden(\2)", text)
    if n != 1:
        raise PatchError(f"AR verifier calls: expected 1, found {n}")
    counts["ar_verify"] = n

    text, n = re.subn(
        r"(?m)^(\s*)nxt = _ar_next_from_hidden\(([^\n]+)\)$",
        r"\1nxt, logits = _ar_next_from_hidden(\2)", text)
    if n != 2:
        raise PatchError(f"AR decode calls: expected 2, found {n}")
    counts["ar_calls"] = n

    text = exact(text,
"""                if stream:
                    print("[STREAM_AR] " + json.dumps({"i": int(ids.size(1) - prompt_len) - 1,
                                                       "text": tok.decode([int(nxt.item())], skip_special_tokens=True)}), flush=True)
""",
"""                if stream:
                    emitted = int(ids.size(1) - prompt_len)
                    print("[STREAM_AR] " + json.dumps(_stream_token_payload(
                        "i", emitted - 1,
                        tok.decode([int(nxt.item())], skip_special_tokens=True),
                        logits, int(nxt.item()), tok, emitted, args.max_new, 8
                    )), flush=True)
""", 2, "AR streams")
    counts["ar_streams"] = 2

    text = exact(text,
"""        passes = max(1, int(args.nat_passes))

        def _nat_history""",
"""        passes = max(1, int(args.nat_passes))
        nat_emitted = 0
        if stream:
            print("[STREAM_BEGIN] " + json.dumps({"mode": "nat", "slots": n_fill}), flush=True)

        def _nat_history""", 1, "NAT begin")
    counts["nat_begin"] = 1

    text = exact(text,
"""            nat_greedy = bool(getattr(args, "nat_greedy", True))
            return _sample(logits_pos, args.temperature, args.top_k, args.top_p, args.min_p, args.greedy or nat_greedy)
""",
"""            nat_greedy = bool(getattr(args, "nat_greedy", True))
            nxt = _sample(logits_pos, args.temperature, args.top_k, args.top_p, args.min_p, args.greedy or nat_greedy)
            return nxt, logits_pos
""", 1, "NAT return")
    counts["nat_return"] = 1

    text, n = re.subn(
        r"(?m)^(\s*)nxt = _nat_pick\(([^\n]+)\)$",
        r"\1nxt, logits_pos = _nat_pick(\2)", text)
    if n != 2:
        raise PatchError(f"NAT decode calls: expected 2, found {n}")
    counts["nat_calls"] = n

    text = exact(text,
"""                ids[0, pos] = int(nxt.reshape(-1)[0])
                remaining.discard(pos)
""",
"""                ids[0, pos] = int(nxt.reshape(-1)[0])
                nat_emitted += 1
                if stream:
                    print("[STREAM_NAT] " + json.dumps(_stream_token_payload(
                        "pos", int(pos) - prompt_len,
                        tok.decode([int(ids[0, pos])], skip_special_tokens=True),
                        logits_pos, int(ids[0, pos]), tok, nat_emitted, n_fill, 16
                    )), flush=True)
                remaining.discard(pos)
""", 1, "NAT primary stream")

    text = exact(text,
"""            for pos in sorted(remaining):
                nxt, logits_pos = _nat_pick(logits[:, pos, :], ids)
                ids[0, pos] = int(nxt.reshape(-1)[0])
""",
"""            for pos in sorted(remaining):
                nxt, logits_pos = _nat_pick(logits[:, pos, :], ids)
                ids[0, pos] = int(nxt.reshape(-1)[0])
                nat_emitted += 1
                if stream:
                    print("[STREAM_NAT] " + json.dumps(_stream_token_payload(
                        "pos", int(pos) - prompt_len,
                        tok.decode([int(ids[0, pos])], skip_special_tokens=True),
                        logits_pos, int(ids[0, pos]), tok, nat_emitted, n_fill, 16
                    )), flush=True)
""", 1, "NAT final stream")
    counts["nat_streams"] = 2

    text = exact(text,
"""    else:
        cached_len = ids.size(1)
""",
"""    else:
        if stream:
            print("[STREAM_BEGIN] " + json.dumps({"mode": "sat", "slots": int(args.max_new)}), flush=True)
        cached_len = ids.size(1)
""", 1, "SAT begin")
    counts["sat_begin"] = 1

    text = exact(text,
"""            ids = torch.cat([ids, nxt], 1)
            added += 1
            sat_prompt_realign = True
""",
"""            ids = torch.cat([ids, nxt], 1)
            added += 1
            if stream:
                print("[STREAM_SAT] " + json.dumps(_stream_token_payload(
                    "i", added - 1,
                    tok.decode([int(nxt.item())], skip_special_tokens=True),
                    logits, int(nxt.item()), tok, added, args.max_new, 8
                )), flush=True)
            sat_prompt_realign = True
""", 1, "SAT alignment stream")

    text = exact(text,
"""                new_tokens.append(nxt)
                ids = torch.cat([ids, nxt], 1)
                added += 1
                if EOS is not None""",
"""                new_tokens.append(nxt)
                ids = torch.cat([ids, nxt], 1)
                added += 1
                if stream:
                    print("[STREAM_SAT] " + json.dumps(_stream_token_payload(
                        "i", added - 1,
                        tok.decode([int(nxt.item())], skip_special_tokens=True),
                        logits, int(nxt.item()), tok, added, args.max_new, 8
                    )), flush=True)
                if EOS is not None""", 1, "SAT block stream")
    counts["sat_streams"] = 2
    ast.parse(text)
    return text, counts

def atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(name, 0o644)
        os.replace(name, path)
    except Exception:
        try: os.unlink(name)
        except FileNotFoundError: pass
        raise

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args()
    raw = args.input.resolve(strict=True).read_bytes()
    out_text, counts = patch(raw.decode())
    out = out_text.encode()
    ast.parse(out_text)
    if not args.check_only:
        atomic_write(args.output, out)
    print(json.dumps({
        "ok": True, "patch_version": PATCH_VERSION,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "output_sha256": hashlib.sha256(out).hexdigest(),
        "replacements": counts,
    }, sort_keys=True))

if __name__ == "__main__":
    main()
