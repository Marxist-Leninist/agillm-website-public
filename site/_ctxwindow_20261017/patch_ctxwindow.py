#!/usr/bin/env python3
"""Pop-out 49.1M context-window fix (2026-10-17).

Long chat/Continue prompts made every pop-out mode fail with
  prompt (N) + requested output (M) exceeds context 1536
Worker now keeps the newest prompt tokens that fit (sliding window) and reports
prompt_truncated / prompt_tokens / original_prompt_tokens; the native backend
folds that receipt into stats so agillm-ui shows "prompt shortened: used last N of M".
Anchored exact-text replacements, each asserted unique; timestamped backups.
"""
import py_compile, shutil, sys, time
TS = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
BASE = "/opt/agillm43_netserve/popup_multimode_next"
WORKERS = [f"{BASE}/operator_candidate_20260902/agillm44_operator_jsonl_server.py",
           f"{BASE}/satvar_cost_20260912/agillm44_operator_jsonl_server.py"]
BACKENDS = [f"{BASE}/operator_candidate_20260902/agillm44_native_jsonl_backend.py",
            f"{BASE}/satvar_cost_20260912/agillm44_native_jsonl_backend.py"]

WORKER_EDITS = [
("""def emit_result(loaded, request, worker_mode):
""",
"""PROMPT_WINDOW_MARGIN=8      # re-tokenisation slack at the cut point
PROMPT_WINDOW_MIN_KEEP=32   # never window the prompt below this; shrink max_new first

def fit_prompt_to_context(loaded, prompt, max_new, max_context):
    \"\"\"Keep the newest prompt tokens so prompt + max_new fits the fixed context.

    The 49.1M pop-out has a hard window (1536 tokens). A chat or Continue session
    grows past it and every runtime mode raised instead of generating. The newest
    tokens carry the conditioning that matters, so drop from the front and report
    exactly what happened. prompt_truncated / prompt_tokens / original_prompt_tokens
    are the keys the agillm-ui status line already renders.
    \"\"\"
    tok=loaded.tokenizer
    ids=tok.encode(prompt,add_special_tokens=True)
    original=len(ids)
    reserve=max_new+2   # NAT reserves one slot beyond max_new
    window={'prompt_truncated':False,'prompt_tokens':original,'original_prompt_tokens':original,
            'prompt_tokens_dropped':0,'context_window':max_context,'effective_max_new':max_new}
    if original+reserve<=max_context:
        return prompt,window
    keep=max_context-reserve-PROMPT_WINDOW_MARGIN
    if keep<PROMPT_WINDOW_MIN_KEEP:
        max_new=max(1,max_context-PROMPT_WINDOW_MIN_KEEP-2-PROMPT_WINDOW_MARGIN)
        reserve=max_new+2
        keep=max_context-reserve-PROMPT_WINDOW_MARGIN
        window['effective_max_new']=max_new
        window['max_new_clamped']=True
    for _ in range(10):
        keep=max(1,min(keep,original))
        text=tok.decode(ids[-keep:],skip_special_tokens=True).lstrip('\\ufffd')
        fitted=tok.encode(text,add_special_tokens=True)
        if len(fitted)+reserve<=max_context:
            window.update(prompt_truncated=True,prompt_tokens=len(fitted),
                          prompt_tokens_dropped=REDACTED
            return text,window
        keep-=(len(fitted)+reserve-max_context)+4
    raise ValueError(f'prompt ({original}) could not be windowed into context {max_context} with {max_new} new tokens')

def emit_result(loaded, request, worker_mode):
"""),
("""    incremental_stream = mode in {'ar','sat_fixed','sat_variable'}
    requested_tokens=int(request.get('max_new',64))
""",
"""    incremental_stream = mode in {'ar','sat_fixed','sat_variable'}
    requested_tokens=int(request.get('max_new',64))
    max_context=int(request.get('max_context',1536))
    # Sliding context window (2026-10-17): see fit_prompt_to_context.
    prompt,prompt_window=fit_prompt_to_context(loaded,prompt,requested_tokens,max_context)
    requested_tokens=int(prompt_window['effective_max_new'])
    min_new_tokens=min(int(request.get('min_new',0)),requested_tokens)
"""),
("""        print('[STREAM_BEGIN] '+json.dumps({'mode':mode,'tokens':requested_tokens,'incremental':True},separators=(',',':')),flush=True)
""",
"""        print('[STREAM_BEGIN] '+json.dumps({'mode':mode,'tokens':requested_tokens,'incremental':True,'prompt_window':prompt_window},separators=(',',':')),flush=True)
"""),
("""        max_new_tokens=requested_tokens,
        min_new_tokens=int(request.get('min_new',0)),
        max_context=int(request.get('max_context',1536)),
""",
"""        max_new_tokens=requested_tokens,
        min_new_tokens=min_new_tokens,
        max_context=max_context,
"""),
("""    print('[decode-telemetry] '+json.dumps(telemetry,separators=(',',':'),sort_keys=True),flush=True)
""",
"""    print('[prompt-window] '+json.dumps(prompt_window,separators=(',',':'),sort_keys=True),flush=True)
    print('[decode-telemetry] '+json.dumps(telemetry,separators=(',',':'),sort_keys=True),flush=True)
"""),
]

BACKEND_EDITS = [
("""        legacy_stats: dict[str, Any] | None = None
        telemetry: Mapping[str, Any] | None = None
""",
"""        legacy_stats: dict[str, Any] | None = None
        telemetry: Mapping[str, Any] | None = None
        prompt_window: dict[str, Any] = {}
"""),
("""            match = _STAT_RE.search(text)
            if match:
""",
"""            if text.startswith("[prompt-window] "):
                # Context-window receipt (2026-10-17): prompt_truncated,
                # prompt_tokens, original_prompt_tokens, context_window...
                window = marker_json(text, "[prompt-window] ")
                if isinstance(window, Mapping):
                    prompt_window = {str(key): value for key, value in window.items()}
                continue
            match = _STAT_RE.search(text)
            if match:
"""),
("""        stats = dict(legacy_stats or {})
        stats["telemetry"] = validated
""",
"""        stats = dict(legacy_stats or {})
        if prompt_window:
            # The UI renders "prompt shortened: used last N of M tokens" from these.
            stats.update(prompt_window)
        stats["telemetry"] = validated
"""),
]

def patch(path, edits):
    src = open(path, encoding="utf-8").read()
    if "fit_prompt_to_context" in src or "[prompt-window] " in src:
        print("ALREADY PATCHED", path); return
    for old, new in edits:
        n = src.count(old)
        assert n == 1, f"{path}: anchor count {n} for {old[:60]!r}"
        src = src.replace(old, new)
    shutil.copy2(path, f"{path}.bak-ctxwindow-{TS}")
    open(path, "w", encoding="utf-8").write(src)
    py_compile.compile(path, doraise=True)
    print("PATCHED", path)

for p in WORKERS: patch(p, WORKER_EDITS)
for p in BACKENDS: patch(p, BACKEND_EDITS)
print("OK", TS)
