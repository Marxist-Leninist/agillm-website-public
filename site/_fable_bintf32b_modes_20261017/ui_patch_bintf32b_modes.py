#!/usr/bin/env python3
"""Enable SAT fixed / SAT variable / NAT for Binary 32B in agillm_completion.html.

Exact-anchor rewrite; every anchor must match exactly once or nothing is written.
The :8513 worker (agillm32b_server_learned, runtime agillm32b-learned-satvar12-serving-v6)
advertises canonical_modes ar/nat/sat_fixed/sat_variable; the page hard-coded ["ar"].
"""
import hashlib, shutil, sys, time
from pathlib import Path

PATH = Path('/opt/agillm43_netserve/agillm_completion.html')
STAMP = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
BACKUP = PATH.with_name(PATH.name + f'.bak-bintf32b-modes-{STAMP}')

EDITS = [
    # 1. profileModeKeys: 32B serves the standard four modes (no diffusion).
    (
        '    if (profileKey === "bintf32b") return ["ar"];\n',
        '    /* BINTF32B_MULTIMODE_UI_20261017: the :8513 worker serves AR, SAT fixed (block 2/3), SAT variable (learned {1,2}) and one-pass NAT. */\n'
        '    if (profileKey === "bintf32b") return STANDARD_MODE_ORDER.slice();\n',
    ),
    # 2. MODEL_PROFILES.bintf32b.modeKeys
    (
        '  bintf32b:Object.freeze({ key:"bintf32b", modeKeys:Object.freeze(["ar"]), short:"Binary 32B",',
        '  bintf32b:Object.freeze({ key:"bintf32b", modeKeys:STANDARD_MODE_KEYS, short:"Binary 32B",',
    ),
    # 3. summary text
    (
        'summaryDetail:"Experimental text completion · 1-bit · cached AR",',
        'summaryDetail:"Experimental text completion · 1-bit · AR / SAT·F / SAT·V / NAT",',
    ),
    # 4. buildBody: mode-aware request for the native 32B worker.
    (
        '''  if (modelKey === "bintf32b") {
    // Native 32B AR accepts only its own sampler controls and retains the generated continuation in its request cache.
    const greedy = dec === "greedy";
    const requestedTokens = Math.min(effectiveTokenLimit(requestProfile, "ar"), Math.max(1, Math.trunc(+lenEl.value || 256)));
    const exactLength = lengthPolicyEl.value === "exact";
    return { prompt, mode:"ar", sampler:"ar", max_new:requestedTokens,
      min_new:exactLength ? requestedTokens : 0, ignore_eos:exactLength,
      greedy, temperature:greedy ? 0 : +tempEl.value, top_p:+toppEl.value, top_k:Math.max(0, Math.trunc(+topkEl.value || 0)) };
  }
''',
        '''  if (modelKey === "bintf32b") {
    // Native 32B worker (agillm32b_server, learned-satvar runtime): ar / nat / sat_fixed / sat_variable.
    // It accepts only its own sampler controls; NAT is exactly one parallel masked pass;
    // SAT fixed takes a block size of 2 or 3; SAT variable uses the checkpoint's learned {1,2} policy.
    const greedy = dec === "greedy";
    const requestedTokens = Math.min(effectiveTokenLimit(requestProfile, selectedModeKey), Math.max(1, Math.trunc(+lenEl.value || 256)));
    const exactLength = lengthPolicyEl.value === "exact";
    const body32 = { prompt, mode:selectedModeKey, sampler:selectedModeKey, max_new:requestedTokens,
      min_new:exactLength ? requestedTokens : 0, ignore_eos:exactLength,
      greedy, temperature:greedy ? 0 : +tempEl.value, top_p:+toppEl.value, top_k:Math.max(0, Math.trunc(+topkEl.value || 0)) };
    if (selectedModeKey === "sat_fixed") body32.sat_block_size = Math.min(3, Math.max(2, Math.trunc(+binarySatBlockSize || 2)));
    if (selectedModeKey === "sat_variable") body32.sat_speed_weight = binarySatSpeedWeight;
    if (selectedModeKey === "nat") { body32.nat_passes = 1; body32.nat_greedy = greedy; }
    return body32;
  }
''',
    ),
    # 5. help text
    (
        '<b>AR</b> generates one token at a time. Choose up to <b>2,048 output tokens</b> (default 256);',
        '<b>AR</b> generates one token at a time; <b>SAT Fixed</b> commits 2 or 3 masked tokens per step, <b>SAT Variable</b> lets the checkpoint\'s learned policy choose 1 or 2, and <b>NAT</b> fills the whole continuation in one parallel masked pass (fastest, lowest quality at this checkpoint). Choose up to <b>2,048 output tokens</b> (default 256);',
    ),
]


def main():
    src = PATH.read_text(encoding='utf-8')
    before = hashlib.sha256(src.encode('utf-8')).hexdigest()
    out = src
    for old, new in EDITS:
        n = out.count(old)
        if n != 1:
            print(f'ANCHOR MISMATCH ({n} matches): {old[:90]!r}')
            sys.exit(2)
        out = out.replace(old, new, 1)
    if '--check' in sys.argv:
        print('all anchors match once; sha before', before[:16])
        return
    shutil.copy2(PATH, BACKUP)
    PATH.write_text(out, encoding='utf-8')
    after = hashlib.sha256(out.encode('utf-8')).hexdigest()
    print(json_line(before=before, after=after, backup=str(BACKUP), bytes=len(out.encode('utf-8'))))


def json_line(**kw):
    import json
    return json.dumps(kw)


if __name__ == '__main__':
    main()
