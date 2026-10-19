#!/usr/bin/env python3
"""AGI44_TELEMETRY_RESTORE_20261002 - undo chart-v2 (2026-09-23, claude-opus55-3dff5aa4) default-hiding
and the 2026-09-30 fleet-snapshot suppression of current-run rows in the AGILLM loss chart.

usage: restore_telemetry_20261002.py IN.html OUT.html
Every anchor must match exactly once; any miss aborts without writing. Idempotent marker check.
"""
import re, sys
from pathlib import Path

MARK = "AGI44_TELEMETRY_RESTORE_20261002"
src, out = sys.argv[1:3]
h = Path(src).read_text(encoding="utf-8")
if MARK in h:
    sys.exit("already patched: " + src)

def rep(old, new):
    global h
    c = h.count(old)
    if c != 1:
        sys.exit(f"anchor matched {c}x (need 1) in {src}: {old[:100]!r}")
    h = h.replace(old, new)

# 1) nothing hidden by default (mixed band/EMA were hidden whenever AR/anchor existed)
rep('function defaultHidden(k){ if(k!=="mix") return false; var m=MODEL&&MODEL.lin===lineage()?MODEL:null; return !!(m&&(m.has.anc||m.has.ar)); }',
    'function defaultHidden(k){ return false; } /* ' + MARK + ': every series visible by default; hiding is only an explicit user toggle */')
# one-time reset of stored hides from the old default-hidden era
rep('if(saved.v2.hide&&typeof saved.v2.hide==="object") ST.hide=saved.v2.hide;',
    'if(saved.v2.hide&&typeof saved.v2.hide==="object"&&saved.v2.v===2026100201) ST.hide=saved.v2.hide;')
rep('v:2026092301}', 'v:2026100201}')

# 2) mixed local-objective EMA series (restored; was a default-visible line before chart v2)
m = list(re.finditer(r'\n(\s*)ref:\{name:"unigram entropy', h))
if len(m) != 1:
    sys.exit("DEFS ref anchor matched %d" % len(m))
ind = m[0].group(1)
ins = ('\n' + ind + 'ema:{name:"mixed local-objective EMA (α=.03 per step, all objectives) · raw training telemetry",'
       'short:"mixed EMA",leg:"mixed EMA (α=.03)",col:"#e3b341",w:1.6,grp:"Mixed",f:"ema",nf:"n",dash:[6,2]},')
h = h[:m[0].start()] + ins + h[m[0].start():]
rep('var ORDER={rpv16:["anc","ar","sat","nat","mix","held","ref","ghost","lr"],standard:["w257","w2048","anc","ar","sat","nat","mix","lr"]};',
    'var ORDER={rpv16:["anc","ar","sat","nat","mix","ema","held","ref","ghost","lr"],standard:["w257","w2048","anc","ar","sat","nat","mix","ema","lr"]};')
rep('var LINES=["mix","ar","sat","nat","anc"];', 'var LINES=["mix","ema","ar","sat","nat","anc"];')
rep('var MINW={mix:32,ar:8,lr:8,sat:48,nat:48,anc:96};', 'var MINW={mix:32,ema:8,ar:8,lr:8,sat:48,nat:48,anc:96};')

# 3) current-run raw rows back in the chart (fleet snapshot of 2026-09-30 had emptied T)
rep('    var T=lin==="rpv16"&&!fleetSnapshot()?tailPts(DATA.train):[];',
    '    var T=liveTail(lin);')
rep('  function buildModel(lin){',
    '  /* ' + MARK + ': current-run raw trainer rows (live_history) are charted again, per objective where the feed has it */\n'
    '  function liveTail(lin){\n'
    '    if(!fleetSnapshot()) return lin==="rpv16"?tailPts(DATA.train):[];\n'
    '    var lh=DATA.live_history&&DATA.live_history[lin], rows=lh&&Array.isArray(lh.rows)?lh.rows:[];\n'
    '    return tailPts(rows.map(function(p){ if(!p) return null; var o=String(p.objective||"").toLowerCase();\n'
    '      return {step:num(p.step),loss:num(p.loss),lr:num(p.lr),tps:num(p.tok_s!=null?p.tok_s:p.raw_tok_s),tok:num(p.seen_tokens),ts:num(p.ts),objective:(o==="ar"||o==="sat"||o==="nat")?o:null,global_anchor:p.global_anchor===true}; }).filter(Boolean));\n'
    '  }\n'
    '  function addEma(m){ var A=0.03; /* count-aware: a bucket of n samples decays the EMA n times */\n'
    '    function rowsEma(rows){ var e=null; (rows||[]).forEach(function(r){ var v=num(r.loss), n=Math.max(1,num(r.n)||1); if(v===null){ r.ema=e; return; } if(e===null) e=v; else { var k=Math.pow(1-A,n); e=e*k+v*(1-k); } r.ema=e; }); }\n'
    '    rowsEma(m.H); rowsEma(m.R); (m.branches||[]).forEach(function(b){ rowsEma(b.rows); });\n'
    '    var e=null; m.T.forEach(function(p){ var v=num(p.loss); if(v===null) return; e=e===null?v:(1-A)*e+A*v; p.ema=e; });\n'
    '  }\n'
    '  function lastEma(m){ var i; for(i=m.T.length-1;i>=0;i--) if(num(m.T[i].ema)!==null) return m.T[i].ema; var rr=m.R.length?m.R:m.H; for(i=rr.length-1;i>=0;i--) if(num(rr[i].ema)!==null) return rr[i].ema; return null; }\n'
    '  function buildModel(lin){')
rep('    m.has={}; ORDER[lin].forEach(function(k){ m.has[k]=seriesAvailable(m,k); });',
    '    addEma(m); m.has={}; ORDER[lin].forEach(function(k){ m.has[k]=seriesAvailable(m,k); });')

# 3b) live rows newer than the frozen archive are always drawn (not only when zoomed to tail resolution)
rep('    if(m.T.length&&(!rHas||tw<m.wR)&&(!hHas||dw<m.wH)){\n      var tb=bucketTail(m.T,tw); if(has(tb)){ out=tb.slice(); cut=tb[0].s0; }\n    }',
    '    if(m.T.length){ var tb=bucketTail(m.T,tw);\n      if(has(tb)){ if((!rHas||tw<m.wR)&&(!hHas||dw<m.wH)){ out=tb.slice(); cut=tb[0].s0; }\n        else { var hrEnd=Math.max(m.R.length?m.R[m.R.length-1].s1:-Infinity,m.H.length?m.H[m.H.length-1].s1:-Infinity); out=tb.filter(function(r){return r.s0>hrEnd;}); } }\n    }')

# 4) draw the mixed mean line even when the band is the AR p10-p90 band, and let it set the y-range
rep('if(k!=="mix"||m.band!=="ar") ys.push(r[f]);', 'ys.push(r[f]);')
rep('LINES.forEach(function(k){ var rows=S[k]; if(!rows||!rows.length||(k==="mix"&&m.band==="ar")) return;',
    'LINES.forEach(function(k){ var rows=S[k]; if(!rows||!rows.length) return;')
rep('var leg=k==="mix"?(m.band==="ar"?"AR p10–p90 band":"mixed band"):(d.leg||d.short)',
    'var leg=k==="mix"?(m.band==="ar"?"mixed mean + AR p10–p90 band":"mixed mean + p10–p90 band"):(d.leg||d.short)')

# 5) mixed EMA summary tile always present (was only shown when no AR/anchor existed)
rep('      if(!m.has.anc&&!m.has.ar){ var ema=num(L.ema); tile("mixed EMA",F(ema,3),\'<i>all objectives</i>\',DEFS.mix.col,"EMA α=.03 of every step\'s loss, whatever its objective"); }',
    '      { var ema=lastEma(m); tile("mixed EMA",F(ema,3),\'<i>all objectives · raw</i>\',DEFS.ema.col,"EMA α=.03 of every step\'s loss, whatever its objective (raw training telemetry, shown by default)"); }')
rep('      tile("probes",N(P.length),\'<i>recovered · no interpolation</i>\',null,"");',
    '      tile("probes",N(P.length),\'<i>recovered · no interpolation</i>\',null,"");\n'
    '      { var ema2=lastEma(m); tile("mixed EMA",F(ema2,3),\'<i>all objectives · raw</i>\',DEFS.ema.col,"EMA α=.03 of every step\'s loss, whatever its objective (raw training telemetry, shown by default)"); }')

# 5b) a series is only "available" from live rows if those rows really carry it
rep('    return m.T.some(function(p){return !!p.objective;});',
    '    return m.T.some(function(p){ return k==="anc"?p.global_anchor===true:String(p.objective||"").toLowerCase()===k; });')

# 6) note text
rep('Current-run samples appear separately in the live cards.',
    'Current-run raw trainer rows are drawn too (dots when zoomed in), and the mixed band, mixed EMA and per-objective AR/SAT/NAT series are all shown by default — toggle any of them off in the legend.')

Path(out).write_text(h, encoding="utf-8")
print("ok", src, "->", out, len(h))
