#!/usr/bin/env python3
"""Splice the AGILLM loss chart v2 into agillm_completion.html.

usage: patch_page.py IN.html OUT.html chart_v2.js chart_v2.css
Anchors are matched exactly once; any miss aborts without writing.
Idempotent: a page that already carries chart v2 gets its v2 blocks replaced.
"""
import re, sys
from pathlib import Path

src_path, out_path, js_path, css_path = sys.argv[1:5]
html = Path(src_path).read_text(encoding="utf-8")
js = Path(js_path).read_text(encoding="utf-8").rstrip("\n")
css = Path(css_path).read_text(encoding="utf-8").strip("\n")

def one(pattern, text, flags=re.S):
    ms = list(re.finditer(pattern, text, flags))
    if len(ms) != 1:
        sys.exit(f"anchor matched {len(ms)}x (need 1): {pattern[:90]!r}")
    return ms[0]

# 1) chart script block (legacy or an earlier v2) -> v2
if 'id="agi44-chart-v2"' in html:
    m = one(r'<script id="agi44-chart-v2">.*?</script>', html)
else:
    m = one(r'<script>\n\(function\(\)\{\n  var ROOT=document\.getElementById\("agi44-curves"\).*?</script>', html)
html = html[:m.start()] + '<script id="agi44-chart-v2">\n' + js + '\n</script>' + html[m.end():]

# 2) CSS after the panel's own <style> (so it wins the cascade), or replace an earlier v2
if 'id="agi44-chart-v2-css"' in html:
    m = one(r'<style id="agi44-chart-v2-css">.*?</style>', html)
    html = html[:m.start()] + css + html[m.end():]
else:
    m = one(r'<style>\n#agi44-curves\{position:fixed;.*?</style>', html)
    html = html[:m.end()] + "\n" + css + html[m.end():]

# 3) RPV16 card chip: headline = AR full-stack CE from the chart, not the mixed-objective EMA
if True:
    old_val = (" out.textContent=ema!==null?ema.toFixed(2):(loss!==null?loss.toFixed(2):'—');\n"
               " const unit=document.createElement('small'); unit.textContent=ema!==null?' EMA':' step'; out.appendChild(unit);\n")
    new_val = (" const hl=window.AGILLMCurves&&typeof window.AGILLMCurves.headline==='function'?window.AGILLMCurves.headline():null, hv=hl&&num(hl.value)!==null?hl.value:null;\n"
               " out.textContent=hv!==null?hv.toFixed(2):(ema!==null?ema.toFixed(2):(loss!==null?loss.toFixed(2):'—'));\n"
               " const unit=document.createElement('small'); unit.textContent=hv!==null?' '+hl.label:(ema!==null?' EMA':' step'); out.appendChild(unit);\n")
    old_ser = (" let series=Array.isArray(t.train)?t.train.map(p=>p&&num(p.ema)!==null?p.ema:(p?p.loss:null)):[];\n"
               " // The feed's EMA restarts at its window's first sample (alpha .03): skip the warm-up so the spark shows the trend, not the seed.\n"
               " if(series.length>=300) series=series.slice(100);\n")
    new_ser = (" let series=hl&&Array.isArray(hl.series)&&hl.series.filter(v=>num(v)!==null).length>=2?hl.series.slice():(Array.isArray(t.train)?t.train.map(p=>p&&num(p.ema)!==null?p.ema:(p?p.loss:null)):[]);\n"
               " // The feed's EMA restarts at its window's first sample (alpha .03): skip the warm-up so the spark shows the trend, not the seed.\n"
               " if(!(hl&&hl.series)&&series.length>=300) series=series.slice(100);\n")
    old_gl = " if(glyph){const g=document.createElement('span');g.className=glyph[1];g.textContent=' '+glyph[0];out.appendChild(g);}\n"
    new_gl = (" const qg=hl&&hl.glyph?[hl.glyph,{ok:'rpv-ok',maintenance:'',warn:'rpv-warn'}[hl.level]||'']:glyph; // model quality vs best, not service health\n"
              " if(qg){const g=document.createElement('span');g.className=qg[1];g.textContent=' '+qg[0];out.appendChild(g);}\n")
    old_tt = " if(chip) chip.title='Open the training curves · loss '"
    new_tt = " if(chip) chip.title='Open the training curves'+(hl&&hv!==null?' · '+hl.title:'')+' · last step loss '"
    if new_val not in html:
        for a, b in ((old_val, new_val), (old_gl, new_gl), (old_ser, new_ser), (old_tt, new_tt)):
            if html.count(a) != 1:
                sys.exit(f"card anchor matched {html.count(a)}x: {a[:70]!r}")
            html = html.replace(a, b)

Path(out_path).write_text(html, encoding="utf-8")
print("ok", len(html))
