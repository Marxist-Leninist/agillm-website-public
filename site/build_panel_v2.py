#!/usr/bin/env python3
"""Rebuild agillm_curves_panel.html (the shared loss-chart panel) from the live
agillm_completion.html so /agillm-curves shows exactly the /agillm-ui chart.
usage: build_panel_v2.py COMPLETION.html PANEL_OUT.html"""
import re, sys
from pathlib import Path
src, out = sys.argv[1], sys.argv[2]
h = Path(src).read_text(encoding="utf-8")
def one(pat):
    ms = list(re.finditer(pat, h, re.S))
    if len(ms) != 1: sys.exit(f"panel anchor matched {len(ms)}x: {pat[:70]!r}")
    return ms[0].group(0)
style = one(r'<style>\n#agi44-curves\{position:fixed;.*?</style>')
style2 = one(r'<style id="agi44-chart-v2-css">.*?</style>')
a = h.index('<div id="agi44-curves">'); b = h.index('<div id="agi43-ckpt-panel">', a)
markup = h[a:b].rstrip()
markup = re.sub(r'<button type="button" class="telemetry-close"[^>]*>.*?</button>', '', markup, count=1, flags=re.S)
prog = one(r'<script>\n(?:(?!</script>).)*?root\.AGILLMProgressV1=api;.*?</script>')
chart = one(r'<script id="agi44-chart-v2">.*?</script>')
Path(out).write_text("\n".join([style, style2, markup, prog, chart]) + "\n", encoding="utf-8")
print("panel", out, len(style), len(style2), len(markup), len(prog), len(chart))
