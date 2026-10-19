#!/usr/bin/env python3
"""Build /opt/agillm43_netserve/agillm_curves_page.html (the full-screen /agillm-curves view)
from the shared panel agillm_curves_panel.html. Re-run after every panel change."""
from pathlib import Path
import sys
PANEL = sys.argv[1] if len(sys.argv) > 1 else '/opt/agillm43_netserve/agillm_curves_panel.html'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/opt/agillm43_netserve/agillm_curves_page.html'
panel = Path(PANEL).read_text(encoding='utf-8').strip('\n')
page = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AGILLM — training curves</title>
<style>
:root{color-scheme:dark}
body{margin:0;background:#0b0e14;color:#e6edf3;font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
.top{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:14px 18px 0;flex-wrap:wrap}
.top h1{font-size:19px;margin:0;letter-spacing:-.01em}
.top a{color:#58a6ff;text-decoration:none;font-size:13px;margin-left:14px}
.top .sub{color:#8b98ab;font-size:12.5px}
/* the shared panel, but static + full width + tall canvas */
#agi44-curves{position:static!important;max-width:none!important;padding:8px 18px 20px}
#agi44-curves-pill{display:none!important}
#agi44-curves-body{display:flex!important;flex-direction:column;width:100%!important;max-height:none!important;box-shadow:none!important;background:transparent!important;border:0!important;padding:0 3px!important;overflow:visible!important;backdrop-filter:none!important}
#agi44-curves canvas{height:min(64vh,720px)!important;min-height:300px}
#agi44-curves h4 a[href="/agillm-curves"]{display:none}
#agi44-curves .telemetry-close{display:none}
.det{padding:0 18px 30px;display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(300px,1fr))}
.det .card{background:#111624;border:1px solid #2a3346;border-radius:12px;padding:12px 14px}
.det h3{margin:0 0 8px;font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:#8b98ab;display:flex;align-items:center;gap:8px}
.det h3 i{width:8px;height:8px;border-radius:50%;display:inline-block;background:#8b98ab}
.det .ok i{background:#3fb950}.det .maintenance i{background:#58a6ff}.det .warn i{background:#d29922}.det .bad i{background:#f85149}
.det dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:3px 12px;font-size:12px}
.det dt{color:#8b98ab;white-space:nowrap}.det dd{margin:0;text-align:right;font-variant-numeric:tabular-nums;overflow-wrap:anywhere;font-family:ui-monospace,Menlo,Consolas,monospace;font-size:11.5px}
.det .flag{font-size:12px;margin-top:6px}.det .flag.warn{color:#d29922}.det .flag.bad{color:#f85149}
.foot{padding:0 18px 24px;color:#8b98ab;font-size:12px}
.foot code{background:#080b11;border:1px solid #2a3346;border-radius:6px;padding:1px 6px}
</style></head><body>
<div class="top"><div><h1>AGILLM training curves</h1><div class="sub">RPV16 2.0B on GB10 and Standard 1.1B, one model per axis · AR full-stack, AR block-local, SAT var, NAT · rollbacks and abandoned branches · drag to zoom · refreshes every minute</div></div>
<div><a href="/agillm-ui">← model UI</a><a href="/api/training.png" target="_blank" rel="noopener">PNG</a><a href="/api/health" target="_blank" rel="noopener">health JSON</a><a href="/api/training" target="_blank" rel="noopener">data JSON</a></div></div>
''' + panel + '''
<script>document.getElementById("agi44-curves").classList.add("open");</script>
<div class="det" id="det"></div>
<div class="foot">Health is heuristic and read-only: trainer/box (log freshness, tracebacks, disk, GPU, pace), whole model (held-out vs best, probe age, governor), AR / SAT fixed / NAT composition anchors (ran, accepted, spikes, finite, trainer-val trends), SAT-variable gate gradients, DBlock local training (spike streaks, commit rate, block skew, per-block local CE), detachable-50M student (error-skips, commit rate). Same verdicts on the MCP: <code>agillm_health</code>, <code>agillm_curves</code> (text, optional image), <code>agillm_curves_image</code> at <code>https://codex.opentransformers.online/mcp</code>.</div>
<script>
(function(){
  var AREAS=[["trainer","Trainer / box"],["model","Whole model (held-out)"],["ar","AR"],["sat_fixed","SAT fixed"],["sat_variable","SAT variable"],["nat","NAT"],["dblock","DBlock local training"],["det50","Detachable 50M student"]];
  function esc(t){return String(t==null?"":t).replace(/[<>&]/g,function(c){return {"<":"&lt;",">":"&gt;","&":"&amp;"}[c];});}
  function fmt(v){ if(v===null||v===undefined) return "—"; if(typeof v==="number") return (Math.abs(v)<1e-3&&v!==0)?v.toExponential(2):(Number.isInteger(v)?v.toLocaleString():(+v.toFixed(4)).toString()); if(typeof v==="object") return Object.keys(v).map(function(k){ var x=v[k]; return esc(k)+"="+((x&&typeof x==="object")?esc(JSON.stringify(x)).slice(0,120):fmt(x)); }).join(" · ").slice(0,480); return esc(String(v)); }
  function paint(h){ var el=document.getElementById("det"); if(!h||!h.areas){ el.innerHTML=""; return; } var html="";
    AREAS.forEach(function(a){ var ar=h.areas[a[0]]||{}, lv=ar.level||"?"; var rows="";
      Object.keys(ar).forEach(function(k){ if(k==="level") return; var v=ar[k]; if(k==="local_ar_ce_by_block"&&v&&typeof v==="object"){ v=Object.keys(v).map(function(b){return b+":"+v[b];}).join(" "); } if(k==="window"&&v&&typeof v==="object"){ v="attempts "+JSON.stringify(v.attempts)+" commits "+JSON.stringify(v.commits)+" err "+JSON.stringify(v.error_skips); } if(k==="last_train"&&Array.isArray(v)){ v=v.map(function(x){return "L"+x.layer+" "+x.sublayer;}).join(", "); } rows+="<dt>"+esc(k)+"</dt><dd>"+fmt(v)+"</dd>"; });
      var flags=(h.flags||[]).filter(function(f){return f.area===a[0];}).map(function(f){return '<div class="flag '+esc(f.level)+'">'+esc(f.level==="bad"?"✖ ":"▲ ")+esc(f.message)+'</div>';}).join("");
      html+='<div class="card '+esc(lv)+'"><h3><i></i>'+esc(a[1])+' <span style="margin-left:auto;font-weight:400">'+esc(lv)+'</span></h3>'+flags+'<dl>'+rows+'</dl></div>'; });
    el.innerHTML=html; }
  function tick(){ fetch("/api/health",{cache:"no-store"}).then(function(r){return r.json();}).then(paint).catch(function(){}); }
  tick(); setInterval(tick,60000);
})();
</script>
</body></html>
'''
Path(OUT).write_text(page, encoding='utf-8')
print(OUT, len(page))
