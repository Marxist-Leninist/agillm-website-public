#!/usr/bin/env python3
"""Fix the AGILLM4.3 /agillm-ui front end: responsive layout + latest-ckpt wiring.
Idempotent, backs up, aborts if anchors are not unique."""
import time, shutil, sys

F = "/opt/agillm43_netserve/agillm_completion.html"
src = open(F, encoding="utf-8").read()
if "AGI43_RESPONSIVE_FIX" in src:
    print("already-patched"); sys.exit(0)

CSS = """
/* AGI43_RESPONSIVE_FIX: inspector was a fixed 320px panel with no mobile rule,
   crushing #app on phones. On mobile make it an off-canvas overlay drawer. */
#inspectorFab { display:none; }
@media (max-width:760px) {
  #inspector { position:fixed; top:0; right:0; bottom:0; z-index:40;
    width:min(92vw,360px); flex:0 0 auto; transform:translateX(0);
    transition:transform .22s ease; box-shadow:-14px 0 40px rgba(0,0,0,.5); }
  #inspector.collapsed { width:min(92vw,360px); flex:0 0 auto; transform:translateX(100%); }
  #app { min-width:0; }
  #inspectorFab { display:flex; align-items:center; gap:7px; position:fixed;
    right:14px; bottom:16px; z-index:35; padding:10px 15px; border-radius:22px;
    border:1px solid var(--line); background:#14202b; color:var(--accent);
    font:600 13px system-ui,sans-serif; box-shadow:0 6px 20px rgba(0,0,0,.45); cursor:pointer; }
}
</style>"""

FAB = """<div id="workspace">
  <button id="inspectorFab" type="button" aria-label="Open logit inspector">\U0001F4CA Logits</button>"""

JS = """<script>
/* AGI43_RESPONSIVE_FIX runtime: mobile inspector drawer + latest-ckpt header wiring */
(function(){
  var insp=document.getElementById('inspector');
  var fab=document.getElementById('inspectorFab');
  var tgl=document.getElementById('toggleInspector');
  if(insp&&fab){
    var mobile=function(){return window.matchMedia('(max-width:760px)').matches;};
    var updFab=function(){ fab.style.display=(mobile()&&insp.classList.contains('collapsed'))?'flex':'none'; };
    if(mobile()) insp.classList.add('collapsed');           // don't cover the page on load
    fab.addEventListener('click',function(){ insp.classList.remove('collapsed');
      try{localStorage.setItem('agillm_inspector_collapsed','0');}catch(e){} updFab(); });
    if(tgl) tgl.addEventListener('click',function(){ setTimeout(updFab,0); });
    window.addEventListener('resize',updFab);
    updFab();
  }
  // Wire the LATEST rescued checkpoint (from the GETH mirror) into the header, live.
  var fmtAge=function(s){ if(s==null)return''; if(s<90)return s+'s ago';
    if(s<5400)return Math.round(s/60)+'m ago'; return Math.round(s/3600)+'h ago'; };
  var hdr=document.querySelector('header');
  if(hdr){
    var chip=document.createElement('span'); chip.className='sub'; chip.id='latestckpt';
    chip.style.opacity='.8'; chip.style.marginLeft='8px';
    hdr.appendChild(chip);
    fetch('/agillm-latest',{cache:'no-store'}).then(function(r){return r.json();}).then(function(d){
      if(d&&d.step){ chip.textContent='· latest ckpt: step '+Number(d.step).toLocaleString('en-US')+
        (d.mirror_age_sec!=null?' ('+fmtAge(d.mirror_age_sec)+')':''); }
    }).catch(function(){});
  }
})();
</script>
</body>"""

for anchor, cnt in (("</style>", src.count("</style>")),
                    ('<div id="workspace">', src.count('<div id="workspace">')),
                    ("</body>", src.count("</body>"))):
    if cnt != 1:
        print("ABORT anchor %r count=%d (expected 1)" % (anchor, cnt)); sys.exit(1)

out = src.replace("</style>", CSS, 1)
out = out.replace('<div id="workspace">', FAB, 1)
out = out.replace("</body>", JS, 1)

bak = F + ".bak-agi43fix-" + time.strftime("%Y%m%dT%H%M%SZ")
shutil.copy(F, bak)
open(F, "w", encoding="utf-8").write(out)
print("patched OK; backup", bak, "| new size", len(out))
