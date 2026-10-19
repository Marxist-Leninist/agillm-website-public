import sys
p=sys.argv[1]; s=open(p,encoding='utf-8').read()
def rep(old,new,n=1):
    global s
    c=s.count(old); assert c==n,(old[:60],c); s=s.replace(old,new)
# CSS: health strip + header links
rep('#agi44-curves .note{margin-top:7px;font-size:10.5px;color:#8b98ab;line-height:1.4}',
'''#agi44-curves .note{margin-top:7px;font-size:10.5px;color:#8b98ab;line-height:1.4}
#agi44-curves h4 a{color:#58a6ff;text-decoration:none;font-weight:600;letter-spacing:0;text-transform:none;margin-left:8px}
#agi44-curves h4 a:hover{text-decoration:underline}
#agi44-curves .hs{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:9px}
#agi44-curves .hs .ov{font-weight:700;font-size:11.5px;letter-spacing:.06em;text-transform:uppercase;padding:3px 10px;border-radius:999px;border:1px solid}
#agi44-curves .hs .ch{display:inline-flex;align-items:center;gap:5px;font-size:11px;color:#c9d1d9;padding:2px 9px;border-radius:999px;border:1px solid #2a3346;background:#0b0e14;cursor:default}
#agi44-curves .hs .ch i{width:7px;height:7px;border-radius:50%;display:inline-block}
#agi44-curves .fl{margin-top:6px;font-size:11px;line-height:1.45}
#agi44-curves .fl div{padding-left:10px;text-indent:-10px}
#agi44-curves .ok{color:#3fb950;border-color:#3fb950}#agi44-curves .warn{color:#d29922;border-color:#d29922}#agi44-curves .bad{color:#f85149;border-color:#f85149}
#agi44-curves .hs .ch.ok i{background:#3fb950}#agi44-curves .hs .ch.warn i{background:#d29922}#agi44-curves .hs .ch.bad i{background:#f85149}#agi44-curves .hs .ch i{background:#8b98ab}''')
# header: links + health strip
rep('''    <h4>Training curves · heldout_v3 CE, lower is better <span id="agi44-upd">—</span></h4>''',
'''    <h4><span style="letter-spacing:.1em;text-transform:uppercase;font-weight:600">Training curves · heldout_v3 CE, lower is better</span><span><span id="agi44-upd">—</span><a href="/agillm-curves" target="_blank" rel="noopener" title="open the tracker on its own tab, full screen">⤢ full view</a><a href="/api/training.png" target="_blank" rel="noopener" title="chart as a PNG (also on the MCP: agillm_curves_image)">PNG</a></span></h4>
    <div class="hs" id="agi44-health"></div><div class="fl" id="agi44-flags"></div>''')
# JS: canvas height from CSS, health painter
rep('''    var dpr=window.devicePixelRatio||1, W=cv.clientWidth||600, H=290;''','''    var dpr=window.devicePixelRatio||1, W=cv.clientWidth||600, H=cv.clientHeight||290;''')
rep('''  function paintPill(){''','''  var AREAS=[["trainer","trainer"],["model","model"],["ar","AR"],["sat_fixed","SAT fixed"],["sat_variable","SAT var"],["nat","NAT"],["dblock","DBlock"],["det50","50M student"]];
  function paintHealth(){ var h=DATA.health, el=document.getElementById("agi44-health"), fl=document.getElementById("agi44-flags"); if(!h){ el.innerHTML=""; fl.innerHTML=""; return; }
    var ov=h.overall||"?", txt={ok:"all fine",warn:"warning",bad:"problem"}[ov]||ov;
    var html='<span class="ov '+esc(ov)+'">'+esc(txt)+'</span>';
    AREAS.forEach(function(a){ var ar=(h.areas||{})[a[0]]||{}, lv=ar.level||"?", fs=(h.flags||[]).filter(function(f){return f.area===a[0];}).map(function(f){return f.message;}).join(" · ");
      html+='<span class="ch '+esc(lv)+'" title="'+esc(fs||(lv==="ok"?"ok":lv))+'"><i></i>'+esc(a[1])+'</span>'; });
    el.innerHTML=html;
    fl.innerHTML=(h.flags||[]).slice(0,6).map(function(f){ return '<div class="'+esc(f.level)+'">'+esc(f.level==="bad"?"✖":"▲")+' <b>'+esc(f.area)+'</b> '+esc(f.message)+'</div>'; }).join("")||('<div style="color:#8b98ab">'+esc(h.summary||"")+'</div>'); }
  function paintPill(){''')
rep('''  function paintPill(){ var lastP=(DATA.probes||[]).slice(-1)[0], L=DATA.latest||{}; lbl.textContent=(lastP?"held-out "+F(lastP.w257,3):"loss "+F(L.loss,2))+" · lr "+E(L.lr); sparkline(); }''',
'''  function paintPill(){ var lastP=(DATA.probes||[]).slice(-1)[0], L=DATA.latest||{}, h=DATA.health||{}, ov=h.overall; lbl.textContent=(lastP?"held-out "+F(lastP.w257,3):"loss "+F(L.loss,2))+" · lr "+E(L.lr)+(ov?(" · "+({ok:"✓ fine",warn:"▲ warn",bad:"✖ problem"}[ov]||ov)):""); pill.style.borderColor=({ok:"#2a3346",warn:"#d29922",bad:"#f85149"}[ov]||"#2a3346"); sparkline(); try{paintHealth();}catch(e){} }''')
rep('''#agi44-curves h4 a{color:#58a6ff;text-decoration:none;font-weight:600;letter-spacing:0;text-transform:none;margin-left:8px}''','''#agi44-curves h4 a{color:#58a6ff;text-decoration:none;font-weight:600;letter-spacing:0;text-transform:none;margin-left:8px;white-space:nowrap}\n#agi44-curves h4>span:last-child{white-space:nowrap;text-align:right}''')
open(p,'w',encoding='utf-8').write(s); print('panel v3 patched',len(s))
