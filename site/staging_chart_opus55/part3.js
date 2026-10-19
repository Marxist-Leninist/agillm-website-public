      var hb=(m.h&&m.h.best)||{}, best=hb[k]?num(hb[k].value):null, v=s.value, d=best===null?null:v-best;
      var glyph=d===null?null:(d<=0.05?"✓":d<=0.15?"◷":"▲"), level=d===null?null:(d<=0.05?"ok":d<=0.15?"maintenance":"warn"), above=!!(m.ref&&v>=m.ref.v);
      var title=DEFS[k].name+" "+F(v,3)+" (mean of the last "+N(W)+" steps)"+(best!==null?" · best "+F(best,3)+" @"+N(hb[k].step)+" ("+(d>=0?"+":"−")+F(Math.abs(d),3)+")":"")+
        (m.ref?" · unigram "+F(m.ref.v,3)+(above?" — ABOVE the unigram baseline: worse than ignoring context":" ("+F(v-m.ref.v,3)+" vs unigram)"):"")+" · ✓ within 0.05 of best, ◷ within 0.15, ▲ further";
      return {value:v,label:k==="anc"?"AR":"AR local",title:title,series:series,glyph:glyph,level:level,above:above}; }
    var tr=(DATA.train||[]).map(function(p){return num(p.ema);}).filter(function(v){return v!==null;});
    return {value:num((DATA.latest||{}).ema),label:"EMA",title:"mixed-objective EMA (per-objective history not attached yet)",series:tr.length>=300?tr.slice(100):tr};
  }
  window.AGILLMCurves={headline:headline,redraw:function(){ try{ draw(); }catch(e){} }};

  /* ---------- status cards (below the chart) ---------- */
  function paintModesLive(m){
    var el=$("agi44-mode-loss"); if(!el) return;
    if(!(m.has.anc||m.has.ar||m.has.sat||m.has.nat)){ paintModesLegacy(false); return; }
    var W=trendW(m);
    function card(name,k,extra,state){ var d=DEFS[k]||{}, s=k&&m.has[k]?winStats(m,k,W):null;
      return '<div class="modecard" data-mode="'+esc(name)+'" data-status="'+(s?"recent":"missing")+'"><strong>'+esc(name)+'</strong><div class="value" style="color:'+(d.col||"#8b98ab")+'">'+(s?"CE "+F(s.value,4):"—")+'</div><div class="status '+(s?"ok":"warn")+'">'+esc(state||(s?"training CE · last "+N(W)+" steps":"no samples"))+'</div><div class="meta">'+(s?trendHtml(s).replace(/<\/?i[^>]*>/g,"")+(s.n?" · "+N(s.n)+" samples":""):"")+(extra?'<span class="detail">'+esc(extra)+'</span>':'')+'</div></div>'; }
    var hb=(m.h&&m.h.best)||{};
    el.innerHTML=card("AR","anc",(m.has.ar?"Block-local AR "+F((winStats(m,"ar",W)||{}).value,4)+". ":"")+(hb.anc&&num(hb.anc.value)!==null?"Best full-stack "+F(hb.anc.value,4)+" @"+N(hb.anc.step)+". ":"")+"Full-stack CE on the anchor steps.")+
      '<div class="modecard" data-mode="sat_fixed" data-status="n/a"><strong>SAT fixed</strong><div class="value" style="color:#8b98ab">—</div><div class="status">not a '+(m.lin==="rpv16"?"RPV16":"")+' training objective</div><div class="meta">This lineage trains AR, SAT var and NAT only.</div></div>'+
      card("SAT var","sat","Variable-block SAT training CE.")+card("NAT","nat","NAT training CE.");
  }
  function paintModesLegacy(unavailable){ var el=$("agi44-mode-loss"); if(!el) return;
    el.innerHTML='<div class="modecard" data-status="'+(unavailable?"stale":"missing")+'"><strong>Mode telemetry</strong><span class="meta">'+(unavailable?"Feed unavailable — last evidence kept on the chart.":"Per-objective AR / SAT var / NAT history not attached by the collector yet.")+'</span></div>'; }

  /* ---------- pill, health, "now" cards (unchanged behaviour) ---------- */
  function sparkline(){ var hl=headline(), vals=hl&&hl.series?hl.series.filter(function(v){return v!==null&&isFinite(v);}):[]; if(vals.length<2){ spark.innerHTML=""; return; }
    var mn=Math.min.apply(null,vals), mx=Math.max.apply(null,vals), d=vals.map(function(v,i){ return (i?"L":"M")+(i/(vals.length-1)*33+0.5).toFixed(1)+","+(13-(v-mn)/Math.max(1e-6,mx-mn)*12).toFixed(1); }).join(" ");
    spark.innerHTML='<path d="'+d+'" fill="none" stroke="#58a6ff" stroke-width="1.5" stroke-linejoin="round"/>'; }
  var AREAS=[["trainer","trainer"],["model","model"],["ar","AR"],["sat_fixed","SAT fixed"],["sat_variable","SAT var"],["nat","NAT"],["dblock","DBlock"],["det50","50M student"]];
  function paintNow(){
    if(!nowEl||!DATA)return; var h=DATA.health||{}, a=h.areas||{}, tr=a.trainer||{}, mo=a.model||{}, sv=a.sat_variable||{}, de=a.det50||{};
    function card(level,key,title,meta){return '<div class="nowcard '+esc(level||'maintenance')+'"><span class="k">'+esc(key)+'</span><strong>'+esc(title)+'</strong><span class="m">'+esc(meta||'')+'</span></div>'; }
    var tl=tr.level||'maintenance', tt=(tr.state==='training'&&tl==='ok')?'TRAINING LIVE':(tl==='bad'?'TRAINER PROBLEM':'TRAINER '+String(tr.state||tl).toUpperCase());
    var L0=DATA.latest||{}, trTok=tr.raw_tok_s!=null?tr.raw_tok_s:(L0.fullstack_tok_s_64!=null?L0.fullstack_tok_s_64:L0.raw_tok_s), trLr=tr.lr!=null?tr.lr:L0.lr, trAge=tr.progress_age_s!=null?tr.progress_age_s:tr.heartbeat_age_s;
    var tm='step '+N(tr.step)+' · '+(trTok==null?'—':N(Math.round(trTok)))+' tok/s · LR '+E(trLr)+' · progress '+(trAge==null?'—':Math.round(trAge)+'s ago');
    var ev=mo.evaluation||{}, qfresh=!!mo.fresh, qlevel=qfresh?(mo.level||'ok'):((['paused','waiting_interval','waiting_checkpoint','running'].indexOf(ev.phase)>=0)?'maintenance':'warn');
    var noProbe=mo.probe_age_s==null&&!Number.isFinite(Number(mo.heldout_w257))&&!(DATA.probes||[]).length; if(noProbe){qlevel='maintenance';}
    var qtitle=noProbe?'NO HELD-OUT PROBE YET':qfresh?'QUALITY EVIDENCE FRESH':({paused:'QUALITY PROBE PAUSED',waiting_interval:'QUALITY PROBE WAITING',waiting_checkpoint:'WAITING FOR CHECKPOINT',running:'QUALITY PROBE RUNNING',suspended:'QUALITY PROBE SUSPENDED',offline:'QUALITY PROBE OFFLINE',stalled:'QUALITY PROBE STALLED'}[ev.phase]||'QUALITY EVIDENCE STALE');
    var qa=mo.probe_age_s==null?'—':Math.round(mo.probe_age_s/60)+' min old'; var qm=noProbe?(mo.note||'no fixed held-out probe on this lineage yet'):qa+' · '+(ev.reason||({paused:'training continues while evaluation is paused',waiting_interval:'monitor alive; next interval pending',waiting_checkpoint:'monitor alive; next checkpoint pending',running:'CPU evaluation in progress'}[ev.phase]||'last-known quality only'));
    var rr=sv.routing_readiness||'unknown', rl=sv.routing_level||sv.level||'maintenance', gt=sv.last_anchor_gate||{};
    var rtitle=(rr==='two_class_predictions_seen')?'ROUTING SIGNAL ACTIVE':(rr.indexOf('fail_closed')===0?'NOT ROUTING-READY':'ROUTING STATUS '+String(rr).toUpperCase());
    var rm='main gate: '+N(gt.predicted_stride2)+' stride-2 predictions / '+N(gt.stride2_labels)+' stride-2 labels · threshold '+F(gt.serving_stride2_threshold,2);
    if(de.sat_gate_routing_status) rm+=' · 50M gate: '+de.sat_gate_routing_status;
    nowEl.innerHTML=card(tl,'Training',tt,tm)+card(qlevel,'Quality evidence',qtitle,qm)+card(rl,'SAT-variable routing',rtitle,rm);
  }
  function paintHealth(){ var h=DATA.health, el=$("agi44-health"), fl=$("agi44-flags"); if(!h){ el.innerHTML=""; fl.innerHTML=""; return; }
    var ov=h.overall||"?", txt={ok:"all fine",maintenance:(((h.areas||{}).model||{}).fresh?"quality checked · qualification pending":"quality check pending"),warn:"warning",bad:"problem"}[ov]||ov;
    var html='<span class="ov '+esc(ov)+'">'+esc(txt)+'</span>';
    AREAS.forEach(function(a){ var ar=(h.areas||{})[a[0]]||{}, lv=ar.level||"?", fs=(h.flags||[]).filter(function(f){return f.area===a[0];}).map(function(f){return f.message;}).join(" · ");
      html+='<span class="ch '+esc(lv)+'" title="'+esc(fs||(lv==="ok"?"ok":lv))+'"><i></i>'+esc(a[1])+'</span>'; });
    el.innerHTML=html; paintNow();
    fl.innerHTML=(h.flags||[]).slice(0,6).map(function(f){ return '<div class="'+esc(f.level)+'">'+esc(f.level==="bad"?"✖":f.level==="maintenance"?"◷":"▲")+' <b>'+esc(f.area)+'</b> '+esc(f.message)+'</div>'; }).join("")||('<div style="color:#8b98ab">'+esc(h.summary||"")+'</div>'); }
  function paintPill(){
    var L=DATA.latest||{}, h=DATA.health||{}, ov=h.overall;
    if(DATA.curves_available===false){
      var step=num(L.step); lbl.textContent="GB10 "+(step!==null?N(step):"checkpoint live")+" · curves reconnecting";
      lbl.title=(h.summary||DATA.reason||"checkpoint feed live; historical curves temporarily unavailable"); pill.style.borderColor="#58a6ff";
      sparkline(); try{paintHealth();}catch(e){} return;
    }
    var hl=headline(), m=(h.areas||{}).model||{}, ev=m.evaluation||{}, pending=ev.phase==="running";
    var evidence=hl&&hl.value!==null?(hl.label+" "+F(hl.value,3)):"loss "+F(L.loss,2);
    lbl.textContent=evidence+" · lr "+E(L.lr)+(pending?" · checking":ov?(" · "+({ok:"✓ fine",maintenance:"◷ checking",warn:"▲ warn",bad:"✖ problem"}[ov]||ov)):"");
    lbl.title=hl?hl.title:"training loss";
    pill.style.borderColor=({ok:"#2a3346",maintenance:"#58a6ff",warn:"#d29922",bad:"#f85149"}[ov]||"#2a3346");
    sparkline(); try{paintHealth();}catch(e){}
  }
  function mergeCurrentHeldout(d){ if(!d) return; var mc=d.multimode_current||[]; var mm=d.modes||(d.modes=[]); mm=d.modes=mm.filter(function(p){return !p||p.source!=="cpu_current";}); mc.forEach(function(p){ if(!p) return; mm.push({step:p.step,t:p.t,ts:p.ts,source:"cpu_current",ar_current_ce:p.ar,sat_fixed_ce:p.sat,nat_ce:(p.nat_pooled!=null?p.nat_pooled:null)}); }); mm.sort(function(a,b){return (a.step||0)-(b.step||0);}); }
  /* AGILLM_TRANSFORMER_NN_PFLOP_TARGET_20260920
     Owner target: the actual executed Transformer training graph over wall time.
     Isolated microkernels and 6*conventional-parameter*tok/s are not completion. */
  function paintPflopTarget(d){
    var c=(d&&d.transformer_compute)||null;
    if(!c || typeof c!=="object") return;
    var r=c.current||{}, runtime=c.runtime||{}, micro=c.microkernel_reference||{};
    var pct=Number(r.target_attainment_percent), pf=Number(r.dense_equivalent_pflop_s), physical=Number(r.physical_nonzero_pflop_s), kernel=Number(micro.verified_sparse_microkernel_pflop_s);
    if(!Number.isFinite(pct) || !Number.isFinite(pf)) return;
    var compactEl=document.getElementById("pflopCompact"), full=document.getElementById("pflopFull"), live=runtime.running===true;
    if(compactEl) compactEl.textContent=(live?"NN ":"Last NN ")+pf.toFixed(3)+" PF";
    var text="1 PF Transformer target: "+pct.toFixed(1)+"% · "+pf.toFixed(3)+" PF dense-equivalent"+(Number.isFinite(physical)?" · "+physical.toFixed(3)+" PF physical nonzero":"")+" · rolling "+Number(r.updates||0)+" executed updates over wall time"+(Number.isFinite(kernel)?" · sparse microkernel "+kernel.toFixed(3)+" PF is reference only":"")+".";
    if(full) full.textContent=text;
    if(compactEl) compactEl.title=text;
  }
  function refreshPflopTarget(){
    var pe=document.getElementById("pflopCompact"); if(document.hidden||!pe||pe.offsetParent===null) return;
    fetch("/agillm-rpv16-training",{cache:"no-store"}).then(function(r){if(!r.ok)throw new Error("HTTP "+r.status);return r.json();}).then(function(d){paintPflopTarget(d);}).catch(function(){});
  }
  refreshPflopTarget(); setInterval(refreshPflopTarget,30000);
  function refreshServed(){
    if(document.hidden) return;
    fetch("/agillm-rpv16-health",{cache:"no-store"}).then(function(r){if(!r.ok)throw new Error("HTTP "+r.status);return r.json();}).then(function(h){
      var s=h&&h.ok===true&&h.model?num(h.model.step):null; var prev=SERVED&&SERVED.step; SERVED=s===null?null:{step:s,ckpt:h.ckpt||h.model.ckpt||null};
      if((SERVED&&SERVED.step)!==prev) draw(); }).catch(function(){});
  }
  /* Bandwidth (owner is often on a metered phone hotspot): poll the lean feed (?history=0, ~11 KB gz) every
     60 s and fetch the full history (~120 KB gz) only on first load, after a rollback, or when the cached
     history no longer overlaps the live tail. Nothing is fetched while the tab is hidden. */
  var HIST=null, HIST_SIG=null, HIST_LAST=null, HIST_AT=0, FULL_BUSY=false, LAST_TICK=0;
  function getJSON(u){ return fetch(u,{cache:"no-store"}).then(function(r){ if(!r.ok) throw new Error("HTTP "+r.status); return r.json(); }); }
  function histSig(d){ var hs=d.history_summary||{}, sm=hs.summary||(d.history&&d.history.summary)||{}; return String(sm.rows_abandoned)+"|"+String(sm.first_step); } // a rollback moves rows_abandoned
  function keepHistory(d){ var h=d.history&&Array.isArray(d.history.rows)?d.history:null; if(!h) return false;
    HIST={history:h,standard_history:d.standard_history||null}; HIST_SIG=histSig(d); HIST_AT=Date.now(); var r=h.rows, c=h.cols||[], i1=c.indexOf("s1"); HIST_LAST=r.length&&i1>=0?num(r[r.length-1][i1]):null; return true; }
  function apply(d){
    if(HIST){ if(!("history" in d)) d.history=HIST.history; if(!("standard_history" in d)) d.standard_history=HIST.standard_history; }
    DATA=d;mergeCurrentHeldout(DATA);window.__AGILLM_TRAINING_STATUS=d;paintPflopTarget(d);
    if(window.AGILLMProgressV1)window.AGILLMProgressV1.render(d);
    if(typeof renderTrainerMirror==="function")renderTrainerMirror();
    paintPill();draw();
  }
  function fetchFull(){ if(FULL_BUSY) return; FULL_BUSY=true;
    getJSON("/api/training").then(function(f){ if(f&&f.ok){ keepHistory(f); apply(f); } }).catch(function(){}).then(function(){ FULL_BUSY=false; }); }
  function tick(force){
    if(document.hidden&&!force) return; LAST_TICK=Date.now();
    function unavailable(reason){
      window.__AGILLM_TRAINING_STATUS={ok:false,reason:reason};
      if(window.AGILLMProgressV1){ window.dispatchEvent(new Event("agi44-feed-unavailable")); window.AGILLMProgressV1.unavailable(); }
      if(!DATA)lbl.textContent="curves: "+reason;
      else{lbl.textContent="last evidence · feed unavailable";lbl.title=reason;}
      if(typeof renderTrainerMirror==="function")renderTrainerMirror();
    }
    getJSON(HIST?"/api/training?history=0":"/api/training").then(function(d){
      if(!d||!d.ok){unavailable((d&&d.reason)||"unavailable");return;}
      if(keepHistory(d)){ apply(d); return; }          // full doc (first load, or a server without ?history=0)
      var tail=d.train||[], t0=tail.length?num(tail[0].step):null;
      var need=histSig(d)!==HIST_SIG||(HIST_LAST!==null&&t0!==null&&t0>HIST_LAST-64)||(Date.now()-HIST_AT>30*60*1000);
      apply(d); if(need) fetchFull();
    }).catch(function(e){unavailable(e&&e.message||"unreachable");});
    refreshServed();
  }
  document.addEventListener("visibilitychange",function(){ if(!document.hidden&&Date.now()-LAST_TICK>20000){ tick(true); refreshPflopTarget(); } });
  window.addEventListener("agi44-feed-unavailable",function(){ if(!MODEL||!(MODEL.has.anc||MODEL.has.ar)) paintModesLegacy(true); });
  var rz=0; window.addEventListener("resize",function(){ if(rz) return; rz=requestAnimationFrame(function(){ rz=0; try{draw();}catch(e){} }); });
  if(window.ResizeObserver) new ResizeObserver(function(){ if(rz) return; rz=requestAnimationFrame(function(){ rz=0; try{draw();}catch(e){} }); }).observe(cv);
  tick(); setInterval(tick,60000);
})();
