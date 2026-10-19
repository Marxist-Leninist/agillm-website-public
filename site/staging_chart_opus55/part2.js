    // restarts: faint ticks, labelled only when sparse
    var RS=m.resumes.filter(function(r){return r.step>x0&&r.step<x1&&r.step<=m.last+1;});
    RS.forEach(function(r){ if(m.branches.some(function(b){return Math.abs(b.cut-r.step)<=1;})) return; var xx=Math.round(X(r.step))+.5; g.strokeStyle=r.rollback?"rgba(240,136,62,.55)":(RS.length>24?"#222c3d":"#2f3b52"); g.setLineDash([2,3]); g.beginPath(); g.moveTo(xx,(RS.length>24&&!r.rollback)?pad.t+ph-8:pad.t); g.lineTo(xx,pad.t+ph); g.stroke(); g.setLineDash([]); });
    var RL=RS.filter(function(r){ return !m.branches.some(function(b){return Math.abs(b.cut-r.step)<=1;}); });
    if(RL.length&&RL.length<=6&&!narrow){ g.fillStyle="#6b7686"; g.textAlign="left"; RL.forEach(function(r){ var xx=X(r.step); g.fillText("restart",Math.min(xx+3,pad.l+pw-40),pad.t+ph-8); }); }
    // CPU serving checkpoint
    if(m.lin==="rpv16"&&SERVED&&num(SERVED.step)!==null&&SERVED.step>=x0&&SERVED.step<=x1){ var sx=Math.round(X(SERVED.step))+.5; g.strokeStyle="#3fb950"; g.lineWidth=1.2; g.setLineDash([6,4]); g.beginPath(); g.moveTo(sx,pad.t); g.lineTo(sx,pad.t+ph); g.stroke(); g.setLineDash([]);
      var st="▼ CPU serving "+N(SERVED.step); g.fillStyle="#3fb950"; g.textAlign=(sx+g.measureText(st).width+6>pad.l+pw)?"right":"left"; g.fillText(st,g.textAlign==="right"?sx+4:sx-4,Math.max(6,pad.t-7)); }
    // rollback cut(s): where the live lineage restarted from an older checkpoint
    m.branches.forEach(function(b){ if(b.cut<x0||b.cut>x1) return; var bx=Math.round(X(b.cut))+.5; g.strokeStyle="#f0883e"; g.lineWidth=1.4; g.setLineDash([]); g.beginPath(); g.moveTo(bx,pad.t); g.lineTo(bx,pad.t+ph); g.stroke();
      var bt=narrow?"⟲ rollback "+N(b.cut):"⟲ rolled back to "+N(b.cut)+(b.depth?" (−"+compact(b.depth)+" steps)":""), bw=g.measureText(bt).width, bxx=bx+6+bw>pad.l+pw?bx-6-bw:bx+6; bxx=Math.max(pad.l+3,Math.min(bxx,pad.l+pw-bw-3)); g.fillStyle="rgba(11,14,20,.85)"; g.fillRect(bxx-3,pad.t+3,bw+6,15); g.fillStyle="#f0883e"; g.textAlign="left"; g.fillText(bt,bxx,pad.t+11); });
    if(x1>m.last+(x1-x0)*0.01&&m.last>=x0){ var lx=Math.round(X(m.last))+.5; g.strokeStyle="rgba(230,237,243,.35)"; g.lineWidth=1; g.setLineDash([1,3]); g.beginPath(); g.moveTo(lx,pad.t); g.lineTo(lx,pad.t+ph); g.stroke(); g.setLineDash([]);
      var lt="live "+N(m.last), lw=g.measureText(lt).width, llx=Math.min(lx+6,pad.l+pw-lw-4); g.fillStyle="rgba(11,14,20,.85)"; g.fillRect(llx-3,pad.t+20,lw+6,15); g.fillStyle="#c9d1d9"; g.textAlign="left"; g.fillText(lt,llx,pad.t+28); }
    // plot
    g.save(); g.beginPath(); g.rect(pad.l,pad.t,pw,ph); g.clip();
    if(REF){ var ry=Math.round(Y(REF.v))+.5; g.strokeStyle="rgba(230,237,243,.55)"; g.lineWidth=1; g.setLineDash([1,3]); g.beginPath(); g.moveTo(pad.l,ry); g.lineTo(pad.l+pw,ry); g.stroke(); g.setLineDash([]); }
    var GAPX=2.5, gapOf=function(p,r){ return r.s0-p.s1>GAPX*Math.max(p.w||1,r.w||1)+1; };
    /* sparse objectives (SAT var / NAT were scheduled in 16-step sweeps, anchors 1 in ~17 steps) can skip a few buckets: bridge those */
    function path(rows,f,yf,k){ GAPX=(k==="sat"||k==="nat"||k==="anc")?8:2.5; var on=false,prev=null; g.beginPath(); rows.forEach(function(r){ var v=r[f]; if(v===null||v===undefined||!isFinite(v)) return; var px=X(r.x), py=yf(v); if(!on||(prev&&gapOf(prev,r))){ g.moveTo(px,py); on=true; } else g.lineTo(px,py); prev=r; }); }
    if(S.mix&&S.mix.length){ // band, split at gaps
      var segs=[], cur=[]; S.mix.forEach(function(r,i){ if(r[BLO]==null||r[BHI]==null){ if(cur.length) segs.push(cur); cur=[]; return; } if(cur.length&&gapOf(cur[cur.length-1],r)){ segs.push(cur); cur=[]; } cur.push(r); }); if(cur.length) segs.push(cur);
      g.fillStyle="rgba(139,152,171,0.16)"; segs.forEach(function(sg){ if(sg.length<2) return; g.beginPath(); sg.forEach(function(r,i){ var px=X(r.x), py=Y(r[BHI]); if(i) g.lineTo(px,py); else g.moveTo(px,py); }); for(var i=sg.length-1;i>=0;i--) g.lineTo(X(sg[i].x),Y(sg[i][BLO])); g.closePath(); g.fill(); });
    }
    GH.forEach(function(gb){ ["ar","sat","nat","anc"].forEach(function(k){ if(!S[k]) return; var d=DEFS[k]; g.globalAlpha=(k==="sat"||k==="nat")?.32:.62; g.strokeStyle=d.col; g.lineWidth=Math.max(1,d.w-.7); g.setLineDash([2,4]); path(gb.rows,"_s"+d.f,Y,k); g.stroke(); }); g.setLineDash([]); g.globalAlpha=1; });
    if(S.lr&&S.lr.length){ g.strokeStyle=DEFS.lr.col; g.lineWidth=DEFS.lr.w; g.setLineDash(DEFS.lr.dash); path(S.lr,"lr",YL); g.stroke(); g.setLineDash([]); }
    LINES.forEach(function(k){ var rows=S[k]; if(!rows||!rows.length||(k==="mix"&&m.band==="ar")) return; var d=DEFS[k], f="_s"+d.f; g.strokeStyle=d.col; g.lineWidth=d.w; g.setLineDash(d.dash||[]); g.lineJoin="round"; path(rows,f,Y,k); g.stroke(); g.setLineDash([]);
      if(rows.length<=36&&k!=="mix"){ g.fillStyle=d.col; rows.forEach(function(r){ var v=r[f]; if(v===null||v===undefined) return; g.beginPath(); g.arc(X(r.x),Y(v),1.8,0,6.283); g.fill(); }); } });
    ["w2048","w257"].forEach(function(k){ if(!m.has[k]||!visible(k)||!PR.length) return; var d=DEFS[k]; g.strokeStyle=d.col; g.lineWidth=d.w; g.beginPath(); var on=false,prev=null;
      PR.forEach(function(p){ var v=p[k]; if(v===null){ on=false; return; } var px=X(p.x), py=Y(v); if(!on||(prev&&p.x-prev.x>60000)){ g.moveTo(px,py); on=true; } else g.lineTo(px,py); prev=p; }); g.stroke();
      if(PR.length<=160){ g.fillStyle=d.col; PR.forEach(function(p){ if(p[k]===null) return; g.beginPath(); g.arc(X(p.x),Y(p[k]),k==="w257"?2.6:1.8,0,6.283); g.fill(); }); } });
    if(RAW.length){ g.globalAlpha=RAW.length>1500?.28:.42; RAW.forEach(function(r){ g.fillStyle=DEFS[r.k].col; g.fillRect(X(r.x)-1,Y(r.v)-1,2.2,2.2); }); g.globalAlpha=1; }
    HELD.forEach(function(p){ g.beginPath(); g.arc(X(p.x),Y(p.ce),4.2,0,6.283); if(p.ab){ g.strokeStyle="#3fb950"; g.lineWidth=1.6; g.stroke(); } else { g.fillStyle="#3fb950"; g.strokeStyle="#0b0e14"; g.lineWidth=1.5; g.fill(); g.stroke(); } });
    g.restore();
    GH.forEach(function(gb){ var k=S.anc?"anc":S.ar?"ar":S.sat?"sat":S.nat?"nat":null; if(!k) return; var fk="_s"+DEFS[k].f, rows=gb.rows.filter(function(r){return r.x>=x0&&r.x<=x1&&r[fk]!==null&&r[fk]!==undefined;}); if(!rows.length) return;
      var r=rows[rows.length-1], lx=X(r.x), ly=Y(r[fk]), txt=narrow?"abandoned":"abandoned branch (rolled back)"; g.fillStyle="#8b98ab"; g.textAlign=lx-g.measureText(txt).width-6<pad.l?"left":"right"; if(ly>pad.t+10&&ly<pad.t+ph-4) g.fillText(txt,g.textAlign==="right"?lx-6:lx+6,ly-9); });
    // best markers
    var bests=[]; if(m.lin==="standard"&&m.has.w257&&visible("w257")&&PR.length){ var bp=PR.reduce(function(a,b){return b.w257<a.w257?b:a;}); bests.push([bp.w257,narrow?"best "+F(bp.w257,4):"best 257 "+F(bp.w257,4)+" @"+N(bp.step),DEFS.w257.col]); }
    var hb=m.h&&m.h.best; if(hb&&hb.anc&&visible("anc")&&num(hb.anc.value)!==null){ var bv=num(hb.anc.value); if(bv>=ymin&&bv<=ymax) bests.push([bv,narrow?"best "+F(bv,3):"best AR full-stack "+F(bv,3)+" @"+N(hb.anc.step),DEFS.anc.col]); }
    if(REF){ var rtxt=narrow?"unigram "+F(REF.v,3):"unigram baseline "+F(REF.v,3)+" · AR above = worse than no context", ryy=Math.round(Y(REF.v))+.5, rtw=g.measureText(rtxt).width+8, rlx=pad.l+pw-rtw-4, rly=ryy-9<pad.t+6?ryy+9:ryy-9; g.fillStyle="rgba(11,14,20,.8)"; g.fillRect(rlx,rly-7,rtw,14); g.fillStyle="#c9d1d9"; g.textAlign="left"; g.fillText(rtxt,rlx+4,rly); }
    bests.forEach(function(b){ var yy=Math.round(Y(b[0]))+.5; g.strokeStyle=b[2]; g.globalAlpha=.55; g.setLineDash([2,4]); g.beginPath(); g.moveTo(pad.l,yy); g.lineTo(pad.l+pw,yy); g.stroke(); g.setLineDash([]); g.globalAlpha=1;
      var ly=yy-9<pad.t+6?yy+9:yy-9, tw=g.measureText(b[1]).width+8; g.fillStyle="rgba(11,14,20,.8)"; g.fillRect(pad.l+2,ly-7,tw,14); g.fillStyle=b[2]; g.textAlign="left"; g.fillText(b[1],pad.l+6,ly); });
    // end-of-line value labels, de-overlapped
    var ends=[]; LINES.slice().reverse().forEach(function(k){ var rows=S[k]; if(!rows||!rows.length||k==="mix") return; var f="_s"+DEFS[k].f, r=null; for(var i=rows.length-1;i>=0;i--){ if(rows[i].x<=x1&&rows[i][f]!==null&&rows[i][f]!==undefined){ r=rows[i]; break; } } if(!r) return; ends.push({y:Y(r[f]),x:X(r.x),t:F(r[f],3),c:DEFS[k].col}); });
    ends.sort(function(a,b){return a.y-b.y;}); for(var ei=1;ei<ends.length;ei++){ if(ends[ei].y-ends[ei-1].y<12) ends[ei].y=ends[ei-1].y+12; }
    ends.forEach(function(e){ if(e.y<pad.t||e.y>pad.t+ph) return; var tw=g.measureText(e.t).width+8, lx=Math.min(e.x+5,pad.l+pw-tw-1); g.fillStyle="rgba(11,14,20,.82)"; g.fillRect(lx,e.y-7,tw,14); g.fillStyle=e.c; g.textAlign="left"; g.fillText(e.t,lx+4,e.y); });
    cv.setAttribute("aria-label",(m.lin==="rpv16"?"RPV16 training loss chart":"Standard 1.1B loss chart")+", steps "+N(x0)+" to "+N(x1)+". "+ends.map(function(e){return e.t;}).join(", "));
    // latest in-view value on each legend chip
    Array.prototype.forEach.call(seriesEl.querySelectorAll("b.v[data-v]"),function(el){ var k=el.getAttribute("data-v"), d=DEFS[k], v=null, i;
      if(k==="held"){ var hc=HELD.filter(function(p){return !p.ab;}), hp=(hc.length?hc:HELD); v=hp.length?F(hp[hp.length-1].ce,4):""; }
      else if(k==="w257"||k==="w2048"){ for(i=PR.length-1;i>=0;i--) if(PR[i][k]!==null){ v=F(PR[i][k],4); break; } }
      else { var rows=S[k], ff=k==="lr"?d.f:"_s"+d.f; if(rows) for(i=rows.length-1;i>=0;i--){ if(rows[i].x<=x1&&rows[i][ff]!==null&&rows[i][ff]!==undefined){ v=k==="lr"?E(rows[i][ff]):F(rows[i][ff],3); break; } } }
      el.textContent=v||""; });
    paintFoot(m,{x0:x0,x1:x1,dw:dw});
    paintModesLive(m);
    upd.textContent="updated "+ago(DATA.generated_at||DATA.polled_at);
  }
  function domainFor(m){
    if(ST.zoom&&ST.zoom.lin===m.lin) return [ST.zoom.x0,ST.zoom.x1];
    var r=ST.range[m.lin], a=m.first, b=m.last;
    if(r==="run"&&m.ghostEnd&&visible("ghost")) b=Math.max(b,m.ghostEnd);
    if(r==="tail"&&m.T.length) a=m.T[0].step;
    else if(r==="branch"&&m.branches.length){ var gb=m.branches[m.branches.length-1], sp=Math.max(b-gb.from,2048); a=Math.max(m.first,Math.floor(gb.from-sp*0.35)); b=Math.ceil(gb.from+sp*1.12); }
    else if(/^\d+[hd]$/.test(r||"")){ var hrs=parseInt(r,10)*(r.slice(-1)==="d"?24:1), te=timeAt(m,b), s=te===null?null:stepAt(m,te-hrs*3600); if(s!==null) a=Math.max(a,Math.floor(s)); }
    if(b-a<16) a=Math.max(m.first,b-16);
    return [a,b];
  }
  function paintFoot(m,v){
    var L=DATA.latest||{}, h=m.h||{}, meta=h.meta||{}, rows=[];
    function row(k,html){ rows.push("<div><em>"+esc(k)+"</em> "+html+"</div>"); }
    if(m.lin==="rpv16"){
      if(m.H.length) row("history",N(m.H.length)+" × "+N(m.wH)+"-step buckets · steps "+N(m.H[0].s0)+"–"+N(m.H[m.H.length-1].s1)+(m.R.length?" · "+N(m.wR)+"-step detail for the last "+compact(m.R[m.R.length-1].s1-m.R[0].s0)+" steps":"")+(meta.built_at?" · built "+ago(meta.built_at):""));
      else row("history","collector has not attached the full-run history yet — showing the live tail ("+N(m.T.length)+" steps since the last restart)");
      var lastRes=m.resumes.length?m.resumes[m.resumes.length-1]:null;
      if(lastRes){ var after=null,before=null; if(m.has.anc){ var src=m.R.length?m.R:m.H; var a=src.filter(function(r){return r.s0>lastRes.step&&r.anc!==null;}), b=src.filter(function(r){return r.s1<=lastRes.step&&r.s1>lastRes.step-2048&&r.anc!==null;}); after=a.length?wmean(a,"anc","anc_n"):null; before=b.length?wmean(b,"anc","anc_n"):null; }
        row("last restart","@"+N(lastRes.step)+(lastRes.t?" · "+fmtStamp(lastRes.t):"")+(after!==null&&before!==null?" · AR full-stack "+F(before,3)+" → "+F(after,3)+" since":"")+" · "+N(m.resumes.length)+" restarts on this lineage"); }
      var hb=h.best||{}; if(hb.anc&&num(hb.anc.value)!==null) row("best AR full-stack",F(hb.anc.value,3)+" @"+N(hb.anc.step)); if(hb.ar&&num(hb.ar.value)!==null) row("best AR block-local",F(hb.ar.value,3)+" @"+N(hb.ar.step));
      var hcur=m.held.filter(function(p){return !p.ab;}), hab=m.held.filter(function(p){return p.ab;});
      if(hcur.length){ var lh=hcur[hcur.length-1]; row("held-out CE",F(lh.ce,4)+" @"+N(lh.step)+(lh.protocol?" · "+esc(lh.protocol):"")+(hab.length?" · abandoned branch "+F(hab[hab.length-1].ce,4)+" @"+N(hab[hab.length-1].step):"")); }
      else if(hab.length){ var la=hab[hab.length-1]; row("held-out CE","abandoned branch "+F(la.ce,4)+" @"+N(la.step)); }
      row("mixed EMA (chip value)",F(L.ema,3)+" · last step "+F(L.loss,3)+(L.objective?" ("+esc(String(L.objective).toUpperCase())+")":""));
      if(meta.legacy_ar_until_step) row("note","steps before "+N(meta.legacy_ar_until_step)+" were AR-only (no objective field logged)");
    } else {
      var P=m.probes; row("3090 archive",N(P.length)+" recovered probes · "+N(P.filter(function(p){return p.basis==="explicit_probe_step";}).length)+" exact-step · through "+(P.length?N(P[P.length-1].step):"—"));
      var rm=DATA.recovered_history_meta||{}; if(rm.provenance) row("provenance",esc(rm.provenance));
      if(m.H.length) row("GB10 1.1B history",N(m.H.length)+" buckets · steps "+N(m.H[0].s0)+"–"+N(m.H[m.H.length-1].s1));
    }
    if(v) row("view","steps "+N(v.x0)+"–"+N(v.x1)+" · ≈"+compact(Math.max(1,v.dw))+" steps per point");
    foot.innerHTML=rows.join("");
  }

  /* ---------- hover + drag-to-zoom ---------- */
  function nearestRow(rows,x){ var b=null,bd=Infinity; for(var i=0;i<rows.length;i++){ var d=Math.abs(rows[i].x-x); if(d<bd){ bd=d; b=rows[i]; } } return b; }
  function hover(ev){
    if(!GEOM||!DATA) return; var r=cv.getBoundingClientRect(), pt=ev.touches?ev.touches[0]:ev, px=pt.clientX-r.left, py=pt.clientY-r.top, G=GEOM, m=G.m;
    if(px<G.pad.l||px>G.pad.l+G.pw){ tip.style.display="none"; return; }
    var s=G.x0+(px-G.pad.l)/G.pw*(G.x1-G.x0), tol=(G.x1-G.x0)*0.03, rows=[], t=timeAt(m,s);
    rows.push("<b>step "+N(s)+"</b>"+(t?" <span style='color:#8b98ab'>"+fmtStamp(t)+"</span>":""));
    LINES.slice().reverse().forEach(function(k){ var S=G.S[k]; if(!S||!S.length) return; var p=nearestRow(S,s), d=DEFS[k]; if(!p||Math.abs(p.x-s)>Math.max(tol,(p.s1-p.s0))) return; var v=p[d.f]; if(v===null||v===undefined) return;
      var n=d.nf?num(p[d.nf]):null; if(k==="mix"&&G.m.band==="ar"){ if(p.ar_p10!=null) rows.push("<span style='color:#8b98ab'>▒ AR p10–p90 "+F(p.ar_p10,3)+"–"+F(p.ar_p90,3)+"</span>"); return; }
      rows.push("<span style='color:"+d.col+"'>●</span> "+esc(d.short)+" <b>"+F(v,3)+"</b>"+(k==="mix"&&p.lo!==null?" <span style='color:#8b98ab'>p10–p90 "+F(p.lo,2)+"–"+F(p.hi,2)+"</span>":"")+(n?" <span style='color:#8b98ab'>n="+N(n)+"</span>":"")); });
    var any=G.S.mix&&G.S.mix.length?nearestRow(G.S.mix,s):(G.S.anc&&G.S.anc.length?nearestRow(G.S.anc,s):null);
    if(any) rows.push("<span style='color:#8b98ab'>steps "+N(any.s0)+"–"+N(any.s1)+(any.lr!==null?" · lr "+E(any.lr):"")+(any.tps?" · "+compact(any.tps)+" tok/s":"")+"</span>");
    (G.GH||[]).forEach(function(gb){ var p=nearestRow(gb.rows,s); if(!p||Math.abs(p.x-s)>Math.max(tol,p.s1-p.s0)) return; var bits=[]; ["anc","ar","sat","nat"].forEach(function(k){ if(G.S[k]&&p[DEFS[k].f]!==null&&p[DEFS[k].f]!==undefined) bits.push(DEFS[k].short+" "+F(p[DEFS[k].f],3)); }); if(bits.length) rows.push("<span style='color:#8b98ab'>┄ abandoned branch: "+bits.join(" · ")+"</span>"); });
    G.HELD.forEach(function(p){ if(Math.abs(p.x-s)<tol) rows.push("<span style='color:#3fb950'>"+(p.ab?"○":"●")+"</span> held-out <b>"+F(p.ce,4)+"</b> @"+N(p.step)+(p.ab?" · abandoned branch":"")+(p.protocol?" · "+esc(p.protocol):"")+(p.tokens?" · "+compact(p.tokens)+" tok":"")); });
    if(G.PR.length){ var pp=nearestRow(G.PR,s); if(pp&&Math.abs(pp.x-s)<tol) rows.push("<span style='color:"+DEFS.w257.col+"'>●</span> held-out 257 <b>"+F(pp.w257,4)+"</b> · 2048 "+F(pp.w2048,4)+" <span style='color:#8b98ab'>@"+N(pp.step)+(pp.basis==="explicit_probe_step"?" · exact step":pp.basis==="archive_checkin_step"?" · check-in step":"")+"</span>"); }
    tip.innerHTML=rows.join("<br>"); tip.style.display="block";
    var tw=tip.offsetWidth, left=px+14; if(left+tw>r.width-4) left=Math.max(4,px-tw-14); tip.style.left=left+"px"; tip.style.top=Math.max(4,Math.min(py-10,r.height-tip.offsetHeight-4))+"px";
    drawCross(px);
  }
  var crossEl=document.createElement("div"); crossEl.className="xh"; wrapEl.appendChild(crossEl);
  function drawCross(px){ if(!GEOM) return; crossEl.style.display="block"; crossEl.style.left=Math.round(px)+"px"; crossEl.style.top=GEOM.pad.t+"px"; crossEl.style.height=GEOM.ph+"px"; }
  function hideHover(){ tip.style.display="none"; crossEl.style.display="none"; }
  var drag=null;
  cv.addEventListener("pointerdown",function(ev){ if(ev.pointerType!=="mouse"||ev.button!==0||!GEOM) return; var r=cv.getBoundingClientRect(), px=ev.clientX-r.left; if(px<GEOM.pad.l||px>GEOM.pad.l+GEOM.pw) return; drag={a:px,b:px}; try{cv.setPointerCapture(ev.pointerId);}catch(e){} });
  cv.addEventListener("pointermove",function(ev){ if(drag){ var r=cv.getBoundingClientRect(); drag.b=Math.max(GEOM.pad.l,Math.min(GEOM.pad.l+GEOM.pw,ev.clientX-r.left)); var lo=Math.min(drag.a,drag.b), w=Math.abs(drag.b-drag.a); selEl.style.display=w>3?"block":"none"; selEl.style.left=lo+"px"; selEl.style.width=w+"px"; selEl.style.top=GEOM.pad.t+"px"; selEl.style.height=GEOM.ph+"px"; } if(ev.pointerType==="mouse") hover(ev); });
  cv.addEventListener("pointerup",function(ev){ if(!drag) return; var d=drag; drag=null; selEl.style.display="none"; if(Math.abs(d.b-d.a)<8||!GEOM) return;
    var G=GEOM, s=function(px){ return G.x0+(px-G.pad.l)/G.pw*(G.x1-G.x0); }, a=s(Math.min(d.a,d.b)), b=s(Math.max(d.a,d.b)); if(b-a<16) return; ST.zoom={lin:G.m.lin,x0:Math.floor(a),x1:Math.ceil(b)}; hideHover(); draw(); });
  cv.addEventListener("pointercancel",function(){ drag=null; selEl.style.display="none"; });
  cv.addEventListener("dblclick",function(){ if(ST.zoom){ ST.zoom=null; draw(); } });
  cv.addEventListener("touchstart",hover,{passive:true}); cv.addEventListener("touchmove",hover,{passive:true});
  cv.addEventListener("mouseleave",function(){ if(!drag) hideHover(); });

  /* ---------- headline for the RPV16 card chip + pill ---------- */
  function headline(){
    if(!DATA||DATA.ok===false) return null; var m; try{ m=model("rpv16"); }catch(e){ return null; }
    if(m.last===null) return null;
    var k=m.has.anc?"anc":(m.has.ar?"ar":null), W=Math.min(1024,trendW(m));
    if(k){ var s=winStats(m,k,W); if(!s) return null;
      var span=Math.min(40000,m.last-m.first), rr=rowsFor(m,k,m.last-span,m.last,Math.max(1,span/48)), sf=smooth(rr,k), series=rr.filter(function(r){return r.x>=m.last-span;}).map(function(r){return r[sf];}).slice(-48);
