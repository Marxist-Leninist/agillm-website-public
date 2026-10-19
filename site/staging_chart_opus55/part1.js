    if(m.T.length&&(!rHas||tw<m.wR)&&(!hHas||dw<m.wH)){
      var tb=bucketTail(m.T,tw); if(has(tb)){ out=tb.slice(); cut=tb[0].s0; }
    }
    if(rHas&&dw<m.wH){ var rr=mergeRows(m.R.filter(function(r){return r.s1<cut;}),Math.max(1,Math.round(dw/m.wR))); if(rr.length){ out=rr.concat(out); cut=Math.min(cut,rr[0].s0); } }
    if(hHas){ var hh=mergeRows(m.H.filter(function(r){return r.s1<cut;}),Math.max(1,Math.round(dw/m.wH))); out=hh.concat(out); }
    var lo=-1,hi=out.length; for(var i=0;i<out.length;i++){ if(out[i].x<x0) lo=i; else break; } for(var j=out.length-1;j>=0;j--){ if(out[j].x>x1) hi=j; else break; }
    return out.slice(Math.max(0,lo),Math.min(out.length,hi+1)).filter(function(r){ var v=r[f]; return v!==null&&v!==undefined; });
  }
  /* Window statistic: mean over the last W steps vs the W before, weighted by sample counts, with a row-scatter standard error. */
  function winStats(m,k,W){
    var d=DEFS[k]; if(!d||!d.f||m.last===null) return null;
    var srcs=[]; if(m.R.length) srcs.push(m.R); if(m.T.length) srcs.push(bucketTail(m.T,16)); if(m.H.length) srcs.push(m.H);
    for(var si=0;si<srcs.length;si++){
      var rows=srcs[si].filter(function(r){ return r[d.f]!==null&&r[d.f]!==undefined; });
      if(!rows.length) continue;
      var end=rows[rows.length-1].s1, a=rows.filter(function(r){return r.x>end-W;}), b=rows.filter(function(r){return r.x<=end-W&&r.x>end-2*W;});
      if(!a.length) continue;
      var ma=wmean(a,d.f,d.nf), mb=b.length?wmean(b,d.f,d.nf):null;
      var se=function(g,mu){ if(g.length<2) return null; var s=0; g.forEach(function(r){ s+=Math.pow(r[d.f]-mu,2); }); return Math.sqrt(s/(g.length-1))/Math.sqrt(g.length); };
      var sa=se(a,ma), sb=b.length?se(b,mb):null, sd=(sa!==null&&sb!==null)?Math.sqrt(sa*sa+sb*sb):null;
      return {value:ma,prev:mb,delta:mb===null?null:ma-mb,se:sd,end:end,W:W,n:sumk(a,d.nf),rows:rows};
    }
    return null;
  }
  function trendW(m){ var span=(m.last||0)-(m.first||0); return Math.max(64,Math.min(5000,Math.round(span/4))); }

  /* ---------- chrome: lineage switch, ranges, legend, quick filters, summary ---------- */
  var sumEl=$("agi44-sum"); if(!sumEl){ sumEl=document.createElement("div"); sumEl.id="agi44-sum"; sumEl.setAttribute("aria-live","polite"); var h4=bodyEl.querySelector("h4"); bodyEl.insertBefore(sumEl,h4?h4.nextSibling:bodyEl.firstChild); }
  var qfEl=document.createElement("span"); qfEl.className="qf";
  var wrapEl=cv.parentNode, selEl=document.createElement("div"); selEl.className="sel"; wrapEl.appendChild(selEl);
  var titleEl=(bodyEl.querySelector("h4 > span")||null);
  var notes=Array.prototype.filter.call(bodyEl.children,function(n){return n.classList&&n.classList.contains("note")&&!n.id;});
  var liveNote=document.createElement("div"); liveNote.className="note"; liveNote.id="agi44-live-note";
  liveNote.textContent="RPV16 curves are training-batch cross-entropy (nats) from the GB10 trainer log, bucketed by step: AR full-stack = full forward on the anchor steps (the headline); AR block-local, SAT var and NAT = per-objective local-block CE; the grey band is the p10–p90 spread of every step's loss, whatever its objective. Held-out dots appear when an eval run is recorded. The Standard 1.1B / 3090 archive is a different model and is never drawn on this axis. Drag across the chart to zoom, double-click to reset.";
  bodyEl.appendChild(liveNote);
  function paintChrome(m){
    var lin=m.lin, sig=lin+"|"+m.branches.length+"|"+ORDER[lin].map(function(k){return k+(m.has[k]?1:0)+(visible(k)?1:0);}).join(",")+"|"+(ST.range[lin])+"|"+(ST.zoom?1:0)+"|"+(ST.lineage?1:0);
    if(sig===CHROME_SIG) return; CHROME_SIG=sig;
    if(titleEl) titleEl.textContent=lin==="rpv16"?"RPV16 2.0B · training CE per objective"+(window.innerWidth<620?"":" · lower is better"):"Standard 1.1B · held-out + training CE"+(window.innerWidth<620?"":" · lower is better");
    notes.forEach(function(n,i){ n.hidden=(lin==="rpv16"); }); liveNote.hidden=(lin!=="rpv16");
    // ranges bar: lineage switch + ranges + zoom reset
    rangesEl.innerHTML="";
    var seg=document.createElement("span"); seg.className="seg2"; seg.setAttribute("role","group"); seg.setAttribute("aria-label","Model lineage");
    LINEAGES.forEach(function(L){ var b=document.createElement("button"); b.type="button"; b.textContent=L[1]; b.className=lin===L[0]?"on":""; b.setAttribute("aria-pressed",String(lin===L[0]));
      b.addEventListener("click",function(){ ST.lineage=(L[0]===modelLineage())?null:L[0]; ST.zoom=null; save(); draw(); }); seg.appendChild(b); });
    rangesEl.appendChild(seg);
    RANGES[lin].forEach(function(r){ if(r[0]==="branch"&&!m.branches.length) return; var b=document.createElement("button"); b.type="button"; b.className="rg"+(!ST.zoom&&ST.range[lin]===r[0]?" on":""); b.textContent=r[1];
      b.addEventListener("click",function(){ ST.range[lin]=r[0]; ST.zoom=null; save(); draw(); }); rangesEl.appendChild(b); });
    if(ST.zoom){ var z=document.createElement("button"); z.type="button"; z.className="rg on"; z.textContent="reset zoom ×"; z.addEventListener("click",function(){ ST.zoom=null; draw(); }); rangesEl.appendChild(z); }
    rangesEl.appendChild(qfEl);
    // legend: only series with data on this lineage
    seriesEl.innerHTML="";
    ORDER[lin].forEach(function(k){ if(!m.has[k]) return; var d=DEFS[k], on=visible(k), l=document.createElement("button");
      l.type="button"; l.dataset.series=k; l.className="sr"+(on?"":" off"); l.setAttribute("aria-pressed",String(on));
      var leg=k==="mix"?(m.band==="ar"?"AR p10–p90 band":"mixed band"):(d.leg||d.short), tt=k==="mix"?(m.band==="ar"?"spread (p10–p90) of the per-step AR loss in each bucket":d.name):(k==="ref"&&m.ref?m.ref.label+" · "+m.ref.meaning:d.name);
      l.title=tt; l.innerHTML='<i style="background:'+d.col+(d.band?';height:8px;opacity:.55':'')+(k==="ghost"?';background:repeating-linear-gradient(90deg,#8b98ab 0 3px,transparent 3px 6px)':'')+(k==="ref"?';background:repeating-linear-gradient(90deg,#e6edf3 0 2px,transparent 2px 5px)':'')+'"></i>'+esc(leg)+' <b class="v" data-v="'+k+'"></b>';
      l.addEventListener("click",function(){ setVisible(k,!visible(k)); save(); draw(); }); seriesEl.appendChild(l); });
    var na=document.createElement("span"); na.className="sr na";
    if(lin==="rpv16") na.textContent="SAT fixed: not an RPV16 objective"+(m.has.anc?"":" · per-objective history loading from the collector…");
    else na.textContent=m.has.anc||m.has.ar?"":"GB10 training CE for the 1.1B is not in the feed yet — showing the 3090 archive probes";
    if(na.textContent) seriesEl.appendChild(na);
    // quick filters
    qfEl.innerHTML=""; qfEl.title="quick filters";
    GROUPS.forEach(function(g){ var keys=ORDER[lin].filter(function(k){return m.has[k]&&(g[0]==="all"||DEFS[k].grp===g[0]||(g[0]==="Mixed"&&DEFS[k].grp==="LR"));}); if(!keys.length) return;
      var b=document.createElement("button"); b.type="button"; b.className="qfb"; b.textContent=g[1];
      b.addEventListener("click",function(){ ORDER[lin].forEach(function(k){ setVisible(k,keys.indexOf(k)>=0); }); save(); draw(); }); qfEl.appendChild(b); });
  }
  function trendHtml(s){
    if(!s||s.delta===null) return '<i>—</i>';
    var sig=s.se!==null?2*s.se:0.02, d=s.delta, cls=d< -sig?"dn":d>sig?"up":"", arrow=d< -sig?"▼":d>sig?"▲":"≈";
    return '<i class="'+cls+'" title="mean of the last '+N(s.W)+' steps minus the '+N(s.W)+' before'+(s.se!==null?' · ±'+F(2*s.se,3)+' (2 s.e.)':'')+'">'+arrow+' '+(d>=0?"+":"−")+F(Math.abs(d),3)+' <span style="opacity:.7">/'+kfmt(s.W)+'</span></i>';
  }
  function paintSummary(m){
    var L=DATA.latest||{}, W=trendW(m), tiles=[];
    function tile(label,val,sub,col,title){ tiles.push('<div class="t"'+(title?' title="'+esc(title)+'"':'')+'><small>'+(col?'<span class="sw" style="background:'+col+'"></span>':'')+esc(label)+'</small><b>'+val+'</b>'+(sub||'<i>&nbsp;</i>')+'</div>'); }
    if(m.lin==="rpv16"){
      tile("step",N(L.step!=null?L.step:m.last),'<i>'+compact(L.seen_tok!=null?L.seen_tok:L.seen_tokens)+' tokens</i>',null,"latest trainer step on GB10");
      ["anc","ar","sat","nat"].forEach(function(k){ if(!m.has[k]) return; var s=winStats(m,k,W); tile(DEFS[k].short,s?F(s.value,3):"—",trendHtml(s),DEFS[k].col,DEFS[k].name+" · mean of the last "+N(W)+" steps"+(s&&s.n?" · "+N(s.n)+" samples":"")); });
      if(m.ref&&(m.has.anc||m.has.ar)){ var ku=m.has.anc?"anc":"ar", su=winStats(m,ku,Math.min(1024,W)); if(su){ var du=su.value-m.ref.v;
        tile("vs unigram",(du>=0?"+":"−")+F(Math.abs(du),3),'<i class="'+(du>-0.25?"up":"")+'">baseline '+F(m.ref.v,3)+'</i>',"#e6edf3",DEFS[ku].name+" minus the unigram entropy "+F(m.ref.v,3)+" (a context-free model). Negative = better than ignoring context; a working LM sits well below it."); } }
      if(!m.has.anc&&!m.has.ar){ var ema=num(L.ema); tile("mixed EMA",F(ema,3),'<i>all objectives</i>',DEFS.mix.col,"EMA α=.03 of every step's loss, whatever its objective"); }
      if(m.branches.length){ var gb=m.branches[m.branches.length-1], k2=m.has.anc?"anc":(m.has.ar?"ar":null);
        if(k2){ var f2=DEFS[k2].f, nf2=DEFS[k2].nf, ok2=function(r){return r[f2]!==null&&r[f2]!==undefined;};
          var cur=(m.R.length?m.R:m.H).filter(function(r){return r.s0>gb.from&&ok2(r);}); if(!cur.length&&m.T.length) cur=bucketTail(m.T,64).filter(function(r){return r.s0>gb.from&&ok2(r);});
          if(cur.length){ var endS=cur[cur.length-1].s1, gh=gb.rows.filter(function(r){return r.s0>=gb.from&&r.s0<=endS&&ok2(r);}), mc=wmean(cur,f2,nf2), mg=gh.length?wmean(gh,f2,nf2):null;
            if(mc!==null&&mg!==null){ var dd=mc-mg; tile("vs abandoned",(dd>=0?"+":"−")+F(Math.abs(dd),3),'<i class="'+(dd<0?"dn":"up")+'">'+esc(k2==="anc"?"AR full":"AR local")+' · '+compact(endS-gb.from)+' steps</i>',"#6e7681","live branch minus the abandoned "+gb.label+" over the same steps ("+N(gb.from)+"–"+N(endS)+"), "+DEFS[k2].name); } } } }
      tile("LR",E(L.lr),'<i>'+compact(L.fullstack_tok_s_64!=null?L.fullstack_tok_s_64:L.raw_tok_s)+' tok/s</i>',DEFS.lr.col,"current learning rate · rolling-64 training throughput");
      if(SERVED&&num(SERVED.step)!==null){ var lag=num(L.step)!==null?L.step-SERVED.step:null; tile("CPU serving",N(SERVED.step),'<i class="'+(lag>0?"up":"")+'">'+(lag>0?compact(lag)+' steps behind':'current')+'</i>',"#3fb950","checkpoint loaded by the GETH CPU inference server"); }
    } else {
      var P=m.probes, lastP=P[P.length-1], best=P.reduce(function(a,b){return (!a||b.w257<a.w257)?b:a;},null);
      tile("held-out 257",lastP?F(lastP.w257,4):"—",'<i>'+(lastP?"@"+N(lastP.step):"")+'</i>',DEFS.w257.col,"last recovered 3090 held-out probe");
      tile("best 257",best?F(best.w257,4):"—",'<i>'+(best?"@"+N(best.step):"")+'</i>',DEFS.w257.col,"best recovered 3090 held-out probe");
      tile("held-out 2048",lastP?F(lastP.w2048,4):"—",'<i>3090 archive</i>',DEFS.w2048.col,"");
      ["anc","ar","sat","nat"].forEach(function(k){ if(!m.has[k]) return; var s=winStats(m,k,W); tile(DEFS[k].short,s?F(s.value,3):"—",trendHtml(s),DEFS[k].col,DEFS[k].name); });
      tile("probes",N(P.length),'<i>recovered · no interpolation</i>',null,"");
    }
    sumEl.innerHTML=tiles.join("");
  }

  /* ---------- drawing ---------- */
  function fitCanvas(){
    var cs=getComputedStyle(bodyEl), mh=parseFloat(cs.maxHeight); if(!isFinite(mh)||mh<=0||bodyEl.offsetParent===null) return; // full page: CSS decides
    var top=wrapEl.offsetTop, avail=Math.floor(mh-top-(parseFloat(cs.borderBottomWidth)||0)-(parseFloat(cs.borderTopWidth)||0)-2), lo=window.innerWidth<620?210:240, h=Math.max(lo,Math.min(560,avail));
    if(Math.abs(cv.getBoundingClientRect().height-h)>1) cv.style.height=h+"px";
  }
  /* Count-weighted centred rolling mean so sparse objectives are not drawn as noise: at least MINN samples per plotted point. */
  var MINN={anc:160,ar:320,sat:160,nat:160,mix:128};
  var SMOOTH_SPAN=Infinity;
  function smooth(rows,k){ var d=DEFS[k], f=d.f, nf=d.nf, need=MINN[k], key="_s"+f, i,j,s,w,add,jmax;
    for(i=0;i<rows.length;i++){ if(rows[i][f]===null||rows[i][f]===undefined){ rows[i][key]=null; continue; }
      s=0; w=0; add=function(r){ var v=r[f]; if(v===null||v===undefined) return; var c=nf?num(r[nf]):1; if(c===null) c=1; if(!c) return; s+=v*c; w+=c; };
      jmax=Math.min(12,Math.floor(SMOOTH_SPAN/Math.max(1,rows[i].w||1)));
      add(rows[i]); for(j=1;need&&w<need&&j<=jmax;j++){ if(i-j>=0) add(rows[i-j]); if(i+j<rows.length) add(rows[i+j]); }
      rows[i][key]=w?s/w:rows[i][f]; }
    if(k==="mix"){ ["lo","hi","ar_p10","ar_p90"].forEach(function(e){ for(i=0;i<rows.length;i++){ var ss=0,ww=0,jm=Math.min(12,Math.floor(SMOOTH_SPAN/Math.max(1,rows[i].w||1))),acc=function(r){ var v=r[e],c=num(r.n)||1; if(v===null||v===undefined) return; ss+=v*c; ww+=c; };
      acc(rows[i]); for(j=1;ww<need&&j<=jm;j++){ if(i-j>=0) acc(rows[i-j]); if(i+j<rows.length) acc(rows[i+j]); } rows[i]["_s"+e]=ww?ss/ww:rows[i][e]; } }); }
    return key; }
  function draw(){
    if(!DATA||!ROOT.classList.contains("open")) return;
    var m=model(); paintChrome(m); paintSummary(m); fitCanvas();
    var dpr=window.devicePixelRatio||1, W=cv.clientWidth||600, H=cv.clientHeight||300;
    if(cv.width!==Math.round(W*dpr)||cv.height!==Math.round(H*dpr)){ cv.width=Math.round(W*dpr); cv.height=Math.round(H*dpr); }
    var g=cv.getContext("2d"); g.setTransform(dpr,0,0,dpr,0,0); g.clearRect(0,0,W,H);
    g.font="10.5px ui-sans-serif,system-ui,sans-serif"; g.textBaseline="middle";
    if(m.first===null||m.last===null){ g.fillStyle="#8b98ab"; g.textAlign="center"; g.fillText(DATA.ok===false?"training feed unavailable":"waiting for training data…",W/2,H/2); GEOM=null; paintFoot(m,null); return; }
    var dom=domainFor(m), x0=dom[0], x1=Math.max(dom[1],dom[0]+1);
    var narrow=W<520, showLR=visible("lr")&&m.has.lr;
    var pad={l:narrow?40:46,r:showLR?(narrow?40:48):12,t:18,b:m.TL.length>=2?34:22}, pw=Math.max(40,W-pad.l-pad.r), ph=Math.max(40,H-pad.t-pad.b);
    var dw=(x1-x0)/Math.max(40,pw/2.5); SMOOTH_SPAN=(x1-x0)*0.04;
    var BLO=m.band==="ar"?"_sar_p10":"_slo", BHI=m.band==="ar"?"_sar_p90":"_shi";
    var S={}, ys=[], band=[], lrs=[];
    LINES.forEach(function(k){ if(!m.has[k]||!visible(k)) return; S[k]=rowsFor(m,k,x0,x1,dw); var f=smooth(S[k],k);
      S[k].forEach(function(r){ if(r.x>=x0&&r.x<=x1){ if(k!=="mix"||m.band!=="ar") ys.push(r[f]); if(k==="mix"){ var bl=r[BLO], bh=r[BHI]; if(bl!=null) band.push(bl); if(bh!=null) band.push(bh); } } }); });
    if(showLR){ S.lr=rowsFor(m,"lr",x0,x1,dw); S.lr.forEach(function(r){ if(r.lr>0) lrs.push(r.lr); }); }
    var GH=[], cmp=!ST.zoom&&ST.range[m.lin]==="branch"; if(m.branches.length&&visible("ghost")) m.branches.forEach(function(b){ var ov=Math.min(x1,b.to)-Math.max(x0,b.from); if(ov<=0||(!cmp&&ov<(x1-x0)*0.1)) return;
      var rows=b.rows.filter(function(r){return r.x>=x0-b.w&&r.x<=x1+b.w;}); if(!rows.length) return; rows=mergeRows(rows,Math.max(1,Math.round(dw/b.w)));
      GH.push({b:b,rows:rows}); ["ar","sat","nat","anc"].forEach(function(k){ if(!S[k]) return; var f=smooth(rows,k); if(!cmp&&(k==="sat"||k==="nat")) return; rows.forEach(function(r){ var v=r[f]; if(v!==null&&v!==undefined&&r.x>=x0&&r.x<=x1) ys.push(v); }); }); });
    var RAW=[]; if(m.T.length&&(x1-x0)<=4096) m.T.forEach(function(p){ if(p.step<x0||p.step>x1) return; var o=p.global_anchor===true?"anc":String(p.objective||"").toLowerCase(); if(!DEFS[o]||!(m.has[o]&&visible(o))){ if(!p.objective&&visible("mix")) o="mix"; else return; } RAW.push({x:p.step,v:p.loss,k:o}); });
    if(RAW.length){ var rv=RAW.map(function(r){return r.v;}).sort(nsort); ys.push(q(rv,.02),q(rv,.98)); }
    var HELD=(m.has.held&&visible("held"))?m.held.filter(function(p){return p.x>=x0&&p.x<=x1&&(!p.ab||visible("ghost"));}):[];
    HELD.forEach(function(p){ ys.push(p.ce); });
    var PR=m.probes.filter(function(p){return p.x>=x0&&p.x<=x1;});
    ["w257","w2048"].forEach(function(k){ if(m.has[k]&&visible(k)) PR.forEach(function(p){ if(p[k]!==null) ys.push(p[k]); }); });
    var REF=(m.ref&&visible("ref")&&(S.anc||S.ar))?m.ref:null; if(REF) ys.push(REF.v);
    ys.sort(nsort); band.sort(nsort);
    var ymin=ys.length?q(ys,.004):null, ymax=ys.length?q(ys,.996):null;
    if(band.length){ var bl=q(band,.02), bh=q(band,.98); ymin=ymin===null?bl:Math.min(ymin,bl); ymax=ymax===null?bh:Math.max(ymax,bh); }
    if(ymin===null){ ymin=0; ymax=1; }
    if(ymax-ymin<0.12){ var c=(ymin+ymax)/2; ymin=c-0.06; ymax=c+0.06; }
    var yp=(ymax-ymin)*0.07; ymin-=yp; ymax+=yp;
    lrs.sort(nsort); var lmin=lrs.length?Math.log10(lrs[0]):-5, lmax=lrs.length?Math.log10(lrs[lrs.length-1]):-3;
    if(lmax-lmin<0.3){ var lc=(lmax+lmin)/2; lmin=lc-0.25; lmax=lc+0.25; } else { var lp=(lmax-lmin)*0.08; lmin-=lp; lmax+=lp; }
    var X=function(s){return pad.l+(s-x0)/(x1-x0)*pw;}, Y=function(v){return pad.t+(ymax-v)/(ymax-ymin)*ph;}, YL=function(v){return pad.t+(lmax-Math.log10(v))/(lmax-lmin)*ph;};
    GEOM={m:m,X:X,Y:Y,x0:x0,x1:x1,pad:pad,pw:pw,ph:ph,S:S,HELD:HELD,PR:PR,GH:GH,W:W,H:H};
    // y grid
    var ystep=niceStep((ymax-ymin)/Math.max(3,Math.floor(ph/46))), yd=ystep<0.1?2:1;
    for(var y=Math.ceil(ymin/ystep)*ystep;y<=ymax+1e-9;y+=ystep){ var yy=Math.round(Y(y))+.5; g.strokeStyle="#1d2536"; g.lineWidth=1; g.beginPath(); g.moveTo(pad.l,yy); g.lineTo(pad.l+pw,yy); g.stroke(); g.fillStyle="#8b98ab"; g.textAlign="right"; g.fillText(y.toFixed(yd),pad.l-6,yy); }
    if(showLR&&lrs.length){ g.textAlign="left"; g.fillStyle=DEFS.lr.col; var mult=(lmax-lmin)<1.2?[1,2,5]:[1];
      for(var e=Math.floor(lmin)-1;e<=Math.ceil(lmax);e++){ mult.forEach(function(k){ var v=k*Math.pow(10,e), lv=Math.log10(v); if(lv<lmin||lv>lmax) return; g.fillText(k+"e"+e,pad.l+pw+5,YL(v)); }); } }
    // x grid: steps, then local time
    var span=x1-x0, nx=Math.max(2,Math.floor(pw/(narrow?80:110))), xs=niceStep(span/nx), tspan=(timeAt(m,x1)||0)-(timeAt(m,x0)||0);
    g.textAlign="center"; var lastDay=null;
    for(var xv=Math.ceil(x0/xs)*xs;xv<=x1;xv+=xs){ var xx=Math.round(X(xv))+.5; g.strokeStyle="#1d2536"; g.beginPath(); g.moveTo(xx,pad.t); g.lineTo(xx,pad.t+ph); g.stroke();
      g.fillStyle="#8b98ab"; g.fillText(fmtStep(xv,span),xx,pad.t+ph+9);
      if(m.TL.length>=2&&xv>=m.TL[0][0]&&xv<=m.TL[m.TL.length-1][0]+1){ var tt=timeAt(m,xv); if(tt){ var dd=new Date(tt*1000), day=dd.toDateString(), lab=fmtTime(tt,tspan);
        if(tspan<=129600&&day!==lastDay) lab=dd.toLocaleDateString(undefined,{day:"numeric",month:"short"})+" "+lab; lastDay=day; g.fillStyle="#6b7686"; g.fillText(lab,xx,pad.t+ph+22); } } }
