(function(){
  "use strict";
  /* AGILLM loss chart v2 (2026-09-23, cowork-opus55-chartfix).
     One model lineage per axis: RPV16 2.0B on GB10, or Standard 1.1B (3090 archive
     probes + GB10 continuation). Full-run history comes from /api/training `history`
     (agillm.rpv16.curve-history.v1, 512-step buckets + 64-step `recent`), the dense
     tail from `train`. Headline = AR full-stack CE; AR / SAT var / NAT per objective;
     the mixed-objective loss is only a faint p10-p90 band. */
  var $=function(id){return document.getElementById(id);};
  var ROOT=$("agi44-curves"), pill=$("agi44-curves-pill"), lbl=$("agi44-curves-lbl"), spark=$("agi44-spark"),
      cv=$("agi44-canvas"), tip=$("agi44-tip"), foot=$("agi44-foot"), upd=$("agi44-upd"), nowEl=$("agi44-now"),
      rangesEl=$("agi44-ranges"), seriesEl=$("agi44-series"), bodyEl=$("agi44-curves-body");
  if(!ROOT||!cv||!bodyEl) return;
  var FIX_STEP=2785394; // Standard 1.1B lineage: 2026-09-06 rollback + fused-CE gradient fix
  var DATA=null, SERVED=null, GEOM=null, MODEL=null, CHROME_SIG="";

  /* ---------- helpers ---------- */
  function num(v){ if(v===null||v===undefined||v===""||typeof v==="boolean") return null; var n=Number(v); return isFinite(n)?n:null; }
  function N(v){ var n=num(v); return n===null?"—":Math.round(n).toLocaleString(); }
  function F(v,d){ var n=num(v); return n===null?"—":n.toFixed(d===undefined?3:d); }
  function E(v){ var n=num(v); return n===null?"—":n.toExponential(2).replace("e-0","e-").replace("e+0","e"); }
  function esc(t){ return String(t==null?"":t).replace(/[<>&"]/g,function(c){return {"<":"&lt;",">":"&gt;","&":"&amp;",'"':"&quot;"}[c];}); }
  function ago(t){ if(!t) return "—"; var s=Math.max(0,Math.floor(Date.now()/1000-t)); return s<90?s+"s ago":s<5400?Math.round(s/60)+"m ago":s<172800?Math.round(s/3600)+"h ago":Math.round(s/86400)+"d ago"; }
  function kfmt(v){ var n=num(v); if(n===null) return "—"; return n>=1000?(Math.round(n/100)/10+"k").replace(".0k","k"):String(Math.round(n)); }
  function compact(v){ var n=num(v); if(n===null) return "—"; var a=Math.abs(n); return a>=1e9?(n/1e9).toFixed(2)+"B":a>=1e6?(n/1e6).toFixed(1)+"M":a>=1e4?(n/1e3).toFixed(1)+"k":Math.round(n).toLocaleString(); }
  function byX(a,b){ return a.x-b.x; }
  function nsort(a,b){ return a-b; }
  function q(sorted,p){ if(!sorted.length) return null; var i=(sorted.length-1)*p, lo=Math.floor(i), hi=Math.ceil(i); return sorted[lo]+(sorted[hi]-sorted[lo])*(i-lo); }
  function mean(a){ if(!a.length) return null; var s=0; for(var i=0;i<a.length;i++) s+=a[i]; return s/a.length; }
  function niceStep(r){ var p=Math.pow(10,Math.floor(Math.log10(r||1))), m=r/p; return (m<1.5?1:m<3.5?2:m<7.5?5:10)*p; }
  function fmtStep(v,span){ if(span>=150000) return Math.round(v/1000)+"k"; if(span>=15000) return (v/1000).toFixed(1).replace(/\.0$/,"")+"k"; return Math.round(v).toLocaleString(); }
  function fmtTime(t,span){ if(!t) return ""; var d=new Date(t*1000);
    if(span>129600) return d.toLocaleDateString(undefined,{day:"numeric",month:"short"});
    return d.toLocaleTimeString(undefined,{hour:"2-digit",minute:"2-digit"}); }
  function fmtStamp(t){ if(!t) return ""; var d=new Date(t*1000); return d.toLocaleDateString(undefined,{day:"numeric",month:"short"})+" "+d.toLocaleTimeString(undefined,{hour:"2-digit",minute:"2-digit"}); }

  /* ---------- series ---------- */
  var DEFS={
    anc:{name:"AR · full-stack CE",short:"AR full-stack",col:"#58a6ff",w:2.5,grp:"AR",f:"anc",nf:"anc_n",tipn:"anchor steps"},
    ar:{name:"AR · block-local CE",short:"AR block-local",col:"#a5d6ff",w:1.3,grp:"AR",f:"ar",nf:"ar_n",dash:[5,3]},
    sat:{name:"SAT var · CE",short:"SAT var",col:"#d2a8ff",w:1.7,grp:"SAT",f:"sat",nf:"sat_n"},
    nat:{name:"NAT · CE",short:"NAT",col:"#ff7b72",w:1.7,grp:"NAT",f:"nat",nf:"nat_n"},
    mix:{name:"all objectives · mean + p10–p90 of every step's loss",short:"mixed",leg:"mixed band",col:"#8b98ab",w:1,grp:"Mixed",f:"loss",nf:"n",band:true},
    ref:{name:"unigram entropy (no context): AR above this line is worse than ignoring context",short:"unigram",leg:"unigram baseline",col:"#e6edf3",grp:"AR"},
    held:{name:"held-out CE (recorded eval runs)",short:"held-out",leg:"held-out eval",col:"#3fb950",grp:"AR",pts:true},
    w257:{name:"held-out 257-tok · 3090 archive",short:"held-out 257",col:"#58a6ff",w:2.2,grp:"AR",pts:true},
    w2048:{name:"held-out 2048-tok · 3090 archive",short:"held-out 2048",col:"#a5d6ff",w:1.4,grp:"AR",pts:true},
    lr:{name:"learning rate · right axis, log scale",short:"LR",leg:"LR (right axis)",col:"#d29922",w:1.2,grp:"LR",f:"lr",axis:"lr",dash:[4,3]},
    ghost:{name:"abandoned (rolled-back) branch · dashed, same colours",short:"abandoned",leg:"abandoned branch",col:"#6e7681",grp:"Mixed"}
  };
  var ORDER={rpv16:["anc","ar","sat","nat","mix","held","ref","ghost","lr"],standard:["w257","w2048","anc","ar","sat","nat","mix","lr"]};
  var LINES=["mix","ar","sat","nat","anc"]; // draw order, headline last
  var LINEAGES=[["rpv16","RPV16 2.0B · GB10"],["standard","Standard 1.1B"]];
  var RANGES={rpv16:[["run","whole run"],["branch","vs abandoned"],["7d","7 d"],["24h","24 h"],["6h","6 h"],["1h","1 h"],["tail","live tail"]],
              standard:[["all","all"],["7d","7 d"],["24h","24 h"]]};
  var GROUPS=[["all","show all"],["AR","AR"],["SAT","SAT"],["NAT","NAT"],["Mixed","mixed + LR"]];

  /* ---------- persisted view state (same localStorage key the layout runtime edits) ---------- */
  var ST={lineage:null,range:{rpv16:"run",standard:"all"},hide:{},zoom:null};
  var saved={}; try{ saved=JSON.parse(localStorage.getItem("agi44_curves")||"{}")||{}; }catch(e){ saved={}; }
  if(saved.open) ROOT.classList.add("open");
  if(saved.v2&&typeof saved.v2==="object"){
    if(saved.v2.range&&typeof saved.v2.range==="object") Object.keys(ST.range).forEach(function(k){ var r=saved.v2.range[k]; if(RANGES[k].some(function(x){return x[0]===r;})) ST.range[k]=r; });
    if(saved.v2.hide&&typeof saved.v2.hide==="object") ST.hide=saved.v2.hide;
  }
  function save(){ try{ var prev=JSON.parse(localStorage.getItem("agi44_curves")||"{}")||{}; prev.open=ROOT.classList.contains("open"); prev.v2={range:ST.range,hide:ST.hide,v:2026092301}; localStorage.setItem("agi44_curves",JSON.stringify(prev)); }catch(e){} }
  function defaultHidden(k){ if(k!=="mix") return false; var m=MODEL&&MODEL.lin===lineage()?MODEL:null; return !!(m&&(m.has.anc||m.has.ar)); }
  function visible(k){ var key=lineage()+":"+k; return (key in ST.hide)?!ST.hide[key]:!defaultHidden(k); }
  function setVisible(k,on){ ST.hide[lineage()+":"+k]=on?0:1; }
  pill.addEventListener("click",function(){ ROOT.classList.toggle("open"); save(); if(ROOT.classList.contains("open")) draw(); });

  /* The chart follows the model selector; a lineage picked in the chart sticks until the selector changes. */
  function modelLineage(){ try{ if(typeof modelKey!=="undefined"&&modelKey==="standard") return "standard"; }catch(e){} return "rpv16"; }
  function lineage(){ return ST.lineage||modelLineage(); }
  (function(){ var seg=$("modelseg"); if(!seg||!window.MutationObserver) return; var last=modelLineage();
    new MutationObserver(function(){ var now=modelLineage(); if(now!==last){ last=now; ST.lineage=null; ST.zoom=null; draw(); } }).observe(seg,{attributes:true,subtree:true,attributeFilter:["class"]}); })();

  /* ---------- data model ---------- */
  var ALIAS={step:"s1",mean:"loss",p10:"lo",p90:"hi",anchor:"anc",anchor_n:"anc_n",tok_s:"tps",grad_norm:"gn",ts:"t1",t:"t1"};
  var NUMK=["loss","lo","hi","ar","ar_n","sat","sat_n","nat","nat_n","anc","anc_n","lr","tps","gn","tok","ar_deep","ar_p10","ar_p90"];
  function tableRows(h){
    if(!h) return [];
    var rows=Array.isArray(h)?h:h.rows, cols=Array.isArray(h.cols)?h.cols:null;
    if(!Array.isArray(rows)) return [];
    return rows.map(function(r){ if(Array.isArray(r)){ if(!cols) return null; var o={}; for(var i=0;i<cols.length;i++) o[cols[i]]=r[i]; return o; } return (r&&typeof r==="object")?r:null; }).filter(Boolean);
  }
  function normRow(o){
    var r={}; for(var k in o) if(Object.prototype.hasOwnProperty.call(o,k)) r[ALIAS[k]||k]=o[k];
    var s0=num(r.s0), s1=num(r.s1); if(s0===null&&s1===null) return null; if(s0===null) s0=s1; if(s1===null) s1=s0;
    var out={s0:s0,s1:s1,x:(s0+s1)/2,t0:num(r.t0),t1:num(r.t1),n:num(r.n)};
    NUMK.forEach(function(k){ out[k]=num(r[k]); });
    [["ar","ar_n"],["sat","sat_n"],["nat","nat_n"],["anc","anc_n"]].forEach(function(p){ if(out[p[1]]===0) out[p[0]]=null; });
    if(out.t0===null) out.t0=out.t1;
    return out;
  }
  function inferW(rows){ if(rows.length<2) return null; var d=[]; for(var i=1;i<Math.min(rows.length,64);i++) d.push(rows[i].s0-rows[i-1].s0); d.sort(nsort); return Math.max(1,q(d,.5)); }
  function tailPts(src){ return (src||[]).filter(function(p){return p&&num(p.step)!==null&&num(p.loss)!==null;}).slice().sort(function(a,b){return a.step-b.step;}); }
  function bucketTail(pts,w){
    var out=[],cur=null,hasObj=pts.some(function(p){return !!p.objective;});
    function flush(){ if(!cur) return; var L=cur.L.slice().sort(nsort);
      var r={s0:cur.s0,s1:cur.s1,x:(cur.s0+cur.s1)/2,w:w,t0:cur.t0,t1:cur.t1,n:L.length,loss:mean(L),lo:q(L,.1),hi:q(L,.9),lr:cur.lr,tps:mean(cur.tps),tok:cur.tok,ema:cur.ema,src:"tail"};
      ["ar","sat","nat","anc"].forEach(function(k){ var a=cur.o[k]; r[k]=hasObj&&a.length?mean(a):null; r[k+"_n"]=hasObj?a.length:null; });
      if(hasObj&&cur.o.ar.length){ var A=cur.o.ar.slice().sort(nsort); r.ar_p10=q(A,.1); r.ar_p90=q(A,.9); } else { r.ar_p10=null; r.ar_p90=null; }
      out.push(r); }
    pts.forEach(function(p){ var b=Math.floor(p.step/w);
      if(!cur||cur.b!==b){ flush(); cur={b:b,s0:p.step,s1:p.step,t0:num(p.ts),t1:num(p.ts),L:[],o:{ar:[],sat:[],nat:[],anc:[]},tps:[],lr:null,tok:null,ema:null}; }
      cur.s1=p.step; var ts=num(p.ts); if(ts!==null){ if(cur.t0===null) cur.t0=ts; cur.t1=ts; }
      cur.L.push(p.loss);
      if(hasObj){ var o=String(p.objective||"").toLowerCase(); if(p.global_anchor===true) cur.o.anc.push(p.loss); else if(cur.o[o]) cur.o[o].push(p.loss); }
      if(num(p.lr)!==null) cur.lr=p.lr; if(num(p.tps)!==null) cur.tps.push(p.tps); if(num(p.tok)!==null) cur.tok=p.tok; if(num(p.ema)!==null) cur.ema=p.ema; });
    flush(); return out;
  }
  function wmean(grp,k,nk){ var s=0,w=0,i,g,v,ww;
    for(i=0;i<grp.length;i++){ g=grp[i]; v=g[k]; if(v===null||v===undefined||!isFinite(v)) continue; ww=nk?num(g[nk]):1; if(ww===null) ww=1; if(!ww) continue; s+=v*ww; w+=ww; }
    return w?s/w:null; }
  function sumk(grp,k){ var s=0,any=false; grp.forEach(function(g){ var v=num(g[k]); if(v!==null){ s+=v; any=true; } }); return any?s:null; }
  function mergeRows(rows,g){ if(g<=1||rows.length<2) return rows; var out=[];
    for(var i=0;i<rows.length;i+=g){ var grp=rows.slice(i,i+g), a=grp[0], z=grp[grp.length-1];
      var r={s0:a.s0,s1:z.s1,x:(a.s0+z.s1)/2,w:sumk(grp,"w")||(z.s1-a.s0+1),t0:a.t0,t1:z.t1,src:a.src,n:sumk(grp,"n"),loss:wmean(grp,"loss","n"),lo:wmean(grp,"lo","n"),hi:wmean(grp,"hi","n"),
             ar_deep:wmean(grp,"ar_deep","ar_n"),ar_p10:wmean(grp,"ar_p10","ar_n"),ar_p90:wmean(grp,"ar_p90","ar_n"),lr:z.lr,tps:wmean(grp,"tps",null),gn:wmean(grp,"gn",null),tok:z.tok,ema:z.ema};
      ["ar","sat","nat","anc"].forEach(function(k){ r[k]=wmean(grp,k,k+"_n"); r[k+"_n"]=sumk(grp,k+"_n"); });
      out.push(r); }
    return out; }
  function recoveredProbes(){
    var pts=Array.isArray(DATA.recovered_probes)?DATA.recovered_probes.slice():[];
    if(!pts.length) pts=(DATA.probes||[]).slice();
    return pts.filter(function(p){return num(p.step)!==null&&num(p.w257)!==null;}).map(function(p){ return {step:p.step,x:p.step,t:num(p.t),w257:num(p.w257),w2048:num(p.w2048),basis:p.step_basis,ts:p.ts}; }).sort(byX);
  }
  function buildModel(lin){
    var h=(lin==="standard"?DATA.standard_history:DATA.history)||null;
    var H=tableRows(h).map(normRow).filter(Boolean).sort(byX);
    var rec=h&&h.recent, R=rec?tableRows(Array.isArray(rec)?{cols:h.cols,rows:rec}:{cols:rec.cols||h.cols,rows:rec.rows}).map(normRow).filter(Boolean).sort(byX):[];
    var T=lin==="rpv16"?tailPts(DATA.train):[];
    var probes=lin==="standard"?recoveredProbes():[];
    var held=((h&&h.heldout)||[]).map(function(p){ var ce=num(p.ce!=null?p.ce:p.value); return (num(p.step)===null||ce===null)?null:{x:p.step,step:p.step,ce:ce,source:p.source,protocol:p.protocol,ckpt:p.ckpt,tokens:num(p.tokens),ab:p.lineage==="abandoned",t:num(p.ts!=null?p.ts:(p.t!=null?p.t:p.mtime))}; }).filter(Boolean).sort(byX);
    var res={}; ((h&&h.resumes)||[]).forEach(function(r){ var s=num(r&&r.step); if(s!==null) res[s]={step:s,t:num(r.ts),path:r.path,max_depth:num(r.max_depth),count:num(r.count)}; });
    if(lin==="rpv16") (DATA.markers||[]).forEach(function(m){ var s=num(m&&m.step); if(s!==null&&m.label==="resume"&&!res[s]) res[s]={step:s,t:null,path:m.checkpoint}; });
    var resumes=Object.keys(res).map(function(k){return res[k];}).sort(function(a,b){return a.step-b.step;});
    var L=DATA.latest||{}, lastS=[], firstS=[];
    if(H.length){ firstS.push(H[0].s0); lastS.push(H[H.length-1].s1); }
    if(R.length){ firstS.push(R[0].s0); lastS.push(R[R.length-1].s1); }
    if(T.length){ firstS.push(T[0].step); lastS.push(T[T.length-1].step); }
    if(probes.length){ firstS.push(probes[0].step); lastS.push(probes[probes.length-1].step); }
    var heldCur=held.filter(function(p){return !p.ab;}); if(heldCur.length){ firstS.push(heldCur[0].step); lastS.push(heldCur[heldCur.length-1].step); }
    if(lin==="rpv16"&&num(L.step)!==null) lastS.push(L.step);
    var first=firstS.length?Math.min.apply(null,firstS):null, last=lastS.length?Math.max.apply(null,lastS):null;
    // step <-> time: bucket ends, dense tail stamps, probe stamps; monotone in step
    var tl=[]; H.concat(R).forEach(function(r){ if(r.t1!==null) tl.push([r.s1,r.t1]); });
    T.forEach(function(p){ var t=num(p.ts); if(t!==null) tl.push([p.step,t]); });
    probes.forEach(function(p){ if(p.t!==null) tl.push([p.step,p.t]); });
    tl.sort(function(a,b){return a[0]-b[0];}); var TL=[]; tl.forEach(function(p){ if(!TL.length||(p[0]>TL[TL.length-1][0]&&p[1]>=TL[TL.length-1][1])) TL.push(p); });
    if(TL.length<2&&last!==null){ // no stamps yet: anchor the latest step to the feed time at the live median step time
      var tEnd=num(L.ts)||num(DATA.polled_at)||num(DATA.generated_at)||Date.now()/1000, sps=num((DATA.gb10_live||{}).median_step_s)||0.82;
      TL=[[first,tEnd-(last-first)*sps],[last,tEnd]];
    }
    var wH=num(h&&h.bucket_steps)||inferW(H)||512, wR=num(rec&&rec.bucket_steps)||inferW(R)||64;
    H.forEach(function(r){ r.w=Math.max(r.s1-r.s0+1,wH); r.src="hist"; });
    R.forEach(function(r){ r.w=Math.max(r.s1-r.s0+1,wR); r.src="recent"; });
    var branches=((h&&h.branches)||[]).map(function(b){ if(!b) return null; var rows=tableRows({cols:b.cols||(h&&h.cols),rows:b.rows}).map(normRow).filter(Boolean).sort(byX), bw=num(b.bucket_steps)||inferW(rows)||wH;
      rows.forEach(function(r){ r.w=Math.max(r.s1-r.s0+1,bw); r.src="ghost"; }); return rows.length?{label:b.label||"abandoned branch",from:num(b.from_step)!==null?num(b.from_step):rows[0].s0,to:num(b.to_step)!==null?num(b.to_step):rows[rows.length-1].s1,ended:num(b.ended_ts),rows:rows,w:bw}:null; }).filter(Boolean);
    branches.forEach(function(b,i){ var src=(h.branches||[]).filter(Boolean)[i]||{}; b.cut=num(src.cut_step)!==null?num(src.cut_step):b.from-1; b.depth=num(src.depth_steps); b.cutTs=num(src.cut_ts)||b.ended; b.meanLast=num(src.ar_mean_last_10k); });
    resumes=resumes.filter(function(r){ return !branches.some(function(b){ return r.step>b.cut&&r.step<=b.to&&r.t!==null&&b.cutTs&&r.t<b.cutTs; }); });
    resumes.forEach(function(r){ r.rollback=branches.some(function(b){return Math.abs(b.cut-r.step)<=1;}); });
    var ref=h&&h.reference&&num(h.reference.unigram_nats)!==null?{v:num(h.reference.unigram_nats),label:h.reference.label||"unigram entropy",meaning:h.reference.meaning||""}:null;
    var bandAR=H.concat(R).some(function(r){return r.ar_p10!==null&&r.ar_p10!==undefined;})||(T.length&&T.some(function(p){return !!p.objective;}));
    var nowT=Date.now()/1000, ghostEnd=null; branches.forEach(function(b){ if(!b.cutTs||nowT-b.cutTs<72*3600) ghostEnd=Math.max(ghostEnd||0,b.to); });
    var m={lin:lin,h:h||{},H:H,R:R,T:T,wH:wH,wR:wR,probes:probes,held:held,resumes:resumes,branches:branches,first:first,last:last,TL:TL,ref:ref,band:bandAR?"ar":"mixed",ghostEnd:ghostEnd};
    m.has={}; ORDER[lin].forEach(function(k){ m.has[k]=seriesAvailable(m,k); });
    return m;
  }
  function seriesAvailable(m,k){ var d=DEFS[k];
    if(k==="held") return m.held.length>0;
    if(k==="ghost") return m.branches.length>0;
    if(k==="ref") return !!m.ref;
    if(k==="w257"||k==="w2048") return m.probes.some(function(p){return p[k]!==null;});
    var any=function(rows){ for(var i=0;i<rows.length;i++){ var v=rows[i][d.f]; if(v!==null&&v!==undefined) return true; } return false; };
    if(any(m.H)||any(m.R)) return true;
    if(!m.T.length) return false;
    if(k==="mix"||k==="lr") return true;
    return m.T.some(function(p){return !!p.objective;});
  }
  function model(lin){ lin=lin||lineage(); if(MODEL&&MODEL.d===DATA&&MODEL.lin===lin) return MODEL; if(HMODEL&&HMODEL.d===DATA&&HMODEL.lin===lin) return HMODEL;
    var m=buildModel(lin); m.d=DATA; if(lin===lineage()) MODEL=m; else HMODEL=m; return m; }
  var HMODEL=null;
  function interp(TL,v,a,b){ if(!TL.length) return null; if(v<=TL[0][a]) return TL[0][b]; var n=TL.length-1; if(v>=TL[n][a]) return TL[n][b];
    var lo=0,hi=n; while(hi-lo>1){ var mid=(lo+hi)>>1; if(TL[mid][a]<=v) lo=mid; else hi=mid; }
    var p=TL[lo],r=TL[hi]; return p[b]+(r[b]-p[b])*((v-p[a])/Math.max(1e-9,r[a]-p[a])); }
  function timeAt(m,s){ return interp(m.TL,s,0,1); }
  function stepAt(m,t){ return interp(m.TL,t,1,0); }

  /* Rows for one series at the resolution the view needs: dense tail, then 64-step recent, then 512-step history. */
  /* Sparse objectives need wider tail buckets: AR is ~3/4 of steps, SAT var and NAT ~1/10 each, anchors 1 in ~17. */
  var MINW={mix:32,ar:8,lr:8,sat:48,nat:48,anc:96};
  function rowsFor(m,k,x0,x1,dw){
    var d=DEFS[k], f=d.f, has=function(rows){ for(var i=0;i<rows.length;i++){ var v=rows[i][f]; if(v!==null&&v!==undefined) return true; } return false; };
    var out=[], cut=Infinity, rHas=m.R.length&&has(m.R), hHas=m.H.length&&has(m.H);
    var tw=Math.max(MINW[k]||8,Math.round(dw));
