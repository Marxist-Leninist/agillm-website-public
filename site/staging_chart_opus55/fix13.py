import sys
from pathlib import Path
p=Path(sys.argv[1]); s=p.read_text(encoding="utf-8")
def rep(a,b):
    global s
    n=s.count(a); assert n==1,(n,a[:90]); s=s.replace(a,b)
# held-out: anything not explicitly on the current lineage is hollow and never sets the live range
rep('''tokens:num(p.tokens),ab:p.lineage==="abandoned",''','''tokens:num(p.tokens),ab:p.lineage!=null&&p.lineage!=="current",''')
# resumes: keep only those that sit on the current lineage's timeline; infer the rollback cut when branches[] is empty
rep('''    resumes=resumes.filter(function(r){ return !branches.some(function(b){ return r.step>b.cut&&r.step<=b.to&&r.t!==null&&b.cutTs&&r.t<b.cutTs; }); });
    resumes.forEach(function(r){ r.rollback=branches.some(function(b){return Math.abs(b.cut-r.step)<=1;}); });''',
'''    resumes=resumes.filter(function(r){ if(last!==null&&r.step>last+1) return false; if(branches.some(function(b){ return r.step>b.cut&&r.step<=b.to&&r.t!==null&&b.cutTs&&r.t<b.cutTs; })) return false;
      if(r.t!==null&&TL.length>=2&&r.step>=TL[0][0]&&r.step<=TL[TL.length-1][0]){ var tAt=interp(TL,r.step,0,1); if(tAt!==null&&r.t<tAt-900) return false; } return true; });
    var rbMin=num(h&&h.rollback_min_steps)||4096, cuts=branches.map(function(b){ return {cut:b.cut,depth:b.depth,t:b.cutTs}; });
    if(!cuts.length&&num(h&&h.summary&&h.summary.rows_abandoned)>0){ var cand=null; resumes.forEach(function(r){ if(num(r.max_depth)===null||r.max_depth<rbMin||r.t===null) return; var prevT=null; for(var i=TL.length-1;i>=0;i--){ if(TL[i][0]<=r.step){ prevT=TL[i][1]; break; } }
      if(prevT!==null&&r.t-prevT>7200&&(!cand||r.max_depth>cand.max_depth)) cand=r; }); if(cand) cuts.push({cut:cand.step,depth:cand.max_depth,t:cand.t}); }
    resumes.forEach(function(r){ r.rollback=cuts.some(function(c){return Math.abs(c.cut-r.step)<=1;}); });''')
rep('''resumes:resumes,branches:branches,first:first,''','''resumes:resumes,branches:branches,cuts:cuts,first:first,''')
# drawing uses the cuts (branch cuts, or the inferred one)
rep('''    RS.forEach(function(r){ if(m.branches.some(function(b){return Math.abs(b.cut-r.step)<=1;})) return;''','''    RS.forEach(function(r){ if(m.cuts.some(function(b){return Math.abs(b.cut-r.step)<=1;})) return;''')
rep('''    var RL=RS.filter(function(r){ return !m.branches.some(function(b){return Math.abs(b.cut-r.step)<=1;}); });''','''    var RL=RS.filter(function(r){ return !m.cuts.some(function(b){return Math.abs(b.cut-r.step)<=1;}); });''')
rep('''    m.branches.forEach(function(b){ if(b.cut<x0||b.cut>x1) return; var bx=Math.round(X(b.cut))+.5;''','''    m.cuts.forEach(function(b){ if(b.cut<x0||b.cut>x1) return; var bx=Math.round(X(b.cut))+.5;''')
p.write_text(s, encoding="utf-8"); print("ok")
