import sys
p=sys.argv[1]; s=open(p,encoding='utf-8').read()
def rep(old,new,n=1):
    global s
    c=s.count(old); assert c==n,(old[:60],c); s=s.replace(old,new)
rep('#agi44-curves-body{display:none;margin-top:8px;','#agi44-curves-body{display:none;box-sizing:border-box;margin-top:8px;')
rep('#agi44-curves canvas{display:block;width:100%;height:290px;','#agi44-curves canvas{display:block;box-sizing:border-box;width:100%;height:290px;')
rep('#agi44-curves .foot{margin-top:8px;','#agi44-curves *{box-sizing:border-box}\n#agi44-curves .foot{margin-top:8px;')
rep('''    g.fillStyle="#8b98ab"; g.textAlign="left"; g.fillText("step",pad.l,pad.t+ph+22>H-4?H-6:H-6);
''','')
rep('''    if(show.train) T.forEach(function(p){ys.push(p.ema);});
''','''    if(show.train&&T.length){ var te=T.map(function(p){return p.ema;}).sort(function(a,b){return a-b;}); ys.push(te[Math.floor(te.length*0.05)], te[Math.min(te.length-1,Math.floor(te.length*0.95))]); }
''')
rep('''    (DATA.markers||[]).forEach(function(m){ if(m.label==="resume"||!inR(m,x0,x1)) return; var xx=X(m.step); g.strokeStyle="#2f3b52"; g.setLineDash([3,3]); g.beginPath(); g.moveTo(xx,pad.t); g.lineTo(xx,pad.t+ph); g.stroke(); g.setLineDash([]); g.fillStyle="#6b7686"; g.textAlign="left"; g.save(); g.translate(xx+3,pad.t+2); g.fillText(hopLabel(m.label),0,4); g.restore(); });
''','''    var hops={}; (DATA.markers||[]).forEach(function(m){ if(m.label!=="resume"&&inR(m,x0,x1)) hops[m.step]=m; });
    var hs=Object.keys(hops).map(Number).sort(function(a,b){return a-b;}), hx=hs.map(X);
    hs.forEach(function(st,i){ var xx=hx[i]; g.strokeStyle="#2f3b52"; g.setLineDash([3,3]); g.beginPath(); g.moveTo(xx,pad.t); g.lineTo(xx,pad.t+ph); g.stroke(); g.setLineDash([]); });
    // labels: one per cluster of nearby marks (the last of the cluster), staggered over 3 rows
    var lastLx=-1e9,row=0; hs.forEach(function(st,i){ var xx=hx[i]; if(i+1<hs.length&&hx[i+1]-xx<72) return; row=(xx-lastLx<72)?(row+1)%3:0; lastLx=xx; g.fillStyle="#6b7686"; g.textAlign="left"; var lab=hopLabel(hops[st].label); if(xx+g.measureText(lab).width>pad.l+pw){ g.textAlign="right"; g.fillText(lab,xx-3,pad.t+6+row*12); } else g.fillText(lab,xx+3,pad.t+6+row*12); });
''')
rep('''g.fillStyle=COL.w257; g.textAlign="right"; g.fillText("best "+F(best.w257,4)+" @"+N(best.step),pad.l+pw-4,Y(best.w257)-8); }''',
    '''g.fillStyle=COL.w257; g.textAlign="left"; g.fillText("best "+F(best.w257,4)+" @"+N(best.step),pad.l+4,Y(best.w257)-8); }''')
rep('''    if(show.lr){ g.textAlign="left"; g.fillStyle=COL.lr; for(var e=Math.ceil(lmin);e<=lmax;e++){ var yl=YL(Math.pow(10,e)); g.fillText("1e"+e,pad.l+pw+5,yl); } }
''','''    if(show.lr){ g.textAlign="left"; g.fillStyle=COL.lr; var mult=(lmax-lmin)<1.6?[1,2,5]:[1]; for(var e=Math.floor(lmin)-1;e<=Math.ceil(lmax);e++){ mult.forEach(function(k){ var v=k*Math.pow(10,e), lv=Math.log10(v); if(lv<lmin||lv>lmax) return; g.fillText((k===1?"1e"+e:k+"e"+e),pad.l+pw+5,YL(v)); }); } }
''')
rep('''    if(show.lr&&T.length) line(T,"step","lr",COL.lr,1,[4,3],0,YL);
''','''    g.save(); g.beginPath(); g.rect(pad.l,pad.t,pw,ph); g.clip();
    if(show.lr&&T.length){ line(T,"step","lr",COL.lr,1,[4,3],0,YL); var lt=T[T.length-1]; g.fillStyle=COL.lr; g.textAlign="right"; g.fillText("lr "+E(lt.lr),X(lt.step)-4,YL(lt.lr)-8); }
''')
rep('''    // best held-out marker
''','''    g.restore();
    // best held-out marker
''')
open(p,'w',encoding='utf-8').write(s); print('patched',p,len(s))
