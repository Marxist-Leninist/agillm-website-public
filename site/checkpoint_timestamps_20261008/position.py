from pathlib import Path
import hashlib,json,os,shutil
root=Path('/opt/agillm43_netserve');work=root/'checkpoint_timestamps_20261008'
p=root/'bintf_telemetry/ui-addon.html';live=root/'agillm_completion.html'
a=p.read_text();page=live.read_text()
pairs=[("+'</strong></div>'+servingExportMarkup(h)+savedCheckpointMarkup(t)+'<details class=\"af-details\"'","+'</strong></div>'+servingExportMarkup(h)+'<details class=\"af-details\"'"),("metric('Train step',compact(t.step))+'</div><div class=\"af-checkpoint\">","metric('Train step',compact(t.step))+'</div>'+savedCheckpointMarkup(t)+'<div class=\"af-checkpoint\">")]
b=a
for old,new in pairs:
 assert b.count(old)==1,'Unexpected source; not applying'
 b=b.replace(old,new,1)
assert page.count(a)==1,'Concurrent addon change; not applying'
newpage=page.replace(a,b,1)
backup=work/'before-position';backup.mkdir(exist_ok=False)
shutil.copy2(p,backup/p.name);shutil.copy2(live,backup/live.name)
assert p.read_text()==a and live.read_text()==page,'Concurrent source change'
for dest,text in [(live,newpage),(p,b)]:
 tmp=dest.with_name(dest.name+'.timestamp-position');tmp.write_text(text);os.chmod(tmp,dest.stat().st_mode&0o777);os.replace(tmp,dest)
r={'ui_sha256':hashlib.sha256(newpage.encode()).hexdigest(),'addon_sha256':hashlib.sha256(b.encode()).hexdigest(),'change':'Latest saved timestamp placed immediately after trainer metrics, before serving row, to fit compact telemetry rail','backup':str(backup)}
(work/'position-receipt.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
