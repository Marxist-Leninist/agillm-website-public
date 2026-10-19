from pathlib import Path
import hashlib,json,os,shutil,tempfile,subprocess
root=Path('/opt/agillm43_netserve');work=root/'checkpoint_timestamps_20261008'
addon=root/'bintf_telemetry/ui-addon.html';live=root/'agillm_completion.html'
a=addon.read_text();page=live.read_text()
old="  checkpointStore.observe('bintf4b',event?{step:event.step,at:event.updated_at,identity:event.sha256}:null,checkpointScope(value));"
new=old+"\n  if(value?.ok!==true||value?.stale===true)checkpointStore.unavailable('bintf4b');"
assert a.count(old)==1 and new not in a,'Source differs or already patched'
b=a.replace(old,new,1);assert page.count(a)==1,'Live addon differs'
updated=page.replace(a,b,1)
script=b.split('<script id="agillm-fleet-ui-runtime-20261008">',1)[1].rsplit('</script>',1)[0]
with tempfile.TemporaryDirectory(prefix='agillm-stale-label-') as directory:
 p=Path(directory)/'check.js';p.write_text(script);subprocess.run(['node','--check',str(p)],check=True)
backup=work/'before-stale-binary-label-v2';backup.mkdir(exist_ok=False)
shutil.copy2(addon,backup/addon.name);shutil.copy2(live,backup/live.name)
assert addon.read_text()==a and live.read_text()==page,'Concurrent edit'
for dest,text in [(live,updated),(addon,b)]:
 tmp=dest.with_name(dest.name+'.stale-label-v2');tmp.write_text(text);os.chmod(tmp,dest.stat().st_mode&0o777);os.replace(tmp,dest)
receipt={'change':'Explicit stale/error Binary collector payloads label checkpoint metadata last-known','ui_sha256':hashlib.sha256(updated.encode()).hexdigest(),'addon_sha256':hashlib.sha256(b.encode()).hexdigest(),'backup':str(backup),'services_restarted':[]}
(work/'stale-binary-label-v2-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
