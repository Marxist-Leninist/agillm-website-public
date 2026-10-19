#!/usr/bin/env python3
"""Narrow checkpoint metadata retention patch. Does not touch any model/service."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, re, shutil, subprocess, tempfile
ROOT=Path('/opt/agillm43_netserve')
WORK=ROOT/'checkpoint_timestamps_20261008'
LIVE=ROOT/'agillm_completion.html'
ADDON=ROOT/'bintf_telemetry/ui-addon.html'
MARKER='AGILLM_CHECKPOINT_RETENTION_V2'
def once(text,old,new):
    if text.count(old)!=1:raise RuntimeError('Source anchor differs: '+old[:100])
    return text.replace(old,new,1)
def sha(text):return hashlib.sha256(text.encode()).hexdigest()
def build(addon):
    if MARKER in addon:raise RuntimeError('Retention v2 already present; inspect before editing.')
    helper=(WORK/'retention-v2.js').read_text()
    addon=once(addon,'<script id="agillm-fleet-ui-runtime-20261008">','<script id="agillm-fleet-ui-runtime-20261008">\n'+helper)
    addon=once(addon,'const number=n=>',"""let checkpointStorage=null;
try { checkpointStorage=window.localStorage; } catch (_) {}
const checkpointStore=window.AGILLMCheckpointRetention.create({parseTime:window.AGILLMCheckpointTime.parse,storage:checkpointStorage});
function checkpointScope(t){
 if(!t||!(t.model_id||t.model||t.lineage||t.instance_id))return '';
 return [t.model_id||t.model||'',t.lineage||'',t.instance_id??''].join('|');
}
function observeCheckpoints(key,value){
 if(key==='fleet')for(const model of ['standard','rpv16']){
  const t=value?.[model+'_live'],event=t?.last_checkpoint_event;
  checkpointStore.observe(model,event?{step:event.step,at:event.ts,identity:event.sha256}:null,checkpointScope(t));
 }
 if(key==='binary'){
  const event=value?.saved?.ckpt;
  checkpointStore.observe('bintf4b',event?{step:event.step,at:event.updated_at,identity:event.sha256}:null,checkpointScope(value));
 }
}
const number=n=>""")
    addon=once(addon,'saved:d.saved?.ckpt?.step,savedAt:d.saved?.ckpt?.updated_at,','...checkpointStore.view(\'bintf4b\'),')
    addon=once(addon,'saved:t.last_checkpoint_event?.step,savedAt:t.last_checkpoint_event?.ts,','...checkpointStore.view(key),')
    start=addon.index('function savedCheckpointMarkup(t){')
    end=addon.index('function servingExportMarkup(h){',start)
    addon=addon[:start]+'''function savedCheckpointMarkup(t){
 const cached=t.savedLastKnown===true;
 const label=cached?'Last known saved':'Latest saved';
 const tip=cached?'Last-known checkpoint metadata retained while the feed is missing or delayed; not confirmation of the newest save.':'Checkpoint update timestamp, not the telemetry refresh.';
 return '<div class="af-save-status" data-saved-step="'+esc(t.saved??'')+'" data-checkpoint-status="'+(cached?'last-known':'current')+'" title="'+esc(tip)+'"><span>'+label+' <strong>'+(number(t.saved)!==null?'#'+fmt(t.saved):'Unavailable')+'</strong></span><div class="af-checkpoint-updated">Last updated: '+checkpointTimeMarkup(t.savedAt)+'</div></div>';
}
'''+addon[end:]
    addon=once(addon,'if(state.busy)return;state.busy=true;','if(state.busy)return;state.busy=true;render();')
    old=" await Promise.all(tasks.map(async([key,url])=>{try{const value=await get(url);if(key==='fleet'||key==='binary')state[key]=value;else state.health[key]=value;delete state.errors[key];}catch(e){state.errors[key]=true;if(key!=='fleet'&&key!=='binary')state.health[key]={ready:false};}}));"
    new=""" await Promise.all(tasks.map(async([key,url])=>{
  try {
   const value=await get(url);
   if(key==='fleet'||key==='binary'){state[key]=value;observeCheckpoints(key,value);}else state.health[key]=value;
   delete state.errors[key];
  } catch(e) {
   state.errors[key]=true;
   if(key==='fleet'){checkpointStore.unavailable('standard');checkpointStore.unavailable('rpv16');}
   else if(key==='binary')checkpointStore.unavailable('bintf4b');
   else state.health[key]={ready:false};
  } finally { render(); }
 }));"""
    addon=once(addon,old,new)
    addon=once(addon,'Delayed cards retain their last known values.','Missing checkpoint metadata keeps a labelled last-known step and timestamp; it never borrows a timestamp from another checkpoint.')
    return addon

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
    test=subprocess.run(['node',str(WORK/'retention-v2.test.js')],capture_output=True,text=True)
    (WORK/'retention-v2-tests.txt').write_text(test.stdout+test.stderr)
    if test.returncode:raise RuntimeError(test.stdout+test.stderr)
    addon=ADDON.read_text();page=LIVE.read_text()
    updated=build(addon);candidate=once(page,addon,updated)
    scripts=re.findall(r'<script\b[^>]*>(.*?)</script\s*>',candidate,re.S|re.I)
    checked=0
    with tempfile.TemporaryDirectory(prefix='agillm-retention-v2-') as tmp:
        for index,script in enumerate(scripts):
            if not script.strip():continue
            path=Path(tmp)/f'{index}.js';path.write_text(script)
            result=subprocess.run(['node','--check',str(path)],capture_output=True,text=True)
            if result.returncode:raise RuntimeError(result.stderr)
            checked+=1
    (WORK/'retention-v2-ui.candidate.html').write_text(candidate)
    (WORK/'retention-v2-addon.candidate.html').write_text(updated)
    receipt={'marker':MARKER,'applied':False,'unit_tests':20,'inline_scripts_checked':checked,'before_ui_sha256':sha(page),'after_ui_sha256':sha(candidate),'addon_sha256':sha(updated),'changes':['Keep last-known checkpoint step/time pairs with explicit label','Accept lower-step rollback metadata','Never substitute poll times or cross-step timestamps','Render each telemetry feed independently as it arrives'],'model_or_service_changes':[]}
    if args.apply:
        if LIVE.read_text()!=page or ADDON.read_text()!=addon:raise RuntimeError('Concurrent UI edit detected; re-read before applying.')
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        backup=WORK/('backup-retention-v2-'+stamp);backup.mkdir()
        shutil.copy2(LIVE,backup/LIVE.name);shutil.copy2(ADDON,backup/ADDON.name)
        for destination,text in [(LIVE,candidate),(ADDON,updated)]:
            temp=destination.with_name(destination.name+'.retention-v2-new');temp.write_text(text)
            os.chmod(temp,destination.stat().st_mode&0o777);os.replace(temp,destination)
        receipt.update(applied=True,applied_at=stamp,backup=str(backup))
    (WORK/'retention-v2-receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
