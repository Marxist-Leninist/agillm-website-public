#!/usr/bin/env python3
"""Add checkpoint timestamps to the current fleet addon; no service restarts."""
from pathlib import Path
import argparse, datetime, hashlib, json, os, re, shutil, subprocess, tempfile
BASE=Path('/opt/agillm43_netserve')
ADDON=BASE/'bintf_telemetry/ui-addon.html'
LIVE=BASE/'agillm_completion.html'
WORK=BASE/'checkpoint_timestamps_20261008'
MARKER='AGILLM_CHECKPOINT_TIMESTAMPS_V1'
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def once(s,a,b):
 n=s.count(a)
 if n!=1:raise RuntimeError(f'Expected one anchor, found {n}: {a[:110]}')
 return s.replace(a,b,1)
def build(old):
 if MARKER in old:raise RuntimeError('Timestamp patch already present; inspect before changing.')
 helper=(WORK/'checkpoint-time.js').read_text()
 new=once(old,'<script id="agillm-fleet-ui-runtime-20261008">','<script id="agillm-fleet-ui-runtime-20261008">\n/* '+MARKER+' */\n'+helper)
 new=once(new,'</style>','''.af-save-status{margin-top:7px;padding-top:7px;border-top:1px solid #2b394b;font-size:11px;line-height:1.55;color:#a3b4c9;overflow-wrap:anywhere}
.af-save-status strong{color:#d7e5f5;font-weight:650}
.af-checkpoint-updated{margin-top:2px;color:#aebed1;font-variant-numeric:tabular-nums}
.af-checkpoint-updated time{color:#d8e3f2}
.af-checkpoint-age{white-space:nowrap;color:#91a6bd}
.af-serving-export{font-size:10px;line-height:1.5;color:#95a9bf;margin-top:4px;overflow-wrap:anywhere}
</style>''')
 new=once(new,'saved:d.saved?.ckpt?.step,validated:','saved:d.saved?.ckpt?.step,savedAt:d.saved?.ckpt?.updated_at,validated:')
 new=once(new,'saved:t.last_checkpoint_event?.step,rateLabel:','saved:t.last_checkpoint_event?.step,savedAt:t.last_checkpoint_event?.ts,rateLabel:')
 functions='''function checkpointTimeMarkup(value){
 const f=window.AGILLMCheckpointTime.format(value);
 if(!f)return '<span class="af-time-unavailable">Time unavailable</span>';
 return '<time datetime="'+esc(f.iso)+'" title="'+esc(f.iso)+'">'+esc(f.absolute)+'</time> <span class="af-checkpoint-age">('+esc(f.relative)+')</span>';
}
function savedCheckpointMarkup(t){
 return '<div class="af-save-status" data-saved-step="'+esc(t.saved??'')+'"><span>Latest saved <strong>'+(number(t.saved)!==null?'#'+fmt(t.saved):'Unavailable')+'</strong></span><div class="af-checkpoint-updated" title="Timestamp of the saved checkpoint, not the telemetry refresh.">Last updated: '+checkpointTimeMarkup(t.savedAt)+'</div></div>';
}
function servingExportMarkup(h){
 const value=h?.model?.exported_at??h?.ckpt_meta?.exported_at;
 return h?.ready===true&&window.AGILLMCheckpointTime.parse(value)!==null?'<div class="af-serving-export">Serving exported: '+checkpointTimeMarkup(value)+'</div>':'';
}
'''
 new=once(new,'function card(key,open){',functions+'function card(key,open){')
 new=once(new,"+'</strong></div><details class=\"af-details\"'","+'</strong></div>'+servingExportMarkup(h)+savedCheckpointMarkup(t)+'<details class=\"af-details\"'")
 return new

def validate(html,directory):
 scripts=re.findall(r'<script\b[^>]*>(.*?)</script\s*>',html,re.S|re.I)
 count=0
 for i,text in enumerate(scripts):
  if not text.strip():continue
  p=directory/f'check-{i}.js';p.write_text(text)
  result=subprocess.run(['node','--check',str(p)],capture_output=True,text=True)
  if result.returncode:raise RuntimeError(result.stderr)
  count+=1
 return count

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
 oldaddon=ADDON.read_text();oldlive=LIVE.read_text()
 newaddon=build(oldaddon)
 newlive=once(oldlive,oldaddon,newaddon)
 (WORK/'agillm_completion.candidate.html').write_text(newlive)
 (WORK/'ui-addon.candidate.html').write_text(newaddon)
 with tempfile.TemporaryDirectory(prefix='agillm-timestamp-jscheck-') as td:
  checked=validate(newlive,Path(td))
 receipt={'schema':'agillm.checkpoint_timestamps.v1','marker':MARKER,'before_ui_sha256':sha(oldlive),'after_ui_sha256':sha(newlive),'before_addon_sha256':sha(oldaddon),'after_addon_sha256':sha(newaddon),'scripts_syntax_checked':checked,'applied':False,'saved_timestamp_sources':{'standard':'standard_live.last_checkpoint_event.ts','rpv16':'rpv16_live.last_checkpoint_event.ts','bintf4b':'saved.ckpt.updated_at'},'serving_timestamp_source':'health.model.exported_at when explicitly available','timezone':'Europe/London','service_restarts':[]}
 if args.apply:
  if sha(LIVE.read_text())!=sha(oldlive) or sha(ADDON.read_text())!=sha(oldaddon):raise RuntimeError('Concurrent source change detected; re-read before applying.')
  stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
  backup=WORK/('backup-'+stamp);backup.mkdir()
  shutil.copy2(LIVE,backup/LIVE.name);shutil.copy2(ADDON,backup/ADDON.name)
  for path,text in [(LIVE,newlive),(ADDON,newaddon)]:
   tmp=path.with_name(path.name+'.timestamps-new');tmp.write_text(text);os.chmod(tmp,path.stat().st_mode&0o777);os.replace(tmp,path)
  receipt.update(applied=True,applied_at=stamp,backup=str(backup))
 (WORK/'receipt.json').write_text(json.dumps(receipt,indent=2))
 print(json.dumps(receipt))
if __name__=='__main__':main()
