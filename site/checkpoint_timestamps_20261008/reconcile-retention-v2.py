"""Preserve live peer scope fix, synchronize its source, and test merged timestamp code."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,re,shutil,subprocess,tempfile
ROOT=Path('/opt/agillm43_netserve');WORK=ROOT/'checkpoint_timestamps_20261008'
LIVE=ROOT/'agillm_completion.html';ADDON=ROOT/'bintf_telemetry/ui-addon.html'
HELPER=WORK/'retention-v2.js';TEST=WORK/'retention-v2.test.js'
pattern=r'(<script id="agillm-fleet-ui-runtime-20261008">)(.*?)(</script>)'
def once(text,old,new):
 assert text.count(old)==1,'Expected one source anchor: '+old[:80]
 return text.replace(old,new,1)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');args=ap.parse_args()
 oldpage=LIVE.read_text();oldaddon=ADDON.read_text();oldhelper=HELPER.read_text();oldtest=TEST.read_text()
 lm=re.search(pattern,oldpage,re.S);am=re.search(pattern,oldaddon,re.S)
 assert lm and am,'Missing runtime block'
 script=lm.group(2)
 assert 'const scopeComplete = scopeParts.length === 3' in script,'Expected peer scope fix not found'
 line="  checkpointStore.observe('bintf4b',event?{step:event.step,at:event.updated_at,identity:event.sha256}:null,checkpointScope(value));"
 script=once(script,line,line+"\n  if(value?.ok!==true||value?.stale===true)checkpointStore.unavailable('bintf4b');")
 newpage=oldpage[:lm.start(2)]+script+oldpage[lm.end(2):]
 newaddon=oldaddon[:am.start(2)]+script+oldaddon[am.end(2):]
 helper=script.split('/* AGILLM_CHECKPOINT_TIMESTAMPS_V1 */',1)[0].strip()+'\n'
 assert helper.startswith('/* AGILLM_CHECKPOINT_RETENTION_V2')
 oldline="test('new lineage without a checkpoint clears old lineage cache',()=>{const c=create();c.observe('rpv16',full(100),'lineage1');c.observe('rpv16',null,'lineage2');assert.equal(c.view('rpv16').saved,null);});"
 newline="test('confirmed new lineage does not inherit previous timestamp',()=>{const c=create();c.observe('rpv16',full(100),'rpv16|lineage1|1');c.observe('rpv16',{step:100},'rpv16|lineage2|1');assert.equal(c.view('rpv16').savedAt,null);});"
 tests=once(oldtest,oldline,newline)
 extra="""test('partial-scope missing event cannot erase last-known checkpoint',()=>{const c=create();c.observe('rpv16',full(100),'rpv16|lineage1|53075549');c.observe('rpv16',null,'rpv16||');assert.equal(c.view('rpv16').saved,100);assert.equal(c.view('rpv16').savedAt,saved*1000);assert.equal(c.view('rpv16').savedLastKnown,true);});
test('unconfirmed scope transition preserves clearly-labelled history',()=>{const c=create();c.observe('rpv16',full(100),'rpv16|lineage1|1');c.observe('rpv16',null,'rpv16|lineage2|2');assert.equal(c.view('rpv16').saved,100);assert.equal(c.view('rpv16').savedLastKnown,true);});
"""
 tests=once(tests,"console.log(JSON.stringify({passed:count,failed:0}));",extra+"console.log(JSON.stringify({passed:count,failed:0}));")
 with tempfile.TemporaryDirectory(prefix='agillm-retention-merged-') as td:
  temp=Path(td);(temp/'retention-v2.js').write_text(helper);(temp/'retention-v2.test.js').write_text(tests);shutil.copy2(WORK/'checkpoint-time.js',temp/'checkpoint-time.js')
  test=subprocess.run(['node',str(temp/'retention-v2.test.js')],capture_output=True,text=True)
  if test.returncode:raise RuntimeError(test.stdout+test.stderr)
  result=json.loads(test.stdout.splitlines()[-1]);checked=0
  for index,body in enumerate(re.findall(r'<script\b[^>]*>(.*?)</script\s*>',newpage,re.S|re.I)):
   if not body.strip():continue
   js=temp/f'inline-{index}.js';js.write_text(body)
   subprocess.run(['node','--check',str(js)],check=True,capture_output=True);checked+=1
 (WORK/'retention-v2-addon.candidate.html').write_text(newaddon)
 (WORK/'retention-v2-ui.candidate.html').write_text(newpage)
 report={'applied':False,'unit_tests':result,'inline_scripts_checked':checked,'peer_scope_fix_preserved':True,'ui_sha256':hashlib.sha256(newpage.encode()).hexdigest(),'addon_sha256':hashlib.sha256(newaddon.encode()).hexdigest(),'service_or_model_changes':[]}
 if args.apply:
  pairs=[(LIVE,oldpage,newpage),(ADDON,oldaddon,newaddon),(HELPER,oldhelper,helper),(TEST,oldtest,tests)]
  assert all(path.read_text()==before for path,before,after in pairs),'Concurrent source change; no writes applied'
  stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
  backup=WORK/('backup-reconciled-'+stamp);backup.mkdir()
  for path,before,after in pairs:shutil.copy2(path,backup/path.name)
  for path,before,after in pairs:
   tmp=path.with_name(path.name+'.retention-merged');tmp.write_text(after);os.chmod(tmp,path.stat().st_mode&0o777);os.replace(tmp,path)
  report.update(applied=True,applied_at=stamp,backup=str(backup))
 (WORK/'reconciled-v2-receipt.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
