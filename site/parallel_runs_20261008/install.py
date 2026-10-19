"""Stage or atomically install the independent parallel-generation workspace."""
import datetime,hashlib,json,os,pathlib,re,stat,subprocess,sys,tempfile
ROOT=pathlib.Path('/opt/agillm43_netserve/parallel_runs_20261008')
p=ROOT.parent/'agillm_completion.html'
raw=p.read_bytes(); text=raw.decode(); before=hashlib.sha256(raw).hexdigest()
start='<!-- AGILLM_PARALLEL_RUNS_V1:'; end='<!-- /AGILLM_PARALLEL_RUNS_V1 -->'
if start in text:
 a=text.index(start); b=text.index(end,a)+len(end); text=text[:a]+text[b:]
if 'const AGI_PARALLEL_SUFFIX =' not in text:
 helper="const AGI_PARALLEL_SUFFIX = (() => { const id = new URLSearchParams(location.search).get('agi_parallel_run') || ''; return /^[a-zA-Z0-9_-]{1,96}$/.test(id) ? '::parallel:' + id : ''; })();\n"
 assert text.count('const LEGACY_KEY =')==1
 text=text.replace('const LEGACY_KEY =',helper+'const LEGACY_KEY =',1)
 for key in ['LEGACY_KEY','CHAT_KEY','ACTIVE_KEY','LIVE_STREAM_MIGRATION_KEY','RUN_HISTORY_KEY','RUN_HISTORY_MIGRATION_KEY','HISTORY_VIEW_KEY','UNDO_KEY']:
  text,n=re.subn(r'(const '+key+r' = "[^"\n]*")\s*;',r'\1 + AGI_PARALLEL_SUFFIX;',text)
  assert n==1,(key,n)
addon=(ROOT/'parallel-addon.html').read_text()
assert text.count('</body>')==1
text=text.replace('</body>',addon+'\n</body>',1)
candidate=ROOT/'candidate.html';candidate.write_text(text)
for i,(attrs,js) in enumerate(re.findall(r'<script\b([^>]*)>(.*?)</script\s*>',text,re.S|re.I)):
 if not js.strip() or re.search(r'\bsrc\s*=',attrs) or re.search(r'application/(?:ld\+)?json',attrs):continue
 with tempfile.NamedTemporaryFile('w',suffix='.js',delete=False) as f:f.write(js);name=f.name
 try:
  checked=subprocess.run(['node','--check',name],capture_output=True,text=True,timeout=20)
  if checked.returncode:raise RuntimeError('Script '+str(i)+': '+checked.stderr)
 finally:os.unlink(name)
result={'before_sha256':before,'candidate_sha256':hashlib.sha256(text.encode()).hexdigest(),'candidate':str(candidate),'javascript_syntax':'passed','applied':False}
if '--apply' in sys.argv:
 if hashlib.sha256(p.read_bytes()).hexdigest()!=before:raise RuntimeError('Live source changed during checks; rerun against the new source.')
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 backup=ROOT/('before-'+stamp+'.html');backup.write_bytes(raw)
 fd,name=tempfile.mkstemp(prefix='.parallel-',suffix='.html',dir=p.parent)
 try:
  with os.fdopen(fd,'w') as f:f.write(text);f.flush();os.fsync(f.fileno())
  os.chmod(name,stat.S_IMODE(p.stat().st_mode));os.replace(name,p)
 finally:
  if os.path.exists(name):os.unlink(name)
 result.update(applied=True,backup=str(backup),installed_at=stamp,live_path=str(p))
 (ROOT/'deployment.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
