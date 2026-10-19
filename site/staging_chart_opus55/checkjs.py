import re, sys, subprocess, tempfile, os
html = open(sys.argv[1], encoding="utf-8").read()
bad = 0; n = 0
for m in re.finditer(r'<script([^>]*)>(.*?)</script>', html, re.S):
    attrs, body = m.group(1), m.group(2)
    if 'src=' in attrs or ('type=' in attrs and 'javascript' not in attrs and 'module' not in attrs): continue
    n += 1
    with tempfile.NamedTemporaryFile('w', suffix='.mjs' if 'module' in attrs else '.js', delete=False, encoding='utf-8') as f: f.write(body); p = f.name
    r = subprocess.run(['node', '--check', p], capture_output=True, text=True); os.unlink(p)
    if r.returncode: bad += 1; print('SYNTAX FAIL', attrs.strip()[:60], r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else '')
print(sys.argv[1], 'scripts', n, 'failures', bad)
