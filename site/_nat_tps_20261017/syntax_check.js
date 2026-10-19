// Parse every inline <script> in the HTML with V8 to catch syntax errors (same approach as /opt/bintf4b_serve/check_ui.js).
const fs = require('fs'), vm = require('vm');
function scripts(html) {
  const out = []; const re = /<script\b([^>]*)>([\s\S]*?)<\/script>/gi; let m;
  while ((m = re.exec(html))) {
    const attrs = m[1];
    if (/\bsrc=/i.test(attrs)) continue;
    const t = (/type\s*=\s*["']([^"']+)["']/i.exec(attrs) || [])[1] || '';
    if (t && !/javascript|module/i.test(t)) continue;
    out.push({ attrs, code: m[2] });
  }
  return out;
}
for (const f of process.argv.slice(2)) {
  const s = scripts(fs.readFileSync(f, 'utf8'));
  let bad = 0;
  s.forEach((x, i) => {
    try { new vm.Script(x.code, { filename: f + '#' + i }); }
    catch (e) { bad++; console.log('SYNTAX', f, i, x.attrs.slice(0, 60), e.message); }
  });
  console.log(f, 'scripts', s.length, 'syntax errors', bad);
  if (bad) process.exitCode = 1;
}
