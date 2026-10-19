import sys
from pathlib import Path
b = Path(sys.argv[1]).read_text(encoding="utf-8")
def rep(a, c):
    global b
    assert b.count(a) == 1, a[:60]; b = b.replace(a, c)
rep("panel = Path('/opt/agillm43_netserve/agillm_curves_panel.html').read_text(encoding='utf-8').strip('\\n')",
    "import sys\nPANEL = sys.argv[1] if len(sys.argv) > 1 else '/opt/agillm43_netserve/agillm_curves_panel.html'\nOUT = sys.argv[2] if len(sys.argv) > 2 else '/opt/agillm43_netserve/agillm_curves_page.html'\npanel = Path(PANEL).read_text(encoding='utf-8').strip('\\n')")
rep("<title>AGILLM-4.4 — training curves</title>", "<title>AGILLM — training curves</title>")
rep("#agi44-curves-body{display:block!important;", "#agi44-curves-body{display:flex!important;flex-direction:column;")
rep("#agi44-curves h4 a[href=\"/agillm-curves\"]{display:none}", "#agi44-curves h4 a[href=\"/agillm-curves\"]{display:none}\n#agi44-curves .telemetry-close{display:none}")
rep("<h1>AGILLM-4.4 training curves</h1><div class=\"sub\">AR, SAT fixed, SAT var, NAT · separate held-out and training curves · LR · refreshes every minute</div>",
    "<h1>AGILLM training curves</h1><div class=\"sub\">RPV16 2.0B on GB10 and Standard 1.1B, one model per axis · AR full-stack, AR block-local, SAT var, NAT · rollbacks and abandoned branches · drag to zoom · refreshes every minute</div>")
rep("Path('/opt/agillm43_netserve/agillm_curves_page.html').write_text(page, encoding='utf-8')\nprint('agillm_curves_page.html', len(page))",
    "Path(OUT).write_text(page, encoding='utf-8')\nprint(OUT, len(page))")
Path(sys.argv[2]).write_text(b, encoding="utf-8"); print("ok")
