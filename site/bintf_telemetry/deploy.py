#!/usr/bin/env python3
"""Deploy the additive monitor and a private, read-only telemetry endpoint."""
from pathlib import Path
import datetime, hashlib, json, os, subprocess

root=Path('/opt/agillm43_netserve')
work=root/'bintf_telemetry'
config=root/'agillm43-infer-edge.nginx.conf'
page=root/'agillm_completion.html'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup=work/('backup-'+stamp)
backup.mkdir()
old_config=config.read_text()
old_page=page.read_text()
(backup/config.name).write_text(old_config)
(backup/page.name).write_text(old_page)
route='''        # Binary 4B read-only telemetry, with the same access policy as the UI.
        location = /agillm-bintf4b-training.json {
            include /opt/agillm43_netserve/agillm_codex_browser_allow.conf;
            include /opt/agillm43_netserve/agillm_acl.conf;
            default_type application/json;
            alias /opt/agillm43_netserve/bintf_telemetry/status.json;
            add_header Cache-Control "no-store, no-cache, must-revalidate" always;
            add_header X-Content-Type-Options "nosniff" always;
        }

'''
if 'location = /agillm-bintf4b-training.json' not in old_config:
    needle='        location = /agillm-ui/ {'
    assert old_config.count(needle)==1, 'UI route is ambiguous'
    config.write_text(old_config.replace(needle,route+needle))
    tested=subprocess.run(['/opt/nginx-goddess/nginx','-t','-c',str(config)],capture_output=True,text=True)
    if tested.returncode:
        config.write_text(old_config)
        raise RuntimeError(tested.stderr)
    subprocess.run(['systemctl','reload','agillm43-infer-edge.service'],check=True)
subprocess.run(['systemctl','daemon-reload'],check=True)
subprocess.run(['systemctl','enable','--now','agillm-bintf-telemetry.timer'],check=True)
subprocess.run(['systemctl','start','agillm-bintf-telemetry.service'],check=True)
assert page.read_text()==old_page, 'Concurrent UI edit detected; page left unchanged'
if 'id="agillm-fleet-ui-20261008"' not in old_page:
    addon=(work/'ui-addon.html').read_text()
    assert old_page.count('</body>')==1
    updated=old_page.replace('</body>',addon+'\n</body>')
    tmp=page.with_suffix('.html.telemetry-tmp')
    tmp.write_text(updated);tmp.chmod(0o644);os.replace(tmp,page)
receipt={'deployed_at':stamp,'backup':str(backup),'page_sha256':hashlib.sha256(page.read_bytes()).hexdigest(),'endpoint':'/agillm-bintf4b-training.json','timer':'agillm-bintf-telemetry.timer'}
(work/'deployment.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
