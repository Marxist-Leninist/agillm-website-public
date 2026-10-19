from pathlib import Path
import os,subprocess,time
root=Path('/opt/agillm43_netserve')
work=root/'bintf_telemetry'
subprocess.run(['python3',str(work/'compact_fleet.py')],check=True)
conf=root/'agillm43-infer-edge.nginx.conf'
old=conf.read_text()
route='''        location = /agillm-fleet-live.json {
            include /opt/agillm43_netserve/agillm_codex_browser_allow.conf;
            include /opt/agillm43_netserve/agillm_acl.conf;
            default_type application/json;
            alias /opt/agillm43_netserve/bintf_telemetry/fleet-compact.json;
            add_header Cache-Control "no-store, no-cache, must-revalidate" always;
            add_header X-Content-Type-Options "nosniff" always;
        }

'''
if 'location = /agillm-fleet-live.json' not in old:
    needle='        location = /agillm-bintf4b-training.json'
    assert old.count(needle)==1
    (work/'nginx.before-compact.conf').write_text(old)
    conf.write_text(old.replace(needle,route+needle))
    tested=subprocess.run(['/opt/nginx-goddess/nginx','-t','-c',str(conf)],capture_output=True,text=True)
    if tested.returncode:
        conf.write_text(old);raise RuntimeError(tested.stderr)
    subprocess.run(['systemctl','reload','agillm43-infer-edge.service'],check=True)
for name in ['agillm_completion.html','bintf_telemetry/ui-addon.html']:
    p=root/name;s=p.read_text()
    needle="['fleet','/api/training?history=0']"
    assert s.count(needle)==1
    backup=work/(p.name+'.before-compact-'+str(int(time.time())))
    backup.write_text(s)
    tmp=p.with_suffix(p.suffix+'.compact-tmp');tmp.write_text(s.replace(needle,"['fleet','/agillm-fleet-live.json']"));tmp.chmod(0o644);os.replace(tmp,p)
drop=Path('/etc/systemd/system/agillm-bintf-telemetry.service.d');drop.mkdir(exist_ok=True)
(drop/'compact-fleet.conf').write_text('[Service]\nExecStartPost=/usr/bin/python3 /opt/agillm43_netserve/bintf_telemetry/compact_fleet.py\n')
subprocess.run(['systemctl','daemon-reload'],check=True)
print('Compact live monitor deployed; existing detailed chart endpoints unchanged.')
