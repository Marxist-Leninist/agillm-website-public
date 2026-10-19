#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path


def one(s, old, new, label):
    n=s.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected 1 match, got {n}")
    return s.replace(old,new,1)


def patch(s):
    s=one(s,
'''    upstream agillm44_popup_backend {
        server 127.0.0.1:18597 max_fails=1 fail_timeout=3s;
        keepalive 4;
    }
''',
'''    upstream agillm44_popup_backend {
        server 127.0.0.1:18597 max_fails=1 fail_timeout=3s;
        keepalive 4;
    }

    # AGILLM-GB10-1PF RPV16 completed checkpoint, derived INT8 serving cache on GETH CPU.
    # Strict identity: no fallback to Standard or Popout.
    upstream agillm_rpv16_cpu_backend {
        server 127.0.0.1:18620 max_fails=1 fail_timeout=3s;
        keepalive 2;
    }
''','upstream')
    marker='''

        # Detachable 49.1M popup routes use a dedicated one-worker AR service.
'''
    routes='''

        # RPV16 1PF / 2.0B AR-only GETH CPU route. The source checkpoint is kept
        # verbatim; this backend loads a derived dynamic-INT8 serving cache.
        location = /agillm-rpv16-health {
            include /opt/agillm43_netserve/agillm_codex_browser_allow.conf;
            include /opt/agillm43_netserve/agillm_acl.conf;
            proxy_pass http://agillm_rpv16_cpu_backend/health;
            proxy_next_upstream off;
            proxy_set_header Host $host;
            proxy_read_timeout 30s;
            proxy_buffering off;
        }

        location = /agillm-rpv16-gen {
            include /opt/agillm43_netserve/agillm_codex_browser_allow.conf;
            include /opt/agillm43_netserve/agillm_acl.conf;
            proxy_pass http://agillm_rpv16_cpu_backend/infer;
            proxy_next_upstream off;
            proxy_set_header Host $host;
            proxy_read_timeout 1800s;
            proxy_buffering off;
            client_max_body_size 2m;
        }

        location = /agillm-rpv16-cancel {
            include /opt/agillm43_netserve/agillm_codex_browser_allow.conf;
            include /opt/agillm43_netserve/agillm_acl.conf;
            proxy_pass http://agillm_rpv16_cpu_backend/cancel;
            proxy_next_upstream off;
            proxy_set_header Host $host;
            proxy_read_timeout 30s;
            proxy_buffering off;
            client_max_body_size 4k;
        }

        location = /agillm-rpv16-stream {
            include /opt/agillm43_netserve/agillm_codex_browser_allow.conf;
            include /opt/agillm43_netserve/agillm_acl.conf;
            proxy_pass http://agillm_rpv16_cpu_backend/stream;
            proxy_next_upstream off;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header Connection "";
            proxy_read_timeout 1800s;
            proxy_buffering off;
            proxy_cache off;
            chunked_transfer_encoding on;
            client_max_body_size 2m;
        }
'''
    s=one(s,marker,routes+marker,'routes')
    return s


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('path'); a=ap.parse_args()
    p=Path(a.path); old=p.read_text(); new=patch(old)
    p.write_text(new); print(f"patched {p}: {len(old)} -> {len(new)} bytes")
if __name__=='__main__': main()
