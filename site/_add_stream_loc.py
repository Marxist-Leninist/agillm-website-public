#!/usr/bin/env python3
"""Idempotently insert the /agillm-stream location into the infer-edge nginx conf,
mirroring /agillm-gen but proxying to the bridge's NDJSON /stream endpoint."""
import sys, time

CONF = "/opt/agillm43_netserve/agillm43-infer-edge.nginx.conf"
BLOCK = """        location = /agillm-stream {
            include /opt/agillm43_netserve/bonsai_acl.conf;
            proxy_pass http://127.0.0.1:8493/stream;
            proxy_http_version 1.1;
            proxy_set_header Host $host;
            proxy_set_header Connection "";
            proxy_set_header X-AGILLM-Token "<REDACTED_ROTATED_AGILLM_TOKEN>";
            proxy_read_timeout 600s;
            proxy_buffering off;
            proxy_cache off;
            chunked_transfer_encoding on;
            client_max_body_size 2m;
        }

"""
ANCHOR = "        location = /models {"

src = open(CONF).read()
if "/agillm-stream" in src:
    print("already present; no change")
    sys.exit(0)
if ANCHOR not in src:
    print("ANCHOR-NOT-FOUND; aborting, no change")
    sys.exit(2)
# backup
bak = CONF + ".bak-stream-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
open(bak, "w").write(src)
out = src.replace(ANCHOR, BLOCK + ANCHOR, 1)
open(CONF, "w").write(out)
print("inserted /agillm-stream; backup:", bak)
