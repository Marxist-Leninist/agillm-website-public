#!/usr/bin/env python3
import time, shutil, sys
F = "/opt/agillm43_netserve/agillm43-infer-edge.nginx.conf"
src = open(F).read()
if "/agillm-latest" in src:
    print("already-present"); sys.exit(0)
anchor = "        location = /agillm-gen {"
if src.count(anchor) != 1:
    print("ABORT anchor count", src.count(anchor)); sys.exit(1)
block = (
    "        # Latest rescued checkpoint status (from GETH mirror, via console backend)\n"
    "        location = /agillm-latest {\n"
    "            include /opt/agillm43_netserve/bonsai_acl.conf;\n"
    "            proxy_pass http://127.0.0.1:8794/api/checkpoint;\n"
    "            proxy_set_header Host $host;\n"
    "            proxy_read_timeout 15s;\n"
    "            add_header Cache-Control \"no-store\" always;\n"
    "        }\n\n"
)
out = src.replace(anchor, block + anchor, 1)
bak = F + ".bak-latest-" + time.strftime("%Y%m%dT%H%M%SZ")
shutil.copy(F, bak)
open(F, "w").write(out)
print("inserted /agillm-latest; backup", bak)
