#!/usr/bin/env python3
"""Forward advisory deployment evidence to the MCP Token Plan supervisor.

Checkpoint bytes never traverse this path. Only the compact, secret-free JSON
evidence is sent. The MCP host owns outbound provider access; the SG1 vault
resolver still releases the credential only in memory.
"""
from __future__ import annotations
import subprocess
import sys

COMMAND = [
    "ssh", "-i", "/root/.ssh/id_ed25519_agillm_supervisor",
    "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
    "root@x.x.x.x", "/opt/goddess/bin/agillm43_deploy_supervisor.py",
]

def main() -> int:
    payload = sys.stdin.read()
    if not payload.strip():
        print('{"ok":false,"error":"empty deployment evidence"}')
        return 2
    try:
        process = subprocess.run(
            COMMAND, input=payload, text=True, capture_output=True, timeout=470,
        )
    except Exception as exc:
        print(
            '{"ok":false,"error":"supervisor transport failed","message":'
            + __import__('json').dumps(f"{type(exc).__name__}: {str(exc)[:500]}") + '}'
        )
        return 1
    if process.stdout:
        sys.stdout.write(process.stdout)
    if process.stderr:
        sys.stderr.write(process.stderr)
    return process.returncode

if __name__ == "__main__":
    raise SystemExit(main())
