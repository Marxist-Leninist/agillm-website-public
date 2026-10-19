# SG Vercel relay edge

Vercel project `sg-mcp-relay` sends its short-lived signed OIDC token to the hidden `/_sg-vercel/` edge on `codex.opentransformers.online`. The edge requires both the exact production Vercel project identity and a separate caller relay capability. Both credentials are consumed at GETH and stripped before requests reach the private SG1/SG2 MCP backends.

Live files:
- `/opt/sg-vercel-oidc/verify.py`
- `/etc/systemd/system/sg-vercel-oidc-auth.service`
- `/etc/sg-vercel-relay/capability` (secret, never committed)
- `/opt/agillm43_netserve/sg-vercel-relay.locations.conf`

The verifier stores only the SHA-256 of the caller capability in source. Production fails closed without the capability and Vercel OIDC. Keep access logging disabled on the token-bearing relay locations.
