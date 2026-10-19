# AGILLM 4.3 GETH Netserve Bridge

Private durability mirror for the AGILLM 4.3 warm HTTP inference bridge running on GETH.

This service exposes the local warm stdin inference server over HTTP for the AGILLM 4.3 GUI/network fallback path. It includes the conditional pipe-resync speed fix from 2026-07-04: after a clean completed request, the bridge skips the old per-request sentinel drain and only resyncs after startup, reload, timeout, child death, or other dirty-pipe conditions.

Do not commit live secrets or runtime state. The repository should track code and templates only:

- `agillm43_http_bridge.py`
- `agillm43_candidate_eval.py`
- `agillm43-bridge.service.unit`
- `service.d/cpuopt.conf`
- `bridge.env.example`

Ignored runtime files include `bridge.env`, `token.txt`, `bridge.log`, backup files, caches, and candidate evaluation outputs.
