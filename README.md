# AGILLM website (codex.opentransformers.online/agillm-ui) — public source snapshot 2026-10-19

Public, scrubbed mirror (see `REDACTIONS.md` and `PUBLIC_SCRUB.md`: credentials, IP addresses and secret-like strings removed). Snapshot of the AGILLM web UI and its serving glue, exported from the live servers.

| dir | what it is | lives on |
|---|---|---|
| `site/` | `/opt/agillm43_netserve`: the UI pages (`agillm_completion.html` = /agillm-ui, `agillm64b.html` = /agillm-64b), nginx edge config and location includes, bridges, collectors, runtimes | GETH |
| `monitor/` | `/root/agillm41_codex_monitor`: the /api/* backend (training curves, health, generate) | GETH |
| `backend-64b-cpu/` | live CPU inference for the binary 64B: `serve64b.py` (streaming HTTP), `bgemm.c` (1-bit x int8 popcount GEMM, build: `gcc -O3 -march=native -fopenmp -shared -fPIC -o libbgemm.so bgemm.c -lm`) | vast 55575647 |
| `deploy/` | `agillm64b-cpu-tunnel.service` (GETH -> box tunnel) and the /agillm-64b nginx locations | GETH |

Excluded on purpose: backups (`*.bak*`, `*.pre-*`, rollbacks), logs, pid/lock files, `.pyc`, checkpoints (`*.pt`), env files, token/secret/cert/key files.
Credentials still embedded in code were replaced with `REDACTED` (see `REDACTIONS.md`); put real values back from the servers or the SG vault when deploying.
