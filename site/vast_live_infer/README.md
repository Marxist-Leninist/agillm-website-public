# Vast FP32 live inference pool

This directory contains the process supervision and checkpoint rollover pieces used by the AGILLM4.3 public inference service.

## Runtime layout

- `agillm43_http_bridge.py` owns the HTTP API and two warm FP32 CPU workers.
- `supervise_bridge.sh` restarts the bridge if its process exits.
- `watch_latest.py` validates `latest.json`, the full checkpoint wrapper, provenance, tokenizer, and shard set. A candidate must be old enough and stable across repeated observations before promotion.
- Normal checkpoint reloads are blue-green. The complete replacement pool warms first, new requests switch atomically, and old workers drain accepted requests before retirement.
- `supervise_rollover.sh` restarts the watcher if needed.
- `start_live_infer.sh` is idempotent and starts both supervisors without duplicating them.

The public nginx edge uses a managed SSH local forward to this pool as its primary upstream and the local GETH bridge as a passive backup. Credentials remain in deployment environment files and are deliberately not stored here.

## Vast on-start hook

```bash
if [ -x /workspace/agillm43_live_infer/start_live_infer.sh ]; then
  setsid nohup /workspace/agillm43_live_infer/start_live_infer.sh \
    >> /workspace/agillm43_live_infer/startup.log 2>&1 &
fi
```
