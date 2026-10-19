# RPV16 native CPU multimode serving

The HTTP server loads `rpv16_multimode_v1.py`. Availability is derived from the loaded checkpoint, not from a UI-only feature flag. The public model remains a base/completion model; a successful decode test is not a capability benchmark.

## Implemented modes

- `ar`: single-token incremental decoding with the existing KV cache.
- `sat_fixed`: native block-causal attention and the saved two-slot embeddings; a two-token block is computed jointly. A final one-token budget is handled explicitly.
- `sat_variable`: the saved CE-regret network decides between one and two tokens using its speed reward and merge cost. The response includes actual committed and policy stride counts. It is not an AR fallback relabelled as variable SAT.
- `nat`: same-position masked-suffix reconstruction with the checkpoint's causal attention mask, confidence-based refinement, and the original prompt held fixed. The tied full-vocabulary projection is batched; sampling, repetition penalties and confidence ordering remain sequential. Limits are 256 output tokens and eight refinement passes.

SAT uses full block-aware recomputation in this implementation. It is correct to report it as slower than cached AR when measured; do not promise a universal speed advantage.

Single-sequence routing is token-local argmax. Do not balance expert assignments across prompt positions: that couples future input positions into the cached autoregressive prefix.

## HTTP and UI contract

`/health` reports `canonical_modes` and a per-mode capability map. `runtime_sampler` is `ar`, `sat` or `nat` as appropriate. Request `mode=sat` with `var=true/false` remains supported, while stream receipts include the unambiguous canonical mode. Unknown modes fail rather than silently becoming AR.

The UI checks those capabilities, caps the RPV NAT token slider at 256, retains the requested NAT greedy/sample setting, and verifies SAT block-count receipts. Public speed is measured from the button click to the final response, including prefill and transport; decode-only speed must remain a separate metric.

## Checkpoint and promotion requirements

Keep the full trained auxiliary state when exporting a v6 checkpoint: slot embeddings, CE-regret weights, update counters and objective configuration. Preserve `tied_bias` and `tied_logit_scale` for corrected-target checkpoints. A filename containing `v6` is not evidence of its payload schema.

The automatic promoter now rejects candidates missing learned multimode state, verifies all four served mode identities after restart, preserves the four-thread CPU setting, and checks GETH free space before requesting a new export. The canonical systemd override is `zz-autopromote.conf`; an extra later-sorting fixed-checkpoint override would prevent future promotions and must not be left behind.

`export_checkpoint_stream.py` streams a CPU-only conversion to stdout as a tar archive. Run it on the machine holding an immutable source checkpoint, with `CUDA_VISIBLE_DEVICES` empty. Supply `--source`, `--tokenizer`, and optionally `--converter`. Progress goes to stderr. Validate every SHA256SUMS entry and the completion marker before using the destination. The source host does not need room for another complete checkpoint or tar archive.

## Tests

Run the portable contract tests:

```sh
python3 -m unittest discover -s rpv16_cpu -p 'test_native_multimode_contract.py'
```

Deployment evidence for 25 September 2026 is under `receipts/dual_serving_20260925/` on GETH, including real CPU-mode tests, NDJSON tests, public browser generation tests, screenshots, and numerical/performance receipts. The Standard 1.1B transfer, all-shard verification, isolated AR/SAT-variable/NAT canary and blue-green promotion are recorded separately in `receipts/latest_standard_20260925_0641/`.
