#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one match, found {count}")
    return text.replace(old, new, 1)


def patch(text: str) -> str:
    text = replace_once(text,
'''        <button type="button" data-model="standard" class="on">Standard 1.1B</button>
        <button type="button" data-model="popout">Popout 49.1M</button>''',
'''        <button type="button" data-model="standard" class="on">Standard 1.1B</button>
        <button type="button" data-model="rpv16">RPV16 1PF · 2.0B</button>
        <button type="button" data-model="popout">Popout 49.1M</button>''', "model button")

    text = replace_once(text,
'''      <div id="popoutHelp" hidden>''',
'''      <div id="rpv16Help" hidden>
        <b>RPV16 1PF · 2.0B</b> serves the completed corrected-target GB10 checkpoint copied to GETH. The completed source checkpoint is pinned by SHA-256; this website lane uses a derived dynamic-INT8 CPU cache so the 1.99B-parameter model fits and runs on the 16-core GETH CPU.
        It is <b>AR-only</b>: one token per decode step, the trained 16-stage MoE stack, exact RPV16 product-vocabulary probability factorisation, and a causal KV cache. The website intentionally caps prompt/output length for CPU responsiveness. This serving cache is for interactive inference, not a replacement training checkpoint or a benchmark of GB10 1-PFLOP training throughput.
      </div>
      <div id="popoutHelp" hidden>''', "rpv help")

    text = replace_once(text,
'''  function profileModeKeys(profileKey) {
    return (profileKey === "popout" ? MODE_ORDER : STANDARD_MODE_ORDER).slice();
  }''',
'''  function profileModeKeys(profileKey) {
    if (profileKey === "rpv16") return ["ar"];
    return (profileKey === "popout" ? MODE_ORDER : STANDARD_MODE_ORDER).slice();
  }''', "contract profile modes")

    text = replace_once(text,
'''  standard:Object.freeze({ key:"standard", modeKeys:STANDARD_MODE_KEYS, short:"Standard 1.1B", stream:"/agillm-stream", gen:"/agillm-gen", cancel:"/agillm-cancel", health:"/agillm43-infer/health", latest:"/agillm-latest", maxNew:2048, maxNatPasses:64 }),
  popout:Object.freeze({ key:"popout", modeKeys:MODE_KEYS, short:"Popout 49.1M", stream:"/agillm-popout-stream", gen:"/agillm-popout-gen", cancel:"/agillm-popout-cancel", health:"/agillm-popout-health", latest:null, params:49097218, parentStep:2570067, maxNew:512, maxNatPasses:1 })''',
'''  standard:Object.freeze({ key:"standard", modeKeys:STANDARD_MODE_KEYS, short:"Standard 1.1B", stream:"/agillm-stream", gen:"/agillm-gen", cancel:"/agillm-cancel", health:"/agillm43-infer/health", latest:"/agillm-latest", maxNew:2048, maxNatPasses:64 }),
  rpv16:Object.freeze({ key:"rpv16", modeKeys:Object.freeze(["ar"]), short:"RPV16 1PF · 2.0B", stream:"/agillm-rpv16-stream", gen:"/agillm-rpv16-gen", cancel:"/agillm-rpv16-cancel", health:"/agillm-rpv16-health", latest:null, params:1990038784, parentStep:15360, maxNew:1024, maxNatPasses:1 }),
  popout:Object.freeze({ key:"popout", modeKeys:MODE_KEYS, short:"Popout 49.1M", stream:"/agillm-popout-stream", gen:"/agillm-popout-gen", cancel:"/agillm-popout-cancel", health:"/agillm-popout-health", latest:null, params:49097218, parentStep:2570067, maxNew:512, maxNatPasses:1 })''', "model profile")

    text = replace_once(text,
'''const runtimeCapabilities = {
  standard:Object.assign({}, DEFAULT_MODE_CAPABILITIES.standard),
  popout:Object.assign({}, DEFAULT_MODE_CAPABILITIES.popout)
};''',
'''const runtimeCapabilities = {
  standard:Object.assign({}, DEFAULT_MODE_CAPABILITIES.standard),
  rpv16:{ ar:{ trained:true, backend_configured:true, worker_ready:false, available:false, requestable:false, reason:"Checking RPV16 GETH CPU worker", semantics:{ runtime_mode:"ar", runtime_sampler:"ar", commit_policy:"exactly_1_token_per_decode_step" } } },
  popout:Object.assign({}, DEFAULT_MODE_CAPABILITIES.popout)
};''', "runtime capabilities")

    text = replace_once(text,
'''const standardHelp = document.getElementById("standardHelp"), popoutHelp = document.getElementById("popoutHelp");''',
'''const standardHelp = document.getElementById("standardHelp"), rpv16Help = document.getElementById("rpv16Help"), popoutHelp = document.getElementById("popoutHelp");''', "help refs")

    text = replace_once(text,
'''let preferredModeKeys = { standard:"ar", popout:"ar" };''',
'''let preferredModeKeys = { standard:"ar", rpv16:"ar", popout:"ar" };''', "preferred defaults")

    text = replace_once(text,
'''  preferredModeKeys = Object.assign({ standard:canonicalModeKey(null, standardMode, standardSatVar), popout:"ar" },''',
'''  preferredModeKeys = Object.assign({ standard:canonicalModeKey(null, standardMode, standardSatVar), rpv16:"ar", popout:"ar" },''', "restore defaults")

    text = replace_once(text,
'''  preferredModeKeys.popout = canonicalModeKey(preferredModeKeys.popout);''',
'''  preferredModeKeys.rpv16 = "ar";
  preferredModeKeys.popout = canonicalModeKey(preferredModeKeys.popout);''', "restore rpv mode")

    text = replace_once(text,
'''      badge.textContent = (entry.modelKey === "popout" ? "49M·" : "1.1B·") + shortMode;''',
'''      badge.textContent = (entry.modelKey === "popout" ? "49M·" : entry.modelKey === "rpv16" ? "2.0B·" : "1.1B·") + shortMode;''', "history badge")

    text = replace_once(text,
'''function updateModeCapabilities(profileKey, health) {
  // Standard keeps its established four-mode contract. Only Popout is derived
  // from the strict capability envelope returned by its independent bridge.
  if (profileKey !== "popout") return false;''',
'''function updateModeCapabilities(profileKey, health) {
  // Standard keeps its established four-mode contract. RPV16 is a separate
  // AR-only GETH CPU route; Popout keeps its strict multi-mode envelope.
  if (profileKey === "rpv16") {
    const raw = health && health.capabilities && health.capabilities.ar;
    const ready = !!health && health.ok === true && health.ready === true && !!raw
      && raw.trained === true && raw.backend_configured === true && raw.worker_ready === true && raw.available === true;
    runtimeCapabilities.rpv16 = { ar:{ trained:true, backend_configured:!!raw && raw.backend_configured === true,
      worker_ready:ready, available:ready, requestable:ready,
      reason:ready ? "" : "RPV16 GETH CPU worker is not ready",
      semantics:raw && raw.semantics && typeof raw.semantics === "object" ? Object.assign({}, raw.semantics)
        : { runtime_mode:"ar", runtime_sampler:"ar", commit_policy:"exactly_1_token_per_decode_step" } } };
    return true;
  }
  if (profileKey !== "popout") return false;''', "rpv health capabilities")

    text = replace_once(text,
'''function applyModelUi(quiet = false) {
  const profile = modelProfile();
  const isPopout = profile.key === "popout";''',
'''function applyModelUi(quiet = false) {
  const profile = modelProfile();
  const isPopout = profile.key === "popout";
  const isRpv16 = profile.key === "rpv16";''', "apply rpv flag")

    text = replace_once(text,
'''  modelTitle.textContent = isPopout ? "AGILLM Popout 49.1M" : "AGILLM 4.3";''',
'''  modelTitle.textContent = isPopout ? "AGILLM Popout 49.1M" : isRpv16 ? "AGILLM RPV16 1PF" : "AGILLM 4.3";''', "model title")

    text = replace_once(text,
'''  helpToggle.textContent = isPopout ? (requestableKeys.length === 1 && requestableKeys[0] === "ar" ? "AR-ONLY" : "POPOUT") : "BASE";
  helpToggle.title = isPopout ? "About the detachable 49.1M popout" : "About the standard 1.1B base model";
  modelSummaryName.textContent = isPopout ? "Detachable 49.1M student" : "Standard 1.1B base";''',
'''  helpToggle.textContent = isPopout ? (requestableKeys.length === 1 && requestableKeys[0] === "ar" ? "AR-ONLY" : "POPOUT") : isRpv16 ? "1PF CPU" : "BASE";
  helpToggle.title = isPopout ? "About the detachable 49.1M popout" : isRpv16 ? "About the RPV16 1PF GETH CPU checkpoint" : "About the standard 1.1B base model";
  modelSummaryName.textContent = isPopout ? "Detachable 49.1M student" : isRpv16 ? "RPV16 1PF · 2.0B" : "Standard 1.1B base";''', "help title")

    text = replace_once(text,
'''  modelSummaryDetail.textContent = isPopout
    ? "independent " + selectedSpec.label + " inference" + (selectedNeedsWarmup ? " · loads on first run" : "")
    : "continues text, not chat";
  standardHelp.hidden = isPopout;
  popoutHelp.hidden = !isPopout;''',
'''  modelSummaryDetail.textContent = isPopout
    ? "independent " + selectedSpec.label + " inference" + (selectedNeedsWarmup ? " · loads on first run" : "")
    : isRpv16 ? "AR-only · GETH CPU · step 15,360" : "continues text, not chat";
  standardHelp.hidden = isPopout || isRpv16;
  rpv16Help.hidden = !isRpv16;
  popoutHelp.hidden = !isPopout;''', "help visibility")

    text = replace_once(text,
'''  capabilityPrefix.textContent = isPopout ? "49.1M" : "1.1B";''',
'''  capabilityPrefix.textContent = isPopout ? "49.1M" : isRpv16 ? "2.0B" : "1.1B";''', "capability prefix")

    text = replace_once(text,
'''      + (selectedNeedsWarmup ? " · first run loads the GETH worker" : (isPopout ? " · independent GETH CPU worker" : " · " + standardServingDetail));''',
'''      + (selectedNeedsWarmup ? " · first run loads the GETH worker" : (isPopout ? " · independent GETH CPU worker" : isRpv16 ? " · GETH CPU INT8 serving cache" : " · " + standardServingDetail));''', "status detail")

    text = replace_once(text,
'''function primeModelCheckpointUi() {
  if (modelKey === "popout") {''',
'''function primeModelCheckpointUi() {
  if (modelKey === "rpv16") {
    setServingCheckpoint("rpv16_1pf_step15360_cpu_int8.pt");
    latestCkpt.textContent = "RPV16 1PF · completed step 15,360 · 1.99B params · checking GETH CPU worker";
    latestCkpt.title = "Source checkpoint SHA-256 c44bc186dad53042626578658d5c96dc81cf1b9eedef6547838e0b47dc081ac2";
    latestCkpt.classList.remove("ahead");
    mirrorCompact.textContent = "RPV16 · CPU";
    mirrorCompact.title = latestCkpt.textContent;
    mirrorCompact.classList.remove("ahead");
  } else if (modelKey === "popout") {''', "prime rpv")

    text = replace_once(text,
'''      mirrorCompact.title = latestCkpt.textContent;
    } else {
      const checkpointObs = checkpointObservedAt(checkpoint);''',
'''      mirrorCompact.title = latestCkpt.textContent;
    } else if (profile.key === "rpv16") {
      const rm = health && health.model && typeof health.model === "object" ? health.model : {};
      const rstep = rm.step != null ? Number(rm.step) : 15360;
      const rparams = rm.parameters != null ? Number(rm.parameters) : 1990038784;
      const rseen = rm.seen_tokens != null ? Number(rm.seen_tokens) : null;
      const rsha = rm.source_checkpoint_sha256 ? String(rm.source_checkpoint_sha256) : "";
      const servingCkpt = health && health.ckpt ? String(health.ckpt) : "rpv16_1pf_step15360_cpu_int8.pt";
      setServingCheckpoint(servingCkpt);
      latestCkpt.textContent = "Completed source step " + rstep.toLocaleString("en-US")
        + " · " + rparams.toLocaleString("en-US") + " params"
        + (rseen !== null ? " · " + rseen.toLocaleString("en-US") + " training tokens seen" : "")
        + " · GETH CPU dynamic INT8 · " + (health && health.ready === true ? "ready" : "warming");
      latestCkpt.title = "Source checkpoint SHA-256 " + (rsha || "unverified")
        + " · CPU cache is derived for serving only";
      latestCkpt.classList.remove("ahead");
      mirrorCompact.textContent = "RPV16 · step " + rstep.toLocaleString("en-US") + " · CPU";
      mirrorCompact.title = latestCkpt.textContent;
      mirrorCompact.classList.remove("ahead");
    } else {
      const checkpointObs = checkpointObservedAt(checkpoint);''', "refresh rpv")

    text = replace_once(text,
'''    if (seq === checkpointRefreshSeq && profile.key === modelKey && profile.key === "popout") {
      failClosedPopoutCapabilities("Popout health could not be verified; all Popout modes are disabled");
      applyModelUi(true);
      status.textContent = "Popout 49.1M · health unverified · no request can be sent";
    }''',
'''    if (seq === checkpointRefreshSeq && profile.key === modelKey && profile.key === "popout") {
      failClosedPopoutCapabilities("Popout health could not be verified; all Popout modes are disabled");
      applyModelUi(true);
      status.textContent = "Popout 49.1M · health unverified · no request can be sent";
    }
    if (seq === checkpointRefreshSeq && profile.key === modelKey && profile.key === "rpv16") {
      runtimeCapabilities.rpv16 = { ar:{ trained:true, backend_configured:false, worker_ready:false, available:false, requestable:false,
        reason:"RPV16 GETH CPU health could not be verified", semantics:{ runtime_mode:"ar", runtime_sampler:"ar", commit_policy:"exactly_1_token_per_decode_step" } } };
      applyModelUi(true);
      status.textContent = "RPV16 1PF · GETH CPU health unverified · inference disabled";
    }''', "catch rpv")

    return text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()
    p = Path(args.path)
    original = p.read_text()
    updated = patch(original)
    if updated == original:
        raise RuntimeError("patch produced no change")
    p.write_text(updated)
    print(f"patched {p}: {len(original)} -> {len(updated)} bytes")


if __name__ == "__main__":
    main()
