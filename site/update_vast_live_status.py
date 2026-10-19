#!/usr/bin/env python3
"""Update /opt/agillm43_netserve/agillm_vast_live_status.json from LIVE state.

Sources (all authoritative, read on every run):
  1. http://127.0.0.1:8498/health  - the live GB10 CUDA bridge (via supervised GETH tunnel)
  2. ssh vast: training_latest.json - the trainer's latest checkpoint metadata (loss, step)
  3. ssh vast: newest completed promotion receipt matching the live serving checkpoint

Writes atomically. Safe to run on a 10-minute timer.
"""
import glob
import hashlib
import ipaddress
import json
import os
import re
import subprocess
import time
import urllib.request

OUT = os.environ.get("AGILLM_LIVE_STATUS_OUT", "/opt/agillm43_netserve/agillm_vast_live_status.json")
TUNNELS = (
    # Probe the live GB10 CUDA primary first, then legacy/remote backups,
    # and finally the local GETH CPU rescue bridge.
    "http://127.0.0.1:8498/health",
    "http://127.0.0.1:8494/health",
    "http://127.0.0.1:8497/health",
    "http://127.0.0.1:8495/health",
    # Last-resort local CPU bridge. This prevents a deleted/unreachable Vast
    # instance from making the website report a total serving outage.
    "http://127.0.0.1:8493/health",
)
HEALTH_TIMEOUT_S = 1.5
HEALTH_ATTEMPTS = 2
LOCAL_CACHE_DIR = "/opt/agillm43_netserve/status_cache"
LOCAL_TRAIN_LATEST = LOCAL_CACHE_DIR + "/training_latest.json"
LOCAL_RUN_STATE = LOCAL_CACHE_DIR + "/run_state.json"
LOCAL_RECEIPT_GLOB = LOCAL_CACHE_DIR + "/receipts/manual_latest_promotion_*.json"
VERIFIED_DIRECT_ROUTE = "/var/lib/agillm43-vast-route/gb10-verified-direct.json"
# AGILLM_V100_HF_TRAINER_SOURCE_20260927: Standard 1.1B now trains on Vast V100
# 52631126 (GB10 51049010 is offline). GETH has no direct route to that box, but the
# trainer publishes every checkpoint to Hugging Face, which GETH already reads for
# serving sync. Use the newest V100 checkpoint there instead of silently presenting
# the 2026-09-15 local cache as the current trainer.
HF_V100_REPO = "MarxistLeninist/AGILLM-4.4"
HF_V100_CACHE = LOCAL_CACHE_DIR + "/hf_v100_training_latest.json"
HF_V100_TTL_S = 600
V100_HOST_LABEL = "V100 52631126"

def _ssh():
    """Build SSH command from the separately verified live Vast inventory route."""
    with open(VERIFIED_DIRECT_ROUTE, "r", encoding="utf-8") as f:
        route = json.load(f)
    if route.get("instance_id") != 51049010:
        raise RuntimeError("verified direct route instance mismatch")
    host = str(ipaddress.ip_address(str(route.get("ssh_host") or "")))
    if not ipaddress.ip_address(host).is_global:
        raise RuntimeError("verified direct route host is not public")
    port = int(route.get("ssh_port") or 0)
    if not 1 <= port <= 65535:
        raise RuntimeError("verified direct route port is invalid")
    verified_at = float(route.get("verified_at_epoch") or 0)
    if time.time() - verified_at > 600:
        raise RuntimeError("verified direct route is stale")
    return [
        "ssh", "-i", "/root/.ssh/goddess_vast",
        "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "UserKnownHostsFile=/var/lib/agillm43-vast-route/gb10_known_hosts",
        "-o", "HostKeyAlias=agillm44-gb10-51049010",
        "-p", str(port), "root@" + host,
    ]
TRAIN_LATEST = (
    "/workspace/gb10_nvfp4_openai_20260915/live_canary/training_latest.json"
)
RUN_STATE = (
    "/workspace/gb10_nvfp4_openai_20260915/live_canary/run_state.json"
)
COMPUTE_STATUS = "/workspace/gb10_nvfp4_openai_20260915/live_canary/compute_status.json"
PROMOTION_RECEIPTS_DIR = "/workspace/agillm43_gb10_infer/receipts"
PROMOTION_RECEIPT_PREFIX = PROMOTION_RECEIPTS_DIR + "/manual_latest_promotion_"
PROMOTION_RECEIPT_GLOB = PROMOTION_RECEIPT_PREFIX + "*.json"

STEP_RE = re.compile(r"step0*([0-9]+)")


def step_of(name: str):
    if not name:
        return None
    m = STEP_RE.search(str(name))
    return int(m.group(1)) if m else None


def _basename(value) -> str:
    return os.path.basename(str(value or ""))


def _nonnegative_int(value) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _health_ready(health: dict) -> bool:
    return bool(
        isinstance(health, dict)
        and health.get("ok") is True
        and health.get("ready") is True
        and _basename(health.get("ckpt"))
    )


def _receipt_health_matches(receipt_health: dict, checkpoint_name: str) -> bool:
    return bool(
        isinstance(receipt_health, dict)
        and receipt_health.get("ok") is True
        and receipt_health.get("ready") is True
        and _basename(receipt_health.get("ckpt")) == checkpoint_name
    )


def promotion_receipt_is_verified(receipt: dict, health: dict):
    """Return (verified, reason) for a receipt bound to current live health."""
    if not _health_ready(health):
        return False, "live_primary_health_unavailable"

    live_ckpt = _basename(health.get("ckpt"))
    live_step = step_of(live_ckpt)
    if not isinstance(receipt, dict) or receipt.get("status") != "complete":
        return False, "receipt_not_complete"
    if _basename(receipt.get("serving_checkpoint")) != live_ckpt:
        return False, "receipt_checkpoint_mismatch"
    if live_step is None or receipt.get("step") != live_step:
        return False, "receipt_step_mismatch"
    if not _receipt_health_matches(receipt.get("primary_health"), live_ckpt):
        return False, "receipt_primary_health_mismatch"
    secondary = receipt.get("secondary_health")
    if secondary is not None:
        if not _receipt_health_matches(secondary, live_ckpt):
            return False, "receipt_secondary_health_mismatch"
    elif receipt.get("schema") != "agillm44.gb10.local-promotion.v1":
        return False, "receipt_secondary_health_missing"

    smoke = receipt.get("smoke")
    if not isinstance(smoke, dict):
        return False, "receipt_smoke_missing"
    if not isinstance(smoke.get("ar"), dict) or smoke["ar"].get("ok") is not True:
        return False, "receipt_ar_smoke_failed"
    if not isinstance(smoke.get("nat"), dict) or smoke["nat"].get("ok") is not True:
        return False, "receipt_nat_smoke_failed"

    sha_total = _nonnegative_int(receipt.get("sha256_total"))
    sha_verified = _nonnegative_int(receipt.get("sha256_verified"))
    if sha_total < 1 or sha_verified < sha_total:
        return False, "receipt_sha256_verification_incomplete"
    return True, "matching_completed_verified_promotion_receipt"


def read_health() -> dict:
    # Tunnel endpoints can briefly reset while a checkpoint reloads. A single
    # short pass used to turn that harmless reload into a cached global outage.
    # Require two complete failed passes before reporting every route down.
    errors = {}
    for attempt in range(HEALTH_ATTEMPTS):
        pass_errors = {}
        for url in TUNNELS:
            try:
                with urllib.request.urlopen(url, timeout=HEALTH_TIMEOUT_S) as r:
                    health = json.loads(r.read().decode())
                if _health_ready(health):
                    health = dict(health)
                    health["health_source"] = url
                    health["health_probe_attempt"] = attempt + 1
                    return health
                pass_errors[url] = health
            except Exception as e:
                pass_errors[url] = {"error": str(e)}
        errors = pass_errors
        if attempt + 1 < HEALTH_ATTEMPTS:
            time.sleep(0.5)
    return {"error": "all configured inference health routes unavailable", "routes": errors}


def read_train_latest() -> dict:
    # GB10 is authoritative. Prefer the freshly verified direct route; retain
    # the local cache only as a control-plane outage fallback.
    try:
        p = subprocess.run(_ssh() + ["cat", TRAIN_LATEST], capture_output=True, text=True, timeout=5)
        if p.returncode != 0:
            raise RuntimeError("training_latest ssh read failed")
        train = json.loads(p.stdout)
        p2 = subprocess.run(_ssh() + ["cat", RUN_STATE], capture_output=True, text=True, timeout=5)
        if p2.returncode == 0:
            rs = json.loads(p2.stdout)
            for key in ("loss", "step", "seen_tok", "effective_seen_tok", "target_tokens", "updated_utc"):
                if rs.get(key) is not None:
                    train[key] = rs[key]
        return train
    except Exception as live_error:
        hf = read_hf_v100_latest()
        if hf.get("checkpoint_name"):
            hf["cache_fallback_reason"] = str(live_error)
            return hf
        try:
            with open(LOCAL_TRAIN_LATEST) as f:
                train = json.load(f)
            with open(LOCAL_RUN_STATE) as f:
                rs = json.load(f)
            for key in ("loss", "step", "seen_tok", "effective_seen_tok", "target_tokens", "updated_utc"):
                if rs.get(key) is not None:
                    train[key] = rs[key]
            train["cache_fallback_reason"] = str(live_error)
            train["trainer_source"] = "stale-local-cache"
            train["hf_error"] = hf.get("error")
            return train
        except Exception:
            return {"error": str(live_error), "hf_error": hf.get("error")}


def read_hf_v100_latest() -> dict:
    """Newest V100 checkpoint metadata published to Hugging Face (cached 10 min)."""
    now = time.time()
    try:
        if now - os.path.getmtime(HF_V100_CACHE) < HF_V100_TTL_S:
            with open(HF_V100_CACHE) as f:
                cached = json.load(f)
            if isinstance(cached, dict) and cached.get("checkpoint_name"):
                return cached
    except (OSError, ValueError):
        pass
    try:
        from huggingface_hub import HfApi, hf_hub_download
        tree = HfApi().list_repo_tree(HF_V100_REPO, path_in_repo="checkpoints", repo_type="model")
        dirs = sorted((x.path for x in tree if "_v100_" in x.path), key=lambda p: step_of(p) or 0)
        if not dirs:
            raise RuntimeError("no V100 checkpoint directories on Hugging Face")
        doc, used, last_err = None, None, None
        for cand in reversed(dirs[-3:]):
            try:
                path = hf_hub_download(HF_V100_REPO, cand + "/training_latest.json", repo_type="model",
                                       cache_dir=LOCAL_CACHE_DIR + "/hf_meta")
                with open(path) as f:
                    doc = json.load(f)
                used = cand
                break
            except Exception as exc:
                last_err = exc
        if not isinstance(doc, dict) or not doc.get("checkpoint_name"):
            raise RuntimeError("no readable training_latest.json in newest V100 dirs: %s" % last_err)
        keep = {k: doc.get(k) for k in ("checkpoint_name", "checkpoint_main_sha256", "step", "seen_tok",
                                        "effective_seen_tok", "target_tokens", "loss", "saved_at", "updated_utc")
                if doc.get(k) is not None}
        keep.update({"trainer_source": "huggingface", "trainer_host": V100_HOST_LABEL,
                     "hf_repo": HF_V100_REPO, "hf_path": used, "hf_checked_at": int(now)})
        tmp = HF_V100_CACHE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(keep, f, indent=2, sort_keys=True)
        os.replace(tmp, HF_V100_CACHE)
        return keep
    except Exception as exc:
        try:
            with open(HF_V100_CACHE) as f:
                cached = json.load(f)
            if isinstance(cached, dict) and cached.get("checkpoint_name"):
                cached["hf_refresh_error"] = "%s: %s" % (type(exc).__name__, str(exc)[:200])
                return cached
        except (OSError, ValueError):
            pass
        return {"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}

# AGI44_COMPUTE_STATUS_MIRROR_20260915: read-only telemetry for the website PFLOP target chip.
def read_compute_status() -> dict:
    try:
        p = subprocess.run(_ssh() + ["cat", COMPUTE_STATUS], capture_output=True, text=True, timeout=5)
        if p.returncode != 0:
            return {}
        d = json.loads(p.stdout)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def read_latest_verified_promotion_receipt(health: dict):
    """Read the newest completed receipt that verifies the live serving checkpoint."""
    if not _health_ready(health):
        return None, None, None, "live_primary_health_unavailable"

    local_paths = sorted(glob.glob(LOCAL_RECEIPT_GLOB), key=os.path.getmtime, reverse=True)
    if local_paths:
        for path in local_paths[:64]:
            try:
                raw = open(path, "rb").read()
                receipt = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            verified, reason = promotion_receipt_is_verified(receipt, health)
            if verified:
                return receipt, path, hashlib.sha256(raw).hexdigest(), reason
        # Local cache may lag a just-completed promotion. Fall through to the
        # verified live route instead of declaring the new serving checkpoint unverified.

    list_command = "ls -1t -- %s 2>/dev/null | head -n 64" % PROMOTION_RECEIPT_GLOB
    try:
        listing = subprocess.run(
            _ssh() + [list_command], capture_output=True, text=True, timeout=20
        )
    except Exception as e:
        return None, None, None, "receipt_listing_failed:%s" % e

    for path in listing.stdout.splitlines()[:2]:
        path = path.strip()
        if not path.startswith(PROMOTION_RECEIPT_PREFIX) or not path.endswith(".json"):
            continue
        try:
            result = subprocess.run(_ssh() + ["cat", path], capture_output=True, timeout=8)
            if result.returncode != 0:
                continue
            receipt = json.loads(result.stdout.decode("utf-8"))
        except Exception:
            continue
        verified, reason = promotion_receipt_is_verified(receipt, health)
        if verified:
            return receipt, path, hashlib.sha256(result.stdout).hexdigest(), reason
    return None, None, None, "no_matching_verified_promotion_receipt"


def build_promotion_metadata(receipt, receipt_path, receipt_sha256, health, reason):
    """Preserve the old metadata shape while deriving every claim from live evidence."""
    primary_ready = _health_ready(health)
    concurrency = _nonnegative_int(health.get("concurrency"))
    live_slots = _nonnegative_int(health.get("live_slots"))
    capacity_ready = primary_ready and concurrency >= 1 and live_slots >= 1
    primary_two_slots = primary_ready and concurrency >= 2 and live_slots >= 2

    if receipt is None:
        acceptance = {
            "checks": {
                "ar_smoke": False,
                "manifest_complete": False,
                "nat_smoke": False,
                "primary_ready": primary_ready,
                "primary_capacity_ready": capacity_ready,
                "primary_two_slots": primary_two_slots,
                "secondary_ready": False,
                "checkpoint_match": False,
                "sha256_all_files": False,
            },
            "primary_concurrency": concurrency,
            "primary_live_slots": live_slots,
            "promotion_receipt_sha256": None,
            "verified_files": 0,
            "reason": reason,
        }
        deployment = {
            "path": None,
            "sha256": None,
            "status": "unavailable",
            "reason": reason,
        }
        return acceptance, deployment

    verified_files = _nonnegative_int(receipt.get("sha256_verified"))
    sha_total = _nonnegative_int(receipt.get("sha256_total"))
    sha_ok = sha_total >= 1 and verified_files >= sha_total
    acceptance = {
        "checks": {
            "ar_smoke": receipt["smoke"]["ar"].get("ok") is True,
            "manifest_complete": sha_ok,
            "nat_smoke": receipt["smoke"]["nat"].get("ok") is True,
            "primary_ready": primary_ready,
            "primary_capacity_ready": capacity_ready,
            "primary_two_slots": primary_two_slots,
            "secondary_ready": _receipt_health_matches(
                receipt.get("secondary_health"), _basename(health.get("ckpt"))
            ),
            "checkpoint_match": (
                _basename(receipt.get("serving_checkpoint"))
                == _basename(health.get("ckpt"))
            ),
            "sha256_all_files": sha_ok,
        },
        "primary_concurrency": concurrency,
        "primary_live_slots": live_slots,
        "promotion_receipt_sha256": receipt_sha256,
        "promotion_step": receipt.get("step"),
        "promotion_completed_at": receipt.get("completed_at"),
        "verified_files": verified_files,
        "source": "latest_matching_completed_promotion_receipt",
    }
    deployment = {
        "path": receipt_path,
        "sha256": receipt_sha256,
        "status": receipt.get("status"),
        "step": receipt.get("step"),
        "completed_at": receipt.get("completed_at"),
    }
    return acceptance, deployment


def main() -> int:
    health = read_health()
    train = read_train_latest()
    compute_status = read_compute_status()
    receipt, receipt_path, receipt_sha256, receipt_reason = (
        read_latest_verified_promotion_receipt(health)
    )
    acceptance, deployment_receipt = build_promotion_metadata(
        receipt, receipt_path, receipt_sha256, health, receipt_reason
    )

    serving_ckpt = health.get("ckpt") or ""
    saved_ckpt = train.get("checkpoint_name") or serving_ckpt
    ckpt = saved_ckpt
    live_step = train.get("trainer_live_step") or train.get("step")
    saved_step = train.get("checkpoint_step") or step_of(saved_ckpt) or train.get("step")
    serving_step = health.get("serving_step") or health.get("step") or step_of(serving_ckpt)
    # Mixed fields: step/file are last saved, never run_state live step.
    step = saved_step or live_step
    now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
    serving_available = bool(_health_ready(health))
    serving_verified = bool(serving_available and receipt is not None)

    status = {
        "authoritative_source": "vast-direct-live",
        "source": "vast.ai",
        "status": ("serving_verified" if serving_verified else
                   "serving_degraded" if serving_available else "serving_unavailable"),
        "status_updated_at": now,
        "step": step,
        "file": ckpt,
        # Operational availability is separate from promotion verification.
        # The UI should remain usable on the local rescue bridge while clearly
        # exposing verification state in promotion_verification/integrity.
        "serving": serving_available,
        "bridge": health,
        "loss": train.get("loss"),
        "seen_tok": train.get("seen_tok"),
        "effective_seen_tok": train.get("effective_seen_tok"),
        "target_tokens": train.get("target_tokens"),
        "compute_status": compute_status,
        "integrity": (
            "live bridge health + matching completed promotion receipt"
            if serving_verified
            else "promotion verification unavailable"
        ),
        "promotion_verification": {
            "ok": serving_verified,
            "reason": receipt_reason,
            "scope": "actual_serving_checkpoint",
            "checkpoint": serving_ckpt,
            "step": serving_step,
        },
        "acceptance": acceptance,
        "deployment_receipt": deployment_receipt,
        "mirror_age_sec": max(0, time.time()-train["observed_at"]) if isinstance(train.get("observed_at"),(int,float)) else None,
        "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime(health["observed_at"])) if isinstance(health.get("observed_at"),(int,float)) else None,
        "content_commit": None if train.get("trainer_source") == "vast-current-fleet" else train.get("content_commit"),
        "updated_utc": train.get("updated_utc"),
        "ui": {
            "max_new_tokens": 2048,
            "nat_default_passes": 1,
            "request_scoped_stop": True,
            "saved_decode_intent": True,
            "url": "https://codex.opentransformers.online/agillm-ui",
        },
    }

    sha = train.get("checkpoint_main_sha256")
    if sha:
        status["checkpoint_sha256"] = sha
        status["training_checkpoint_sha256"] = sha
    if train.get("saved_at"):
        status["checkpoint_saved_at"] = train["saved_at"]
    if serving_step:
        status["package_path"] = health.get("checkpoint_path") or health.get("package_path")
        status["serving_file"] = serving_ckpt
        status["serving_step"] = serving_step
    status["trainer_live_step"] = live_step if train.get("trainer_status") == "training" and train.get("trainer_fresh") else None
    status["trainer_status"] = train.get("trainer_status")
    status["trainer_fresh"] = train.get("trainer_fresh")
    status["training_observed_at"] = train.get("observed_at")
    status["observed_at"] = train.get("observed_at")
    status["model_id"] = "standard"
    status["lineage"] = train.get("lineage") or "standard-1p1b-400b"
    status["trainer_seen_tok"] = train.get("trainer_seen_tok")
    status["training_source_observed_at"] = train.get("source_observed_at")
    status["last_saved_file"] = saved_ckpt
    status["last_saved_step"] = saved_step
    trainer_source = train.get("trainer_source") or ("vast-direct" if "cache_fallback_reason" not in train else "stale-local-cache")
    status["trainer_source"] = trainer_source
    if trainer_source == "huggingface":
        # A published checkpoint is not a live heartbeat: report it as "latest", never "trainer live".
        status["trainer_live_step"] = None
        status["authoritative_source"] = "huggingface"
        status["trainer_host"] = train.get("trainer_host") or V100_HOST_LABEL
        status["trainer_feed"] = "checkpoints via Hugging Face (%s); no live heartbeat link" % train.get("hf_path")
        status["repo"] = train.get("hf_repo")
    elif trainer_source == "stale-local-cache":
        status["trainer_feed"] = "stale local cache; trainer unreachable"

    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(status, f, indent=2, sort_keys=True)
        f.write("\n")
    os.chmod(tmp, 0o644)
    os.replace(tmp, OUT)
    print(json.dumps({
        "step": step,
        "ckpt": ckpt,
        "health_ok": health.get("ok"),
        "promotion_verified": serving_verified,
        "promotion_reason": receipt_reason,
        "promotion_receipt": receipt_path,
        "wrote": OUT,
    }, indent=2))
    return 0


# Current fleet source (2026-09-30), with clocks kept separate from updater attempts.
_FLEET_FILE = "/opt/agillm43_netserve/fleet_training_status.json"
_legacy_read_train_latest = read_train_latest
_legacy_read_health = read_health
def _current_fleet():
    try:
        with open(_FLEET_FILE) as stream:
            data = json.load(stream)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}
def read_train_latest():
    data = _current_fleet()
    model = (data.get("models") or {}).get("standard") or {}
    ckpt = model.get("last_checkpoint_event") or {}
    if not ckpt.get("path"):
        return _legacy_read_train_latest()
    observed = model.get("observed_at")
    fresh = bool(model.get("ok") and model.get("fresh") and isinstance(observed, (int,float)) and -10 <= time.time()-observed <= 100)
    def iso(value):
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(value)) if isinstance(value,(int,float)) else None
    # Every provenance field below comes from this same final checkpoint record.
    return {"checkpoint_name":os.path.basename(ckpt["path"]), "checkpoint_path":ckpt["path"],
            "checkpoint_step":ckpt.get("step"), "step":ckpt.get("step"),
            "checkpoint_main_sha256":ckpt.get("sha256") or "",
            "saved_at":iso(ckpt.get("ts")), "seen_tok":ckpt.get("seen_tokens"),
            "loss":model.get("loss"), "target_tokens":model.get("target_tokens"),
            "lineage":model.get("lineage"), "trainer_live_step":model.get("step"),
            "trainer_seen_tok":model.get("seen_tokens"),
            "trainer_source":"vast-current-fleet", "trainer_host":"V100 52631126",
            "trainer_status":model.get("status") if fresh else "stale",
            "trainer_fresh":fresh, "observed_at":observed,
            "source_observed_at":model.get("source_observed_at"),
            "updated_utc":iso(model.get("source_observed_at"))}
def read_health():
    health = ((_current_fleet().get("serving") or {}).get("standard"))
    if not isinstance(health,dict):
        return _legacy_read_health()
    result = dict(health)
    observed=result.get("observed_at")
    fresh=isinstance(observed,(int,float)) and -10 <= time.time()-observed <= 100
    result["fresh"]=fresh
    result["age_s"]=max(0,time.time()-observed) if isinstance(observed,(int,float)) else None
    if not fresh:
        result.update(ok=False,ready=False,error="Public serving health evidence is stale or unavailable")
    return result
def read_compute_status():
    # Obsolete GB10 estimates must not be assigned to today's Standard or RPV16.
    return {}

# A pinned new receipt has a dedicated strict verifier; prior receipts keep their existing checks.
_legacy_read_latest_verified_promotion_receipt = read_latest_verified_promotion_receipt
_legacy_build_promotion_metadata = build_promotion_metadata
def read_latest_verified_promotion_receipt(health):
    if health.get("ckpt") == "pretrain_final.pt" and health.get("host") == "geth-cpu":
        verifier_path = "/opt/agillm43_netserve/verify_geth_final_promotion.py"
        if hashlib.sha256(open(verifier_path,"rb").read()).hexdigest() != "9962be9b16f323425a282bfcb5e23761f304fadc67db6c1fe65ca67a034c84f5":
            return None, None, None, "geth_final_verifier_source_changed"
        import verify_geth_final_promotion
        return verify_geth_final_promotion.verify(health)
    if health.get("ckpt") == "pretrain_final.pt":
        verifier_path = "/opt/agillm43_netserve/verify_standard_final_promotion.py"
        if hashlib.sha256(open(verifier_path,"rb").read()).hexdigest() != "4f1b3f952c784264b4bab00a999beab5826c05042ef543403061108ea669a79c":
            return None, None, None, "standard_final_verifier_source_changed"
        import verify_standard_final_promotion
        return verify_standard_final_promotion.verify(health)
    return _legacy_read_latest_verified_promotion_receipt(health)
def build_promotion_metadata(receipt, receipt_path, receipt_sha256, health, reason):
    acceptance, deployment = _legacy_build_promotion_metadata(receipt, receipt_path, receipt_sha256, health, reason)
    if isinstance(receipt,dict) and receipt.get("schema") == "agillm.standard.final-promotion.v1":
        acceptance["source"] = "pinned_standard_final_cutover_with_current_public_tunnel_runtime_identity"
        acceptance["verification_scope"] = "actual_serving_checkpoint"
        deployment["verification"] = receipt.get("verification")
    if isinstance(receipt,dict) and receipt.get("schema") == "agillm.standard.geth-final-promotion.v1":
        acceptance["source"] = "pinned_geth_final_cutover_with_current_public_local_runtime_identity"
        acceptance["verification_scope"] = "actual_serving_checkpoint"
        deployment["verification"] = receipt.get("verification")
    return acceptance, deployment

if __name__ == "__main__":
    raise SystemExit(main())
