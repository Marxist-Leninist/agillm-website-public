#!/usr/bin/env python3
"""Isolated four-mode runtime for the AGILLM-4.4 detachable 49.1M export.

The released step-2,570,067 artifact is AR-trained.  Its tied projection can be
executed under SAT-fixed and NAT masks, and it contains a 514-parameter SAT gate,
but those three paths have not yet received their own student objectives.  This
runtime therefore exposes all four algorithms while reporting their trained state
honestly.  It never upgrades capability metadata merely because a path executes.

Checkpoint loading remains bounded and uses ``weights_only=True`` without an
unsafe fallback.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import pathlib
import re
import struct
import sys
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


LEGACY_EXPORT_SCHEMA = "agillm5.detachable-50m.ar-export.v1"
MULTIMODE_EXPORT_SCHEMA = "agillm5.detachable-50m.multimode-export.v2"
CAPABILITY_SCHEMA = "agillm5.detachable-50m.capabilities.v1"
DECODE_TELEMETRY_SCHEMA = "agillm.decode-telemetry.v1"
FOUR_MODES = ("ar", "sat_fixed", "sat_variable", "nat")
TRAINING_RECEIPT_SCHEMA = "agillm5.detachable-50m.multimode-training-receipt.v1"
CHECKPOINT_BINDING_SCHEMA = "agillm5.detachable-50m.checkpoint-binding.v1"
EVALUATOR_SCHEMA = "agillm5.detachable-50m.heldout-evaluator.v1"
MIN_TRAINING_COMMITS = {
    "ar": 128,
    "sat_fixed": 128,
    "sat_variable": 128,
    "nat": 128,
}
MIN_HELDOUT_TOKENS = 4_096
MIN_GATE_CENSUS_ROWS = 512
MIN_GATE_LABELS_PER_CLASS = 32
MIN_THROUGHPUT_PROMPTS = 32
MIN_THROUGHPUT_TOKENS_PER_MODE = 512
MIN_PROMPT_PARITY_CASES = 32
NAT_REFINEMENT_RATIO_DECAY = 0.5
WRAPPER_SCHEMA = "agillm43_zstd_torch_v1"
RELEASE_ARTIFACT_SHA256 = (
    "02b945bf96a7cc15a9cfb13f144929b91bc2eacc3624b4d806e0b7f700233f6a"
)
RELEASE_TOKENIZER_JSON_SHA256 = (
    "0e546c51529290ed4b18dc8b3e481a9ea39a2df2175b65ecb281af6bac79af73"
)
VOCAB_SIZE = 129_280
CORE_PARAMETERS = 49_096_704
EXPORT_PARAMETERS = 49_097_218
MAX_FILE_BYTES = 512 * 1024 * 1024
MAX_RAW_BYTES = 1024 * 1024 * 1024
EXPECTED_CONFIG = {
    "d": 256,
    "layers": 22,
    "heads": 4,
    "rank": 64,
    "tie_kv": True,
    "moe_ffn": False,
}
CHECKPOINT_BINDING_FIELDS = frozenset({
    "schema",
    "source_checkpoint_sha256",
    "source_manifest_sha256",
    "model_state_sha256",
    "sat_variable_gate_state_sha256",
    "export_state_sha256",
    "parent_step",
    "student_popup_step",
})
EVALUATOR_CONTRACT_FIELDS = frozenset({
    "schema",
    "thresholds_frozen_before_evaluation",
    "split",
    "evaluator_sha256",
    "dataset_sha256",
    "evaluation_config_sha256",
})


class CheckpointError(RuntimeError):
    """Raised when an export fails a safety or compatibility contract."""


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_state_sha256(payload: dict[str, Any]) -> str:
    """Canonical digest of the shared core plus the learned SAT gate.

    Projection aliases are intentionally excluded because they are tied to the
    core embedding.  Framing is architecture-independent; numeric bytes are
    explicitly the little-endian tensor storage used by supported deployments.
    """
    if sys.byteorder != "little":
        raise CheckpointError("canonical export-state hashing requires little-endian storage")
    core = payload.get("core")
    sat = payload.get("sat")
    if not isinstance(core, dict) or not isinstance(sat, dict):
        raise CheckpointError("export-state hashing requires core and SAT state")
    named: list[tuple[str, torch.Tensor]] = []
    for key, tensor in core.items():
        if not isinstance(key, str) or not torch.is_tensor(tensor):
            raise CheckpointError("core state has an unhashable entry")
        named.append((f"core.{key}", tensor))
    for key in ("gate.weight", "gate.bias"):
        tensor = sat.get(key)
        if not torch.is_tensor(tensor):
            raise CheckpointError(f"SAT state lacks hash-bound {key}")
        named.append((f"sat.{key}", tensor))
    digest = hashlib.sha256()
    digest.update(b"agillm5.det50.export-state-sha256.v1\0")
    for name, tensor in sorted(named, key=lambda item: item[0]):
        if tensor.layout != torch.strided:
            raise CheckpointError("canonical export-state hashing requires dense tensors")
        value = tensor.detach().cpu().contiguous()
        header = json.dumps(
            {
                "name": name,
                "dtype": str(value.dtype).removeprefix("torch."),
                "shape": list(value.shape),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C")
        digest.update(struct.pack(">Q", len(header)))
        digest.update(header)
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def _checked_sha256(value: Optional[str], label: str) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise CheckpointError(f"{label} must be a 64-character lowercase SHA-256")
    return value


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _state_dict_sha256(state: Any) -> str:
    """Match the v26 trainer's checkpoint-binding tensor digest exactly."""
    if not isinstance(state, dict) or not state:
        raise CheckpointError("checkpoint-binding tensor state is missing")
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise CheckpointError("checkpoint-binding tensor state is malformed")
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(
            json.dumps(list(value.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _payload_gate_binding_state(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    sat = payload.get("sat")
    if not isinstance(sat, dict):
        raise CheckpointError("SAT state is missing from checkpoint binding")
    return {
        "export_sat_gate.weight": sat.get("gate.weight"),
        "export_sat_gate.bias": sat.get("gate.bias"),
    }


def _receipt_evaluator_contract(receipt: Any) -> tuple[Optional[dict[str, Any]], str]:
    if not isinstance(receipt, dict):
        return None, "missing_or_wrong_receipt_schema"
    evaluator = receipt.get("evaluator_contract")
    if not isinstance(evaluator, dict):
        return None, "missing_heldout_evaluator_contract"
    if not EVALUATOR_CONTRACT_FIELDS.issubset(evaluator):
        return None, "incomplete_heldout_evaluator_contract"
    if (
        evaluator.get("schema") != EVALUATOR_SCHEMA
        or evaluator.get("thresholds_frozen_before_evaluation") is not True
        or evaluator.get("split") != "heldout"
    ):
        return None, "invalid_heldout_evaluator_contract"
    for key in (
        "evaluator_sha256",
        "dataset_sha256",
        "evaluation_config_sha256",
    ):
        if not isinstance(evaluator.get(key), str) or not re.fullmatch(
            r"[0-9a-f]{64}", evaluator[key]
        ):
            return None, f"invalid_heldout_evaluator_{key}"
    return evaluator, "qualified"


def _receipt_checkpoint_binding(receipt: Any) -> tuple[Optional[dict[str, Any]], str]:
    """Validate the non-circular receipt binding shared with the v26 exporter."""
    if not isinstance(receipt, dict) or receipt.get("schema") != TRAINING_RECEIPT_SCHEMA:
        return None, "missing_or_wrong_receipt_schema"
    binding = receipt.get("checkpoint_binding")
    if not isinstance(binding, dict) or binding.get("schema") != CHECKPOINT_BINDING_SCHEMA:
        return None, "missing_or_wrong_checkpoint_binding_schema"
    if set(binding) != CHECKPOINT_BINDING_FIELDS:
        return None, "checkpoint_binding_fields_do_not_match_v26_exporter"
    for key in (
        "source_checkpoint_sha256",
        "source_manifest_sha256",
        "model_state_sha256",
        "sat_variable_gate_state_sha256",
        "export_state_sha256",
    ):
        if not isinstance(binding.get(key), str) or not re.fullmatch(
            r"[0-9a-f]{64}", binding[key]
        ):
            return None, f"invalid_checkpoint_binding_{key}"
    for key in ("parent_step", "student_popup_step"):
        if type(binding.get(key)) is not int or int(binding[key]) < 0:
            return None, f"invalid_checkpoint_binding_{key}"
    receipt_digest = receipt.get("receipt_sha256")
    if not isinstance(receipt_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", receipt_digest
    ):
        return None, "missing_or_invalid_receipt_sha256"
    unsigned = copy.deepcopy(receipt)
    unsigned.pop("receipt_sha256", None)
    try:
        canonical_digest = _canonical_json_sha256(unsigned)
    except (TypeError, ValueError):
        return None, "training_receipt_not_canonical_json"
    if canonical_digest != receipt_digest:
        return None, "training_receipt_digest_mismatch"
    return binding, "qualified"


def _export_serving_contract(payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve immutable inference settings from their exporter-owned location."""
    meta = payload.get("agillm5_detachable_50m_export")
    if not isinstance(meta, dict):
        raise CheckpointError("detachable export metadata is missing")
    schema = meta.get("schema")
    if schema == LEGACY_EXPORT_SCHEMA:
        mask_id = payload.get("nat_mask_token_id")
        threshold = 0.95
        refinement_steps = 2
        refinement_ratio_decay = NAT_REFINEMENT_RATIO_DECAY
    elif schema == MULTIMODE_EXPORT_SCHEMA:
        if meta.get("supported_modes") != list(FOUR_MODES):
            raise CheckpointError("multimode export must advertise the exact four modes")
        if (
            type(meta.get("sat_fixed_block")) is not int
            or meta["sat_fixed_block"] != 2
            or type(meta.get("sat_variable_block")) is not int
            or meta["sat_variable_block"] != 2
            or meta.get("sat_prompt_relative_pairing") is not True
        ):
            raise CheckpointError("multimode SAT export contract is malformed")
        threshold = meta.get("sat_variable_stride2_threshold")
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or not 0.5 < float(threshold) < 1.0
        ):
            raise CheckpointError("multimode SAT gate threshold is invalid")
        threshold = float(threshold)
        mask_id = meta.get("nat_mask_token_id")
        refinement_steps = meta.get("nat_refinement_steps")
        if (
            type(refinement_steps) is not int
            or not 1 <= int(refinement_steps) <= 8
        ):
            raise CheckpointError("multimode NAT refinement count is invalid")
        top_level_mask_id = payload.get("nat_mask_token_id")
        if type(top_level_mask_id) is not int or top_level_mask_id != mask_id:
            raise CheckpointError("top-level and export-metadata NAT mask IDs disagree")
        top_level_threshold = payload.get("sat_variable_stride2_threshold")
        if (
            isinstance(top_level_threshold, bool)
            or not isinstance(top_level_threshold, (int, float))
            or not math.isclose(
                float(top_level_threshold), threshold, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise CheckpointError(
                "top-level and export-metadata SAT gate thresholds disagree"
            )
        top_level_refinement_steps = payload.get("nat_refinement_steps")
        if (
            type(top_level_refinement_steps) is not int
            or top_level_refinement_steps != refinement_steps
        ):
            raise CheckpointError(
                "top-level and export-metadata NAT refinement counts disagree"
            )
        # The frozen v26 source manifest and held-out evaluator bind this exact
        # decay. Early v2 exporters did not duplicate it into the detachable
        # payload, so absence means the schema-defined 0.5 rather than a freely
        # selectable default. Any explicit declaration must match exactly.
        refinement_ratio_decay = meta.get(
            "nat_refinement_ratio_decay", NAT_REFINEMENT_RATIO_DECAY
        )
        if (
            isinstance(refinement_ratio_decay, bool)
            or not isinstance(refinement_ratio_decay, (int, float))
            or not math.isfinite(float(refinement_ratio_decay))
            or not math.isclose(
                float(refinement_ratio_decay),
                NAT_REFINEMENT_RATIO_DECAY,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise CheckpointError("multimode NAT refinement decay must be exactly 0.5")
        if "nat_refinement_ratio_decay" in payload:
            top_level_refinement_ratio_decay = payload[
                "nat_refinement_ratio_decay"
            ]
            if (
                isinstance(top_level_refinement_ratio_decay, bool)
                or not isinstance(top_level_refinement_ratio_decay, (int, float))
                or not math.isclose(
                    float(top_level_refinement_ratio_decay),
                    float(refinement_ratio_decay),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise CheckpointError(
                    "top-level and export-metadata NAT refinement decays disagree"
                )
    else:
        raise CheckpointError("unsupported detachable export schema")
    if type(mask_id) is not int or not 0 <= int(mask_id) < VOCAB_SIZE:
        raise CheckpointError("valid NAT mask token ID is required for four-mode inference")
    return {
        "sat_variable_stride2_threshold": float(threshold),
        "nat_mask_token_id": int(mask_id),
        "nat_refinement_steps": int(refinement_steps),
        "nat_refinement_ratio_decay": float(refinement_ratio_decay),
    }


def _weights_only_load(source: Any) -> Any:
    try:
        return torch.load(source, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise CheckpointError(
            "This PyTorch build does not support weights_only=True; refusing unsafe load"
        ) from exc
    except Exception as exc:
        raise CheckpointError(f"safe checkpoint load failed: {type(exc).__name__}: {exc}") from exc


def load_export_payload(
    checkpoint: str | pathlib.Path,
    *,
    expected_sha256: Optional[str] = RELEASE_ARTIFACT_SHA256,
) -> dict[str, Any]:
    """Load the bounded zstd export without enabling arbitrary pickle globals."""
    path = pathlib.Path(checkpoint)
    if not path.is_file():
        raise CheckpointError(f"checkpoint is not a regular file: {path}")
    size = path.stat().st_size
    if size <= 0 or size > MAX_FILE_BYTES:
        raise CheckpointError(f"checkpoint size outside allowed range: {size} bytes")
    expected = _checked_sha256(expected_sha256, "expected checkpoint hash")
    if expected is not None:
        actual = sha256_file(path)
        if actual != expected:
            raise CheckpointError(
                f"checkpoint SHA-256 mismatch: expected {expected}, got {actual}"
            )

    wrapper = _weights_only_load(path)
    if not isinstance(wrapper, dict):
        raise CheckpointError("checkpoint wrapper must be a dictionary")
    if wrapper.get("__agillm43_payload_codec__") != WRAPPER_SCHEMA:
        raise CheckpointError("unsupported checkpoint wrapper schema")
    if wrapper.get("codec") != "zstd":
        raise CheckpointError("checkpoint wrapper is not zstd")
    raw_nbytes = wrapper.get("raw_nbytes")
    packed_nbytes = wrapper.get("packed_nbytes")
    if type(raw_nbytes) is not int or not (0 < raw_nbytes <= MAX_RAW_BYTES):
        raise CheckpointError("invalid or excessive declared decompressed size")
    if type(packed_nbytes) is not int or not (0 < packed_nbytes <= MAX_FILE_BYTES):
        raise CheckpointError("invalid or excessive declared packed size")
    packed = wrapper.get("payload")
    if (
        not torch.is_tensor(packed)
        or packed.device.type != "cpu"
        or packed.dtype != torch.uint8
        or packed.ndim != 1
        or packed.numel() != packed_nbytes
    ):
        raise CheckpointError("wrapper payload must be a bounded one-dimensional uint8 tensor")

    try:
        import zstandard as zstd
    except Exception as exc:
        raise CheckpointError("zstandard is required to decompress this export") from exc
    packed_bytes = packed.contiguous().numpy().tobytes()
    try:
        raw = zstd.ZstdDecompressor().decompress(
            packed_bytes, max_output_size=raw_nbytes
        )
    except Exception as exc:
        raise CheckpointError(f"zstd decompression failed: {type(exc).__name__}: {exc}") from exc
    if len(raw) != raw_nbytes:
        raise CheckpointError(
            f"decompressed size mismatch: expected {raw_nbytes}, got {len(raw)}"
        )
    payload = _weights_only_load(io.BytesIO(raw))
    if not isinstance(payload, dict):
        raise CheckpointError("decompressed payload must be a dictionary")
    validate_export_payload(payload)
    return payload


def validate_export_payload(payload: dict[str, Any]) -> None:
    cfg = payload.get("cfg")
    if not isinstance(cfg, dict) or any(cfg.get(k) != v for k, v in EXPECTED_CONFIG.items()):
        raise CheckpointError(f"unexpected detachable model config: {cfg!r}")
    if type(cfg.get("nat_head")) is not bool:
        raise CheckpointError("detachable nat_head config flag must be boolean")
    if payload.get("tie_weights") is not True:
        raise CheckpointError("export must use tied vocabulary weights")
    mode = payload.get("alibi_mode")
    scale = payload.get("alibi_scale")
    if mode not in {"legacy", "corrected"}:
        raise CheckpointError(f"unsupported ALiBi mode: {mode!r}")
    if not isinstance(scale, (int, float)) or not math.isfinite(float(scale)) or float(scale) < 0:
        raise CheckpointError("invalid ALiBi scale")
    meta = payload.get("agillm5_detachable_50m_export")
    schema = meta.get("schema") if isinstance(meta, dict) else None
    if schema not in {LEGACY_EXPORT_SCHEMA, MULTIMODE_EXPORT_SCHEMA}:
        raise CheckpointError("missing detachable export manifest")
    serving_contract = _export_serving_contract(payload)
    expected_nat_head = schema == MULTIMODE_EXPORT_SCHEMA
    if cfg["nat_head"] is not expected_nat_head:
        raise CheckpointError(
            f"{schema} requires cfg.nat_head={expected_nat_head}"
        )
    if int(meta.get("core_ar_unique_parameters", -1)) != CORE_PARAMETERS:
        raise CheckpointError("core parameter contract mismatch")
    if int(meta.get("unique_parameters", -1)) != EXPORT_PARAMETERS:
        raise CheckpointError("standalone parameter contract mismatch")
    if meta.get("standalone") is not True or meta.get("tied_biasless_vocabulary") is not True:
        raise CheckpointError("export standalone/tied-vocabulary contract mismatch")
    core = payload.get("core")
    if not isinstance(core, dict) or not core:
        raise CheckpointError("export has no core state dictionary")
    if any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in core.items()):
        raise CheckpointError("core state dictionary contains unsupported entries")
    embedding = core.get("emb.weight")
    if not torch.is_tensor(embedding) or tuple(embedding.shape) != (VOCAB_SIZE, 256):
        raise CheckpointError("embedding shape does not match the release contract")
    ar = payload.get("ar")
    if not isinstance(ar, dict) or tuple(getattr(ar.get("proj.weight"), "shape", ())) != (VOCAB_SIZE, 256):
        raise CheckpointError("tied AR projection is missing or malformed")
    sat = payload.get("sat")
    if not isinstance(sat, dict):
        raise CheckpointError("inert SAT compatibility state is missing")
    gate_weight, gate_bias = sat.get("gate.weight"), sat.get("gate.bias")
    if tuple(getattr(gate_weight, "shape", ())) != (2, 256) or tuple(getattr(gate_bias, "shape", ())) != (2,):
        raise CheckpointError("inert SAT compatibility gate is malformed")
    if schema == LEGACY_EXPORT_SCHEMA:
        if torch.count_nonzero(gate_weight).item() or torch.count_nonzero(gate_bias).item():
            raise CheckpointError("legacy SAT compatibility gate is not inert")
    else:
        capabilities = validate_capabilities(
            payload.get("agillm5_detachable_50m_capabilities")
        )
        receipt = payload.get("agillm5_detachable_50m_training_receipt")
        receipt_present = isinstance(receipt, dict)
        if meta.get("training_receipt_present") is not receipt_present:
            raise CheckpointError("multimode receipt-presence metadata is inconsistent")
        if meta.get("trained_sat_variable_gate") is not bool(
            capabilities["modes"]["sat_variable"]["trained"]
        ):
            raise CheckpointError("multimode SAT-gate trained metadata is inconsistent")
        declared_export_digest = _checked_sha256(
            meta.get("export_state_sha256"), "multimode export-state hash"
        )
        actual_export_digest = export_state_sha256(payload)
        if declared_export_digest != actual_export_digest:
            raise CheckpointError(
                "multimode export-state SHA-256 does not bind the core and SAT gate"
            )
        trained_modes = [
            mode_name for mode_name in FOUR_MODES
            if capabilities["modes"][mode_name]["trained"] is True
        ]
        if trained_modes:
            if set(trained_modes) != set(FOUR_MODES):
                raise CheckpointError(
                    "v26 qualified exports must bind all four current objectives"
                )
            binding, binding_reason = _receipt_checkpoint_binding(receipt)
            if binding is None:
                raise CheckpointError(
                    f"trained multimode receipt binding failed: {binding_reason}"
                )
            if meta.get("checkpoint_binding") != binding:
                raise CheckpointError(
                    "export metadata/checkpoint receipt binding mismatch"
                )
            _, evaluator_reason = _receipt_evaluator_contract(receipt)
            if evaluator_reason != "qualified":
                raise CheckpointError(
                    f"trained multimode evaluator contract failed: {evaluator_reason}"
                )
            declared_source_manifest = _checked_sha256(
                meta.get("source_manifest_sha256"), "multimode source-manifest hash"
            )
            if binding["source_manifest_sha256"] != declared_source_manifest:
                raise CheckpointError(
                    "training receipt/source manifest binding mismatch"
                )
            if binding["export_state_sha256"] != actual_export_digest:
                raise CheckpointError(
                    "training receipt/export state binding mismatch"
                )
            if binding["model_state_sha256"] != _state_dict_sha256(core):
                raise CheckpointError(
                    "training receipt/core model-state binding mismatch"
                )
            if binding["sat_variable_gate_state_sha256"] != _state_dict_sha256(
                _payload_gate_binding_state(payload)
            ):
                raise CheckpointError(
                    "training receipt/SAT gate-state binding mismatch"
                )
        elif receipt_present:
            raise CheckpointError("untrained multimode export must not carry a receipt")
        for mode_name in FOUR_MODES:
            if capabilities["modes"][mode_name]["trained"] is True:
                qualified, committed, reason = _receipt_mode_qualification(
                    receipt,
                    mode_name,
                    export_metadata=meta,
                )
                if not qualified:
                    raise CheckpointError(
                        f"{mode_name} is marked trained without a qualifying receipt: {reason}"
                    )
                receipt_id = receipt["heldout_validation"][mode_name]["receipt_id"]
                if capabilities["modes"][mode_name].get(
                    "training_receipt_id"
                ) != receipt_id:
                    raise CheckpointError(
                        f"{mode_name} capability/heldout receipt ID mismatch"
                    )
                if (
                    type(capabilities["modes"][mode_name].get("committed_updates"))
                    is not int
                    or capabilities["modes"][mode_name]["committed_updates"]
                    != committed
                ):
                    raise CheckpointError(
                        f"{mode_name} capability/receipt committed-update mismatch"
                    )
        if capabilities["modes"]["sat_variable"]["trained"] is True:
            throughput_receipt_id = receipt["throughput_benchmark"]["receipt_id"]
            mode_receipt_ids = {
                item.get("training_receipt_id")
                for item in capabilities["modes"].values()
                if item.get("trained") is True
            }
            if throughput_receipt_id in mode_receipt_ids:
                raise CheckpointError(
                    "throughput and per-mode heldout receipt IDs must be distinct"
                )
        if serving_contract["nat_refinement_steps"] != int(
            meta["nat_refinement_steps"]
        ):
            raise CheckpointError("NAT serving contract did not resolve exactly")
        nat = payload.get("nat")
        if not isinstance(nat, dict) or tuple(
            getattr(nat.get("proj.weight"), "shape", ())
        ) != (VOCAB_SIZE, 256):
            raise CheckpointError("multimode export has no tied NAT projection")
    special = payload.get("tokenizer_special")
    if not isinstance(special, dict) or int(special.get("vocab_size", -1)) != VOCAB_SIZE:
        raise CheckpointError("tokenizer vocabulary contract is missing or malformed")
    if not isinstance(payload.get("tokenizer_json"), str) or not payload["tokenizer_json"]:
        raise CheckpointError("exact embedded tokenizer JSON is required")


def legacy_capabilities(payload: dict[str, Any]) -> dict[str, Any]:
    """Describe what legacy AR weights can do without claiming extra training."""
    gate = payload["sat"]
    gate_is_zero = bool(
        torch.count_nonzero(gate["gate.weight"]).item() == 0
        and torch.count_nonzero(gate["gate.bias"]).item() == 0
    )
    return {
        "schema": CAPABILITY_SCHEMA,
        "source_schema": LEGACY_EXPORT_SCHEMA,
        "modes": {
            "ar": {
                "inference": True, "training": True, "trained": True,
                "objective": "causal_shift_1", "status": "released_ar_student",
                "state_binding": "core+tied_vocab:causal_shift_1",
                "telemetry_contract": DECODE_TELEMETRY_SCHEMA,
                "training_receipt_id": "released-ar-export-step-2570067",
            },
            "sat_fixed": {
                "inference": True, "training": True, "trained": False,
                "objective": "sat_block_2_shift_2",
                "status": "executable_shared_weights_requires_sat_training",
                "state_binding": "core+tied_vocab:sat_block_2_shift_2",
                "telemetry_contract": DECODE_TELEMETRY_SCHEMA,
            },
            "sat_variable": {
                "inference": True, "training": True, "trained": False,
                "objective": "sat_fixed_plus_learned_stride_1_or_2_gate",
                "status": (
                    "zero_gate_collapsed_to_stride_1_requires_training"
                    if gate_is_zero else "unverified_gate_requires_training_receipt"
                ),
                "state_binding": "core+tied_vocab+sat.gate:sat_variable_stride",
                "telemetry_contract": DECODE_TELEMETRY_SCHEMA,
            },
            "nat": {
                "inference": True, "training": True, "trained": False,
                "objective": "masked_bidirectional_denoising",
                "status": "executable_tied_head_requires_nat_training",
                "state_binding": "core+tied_vocab:nat_bidirectional_masked_suffix",
                "telemetry_contract": DECODE_TELEMETRY_SCHEMA,
            },
        },
    }


def validate_capabilities(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != CAPABILITY_SCHEMA:
        raise CheckpointError("missing multimode capability contract")
    modes = value.get("modes")
    if not isinstance(modes, dict) or set(modes) != set(FOUR_MODES):
        raise CheckpointError("capability contract must name four distinct modes")
    state_bindings: list[str] = []
    trained_receipt_ids: list[str] = []
    for name in FOUR_MODES:
        item = modes[name]
        if not isinstance(item, dict):
            raise CheckpointError(f"malformed capability record for {name}")
        for flag in ("inference", "training", "trained"):
            if type(item.get(flag)) is not bool:
                raise CheckpointError(f"{name}.{flag} must be boolean")
        if not isinstance(item.get("objective"), str) or not item["objective"]:
            raise CheckpointError(f"{name} objective is missing")
        if not isinstance(item.get("status"), str) or not item["status"]:
            raise CheckpointError(f"{name} status is missing")
        state_binding = item.get("state_binding")
        if not isinstance(state_binding, str) or not state_binding:
            raise CheckpointError(f"{name} state binding is missing")
        if item.get("telemetry_contract") != DECODE_TELEMETRY_SCHEMA:
            raise CheckpointError(f"{name} decode telemetry contract is missing")
        state_bindings.append(state_binding)
        if item["trained"] is True:
            receipt_id = item.get("training_receipt_id")
            if not _valid_receipt_id(receipt_id):
                raise CheckpointError(f"{name} trained state has no valid receipt ID")
            trained_receipt_ids.append(receipt_id)
    if len(set(state_bindings)) != len(FOUR_MODES):
        raise CheckpointError("four-mode state bindings must be distinct")
    if len(set(trained_receipt_ids)) != len(trained_receipt_ids):
        raise CheckpointError("trained mode receipt IDs must be distinct")
    return value


def _valid_receipt_id(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and 8 <= len(value) <= 128
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", value)
    )


def _receipt_mode_qualification(
    receipt: Any,
    mode: str,
    *,
    export_metadata: Optional[dict[str, Any]] = None,
) -> tuple[bool, int, str]:
    """Return strict receipt qualification, committed count, and failure reason."""
    if mode not in MIN_TRAINING_COMMITS:
        raise ValueError(f"no receipt threshold for {mode!r}")
    if not isinstance(receipt, dict) or receipt.get("schema") != TRAINING_RECEIPT_SCHEMA:
        return False, 0, "missing_or_wrong_receipt_schema"
    _, binding_reason = _receipt_checkpoint_binding(receipt)
    if binding_reason != "qualified":
        return False, 0, binding_reason
    _, evaluator_reason = _receipt_evaluator_contract(receipt)
    if evaluator_reason != "qualified":
        return False, 0, evaluator_reason
    commits = receipt.get("committed_objective_updates")
    if not isinstance(commits, dict) or type(commits.get(mode)) is not int:
        return False, 0, "missing_integer_committed_updates"
    count = int(commits[mode])
    if count < int(MIN_TRAINING_COMMITS[mode]):
        return False, count, "committed_updates_below_threshold"
    validation = receipt.get("heldout_validation")
    check = validation.get(mode) if isinstance(validation, dict) else None
    if not isinstance(check, dict) or check.get("passed") is not True:
        return False, count, "heldout_validation_not_passed"
    if not _valid_receipt_id(check.get("receipt_id")):
        return False, count, "missing_or_invalid_mode_receipt_id"
    if type(check.get("tokens")) is not int or int(check["tokens"]) < MIN_HELDOUT_TOKENS:
        return False, count, "heldout_token_count_below_threshold"
    cross_entropy = check.get("cross_entropy")
    if (
        isinstance(cross_entropy, bool)
        or not isinstance(cross_entropy, (int, float))
        or not math.isfinite(float(cross_entropy))
        or float(cross_entropy) < 0.0
    ):
        return False, count, "heldout_cross_entropy_not_finite"
    maximum_cross_entropy = check.get("maximum_cross_entropy")
    baseline_cross_entropy = check.get("baseline_cross_entropy")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in (maximum_cross_entropy, baseline_cross_entropy)
    ):
        return False, count, "heldout_ce_thresholds_not_finite"
    if not (
        float(cross_entropy)
        <= float(maximum_cross_entropy)
        <= float(baseline_cross_entropy)
    ):
        return False, count, "heldout_ce_not_below_frozen_threshold_and_baseline"
    if mode == "sat_fixed" and check.get("fixed_pair_contract_passed") is not True:
        return False, count, "fixed_pair_contract_not_passed"
    if mode in {"sat_fixed", "sat_variable"}:
        if check.get("prompt_relative_pairing_contract_passed") is not True:
            return False, count, "prompt_relative_pairing_contract_not_passed"
        if (
            type(check.get("odd_prompt_cases")) is not int
            or type(check.get("even_prompt_cases")) is not int
            or int(check["odd_prompt_cases"]) < MIN_PROMPT_PARITY_CASES
            or int(check["even_prompt_cases"]) < MIN_PROMPT_PARITY_CASES
        ):
            return False, count, "odd_even_prompt_census_below_threshold"
    if mode == "sat_variable":
        if check.get("realignment_contract_passed") is not True:
            return False, count, "sat_variable_realignment_contract_not_passed"
        gate = check.get("gate_census")
        if not isinstance(gate, dict):
            return False, count, "missing_gate_census"
        required_integer_fields = (
            "rows", "stride1", "stride2", "label_stride1", "label_stride2"
        )
        if any(type(gate.get(name)) is not int for name in required_integer_fields):
            return False, count, "malformed_gate_census_counts"
        rows = int(gate["rows"])
        stride1, stride2 = int(gate["stride1"]), int(gate["stride2"])
        label_stride1 = int(gate["label_stride1"])
        label_stride2 = int(gate["label_stride2"])
        accuracy = gate.get("balanced_accuracy")
        if (
            isinstance(accuracy, bool)
            or not isinstance(accuracy, (int, float))
            or not math.isfinite(float(accuracy))
        ):
            return False, count, "malformed_gate_balanced_accuracy"
        if rows < MIN_GATE_CENSUS_ROWS or stride1 <= 0 or stride2 <= 0:
            return False, count, "gate_census_too_small_or_single_class"
        if stride1 + stride2 != rows:
            return False, count, "gate_census_counts_do_not_sum"
        if (
            label_stride1 < MIN_GATE_LABELS_PER_CLASS
            or label_stride2 < MIN_GATE_LABELS_PER_CLASS
            or label_stride1 + label_stride2 != rows
        ):
            return False, count, "gate_label_census_too_small_or_inconsistent"
        calibration_error = gate.get("ece")
        brier_score = gate.get("brier")
        false_accept_rate = gate.get("false_accept_rate")
        metrics = (calibration_error, brier_score, false_accept_rate)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in metrics
        ):
            return False, count, "missing_or_nonfinite_gate_calibration_metrics"
        gate_threshold = gate.get("threshold")
        expected_gate_threshold = (
            export_metadata.get("sat_variable_stride2_threshold")
            if isinstance(export_metadata, dict) else None
        )
        if (
            not isinstance(gate_threshold, (int, float))
            or isinstance(gate_threshold, bool)
            or not isinstance(expected_gate_threshold, (int, float))
            or isinstance(expected_gate_threshold, bool)
            or not math.isclose(
                float(gate_threshold),
                float(expected_gate_threshold),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return False, count, "gate_census_threshold_must_match_export_metadata"
        if (
            float(accuracy) < 0.55
            or float(calibration_error) > 0.12
            or float(brier_score) > 0.24
            or float(false_accept_rate) > 0.05
            or gate.get("collapsed") is not False
        ):
            return False, count, "gate_accuracy_or_collapse_check_failed"
        throughput = receipt.get("throughput_benchmark")
        if not isinstance(throughput, dict) or throughput.get("passed") is not True:
            return False, count, "missing_or_failed_throughput_benchmark"
        if not _valid_receipt_id(throughput.get("receipt_id")):
            return False, count, "missing_or_invalid_throughput_receipt_id"
        if (
            type(throughput.get("prompt_count")) is not int
            or int(throughput["prompt_count"]) < MIN_THROUGHPUT_PROMPTS
        ):
            return False, count, "throughput_prompt_count_below_threshold"
        benchmark_tokens = throughput.get("generated_tokens")
        rates = throughput.get("tokens_per_second")
        if not isinstance(benchmark_tokens, dict) or not isinstance(rates, dict):
            return False, count, "malformed_throughput_benchmark"
        speed_modes = ("ar", "sat_variable", "sat_fixed")
        if any(
            type(benchmark_tokens.get(name)) is not int
            or int(benchmark_tokens[name]) < MIN_THROUGHPUT_TOKENS_PER_MODE
            for name in speed_modes
        ):
            return False, count, "throughput_token_count_below_threshold"
        if any(
            isinstance(rates.get(name), bool)
            or not isinstance(rates.get(name), (int, float))
            or not math.isfinite(float(rates[name]))
            or float(rates[name]) <= 0.0
            for name in speed_modes
        ):
            return False, count, "throughput_rates_not_finite_positive"
        if not (
            float(rates["ar"])
            < float(rates["sat_variable"])
            < float(rates["sat_fixed"])
        ):
            return False, count, "throughput_order_not_ar_lt_variable_lt_fixed"
    if mode == "nat":
        if check.get("clean_prefix_masked_suffix") is not True:
            return False, count, "nat_clean_prefix_masked_suffix_not_verified"
        expected_refinement_steps = (
            export_metadata.get("nat_refinement_steps")
            if isinstance(export_metadata, dict) else None
        )
        if (
            type(check.get("refinement_passes")) is not int
            or type(expected_refinement_steps) is not int
            or int(check["refinement_passes"]) != int(expected_refinement_steps)
        ):
            return False, count, "nat_refinement_contract_not_verified"
    return True, count, "qualified"


def build_multimode_export_payload(
    legacy_payload: dict[str, Any],
    training_receipt: Optional[dict[str, Any]] = None,
    *,
    sat_variable_stride2_threshold: float = 0.95,
    nat_mask_token_id: Optional[int] = None,
    nat_refinement_steps: int = 2,
) -> dict[str, Any]:
    """Create a v2 multimode payload without forging training completion.

    A mode is marked trained only after a thresholded update count and a held-out
    validation receipt.  SAT-variable additionally needs a non-collapsed gate
    census.  The tied SAT/NAT projections add no unique parameters.
    """
    validate_export_payload(legacy_payload)
    result = copy.deepcopy(legacy_payload)
    capabilities = legacy_capabilities(result)
    receipt = training_receipt if isinstance(training_receipt, dict) else {}
    resolved_mask_id = (
        legacy_payload.get("nat_mask_token_id")
        if nat_mask_token_id is None else nat_mask_token_id
    )
    if type(resolved_mask_id) is not int or not 0 <= resolved_mask_id < VOCAB_SIZE:
        raise CheckpointError("multimode export requires a valid NAT mask token ID")
    if (
        isinstance(sat_variable_stride2_threshold, bool)
        or not isinstance(sat_variable_stride2_threshold, (int, float))
        or not math.isfinite(float(sat_variable_stride2_threshold))
        or not 0.5 < float(sat_variable_stride2_threshold) < 1.0
    ):
        raise CheckpointError("multimode export SAT gate threshold is invalid")
    if type(nat_refinement_steps) is not int or not 1 <= nat_refinement_steps <= 8:
        raise CheckpointError("multimode export NAT refinement count is invalid")
    result["nat"] = {"proj.weight": result["core"]["emb.weight"]}
    result["cfg"]["nat_head"] = True
    result["nat_mask_token_id"] = resolved_mask_id
    result["sat_variable_stride2_threshold"] = float(
        sat_variable_stride2_threshold
    )
    result["nat_refinement_steps"] = nat_refinement_steps
    result["nat_refinement_ratio_decay"] = NAT_REFINEMENT_RATIO_DECAY
    meta = dict(result["agillm5_detachable_50m_export"])
    binding = receipt.get("checkpoint_binding") if isinstance(receipt, dict) else None
    state_digest = export_state_sha256(result)
    meta.update(
        schema=MULTIMODE_EXPORT_SCHEMA,
        legacy_source_schema=LEGACY_EXPORT_SCHEMA,
        mode_names=list(FOUR_MODES),
        supported_modes=list(FOUR_MODES),
        sat_fixed_block=2,
        sat_variable_block=2,
        sat_prompt_relative_pairing=True,
        sat_variable_stride2_threshold=float(sat_variable_stride2_threshold),
        nat_mask_token_id=REDACTED
        nat_refinement_steps=nat_refinement_steps,
        nat_refinement_ratio_decay=NAT_REFINEMENT_RATIO_DECAY,
        mode_mixture={
            "ar": 0.80,
            "sat_fixed": 0.05,
            "sat_variable": 0.05,
            "nat": 0.10,
        },
        training_receipt_present=isinstance(training_receipt, dict),
        trained_sat_variable_gate=False,
        export_state_sha256=state_digest,
    )
    if isinstance(binding, dict):
        meta["checkpoint_binding"] = copy.deepcopy(binding)
        if isinstance(binding.get("source_manifest_sha256"), str):
            meta["source_manifest_sha256"] = binding["source_manifest_sha256"]
    for mode in FOUR_MODES:
        item = capabilities["modes"][mode]
        item["trained"] = False
        item.pop("training_receipt_id", None)
        item.pop("committed_updates", None)
        item["status"] = "current_multimode_weights_require_heldout_receipt"
        threshold = int(MIN_TRAINING_COMMITS[mode])
        passed, count, reason = _receipt_mode_qualification(
            receipt,
            mode,
            export_metadata=meta,
        )
        if passed:
            item["trained"] = True
            item["status"] = "trained_and_heldout_validated"
            item["committed_updates"] = count
            item["training_receipt_id"] = receipt[
                "heldout_validation"
            ][mode]["receipt_id"]
        else:
            item["required_committed_updates"] = threshold
            item["observed_committed_updates"] = count
            item["qualification_failure"] = reason
    all_trained = all(capabilities["modes"][mode]["trained"] for mode in FOUR_MODES)
    if isinstance(training_receipt, dict) and not all_trained:
        failures = {
            mode: capabilities["modes"][mode].get("qualification_failure")
            for mode in FOUR_MODES
            if capabilities["modes"][mode]["trained"] is not True
        }
        raise CheckpointError(
            f"training receipt does not qualify the exact v26 export: {failures}"
        )
    meta["trained_sat_variable_gate"] = bool(
        capabilities["modes"]["sat_variable"]["trained"]
    )
    result["agillm5_detachable_50m_capabilities"] = capabilities
    if isinstance(training_receipt, dict):
        result["agillm5_detachable_50m_training_receipt"] = copy.deepcopy(
            training_receipt
        )
    result["agillm5_detachable_50m_export"] = meta
    validate_export_payload(result)
    return result


def _alibi_slopes(heads: int, device: torch.device) -> torch.Tensor:
    def powers_of_two(count: int) -> list[float]:
        start = 2 ** (-2 ** -(math.log2(count) - 3))
        return [start * (start**i) for i in range(count)]

    if math.log2(heads).is_integer():
        values = powers_of_two(heads)
    else:
        closest = 2 ** math.floor(math.log2(heads))
        values = powers_of_two(closest)
        values += powers_of_two(2 * closest)[0::2][: heads - closest]
    return torch.tensor(values, device=device, dtype=torch.float32).view(1, heads, 1, 1)


class AGILLM44Attention(nn.Module):
    def __init__(self, d: int, heads: int, rank: int, tie_kv: bool = True):
        super().__init__()
        if d % heads:
            raise ValueError("hidden width must be divisible by attention heads")
        self.heads, self.head_dim, self.rank = heads, d // heads, rank
        self.tie_kv = bool(tie_kv)
        self.qkv = nn.Linear(d, (2 if self.tie_kv else 3) * d, bias=False)
        self.U = nn.Parameter(torch.empty(self.head_dim, rank))
        nn.init.orthogonal_(self.U)
        self.proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(0.1)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = value.shape
        return value.view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        x: torch.Tensor,
        *,
        past: Optional[tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool,
        alibi_mode: str,
        alibi_scale: float,
        attention_mode: str,
        sat_block: int = 2,
        sat_phase: int = 0,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        if self.tie_kv:
            q_linear, kv_linear = self.qkv(x).chunk(2, dim=-1)
            k_linear = v_linear = kv_linear
        else:
            q_linear, k_linear, v_linear = self.qkv(x).chunk(3, dim=-1)
        if self.rank > self.head_dim:
            metric = self.U @ self.U.T
            q = self._heads(q_linear) @ metric
            k_new = self._heads(k_linear)
            v_new = k_new if self.tie_kv else self._heads(v_linear)
        else:
            q = self._heads(q_linear) @ self.U
            k_new = self._heads(k_linear) @ self.U
            v_new = self._heads(v_linear)
        if past is None:
            k, v = k_new, v_new
        else:
            old_k, old_v = past
            if old_k.ndim != 4 or old_v.ndim != 4 or old_k.size(2) != old_v.size(2):
                raise ValueError("malformed KV cache")
            k = torch.cat((old_k, k_new), dim=2)
            v = torch.cat((old_v, v_new), dim=2)
        query_len, key_len = q.size(2), k.size(2)
        query_start = key_len - query_len
        q_pos = torch.arange(query_start, key_len, device=x.device).view(query_len, 1)
        k_pos = torch.arange(key_len, device=x.device).view(1, key_len)
        if attention_mode == "causal":
            allowed = k_pos <= q_pos
        elif attention_mode == "sat":
            if int(sat_block) != 2:
                raise ValueError("detachable SAT currently requires block size 2")
            if int(sat_phase) not in {0, 1}:
                raise ValueError("detachable SAT phase must be 0 or 1")
            phase = int(sat_phase)
            query_before_phase = q_pos < phase
            key_before_phase = k_pos < phase
            query_block = torch.div(
                q_pos - phase, sat_block, rounding_mode="floor"
            )
            key_block = torch.div(
                k_pos - phase, sat_block, rounding_mode="floor"
            )
            paired_allowed = key_before_phase | (
                (~key_before_phase) & (key_block <= query_block)
            )
            allowed = torch.where(query_before_phase, k_pos <= q_pos, paired_allowed)
        elif attention_mode == "nat":
            if past is not None:
                raise ValueError("NAT bidirectional attention cannot reuse a KV cache")
            allowed = torch.ones(
                (query_len, key_len), dtype=torch.bool, device=x.device
            )
        else:
            raise ValueError(f"unknown attention mode {attention_mode!r}")
        mask = torch.where(
            allowed,
            torch.zeros((), device=x.device, dtype=q.dtype),
            torch.full((), float("-inf"), device=x.device, dtype=q.dtype),
        ).view(1, 1, query_len, key_len)
        if float(alibi_scale) != 0.0:
            if alibi_mode == "corrected":
                distance = (q_pos - k_pos).abs()
            else:
                distance = (k_pos - q_pos).clamp_min(0)
            bias = -_alibi_slopes(self.heads, x.device) * distance.view(1, 1, query_len, key_len)
            mask = mask + bias.to(mask.dtype) * float(alibi_scale)
        try:
            attended = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0,
                scale=1.0 / math.sqrt(self.head_dim),
            )
        except TypeError:
            q = q * math.sqrt(q.size(-1) / self.head_dim)
            attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(x.size(0), x.size(1), -1)
        output = self.drop(self.proj(attended))
        return output, ((k, v) if use_cache else None)


class AGILLM44Block(nn.Module):
    def __init__(self, d: int, heads: int, rank: int, tie_kv: bool):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mha = AGILLM44Attention(d, heads, rank, tie_kv)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))

    def forward(
        self,
        x: torch.Tensor,
        *,
        past: Optional[tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool,
        alibi_mode: str,
        alibi_scale: float,
        attention_mode: str,
        sat_block: int = 2,
        sat_phase: int = 0,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        attended, present = self.mha(
            self.ln1(x), past=past, use_cache=use_cache,
            alibi_mode=alibi_mode, alibi_scale=alibi_scale,
            attention_mode=attention_mode, sat_block=sat_block,
            sat_phase=sat_phase,
        )
        x = x + attended
        x = x + self.ff(self.ln2(x))
        return x, present


class AGILLM44Detachable50M(nn.Module):
    """49,096,704-parameter core with causal, block-SAT and NAT masks."""

    def __init__(self, *, alibi_mode: str, alibi_scale: float):
        super().__init__()
        self.alibi_mode = alibi_mode
        self.alibi_scale = float(alibi_scale)
        self.emb = nn.Embedding(VOCAB_SIZE, 256)
        self.blocks = nn.ModuleList(
            [AGILLM44Block(256, 4, 64, True) for _ in range(22)]
        )
        self.ln = nn.LayerNorm(256)

    def hidden(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: Optional[Sequence[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        attention_mode: str = "causal",
        sat_block: int = 2,
        sat_phase: int = 0,
    ) -> tuple[torch.Tensor, Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        if input_ids.ndim != 2 or input_ids.dtype not in {torch.int32, torch.int64}:
            raise ValueError("input_ids must be a rank-2 integer tensor")
        if input_ids.numel() == 0:
            raise ValueError("input_ids cannot be empty")
        if past_key_values is not None and len(past_key_values) != len(self.blocks):
            raise ValueError("KV-cache layer count mismatch")
        hidden = self.emb(input_ids)
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        for index, block in enumerate(self.blocks):
            past = None if past_key_values is None else past_key_values[index]
            hidden, present = block(
                hidden, past=past, use_cache=use_cache,
                alibi_mode=self.alibi_mode, alibi_scale=self.alibi_scale,
                attention_mode=attention_mode, sat_block=sat_block,
                sat_phase=sat_phase,
            )
            if use_cache:
                assert present is not None
                presents.append(present)
        return self.ln(hidden), (presents if use_cache else None)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: Optional[Sequence[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        attention_mode: str = "causal",
        sat_block: int = 2,
        sat_phase: int = 0,
    ) -> tuple[torch.Tensor, Optional[list[tuple[torch.Tensor, torch.Tensor]]]]:
        hidden, presents = self.hidden(
            input_ids,
            past_key_values=REDACTED
            use_cache=use_cache,
            attention_mode=attention_mode,
            sat_block=sat_block,
            sat_phase=sat_phase,
        )
        return F.linear(hidden, self.emb.weight, None), presents


@dataclass(frozen=True)
class RestoredTokenizer:
    backend: Any
    tokenizer_id: str
    eos_token_id: int
    pad_token_id: int
    tokenizer_json_sha256: str

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return list(self.backend.encode(text, add_special_tokens=add_special_tokens).ids)

    def decode(self, ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        return self.backend.decode([int(v) for v in ids], skip_special_tokens=skip_special_tokens)


def restore_tokenizer(
    payload: dict[str, Any],
    *,
    expected_json_sha256: Optional[str] = RELEASE_TOKENIZER_JSON_SHA256,
) -> RestoredTokenizer:
    tokenizer_json = payload.get("tokenizer_json")
    if not isinstance(tokenizer_json, str) or not tokenizer_json:
        raise CheckpointError("checkpoint has no embedded tokenizer JSON")
    actual_hash = hashlib.sha256(tokenizer_json.encode("utf-8")).hexdigest()
    expected = _checked_sha256(expected_json_sha256, "expected tokenizer JSON hash")
    if expected is not None and actual_hash != expected:
        raise CheckpointError(
            f"tokenizer JSON SHA-256 mismatch: expected {expected}, got {actual_hash}"
        )
    try:
        from tokenizers import Tokenizer
        backend = Tokenizer.from_str(tokenizer_json)
    except Exception as exc:
        raise CheckpointError(f"embedded tokenizer restoration failed: {type(exc).__name__}: {exc}") from exc
    vocab = backend.get_vocab(with_added_tokens=True)
    if not vocab or len(vocab) != VOCAB_SIZE or max(vocab.values()) + 1 != VOCAB_SIZE:
        raise CheckpointError("restored tokenizer vocabulary does not match 129,280 IDs")
    special = payload.get("tokenizer_special")
    if not isinstance(special, dict):
        raise CheckpointError("tokenizer special-token contract is missing")
    eos_id, pad_id = special.get("eos_token_id"), special.get("pad_token_id")
    if type(eos_id) is not int or type(pad_id) is not int:
        raise CheckpointError("integer EOS and PAD token IDs are required")
    if not (0 <= eos_id < VOCAB_SIZE and 0 <= pad_id < VOCAB_SIZE):
        raise CheckpointError("EOS or PAD token ID is outside the vocabulary")
    for token_key, id_key in (("eos_token", "eos_token_id"), ("pad_token", "pad_token_id")):
        token = REDACTED
        token_id = REDACTED
        if isinstance(token, str) and token and backend.token_to_id(token) != token_id:
            raise CheckpointError(f"{token_key} does not resolve to the declared ID")
    tokenizer_id = payload.get("tokenizer_id")
    if not isinstance(tokenizer_id, str) or not tokenizer_id.strip():
        raise CheckpointError("tokenizer ID is missing")
    return RestoredTokenizer(backend, tokenizer_id, eos_id, pad_id, actual_hash)


@dataclass
class LoadedModel:
    model: AGILLM44Detachable50M
    sat_gate: nn.Linear
    tokenizer: RestoredTokenizer
    metadata: dict[str, Any]
    capabilities: dict[str, Any]
    nat_mask_token_id: int
    sat_variable_stride2_threshold: float
    nat_refinement_steps: int
    nat_refinement_ratio_decay: float
    device: torch.device
    dtype: torch.dtype


def _resolve_device(value: str) -> torch.device:
    value = str(value).strip().lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("only CPU and CUDA devices are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    return device


def _resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    value = str(value).strip().lower()
    if value == "auto":
        if device.type == "cpu":
            return torch.float32
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    choices = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if value not in choices:
        raise ValueError(f"unsupported dtype: {value}")
    dtype = choices[value]
    if device.type == "cpu" and dtype is not torch.float32:
        raise ValueError("CPU inference is restricted to float32 for portability")
    if device.type == "cuda" and dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("this CUDA device does not support bfloat16")
    return dtype


def _place_sat_gate_fp32(sat_gate: nn.Linear, device: torch.device) -> nn.Linear:
    """Keep the qualified SAT-variable gate in FP32 on the serving device."""
    return sat_gate.to(device=device, dtype=torch.float32).eval()


def load_model(
    checkpoint: str | pathlib.Path,
    *,
    device: str = "auto",
    dtype: str = "auto",
    expected_sha256: Optional[str] = RELEASE_ARTIFACT_SHA256,
    expected_tokenizer_sha256: Optional[str] = RELEASE_TOKENIZER_JSON_SHA256,
) -> LoadedModel:
    payload = load_export_payload(checkpoint, expected_sha256=expected_sha256)
    tokenizer = restore_tokenizer(payload, expected_json_sha256=expected_tokenizer_sha256)
    target_device = _resolve_device(device)
    target_dtype = _resolve_dtype(dtype, target_device)
    model = AGILLM44Detachable50M(
        alibi_mode=str(payload["alibi_mode"]), alibi_scale=float(payload["alibi_scale"])
    )
    missing, unexpected = model.load_state_dict(payload["core"], strict=True)
    if missing or unexpected:
        raise CheckpointError(f"strict state load mismatch: missing={missing}, unexpected={unexpected}")
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != CORE_PARAMETERS:
        raise CheckpointError(f"loaded core has {count:,} parameters, expected {CORE_PARAMETERS:,}")
    sat_gate = nn.Linear(256, 2, bias=True)
    sat_gate.load_state_dict(
        {
            "weight": payload["sat"]["gate.weight"],
            "bias": payload["sat"]["gate.bias"],
        },
        strict=True,
    )
    if count + sum(parameter.numel() for parameter in sat_gate.parameters()) != EXPORT_PARAMETERS:
        raise CheckpointError("core plus SAT-variable gate parameter contract mismatch")
    model.to(device=target_device, dtype=target_dtype).eval()
    sat_gate = _place_sat_gate_fp32(sat_gate, target_device)
    if (
        sat_gate.weight.dtype != torch.float32
        or sat_gate.bias is None
        or sat_gate.bias.dtype != torch.float32
    ):
        raise CheckpointError("SAT-variable serving gate must remain FP32")
    serving_contract = _export_serving_contract(payload)
    nat_mask_token_id = serving_contract["nat_mask_token_id"]
    capabilities = (
        validate_capabilities(payload["agillm5_detachable_50m_capabilities"])
        if payload["agillm5_detachable_50m_export"]["schema"] == MULTIMODE_EXPORT_SCHEMA
        else legacy_capabilities(payload)
    )
    metadata = dict(payload["agillm5_detachable_50m_export"])
    metadata.update(
        alibi_mode=payload["alibi_mode"], alibi_scale=float(payload["alibi_scale"]),
        tokenizer_id=REDACTED
        tokenizer_json_sha256=REDACTED
        capabilities=copy.deepcopy(capabilities),
        nat_mask_token_id=REDACTED
        sat_variable_stride2_threshold=serving_contract[
            "sat_variable_stride2_threshold"
        ],
        nat_refinement_steps=serving_contract["nat_refinement_steps"],
        nat_refinement_ratio_decay=serving_contract[
            "nat_refinement_ratio_decay"
        ],
    )
    return LoadedModel(
        model, sat_gate, tokenizer, metadata, capabilities,
        nat_mask_token_id,
        serving_contract["sat_variable_stride2_threshold"],
        serving_contract["nat_refinement_steps"],
        serving_contract["nat_refinement_ratio_decay"],
        target_device,
        target_dtype,
    )


def _penalize(
    logits: torch.Tensor,
    history: torch.Tensor,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    penalty_last_n: int,
) -> torch.Tensor:
    if penalty_last_n > 0:
        history = history[:, -penalty_last_n:]
    counts = torch.bincount(history.reshape(-1), minlength=logits.size(-1)).to(logits.dtype)
    seen = counts > 0
    if repetition_penalty != 1.0:
        values = logits[..., seen]
        logits[..., seen] = torch.where(
            values > 0, values / repetition_penalty, values * repetition_penalty
        )
    if presence_penalty:
        logits = logits - seen.to(logits.dtype) * presence_penalty
    if frequency_penalty:
        logits = logits - counts * frequency_penalty
    return logits


def _sample(logits: torch.Tensor, *, greedy: bool, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    if greedy:
        return logits.argmax(dim=-1, keepdim=True)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive unless --greedy is used")
    logits = logits / temperature
    if top_k > 0:
        threshold = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p < 1.0:
        if not (0.0 < top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        remove = probabilities.cumsum(dim=-1) > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_indices, sorted_logits)
    return torch.multinomial(torch.softmax(logits, dim=-1), 1)


@torch.inference_mode()
def generate(
    loaded: LoadedModel,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    max_context: int = 1536,
    greedy: bool = False,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.3,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.3,
    penalty_last_n: int = 128,
    min_new_tokens: int = 0,
    add_special_tokens: bool = True,
) -> tuple[list[int], list[int]]:
    if max_new_tokens < 0 or min_new_tokens < 0 or min_new_tokens > max_new_tokens:
        raise ValueError("invalid new-token bounds")
    prompt_ids = loaded.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    if not prompt_ids:
        prompt_ids = [loaded.tokenizer.eos_token_id]
    if any(token < 0 or token >= VOCAB_SIZE for token in prompt_ids):
        raise ValueError("prompt tokenizer produced an out-of-range ID")
    if len(prompt_ids) + max_new_tokens > max_context:
        raise ValueError(
            f"prompt ({len(prompt_ids)}) + requested output ({max_new_tokens}) exceeds context {max_context}"
        )
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=loaded.device)
    logits, cache = loaded.model(ids, use_cache=True)
    generated: list[int] = []
    for index in range(max_new_tokens):
        next_logits = logits[:, -1].float().clone()
        next_logits = _penalize(
            next_logits, ids, repetition_penalty, presence_penalty,
            frequency_penalty, penalty_last_n,
        )
        if index < min_new_tokens:
            next_logits[..., loaded.tokenizer.eos_token_id] = float("-inf")
        token = _sample(
            next_logits, greedy=greedy, temperature=temperature,
            top_k=top_k, top_p=top_p,
        )
        value = int(token.item())
        generated.append(value)
        ids = torch.cat((ids, token), dim=1)
        if value == loaded.tokenizer.eos_token_id and index >= min_new_tokens:
            break
        logits, cache = loaded.model(token, past_key_values=cache, use_cache=True)
    return prompt_ids, generated


@dataclass
class GenerationResult:
    prompt_ids: list[int]
    generated_ids: list[int]
    mode: str
    diagnostics: dict[str, Any]


def _encode_prompt(loaded: LoadedModel, prompt: str, add_special_tokens: bool) -> list[int]:
    prompt_ids = loaded.tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
    if not prompt_ids:
        prompt_ids = [loaded.tokenizer.eos_token_id]
    if any(token < 0 or token >= VOCAB_SIZE for token in prompt_ids):
        raise ValueError("prompt tokenizer produced an out-of-range ID")
    return prompt_ids


def _pick_mode_token(
    loaded: LoadedModel,
    logits: torch.Tensor,
    history: torch.Tensor,
    generated_count: int,
    *,
    min_new_tokens: int,
    greedy: bool,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    presence_penalty: float,
    frequency_penalty: float,
    penalty_last_n: int,
) -> torch.Tensor:
    logits = logits.float().clone()
    logits[..., loaded.nat_mask_token_id] = float("-inf")
    logits = _penalize(
        logits, history, repetition_penalty, presence_penalty,
        frequency_penalty, penalty_last_n,
    )
    if generated_count < min_new_tokens:
        logits[..., loaded.tokenizer.eos_token_id] = float("-inf")
    return _sample(
        logits, greedy=greedy, temperature=temperature, top_k=top_k, top_p=top_p,
    )


def _sat_gate_logits(sat_gate: nn.Linear, features: torch.Tensor) -> torch.Tensor:
    """Run the SAT gate in its parameter dtype/device, returning FP32 logits."""
    weight = sat_gate.weight
    gate_input = features.to(device=weight.device, dtype=weight.dtype)
    return sat_gate(gate_input).float()


def _nat_next_unresolved_count(current_count: int) -> int:
    """Apply the v26 NAT trailing-suffix refinement schedule exactly."""
    if type(current_count) is not int or current_count <= 0:
        raise ValueError("NAT unresolved count must be a positive integer")
    return max(
        1,
        int(math.ceil(current_count * NAT_REFINEMENT_RATIO_DECAY)),
    )


@torch.inference_mode()
def generate_sat(
    loaded: LoadedModel,
    prompt: str,
    *,
    variable: bool,
    max_new_tokens: int = 32,
    min_new_tokens: int = 0,
    max_context: int = 1536,
    gate_threshold: Optional[float] = None,
    greedy: bool = False,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.3,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.3,
    penalty_last_n: int = 128,
    add_special_tokens: bool = True,
) -> GenerationResult:
    """Run native shift-two SAT with fixed or learned variable stride.

    SAT pairing is prompt-relative: phase is ``len(prompt_ids) % 2``, so the
    prompt's final two tokens are one SAT context pair without emitting an AR
    alignment token. A one-token prompt is left-padded with the checkpoint BOS
    token so short prompts such as ``Hi`` still form a complete, non-visible
    SAT context pair. Fixed SAT commits two tokens on every ordinary decode step.
    A one-token terminal step is allowed only when EOS is the first token of a
    pair or an odd ``max_new_tokens`` request leaves one slot.
    """
    if max_new_tokens < 0 or min_new_tokens < 0 or min_new_tokens > max_new_tokens:
        raise ValueError("invalid new-token bounds")
    bound_gate_threshold = float(loaded.sat_variable_stride2_threshold)
    if gate_threshold is None:
        gate_threshold = bound_gate_threshold
    elif (
        isinstance(gate_threshold, bool)
        or not isinstance(gate_threshold, (int, float))
        or not math.isclose(
            float(gate_threshold), bound_gate_threshold, rel_tol=0.0, abs_tol=1e-12
        )
    ):
        raise CheckpointError(
            "SAT gate threshold must match the checkpoint export metadata"
        )
    gate_threshold = float(gate_threshold)
    prompt_ids = _encode_prompt(loaded, prompt, add_special_tokens)
    synthetic_prompt_bos_tokens = 0
    if len(prompt_ids) < 2:
        # The frozen DeepSeek tokenizer does not auto-insert BOS. SAT needs a
        # complete two-token context pair, so add BOS as model-only context
        # rather than rejecting perfectly valid one-token user prompts.
        bos_token_id = loaded.tokenizer.backend.token_to_id("<｜begin▁of▁sentence｜>")
        if type(bos_token_id) is not int or not (0 <= bos_token_id < VOCAB_SIZE):
            raise CheckpointError("checkpoint tokenizer has no valid BOS token for SAT bootstrap")
        prompt_ids = [int(bos_token_id), *prompt_ids]
        synthetic_prompt_bos_tokens = 1
    if len(prompt_ids) + max_new_tokens > max_context:
        raise ValueError("SAT prompt plus aligned output exceeds context")
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=loaded.device)
    generated: list[int] = []
    core_forwards = 0
    prompt_alignment = 0
    stride1 = 0
    stride2 = 0
    gate_probabilities: list[float] = []
    sat_phase = int(ids.size(1) % 2)
    stop_reason: Optional[str] = None
    terminal_eos_partial_pair = 0
    request_boundary_partial_pair = 0
    request_boundary_realignment_omitted = 0
    terminal_realign_skipped = 0
    sat_tokens_committed = 0
    ar_realign_tokens = 0

    hidden_buffer: Optional[torch.Tensor] = None
    cache: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None
    if stop_reason is None and len(generated) < max_new_tokens:
        hidden, cache = loaded.model.hidden(
            ids, attention_mode="sat", use_cache=True, sat_block=2,
            sat_phase=sat_phase,
        )
        core_forwards += 1
        hidden_buffer = hidden[:, -2:]
    while stop_reason is None and len(generated) < max_new_tokens:
        assert hidden_buffer is not None and cache is not None
        pair_logits = F.linear(hidden_buffer, loaded.model.emb.weight, None).float()
        stride = 2
        if variable:
            gate_logits = _sat_gate_logits(loaded.sat_gate, hidden_buffer[:, 0])
            probability = float(gate_logits.softmax(dim=-1)[0, 1].item())
            gate_probabilities.append(probability)
            stride = 2 if probability >= gate_threshold else 1
        if stride == 2:
            stride2 += 1
            new_tokens: list[torch.Tensor] = []
            emit_count = min(2, max_new_tokens - len(generated))
            if emit_count == 1:
                request_boundary_partial_pair += 1
            for index in range(emit_count):
                token = REDACTED
                    loaded, pair_logits[:, index], ids, len(generated),
                    min_new_tokens=min_new_tokens, greedy=greedy,
                    temperature=temperature, top_k=top_k, top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    presence_penalty=presence_penalty,
                    frequency_penalty=frequency_penalty,
                    penalty_last_n=penalty_last_n,
                )
                new_tokens.append(token)
                ids = torch.cat((ids, token), dim=1)
                value = int(token.item())
                generated.append(value)
                sat_tokens_committed += 1
                if value == loaded.tokenizer.eos_token_id:
                    stop_reason = "eos"
                    if index == 0 and emit_count == 2:
                        terminal_eos_partial_pair += 1
                    break
            if stop_reason is not None or len(generated) >= max_new_tokens:
                if stop_reason is None:
                    stop_reason = "request_boundary"
                break
            if len(new_tokens) != 2:
                raise RuntimeError("continuing fixed SAT requires a complete pair")
            pair = torch.cat(new_tokens, dim=1)
            hidden_new, cache = loaded.model.hidden(
                pair, past_key_values=cache, use_cache=True,
                attention_mode="sat", sat_block=2, sat_phase=sat_phase,
            )
            core_forwards += 1
            hidden_buffer = hidden_new[:, -2:]
        else:
            stride1 += 1
            sat_token = REDACTED
                loaded, pair_logits[:, 0], ids, len(generated),
                min_new_tokens=min_new_tokens, greedy=greedy,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                penalty_last_n=penalty_last_n,
            )
            ids = torch.cat((ids, sat_token), dim=1)
            sat_value = int(sat_token.item())
            generated.append(sat_value)
            sat_tokens_committed += 1
            if sat_value == loaded.tokenizer.eos_token_id:
                terminal_realign_skipped += 1
                stop_reason = "eos"
                break
            if len(generated) >= max_new_tokens:
                request_boundary_realignment_omitted += 1
                terminal_realign_skipped += 1
                stop_reason = "request_boundary_after_stride_1"
                break
            singleton_hidden, _ = loaded.model.hidden(
                sat_token, past_key_values=cache, use_cache=True,
                attention_mode="sat", sat_block=2, sat_phase=sat_phase,
            )
            core_forwards += 1
            ar_token = REDACTED
                loaded,
                F.linear(singleton_hidden[:, -1], loaded.model.emb.weight, None),
                ids,
                len(generated),
                min_new_tokens=min_new_tokens, greedy=greedy,
                temperature=temperature, top_k=top_k, top_p=top_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                penalty_last_n=penalty_last_n,
            )
            ids = torch.cat((ids, ar_token), dim=1)
            ar_value = int(ar_token.item())
            generated.append(ar_value)
            ar_realign_tokens += 1
            if ar_value == loaded.tokenizer.eos_token_id:
                stop_reason = "eos_during_stride_1_realignment"
                break
            if len(generated) >= max_new_tokens:
                stop_reason = "request_boundary"
                break
            # The singleton hidden state did not see its future block partner.
            # Recompute the aligned sequence so the next SAT block uses exact
            # bidirectional-within-pair state and cache.
            hidden, cache = loaded.model.hidden(
                ids, attention_mode="sat", use_cache=True, sat_block=2,
                sat_phase=sat_phase,
            )
            core_forwards += 1
            hidden_buffer = hidden[:, -2:]

    decisions = stride1 + stride2
    gate_parameters_zero = bool(
        variable
        and torch.count_nonzero(loaded.sat_gate.weight).item() == 0
        and torch.count_nonzero(loaded.sat_gate.bias).item() == 0
    )
    diagnostics = {
        "core_forwards": core_forwards,
        "prompt_alignment_ar_tokens": prompt_alignment,
        "synthetic_prompt_bos_tokens": synthetic_prompt_bos_tokens,
        "sat_phase": sat_phase,
        "sat_stride_1": stride1,
        "sat_stride_2": stride2,
        "sat_decisions": decisions,
        "gate_threshold": gate_threshold if variable else None,
        "gate_probabilities_stride_2": gate_probabilities,
        "gate_parameters_zero": gate_parameters_zero,
        "gate_collapsed": bool(variable and decisions > 0 and (stride1 == 0 or stride2 == 0)),
        "requested_tokens": max_new_tokens,
        "emitted_tokens": len(generated),
        "stop_reason": stop_reason or "empty_request",
        "terminal_eos_partial_pair": terminal_eos_partial_pair,
        "request_boundary_partial_pair": request_boundary_partial_pair,
        "request_boundary_realignment_omitted": request_boundary_realignment_omitted,
        "terminal_realign_skipped": terminal_realign_skipped,
        "sat_tokens_committed": sat_tokens_committed,
        "ar_realign_tokens": ar_realign_tokens,
    }
    return GenerationResult(
        prompt_ids, generated,
        "sat_variable" if variable else "sat_fixed",
        diagnostics,
    )


@torch.inference_mode()
def generate_nat(
    loaded: LoadedModel,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    max_context: int = 1536,
    passes: Optional[int] = None,
    greedy: bool = True,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 1.0,
    add_special_tokens: bool = True,
) -> GenerationResult:
    """Run v26 NAT with a clean EOS boundary and 0.5 suffix refinement.

    The model input is ``prompt + masked suffix + EOS``. The terminal EOS is
    never masked, predicted, remasked, or returned as an extra generated token.
    Each non-final pass retains ``ceil(current_unresolved * 0.5)`` positions as
    one contiguous trailing unresolved suffix, matching training/evaluation.
    """
    bound_refinement_steps = int(loaded.nat_refinement_steps)
    if passes is None:
        passes = REDACTED
    elif type(passes) is not int or int(passes) != bound_refinement_steps:
        raise CheckpointError(
            "NAT refinement passes must match the checkpoint export metadata"
        )
    passes = int(passes)
    if max_new_tokens < 0 or passes <= 0:
        raise ValueError("invalid NAT token/pass count")
    prompt_ids = _encode_prompt(loaded, prompt, add_special_tokens)
    refinement_ratio_decay = float(loaded.nat_refinement_ratio_decay)
    if not math.isclose(
        refinement_ratio_decay,
        NAT_REFINEMENT_RATIO_DECAY,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise CheckpointError("NAT runtime refinement decay must be exactly 0.5")
    if len(prompt_ids) + max_new_tokens + 1 > max_context:
        raise ValueError("NAT prompt plus output and terminal EOS exceed context")
    if max_new_tokens == 0:
        return GenerationResult(
            prompt_ids,
            [],
            "nat",
            {
                "passes": 0,
                "core_forwards": 0,
                "pass_rows": [],
                "refinement_ratio_decay": refinement_ratio_decay,
                "terminal_eos_boundary": True,
                "terminal_eos_token_id": loaded.tokenizer.eos_token_id,
            },
        )
    prompt_len = len(prompt_ids)
    nat_suffix_end = prompt_len + max_new_tokens
    ids = torch.tensor(
        [
            prompt_ids
            + [loaded.nat_mask_token_id] * max_new_tokens
            + [loaded.tokenizer.eos_token_id]
        ],
        dtype=torch.long, device=loaded.device,
    )
    pass_rows: list[dict[str, int]] = []
    core_forwards = 0
    for pass_index in range(passes):
        if int(ids[0, nat_suffix_end].item()) != loaded.tokenizer.eos_token_id:
            raise RuntimeError("NAT terminal EOS boundary was modified")
        suffix = ids[:, prompt_len:nat_suffix_end]
        masked = suffix.reshape(-1).eq(loaded.nat_mask_token_id).nonzero().flatten()
        if masked.numel() == 0:
            break
        unresolved_before = int(masked.numel())
        expected_start = max_new_tokens - unresolved_before
        expected_masked = torch.arange(
            expected_start,
            max_new_tokens,
            dtype=masked.dtype,
            device=masked.device,
        )
        if not torch.equal(masked, expected_masked):
            raise RuntimeError("NAT unresolved positions are not one trailing suffix")
        hidden, _ = loaded.model.hidden(ids, attention_mode="nat", use_cache=False)
        core_forwards += 1
        rows = hidden[:, prompt_len:nat_suffix_end].reshape(
            -1, hidden.size(-1)
        ).index_select(0, masked)
        logits = F.linear(rows, loaded.model.emb.weight, None).float()
        logits[..., loaded.nat_mask_token_id] = float("-inf")
        if greedy:
            picked = logits.argmax(dim=-1)
        else:
            picked = _sample(
                logits, greedy=False, temperature=temperature,
                top_k=top_k, top_p=top_p,
            ).reshape(-1)
        flat = suffix.reshape(-1).clone()
        flat.index_copy_(0, masked, picked.to(flat.dtype))
        ids[:, prompt_len:nat_suffix_end] = flat.view(1, max_new_tokens)
        remasked = 0
        if pass_index + 1 < passes:
            remasked = _nat_next_unresolved_count(unresolved_before)
            # Keep one contiguous unresolved suffix.  The next refinement pass
            # therefore sees the same clean-prefix/masked-suffix shape used by
            # detachable NAT training, rather than arbitrary scattered holes.
            unresolved_start = nat_suffix_end - remasked
            ids[0, unresolved_start:nat_suffix_end] = loaded.nat_mask_token_id
        if int(ids[0, nat_suffix_end].item()) != loaded.tokenizer.eos_token_id:
            raise RuntimeError("NAT terminal EOS boundary was modified")
        pass_rows.append({
            "pass": pass_index + 1,
            "predicted": unresolved_before,
            "remasked": remasked,
            "committed_prefix": max_new_tokens - remasked,
        })
    generated = [
        int(value) for value in ids[0, prompt_len:nat_suffix_end].tolist()
    ]
    if loaded.nat_mask_token_id in generated:
        raise RuntimeError("NAT refinement left unresolved mask tokens")
    if int(ids[0, nat_suffix_end].item()) != loaded.tokenizer.eos_token_id:
        raise RuntimeError("NAT terminal EOS boundary was modified")
    return GenerationResult(
        prompt_ids, generated, "nat",
        {
            "passes": len(pass_rows),
            "core_forwards": core_forwards,
            "pass_rows": pass_rows,
            "refinement_ratio_decay": refinement_ratio_decay,
            "terminal_eos_boundary": True,
            "terminal_eos_token_id": loaded.tokenizer.eos_token_id,
        },
    )


def generate_mode(
    loaded: LoadedModel,
    prompt: str,
    *,
    mode: str,
    max_new_tokens: int = 32,
    min_new_tokens: int = 0,
    max_context: int = 1536,
    nat_passes: Optional[int] = None,
    gate_threshold: Optional[float] = None,
    greedy: bool = False,
    temperature: float = 0.7,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.3,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.3,
    penalty_last_n: int = 128,
    add_special_tokens: bool = True,
    allow_untrained_diagnostic: bool = False,
) -> GenerationResult:
    mode = str(mode).strip().lower().replace("-", "_")
    if mode not in FOUR_MODES:
        raise ValueError(f"mode must be one of {', '.join(FOUR_MODES)}")
    capability = loaded.capabilities["modes"][mode]
    if capability["trained"] is not True and not allow_untrained_diagnostic:
        raise CheckpointError(
            f"{mode} weights are not training-qualified: {capability['status']}; "
            "use allow_untrained_diagnostic=True only for isolated diagnostics"
        )
    if mode == "ar":
        prompt_ids, generated = generate(
            loaded, prompt, max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens, max_context=max_context,
            greedy=greedy, temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            penalty_last_n=penalty_last_n,
            add_special_tokens=REDACTED
        )
        return GenerationResult(
            prompt_ids, generated, "ar",
            {"core_forwards": 1 + len(generated), "tokens_per_commit": 1},
        )
    if mode in {"sat_fixed", "sat_variable"}:
        return generate_sat(
            loaded, prompt, variable=mode == "sat_variable",
            max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
            max_context=max_context, gate_threshold=gate_threshold,
            greedy=greedy, temperature=temperature, top_k=top_k, top_p=top_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            penalty_last_n=penalty_last_n,
            add_special_tokens=REDACTED
        )
    return generate_nat(
        loaded, prompt, max_new_tokens=max_new_tokens,
        max_context=max_context, passes=nat_passes,
        greedy=greedy, temperature=temperature, top_k=top_k, top_p=top_p,
        add_special_tokens=REDACTED
    )


class _StreamedVocabCE(torch.autograd.Function):
    """Exact mean CE with only ``[rows, vocab_chunk]`` transient logits.

    The backward pass recomputes each vocabulary chunk, matching the reviewed
    mature-v25 fused-CE contract.  This avoids retaining a full
    ``[batch * sequence, 129280]`` activation in detachable training.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        targets: torch.Tensor,
        vocab_chunk: int,
    ) -> torch.Tensor:
        hidden_float = hidden.float()
        weight_float = weight.float()
        rows = hidden.size(0)
        vocab = weight.size(0)
        maximum = torch.full(
            (rows,), -float("inf"), device=hidden.device, dtype=torch.float32
        )
        normalizer = torch.zeros(
            rows, device=hidden.device, dtype=torch.float32
        )
        target_logits = torch.empty(
            rows, device=hidden.device, dtype=torch.float32
        )
        for start in range(0, vocab, vocab_chunk):
            stop = min(start + vocab_chunk, vocab)
            logits = hidden_float @ weight_float[start:stop].T
            chunk_maximum = logits.max(dim=1).values
            next_maximum = torch.maximum(maximum, chunk_maximum)
            normalizer = (
                normalizer * torch.exp(maximum - next_maximum)
                + torch.exp(logits - next_maximum[:, None]).sum(dim=1)
            )
            maximum = next_maximum
            selected = targets.ge(start) & targets.lt(stop)
            if bool(selected.any()):
                target_logits[selected] = logits[
                    selected, targets[selected] - start
                ]
        logsumexp = maximum + torch.log(normalizer)
        log_probability = target_logits - logsumexp
        ctx.save_for_backward(hidden, weight, targets, logsumexp)
        ctx.vocab_chunk = int(vocab_chunk)
        return (-log_probability).mean()

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:  # type: ignore[override]
        hidden, weight, targets, logsumexp = ctx.saved_tensors
        vocab_chunk = int(ctx.vocab_chunk)
        hidden_float = hidden.float()
        weight_float = weight.float()
        rows = hidden.size(0)
        vocab = weight.size(0)
        hidden_gradient = torch.zeros_like(hidden_float)
        weight_gradient = torch.zeros_like(weight_float)
        scale = grad_output.float() / float(rows)
        for start in range(0, vocab, vocab_chunk):
            stop = min(start + vocab_chunk, vocab)
            weight_chunk = weight_float[start:stop]
            probabilities = torch.exp(
                hidden_float @ weight_chunk.T - logsumexp[:, None]
            )
            selected = targets.ge(start) & targets.lt(stop)
            if bool(selected.any()):
                probabilities[selected, targets[selected] - start] -= 1.0
            probabilities = probabilities * scale
            hidden_gradient.add_(probabilities @ weight_chunk)
            weight_gradient[start:stop].add_(probabilities.T @ hidden_float)
        return (
            hidden_gradient.to(hidden.dtype),
            weight_gradient.to(weight.dtype),
            None,
            None,
        )


def streamed_vocab_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    *,
    vocab_chunk: int = 16_384,
) -> torch.Tensor:
    """Exact streamed mean cross-entropy over a tied vocabulary matrix."""
    if hidden.ndim < 2 or weight.ndim != 2:
        raise ValueError("hidden and vocabulary weight must have rank >=2 and 2")
    if hidden.size(-1) != weight.size(1):
        raise ValueError("hidden width and vocabulary projection width differ")
    if targets.numel() != hidden.numel() // hidden.size(-1):
        raise ValueError("one target ID is required per hidden row")
    if targets.dtype not in {torch.int32, torch.int64}:
        raise ValueError("streamed CE targets must be integer IDs")
    if int(vocab_chunk) <= 0:
        raise ValueError("vocab_chunk must be positive")
    flattened_targets = targets.reshape(-1).long()
    if flattened_targets.numel() == 0:
        raise ValueError("streamed CE requires at least one target")
    if int(flattened_targets.min().item()) < 0 or int(flattened_targets.max().item()) >= weight.size(0):
        raise ValueError("streamed CE target outside vocabulary")
    return _StreamedVocabCE.apply(
        hidden.reshape(-1, hidden.size(-1)),
        weight,
        flattened_targets,
        int(vocab_chunk),
    )


@torch.no_grad()
def streamed_vocab_argmax(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    vocab_chunk: int = 16_384,
) -> torch.Tensor:
    """Argmax a tied vocabulary projection without materializing full logits."""
    if hidden.ndim < 2 or weight.ndim != 2 or hidden.size(-1) != weight.size(1):
        raise ValueError("malformed streamed argmax tensors")
    if int(vocab_chunk) <= 0:
        raise ValueError("vocab_chunk must be positive")
    original_shape = hidden.shape[:-1]
    rows = hidden.reshape(-1, hidden.size(-1)).float()
    weight_float = weight.float()
    best_values = torch.full(
        (rows.size(0),), -float("inf"), device=rows.device, dtype=torch.float32
    )
    best_ids = torch.zeros(rows.size(0), device=rows.device, dtype=torch.long)
    for start in range(0, weight.size(0), int(vocab_chunk)):
        stop = min(start + int(vocab_chunk), weight.size(0))
        values, local_ids = (rows @ weight_float[start:stop].T).max(dim=1)
        better = values > best_values
        best_values = torch.where(better, values, best_values)
        best_ids = torch.where(better, local_ids + start, best_ids)
    return best_ids.view(original_shape)


def multimode_training_loss(
    loaded: LoadedModel,
    ids: torch.Tensor,
    *,
    mode: str,
    nat_mask_ratio: float = 0.5,
    nat_suffix_start: Optional[int] = None,
    sat_phase: Optional[int] = None,
    sat_gate_weight: float = 0.1,
    vocab_chunk: int = 16_384,
    generator: Optional[torch.Generator] = None,
) -> dict[str, Any]:
    """Small, explicit objective surface for continued four-mode training.

    This function does not own an optimizer or mutate the mature model.  Callers
    must optimize ``loaded.model.parameters()`` plus ``loaded.sat_gate.parameters()``
    and persist committed-update counters in the export training receipt.
    """
    mode = str(mode).strip().lower().replace("-", "_")
    if mode not in FOUR_MODES:
        raise ValueError(f"unknown training mode {mode!r}")
    if ids.ndim != 2 or ids.dtype not in {torch.int32, torch.int64}:
        raise ValueError("training ids must be a rank-2 integer tensor")
    if ids.size(1) < (4 if mode.startswith("sat_") else 2):
        raise ValueError("sequence is too short for selected objective")
    ids = ids.to(loaded.device)
    weight = loaded.model.emb.weight

    if mode == "ar":
        hidden, _ = loaded.model.hidden(ids, attention_mode="causal")
        loss = streamed_vocab_cross_entropy(
            hidden[:, :-1], weight, ids[:, 1:], vocab_chunk=vocab_chunk
        )
        return {"mode": mode, "loss": loss, "tokens": int(ids[:, 1:].numel())}

    if mode in {"sat_fixed", "sat_variable"}:
        phase = (
            int(torch.randint(0, 2, (1,), generator=generator).item())
            if sat_phase is None else int(sat_phase)
        )
        if phase not in {0, 1}:
            raise ValueError("SAT training phase must be 0 or 1")
        hidden, _ = loaded.model.hidden(
            ids, attention_mode="sat", sat_block=2, sat_phase=phase
        )
        context = hidden[:, :-2]
        targets = ids[:, 2:]
        if phase == 1:
            # Position zero is the prompt-relative singleton and is not a SAT
            # shift-two training row.
            context = context[:, 1:]
            targets = targets[:, 1:]
        paired_rows = (context.size(1) // 2) * 2
        if paired_rows <= 0:
            raise ValueError("no complete prompt-relative SAT training pair")
        context = context[:, :paired_rows]
        targets = targets[:, :paired_rows]
        token_loss = REDACTED
            context, weight, targets, vocab_chunk=vocab_chunk
        )
        result: dict[str, Any] = {
            "mode": mode, "loss": token_loss, "token_loss": token_loss,
            "tokens": int(targets.numel()), "sat_phase": phase,
        }
        if mode == "sat_variable":
            first = context[:, ::2]
            second = context[:, 1::2]
            gold_second = targets[:, 1::2]
            blocks = min(first.size(1), second.size(1), gold_second.size(1))
            if blocks <= 0:
                raise ValueError("no complete SAT-variable training blocks")
            first = first[:, :blocks]
            second = second[:, :blocks]
            gold_second = gold_second[:, :blocks]
            with torch.no_grad():
                predicted_second = streamed_vocab_argmax(
                    second, weight, vocab_chunk=vocab_chunk
                )
                gate_targets = predicted_second.eq(gold_second).long()
            gate_logits = loaded.sat_gate(first.detach())
            gate_loss = F.cross_entropy(
                gate_logits.reshape(-1, 2), gate_targets.reshape(-1)
            )
            loss = token_loss + float(sat_gate_weight) * gate_loss
            result.update(
                loss=loss,
                gate_loss=gate_loss,
                gate_blocks=int(gate_targets.numel()),
                gate_stride2_labels=int(gate_targets.sum().item()),
            )
        return result

    if not (0.0 < nat_mask_ratio <= 1.0):
        raise ValueError("nat_mask_ratio must be in (0, 1]")
    sequence = ids.size(1)
    suffix_start = int(sequence // 2 if nat_suffix_start is None else nat_suffix_start)
    if not (1 <= suffix_start < sequence):
        raise ValueError("NAT requires a non-empty clean prefix and masked suffix")
    work = ids.clone()
    suffix = ids[:, suffix_start:]
    eligible = suffix.ne(loaded.nat_mask_token_id)
    random_rows = torch.rand(
        suffix.shape, device=ids.device, generator=generator
    )
    suffix_mask = random_rows.lt(float(nat_mask_ratio)) & eligible
    for row in range(ids.size(0)):
        if not bool(suffix_mask[row].any()):
            candidates = eligible[row].nonzero().flatten()
            if candidates.numel() == 0:
                raise ValueError("NAT suffix has no clean target token")
            suffix_mask[row, int(candidates[0].item())] = True
    mask = torch.zeros_like(ids, dtype=torch.bool)
    mask[:, suffix_start:] = suffix_mask
    original = ids[mask]
    work[mask] = loaded.nat_mask_token_id
    hidden, _ = loaded.model.hidden(work, attention_mode="nat")
    loss = streamed_vocab_cross_entropy(
        hidden[mask], weight, original, vocab_chunk=vocab_chunk
    )
    return {
        "mode": mode,
        "loss": loss,
        "tokens": int(original.numel()),
        "clean_prefix_tokens_per_row": suffix_start,
        "suffix_tokens_per_row": sequence - suffix_start,
        "masked_suffix_positions": int(mask.sum().item()),
    }


def sat_variable_gate_census(
    loaded: LoadedModel,
    feature_batches: Sequence[torch.Tensor],
    *,
    threshold: float = 0.95,
    label_batches: Optional[Sequence[torch.Tensor]] = None,
) -> dict[str, Any]:
    probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    if label_batches is not None and len(label_batches) != len(feature_batches):
        raise ValueError("SAT gate feature/label batch counts differ")
    with torch.no_grad():
        for index, features in enumerate(feature_batches):
            if features.ndim != 2 or features.size(-1) != 256:
                raise ValueError("SAT gate census expects [rows, 256] features")
            logits = loaded.sat_gate(features.to(loaded.device, loaded.dtype))
            probabilities.append(logits.float().softmax(dim=-1)[:, 1].cpu())
            if label_batches is not None:
                batch_labels = label_batches[index].reshape(-1).long().cpu()
                if batch_labels.numel() != features.size(0):
                    raise ValueError("SAT gate feature/label row counts differ")
                if batch_labels.numel() and (
                    int(batch_labels.min().item()) < 0
                    or int(batch_labels.max().item()) > 1
                ):
                    raise ValueError("SAT gate labels must be 0 or 1")
                labels.append(batch_labels)
    values = torch.cat(probabilities) if probabilities else torch.empty(0)
    predictions = values.ge(float(threshold))
    stride2 = int(predictions.sum().item())
    rows = int(predictions.numel())
    parameter_zero = bool(
        torch.count_nonzero(loaded.sat_gate.weight).item() == 0
        and torch.count_nonzero(loaded.sat_gate.bias).item() == 0
    )
    report: dict[str, Any] = {
        "rows": rows,
        "threshold": float(threshold),
        "stride1": rows - stride2,
        "stride2": stride2,
        "stride2_rate": (float(stride2) / rows if rows else 0.0),
        "parameter_zero": parameter_zero,
        "collapsed": bool(parameter_zero or (rows > 1 and stride2 in {0, rows})),
        "mean_stride2_probability": (float(values.mean().item()) if rows else None),
    }
    if label_batches is not None:
        truth = torch.cat(labels) if labels else torch.empty(0, dtype=torch.long)
        if truth.numel() != rows:
            raise ValueError("SAT gate census labels do not cover every feature row")
        recalls: list[float] = []
        for label in (0, 1):
            selected = truth.eq(label)
            if bool(selected.any()):
                recalls.append(float(predictions[selected].eq(bool(label)).float().mean().item()))
        report["balanced_accuracy"] = (
            sum(recalls) / len(recalls) if len(recalls) == 2 else None
        )
        report["label_stride1"] = int(truth.eq(0).sum().item())
        report["label_stride2"] = int(truth.eq(1).sum().item())
        if rows:
            truth_float = truth.float()
            report["brier"] = float(
                (values - truth_float).square().mean().item()
            )
            calibration_error = 0.0
            for bin_index in range(10):
                lower = float(bin_index) / 10.0
                upper = float(bin_index + 1) / 10.0
                in_bin = values.ge(lower) & (
                    values.le(upper) if bin_index == 9 else values.lt(upper)
                )
                if bool(in_bin.any()):
                    calibration_error += float(in_bin.float().mean().item()) * abs(
                        float(values[in_bin].mean().item())
                        - float(truth_float[in_bin].mean().item())
                    )
            report["ece"] = calibration_error
            negative = truth.eq(0)
            report["false_accept_rate"] = (
                float(predictions[negative].float().mean().item())
                if bool(negative.any()) else None
            )
        else:
            report.update(brier=None, ece=None, false_accept_rate=None)
    return report


def sat_throughput_contract(stride1: int, stride2: int) -> dict[str, Any]:
    """Verify model-forward efficiency AR < SAT-variable < SAT-fixed.

    Rates are tokens committed per post-prefill core forward.  A stride-1 SAT
    decision needs one singleton forward plus one aligned refresh for its AR
    partner; stride-2 needs one pair forward.
    """
    stride1, stride2 = int(stride1), int(stride2)
    if stride1 <= 0 or stride2 <= 0:
        return {
            "ok": False,
            "reason": "SAT-variable gate must exercise both stride classes",
            "metric": "post_prefill_core_forward_efficiency_not_wall_clock",
            "ar_tokens_per_forward": 1.0,
            "sat_variable_tokens_per_forward": None,
            "sat_fixed_tokens_per_forward": 2.0,
        }
    variable_tokens = 2 * (stride1 + stride2)
    variable_forwards = 2 * stride1 + stride2
    variable_rate = variable_tokens / variable_forwards
    return {
        "ok": bool(1.0 < variable_rate < 2.0),
        "metric": "post_prefill_core_forward_efficiency_not_wall_clock",
        "ar_tokens_per_forward": 1.0,
        "sat_variable_tokens_per_forward": variable_rate,
        "sat_fixed_tokens_per_forward": 2.0,
        "stride1": stride1,
        "stride2": stride2,
    }


def measured_sat_speed_order(
    *,
    ar_tokens_per_second: float,
    sat_variable_tokens_per_second: float,
    sat_fixed_tokens_per_second: float,
    capabilities: dict[str, Any],
    gate_census: dict[str, Any],
) -> dict[str, Any]:
    """Qualify the measured AR < SAT-variable < SAT-fixed wall-clock claim.

    Current legacy weights deliberately return ``eligible=False`` because their
    SAT modes are untrained and their zero gate is collapsed.  A future export
    is held to the ordering only after both SAT capabilities are receipt-qualified
    and the measured gate census exercises both decisions.
    """
    rates = {
        "ar": float(ar_tokens_per_second),
        "sat_variable": float(sat_variable_tokens_per_second),
        "sat_fixed": float(sat_fixed_tokens_per_second),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in rates.values()):
        raise ValueError("measured token rates must be finite and positive")
    modes = validate_capabilities(capabilities)["modes"]
    reasons: list[str] = []
    if modes["sat_fixed"]["trained"] is not True:
        reasons.append("sat_fixed_not_training_qualified")
    if modes["sat_variable"]["trained"] is not True:
        reasons.append("sat_variable_not_training_qualified")
    if gate_census.get("collapsed") is not False:
        reasons.append("sat_variable_gate_collapsed_or_unverified")
    if int(gate_census.get("stride1", 0) or 0) <= 0 or int(
        gate_census.get("stride2", 0) or 0
    ) <= 0:
        reasons.append("both_stride_classes_not_observed")
    observed = bool(
        rates["ar"] < rates["sat_variable"] < rates["sat_fixed"]
    )
    eligible = not reasons
    return {
        "metric": "measured_wall_clock_tokens_per_second",
        "rates": rates,
        "observed_order": observed,
        "eligible": eligible,
        "passed": observed if eligible else None,
        "reasons": reasons,
    }


def decode_telemetry_payload(
    result: GenerationResult,
    capability: dict[str, Any],
    *,
    requested_tokens: int,
    elapsed_seconds: float,
    diagnostic_override: bool,
) -> dict[str, Any]:
    """Flatten mode diagnostics into the bridge's explicit proof contract."""
    elapsed_seconds = float(elapsed_seconds)
    if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
        raise ValueError("decode elapsed time must be finite and non-negative")
    diagnostics = result.diagnostics
    emitted = len(result.generated_ids)
    core_forwards = int(diagnostics.get("core_forwards", 0) or 0)
    if result.mode == "ar":
        decode_steps = emitted
        gate_decisions = gate_commit_1 = gate_commit_2 = 0
        sat_tokens = ar_realign = prompt_align = refinement_steps = 0
        committed_tokens = emitted
        terminal_boundary_partial = terminal_realign_skipped = False
    elif result.mode in {"sat_fixed", "sat_variable"}:
        gate_decisions = int(diagnostics.get("sat_decisions", 0) or 0)
        gate_commit_1 = int(diagnostics.get("sat_stride_1", 0) or 0)
        gate_commit_2 = int(diagnostics.get("sat_stride_2", 0) or 0)
        prompt_align = int(diagnostics.get("prompt_alignment_ar_tokens", 0) or 0)
        decode_steps = gate_decisions
        sat_tokens = int(diagnostics.get("sat_tokens_committed", 0) or 0)
        committed_tokens = sat_tokens
        ar_realign = int(diagnostics.get("ar_realign_tokens", 0) or 0)
        refinement_steps = 0
        terminal_boundary_partial = bool(
            int(diagnostics.get("terminal_eos_partial_pair", 0) or 0)
            or int(diagnostics.get("request_boundary_partial_pair", 0) or 0)
        )
        terminal_realign_skipped = bool(
            int(diagnostics.get("terminal_realign_skipped", 0) or 0)
        )
    elif result.mode == "nat":
        refinement_steps = int(diagnostics.get("passes", 0) or 0)
        decode_steps = refinement_steps
        gate_decisions = gate_commit_1 = gate_commit_2 = 0
        sat_tokens = ar_realign = prompt_align = 0
        committed_tokens = emitted
        terminal_boundary_partial = terminal_realign_skipped = False
    else:
        raise ValueError(f"unknown telemetry mode {result.mode!r}")
    gate_commit_2_rate = (
        float(gate_commit_2) / float(gate_decisions)
        if gate_decisions > 0 else 0.0
    )
    return {
        "schema": DECODE_TELEMETRY_SCHEMA,
        "mode": result.mode,
        "state_binding": capability["state_binding"],
        "trained": capability["trained"],
        "diagnostic_override": bool(diagnostic_override),
        "requested_tokens": int(requested_tokens),
        "decode_steps": decode_steps,
        "committed_tokens": committed_tokens,
        "gate_decisions": gate_decisions,
        "gate_commit_1": gate_commit_1,
        "gate_commit_2": gate_commit_2,
        "gate_commit_2_rate": gate_commit_2_rate,
        "sat_tokens_committed": sat_tokens,
        "ar_realign_tokens": ar_realign,
        "prompt_align_tokens": prompt_align,
        "total_emitted_tokens": emitted,
        "core_forwards": core_forwards,
        "refinement_steps": refinement_steps,
        "terminal_boundary_partial": terminal_boundary_partial,
        "terminal_realign_skipped": terminal_realign_skipped,
        "elapsed_s": elapsed_seconds,
        "tok_s": emitted / elapsed_seconds if elapsed_seconds > 0.0 else None,
        "diagnostics": diagnostics,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt")
    parser.add_argument(
        "--mode", default="ar", choices=FOUR_MODES,
        help="AR, SAT fixed, SAT variable, or NAT",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--dtype", default="auto", choices=("auto", "float32", "float16", "bfloat16"))
    parser.add_argument("--expected-sha256", default=RELEASE_ARTIFACT_SHA256)
    parser.add_argument("--expected-tokenizer-sha256", default=RELEASE_TOKENIZER_JSON_SHA256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--max-context", type=int, default=1536)
    parser.add_argument(
        "--nat-passes", type=int, default=None,
        help="must equal the export-bound NAT refinement count",
    )
    parser.add_argument(
        "--sat-gate-threshold", type=float, default=None,
        help="must equal the export-bound SAT Variable gate threshold",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.3)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.3)
    parser.add_argument("--penalty-last-n", type=int, default=128)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--no-special-tokens", action="store_true")
    parser.add_argument("--completion-only", action="store_true")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--capabilities-json", action="store_true")
    parser.add_argument(
        "--verify-only", action="store_true",
        help="verify artifact, embedded tokenizer, schema and receipts without inference",
    )
    parser.add_argument(
        "--allow-untrained-diagnostic", action="store_true",
        help="permit SAT/NAT execution before their training receipt qualifies",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    if args.verify_only:
        checkpoint_path = pathlib.Path(args.checkpoint)
        payload = load_export_payload(
            checkpoint_path, expected_sha256=args.expected_sha256,
        )
        tokenizer = REDACTED
            payload, expected_json_sha256=args.expected_tokenizer_sha256,
        )
        schema = payload["agillm5_detachable_50m_export"]["schema"]
        capabilities = (
            validate_capabilities(payload["agillm5_detachable_50m_capabilities"])
            if schema == MULTIMODE_EXPORT_SCHEMA else legacy_capabilities(payload)
        )
        export_metadata = dict(payload["agillm5_detachable_50m_export"])
        training_receipt = payload.get("agillm5_detachable_50m_training_receipt")
        serving_contract = _export_serving_contract(payload)
        print(json.dumps({
            "ok": True,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "artifact_schema": schema,
            "capability_schema": capabilities["schema"],
            "capabilities": capabilities,
            "training_receipt": training_receipt,
            "checkpoint_binding": export_metadata.get("checkpoint_binding"),
            "export_metadata": export_metadata,
            "nat_mask_token_id": serving_contract["nat_mask_token_id"],
            "sat_variable_stride2_threshold": serving_contract[
                "sat_variable_stride2_threshold"
            ],
            "nat_refinement_steps": serving_contract["nat_refinement_steps"],
            "nat_refinement_ratio_decay": serving_contract[
                "nat_refinement_ratio_decay"
            ],
            "tokenizer_id": tokenizer.tokenizer_id,
            "tokenizer_sha256": tokenizer.tokenizer_json_sha256,
            "tokenizer_json_sha256": tokenizer.tokenizer_json_sha256,
            "core_parameters": CORE_PARAMETERS,
            "standalone_parameters": EXPORT_PARAMETERS,
        }, sort_keys=True))
        return 0
    if not isinstance(args.prompt, str):
        raise SystemExit("--prompt is required unless --verify-only is used")
    loaded = load_model(
        args.checkpoint, device=args.device, dtype=args.dtype,
        expected_sha256=args.expected_sha256,
        expected_tokenizer_sha256=REDACTED
    )
    if args.capabilities_json:
        print(json.dumps(loaded.capabilities, sort_keys=True))
    generation_started = time.perf_counter()
    result = generate_mode(
        loaded, args.prompt, mode=args.mode,
        max_new_tokens=args.max_new_tokens, min_new_tokens=args.min_new_tokens,
        max_context=args.max_context, greedy=args.greedy,
        nat_passes=args.nat_passes, gate_threshold=args.sat_gate_threshold,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        penalty_last_n=args.penalty_last_n,
        add_special_tokens=not args.no_special_tokens,
        allow_untrained_diagnostic=args.allow_untrained_diagnostic,
    )
    generation_elapsed = time.perf_counter() - generation_started
    output_ids = (
        result.generated_ids
        if args.completion_only
        else result.prompt_ids + result.generated_ids
    )
    print(loaded.tokenizer.decode(output_ids, skip_special_tokens=True))
    elapsed = time.perf_counter() - started
    capability = loaded.capabilities["modes"][result.mode]
    telemetry = decode_telemetry_payload(
        result,
        capability,
        requested_tokens=REDACTED
        elapsed_seconds=generation_elapsed,
        diagnostic_override=bool(
            args.allow_untrained_diagnostic and capability["trained"] is not True
        ),
    )
    print(
        "[decode-telemetry] " + json.dumps(telemetry, sort_keys=True),
        file=sys.stderr,
    )
    print(
        f"[agillm44] mode={result.mode} device={loaded.device} "
        f"dtype={str(loaded.dtype).replace('torch.', '')} "
        f"generated={len(result.generated_ids)} elapsed={elapsed:.2f}s "
        f"diagnostics={result.diagnostics}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
