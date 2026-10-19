#!/usr/bin/env python3
"""Fail-closed serving contract for detachable AGILLM pop-out checkpoints.

The contract deliberately separates *state present in a checkpoint* from a
mode that was actually trained.  In particular, the legacy 49.1M AR export
contains an inert SAT compatibility gate.  Tensor-name inspection must never
turn that compatibility state into a serving capability.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


CONTRACT_SCHEMA = "agillm.popout-http.v1"
TELEMETRY_SCHEMA = "agillm.decode-telemetry.v1"
LEGACY_AR_SCHEMA = "agillm5.detachable-50m.ar-export.v1"
CAPABILITY_SCHEMA = "agillm5.detachable-50m.capabilities.v1"
MULTIMODE_EXPORT_SCHEMA = "agillm5.detachable-50m.multimode-export.v2"
TRAINING_RECEIPT_SCHEMA = "agillm5.detachable-50m.multimode-training-receipt.v1"
CHECKPOINT_BINDING_SCHEMA = "agillm5.detachable-50m.checkpoint-binding.v1"
EVALUATOR_SCHEMA = "agillm5.detachable-50m.heldout-evaluator.v1"
CANONICAL_MODES = ("ar", "sat_fixed", "sat_variable", "nat")

MIN_TRAINING_COMMITS = 128
MIN_HELDOUT_TOKENS = 4_096
MIN_PROMPT_PARITY_CASES = 32
MIN_GATE_CENSUS_ROWS = 512
MIN_GATE_LABELS_PER_CLASS = 32
MIN_THROUGHPUT_PROMPTS = 32
MIN_THROUGHPUT_TOKENS_PER_MODE = 512
SAT_VARIABLE_STRIDE2_THRESHOLD = 0.95
NAT_MASK_TOKEN_ID = 2
NAT_REFINEMENT_STEPS = 2
CORE_PARAMETERS = 49_096_704
STANDALONE_PARAMETERS = 49_097_218

MODE_OBJECTIVES = {
    "ar": "causal_shift_1",
    "sat_fixed": "sat_block_2_shift_2",
    "sat_variable": "sat_fixed_plus_learned_stride_1_or_2_gate",
    "nat": "masked_bidirectional_denoising",
}

MODE_STATE_BINDINGS = {
    "ar": "core+tied_vocab:causal_shift_1",
    "sat_fixed": "core+tied_vocab:sat_block_2_shift_2",
    "sat_variable": "core+tied_vocab+sat.gate:sat_variable_stride",
    "nat": "core+tied_vocab:nat_bidirectional_masked_suffix",
}

MODE_SEMANTICS = {
    "ar": {
        "commit_policy": "exactly_1_token_per_decode_step",
        "runtime_mode": "ar",
        "runtime_sampler": "ar",
    },
    "sat_fixed": {
        "commit_policy": "exactly_2_sat_tokens_per_gate_decision_except_one_explicit_terminal_boundary_partial",
        "runtime_mode": "sat",
        "runtime_sampler": "sat",
    },
    "sat_variable": {
        "commit_policy": "gate_selects_1_or_2_sat_tokens; stride_1 adds one AR realignment unless terminally stopped",
        "runtime_mode": "sat",
        "runtime_sampler": "sat",
    },
    "nat": {
        "commit_policy": "parallel_masked_refinement",
        "runtime_mode": "nat",
        "runtime_sampler": "nat",
    },
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_RECEIPT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")


class ContractError(ValueError):
    """A client or deployment contract violation with an HTTP mapping."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int = 400,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = int(status)
        self.extra = dict(extra or {})

    def payload(self) -> dict[str, Any]:
        return {
            "ok": False,
            "error": self.message,
            "error_code": self.code,
            **self.extra,
        }


@dataclass(frozen=True)
class ModeDeclaration:
    trained: bool
    state_binding: str | None
    telemetry_contract: str | None
    training_receipt_id: str | None
    reason: str


def _canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractError("invalid_manifest", "manifest trust data must be finite canonical JSON") from exc
    return hashlib.sha256(encoded).hexdigest()


def _exact_json_equal(left: Any, right: Any) -> bool:
    """Type-strict JSON equality (unlike Python, ``True`` is not ``1``)."""

    try:
        return json.dumps(
            left,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ) == json.dumps(
            right,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return False


def _copy_finite_json(value: Any, label: str) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ContractError("invalid_manifest", f"{label} must be finite JSON") from exc


def _legacy_capability_contract() -> dict[str, Any]:
    return {
        "schema": CAPABILITY_SCHEMA,
        "source_schema": LEGACY_AR_SCHEMA,
        "modes": {
            "ar": {
                "inference": True,
                "training": True,
                "trained": True,
                "objective": MODE_OBJECTIVES["ar"],
                "status": "released_ar_student",
                "state_binding": MODE_STATE_BINDINGS["ar"],
                "telemetry_contract": TELEMETRY_SCHEMA,
                "training_receipt_id": "released-ar-export-step-2570067",
            },
            "sat_fixed": {
                "inference": True,
                "training": True,
                "trained": False,
                "objective": MODE_OBJECTIVES["sat_fixed"],
                "status": "executable_shared_weights_requires_sat_training",
                "state_binding": MODE_STATE_BINDINGS["sat_fixed"],
                "telemetry_contract": TELEMETRY_SCHEMA,
            },
            "sat_variable": {
                "inference": True,
                "training": True,
                "trained": False,
                "objective": MODE_OBJECTIVES["sat_variable"],
                "status": "zero_gate_collapsed_to_stride_1_requires_training",
                "state_binding": MODE_STATE_BINDINGS["sat_variable"],
                "telemetry_contract": TELEMETRY_SCHEMA,
            },
            "nat": {
                "inference": True,
                "training": True,
                "trained": False,
                "objective": MODE_OBJECTIVES["nat"],
                "status": "executable_tied_head_requires_nat_training",
                "state_binding": MODE_STATE_BINDINGS["nat"],
                "telemetry_contract": TELEMETRY_SCHEMA,
            },
        },
    }


def _validate_export_metadata(
    raw: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
) -> None:
    if not isinstance(raw, Mapping) or raw.get("schema") != MULTIMODE_EXPORT_SCHEMA:
        raise ContractError("invalid_manifest", "v2 manifest requires exact multimode export_metadata")
    if not _exact_json_equal(raw.get("checkpoint_binding"), checkpoint_binding):
        raise ContractError(
            "invalid_manifest",
            "export_metadata.checkpoint_binding must exactly match the training receipt binding",
        )
    if raw.get("source_manifest_sha256") != checkpoint_binding.get("source_manifest_sha256"):
        raise ContractError("invalid_manifest", "export metadata/source manifest binding mismatch")
    if raw.get("export_state_sha256") != checkpoint_binding.get("export_state_sha256"):
        raise ContractError("invalid_manifest", "export metadata/export-state binding mismatch")
    if raw.get("supported_modes") != list(CANONICAL_MODES):
        raise ContractError("invalid_manifest", "export metadata must preserve the exact four-mode order")
    if not isinstance(raw.get("intent_id"), str) or not raw["intent_id"].strip():
        raise ContractError("invalid_manifest", "export_metadata.intent_id is required")
    exact_integer_settings = {
        "core_ar_unique_parameters": CORE_PARAMETERS,
        "unique_parameters": STANDALONE_PARAMETERS,
        "maximum_parameters": 50_000_000,
        "sat_fixed_block": 2,
        "sat_variable_block": 2,
        "nat_mask_token_id": NAT_MASK_TOKEN_ID,
        "nat_refinement_steps": NAT_REFINEMENT_STEPS,
    }
    for key, expected in exact_integer_settings.items():
        if type(raw.get(key)) is not int or raw[key] != expected:
            raise ContractError("invalid_manifest", f"export_metadata.{key} must equal {expected}")
    threshold = raw.get("sat_variable_stride2_threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not math.isfinite(float(threshold))
        or not math.isclose(
            float(threshold),
            SAT_VARIABLE_STRIDE2_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ContractError(
            "invalid_manifest",
            f"export_metadata.sat_variable_stride2_threshold must equal {SAT_VARIABLE_STRIDE2_THRESHOLD}",
        )
    for key in (
        "tied_biasless_vocabulary",
        "standalone",
        "sat_prompt_relative_pairing",
        "trained_sat_variable_gate",
        "training_receipt_present",
    ):
        if raw.get(key) is not True:
            raise ContractError("invalid_manifest", f"export_metadata.{key} must be true")


def _validate_training_receipt(
    raw: Any,
    modes: Mapping[str, ModeDeclaration],
    export_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or raw.get("schema") != TRAINING_RECEIPT_SCHEMA:
        raise ContractError(
            "invalid_training_receipt",
            f"multimode export requires training receipt schema {TRAINING_RECEIPT_SCHEMA}",
        )

    try:
        receipt = _copy_finite_json(raw, "training receipt")
    except (TypeError, ValueError) as exc:
        raise ContractError(
            "invalid_training_receipt",
            "training receipt must be finite canonical JSON",
        ) from exc

    receipt_digest = receipt.get("receipt_sha256")
    unsigned_receipt = dict(receipt)
    unsigned_receipt.pop("receipt_sha256", None)
    if (
        not isinstance(receipt_digest, str)
        or not _SHA256_RE.fullmatch(receipt_digest)
        or receipt_digest != _canonical_json_sha256(unsigned_receipt)
    ):
        raise ContractError(
            "invalid_training_receipt",
            "training receipt has no valid canonical receipt_sha256 binding",
        )

    def positive_int(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ContractError("invalid_training_receipt", f"{label} must be a positive integer")
        return value

    def nonnegative_int(value: Any, label: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError("invalid_training_receipt", f"{label} must be a non-negative integer")
        return value

    def finite_number(value: Any, label: str, *, minimum: float = 0.0, maximum: float | None = None) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ContractError("invalid_training_receipt", f"{label} must be finite numeric")
        parsed = float(value)
        if parsed < minimum or (maximum is not None and parsed > maximum):
            raise ContractError("invalid_training_receipt", f"{label} is outside its valid range")
        return parsed

    binding = receipt.get("checkpoint_binding")
    required_binding_keys = {
        "schema",
        "source_checkpoint_sha256",
        "source_manifest_sha256",
        "model_state_sha256",
        "sat_variable_gate_state_sha256",
        "export_state_sha256",
        "parent_step",
        "student_popup_step",
    }
    if not isinstance(binding, Mapping) or set(binding) != required_binding_keys:
        raise ContractError(
            "invalid_training_receipt",
            "checkpoint_binding must contain the exact full v1 binding fields",
        )
    if binding.get("schema") != CHECKPOINT_BINDING_SCHEMA:
        raise ContractError(
            "invalid_training_receipt",
            f"checkpoint_binding.schema must be {CHECKPOINT_BINDING_SCHEMA}",
        )
    for key in (
        "source_checkpoint_sha256",
        "source_manifest_sha256",
        "model_state_sha256",
        "sat_variable_gate_state_sha256",
        "export_state_sha256",
    ):
        if not isinstance(binding.get(key), str) or not _SHA256_RE.fullmatch(binding[key]):
            raise ContractError("invalid_training_receipt", f"checkpoint_binding.{key} is invalid")
    nonnegative_int(binding.get("parent_step"), "checkpoint_binding.parent_step")
    positive_int(binding.get("student_popup_step"), "checkpoint_binding.student_popup_step")

    evaluator = receipt.get("evaluator_contract")
    required_evaluator_keys = {
        "schema",
        "evaluator_sha256",
        "dataset_sha256",
        "evaluation_config_sha256",
        "split",
        "thresholds_frozen_before_evaluation",
    }
    if not isinstance(evaluator, Mapping) or set(evaluator) != required_evaluator_keys:
        raise ContractError(
            "invalid_training_receipt",
            "evaluator_contract must contain the exact frozen held-out evaluator fields",
        )
    if (
        evaluator.get("schema") != EVALUATOR_SCHEMA
        or evaluator.get("split") != "heldout"
        or evaluator.get("thresholds_frozen_before_evaluation") is not True
    ):
        raise ContractError("invalid_training_receipt", "held-out evaluator contract is invalid")
    for key in ("evaluator_sha256", "dataset_sha256", "evaluation_config_sha256"):
        if not isinstance(evaluator.get(key), str) or not _SHA256_RE.fullmatch(evaluator[key]):
            raise ContractError("invalid_training_receipt", f"evaluator_contract.{key} is invalid")

    updates = receipt.get("committed_objective_updates")
    heldout = receipt.get("heldout_validation")
    if not isinstance(updates, Mapping) or not isinstance(heldout, Mapping):
        raise ContractError(
            "invalid_training_receipt",
            "training receipt requires committed_objective_updates and heldout_validation objects",
        )

    if set(updates) != set(CANONICAL_MODES) or set(heldout) != set(CANONICAL_MODES):
        raise ContractError(
            "invalid_training_receipt",
            "receipt must contain exact AR/SAT Fixed/SAT Variable/NAT commit and held-out records",
        )

    receipt_ids: list[str] = []
    for mode in CANONICAL_MODES:
        if modes[mode].trained is not True:
            raise ContractError(
                "invalid_training_receipt",
                f"v2 multimode export cannot carry an untrained capability: {mode}",
            )
        update_count = positive_int(updates.get(mode), f"committed_objective_updates.{mode}")
        if update_count < MIN_TRAINING_COMMITS:
            raise ContractError(
                "invalid_training_receipt",
                f"trained capability {mode} has fewer than {MIN_TRAINING_COMMITS} committed objective updates",
            )
        validation = heldout.get(mode)
        if not isinstance(validation, Mapping):
            raise ContractError("invalid_training_receipt", f"heldout_validation.{mode} is required")
        receipt_id = validation.get("receipt_id")
        if not isinstance(receipt_id, str) or not _RECEIPT_ID_RE.fullmatch(receipt_id):
            raise ContractError("invalid_training_receipt", f"heldout_validation.{mode}.receipt_id is invalid")
        receipt_ids.append(receipt_id)
        if receipt_id != modes[mode].training_receipt_id:
            raise ContractError(
                "invalid_training_receipt",
                f"heldout_validation.{mode}.receipt_id does not match capability training_receipt_id",
            )
        if validation.get("passed") is not True:
            raise ContractError("invalid_training_receipt", f"heldout validation did not pass for {mode}")
        tokens = positive_int(validation.get("tokens"), f"heldout_validation.{mode}.tokens")
        if tokens < MIN_HELDOUT_TOKENS:
            raise ContractError(
                "invalid_training_receipt",
                f"heldout_validation.{mode}.tokens is below {MIN_HELDOUT_TOKENS}",
            )
        cross_entropy = finite_number(
            validation.get("cross_entropy"), f"heldout_validation.{mode}.cross_entropy"
        )
        maximum_cross_entropy = finite_number(
            validation.get("maximum_cross_entropy"),
            f"heldout_validation.{mode}.maximum_cross_entropy",
        )
        baseline_cross_entropy = finite_number(
            validation.get("baseline_cross_entropy"),
            f"heldout_validation.{mode}.baseline_cross_entropy",
        )
        if not cross_entropy <= maximum_cross_entropy <= baseline_cross_entropy:
            raise ContractError(
                "invalid_training_receipt",
                f"heldout_validation.{mode} must prove CE <= frozen maximum <= baseline",
            )

    if len(set(receipt_ids)) != len(CANONICAL_MODES):
        raise ContractError("invalid_training_receipt", "per-mode held-out receipt IDs must be distinct")

    sat_fixed = heldout["sat_fixed"]
    if (
        sat_fixed.get("fixed_pair_contract_passed") is not True
        or sat_fixed.get("prompt_relative_pairing_contract_passed") is not True
        or positive_int(sat_fixed.get("odd_prompt_cases"), "sat_fixed.odd_prompt_cases")
        < MIN_PROMPT_PARITY_CASES
        or positive_int(sat_fixed.get("even_prompt_cases"), "sat_fixed.even_prompt_cases")
        < MIN_PROMPT_PARITY_CASES
    ):
        raise ContractError("invalid_training_receipt", "SAT Fixed prompt-relative pair contract failed")

    sat_variable = heldout["sat_variable"]
    census = sat_variable.get("gate_census")
    if not isinstance(census, Mapping):
        raise ContractError("invalid_training_receipt", "sat_variable gate_census is required")
    rows = positive_int(census.get("rows"), "gate_census.rows")
    stride1 = nonnegative_int(census.get("stride1"), "gate_census.stride1")
    stride2 = nonnegative_int(census.get("stride2"), "gate_census.stride2")
    label1 = nonnegative_int(census.get("label_stride1"), "gate_census.label_stride1")
    label2 = nonnegative_int(census.get("label_stride2"), "gate_census.label_stride2")
    balanced_accuracy = finite_number(census.get("balanced_accuracy"), "gate_census.balanced_accuracy")
    ece = finite_number(census.get("ece"), "gate_census.ece")
    brier = finite_number(census.get("brier"), "gate_census.brier")
    false_accept_rate = finite_number(census.get("false_accept_rate"), "gate_census.false_accept_rate")
    threshold = finite_number(census.get("threshold"), "gate_census.threshold")
    if (
        sat_variable.get("realignment_contract_passed") is not True
        or sat_variable.get("prompt_relative_pairing_contract_passed") is not True
        or positive_int(sat_variable.get("odd_prompt_cases"), "sat_variable.odd_prompt_cases")
        < MIN_PROMPT_PARITY_CASES
        or positive_int(sat_variable.get("even_prompt_cases"), "sat_variable.even_prompt_cases")
        < MIN_PROMPT_PARITY_CASES
        or rows < MIN_GATE_CENSUS_ROWS
        or stride1 + stride2 != rows
        or label1 + label2 != rows
        or min(stride1, stride2) <= 0
        or min(label1, label2) < MIN_GATE_LABELS_PER_CLASS
        or balanced_accuracy < 0.55
        or ece > 0.12
        or brier > 0.24
        or false_accept_rate > 0.05
        or census.get("collapsed") is not False
        or not math.isclose(
            threshold,
            SAT_VARIABLE_STRIDE2_THRESHOLD,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ContractError("invalid_training_receipt", "SAT Variable gate qualification failed")

    nat = heldout["nat"]
    if (
        nat.get("clean_prefix_masked_suffix") is not True
        or positive_int(nat.get("refinement_passes"), "heldout_validation.nat.refinement_passes")
        != NAT_REFINEMENT_STEPS
    ):
        raise ContractError("invalid_training_receipt", "NAT refinement contract failed")

    benchmark = receipt.get("throughput_benchmark")
    if not isinstance(benchmark, Mapping) or benchmark.get("passed") is not True:
        raise ContractError("invalid_training_receipt", "passed throughput_benchmark is required")
    throughput_receipt_id = benchmark.get("receipt_id")
    if (
        not isinstance(throughput_receipt_id, str)
        or not _RECEIPT_ID_RE.fullmatch(throughput_receipt_id)
        or throughput_receipt_id in receipt_ids
    ):
        raise ContractError("invalid_training_receipt", "throughput receipt ID is invalid or reused")
    prompt_count = positive_int(benchmark.get("prompt_count"), "throughput_benchmark.prompt_count")
    if prompt_count < MIN_THROUGHPUT_PROMPTS:
        raise ContractError("invalid_training_receipt", "throughput prompt census is too small")
    generated = benchmark.get("generated_tokens")
    speeds = benchmark.get("tokens_per_second")
    speed_modes = ("ar", "sat_variable", "sat_fixed")
    if (
        not isinstance(generated, Mapping)
        or not isinstance(speeds, Mapping)
        or set(generated) != set(speed_modes)
        or set(speeds) != set(speed_modes)
    ):
        raise ContractError("invalid_training_receipt", "throughput benchmark mode set is malformed")
    for mode in speed_modes:
        if positive_int(generated.get(mode), f"throughput_benchmark.generated_tokens.{mode}") < MIN_THROUGHPUT_TOKENS_PER_MODE:
            raise ContractError("invalid_training_receipt", f"throughput token census is too small for {mode}")
    ar_tps = finite_number(speeds.get("ar"), "throughput_benchmark.tokens_per_second.ar", minimum=1e-12)
    variable_tps = finite_number(
        speeds.get("sat_variable"),
        "throughput_benchmark.tokens_per_second.sat_variable",
        minimum=1e-12,
    )
    fixed_tps = finite_number(
        speeds.get("sat_fixed"),
        "throughput_benchmark.tokens_per_second.sat_fixed",
        minimum=1e-12,
    )
    if not ar_tps < variable_tps < fixed_tps:
        raise ContractError(
            "invalid_training_receipt",
            "throughput receipt must prove AR < SAT variable < SAT fixed tokens/second",
        )

    _validate_export_metadata(export_metadata, binding)
    return receipt


@dataclass(frozen=True)
class CheckpointManifest:
    model_id: str
    artifact_path: str
    artifact_sha256: str
    artifact_schema: str
    tokenizer_sha256: str | None
    modes: Mapping[str, ModeDeclaration]
    capability_contract: Mapping[str, Any]
    export_metadata: Mapping[str, Any] | None
    model_metadata: Mapping[str, Any]
    training_receipt: Mapping[str, Any] | None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "CheckpointManifest":
        if not isinstance(raw, Mapping):
            raise ContractError("invalid_manifest", "manifest must be a JSON object")

        model_id = str(raw.get("model_id") or "").strip()
        artifact = raw.get("artifact")
        if not model_id:
            raise ContractError("invalid_manifest", "manifest.model_id is required")
        if not isinstance(artifact, Mapping):
            raise ContractError("invalid_manifest", "manifest.artifact must be an object")

        artifact_path = str(artifact.get("path") or "").strip()
        artifact_sha256 = str(artifact.get("sha256") or "").strip().lower()
        artifact_schema = str(artifact.get("schema") or "").strip()
        tokenizer_sha256 = str(raw.get("tokenizer_sha256") or "").strip().lower() or None
        if not artifact_path:
            raise ContractError("invalid_manifest", "artifact.path is required")
        if not _SHA256_RE.fullmatch(artifact_sha256):
            raise ContractError("invalid_manifest", "artifact.sha256 must be 64 lowercase hex characters")
        if tokenizer_sha256 is not None and not _SHA256_RE.fullmatch(tokenizer_sha256):
            raise ContractError("invalid_manifest", "tokenizer_sha256 must be 64 lowercase hex characters")

        training_receipt: Mapping[str, Any] | None = None
        export_metadata: Mapping[str, Any] | None = None
        if artifact_schema == LEGACY_AR_SCHEMA:
            # Security/quality boundary: this schema means AR-only, regardless
            # of extra `sat` keys, a zero gate, or any caller-supplied claims.
            capability_contract = _legacy_capability_contract()
            modes = {
                mode: ModeDeclaration(
                    trained=capability_contract["modes"][mode]["trained"],
                    state_binding=capability_contract["modes"][mode]["state_binding"],
                    telemetry_contract=capability_contract["modes"][mode]["telemetry_contract"],
                    training_receipt_id=capability_contract["modes"][mode].get("training_receipt_id"),
                    reason=capability_contract["modes"][mode]["status"],
                )
                for mode in CANONICAL_MODES
            }
        elif artifact_schema == MULTIMODE_EXPORT_SCHEMA:
            if tokenizer_sha256 is None:
                raise ContractError("invalid_manifest", "multimode manifest requires tokenizer_sha256")
            if raw.get("capability_schema") != CAPABILITY_SCHEMA:
                raise ContractError(
                    "invalid_manifest",
                    f"multimode export requires capability_schema={CAPABILITY_SCHEMA}",
                )
            declared_contract = raw.get("capabilities")
            if (
                not isinstance(declared_contract, Mapping)
                or set(declared_contract) != {"schema", "source_schema", "modes"}
                or declared_contract.get("schema") != CAPABILITY_SCHEMA
                or declared_contract.get("source_schema") != LEGACY_AR_SCHEMA
                or not isinstance(declared_contract.get("modes"), Mapping)
            ):
                raise ContractError(
                    "invalid_manifest",
                    "manifest.capabilities must be the exact v1 capability contract reported by the loader",
                )
            declared = declared_contract["modes"]
            missing = sorted(set(CANONICAL_MODES) - set(declared))
            unknown = sorted(set(declared) - set(CANONICAL_MODES))
            if missing or unknown:
                raise ContractError(
                    "invalid_manifest",
                    "multimode manifest must contain exactly four canonical capability records",
                    extra={"missing_modes": missing, "unknown_modes": unknown},
                )
            modes = {}
            required_capability_keys = {
                "inference",
                "training",
                "trained",
                "objective",
                "status",
                "state_binding",
                "telemetry_contract",
                "training_receipt_id",
                "committed_updates",
            }
            for mode in CANONICAL_MODES:
                item = declared.get(mode, {})
                if not isinstance(item, Mapping):
                    raise ContractError("invalid_manifest", f"capabilities.{mode} must be an object")
                if set(item) != required_capability_keys:
                    raise ContractError(
                        "invalid_manifest",
                        f"capabilities.{mode} must contain the exact qualified v2 fields",
                    )
                for flag in ("inference", "training", "trained"):
                    if type(item.get(flag)) is not bool:
                        raise ContractError("invalid_manifest", f"capabilities.{mode}.{flag} must be boolean")
                if item.get("inference") is not True or item.get("training") is not True or item.get("trained") is not True:
                    raise ContractError(
                        "invalid_manifest",
                        f"qualified v2 capability {mode} must be trainable, inferable, and trained",
                    )
                if item.get("objective") != MODE_OBJECTIVES[mode]:
                    raise ContractError("invalid_manifest", f"capabilities.{mode}.objective is not canonical")
                if item.get("status") != "trained_and_heldout_validated":
                    raise ContractError(
                        "invalid_manifest",
                        f"capabilities.{mode}.status must be trained_and_heldout_validated",
                    )
                trained = True
                state_binding = str(item.get("state_binding") or "").strip() or None
                telemetry_contract = str(item.get("telemetry_contract") or "").strip() or None
                training_receipt_id = str(item.get("training_receipt_id") or "").strip() or None
                if state_binding != MODE_STATE_BINDINGS[mode]:
                    raise ContractError(
                        "invalid_manifest",
                        f"capabilities.{mode}.state_binding is not canonical",
                    )
                if telemetry_contract != TELEMETRY_SCHEMA:
                    raise ContractError(
                        "invalid_manifest",
                        f"trained capability {mode} requires telemetry_contract={TELEMETRY_SCHEMA}",
                    )
                if not training_receipt_id or not _RECEIPT_ID_RE.fullmatch(training_receipt_id):
                    raise ContractError(
                        "invalid_manifest",
                        f"trained capability {mode} requires a valid training_receipt_id",
                    )
                committed_updates = item.get("committed_updates")
                if (
                    type(committed_updates) is not int
                    or committed_updates < MIN_TRAINING_COMMITS
                ):
                    raise ContractError(
                        "invalid_manifest",
                        f"capabilities.{mode}.committed_updates must be >= {MIN_TRAINING_COMMITS}",
                    )
                modes[mode] = ModeDeclaration(
                    trained=trained,
                    state_binding=state_binding,
                    telemetry_contract=telemetry_contract,
                    training_receipt_id=training_receipt_id,
                    reason=str(item.get("reason") or item["status"]),
                )
            trained_bindings = [item.state_binding for item in modes.values() if item.trained]
            trained_receipts = [item.training_receipt_id for item in modes.values() if item.trained]
            if len(set(trained_bindings)) != len(trained_bindings):
                raise ContractError("invalid_manifest", "trained capabilities require distinct state_binding values")
            if len(set(trained_receipts)) != len(trained_receipts):
                raise ContractError("invalid_manifest", "trained capabilities require distinct training_receipt_id values")
            capability_contract = _copy_finite_json(
                declared_contract,
                "manifest capabilities",
            )
            export_metadata_raw = raw.get("export_metadata")
            if not isinstance(export_metadata_raw, Mapping):
                raise ContractError("invalid_manifest", "multimode manifest requires export_metadata")
            export_metadata = _copy_finite_json(export_metadata_raw, "manifest export_metadata")
            training_receipt = _validate_training_receipt(
                raw.get("agillm5_detachable_50m_training_receipt"),
                modes,
                export_metadata,
            )
            for mode in CANONICAL_MODES:
                item = capability_contract["modes"][mode]
                if item["committed_updates"] != training_receipt["committed_objective_updates"][mode]:
                    raise ContractError(
                        "invalid_manifest",
                        f"capabilities.{mode}.committed_updates does not match its training receipt",
                    )
        else:
            raise ContractError(
                "unsupported_checkpoint_schema",
                "checkpoint schema is not trusted for pop-out serving",
                status=503,
                extra={"artifact_schema": artifact_schema},
            )

        model_metadata = raw.get("model_metadata", {})
        if not isinstance(model_metadata, Mapping):
            raise ContractError("invalid_manifest", "model_metadata must be an object")
        return cls(
            model_id=model_id,
            artifact_path=artifact_path,
            artifact_sha256=artifact_sha256,
            artifact_schema=artifact_schema,
            tokenizer_sha256=REDACTED
            modes=modes,
            capability_contract=capability_contract,
            export_metadata=export_metadata,
            model_metadata=dict(model_metadata),
            training_receipt=training_receipt,
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "CheckpointManifest":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_mapping(json.load(handle))

    def validate_preflight_report(self, report: Any) -> None:
        """Bind a safe loader's verified payload facts to this pinned manifest."""

        if not isinstance(report, Mapping):
            raise ContractError("preflight_failed", "loader preflight report must be an object", status=503)
        common_expected = {
            "checkpoint": self.artifact_path,
            "checkpoint_sha256": self.artifact_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "tokenizer_json_sha256": self.tokenizer_sha256,
            "artifact_schema": self.artifact_schema,
            "capability_schema": CAPABILITY_SCHEMA,
            "core_parameters": CORE_PARAMETERS,
            "standalone_parameters": STANDALONE_PARAMETERS,
        }
        if report.get("ok") is not True:
            raise ContractError("preflight_failed", "loader preflight did not report ok=true", status=503)
        for field, expected in common_expected.items():
            if not _exact_json_equal(report.get(field), expected):
                raise ContractError(
                    "preflight_failed",
                    f"loader preflight {field} does not match the pinned manifest contract",
                    status=503,
                )
        if not _exact_json_equal(report.get("capabilities"), self.capability_contract):
            raise ContractError(
                "preflight_failed",
                "loader-reported capabilities do not exactly match the pinned manifest",
                status=503,
            )

        if self.artifact_schema == LEGACY_AR_SCHEMA:
            if "training_receipt" not in report or report.get("training_receipt") is not None:
                raise ContractError(
                    "preflight_failed",
                    "legacy loader must explicitly report training_receipt=null",
                    status=503,
                )
            if "checkpoint_binding" not in report or report.get("checkpoint_binding") is not None:
                raise ContractError(
                    "preflight_failed",
                    "legacy loader must explicitly report checkpoint_binding=null",
                    status=503,
                )
            return

        assert self.training_receipt is not None
        assert self.export_metadata is not None
        expected_binding = self.training_receipt["checkpoint_binding"]
        exact_fields = {
            "training_receipt": self.training_receipt,
            "checkpoint_binding": expected_binding,
            "export_metadata": self.export_metadata,
            "nat_mask_token_id": self.export_metadata["nat_mask_token_id"],
            "sat_variable_stride2_threshold": self.export_metadata[
                "sat_variable_stride2_threshold"
            ],
            "nat_refinement_steps": self.export_metadata["nat_refinement_steps"],
        }
        for field, expected in exact_fields.items():
            if field not in report or not _exact_json_equal(report[field], expected):
                raise ContractError(
                    "preflight_failed",
                    f"loader-reported {field} does not exactly match the pinned manifest",
                    status=503,
                )

    def verify_artifact(self, *, chunk_size: int = 8 * 1024 * 1024) -> None:
        artifact = Path(self.artifact_path)
        if not artifact.is_file():
            raise ContractError(
                "checkpoint_missing",
                "pinned checkpoint file is missing",
                status=503,
                extra={"artifact_path": str(artifact)},
            )
        digest = hashlib.sha256()
        with artifact.open("rb") as handle:
            while chunk := handle.read(chunk_size):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != self.artifact_sha256:
            raise ContractError(
                "checkpoint_hash_mismatch",
                "pinned checkpoint SHA-256 does not match",
                status=503,
                extra={"expected_sha256": self.artifact_sha256, "actual_sha256": actual},
            )


def parse_request_id(value: Any) -> str | None:
    if value is None or value == "":
        return None
    request_id = str(value).strip()
    if not _REQUEST_ID_RE.fullmatch(request_id):
        raise ContractError(
            "invalid_request_id",
            "request_id must be 1-128 characters using letters, digits, dot, underscore, colon, or dash",
        )
    return request_id


def _strict_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
        return value.strip().lower() in {"true", "1"}
    raise ContractError("invalid_request", f"{field} must be boolean")


def canonical_mode(payload: Mapping[str, Any]) -> str:
    """Resolve canonical mode while preserving legacy UI SAT + var requests."""

    if not isinstance(payload, Mapping):
        raise ContractError("invalid_request", "request body must be a JSON object")
    raw_mode = payload.get("mode")
    raw_sampler = payload.get("sampler")

    if raw_mode is None or str(raw_mode).strip() == "":
        if raw_sampler is None or str(raw_sampler).strip() == "":
            raise ContractError(
                "mode_required",
                "one of ar, sat_fixed, sat_variable, nat is required; there is no implicit fallback",
            )
        sampler_only = str(raw_sampler).strip().lower()
        if sampler_only == "sat":
            if "var" not in payload:
                raise ContractError(
                    "ambiguous_sat_mode",
                    "sampler=sat requires mode=sat_fixed/sat_variable or an explicit legacy var boolean",
                )
            return "sat_variable" if _strict_bool(payload["var"], field="var") else "sat_fixed"
        raw_mode = sampler_only

    normalized = str(raw_mode).strip().lower().replace("-", "_")
    aliases = {
        "ar": "ar",
        "sat_fixed": "sat_fixed",
        "satfixed": "sat_fixed",
        "sat_variable": "sat_variable",
        "satvariable": "sat_variable",
        "sat_var": "sat_variable",
        "nat": "nat",
    }
    if normalized == "sat":
        var = _strict_bool(payload.get("var", False), field="var")
        mode = "sat_variable" if var else "sat_fixed"
    else:
        mode = aliases.get(normalized)
    if mode is None:
        raise ContractError(
            "unknown_mode",
            "mode must be one of ar, sat_fixed, sat_variable, nat",
            extra={"requested_mode": str(raw_mode), "canonical_modes": list(CANONICAL_MODES)},
        )

    if "var" in payload and normalized != "sat":
        var = _strict_bool(payload["var"], field="var")
        expected = mode == "sat_variable"
        if mode not in {"sat_fixed", "sat_variable"} or var != expected:
            raise ContractError(
                "mode_var_mismatch",
                "legacy var flag conflicts with canonical mode",
                extra={"mode": mode, "var": var},
            )
    return mode


def validate_sampler(payload: Mapping[str, Any], mode: str) -> None:
    raw = payload.get("sampler")
    if raw is None or str(raw).strip() == "":
        return
    sampler = str(raw).strip().lower().replace("-", "_")
    allowed = {
        "ar": {"ar"},
        "sat_fixed": {"sat", "sat_fixed"},
        "sat_variable": {"sat", "sat_variable", "sat_var"},
        "nat": {"nat"},
    }[mode]
    if sampler not in allowed:
        raise ContractError(
            "sampler_mode_mismatch",
            "sampler does not match the requested canonical mode",
            extra={"mode": mode, "sampler": sampler, "allowed_samplers": sorted(allowed)},
        )


def _bounded_int(value: Any, *, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError("invalid_request", f"{field} must be an integer")
    if value < minimum or value > maximum:
        raise ContractError(
            "invalid_request",
            f"{field} must be between {minimum} and {maximum}",
            extra={"field": field, "minimum": minimum, "maximum": maximum},
        )
    return value


def _bounded_float(value: Any, *, field: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError("invalid_request", f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < minimum or parsed > maximum:
        raise ContractError(
            "invalid_request",
            f"{field} must be finite and between {minimum} and {maximum}",
            extra={"field": field, "minimum": minimum, "maximum": maximum},
        )
    return parsed


def runtime_payload(payload: Mapping[str, Any], mode: str) -> dict[str, Any]:
    """Build a sanitized native-runtime request with non-overridable routing."""

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ContractError("invalid_prompt", "prompt must be a non-empty string")
    if len(prompt.encode("utf-8")) > 256 * 1024:
        raise ContractError("prompt_too_large", "prompt exceeds 256 KiB")

    max_new = _bounded_int(payload.get("max_new", 64), field="max_new", minimum=1, maximum=512)
    min_new = _bounded_int(payload.get("min_new", 0), field="min_new", minimum=0, maximum=max_new)
    out: dict[str, Any] = {"prompt": prompt, "max_new": max_new, "min_new": min_new}

    integer_bounds = {
        "top_k": (0, 1024),
        "penalty_last_n": (0, 8192),
        "seed": (0, 2**63 - 1),
    }
    float_bounds = {
        "temperature": (0.0, 5.0),
        "top_p": (0.0, 1.0),
        "min_p": (0.0, 1.0),
        "repetition_penalty": (0.01, 10.0),
        "presence_penalty": (-10.0, 10.0),
        "frequency_penalty": (-10.0, 10.0),
    }
    for field, (minimum, maximum) in integer_bounds.items():
        if field in payload:
            out[field] = _bounded_int(payload[field], field=field, minimum=minimum, maximum=maximum)
    for field, (minimum, maximum) in float_bounds.items():
        if field in payload:
            out[field] = _bounded_float(payload[field], field=field, minimum=minimum, maximum=maximum)
    for field in ("greedy", "ignore_eos"):
        if field in payload:
            out[field] = _strict_bool(payload[field], field=field)
    if mode == "nat":
        out["nat_passes"] = _bounded_int(
            payload.get("nat_passes", 2),
            field="nat_passes",
            minimum=1,
            maximum=8,
        )
    semantics = MODE_SEMANTICS[mode]
    out["mode"] = semantics["runtime_mode"]
    out["sampler"] = semantics["runtime_sampler"]
    out["var"] = mode == "sat_variable"
    out["stream"] = True
    return out


def validate_telemetry(mode: str, raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate proof telemetry rather than trusting a mode label."""

    if not isinstance(raw, Mapping):
        raise ContractError("invalid_backend_telemetry", "backend telemetry must be an object", status=502)
    if raw.get("schema") != TELEMETRY_SCHEMA:
        raise ContractError(
            "invalid_backend_telemetry",
            f"backend telemetry schema must be {TELEMETRY_SCHEMA}",
            status=502,
        )
    if raw.get("mode") != mode:
        raise ContractError(
            "invalid_backend_telemetry",
            "backend telemetry mode does not match routed mode",
            status=502,
            extra={"routed_mode": mode, "telemetry_mode": raw.get("mode")},
        )

    integer_fields = (
        "decode_steps",
        "committed_tokens",
        "gate_decisions",
        "gate_commit_1",
        "gate_commit_2",
        "sat_tokens_committed",
        "ar_realign_tokens",
        "prompt_align_tokens",
        "total_emitted_tokens",
        "core_forwards",
        "refinement_steps",
    )
    parsed: dict[str, Any] = {"schema": TELEMETRY_SCHEMA, "mode": mode}
    for field in integer_fields:
        value = raw.get(field, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError(
                "invalid_backend_telemetry",
                f"telemetry.{field} must be a non-negative integer",
                status=502,
            )
        parsed[field] = value
    for field in ("elapsed_s", "tok_s", "gate_commit_2_rate"):
        value = raw.get(field, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0:
            raise ContractError(
                "invalid_backend_telemetry",
                f"telemetry.{field} must be a finite non-negative number",
                status=502,
            )
        parsed[field] = float(value)

    partial = raw.get("terminal_boundary_partial", False)
    if not isinstance(partial, bool):
        raise ContractError(
            "invalid_backend_telemetry",
            "telemetry.terminal_boundary_partial must be boolean",
            status=502,
        )
    parsed["terminal_boundary_partial"] = partial
    realign_skipped = raw.get("terminal_realign_skipped", False)
    if not isinstance(realign_skipped, bool):
        raise ContractError(
            "invalid_backend_telemetry",
            "telemetry.terminal_realign_skipped must be boolean",
            status=502,
        )
    parsed["terminal_realign_skipped"] = realign_skipped

    steps = parsed["decode_steps"]
    committed = parsed["committed_tokens"]
    decisions = parsed["gate_decisions"]
    gate1 = parsed["gate_commit_1"]
    gate2 = parsed["gate_commit_2"]
    sat_committed = parsed["sat_tokens_committed"]
    realign = parsed["ar_realign_tokens"]
    prompt_align = parsed["prompt_align_tokens"]
    emitted = parsed["total_emitted_tokens"]
    core_forwards = parsed["core_forwards"]
    refinement = parsed["refinement_steps"]
    expected_rate = (gate2 / decisions) if decisions else 0.0
    if not math.isclose(parsed["gate_commit_2_rate"], expected_rate, rel_tol=1e-6, abs_tol=1e-6):
        raise ContractError("invalid_backend_telemetry", "gate_commit_2_rate does not match gate counts", status=502)

    valid = False
    if mode == "ar":
        valid = (
            committed == steps == emitted
            and core_forwards >= steps
            and decisions == gate1 == gate2 == sat_committed == realign == prompt_align == refinement == 0
            and not partial
            and not realign_skipped
        )
    elif mode == "sat_fixed":
        # A fixed stride-2 decision can expose only one token at a terminal
        # max_new/EOS boundary. The explicit partial bit accounts for that one
        # truncation without weakening the always-stride-2 rule.
        valid = (
            decisions == steps == gate2
            and gate1 == 0
            and sat_committed == 2 * gate2 - int(partial)
            and committed == sat_committed
            and emitted == sat_committed + realign
            and prompt_align == 0
            and refinement == 0
            and core_forwards >= steps
            and not realign_skipped
        )
    elif mode == "sat_variable":
        valid = (
            decisions == steps == gate1 + gate2
            and sat_committed == gate1 + 2 * gate2 - int(partial)
            and committed == sat_committed
            and realign == gate1 - int(realign_skipped)
            and (not realign_skipped or gate1 > 0)
            and emitted == sat_committed + realign
            and prompt_align == 0
            and refinement == 0
            and core_forwards >= steps + realign
        )
    elif mode == "nat":
        valid = (
            refinement == steps
            and committed == emitted
            and decisions == gate1 == gate2 == sat_committed == realign == prompt_align == 0
            and core_forwards >= refinement
            and not partial
            and not realign_skipped
        )
    if not valid:
        raise ContractError(
            "decode_semantics_violation",
            "backend telemetry does not prove the routed mode's commit semantics",
            status=502,
            extra={"mode": mode, "telemetry": parsed, "semantics": MODE_SEMANTICS[mode]},
        )
    return parsed
