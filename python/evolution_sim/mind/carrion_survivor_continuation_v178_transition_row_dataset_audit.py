from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import json
from math import isfinite
from pathlib import Path
import re
from statistics import mean

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import decode_observation_input
from evolution_sim.mind.candidate_campaign import _int, _mapping, _round, write_json
from evolution_sim.mind.carrion_survivor_continuation_action_value_audit import (
    _action_order,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.carrion_survivor_continuation_v171_replay_expansion import (
    load_jsonl_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v177_exact_branch_replay_expansion import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V177_REPORT_PATH,
    DEFAULT_TRANSITION_DATASET_OUTPUT_PATH,
    FORBIDDEN_TRAINABLE_KEY_TOKENS,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_FEATURE_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_ROW_SCHEMA_VERSION,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_POLICY,
    M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_SCHEMA_VERSION,
    TRAINABLE_PUBLIC_FEATURE_KEYS,
    trainable_payload_leakage_scan,
    validate_v177_transition_rows,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V178_TRANSITION_ROW_DATASET_AUDIT_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V178_TRANSITION_ROW_DATASET_AUDIT_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_v1"
)
EXPECTED_V177_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v177_exact_branch_replay_expansion_"
    "compact_transition_rows_ready_no_training"
)
EXPECTED_V179_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion_report_v1"
)
EXPECTED_V179_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion_v1"
)
EXPECTED_V179_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion_"
    "compact_transition_rows_support_ready_for_v178_default_audit_no_training"
)
EXPECTED_V183_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v183_exact_transition_support_expansion_report_v1"
)
EXPECTED_V183_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v183_exact_transition_support_expansion_v1"
)
EXPECTED_V183_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v183_exact_transition_support_expansion_"
    "targeted_exact_support_ready_for_fresh_v178_audit_no_training"
)
EXPECTED_V183_REPORT_EXACT_DIGEST = (
    "7281380512c4a3ce9eb0951ce8f6a1b132a74b78f7fcaafe6af3adf2bb5d16da"
)
EXPECTED_V183_DATASET_DIGEST = (
    "e83424b8bb6e00a03e2afbbabd4d62c71dedfa0dec3beb482bdc73c2de1a81ef"
)
EXPECTED_V184_REPORT_EXACT_DIGEST = (
    "ad9a43a670c4fce5c4ceb7a3ce54abfbc153f3d0545a3762d78c65d55d3302aa"
)
EXPECTED_V185_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v185_v183_target_resolution_repair_report_v1"
)
EXPECTED_V185_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v185_v183_target_resolution_repair_v1"
)
EXPECTED_V185_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v185_v183_target_resolution_repair_"
    "strict_filter_ready_for_repaired_dataset_audit_no_training"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v178-carrion-survivor-continuation-transition-row-dataset-audit.json"
)
DEFAULT_MIN_ROW_COUNT = 128
DEFAULT_MIN_SEED_COUNT = 3
DEFAULT_MIN_BRANCH_COUNT = 24
DEFAULT_MIN_FORCED_ACTION_COUNT = 6
ENCODED_OBSERVATION_INPUT_KEYS = {
    "schema_version",
    "encoder_version",
    "decoded_dtype",
    "storage_dtype",
    "storage_encoding",
    "shape",
    "value_range",
    "data",
}
PREVIOUS_SAME_AGENT_PUBLIC_CONTEXT_KEYS = {
    "available",
    "public_observation",
    "public_action_mask",
    "public_action",
    "moved",
}
V177_SOURCE_PRODUCER = "v177"
V179_SOURCE_PRODUCER = "v179_exact_branch_transition_row_expansion"
V183_SOURCE_PRODUCER = "v183_exact_transition_support_expansion"
V185_SOURCE_PRODUCER = "v185_v183_target_resolution_repair"
V185_SLICE_2_TRAINING_ROUTE = "v185_transition_row_policy_training_slice_2_opt_in"
V186_SLICE_2_TRAINING_ROUTE = "v186_transition_row_policy_training_slice_2_opt_in"


def run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
    *,
    transition_dataset_path: str | Path = DEFAULT_TRANSITION_DATASET_OUTPUT_PATH,
    v177_report_path: str | Path = DEFAULT_V177_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v177_report_exact_digest: str | None = None,
    expected_dataset_digest: str | None = None,
    min_row_count: int = DEFAULT_MIN_ROW_COUNT,
    min_seed_count: int = DEFAULT_MIN_SEED_COUNT,
    min_branch_count: int = DEFAULT_MIN_BRANCH_COUNT,
    min_forced_action_count: int = DEFAULT_MIN_FORCED_ACTION_COUNT,
) -> dict[str, object]:
    v177_report = load_json_report(v177_report_path)
    rows = [dict(row) for row in load_jsonl_dataset(transition_dataset_path)]
    source_validation = validate_v178_sources(
        v177_report=v177_report,
        rows=rows,
        expected_v177_report_exact_digest=expected_v177_report_exact_digest,
        expected_dataset_digest=expected_dataset_digest,
    )
    row_schema_validation = validate_v177_transition_rows(rows)
    leakage_scan = trainable_payload_leakage_scan(
        [_mapping(row.get("trainable_public_features")) for row in rows]
    )
    value_leakage_scan = trainable_value_leakage_scan(
        [_mapping(row.get("trainable_public_features")) for row in rows]
    )
    feature_contract_audit = trainable_feature_contract_audit(rows)
    identity_audit = transition_row_identity_audit(rows)
    action_mask_audit = transition_row_action_mask_audit(rows)
    observation_audit = transition_row_observation_audit(rows)
    coverage_audit = transition_row_coverage_audit(rows)
    target_audit = transition_row_target_audit(rows)
    requested_support_minimums = _support_minimums(
        min_row_count=int(min_row_count),
        min_seed_count=int(min_seed_count),
        min_branch_count=int(min_branch_count),
        min_forced_action_count=int(min_forced_action_count),
    )
    default_support_minimums = _default_support_minimums()
    support_thresholds_match_defaults = (
        requested_support_minimums == default_support_minimums
    )
    support_readiness = transition_row_support_readiness(
        rows=rows,
        coverage_audit=coverage_audit,
        identity_audit=identity_audit,
        min_row_count=int(min_row_count),
        min_seed_count=int(min_seed_count),
        min_branch_count=int(min_branch_count),
        min_forced_action_count=int(min_forced_action_count),
    )
    default_support_readiness = transition_row_support_readiness(
        rows=rows,
        coverage_audit=coverage_audit,
        identity_audit=identity_audit,
        min_row_count=DEFAULT_MIN_ROW_COUNT,
        min_seed_count=DEFAULT_MIN_SEED_COUNT,
        min_branch_count=DEFAULT_MIN_BRANCH_COUNT,
        min_forced_action_count=DEFAULT_MIN_FORCED_ACTION_COUNT,
    )
    training_authorization = transition_row_training_authorization(
        source_validation=source_validation,
        row_schema_validation=row_schema_validation,
        leakage_scan=leakage_scan,
        value_leakage_scan=value_leakage_scan,
        feature_contract_audit=feature_contract_audit,
        identity_audit=identity_audit,
        action_mask_audit=action_mask_audit,
        observation_audit=observation_audit,
        target_audit=target_audit,
        default_support_readiness=default_support_readiness,
        requested_support_minimums=requested_support_minimums,
        default_support_minimums=default_support_minimums,
        support_thresholds_match_defaults=support_thresholds_match_defaults,
    )
    classification = _classification(
        source_validation=source_validation,
        row_schema_validation=row_schema_validation,
        leakage_scan=leakage_scan,
        value_leakage_scan=value_leakage_scan,
        feature_contract_audit=feature_contract_audit,
        identity_audit=identity_audit,
        action_mask_audit=action_mask_audit,
        observation_audit=observation_audit,
        target_audit=target_audit,
        default_support_readiness=default_support_readiness,
        support_thresholds_match_defaults=support_thresholds_match_defaults,
        training_authorization=training_authorization,
    )
    route_recommendation = _route_recommendation(
        classification=classification,
        support_readiness=support_readiness,
        default_support_readiness=default_support_readiness,
        training_authorization=training_authorization,
        source_validation=source_validation,
    )
    training_authorized = bool(
        route_recommendation.get("transition_row_training_authorized")
    )
    dataset_digest = stable_payload_digest(rows)
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V178_TRANSITION_ROW_DATASET_AUDIT_SCHEMA_VERSION
        ),
        "policy": M3_CARRION_SURVIVOR_CONTINUATION_V178_TRANSITION_ROW_DATASET_AUDIT_POLICY,
        "contract": _diagnostics_only_contract(
            transition_row_training_authorized=training_authorized
        ),
        "inputs": {
            "source_report": str(v177_report_path),
            "v177_report": str(v177_report_path),
            "transition_dataset": str(transition_dataset_path),
            "expected_source_report_exact_digest": (
                expected_v177_report_exact_digest
            ),
            "expected_v177_report_exact_digest": expected_v177_report_exact_digest,
            "expected_dataset_digest": expected_dataset_digest,
            "min_row_count": int(min_row_count),
            "min_seed_count": int(min_seed_count),
            "min_branch_count": int(min_branch_count),
            "min_forced_action_count": int(min_forced_action_count),
            "requested_support_minimums": requested_support_minimums,
            "default_support_minimums": default_support_minimums,
            "support_thresholds_match_defaults": support_thresholds_match_defaults,
        },
        "source_validation": source_validation,
        "row_schema_validation": row_schema_validation,
        "leakage_scan": leakage_scan,
        "value_leakage_scan": value_leakage_scan,
        "feature_contract_audit": feature_contract_audit,
        "identity_audit": identity_audit,
        "action_mask_audit": action_mask_audit,
        "observation_audit": observation_audit,
        "coverage_audit": coverage_audit,
        "target_audit": target_audit,
        "support_readiness": support_readiness,
        "default_support_readiness": default_support_readiness,
        "training_authorization": training_authorization,
        "dataset": {
            "path": str(transition_dataset_path),
            "row_count": len(rows),
            "dataset_digest": dataset_digest,
            "row_schema_version": (
                M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_ROW_SCHEMA_VERSION
            ),
            "feature_policy_id": (
                M3_CARRION_SURVIVOR_CONTINUATION_V177_COMPACT_TRANSITION_FEATURE_POLICY
            ),
        },
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": route_recommendation,
        **_lifecycle_flags(),
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def trainable_value_leakage_scan(
    feature_payloads: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for row_index, payload in enumerate(feature_payloads):
        _scan_trainable_values(
            value=payload,
            row_index=row_index,
            path=("trainable_public_features",),
            failures=failures,
            parent=None,
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v178_trainable_value_leakage_scan_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:96],
        "forbidden_value_tokens": list(FORBIDDEN_TRAINABLE_KEY_TOKENS),
        "opaque_encoded_observation_data_allowed": True,
        "payload_count": len(feature_payloads),
    }


def trainable_feature_contract_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        features = _mapping(row.get("trainable_public_features"))
        _require_key_set(
            failures=failures,
            row_index=row_index,
            path=("trainable_public_features",),
            observed=features,
            expected=TRAINABLE_PUBLIC_FEATURE_KEYS,
        )
        _audit_encoded_observation_contract(
            failures=failures,
            row_index=row_index,
            path=("trainable_public_features", "current_public_observation"),
            payload=features.get("current_public_observation"),
            required=True,
        )
        _audit_action_mask_contract(
            failures=failures,
            row_index=row_index,
            path=("trainable_public_features", "current_public_action_mask"),
            payload=features.get("current_public_action_mask"),
            required=True,
        )
        forced_action = features.get("forced_action")
        if forced_action not in ACTION_NAMES:
            failures.append(
                {
                    "row_index": row_index,
                    "path": "trainable_public_features.forced_action",
                    "reason": "invalid_forced_action",
                    "observed": forced_action,
                }
            )
        previous = _mapping(features.get("previous_same_agent_public_context"))
        _require_key_set(
            failures=failures,
            row_index=row_index,
            path=(
                "trainable_public_features",
                "previous_same_agent_public_context",
            ),
            observed=previous,
            expected=PREVIOUS_SAME_AGENT_PUBLIC_CONTEXT_KEYS,
        )
        _audit_previous_context_contract(
            failures=failures,
            row_index=row_index,
            previous=previous,
        )
        done = row.get("transition_done") is True
        _audit_encoded_observation_contract(
            failures=failures,
            row_index=row_index,
            path=("trainable_public_features", "next_public_observation"),
            payload=features.get("next_public_observation"),
            required=not done,
        )
        _audit_action_mask_contract(
            failures=failures,
            row_index=row_index,
            path=("trainable_public_features", "next_public_action_mask"),
            payload=features.get("next_public_action_mask"),
            required=not done,
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v178_trainable_feature_contract_audit_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:128],
        "row_count": len(rows),
        "top_level_feature_keys": sorted(TRAINABLE_PUBLIC_FEATURE_KEYS),
        "previous_context_keys": sorted(PREVIOUS_SAME_AGENT_PUBLIC_CONTEXT_KEYS),
        "encoded_observation_keys": sorted(ENCODED_OBSERVATION_INPUT_KEYS),
    }


def validate_v178_sources(
    *,
    v177_report: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    expected_v177_report_exact_digest: str | None,
    expected_dataset_digest: str | None,
) -> dict[str, object]:
    failures: list[str] = []
    schema_version = str(v177_report.get("schema_version") or "")
    producer = V177_SOURCE_PRODUCER
    failure_prefix = "v177"
    is_v179_source = False
    is_v183_source = False
    is_v185_source = False
    expected_schema_version = (
        M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_SCHEMA_VERSION
    )
    expected_policy = (
        M3_CARRION_SURVIVOR_CONTINUATION_V177_EXACT_BRANCH_REPLAY_EXPANSION_POLICY
    )
    expected_classification = EXPECTED_V177_CLASSIFICATION
    if schema_version == EXPECTED_V179_SCHEMA_VERSION:
        producer = V179_SOURCE_PRODUCER
        failure_prefix = "v179"
        is_v179_source = True
        expected_schema_version = EXPECTED_V179_SCHEMA_VERSION
        expected_policy = EXPECTED_V179_POLICY
        expected_classification = EXPECTED_V179_CLASSIFICATION
    elif schema_version == EXPECTED_V183_SCHEMA_VERSION:
        producer = V183_SOURCE_PRODUCER
        failure_prefix = "v183"
        is_v183_source = True
        expected_schema_version = EXPECTED_V183_SCHEMA_VERSION
        expected_policy = EXPECTED_V183_POLICY
        expected_classification = EXPECTED_V183_CLASSIFICATION
    elif schema_version == EXPECTED_V185_SCHEMA_VERSION:
        producer = V185_SOURCE_PRODUCER
        failure_prefix = "v185"
        is_v185_source = True
        expected_schema_version = EXPECTED_V185_SCHEMA_VERSION
        expected_policy = EXPECTED_V185_POLICY
        expected_classification = EXPECTED_V185_CLASSIFICATION
    elif schema_version != expected_schema_version:
        failures.append("v177_schema_version_mismatch")
    if v177_report.get("policy") != expected_policy:
        failures.append(f"{failure_prefix}_policy_mismatch")
    observed_classification = str(
        _mapping(v177_report.get("classification")).get("primary") or ""
    )
    if observed_classification != expected_classification:
        failures.append(f"{failure_prefix}_unexpected_classification")
    exact = exact_digest_validation_report(v177_report)
    observed_exact = str(v177_report.get("exact_digest") or "")
    if exact.get("passed") is not True:
        failures.append(f"{failure_prefix}_exact_digest_mismatch")
    if is_v183_source and observed_exact != EXPECTED_V183_REPORT_EXACT_DIGEST:
        failures.append("v183_canonical_exact_digest_mismatch")
    if (
        expected_v177_report_exact_digest
        and observed_exact != expected_v177_report_exact_digest
    ):
        failures.append(f"{failure_prefix}_unexpected_exact_digest")
    observed_dataset_digest = stable_payload_digest([dict(row) for row in rows])
    if is_v183_source and observed_dataset_digest != EXPECTED_V183_DATASET_DIGEST:
        failures.append("v183_canonical_dataset_digest_mismatch")
    if expected_dataset_digest and observed_dataset_digest != expected_dataset_digest:
        failures.append(f"{failure_prefix}_dataset_digest_mismatch")
    dataset = _mapping(v177_report.get("dataset"))
    reported_digest = dataset.get("dataset_digest")
    reported_row_count = dataset.get("row_count")
    if reported_digest not in (None, observed_dataset_digest):
        failures.append(f"{failure_prefix}_reported_dataset_digest_mismatch")
    if (
        reported_row_count is not None
        and _int(reported_row_count, default=-1) != len(rows)
    ):
        failures.append(f"{failure_prefix}_reported_dataset_row_count_mismatch")
    lifecycle = _source_lifecycle_validation(
        v177_report,
        producer=producer,
        is_v179_source=is_v179_source,
        is_v183_source=is_v183_source,
        is_v185_source=is_v185_source,
    )
    if lifecycle.get("passed") is not True:
        failures.append(f"{failure_prefix}_lifecycle_not_diagnostics_only")
    v179_replay_validation = _empty_v179_replay_validation()
    if is_v179_source:
        v179_replay_validation = _v179_replay_validation(v177_report)
        if v179_replay_validation.get("passed") is not True:
            failures.append("v179_replay_verification_not_proven")
        if v179_replay_validation.get("v177_source_digests_pinned") is not True:
            failures.append("v179_v177_source_digests_not_pinned")
    v183_expansion_validation = _empty_v183_expansion_validation()
    if is_v183_source:
        v183_expansion_validation = _v183_expansion_validation(
            v177_report,
            observed_dataset_digest=observed_dataset_digest,
        )
        failures.extend(str(item) for item in v183_expansion_validation.get("failures") or [])
    v185_repair_validation = _empty_v185_repair_validation()
    if is_v185_source:
        v185_repair_validation = _v185_repair_validation(
            v177_report,
            observed_dataset_digest=observed_dataset_digest,
        )
        failures.extend(
            str(item) for item in v185_repair_validation.get("failures") or []
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v178_source_validation_v1",
        "passed": bool(rows) and not failures,
        "failures": sorted(set(failures)),
        "source_producer": producer,
        "is_v183_source": is_v183_source,
        "is_v185_source": is_v185_source,
        "source_report_schema_version": schema_version,
        "expected_source_report_schema_version": expected_schema_version,
        "expected_source_policy": expected_policy,
        "expected_source_classification": expected_classification,
        "observed_source_classification": observed_classification,
        "expected_v177_classification": (
            EXPECTED_V177_CLASSIFICATION
            if not is_v179_source and not is_v183_source and not is_v185_source
            else None
        ),
        "observed_v177_classification": (
            observed_classification
            if not is_v179_source and not is_v183_source and not is_v185_source
            else None
        ),
        "expected_v177_report_exact_digest": expected_v177_report_exact_digest,
        "observed_v177_report_exact_digest": observed_exact,
        "v177_exact_digest_validation": exact,
        "expected_source_report_exact_digest": expected_v177_report_exact_digest,
        "observed_source_report_exact_digest": observed_exact,
        "source_report_exact_digest_validation": exact,
        "expected_source_report_exact_digest_provided": bool(
            expected_v177_report_exact_digest
        ),
        "expected_dataset_digest": expected_dataset_digest,
        "observed_dataset_digest": observed_dataset_digest,
        "expected_dataset_digest_provided": bool(expected_dataset_digest),
        "reported_source_dataset_digest": reported_digest,
        "reported_source_dataset_row_count": reported_row_count,
        "v177_reported_dataset_digest": reported_digest,
        "v177_reported_dataset_row_count": reported_row_count,
        "observed_dataset_row_count": len(rows),
        "v177_lifecycle_validation": lifecycle,
        "source_lifecycle_validation": lifecycle,
        "v179_replay_validation": v179_replay_validation,
        "expected_v183_report_exact_digest": (
            EXPECTED_V183_REPORT_EXACT_DIGEST if is_v183_source else None
        ),
        "expected_v183_dataset_digest": (
            EXPECTED_V183_DATASET_DIGEST if is_v183_source else None
        ),
        "v183_report_exact_digest_matches_canonical": (
            observed_exact == EXPECTED_V183_REPORT_EXACT_DIGEST
            if is_v183_source
            else None
        ),
        "v183_dataset_digest_matches_canonical": (
            observed_dataset_digest == EXPECTED_V183_DATASET_DIGEST
            if is_v183_source
            else None
        ),
        "v183_expansion_validation": v183_expansion_validation,
        "v183_source_digests_pinned": (
            v183_expansion_validation.get("upstream_evidence_pinned") is True
            if is_v183_source
            else None
        ),
        "expected_v184_report_exact_digest": (
            EXPECTED_V184_REPORT_EXACT_DIGEST if is_v185_source else None
        ),
        "expected_v185_report_exact_digest": (
            expected_v177_report_exact_digest if is_v185_source else None
        ),
        "v185_report_exact_digest_matches_expected": (
            observed_exact == expected_v177_report_exact_digest
            if is_v185_source and expected_v177_report_exact_digest
            else None
        ),
        "v185_repair_validation": v185_repair_validation,
        "v185_source_digests_pinned": (
            v185_repair_validation.get("source_evidence_pinned") is True
            if is_v185_source
            else None
        ),
    }


def transition_row_identity_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    branch_action_counts: Counter[tuple[str, str]] = Counter()
    feature_digest_counts: Counter[str] = Counter()
    current_state_digest_counts: Counter[str] = Counter()
    branch_digest_counts: Counter[str] = Counter()
    source_identities_by_branch: dict[str, set[tuple[object, ...]]] = defaultdict(set)
    source_record_digests_by_branch: dict[str, set[str]] = defaultdict(set)
    materialized_record_digests_by_branch: dict[str, set[str]] = defaultdict(set)
    branch_state_digests_by_branch: dict[str, set[str]] = defaultdict(set)
    current_state_digests_by_branch: dict[str, set[str]] = defaultdict(set)
    source_identity_branch_ids: dict[tuple[object, ...], set[str]] = defaultdict(set)
    source_record_digest_branch_ids: dict[str, set[str]] = defaultdict(set)
    materialized_record_digest_branch_ids: dict[str, set[str]] = defaultdict(set)
    current_state_digest_branch_ids: dict[str, set[str]] = defaultdict(set)
    failures: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        metadata = _mapping(row.get("metadata"))
        branch_id = str(metadata.get("branch_id") or "")
        action = str(row.get("forced_action") or "")
        branch_action_counts.update([(branch_id, action)])
        feature_digest_counts.update(
            [stable_payload_digest(_mapping(row.get("trainable_public_features")))]
        )
        current_state_digest = stable_payload_digest(
            {
                "current_public_observation": row.get("current_public_observation"),
                "current_public_action_mask": row.get("current_public_action_mask"),
                "previous_same_agent_public_context": row.get(
                    "previous_same_agent_public_context"
                ),
            }
        )
        current_state_digest_counts.update([current_state_digest])
        if branch_id:
            current_state_digest_branch_ids[current_state_digest].add(branch_id)
        branch_state_digest = str(metadata.get("branch_state_digest") or "")
        if branch_state_digest:
            branch_digest_counts.update([branch_state_digest])
        source_record_digest = str(metadata.get("source_record_digest") or "")
        materialized_record_digest = str(
            metadata.get("materialized_record_digest") or ""
        )
        source_identity = (
            _int(metadata.get("seed"), default=-1),
            str(metadata.get("source_path") or ""),
            _int(metadata.get("line_number"), default=-1),
            _int(metadata.get("branch_tick"), default=-1),
            _int(metadata.get("agent_id"), default=-1),
        )
        if branch_id:
            if (
                source_identity[0] >= 0
                and source_identity[1]
                and source_identity[2] >= 0
                and source_identity[3] >= 0
                and source_identity[4] >= 0
            ):
                source_identity_branch_ids[source_identity].add(branch_id)
                source_identities_by_branch[branch_id].add(source_identity)
            if source_record_digest:
                source_record_digest_branch_ids[source_record_digest].add(branch_id)
                source_record_digests_by_branch[branch_id].add(source_record_digest)
            if materialized_record_digest:
                materialized_record_digest_branch_ids[
                    materialized_record_digest
                ].add(branch_id)
                materialized_record_digests_by_branch[branch_id].add(
                    materialized_record_digest
                )
            if branch_state_digest:
                branch_state_digests_by_branch[branch_id].add(branch_state_digest)
            current_state_digests_by_branch[branch_id].add(current_state_digest)
        if not branch_id:
            failures.append({"row_index": row_index, "reason": "missing_branch_id"})
        if action not in ACTION_NAMES:
            failures.append(
                {"row_index": row_index, "reason": "invalid_forced_action"}
            )
        if not source_record_digest:
            failures.append(
                {"row_index": row_index, "reason": "missing_source_record_digest"}
            )
        if not materialized_record_digest:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "missing_materialized_record_digest",
                }
            )
    duplicate_branch_actions = [
        {"branch_id": branch_id, "forced_action": action, "count": int(count)}
        for (branch_id, action), count in sorted(branch_action_counts.items())
        if count > 1
    ]
    for duplicate in duplicate_branch_actions[:32]:
        failures.append(
            {
                "reason": "duplicate_branch_forced_action",
                "branch_id": duplicate["branch_id"],
                "forced_action": duplicate["forced_action"],
                "count": duplicate["count"],
            }
        )
    failures.extend(
        _multi_branch_identity_failures(
            reason="source_materialization_identity_reused_across_branches",
            branch_ids_by_identity=source_identity_branch_ids,
        )
    )
    failures.extend(
        _multi_branch_identity_failures(
            reason="source_record_digest_reused_across_branches",
            branch_ids_by_identity=source_record_digest_branch_ids,
        )
    )
    failures.extend(
        _multi_branch_identity_failures(
            reason="materialized_record_digest_reused_across_branches",
            branch_ids_by_identity=materialized_record_digest_branch_ids,
        )
    )
    failures.extend(
        _multi_branch_identity_failures(
            reason="current_state_payload_reused_across_branches",
            branch_ids_by_identity=current_state_digest_branch_ids,
        )
    )
    failures.extend(
        _multi_identity_branch_failures(
            reason="branch_id_mixes_source_materialization_identity",
            identities_by_branch=source_identities_by_branch,
        )
    )
    failures.extend(
        _multi_identity_branch_failures(
            reason="branch_id_mixes_source_record_digest",
            identities_by_branch=source_record_digests_by_branch,
        )
    )
    failures.extend(
        _multi_identity_branch_failures(
            reason="branch_id_mixes_materialized_record_digest",
            identities_by_branch=materialized_record_digests_by_branch,
        )
    )
    failures.extend(
        _multi_identity_branch_failures(
            reason="branch_id_mixes_branch_state_digest",
            identities_by_branch=branch_state_digests_by_branch,
        )
    )
    failures.extend(
        _multi_identity_branch_failures(
            reason="branch_id_mixes_current_state_payload",
            identities_by_branch=current_state_digests_by_branch,
        )
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v178_identity_audit_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:96],
        "row_count": len(rows),
        "unique_branch_action_count": len(branch_action_counts),
        "duplicate_branch_action_count": sum(
            int(item["count"]) - 1 for item in duplicate_branch_actions
        ),
        "duplicate_branch_action_examples": duplicate_branch_actions[:16],
        "duplicate_trainable_feature_payload_count": _duplicate_member_count(
            feature_digest_counts
        ),
        "duplicate_current_state_payload_count": _duplicate_member_count(
            current_state_digest_counts
        ),
        "branch_state_digest_count": len(branch_digest_counts),
        "branch_state_digest_duplicate_count": _duplicate_member_count(
            branch_digest_counts
        ),
    }


def _multi_branch_identity_failures(
    *,
    reason: str,
    branch_ids_by_identity: Mapping[object, set[str]],
) -> list[dict[str, object]]:
    failures = []
    for identity, branch_ids in branch_ids_by_identity.items():
        if len(branch_ids) > 1:
            failures.append(
                {
                    "reason": reason,
                    "identity": repr(identity),
                    "branch_count": len(branch_ids),
                    "branch_ids": sorted(branch_ids)[:16],
                }
            )
    return failures[:32]


def _multi_identity_branch_failures(
    *,
    reason: str,
    identities_by_branch: Mapping[str, set[object]],
) -> list[dict[str, object]]:
    failures = []
    for branch_id, identities in sorted(identities_by_branch.items()):
        if len(identities) > 1:
            failures.append(
                {
                    "reason": reason,
                    "branch_id": branch_id,
                    "identity_count": len(identities),
                    "identities": sorted(repr(identity) for identity in identities)[
                        :16
                    ],
                }
            )
    return failures[:32]


def transition_row_action_mask_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    current_widths: Counter[int] = Counter()
    next_widths: Counter[int] = Counter()
    forced_still_supported = 0
    gained_counts: Counter[str] = Counter()
    lost_counts: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        action = str(row.get("forced_action") or "")
        current_mask = _complete_action_mask(_mapping(row.get("current_public_action_mask")))
        next_mask_payload = row.get("next_public_action_mask")
        next_mask = (
            _complete_action_mask(_mapping(next_mask_payload))
            if isinstance(next_mask_payload, Mapping)
            else None
        )
        current_widths.update([_mask_width(current_mask)])
        if action not in ACTION_NAMES:
            failures.append(
                {"row_index": row_index, "reason": "invalid_forced_action"}
            )
        elif current_mask.get(action) is not True:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "forced_action_not_current_mask_supported",
                    "forced_action": action,
                }
            )
        if row.get("transition_done") is True:
            if next_mask is not None:
                failures.append(
                    {"row_index": row_index, "reason": "done_transition_has_next_mask"}
                )
            continue
        if next_mask is None:
            failures.append(
                {"row_index": row_index, "reason": "next_action_mask_missing"}
            )
            continue
        next_widths.update([_mask_width(next_mask)])
        if action in ACTION_NAMES and next_mask.get(action) is True:
            forced_still_supported += 1
        for candidate in ACTION_NAMES:
            if current_mask.get(candidate) is not True and next_mask.get(candidate) is True:
                gained_counts.update([candidate])
            if current_mask.get(candidate) is True and next_mask.get(candidate) is not True:
                lost_counts.update([candidate])
    return {
        "policy": "m3_carrion_survivor_continuation_v178_action_mask_audit_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:96],
        "row_count": len(rows),
        "current_mask_width_counts": _counter_dict(current_widths),
        "next_mask_width_counts": _counter_dict(next_widths),
        "forced_action_supported_in_current_count": len(rows) - len(
            [
                failure
                for failure in failures
                if failure.get("reason") == "forced_action_not_current_mask_supported"
            ]
        ),
        "forced_action_supported_in_next_count": forced_still_supported,
        "next_mask_action_gained_counts": _action_counter_dict(gained_counts),
        "next_mask_action_lost_counts": _action_counter_dict(lost_counts),
    }


def transition_row_observation_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    current_next_deltas: list[float] = []
    current_next_changed_counts: list[int] = []
    previous_current_deltas: list[float] = []
    current_decoded = 0
    next_decoded = 0
    previous_decoded = 0
    current_next_exact = 0
    previous_current_exact = 0
    shape_counts: Counter[str] = Counter()
    encoder_counts: Counter[str] = Counter()
    for row_index, row in enumerate(rows):
        current_payload = _mapping(row.get("current_public_observation"))
        current_values, current_error = _decode_values(current_payload)
        if current_error:
            failures.append(
                {
                    "row_index": row_index,
                    "field": "current_public_observation",
                    "reason": current_error,
                }
            )
        else:
            current_decoded += 1
            shape_counts.update([_shape_key(current_payload.get("shape"))])
            encoder_counts.update([str(current_payload.get("encoder_version") or "")])
        next_payload = row.get("next_public_observation")
        next_values: list[float] | None = None
        if next_payload is not None:
            next_values, next_error = _decode_values(_mapping(next_payload))
            if next_error:
                failures.append(
                    {
                        "row_index": row_index,
                        "field": "next_public_observation",
                        "reason": next_error,
                    }
                )
            else:
                next_decoded += 1
        previous_context = _mapping(row.get("previous_same_agent_public_context"))
        previous_payload = previous_context.get("public_observation")
        previous_values: list[float] | None = None
        if previous_context.get("available") is True:
            previous_values, previous_error = _decode_values(_mapping(previous_payload))
            if previous_error:
                failures.append(
                    {
                        "row_index": row_index,
                        "field": "previous_same_agent_public_context.public_observation",
                        "reason": previous_error,
                    }
                )
            else:
                previous_decoded += 1
        if current_values is not None and next_values is not None:
            delta = _vector_delta(current_values, next_values)
            current_next_deltas.append(delta["mean_absolute_delta"])
            current_next_changed_counts.append(delta["changed_element_count"])
            if delta["changed_element_count"] == 0:
                current_next_exact += 1
        if current_values is not None and previous_values is not None:
            delta = _vector_delta(previous_values, current_values)
            previous_current_deltas.append(delta["mean_absolute_delta"])
            if delta["changed_element_count"] == 0:
                previous_current_exact += 1
    return {
        "policy": "m3_carrion_survivor_continuation_v178_observation_audit_v1",
        "passed": not failures and current_decoded == len(rows),
        "failure_count": len(failures),
        "failures": failures[:96],
        "row_count": len(rows),
        "current_observation_decoded_count": current_decoded,
        "next_observation_decoded_count": next_decoded,
        "previous_observation_decoded_count": previous_decoded,
        "current_observation_shape_counts": _counter_dict(shape_counts),
        "current_observation_encoder_counts": _counter_dict(encoder_counts),
        "current_next_exact_match_count": current_next_exact,
        "current_next_mean_absolute_delta": _number_stats(current_next_deltas),
        "current_next_changed_element_count": _integer_stats(
            current_next_changed_counts
        ),
        "previous_current_exact_match_count": previous_current_exact,
        "previous_current_mean_absolute_delta": _number_stats(
            previous_current_deltas
        ),
    }


def transition_row_coverage_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    seeds: Counter[int] = Counter()
    branches_by_seed: dict[int, set[str]] = defaultdict(set)
    actions: Counter[str] = Counter()
    actions_by_seed: dict[int, Counter[str]] = defaultdict(Counter)
    branch_actions: dict[str, set[str]] = defaultdict(set)
    failure_types: Counter[str] = Counter()
    failed_safe_actions: Counter[str] = Counter()
    source_paths: Counter[str] = Counter()
    branch_ticks: Counter[int] = Counter()
    previous_count = 0
    next_count = 0
    done_count = 0
    for row in rows:
        metadata = _mapping(row.get("metadata"))
        seed = _int(metadata.get("seed"), default=-1)
        branch_id = str(metadata.get("branch_id") or "")
        action = str(row.get("forced_action") or "")
        if seed >= 0:
            seeds.update([seed])
        if seed >= 0 and branch_id:
            branches_by_seed[seed].add(branch_id)
        if action:
            actions.update([action])
            if seed >= 0:
                actions_by_seed[seed].update([action])
            if branch_id:
                branch_actions[branch_id].add(action)
        for failure_type in _strings(metadata.get("failure_types")):
            failure_types.update([failure_type])
        failed_safe = str(metadata.get("failed_safe_action") or "")
        if failed_safe:
            failed_safe_actions.update([failed_safe])
        source_path = str(metadata.get("source_path") or "")
        if source_path:
            source_paths.update([source_path])
        branch_tick = _int(metadata.get("branch_tick"), default=-1)
        if branch_tick >= 0:
            branch_ticks.update([branch_tick])
        if _mapping(row.get("previous_same_agent_public_context")).get("available") is True:
            previous_count += 1
        if row.get("next_public_observation_available") is True:
            next_count += 1
        if row.get("transition_done") is True:
            done_count += 1
    action_widths = Counter(len(action_set) for action_set in branch_actions.values())
    return {
        "policy": "m3_carrion_survivor_continuation_v178_coverage_audit_v1",
        "row_count": len(rows),
        "seed_count": len(seeds),
        "branch_count": len(branch_actions),
        "forced_action_count": len(actions),
        "row_counts_by_seed": {
            str(seed): int(count) for seed, count in sorted(seeds.items())
        },
        "branch_counts_by_seed": {
            str(seed): len(branches)
            for seed, branches in sorted(branches_by_seed.items())
        },
        "forced_action_counts": _action_counter_dict(actions),
        "forced_actions_by_seed": {
            str(seed): _action_counter_dict(counter)
            for seed, counter in sorted(actions_by_seed.items())
        },
        "branch_forced_action_width_counts": _counter_dict(action_widths),
        "failure_type_counts": _counter_dict(failure_types),
        "failed_safe_action_counts": _action_counter_dict(failed_safe_actions),
        "source_path_count": len(source_paths),
        "source_path_row_counts": _counter_dict(source_paths),
        "branch_tick_min": min(branch_ticks) if branch_ticks else None,
        "branch_tick_max": max(branch_ticks) if branch_ticks else None,
        "rows_with_previous_same_agent_public_context": previous_count,
        "rows_with_next_public_observation": next_count,
        "transition_done_count": done_count,
        "consumed_support_seed_note": (
            "These seeds are source/provenance support for this lane, not clean "
            "promotion-heldout evidence."
        ),
    }


def transition_row_target_audit(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    requested_actions: Counter[str] = Counter()
    resolved_actions: Counter[str] = Counter()
    action_valid_counts: Counter[str] = Counter()
    resolution_valid_counts: Counter[str] = Counter()
    moved_counts: Counter[str] = Counter()
    target_terminal_count = 0
    forced_used_count = 0
    reward_by_action: dict[str, list[float]] = defaultdict(list)
    resource_by_action: dict[str, list[float]] = defaultdict(list)
    alive_values: list[int] = []
    birth_values: list[int] = []
    death_values: list[int] = []
    for row_index, row in enumerate(rows):
        action = str(row.get("forced_action") or "")
        summary = _mapping(row.get("short_horizon_public_outcome_summary"))
        if not isinstance(row.get("short_horizon_public_outcome_summary"), Mapping):
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "short_horizon_public_outcome_summary_missing",
                }
            )
            continue
        if summary.get("forced_action_used") is not True:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "forced_action_not_used",
                    "observed": summary.get("forced_action_used"),
                }
            )
        if summary.get("current_requested_action") != action:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "current_requested_action_mismatch",
                    "expected": action,
                    "observed": summary.get("current_requested_action"),
                }
            )
        if summary.get("current_resolved_action") not in ACTION_NAMES:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "current_resolved_action_invalid",
                    "observed": summary.get("current_resolved_action"),
                }
            )
        elif summary.get("current_resolved_action") != action:
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "current_resolved_action_mismatch",
                    "expected": action,
                    "observed": summary.get("current_resolved_action"),
                }
            )
        for field in (
            "current_action_valid",
            "current_resolution_action_valid",
            "current_moved",
        ):
            if not isinstance(summary.get(field), bool):
                failures.append(
                    {
                        "row_index": row_index,
                        "reason": f"{field}_not_bool",
                        "observed": summary.get(field),
                    }
                )
        for field in ("current_action_valid", "current_resolution_action_valid"):
            if isinstance(summary.get(field), bool) and summary.get(field) is not True:
                failures.append(
                    {
                        "row_index": row_index,
                        "reason": f"{field}_not_true",
                        "observed": summary.get(field),
                    }
                )
        for field in ("current_reward_total", "current_resource_gain"):
            value = _float(summary.get(field))
            if value is None:
                failures.append(
                    {
                        "row_index": row_index,
                        "reason": f"{field}_not_finite_number",
                        "observed": summary.get(field),
                    }
                )
        for field in ("alive_agents", "births", "deaths"):
            value = _int(summary.get(field), default=-1)
            if value < 0:
                failures.append(
                    {
                        "row_index": row_index,
                        "reason": f"{field}_negative_or_missing",
                        "observed": summary.get(field),
                    }
                )
        target_terminal = summary.get("target_terminal")
        if (
            not isinstance(target_terminal, Mapping)
            or not isinstance(target_terminal.get("alive"), bool)
        ):
            failures.append(
                {
                    "row_index": row_index,
                    "reason": "target_terminal_contract_invalid",
                    "observed": target_terminal,
                }
            )
        requested_actions.update([str(summary.get("current_requested_action") or "")])
        resolved_actions.update([str(summary.get("current_resolved_action") or "")])
        action_valid_counts.update([str(summary.get("current_action_valid"))])
        resolution_valid_counts.update(
            [str(summary.get("current_resolution_action_valid"))]
        )
        moved_counts.update([str(summary.get("current_moved"))])
        target_terminal_for_count = summary.get("target_terminal")
        if (
            isinstance(target_terminal_for_count, Mapping)
            and target_terminal_for_count.get("alive") is True
        ):
            target_terminal_count += 1
        if summary.get("forced_action_used") is True:
            forced_used_count += 1
        reward = _float(summary.get("current_reward_total"))
        if reward is not None:
            reward_by_action[action].append(reward)
        resource_gain = _float(summary.get("current_resource_gain"))
        if resource_gain is not None:
            resource_by_action[action].append(resource_gain)
        alive_values.append(_int(summary.get("alive_agents"), default=0))
        birth_values.append(_int(summary.get("births"), default=0))
        death_values.append(_int(summary.get("deaths"), default=0))
    return {
        "policy": "m3_carrion_survivor_continuation_v178_target_audit_v1",
        "passed": bool(rows) and not failures,
        "failure_count": len(failures),
        "failures": failures[:96],
        "row_count": len(rows),
        "forced_action_used_count": forced_used_count,
        "all_forced_actions_used": len(rows) > 0 and forced_used_count == len(rows),
        "current_requested_action_counts": _action_counter_dict(requested_actions),
        "current_resolved_action_counts": _action_counter_dict(resolved_actions),
        "current_action_valid_counts": _counter_dict(action_valid_counts),
        "current_resolution_action_valid_counts": _counter_dict(
            resolution_valid_counts
        ),
        "current_moved_counts": _counter_dict(moved_counts),
        "target_terminal_count": target_terminal_count,
        "reward_stats_by_forced_action": {
            action: _number_stats(values)
            for action, values in sorted(
                reward_by_action.items(), key=lambda item: _action_order(item[0])
            )
        },
        "resource_gain_stats_by_forced_action": {
            action: _number_stats(values)
            for action, values in sorted(
                resource_by_action.items(), key=lambda item: _action_order(item[0])
            )
        },
        "alive_agents_stats": _integer_stats(alive_values),
        "births_stats": _integer_stats(birth_values),
        "deaths_stats": _integer_stats(death_values),
    }


def transition_row_support_readiness(
    *,
    rows: Sequence[Mapping[str, object]],
    coverage_audit: Mapping[str, object],
    identity_audit: Mapping[str, object],
    min_row_count: int,
    min_seed_count: int,
    min_branch_count: int,
    min_forced_action_count: int,
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    checks = {
        "row_count": (len(rows), int(min_row_count)),
        "seed_count": (_int(coverage_audit.get("seed_count")), int(min_seed_count)),
        "branch_count": (
            _int(coverage_audit.get("branch_count")),
            int(min_branch_count),
        ),
        "forced_action_count": (
            _int(coverage_audit.get("forced_action_count")),
            int(min_forced_action_count),
        ),
    }
    for name, (observed, minimum) in checks.items():
        if observed < minimum:
            failures.append(
                {
                    "reason": f"{name}_below_minimum",
                    "observed": observed,
                    "minimum": minimum,
                }
            )
    if identity_audit.get("duplicate_branch_action_count") not in (None, 0):
        failures.append(
            {
                "reason": "duplicate_branch_action_rows",
                "observed": identity_audit.get("duplicate_branch_action_count"),
                "minimum": 0,
            }
        )
    support_gate_passed = not failures
    return {
        "policy": "m3_carrion_survivor_continuation_v178_support_readiness_v1",
        "passed": support_gate_passed,
        "failure_count": len(failures),
        "failures": failures,
        "minimums": {
            "row_count": int(min_row_count),
            "seed_count": int(min_seed_count),
            "branch_count": int(min_branch_count),
            "forced_action_count": int(min_forced_action_count),
        },
        "observed": {
            "row_count": len(rows),
            "seed_count": _int(coverage_audit.get("seed_count")),
            "branch_count": _int(coverage_audit.get("branch_count")),
            "forced_action_count": _int(coverage_audit.get("forced_action_count")),
        },
        "diagnostic_support_minimums_met": support_gate_passed,
        "authorization_scope": "diagnostic_support_thresholds_only",
        "training_authorized": False,
        "promotion_authorized": False,
    }


def transition_row_training_authorization(
    *,
    source_validation: Mapping[str, object],
    row_schema_validation: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    value_leakage_scan: Mapping[str, object],
    feature_contract_audit: Mapping[str, object],
    identity_audit: Mapping[str, object],
    action_mask_audit: Mapping[str, object],
    observation_audit: Mapping[str, object],
    target_audit: Mapping[str, object],
    default_support_readiness: Mapping[str, object],
    requested_support_minimums: Mapping[str, object],
    default_support_minimums: Mapping[str, object],
    support_thresholds_match_defaults: bool,
) -> dict[str, object]:
    checks = {
        "source_validation_passed": source_validation.get("passed") is True,
        "row_schema_validation_passed": row_schema_validation.get("passed") is True,
        "key_leakage_scan_passed": leakage_scan.get("passed") is True,
        "value_leakage_scan_passed": value_leakage_scan.get("passed") is True,
        "feature_contract_audit_passed": feature_contract_audit.get("passed")
        is True,
        "identity_audit_passed": identity_audit.get("passed") is True,
        "action_mask_audit_passed": action_mask_audit.get("passed") is True,
        "observation_audit_passed": observation_audit.get("passed") is True,
        "target_audit_passed": target_audit.get("passed") is True,
        "default_support_readiness_passed": default_support_readiness.get("passed")
        is True,
        "support_thresholds_match_defaults": bool(support_thresholds_match_defaults),
        "expected_source_report_exact_digest_provided": source_validation.get(
            "expected_source_report_exact_digest_provided"
        )
        is True,
        "expected_dataset_digest_provided": source_validation.get(
            "expected_dataset_digest_provided"
        )
        is True,
    }
    failures = [name for name, passed in checks.items() if not passed]
    authorized = not failures
    return {
        "policy": (
            "m3_carrion_survivor_continuation_v178_transition_row_training_"
            "authorization_v1"
        ),
        **checks,
        "authorized": authorized,
        "training_authorized": authorized,
        "transition_row_training_authorized": authorized,
        "next_same_lane_opt_in_training_slice_authorized": authorized,
        "support_threshold_overrides_authorize_training": False,
        "requested_support_minimums": dict(requested_support_minimums),
        "default_support_minimums": dict(default_support_minimums),
        "failure_count": len(failures),
        "failures": failures,
    }


def validate_v178_transition_row_training_authorization_report(
    report: Mapping[str, object],
    *,
    expected_exact_digest: str,
    expected_dataset_digest: str,
    expected_classification: str,
    expected_source_producer: str = V179_SOURCE_PRODUCER,
) -> dict[str, object]:
    exact_validation = exact_digest_validation_report(report)
    authorization = _mapping(report.get("training_authorization"))
    route = _mapping(report.get("route_recommendation"))
    contract = _mapping(report.get("contract"))
    dataset = _mapping(report.get("dataset"))
    source = _mapping(report.get("source_validation"))
    classification = _mapping(report.get("classification"))
    failures: list[str] = []
    checks = {
        "exact_digest_valid": exact_validation.get("passed") is True,
        "expected_exact_digest_matches": (
            str(report.get("exact_digest") or "") == str(expected_exact_digest)
        ),
        "classification_matches": (
            str(classification.get("primary") or "") == str(expected_classification)
        ),
        "dataset_digest_matches": (
            str(dataset.get("dataset_digest") or "") == str(expected_dataset_digest)
        ),
        "source_producer_matches": (
            str(source.get("source_producer") or "") == str(expected_source_producer)
        ),
        "training_authorization_field_true": (
            authorization.get("next_same_lane_opt_in_training_slice_authorized")
            is True
        ),
        "training_authorization_failures_empty": (
            list(authorization.get("failures") or []) == []
        ),
        "route_training_authorized": (
            route.get("transition_row_training_authorized") is True
        ),
        "contract_training_authorized": (
            contract.get("next_same_lane_opt_in_training_slice_authorized") is True
        ),
        "support_thresholds_match_defaults": (
            authorization.get("support_thresholds_match_defaults") is True
        ),
        "default_support_readiness_passed": (
            authorization.get("default_support_readiness_passed") is True
        ),
        "source_validation_passed": (
            authorization.get("source_validation_passed") is True
        ),
        "row_schema_validation_passed": (
            authorization.get("row_schema_validation_passed") is True
        ),
        "key_leakage_scan_passed": (
            authorization.get("key_leakage_scan_passed") is True
        ),
        "value_leakage_scan_passed": (
            authorization.get("value_leakage_scan_passed") is True
        ),
        "feature_contract_audit_passed": (
            authorization.get("feature_contract_audit_passed") is True
        ),
        "identity_audit_passed": authorization.get("identity_audit_passed")
        is True,
        "action_mask_audit_passed": (
            authorization.get("action_mask_audit_passed") is True
        ),
        "observation_audit_passed": (
            authorization.get("observation_audit_passed") is True
        ),
        "target_audit_passed": authorization.get("target_audit_passed") is True,
        "expected_source_report_exact_digest_provided": (
            authorization.get("expected_source_report_exact_digest_provided") is True
        ),
        "expected_dataset_digest_provided": (
            authorization.get("expected_dataset_digest_provided") is True
        ),
        "v178_audit_training_did_not_run": report.get("training_ran") is False,
        "v178_audit_runtime_artifact_not_created": (
            report.get("runtime_artifact_created") is False
        ),
        "v178_audit_runtime_action_selection_unchanged": (
            report.get("runtime_action_selection_changed") is False
        ),
        "v178_audit_promotion_not_authorized": (
            report.get("promotion_authorized") is False
        ),
    }
    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "policy": (
            "m3_carrion_survivor_continuation_v178_transition_row_training_"
            "authorization_report_validation_v1"
        ),
        **checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_exact_digest": expected_exact_digest,
        "observed_exact_digest": report.get("exact_digest"),
        "exact_digest_validation": exact_validation,
        "expected_classification": expected_classification,
        "observed_classification": classification.get("primary"),
        "expected_dataset_digest": expected_dataset_digest,
        "observed_dataset_digest": dataset.get("dataset_digest"),
        "expected_source_producer": expected_source_producer,
        "observed_source_producer": source.get("source_producer"),
        "training_authorization": dict(authorization),
        "route_recommendation": dict(route),
        "contract": dict(contract),
    }


def _classification(
    *,
    source_validation: Mapping[str, object],
    row_schema_validation: Mapping[str, object],
    leakage_scan: Mapping[str, object],
    value_leakage_scan: Mapping[str, object],
    feature_contract_audit: Mapping[str, object],
    identity_audit: Mapping[str, object],
    action_mask_audit: Mapping[str, object],
    observation_audit: Mapping[str, object],
    target_audit: Mapping[str, object],
    default_support_readiness: Mapping[str, object],
    support_thresholds_match_defaults: bool,
    training_authorization: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v178_transition_row_dataset_audit_"
    if source_validation.get("passed") is not True:
        return prefix + "source_invalid_closed_no_training"
    if (
        row_schema_validation.get("passed") is not True
        or leakage_scan.get("passed") is not True
        or value_leakage_scan.get("passed") is not True
        or feature_contract_audit.get("passed") is not True
        or identity_audit.get("passed") is not True
        or action_mask_audit.get("passed") is not True
        or observation_audit.get("passed") is not True
        or target_audit.get("passed") is not True
    ):
        return prefix + "dataset_contract_invalid_closed_no_training"
    if default_support_readiness.get("passed") is not True:
        return prefix + "valid_support_limited_expand_before_training"
    if not support_thresholds_match_defaults:
        return prefix + "valid_support_ready_default_threshold_recheck_required_no_training"
    if training_authorization.get("authorized") is not True:
        return prefix + "valid_support_ready_digest_pins_required_no_training"
    return prefix + "valid_support_ready_transition_row_training_authorized"


def _route_recommendation(
    *,
    classification: str,
    support_readiness: Mapping[str, object],
    default_support_readiness: Mapping[str, object],
    training_authorization: Mapping[str, object],
    source_validation: Mapping[str, object],
) -> dict[str, object]:
    support_ready_classification = classification.endswith(
        "valid_support_ready_transition_row_training_authorized"
    )
    default_support_ready = default_support_readiness.get("passed") is True
    diagnostic_support_ready = support_readiness.get("passed") is True
    training_authorized = (
        support_ready_classification
        and training_authorization.get("authorized") is True
    )
    source_producer = str(source_validation.get("source_producer") or "")
    slice_2_route = source_producer in {V183_SOURCE_PRODUCER, V185_SOURCE_PRODUCER}
    repaired_slice_2_route = source_producer == V185_SOURCE_PRODUCER
    contract_valid = support_ready_classification or classification.endswith(
        "valid_support_limited_expand_before_training"
    ) or classification.endswith(
        "valid_support_ready_default_threshold_recheck_required_no_training"
    ) or classification.endswith(
        "valid_support_ready_digest_pins_required_no_training"
    )
    if not contract_valid:
        route = (
            "repair_v185_repaired_transition_rows_before_slice_2_training"
            if repaired_slice_2_route
            else
            "repair_v183_transition_rows_before_slice_2_training"
            if slice_2_route
            else "repair_v177_transition_rows_before_capacity_work"
        )
    elif not default_support_ready:
        route = (
            "repair_or_expand_v185_repaired_transition_rows_before_slice_2_training"
            if repaired_slice_2_route
            else
            "repair_or_expand_v183_transition_rows_before_slice_2_training"
            if slice_2_route
            else "v179_expand_exact_branch_transition_rows_no_training"
        )
    elif not training_authorized:
        failures = set(training_authorization.get("failures") or ())
        if {
            "expected_source_report_exact_digest_provided",
            "expected_dataset_digest_provided",
        } & failures:
            route = (
                "rerun_v185_repaired_dataset_audit_with_expected_source_and_dataset_"
                "digests_before_slice_2_training"
                if repaired_slice_2_route
                else
                "rerun_v178_style_audit_with_expected_v183_source_and_dataset_"
                "digests_before_slice_2_training"
                if slice_2_route
                else "rerun_v178_with_expected_source_and_dataset_digests_before_"
                "training_authorization"
            )
        else:
            route = (
                "rerun_v185_repaired_dataset_audit_with_default_support_thresholds_"
                "before_slice_2_training"
                if repaired_slice_2_route
                else
                "rerun_v178_style_audit_with_default_support_thresholds_before_"
                "slice_2_training"
                if slice_2_route
                else "rerun_v178_with_default_support_thresholds_before_"
                "training_authorization"
            )
    else:
        route = (
            V186_SLICE_2_TRAINING_ROUTE
            if repaired_slice_2_route
            else
            V185_SLICE_2_TRAINING_ROUTE
            if slice_2_route
            else "v179_transition_row_policy_training_slice_opt_in"
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v178_route_recommendation_v1",
        "recommended_next_route": route,
        "source_producer": source_producer,
        "dataset_contract_valid": contract_valid,
        "support_minimums_met": default_support_ready,
        "default_support_minimums_met": default_support_ready,
        "diagnostic_support_minimums_met": diagnostic_support_ready,
        "support_thresholds_match_defaults": (
            training_authorization.get("support_thresholds_match_defaults") is True
        ),
        "transition_row_training_authorized": training_authorized,
        "training_authorized": training_authorized,
        "first_opt_in_training_slice_authorized": (
            training_authorized and not slice_2_route
        ),
        "slice_2_opt_in_training_route_authorized": (
            training_authorized and slice_2_route
        ),
        "slice_2_training_authorized": training_authorized and slice_2_route,
        "training_authorization_scope": (
            "next_same_lane_opt_in_slice_2_training"
            if training_authorized and slice_2_route
            else "next_same_lane_opt_in_training_slice"
            if training_authorized
            else "closed"
        ),
        "runtime_integration_authorized": False,
        "shadow_or_live_eval_authorized": False,
        "promotion_authorized": False,
    }


def _diagnostics_only_contract(
    *,
    transition_row_training_authorized: bool,
) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "training_allowed": False,
        "training_authorized_for_this_command": False,
        "next_same_lane_opt_in_training_slice_authorized": bool(
            transition_row_training_authorized
        ),
        "fit_allowed": False,
        "runtime_artifact_allowed": False,
        "runtime_action_change_allowed": False,
        "shadow_or_live_eval_allowed": False,
        "promotion_allowed": False,
        "gate_relaxation_allowed": False,
        "replay_viewer_schema_change_allowed": False,
        "input_rows_are_public_transition_rows": True,
        "input_rows_use_v177_compact_transition_row_schema": True,
        "source_report_can_be_v177_or_v179_transition_row_expansion": True,
        "source_report_can_be_v177_v179_or_v183_transition_support_expansion": True,
        "input_rows_are_v177_public_transition_rows": True,
        "short_horizon_outcomes_remain_diagnostic_targets_only": True,
        "default_support_thresholds_required_for_training_authorization": True,
        "support_threshold_overrides_authorize_training": False,
        "source_report_can_be_v177_v179_v183_or_v185_transition_rows": True,
        "v185_repaired_rows_remain_v177_compact_public_transition_rows": True,
        "audit_command_spends_training_slice": False,
    }


def _lifecycle_flags() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "training_ran": False,
        "training_authorized": False,
        "fit_ran": False,
        "scorer_retraining_ran": False,
        "scorer_retraining_authorized": False,
        "runtime_artifact_created": False,
        "runtime_action_selection_changed": False,
        "runtime_observation_schema_changed": False,
        "runtime_policy_changed": False,
        "shadow_eval_ran": False,
        "live_ab_ran": False,
        "live_ab_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_ran": False,
        "replay_viewer_schema_changed": False,
        "non_promoted": True,
    }


def _support_minimums(
    *,
    min_row_count: int,
    min_seed_count: int,
    min_branch_count: int,
    min_forced_action_count: int,
) -> dict[str, int]:
    return {
        "row_count": int(min_row_count),
        "seed_count": int(min_seed_count),
        "branch_count": int(min_branch_count),
        "forced_action_count": int(min_forced_action_count),
    }


def _default_support_minimums() -> dict[str, int]:
    return _support_minimums(
        min_row_count=DEFAULT_MIN_ROW_COUNT,
        min_seed_count=DEFAULT_MIN_SEED_COUNT,
        min_branch_count=DEFAULT_MIN_BRANCH_COUNT,
        min_forced_action_count=DEFAULT_MIN_FORCED_ACTION_COUNT,
    )


def _source_lifecycle_validation(
    report: Mapping[str, object],
    *,
    producer: str,
    is_v179_source: bool = False,
    is_v183_source: bool = False,
    is_v185_source: bool = False,
) -> dict[str, object]:
    failures = []
    required_true_fields = (
        "diagnostics_only",
        "non_promoted",
    )
    required_false_fields = [
        "training_ran",
        "fit_ran",
        "scorer_retraining_ran",
        "scorer_retraining_authorized",
        "runtime_artifact_created",
        "runtime_action_selection_changed",
        "runtime_observation_schema_changed",
        "runtime_policy_changed",
        "shadow_eval_ran",
        "live_ab_ran",
        "live_ab_allowed",
        "promotion_authorized",
        "gate_relaxation_ran",
        "replay_viewer_schema_changed",
    ]
    if is_v183_source or is_v185_source:
        required_false_fields.extend(
            [
                "training_artifact_created",
                "slice_2_training_consumed",
            ]
        )
        if is_v185_source:
            required_false_fields.append("training_authorized")
    else:
        required_false_fields.append("training_authorized")
    for field in required_false_fields:
        if report.get(field) is not False:
            failures.append(
                {
                    "field": field,
                    "expected": False,
                    "observed": report.get(field),
                }
            )
    for field in required_true_fields:
        if report.get(field) is not True:
            failures.append(
                {
                    "field": field,
                    "expected": True,
                    "observed": report.get(field),
                }
            )
    contract = _mapping(report.get("contract"))
    if not contract:
        failures.append(
            {"field": "contract", "expected": "mapping", "observed": report.get("contract")}
        )
    required_true_contract_fields = ["diagnostics_only"]
    if is_v183_source:
        required_true_contract_fields.extend(
            [
                "targets_v182_carrion_observed_support_zero",
                "targets_v182_broad_seed_19_regression_states",
                "uses_v182_observed_imputed_support_fields",
                "legacy_supported_prediction_is_not_strict_observed_support",
                "fresh_v178_style_audit_required_before_slice_2_training",
            ]
        )
    elif is_v185_source:
        required_true_contract_fields.extend(
            [
                "repairs_v183_target_resolution_blocker",
                "strict_target_resolution_filter_required",
                "runtime_resolution_validity_is_diagnostic_metadata_only",
                "public_action_masks_remain_trainable_features",
                "repaired_dataset_audit_required_before_slice_2_training",
            ]
        )
    else:
        required_true_contract_fields.extend(
            [
                "source_identity_metadata_only",
                "short_horizon_outcomes_are_diagnostic_targets_only",
            ]
        )
    if is_v179_source:
        required_true_contract_fields.extend(
            [
                "v172_selector_evidence_is_not_trainable_transition_rows",
                "current_and_next_public_fields_are_materialized_by_exact_replay",
            ]
        )
    elif not is_v183_source and not is_v185_source:
        required_true_contract_fields.append(
            "current_and_next_public_fields_are_dataset_inputs"
        )
    for field in required_true_contract_fields:
        if contract.get(field) is not True:
            failures.append(
                {
                    "field": f"contract.{field}",
                    "expected": True,
                    "observed": contract.get(field),
                }
            )
    required_false_contract_fields = [
        "training_allowed",
        "runtime_artifact_allowed",
        "promotion_authorized" if is_v183_source else "promotion_allowed",
        "gate_relaxation_allowed",
    ]
    if is_v183_source:
        required_false_contract_fields.extend(
            [
                "training_artifact_created",
                "slice_2_training_consumed",
                "runtime_artifact_created",
                "runtime_integration_allowed",
                "runtime_action_selection_changed",
                "default_runtime_behavior_changed",
            ]
        )
    elif is_v185_source:
        required_false_contract_fields.extend(
            [
                "training_artifact_created",
                "slice_2_training_consumed",
                "runtime_artifact_created",
                "runtime_integration_allowed",
                "runtime_action_selection_changed",
                "runtime_semantics_changed",
                "default_runtime_behavior_changed",
            ]
        )
    else:
        required_false_contract_fields.extend(
            [
                "fit_allowed",
                "runtime_action_change_allowed",
                "shadow_or_live_eval_allowed",
                "replay_viewer_schema_change_allowed",
            ]
        )
    for field in required_false_contract_fields:
        if contract.get(field) is not False:
            failures.append(
                {
                    "field": f"contract.{field}",
                    "expected": False,
                    "observed": contract.get(field),
                }
            )
    if (is_v183_source or is_v185_source) and report.get(
        "diagnostic_dataset_created"
    ) is not True:
        failures.append(
            {
                "field": "diagnostic_dataset_created",
                "expected": True,
                "observed": report.get("diagnostic_dataset_created"),
            }
        )
    return {
        "policy": (
            "m3_carrion_survivor_continuation_v178_source_lifecycle_"
            "validation_v1"
        ),
        "source_producer": producer,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
    }


def _empty_v179_replay_validation() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v178_v179_replay_validation_v1",
        "source_producer": V177_SOURCE_PRODUCER,
        "passed": True,
        "failure_count": 0,
        "failures": [],
        "not_applicable": True,
    }


def _empty_v183_expansion_validation() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v178_v183_expansion_validation_v1",
        "source_producer": V177_SOURCE_PRODUCER,
        "passed": True,
        "failure_count": 0,
        "failures": [],
        "not_applicable": True,
    }


def _empty_v185_repair_validation() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v178_v185_repair_validation_v1",
        "source_producer": V177_SOURCE_PRODUCER,
        "passed": True,
        "failure_count": 0,
        "failures": [],
        "not_applicable": True,
    }


def _v179_replay_validation(report: Mapping[str, object]) -> dict[str, object]:
    failures = []
    inputs = _mapping(report.get("inputs"))
    metrics = _mapping(report.get("metrics"))
    materialization = _mapping(report.get("branch_materialization"))
    source_validation = _mapping(report.get("source_validation"))
    if not source_validation:
        failures.append(
            {
                "field": "source_validation",
                "expected": "mapping",
                "observed": report.get("source_validation"),
            }
        )
    if source_validation.get("passed") is not True:
        failures.append(
            {
                "field": "source_validation.passed",
                "expected": True,
                "observed": source_validation.get("passed"),
            }
        )
    if source_validation.get("v177_source_digests_pinned") is not True:
        failures.append(
            {
                "field": "source_validation.v177_source_digests_pinned",
                "expected": True,
                "observed": source_validation.get("v177_source_digests_pinned"),
            }
        )
    for field in (
        "expected_v177_report_exact_digest_provided",
        "expected_v177_dataset_digest_provided",
    ):
        if source_validation.get(field) is not True:
            failures.append(
                {
                    "field": f"source_validation.{field}",
                    "expected": True,
                    "observed": source_validation.get(field),
                }
            )
    for field in (
        "expected_v177_report_exact_digest",
        "observed_v177_report_exact_digest",
        "expected_v177_dataset_digest",
        "observed_v177_dataset_digest",
    ):
        if not str(source_validation.get(field) or ""):
            failures.append(
                {
                    "field": f"source_validation.{field}",
                    "expected": "nonempty digest",
                    "observed": source_validation.get(field),
                }
            )
    if (
        source_validation.get("expected_v177_report_exact_digest")
        != source_validation.get("observed_v177_report_exact_digest")
    ):
        failures.append(
            {
                "field": "source_validation.expected_v177_report_exact_digest",
                "expected": source_validation.get(
                    "observed_v177_report_exact_digest"
                ),
                "observed": source_validation.get(
                    "expected_v177_report_exact_digest"
                ),
            }
        )
    if (
        source_validation.get("expected_v177_dataset_digest")
        != source_validation.get("observed_v177_dataset_digest")
    ):
        failures.append(
            {
                "field": "source_validation.expected_v177_dataset_digest",
                "expected": source_validation.get("observed_v177_dataset_digest"),
                "observed": source_validation.get("expected_v177_dataset_digest"),
            }
        )
    v177_exact = _mapping(source_validation.get("v177_exact_digest_validation"))
    if v177_exact.get("passed") is not True:
        failures.append(
            {
                "field": "source_validation.v177_exact_digest_validation.passed",
                "expected": True,
                "observed": v177_exact.get("passed"),
            }
        )
    if inputs.get("verify_replay") is not True:
        failures.append(
            {
                "field": "inputs.verify_replay",
                "expected": True,
                "observed": inputs.get("verify_replay"),
            }
        )
    if metrics.get("replay_verification_enabled") is not True:
        failures.append(
            {
                "field": "metrics.replay_verification_enabled",
                "expected": True,
                "observed": metrics.get("replay_verification_enabled"),
            }
        )
    if metrics.get("all_replays_verified") is not True:
        failures.append(
            {
                "field": "metrics.all_replays_verified",
                "expected": True,
                "observed": metrics.get("all_replays_verified"),
            }
        )
    if _int(metrics.get("replay_verified_row_count"), default=0) <= 0:
        failures.append(
            {
                "field": "metrics.replay_verified_row_count",
                "expected": ">0",
                "observed": metrics.get("replay_verified_row_count"),
            }
        )
    if materialization.get("passed") is not True:
        failures.append(
            {
                "field": "branch_materialization.passed",
                "expected": True,
                "observed": materialization.get("passed"),
            }
        )
    if materialization.get("exact_materialization_proven") is not True:
        failures.append(
            {
                "field": "branch_materialization.exact_materialization_proven",
                "expected": True,
                "observed": materialization.get("exact_materialization_proven"),
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v178_v179_replay_validation_v1",
        "source_producer": V179_SOURCE_PRODUCER,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
        "source_validation_passed": source_validation.get("passed"),
        "v177_source_digests_pinned": source_validation.get(
            "v177_source_digests_pinned"
        ),
        "expected_v177_report_exact_digest_provided": source_validation.get(
            "expected_v177_report_exact_digest_provided"
        ),
        "expected_v177_dataset_digest_provided": source_validation.get(
            "expected_v177_dataset_digest_provided"
        ),
        "inputs_verify_replay": inputs.get("verify_replay"),
        "replay_verification_enabled": metrics.get("replay_verification_enabled"),
        "all_replays_verified": metrics.get("all_replays_verified"),
        "replay_verified_row_count": metrics.get("replay_verified_row_count"),
        "exact_materialization_proven": materialization.get(
            "exact_materialization_proven"
        ),
    }


def _v183_expansion_validation(
    report: Mapping[str, object],
    *,
    observed_dataset_digest: str,
) -> dict[str, object]:
    failures: list[str] = []
    source_validation = _mapping(report.get("source_validation"))
    support = _mapping(report.get("support_summary"))
    metrics = _mapping(report.get("metrics"))
    materialization = _mapping(report.get("branch_materialization"))
    route = _mapping(report.get("route_recommendation"))
    dataset = _mapping(report.get("dataset"))
    upstream_true_fields = (
        "v182_schema_version_matches",
        "v182_policy_matches",
        "v182_exact_digest_valid",
        "v182_exact_digest_matches_expected",
        "v182_classification_matches_expected",
        "v182_source_validation_passed",
        "v182_routes_to_exact_support_expansion",
        "v182_training_not_run",
        "v182_slice_2_training_not_consumed",
        "v182_runtime_action_selection_unchanged",
        "v181_exact_digest_valid",
        "v181_exact_digest_matches_expected",
        "v181_classification_matches_expected",
        "v181_training_not_run",
        "v180_exact_digest_valid",
        "v180_exact_digest_matches_expected",
        "v180_classification_matches_expected",
        "v180_training_slice_1_ran",
        "v180_runtime_action_selection_unchanged",
        "v180_promotion_not_authorized",
        "v180_artifact_digest_matches_expected",
        "v180_artifact_digest_matches_report",
        "v179_exact_digest_valid",
        "v179_exact_digest_matches_expected",
        "v179_classification_matches_expected",
        "v179_source_validation_passed",
        "v179_support_summary_passed",
        "v179_dataset_digest_matches_expected",
        "v179_dataset_digest_matches_report",
        "v179_dataset_digest_matches_v180_report",
    )
    if source_validation.get("passed") is not True:
        failures.append("v183_upstream_source_validation_not_passed")
    missing_upstream_flags = [
        field for field in upstream_true_fields if source_validation.get(field) is not True
    ]
    if missing_upstream_flags:
        failures.append("v183_upstream_evidence_not_pinned")
    digest_pairs = (
        ("expected_v182_report_exact_digest", "observed_v182_report_exact_digest"),
        ("expected_v181_report_exact_digest", "observed_v181_report_exact_digest"),
        ("expected_v180_report_exact_digest", "observed_v180_report_exact_digest"),
        ("expected_v180_artifact_digest", "observed_v180_artifact_digest"),
        ("expected_v179_report_exact_digest", "observed_v179_report_exact_digest"),
        ("expected_v179_dataset_digest", "observed_v179_dataset_digest"),
    )
    mismatched_digest_pairs = []
    for expected_field, observed_field in digest_pairs:
        expected = str(source_validation.get(expected_field) or "")
        observed = str(source_validation.get(observed_field) or "")
        if not expected or not observed or expected != observed:
            mismatched_digest_pairs.append(
                {
                    "expected_field": expected_field,
                    "observed_field": observed_field,
                    "expected": expected,
                    "observed": observed,
                }
            )
    if mismatched_digest_pairs:
        failures.append("v183_upstream_digest_pin_mismatch")
    if support.get("passed") is not True:
        failures.append("v183_support_summary_not_passed")
    if support.get("v178_default_support_thresholds_met") is not True:
        failures.append("v183_default_support_thresholds_not_met")
    if metrics.get("replay_verification_enabled") is not True:
        failures.append("v183_replay_verification_not_enabled")
    if metrics.get("all_replays_verified") is not True:
        failures.append("v183_replay_verification_not_proven")
    if _int(metrics.get("replay_verified_row_count"), default=0) <= 0:
        failures.append("v183_replay_verified_row_count_missing")
    if metrics.get("all_forced_actions_used") is not True:
        failures.append("v183_forced_actions_not_used")
    if materialization.get("passed") is not True:
        failures.append("v183_materialization_not_passed")
    if materialization.get("exact_materialization_proven") is not True:
        failures.append("v183_exact_materialization_not_proven")
    if route.get("recommended_next_route") != (
        "fresh_v178_style_transition_row_dataset_audit_before_any_slice_2_training"
    ):
        failures.append("v183_route_not_fresh_v178_audit")
    if route.get("v178_style_audit_recommended") is not True:
        failures.append("v183_v178_style_audit_not_recommended")
    if route.get("slice_2_training_authorized") is not False:
        failures.append("v183_source_authorized_slice_2_training")
    if route.get("transition_row_training_authorized") is not False:
        failures.append("v183_source_authorized_transition_training")
    if str(dataset.get("dataset_digest") or "") != observed_dataset_digest:
        failures.append("v183_reported_dataset_digest_mismatch")
    if str(dataset.get("dataset_digest") or "") != EXPECTED_V183_DATASET_DIGEST:
        failures.append("v183_reported_dataset_digest_not_canonical")
    if _int(dataset.get("row_count"), default=-1) <= 0:
        failures.append("v183_reported_dataset_row_count_missing")
    return {
        "policy": "m3_carrion_survivor_continuation_v178_v183_expansion_validation_v1",
        "source_producer": V183_SOURCE_PRODUCER,
        "passed": not failures,
        "failure_count": len(sorted(set(failures))),
        "failures": sorted(set(failures)),
        "upstream_evidence_pinned": (
            source_validation.get("passed") is True
            and not missing_upstream_flags
            and not mismatched_digest_pairs
        ),
        "missing_upstream_flags": missing_upstream_flags,
        "mismatched_digest_pairs": mismatched_digest_pairs,
        "support_summary_passed": support.get("passed"),
        "v178_default_support_thresholds_met": support.get(
            "v178_default_support_thresholds_met"
        ),
        "replay_verification_enabled": metrics.get("replay_verification_enabled"),
        "all_replays_verified": metrics.get("all_replays_verified"),
        "replay_verified_row_count": metrics.get("replay_verified_row_count"),
        "all_forced_actions_used": metrics.get("all_forced_actions_used"),
        "branch_materialization_passed": materialization.get("passed"),
        "exact_materialization_proven": materialization.get(
            "exact_materialization_proven"
        ),
        "reported_dataset_digest": dataset.get("dataset_digest"),
        "observed_dataset_digest": observed_dataset_digest,
        "expected_v183_dataset_digest": EXPECTED_V183_DATASET_DIGEST,
        "recommended_next_route": route.get("recommended_next_route"),
    }


def _v185_repair_validation(
    report: Mapping[str, object],
    *,
    observed_dataset_digest: str,
) -> dict[str, object]:
    failures: list[str] = []
    source_validation = _mapping(report.get("source_validation"))
    repair = _mapping(report.get("repair_validation"))
    support = _mapping(report.get("support_summary"))
    dataset = _mapping(report.get("dataset"))
    route = _mapping(report.get("route_recommendation"))
    pre_target = _mapping(repair.get("pre_repair_target_audit"))
    repaired_target = _mapping(repair.get("repaired_target_audit"))
    expected_source_dataset_digest = str(
        source_validation.get("expected_v183_dataset_digest")
        or EXPECTED_V183_DATASET_DIGEST
    )
    if source_validation.get("passed") is not True:
        failures.append("v185_source_validation_not_passed")
    source_checks = {
        "v184_schema_version_matches",
        "v184_policy_matches",
        "v184_exact_digest_valid",
        "v184_exact_digest_matches_expected",
        "v184_classification_matches_expected",
        "v184_source_validation_passed",
        "v184_routes_to_v183_target_resolution_repair",
        "v184_source_producer_is_v183",
        "v184_dataset_digest_matches_canonical_v183",
        "v184_target_audit_failed_with_expected_count",
        "v184_training_not_run",
        "v184_slice_2_training_not_consumed",
        "v184_runtime_action_selection_unchanged",
        "v184_promotion_not_authorized",
    }
    missing_source_checks = [
        name for name in sorted(source_checks) if source_validation.get(name) is not True
    ]
    if missing_source_checks:
        failures.append("v185_v184_source_evidence_not_pinned")
    for expected_field, observed_field in (
        ("expected_v184_report_exact_digest", "observed_v184_report_exact_digest"),
        ("expected_v183_report_exact_digest", "observed_v183_report_exact_digest"),
        ("expected_v183_dataset_digest", "observed_v183_dataset_digest"),
    ):
        expected = str(source_validation.get(expected_field) or "")
        observed = str(source_validation.get(observed_field) or "")
        if not expected or not observed or expected != observed:
            failures.append("v185_source_digest_pin_mismatch")
            break
    if repair.get("passed") is not True:
        failures.append("v185_repair_validation_not_passed")
    if repair.get("strict_filter_used") is not True:
        failures.append("v185_strict_filter_not_used")
    if repair.get("backfill_used") is not False:
        failures.append("v185_backfill_used")
    if _int(repair.get("invalid_input_row_count"), default=-1) != 7:
        failures.append("v185_invalid_input_row_count_unexpected")
    if _int(repair.get("removed_or_replaced_invalid_row_count"), default=-1) != 7:
        failures.append("v185_removed_invalid_row_count_unexpected")
    if pre_target.get("passed") is not False:
        failures.append("v185_pre_repair_target_audit_not_failed")
    if _int(pre_target.get("failure_count"), default=-1) != 14:
        failures.append("v185_pre_repair_target_failure_count_unexpected")
    if repaired_target.get("passed") is not True:
        failures.append("v185_repaired_target_audit_not_passed")
    if _int(repaired_target.get("failure_count"), default=-1) != 0:
        failures.append("v185_repaired_target_failures_present")
    if repair.get("all_repaired_rows_force_resolve_to_forced_action") is not True:
        failures.append("v185_repaired_rows_do_not_force_resolve_to_forced_action")
    if (
        repair.get("runtime_resolution_validity_used_as_trainable_input")
        is not False
    ):
        failures.append("v185_runtime_resolution_validity_trainable_input")
    if support.get("passed") is not True:
        failures.append("v185_support_summary_not_passed")
    if support.get("v178_default_support_thresholds_met") is not True:
        failures.append("v185_default_support_thresholds_not_met")
    if str(dataset.get("dataset_digest") or "") != observed_dataset_digest:
        failures.append("v185_reported_dataset_digest_mismatch")
    if _int(dataset.get("row_count"), default=-1) <= 0:
        failures.append("v185_reported_dataset_row_count_missing")
    if str(dataset.get("source_dataset_digest") or "") != expected_source_dataset_digest:
        failures.append("v185_source_dataset_digest_not_canonical_v183")
    if route.get("recommended_next_route") != (
        "v185_repaired_transition_row_dataset_audit_before_slice_2_training"
    ):
        failures.append("v185_route_not_repaired_dataset_audit")
    if route.get("repaired_dataset_audit_recommended") is not True:
        failures.append("v185_repaired_dataset_audit_not_recommended")
    if route.get("slice_2_training_authorized") is not False:
        failures.append("v185_source_authorized_slice_2_training")
    if route.get("transition_row_training_authorized") is not False:
        failures.append("v185_source_authorized_transition_training")
    unique_failures = sorted(set(failures))
    return {
        "policy": "m3_carrion_survivor_continuation_v178_v185_repair_validation_v1",
        "source_producer": V185_SOURCE_PRODUCER,
        "passed": not unique_failures,
        "failure_count": len(unique_failures),
        "failures": unique_failures,
        "source_evidence_pinned": (
            source_validation.get("passed") is True
            and not missing_source_checks
        ),
        "missing_source_checks": missing_source_checks,
        "source_validation_passed": source_validation.get("passed"),
        "repair_validation_passed": repair.get("passed"),
        "strict_filter_used": repair.get("strict_filter_used"),
        "backfill_used": repair.get("backfill_used"),
        "invalid_input_row_count": repair.get("invalid_input_row_count"),
        "removed_or_replaced_invalid_row_count": repair.get(
            "removed_or_replaced_invalid_row_count"
        ),
        "pre_repair_target_failure_count": pre_target.get("failure_count"),
        "repaired_target_failure_count": repaired_target.get("failure_count"),
        "support_summary_passed": support.get("passed"),
        "v178_default_support_thresholds_met": support.get(
            "v178_default_support_thresholds_met"
        ),
        "reported_dataset_digest": dataset.get("dataset_digest"),
        "observed_dataset_digest": observed_dataset_digest,
        "source_dataset_digest": dataset.get("source_dataset_digest"),
        "expected_source_dataset_digest": expected_source_dataset_digest,
        "recommended_next_route": route.get("recommended_next_route"),
    }


def _audit_previous_context_contract(
    *,
    failures: list[dict[str, object]],
    row_index: int,
    previous: Mapping[str, object],
) -> None:
    available = previous.get("available")
    if not isinstance(available, bool):
        failures.append(
            {
                "row_index": row_index,
                "path": (
                    "trainable_public_features."
                    "previous_same_agent_public_context.available"
                ),
                "reason": "available_not_bool",
                "observed": available,
            }
        )
        return
    if available:
        _audit_encoded_observation_contract(
            failures=failures,
            row_index=row_index,
            path=(
                "trainable_public_features",
                "previous_same_agent_public_context",
                "public_observation",
            ),
            payload=previous.get("public_observation"),
            required=True,
        )
        _audit_action_mask_contract(
            failures=failures,
            row_index=row_index,
            path=(
                "trainable_public_features",
                "previous_same_agent_public_context",
                "public_action_mask",
            ),
            payload=previous.get("public_action_mask"),
            required=True,
        )
        action = previous.get("public_action")
        if action not in ACTION_NAMES:
            failures.append(
                {
                    "row_index": row_index,
                    "path": (
                        "trainable_public_features."
                        "previous_same_agent_public_context.public_action"
                    ),
                    "reason": "invalid_previous_public_action",
                    "observed": action,
                }
            )
        if not isinstance(previous.get("moved"), bool):
            failures.append(
                {
                    "row_index": row_index,
                    "path": (
                        "trainable_public_features."
                        "previous_same_agent_public_context.moved"
                    ),
                    "reason": "previous_moved_not_bool",
                    "observed": previous.get("moved"),
                }
            )
    else:
        for field in (
            "public_observation",
            "public_action_mask",
            "public_action",
            "moved",
        ):
            if previous.get(field) is not None:
                failures.append(
                    {
                        "row_index": row_index,
                        "path": (
                            "trainable_public_features."
                            f"previous_same_agent_public_context.{field}"
                        ),
                        "reason": "unavailable_previous_context_field_not_null",
                        "observed": previous.get(field),
                    }
                )


def _audit_encoded_observation_contract(
    *,
    failures: list[dict[str, object]],
    row_index: int,
    path: tuple[str, ...],
    payload: object,
    required: bool,
) -> None:
    if payload is None:
        if required:
            failures.append(
                {
                    "row_index": row_index,
                    "path": ".".join(path),
                    "reason": "encoded_observation_missing",
                }
            )
        return
    if not isinstance(payload, Mapping):
        failures.append(
            {
                "row_index": row_index,
                "path": ".".join(path),
                "reason": "encoded_observation_not_object",
            }
        )
        return
    _require_key_set(
        failures=failures,
        row_index=row_index,
        path=path,
        observed=payload,
        expected=ENCODED_OBSERVATION_INPUT_KEYS,
    )


def _audit_action_mask_contract(
    *,
    failures: list[dict[str, object]],
    row_index: int,
    path: tuple[str, ...],
    payload: object,
    required: bool,
) -> None:
    if payload is None:
        if required:
            failures.append(
                {
                    "row_index": row_index,
                    "path": ".".join(path),
                    "reason": "action_mask_missing",
                }
            )
        return
    if not isinstance(payload, Mapping):
        failures.append(
            {
                "row_index": row_index,
                "path": ".".join(path),
                "reason": "action_mask_not_object",
            }
        )
        return
    _require_key_set(
        failures=failures,
        row_index=row_index,
        path=path,
        observed=payload,
        expected=set(ACTION_NAMES),
    )
    for action in ACTION_NAMES:
        if not isinstance(payload.get(action), bool):
            failures.append(
                {
                    "row_index": row_index,
                    "path": ".".join((*path, action)),
                    "reason": "action_mask_value_not_bool",
                    "observed": payload.get(action),
                }
            )


def _require_key_set(
    *,
    failures: list[dict[str, object]],
    row_index: int,
    path: tuple[str, ...],
    observed: Mapping[str, object],
    expected: set[str],
) -> None:
    observed_keys = {str(key) for key in observed}
    missing = sorted(expected - observed_keys)
    extra = sorted(observed_keys - expected)
    if missing or extra:
        failures.append(
            {
                "row_index": row_index,
                "path": ".".join(path),
                "reason": "key_set_mismatch",
                "missing": missing,
                "extra": extra,
            }
        )


def _scan_trainable_values(
    *,
    value: object,
    row_index: int,
    path: tuple[str, ...],
    failures: list[dict[str, object]],
    parent: Mapping[str, object] | None,
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            current_path = (*path, str(key))
            if _is_opaque_encoded_observation_data(
                key=str(key),
                value=item,
                parent=value,
            ):
                continue
            _scan_trainable_values(
                value=item,
                row_index=row_index,
                path=current_path,
                failures=failures,
                parent=value,
            )
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _scan_trainable_values(
                value=item,
                row_index=row_index,
                path=(*path, str(index)),
                failures=failures,
                parent=parent,
            )
        return
    if isinstance(value, str):
        token = _matching_forbidden_value_token(value)
        if token:
            failures.append(
                {
                    "row_index": row_index,
                    "path": ".".join(path),
                    "reason": "forbidden_string_value_token",
                    "token": token,
                }
            )
        elif _looks_like_private_path_value(value):
            failures.append(
                {
                    "row_index": row_index,
                    "path": ".".join(path),
                    "reason": "path_like_string_value",
                    "token": "path",
                }
            )
    elif isinstance(value, (int, float)) and parent is not None:
        key = path[-1] if path else ""
        if key not in {"available", "moved"} and _parent_has_unexpected_keys(parent):
            failures.append(
                {
                    "row_index": row_index,
                    "path": ".".join(path),
                    "reason": "numeric_value_under_unexpected_feature_key",
                }
            )


def _is_opaque_encoded_observation_data(
    *,
    key: str,
    value: object,
    parent: Mapping[str, object],
) -> bool:
    return (
        key == "data"
        and isinstance(value, str)
        and set(str(item) for item in parent) == ENCODED_OBSERVATION_INPUT_KEYS
        and parent.get("storage_encoding") == "zlib_base64_little_endian_int16"
    )


def _matching_forbidden_value_token(text: str) -> str | None:
    lowered = text.lower()
    parts = set(re.findall(r"[a-z0-9]+", lowered))
    for token in FORBIDDEN_TRAINABLE_KEY_TOKENS:
        token_parts = tuple(part for part in token.split("_") if part)
        if len(token_parts) == 1:
            if token_parts[0] in parts:
                return token
        elif token in lowered:
            return token
    return None


def _looks_like_private_path_value(text: str) -> bool:
    if "/" in text or "\\" in text:
        return True
    return text.endswith((".json", ".jsonl", ".jsonl.gz", ".gz"))


def _parent_has_unexpected_keys(parent: Mapping[str, object]) -> bool:
    keys = {str(key) for key in parent}
    known_sets = (
        TRAINABLE_PUBLIC_FEATURE_KEYS,
        PREVIOUS_SAME_AGENT_PUBLIC_CONTEXT_KEYS,
        ENCODED_OBSERVATION_INPUT_KEYS,
        set(ACTION_NAMES),
    )
    return not any(keys <= known for known in known_sets)


def _decode_values(payload: Mapping[str, object]) -> tuple[list[float] | None, str | None]:
    try:
        return decode_observation_input(dict(payload)), None
    except (TypeError, ValueError) as exc:
        return None, str(exc)


def _vector_delta(left: Sequence[float], right: Sequence[float]) -> dict[str, object]:
    if len(left) != len(right):
        return {
            "mean_absolute_delta": 0.0,
            "changed_element_count": 0,
            "length_mismatch": True,
        }
    differences = [abs(float(a) - float(b)) for a, b in zip(left, right)]
    return {
        "mean_absolute_delta": _round(mean(differences)) if differences else 0.0,
        "changed_element_count": sum(1 for value in differences if value > 1e-9),
        "length_mismatch": False,
    }


def _complete_action_mask(value: Mapping[str, object]) -> dict[str, bool]:
    return {action: value.get(action) is True for action in ACTION_NAMES}


def _mask_width(mask: Mapping[str, bool]) -> int:
    return sum(1 for action in ACTION_NAMES if mask.get(action) is True)


def _duplicate_member_count(counter: Counter[object]) -> int:
    return sum(int(count) - 1 for count in counter.values() if count > 1)


def _strings(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [str(item) for item in value if str(item)]


def _float(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if isfinite(number) else None


def _shape_key(value: object) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "x".join(str(item) for item in value)
    return ""


def _counter_dict(counter: Counter[object]) -> dict[str, int]:
    return {str(key): int(count) for key, count in sorted(counter.items())}


def _action_counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {
        str(action): int(count)
        for action, count in sorted(counter.items(), key=lambda item: _action_order(item[0]))
        if str(action)
    }


def _number_stats(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": _round(min(values)),
        "max": _round(max(values)),
        "mean": _round(mean(values)),
    }


def _integer_stats(values: Sequence[int]) -> dict[str, object]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": _round(mean(values)),
    }


def _json_round_trip_digest(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(
        json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    )
