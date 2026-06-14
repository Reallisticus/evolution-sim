from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind import (
    carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair as v198,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit as v199,
)
from evolution_sim.mind.candidate_campaign import _int, _mapping, _round, write_json
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V200_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_REPAIR_DESIGN_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V200_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_REPAIR_DESIGN_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design_v1"
)

DEFAULT_V198_REPORT_PATH = v198.DEFAULT_OUTPUT_PATH
DEFAULT_V199_REPORT_PATH = v199.DEFAULT_OUTPUT_PATH
DEFAULT_V195_ARTIFACT_PATH = v199.DEFAULT_V195_ARTIFACT_PATH
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v200-carrion-survivor-continuation-high-specificity-action-complete-contract-repair-design.json"
)

EXPECTED_V197_COMMIT = "da8fac5"
EXPECTED_V197_REPORT_EXACT_DIGEST = (
    "8523d9cfcfbee2185ff638e463808818f29459d6ec4a51590072b15a3e341c0a"
)
EXPECTED_V197_ARCHIVE_PATH = (
    "gdrive:evolution-sim-backups/archives/"
    "20260614T180236Z-v197-coverage-abstention-repair-design.tar.zst"
)
EXPECTED_V197_ARCHIVE_SHA256 = (
    "f11dd5c9cd3eea3f30c2854320a435035277b74b4792f008cd687235ad2b2b40"
)
EXPECTED_V198_COMMIT = "029ed5d8c1b3e35dab52228a6d38fce30a084062"
EXPECTED_V198_REPORT_EXACT_DIGEST = v199.EXPECTED_V198_REPORT_EXACT_DIGEST
EXPECTED_V198_ARCHIVE_PATH = v199.EXPECTED_V198_ARCHIVE_PATH
EXPECTED_V198_ARCHIVE_SHA256 = v199.EXPECTED_V198_ARCHIVE_SHA256
EXPECTED_V199_COMMIT = "b68c4e8949a8c652dfbbb8b64b23315d563d7cc7"
EXPECTED_V199_REPORT_EXACT_DIGEST = (
    "d1111534a01d8cd36d6a8f309b6b05eb28f829c382e2cba6da1f58e011b4d04c"
)
EXPECTED_V199_ARCHIVE_PATH = (
    "gdrive:evolution-sim-backups/archives/"
    "20260614T195931Z-v199-high-specificity-action-complete-contract-audit.tar.zst"
)
EXPECTED_V199_ARCHIVE_SHA256 = (
    "38704355bb89da1a9a12c73284baeaa3f6a9f4984ea7d101bf8d249b92e58bdb"
)
EXPECTED_V199_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v199_high_specificity_action_complete_gap_confirmed_routes_to_repair_design_no_training"
)
EXPECTED_V199_ROUTE = v199.ACTION_COMPLETE_REPAIR_ROUTE
EXPECTED_V199_EVALUATED_COUNT = 54536
EXPECTED_V199_ABSENT_COUNT = 53842
EXPECTED_V199_PRESENT_COUNT = 694
EXPECTED_V199_PRESENT_INCOMPLETE_COUNT = 694
EXPECTED_V199_SUPPORT_FLOOR_FAILURE_COUNT = 0
EXPECTED_V195_ARTIFACT_DIGEST = v199.EXPECTED_V195_ARTIFACT_DIGEST

STOP_ROUTE = "stop_v200_source_pins_invalid_no_training"
V199_AUDIT_REPAIR_ROUTE = "v201_v199_action_complete_audit_repair_no_training"
SOURCE_CONTRACT_AUDIT_ROUTE = (
    "v201_high_specificity_action_complete_source_contract_audit_no_training"
)
PUBLIC_MODEL_CAPACITY_CONTRACT_ROUTE = (
    "v201_public_masked_model_capacity_harness_contract_no_training"
)
INSTRUMENTATION_REPAIR_ROUTE = (
    "v201_high_specificity_action_complete_instrumentation_repair_no_training"
)


def run_carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design(
    *,
    v198_report_path: str | Path = DEFAULT_V198_REPORT_PATH,
    v199_report_path: str | Path = DEFAULT_V199_REPORT_PATH,
    v195_artifact_path: str | Path = DEFAULT_V195_ARTIFACT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v198_report_exact_digest: str = EXPECTED_V198_REPORT_EXACT_DIGEST,
    expected_v199_report_exact_digest: str = EXPECTED_V199_REPORT_EXACT_DIGEST,
    expected_v199_classification: str = EXPECTED_V199_CLASSIFICATION,
    expected_v199_route: str = EXPECTED_V199_ROUTE,
    expected_v195_artifact_digest: str = EXPECTED_V195_ARTIFACT_DIGEST,
) -> dict[str, object]:
    v198_report, v198_load_error = _load_optional_json_report(v198_report_path)
    v199_report = load_json_report(v199_report_path)
    v195_artifact = load_json_report(v195_artifact_path)
    source_validation = validate_v200_source_pins(
        v198_report=v198_report,
        v198_load_error=v198_load_error,
        v199_report=v199_report,
        v195_artifact=v195_artifact,
        expected_v198_report_exact_digest=expected_v198_report_exact_digest,
        expected_v199_report_exact_digest=expected_v199_report_exact_digest,
        expected_v199_classification=expected_v199_classification,
        expected_v199_route=expected_v199_route,
        expected_v195_artifact_digest=expected_v195_artifact_digest,
    )
    missing_action_matrix = missing_action_matrix_from_v199(v199_report)
    facts = assess_v199_action_complete_facts(v199_report)
    repair_design = action_complete_repair_design(
        v199_report=v199_report,
        missing_action_matrix=missing_action_matrix,
    )
    route_decision = route_decision_for_v200(
        source_validation=source_validation,
        facts=facts,
        repair_design=repair_design,
    )
    classification = classification_for_v200(
        source_validation=source_validation,
        route_decision=route_decision,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V200_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_REPAIR_DESIGN_SCHEMA_VERSION
        ),
        "policy": (
            M3_CARRION_SURVIVOR_CONTINUATION_V200_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_REPAIR_DESIGN_POLICY
        ),
        "contract": contract(
            expected_v199_report_exact_digest=expected_v199_report_exact_digest,
            expected_v199_route=expected_v199_route,
        ),
        "historical_evidence_checkpoint": historical_evidence_checkpoint(),
        "inputs": {
            "v198_report": str(v198_report_path),
            "v199_report": str(v199_report_path),
            "v195_artifact": str(v195_artifact_path),
            "expected_v198_report_exact_digest": expected_v198_report_exact_digest,
            "expected_v199_report_exact_digest": expected_v199_report_exact_digest,
            "expected_v199_classification": expected_v199_classification,
            "expected_v199_route": expected_v199_route,
            "expected_v195_artifact_digest": expected_v195_artifact_digest,
        },
        "source_pin_validation": source_validation,
        "v199_fact_assessment": facts,
        "missing_action_matrix": missing_action_matrix,
        "action_complete_repair_design": repair_design,
        "route_decision": route_decision,
        "classification": {
            "primary": classification,
            "labels": [
                classification,
                "diagnostics_only",
                "high_specificity_action_complete_contract_repair_design",
                "no_training",
                "no_slice_4_consumption",
                "no_runtime_integration",
                "no_gate_relaxation",
                "no_support_generation",
                "no_support_expansion",
                "no_promotion",
            ],
        },
        **lifecycle_flags(),
    }
    report["exact_digest"] = digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v200_source_pins(
    *,
    v198_report: Mapping[str, object],
    v198_load_error: str | None = None,
    v199_report: Mapping[str, object],
    v195_artifact: Mapping[str, object],
    expected_v198_report_exact_digest: str,
    expected_v199_report_exact_digest: str,
    expected_v199_classification: str,
    expected_v199_route: str,
    expected_v195_artifact_digest: str,
) -> dict[str, object]:
    v198_exact_validation = exact_digest_validation_report(v198_report)
    v199_exact_validation = exact_digest_validation_report(v199_report)
    v198_source = _mapping(v198_report.get("source_pin_validation"))
    v198_source_checks = _mapping(v198_source.get("checks"))
    classification = _mapping(v199_report.get("classification"))
    route = _mapping(v199_report.get("route_decision"))
    source = _mapping(v199_report.get("source_pin_validation"))
    artifact_digest = stable_payload_digest(v195_artifact)
    lifecycle_keys = (
        "training_ran",
        "fit_ran",
        "training_artifact_created",
        "slice_4_training_started",
        "slice_4_training_consumed",
        "runtime_artifact_created",
        "runtime_integration_ran",
        "runtime_action_selection_changed",
        "runtime_policy_changed",
        "gate_relaxation_ran",
        "gate_relaxation_allowed",
        "support_generation_ran",
        "support_expansion_ran",
        "promotion_authorized",
    )
    checks = {
        "v198_report_loaded": v198_load_error is None,
        "v198_report_schema_matches": (
            v198_report.get("schema_version")
            == v198.M3_CARRION_SURVIVOR_CONTINUATION_V198_HIGH_SPECIFICITY_COVERAGE_OR_MODEL_CAPACITY_REPAIR_SCHEMA_VERSION
        ),
        "v198_report_policy_matches": (
            v198_report.get("policy")
            == v198.M3_CARRION_SURVIVOR_CONTINUATION_V198_HIGH_SPECIFICITY_COVERAGE_OR_MODEL_CAPACITY_REPAIR_POLICY
        ),
        "v198_report_exact_digest_valid": (
            v198_exact_validation.get("passed") is True
        ),
        "v198_report_exact_digest_matches_expected": (
            v198_report.get("exact_digest") == expected_v198_report_exact_digest
        ),
        "v198_source_pin_validation_passed": v198_source.get("passed") is True,
        "v198_observed_v197_report_digest_matches_expected": (
            v198_source.get("observed_v197_report_exact_digest")
            == EXPECTED_V197_REPORT_EXACT_DIGEST
        ),
        "v198_observed_v197_route_matches_expected": (
            v198_source.get("observed_v197_route") == v198.EXPECTED_V197_ROUTE
        ),
        "v198_inherited_v196_pin_validated": (
            v198_source_checks.get("v197_inherited_v196_digest_matches") is True
        ),
        "v198_inherited_v195_report_pin_validated": (
            v198_source_checks.get("v197_inherited_v195_report_digest_matches")
            is True
        ),
        "v198_inherited_v195_artifact_pin_validated": (
            v198_source_checks.get("v197_inherited_v195_artifact_digest_matches")
            is True
        ),
        "v199_report_schema_matches": (
            v199_report.get("schema_version")
            == v199.M3_CARRION_SURVIVOR_CONTINUATION_V199_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_AUDIT_SCHEMA_VERSION
        ),
        "v199_report_policy_matches": (
            v199_report.get("policy")
            == v199.M3_CARRION_SURVIVOR_CONTINUATION_V199_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_AUDIT_POLICY
        ),
        "v199_report_exact_digest_valid": v199_exact_validation.get("passed") is True,
        "v199_report_exact_digest_matches_expected": (
            v199_report.get("exact_digest") == expected_v199_report_exact_digest
        ),
        "v199_classification_matches_expected": (
            classification.get("primary") == expected_v199_classification
        ),
        "v199_route_matches_expected": (
            route.get("selected_route") == expected_v199_route
            and route.get("recommended_next_route") == expected_v199_route
        ),
        "v199_source_pin_validation_passed": source.get("passed") is True,
        "v199_lifecycle_closed": all(
            v199_report.get(key) is False for key in lifecycle_keys
        )
        and v199_report.get("non_promoted") is True,
        "v197_commit_pin_recorded": EXPECTED_V197_COMMIT == "da8fac5",
        "v198_digest_pin_matches_v199_source": (
            source.get("expected_v198_report_exact_digest")
            == expected_v198_report_exact_digest
        ),
        "v199_commit_pin_matches_expected": (
            EXPECTED_V199_COMMIT
            == "b68c4e8949a8c652dfbbb8b64b23315d563d7cc7"
        ),
        "v199_archive_pin_matches_expected": (
            EXPECTED_V199_ARCHIVE_PATH
            == "gdrive:evolution-sim-backups/archives/20260614T195931Z-v199-high-specificity-action-complete-contract-audit.tar.zst"
            and EXPECTED_V199_ARCHIVE_SHA256
            == "38704355bb89da1a9a12c73284baeaa3f6a9f4984ea7d101bf8d249b92e58bdb"
        ),
        "v195_artifact_digest_matches_expected": (
            artifact_digest == expected_v195_artifact_digest
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v200_source_pin_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "v198_report_load_error": v198_load_error,
        "expected_v197": {
            "commit": EXPECTED_V197_COMMIT,
            "report_exact_digest": EXPECTED_V197_REPORT_EXACT_DIGEST,
            "archive": EXPECTED_V197_ARCHIVE_PATH,
            "archive_sha256": EXPECTED_V197_ARCHIVE_SHA256,
        },
        "expected_v198": {
            "commit": EXPECTED_V198_COMMIT,
            "report_exact_digest": expected_v198_report_exact_digest,
            "archive": EXPECTED_V198_ARCHIVE_PATH,
            "archive_sha256": EXPECTED_V198_ARCHIVE_SHA256,
        },
        "observed_v198_report_exact_digest": v198_report.get("exact_digest"),
        "v198_report_exact_digest_validation": v198_exact_validation,
        "observed_v198_v197_report_exact_digest": v198_source.get(
            "observed_v197_report_exact_digest"
        ),
        "observed_v198_v197_route": v198_source.get("observed_v197_route"),
        "expected_v199": {
            "commit": EXPECTED_V199_COMMIT,
            "report_exact_digest": expected_v199_report_exact_digest,
            "classification": expected_v199_classification,
            "route": expected_v199_route,
            "archive": EXPECTED_V199_ARCHIVE_PATH,
            "archive_sha256": EXPECTED_V199_ARCHIVE_SHA256,
        },
        "observed_v199_report_exact_digest": v199_report.get("exact_digest"),
        "v199_report_exact_digest_validation": v199_exact_validation,
        "observed_v199_classification": classification.get("primary"),
        "observed_v199_route": route.get("selected_route"),
        "expected_v195_artifact_digest": expected_v195_artifact_digest,
        "observed_v195_artifact_digest": artifact_digest,
        "checks": checks,
    }


def _load_optional_json_report(path: str | Path) -> tuple[dict[str, object], str | None]:
    try:
        return load_json_report(path), None
    except (OSError, ValueError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def assess_v199_action_complete_facts(
    v199_report: Mapping[str, object],
) -> dict[str, object]:
    audit = _mapping(v199_report.get("action_complete_audit"))
    combined = _mapping(audit.get("combined"))
    evaluated = _int(combined.get("high_specificity_candidate_evaluated_count"))
    absent = _int(combined.get("key_absent_count"))
    present = _int(combined.get("key_present_count"))
    incomplete = _int(combined.get("present_but_action_incomplete_count"))
    floor_failed = _int(
        combined.get("present_complete_but_observed_support_floor_failed_count")
    )
    runtime_changes = _int(combined.get("runtime_action_selection_changed_count"))
    facts_consistent = (
        evaluated == EXPECTED_V199_EVALUATED_COUNT
        and absent == EXPECTED_V199_ABSENT_COUNT
        and present == EXPECTED_V199_PRESENT_COUNT
        and incomplete == EXPECTED_V199_PRESENT_INCOMPLETE_COUNT
        and floor_failed == EXPECTED_V199_SUPPORT_FLOOR_FAILURE_COUNT
        and evaluated == absent + present
        and present == incomplete + floor_failed
        and runtime_changes == 0
    )
    instrumentation_present = (
        audit.get("ran") is True
        and audit.get("diagnostics_provenance") == "real_replay"
        and _int(combined.get("candidate_key_action_coverage_count")) > 0
    )
    artifact_can_satisfy = (
        present > 0
        and incomplete == 0
        and floor_failed == 0
        and _int(combined.get("imputed_valid_action_score_count")) == 0
    )
    public_overlap_too_sparse = present == 0
    primary_blocker = "existing_artifact_action_incomplete_for_current_valid_actions"
    if not facts_consistent:
        primary_blocker = "v199_facts_inconsistent"
    elif not instrumentation_present:
        primary_blocker = "instrumentation_missing"
    elif public_overlap_too_sparse:
        primary_blocker = "high_specificity_public_state_overlap_too_sparse"
    elif artifact_can_satisfy:
        primary_blocker = "existing_artifact_selection_probe_only_possible"
    return {
        "policy": "m3_carrion_survivor_continuation_v200_v199_fact_assessment_v1",
        "facts_consistent": facts_consistent,
        "instrumentation_present": instrumentation_present,
        "existing_artifact_can_satisfy_action_complete_contract": artifact_can_satisfy,
        "high_specificity_public_state_overlap_too_sparse_for_source_contract": (
            public_overlap_too_sparse
        ),
        "primary_blocker": primary_blocker,
        "counts": {
            "high_specificity_candidate_evaluated_count": evaluated,
            "key_absent_count": absent,
            "key_present_count": present,
            "present_but_action_incomplete_count": incomplete,
            "present_complete_but_observed_support_floor_failed_count": floor_failed,
            "candidate_key_action_coverage_count": _int(
                combined.get("candidate_key_action_coverage_count")
            ),
            "runtime_action_selection_changed_count": runtime_changes,
        },
    }


def missing_action_matrix_from_v199(
    v199_report: Mapping[str, object],
) -> dict[str, object]:
    combined = _mapping(_mapping(v199_report.get("action_complete_audit")).get("combined"))
    by_seed = _mapping(combined.get("by_seed"))
    rows: list[dict[str, object]] = []
    for seed_key, seed_stats in sorted(by_seed.items()):
        stats = _mapping(seed_stats)
        scope, _, seed = str(seed_key).partition(":")
        rows.append(
            {
                "scope": scope,
                "seed": int(seed) if seed else None,
                "key_absent_count": _int(stats.get("key_absent_count")),
                "key_present_count": _int(stats.get("key_present_count")),
                "present_but_action_incomplete_count": _int(
                    stats.get("present_but_action_incomplete_count")
                ),
                "missing_current_valid_action_counts": dict(
                    sorted(
                        {
                            str(action): _int(count)
                            for action, count in _mapping(
                                stats.get("missing_current_valid_action_counts")
                            ).items()
                        }.items()
                    )
                ),
            }
        )
    return {
        "policy": "m3_carrion_survivor_continuation_v200_missing_action_matrix_from_v199_v1",
        "source": "v199 action_complete_audit.combined.by_seed",
        "combined_missing_current_valid_action_counts": dict(
            sorted(
                {
                    str(action): _int(count)
                    for action, count in _mapping(
                        combined.get("missing_current_valid_action_counts")
                    ).items()
                }.items()
            )
        ),
        "rows": rows,
    }


def action_complete_repair_design(
    *,
    v199_report: Mapping[str, object],
    missing_action_matrix: Mapping[str, object],
) -> dict[str, object]:
    combined = _mapping(_mapping(v199_report.get("action_complete_audit")).get("combined"))
    incomplete = _int(combined.get("present_but_action_incomplete_count"))
    present = _int(combined.get("key_present_count"))
    absent = _int(combined.get("key_absent_count"))
    return {
        "policy": "m3_carrion_survivor_continuation_v200_action_complete_repair_design_v1",
        "selected_repair_class": "source_contract_audit_for_missing_current_valid_actions",
        "repair_options": {
            "existing_artifact_selection_or_probe_only": {
                "viable": False,
                "reason": (
                    f"v199 found {incomplete} present high-specific candidates and "
                    "all were missing one or more current-valid actions"
                ),
            },
            "source_contract_audit_for_missing_current_valid_actions": {
                "viable": present > 0 and incomplete > 0,
                "selected": True,
                "reason": (
                    "High-specific public states overlap live replay, but the "
                    "existing artifact lacks complete current-valid action support."
                ),
            },
            "public_masked_model_or_sequence_capacity_harness_contract": {
                "viable": False,
                "reason": (
                    "Deferred until a source-contract audit proves high-specific "
                    "public-state rows cannot be made action-complete without "
                    "unsafe imputation or leakage."
                ),
            },
            "instrumentation_repair": {
                "viable": False,
                "reason": "v199 proved real replay provenance and emitted candidate-key rows",
            },
            "closed_no_training": {
                "viable": False,
                "reason": "pins and facts validate, so v200 can select one repair design",
            },
        },
        "future_dataset_or_harness_must_prove": {
            "public_observation_identity": True,
            "public_action_mask_identity": True,
            "high_specificity_source_key": True,
            "current_valid_actions": True,
            "observed_support_count_per_action": True,
            "no_private_world_state": True,
            "no_fixture_identity_or_seed_leakage": True,
            "real_replay_provenance": True,
            "source_report_and_dataset_exact_digest_pins": True,
            "closed_lifecycle_before_training_authorization": True,
        },
        "unsafe_non_authorizing_patterns": {
            "lower_specificity_imputation_for_missing_current_valid_actions": (
                "unsafe_non_authorizing"
            ),
            "default_stay_fallback_for_misses": "unsafe_non_authorizing",
            "mock_or_override_provenance": "unsafe_non_authorizing",
            "blind_support_generation_or_expansion": "unsafe_non_authorizing",
            "gate_relaxation": "unsafe_non_authorizing",
        },
        "v199_missing_action_matrix": missing_action_matrix,
        "v199_present_key_count": present,
        "v199_absent_key_count": absent,
        "v199_present_but_action_incomplete_count": incomplete,
    }


def route_decision_for_v200(
    *,
    source_validation: Mapping[str, object],
    facts: Mapping[str, object],
    repair_design: Mapping[str, object],
) -> dict[str, object]:
    if source_validation.get("passed") is not True:
        route = STOP_ROUTE
    elif facts.get("facts_consistent") is not True:
        route = V199_AUDIT_REPAIR_ROUTE
    elif facts.get("instrumentation_present") is not True:
        route = INSTRUMENTATION_REPAIR_ROUTE
    elif (
        facts.get("high_specificity_public_state_overlap_too_sparse_for_source_contract")
        is True
    ):
        route = PUBLIC_MODEL_CAPACITY_CONTRACT_ROUTE
    elif (
        repair_design.get("selected_repair_class")
        == "source_contract_audit_for_missing_current_valid_actions"
    ):
        route = SOURCE_CONTRACT_AUDIT_ROUTE
    else:
        route = V199_AUDIT_REPAIR_ROUTE
    return {
        "policy": "m3_carrion_survivor_continuation_v200_route_decision_v1",
        "recommended_next_route": route,
        "selected_route": route,
        "exactly_one_next_route_recommended": True,
        "source_pins_valid": source_validation.get("passed") is True,
        "primary_blocker": facts.get("primary_blocker"),
        "selected_repair_class": repair_design.get("selected_repair_class"),
        "direct_slice_4_training_allowed": False,
        "future_explicit_slice_4_training_route_authorized": False,
        "slice_4_training_consumed": False,
        "runtime_integration_allowed": False,
        "runtime_action_selection_change_allowed": False,
        "gate_relaxation_allowed": False,
        "support_generation_allowed": False,
        "support_expansion_allowed": False,
        "promotion_authorized": False,
        "rationale": route_rationale(route),
    }


def classification_for_v200(
    *,
    source_validation: Mapping[str, object],
    route_decision: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v200_"
    if source_validation.get("passed") is not True:
        return prefix + "source_pins_invalid_closed_no_training"
    route = str(route_decision.get("selected_route") or "")
    if route == SOURCE_CONTRACT_AUDIT_ROUTE:
        return (
            prefix
            + "existing_artifact_action_incomplete_routes_to_source_contract_audit_no_training"
        )
    if route == PUBLIC_MODEL_CAPACITY_CONTRACT_ROUTE:
        return prefix + "public_masked_model_capacity_harness_contract_no_training"
    if route == INSTRUMENTATION_REPAIR_ROUTE:
        return prefix + "instrumentation_repair_no_training"
    if route == V199_AUDIT_REPAIR_ROUTE:
        return prefix + "v199_action_complete_audit_repair_no_training"
    return prefix + "closed_no_training"


def contract(
    *,
    expected_v199_report_exact_digest: str,
    expected_v199_route: str,
) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "training_allowed": False,
        "slice_4_training_consumption_allowed": False,
        "runtime_artifact_creation_allowed": False,
        "runtime_integration_allowed": False,
        "runtime_action_selection_change_allowed": False,
        "runtime_policy_change_allowed": False,
        "gate_relaxation_allowed": False,
        "support_generation_allowed": False,
        "support_expansion_allowed": False,
        "promotion_authorized": False,
        "requires_v199_report_digest": expected_v199_report_exact_digest,
        "requires_v199_route": expected_v199_route,
        "route_logic": {
            "source_pin_failure": STOP_ROUTE,
            "v199_fact_inconsistency": V199_AUDIT_REPAIR_ROUTE,
            "existing_artifact_action_incomplete": SOURCE_CONTRACT_AUDIT_ROUTE,
            "high_specificity_public_state_overlap_too_sparse": (
                PUBLIC_MODEL_CAPACITY_CONTRACT_ROUTE
            ),
            "instrumentation_missing": INSTRUMENTATION_REPAIR_ROUTE,
            "no_direct_slice_4_training_route": True,
        },
    }


def historical_evidence_checkpoint() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v200_historical_evidence_checkpoint_v1",
        "v53_v63": (
            "scalar IQL coefficient/prior/action-distribution/global-bias/"
            "extraction tuning is closed"
        ),
        "v154_v176": (
            "tiny support-archive / nearest-neighbor scorer loop is strategically "
            "saturated"
        ),
        "v180_v186_v195": "training slices failed and are not promotion evidence",
        "v196": "low-specificity artifact coverage collapsed to eat with miss abstention",
        "v197": "low-specificity overrides were blocked without training",
        "v198": "high-specific keys exist but none were complete for current valid actions",
        "v199": (
            "all 694 present high-specific candidate occurrences were "
            "present-but-action-incomplete; v200 designs the no-training repair "
            "contract"
        ),
    }


def lifecycle_flags() -> dict[str, object]:
    return {
        "training_ran": False,
        "fit_ran": False,
        "training_artifact_created": False,
        "slice_4_training_started": False,
        "slice_4_training_consumed": False,
        "runtime_artifact_created": False,
        "runtime_integration_ran": False,
        "runtime_action_selection_changed": False,
        "runtime_policy_changed": False,
        "gate_relaxation_ran": False,
        "gate_relaxation_allowed": False,
        "support_generation_ran": False,
        "support_expansion_ran": False,
        "promotion_authorized": False,
        "non_promoted": True,
    }


def digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)


def route_rationale(route: str) -> str:
    if route == SOURCE_CONTRACT_AUDIT_ROUTE:
        return (
            "The existing artifact cannot satisfy the high-specificity "
            "action-complete contract because present keys miss current-valid "
            "actions. The next route audits the source contract before any "
            "dataset, harness, or future training can be considered."
        )
    if route == PUBLIC_MODEL_CAPACITY_CONTRACT_ROUTE:
        return (
            "High-specific public-state overlap is too sparse for a source "
            "contract repair, so the next no-training route is a public masked "
            "model/sequence capacity harness contract."
        )
    if route == INSTRUMENTATION_REPAIR_ROUTE:
        return "Required v199 instrumentation/provenance is missing."
    if route == V199_AUDIT_REPAIR_ROUTE:
        return "v199 facts are inconsistent and must be repaired before design work."
    if route == STOP_ROUTE:
        return "Required v197/v198/v199 source pins or lifecycle facts failed."
    return "Closed no-training route."
