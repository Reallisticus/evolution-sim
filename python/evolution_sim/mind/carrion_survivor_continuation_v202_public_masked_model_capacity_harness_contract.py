from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind import (
    carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair as v198,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit as v199,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design as v200,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit as v201,
)
from evolution_sim.mind.candidate_campaign import _int, _mapping, _round, write_json
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V202_PUBLIC_MASKED_MODEL_CAPACITY_HARNESS_CONTRACT_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V202_PUBLIC_MASKED_MODEL_CAPACITY_HARNESS_CONTRACT_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract_v1"
)

DEFAULT_V198_REPORT_PATH = v198.DEFAULT_OUTPUT_PATH
DEFAULT_V199_REPORT_PATH = v199.DEFAULT_OUTPUT_PATH
DEFAULT_V200_REPORT_PATH = v200.DEFAULT_OUTPUT_PATH
DEFAULT_V201_REPORT_PATH = v201.DEFAULT_OUTPUT_PATH
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v202-carrion-survivor-continuation-public-masked-model-capacity-harness-contract.json"
)

EXPECTED_V197_COMMIT = v201.EXPECTED_V197_COMMIT
EXPECTED_V197_REPORT_EXACT_DIGEST = v201.EXPECTED_V197_REPORT_EXACT_DIGEST
EXPECTED_V197_ARCHIVE_PATH = v201.EXPECTED_V197_ARCHIVE_PATH
EXPECTED_V197_ARCHIVE_SHA256 = v201.EXPECTED_V197_ARCHIVE_SHA256
EXPECTED_V198_COMMIT = v201.EXPECTED_V198_COMMIT
EXPECTED_V198_REPORT_EXACT_DIGEST = v201.EXPECTED_V198_REPORT_EXACT_DIGEST
EXPECTED_V198_ARCHIVE_PATH = v201.EXPECTED_V198_ARCHIVE_PATH
EXPECTED_V198_ARCHIVE_SHA256 = v201.EXPECTED_V198_ARCHIVE_SHA256
EXPECTED_V198_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v198_high_specificity_action_complete_or_support_floor_gap_routes_to_contract_audit_no_training"
)
EXPECTED_V198_ROUTE = "v199_high_specificity_action_complete_contract_audit_no_training"
EXPECTED_V199_COMMIT = v201.EXPECTED_V199_COMMIT
EXPECTED_V199_REPORT_EXACT_DIGEST = v201.EXPECTED_V199_REPORT_EXACT_DIGEST
EXPECTED_V199_ARCHIVE_PATH = v201.EXPECTED_V199_ARCHIVE_PATH
EXPECTED_V199_ARCHIVE_SHA256 = v201.EXPECTED_V199_ARCHIVE_SHA256
EXPECTED_V199_CLASSIFICATION = v201.EXPECTED_V199_CLASSIFICATION
EXPECTED_V199_ROUTE = v201.EXPECTED_V199_ROUTE
EXPECTED_V200_COMMIT = v201.EXPECTED_V200_COMMIT
EXPECTED_V200_REPORT_EXACT_DIGEST = v201.EXPECTED_V200_REPORT_EXACT_DIGEST
EXPECTED_V200_ARCHIVE_PATH = v201.EXPECTED_V200_ARCHIVE_PATH
EXPECTED_V200_ARCHIVE_SHA256 = v201.EXPECTED_V200_ARCHIVE_SHA256
EXPECTED_V200_CLASSIFICATION = v201.EXPECTED_V200_CLASSIFICATION
EXPECTED_V200_ROUTE = v201.EXPECTED_V200_ROUTE
EXPECTED_V201_COMMIT = "18c262f51467d8c9b4654232d4eb3dbfb40556d7"
EXPECTED_V201_REPORT_EXACT_DIGEST = (
    "ed401b7a4e5ef5ae572040d6a9973e23b51951f2bd273e48797c094132af94d2"
)
EXPECTED_V201_ARCHIVE_PATH = (
    "gdrive:evolution-sim-backups/archives/"
    "20260614T205238Z-v201-high-specificity-action-complete-source-contract-audit.tar.zst"
)
EXPECTED_V201_ARCHIVE_SHA256 = (
    "8502ba0adb72b8052498fb1ce630dc73a0baa06c02c415af0443a1c7e97d1162"
)
EXPECTED_V201_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v201_lower_specificity_only_evidence_dominates_routes_to_public_masked_model_capacity_harness_contract_no_training"
)
EXPECTED_V201_ROUTE = v201.PUBLIC_MODEL_CAPACITY_ROUTE

EXPECTED_V201_MISSING_CURRENT_VALID_ACTION_DEMANDS = 2714
EXPECTED_V201_SAME_HIGH_SPECIFIC_ACTION_SUPPORT = 0
EXPECTED_V201_SAME_HIGH_SPECIFIC_BELOW_FLOOR_SUPPORT = 0
EXPECTED_V201_LOWER_SPECIFICITY_ONLY_DEMANDS = 2362
EXPECTED_V201_LOWER_SPECIFICITY_ONLY_SHARE = 0.870302
EXPECTED_V201_ABSENT_AT_SAME_AND_KNOWN_LOWER_KEYS = 352
EXPECTED_V201_CANDIDATE_KEY_ACTION_COVERAGE_COUNT = 215
EXPECTED_V201_ARTIFACT_FEATURE_KEY_COUNT = 8575
TRAINING_SLICES_CONSUMED = 3
TRAINING_SLICE_BUDGET = 10

STOP_ROUTE = "v203_v202_source_pin_repair_no_training"
V201_RECONCILIATION_ROUTE = "v203_v201_contract_reconciliation_no_training"
INTERFACE_GAP_REPAIR_ROUTE = (
    "v203_public_masked_model_capacity_interface_gap_repair_no_training"
)
LEAKAGE_CONTRACT_REPAIR_ROUTE = (
    "v203_public_feature_leakage_contract_repair_no_training"
)
HARNESS_SCAFFOLD_ROUTE = (
    "v203_public_masked_model_capacity_harness_scaffold_no_training"
)


def run_carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract(
    *,
    v198_report_path: str | Path = DEFAULT_V198_REPORT_PATH,
    v199_report_path: str | Path = DEFAULT_V199_REPORT_PATH,
    v200_report_path: str | Path = DEFAULT_V200_REPORT_PATH,
    v201_report_path: str | Path = DEFAULT_V201_REPORT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v198_report_exact_digest: str = EXPECTED_V198_REPORT_EXACT_DIGEST,
    expected_v199_report_exact_digest: str = EXPECTED_V199_REPORT_EXACT_DIGEST,
    expected_v200_report_exact_digest: str = EXPECTED_V200_REPORT_EXACT_DIGEST,
    expected_v201_report_exact_digest: str = EXPECTED_V201_REPORT_EXACT_DIGEST,
) -> dict[str, object]:
    v198_report, v198_load_error = _load_optional_json_report(v198_report_path)
    v199_report, v199_load_error = _load_optional_json_report(v199_report_path)
    v200_report, v200_load_error = _load_optional_json_report(v200_report_path)
    v201_report, v201_load_error = _load_optional_json_report(v201_report_path)
    source_validation = validate_v202_source_pins(
        v198_report=v198_report,
        v198_load_error=v198_load_error,
        v199_report=v199_report,
        v199_load_error=v199_load_error,
        v200_report=v200_report,
        v200_load_error=v200_load_error,
        v201_report=v201_report,
        v201_load_error=v201_load_error,
        expected_v198_report_exact_digest=expected_v198_report_exact_digest,
        expected_v199_report_exact_digest=expected_v199_report_exact_digest,
        expected_v200_report_exact_digest=expected_v200_report_exact_digest,
        expected_v201_report_exact_digest=expected_v201_report_exact_digest,
    )
    v201_facts = assess_v201_required_facts(v201_report)
    harness_contract = public_masked_model_capacity_harness_contract()
    route_decision = route_decision_for_v202(
        source_validation=source_validation,
        v201_facts=v201_facts,
        harness_contract=harness_contract,
    )
    classification = classification_for_v202(
        source_validation=source_validation,
        route_decision=route_decision,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V202_PUBLIC_MASKED_MODEL_CAPACITY_HARNESS_CONTRACT_SCHEMA_VERSION
        ),
        "policy": (
            M3_CARRION_SURVIVOR_CONTINUATION_V202_PUBLIC_MASKED_MODEL_CAPACITY_HARNESS_CONTRACT_POLICY
        ),
        "selected_input_route": EXPECTED_V201_ROUTE,
        "historical_evidence_checkpoint": historical_evidence_checkpoint(),
        "inputs": {
            "v198_report": str(v198_report_path),
            "v199_report": str(v199_report_path),
            "v200_report": str(v200_report_path),
            "v201_report": str(v201_report_path),
            "expected_v198_report_exact_digest": expected_v198_report_exact_digest,
            "expected_v199_report_exact_digest": expected_v199_report_exact_digest,
            "expected_v200_report_exact_digest": expected_v200_report_exact_digest,
            "expected_v201_report_exact_digest": expected_v201_report_exact_digest,
        },
        "source_pin_validation": source_validation,
        "v201_fact_assessment": v201_facts,
        "public_masked_model_capacity_harness_contract": harness_contract,
        "budget_state": {
            "training_slices_consumed": TRAINING_SLICES_CONSUMED,
            "training_slice_budget": TRAINING_SLICE_BUDGET,
            "campaign_budget": f"{TRAINING_SLICES_CONSUMED}/{TRAINING_SLICE_BUDGET}",
            "slice_4_available_but_not_started": True,
            "slice_4_training_authorized": False,
        },
        "route_decision": route_decision,
        "classification": {
            "primary": classification,
            "labels": [
                classification,
                "diagnostics_only",
                "public_masked_model_capacity_harness_contract",
                "no_training",
                "no_slice_4_consumption",
                "no_runtime_integration",
                "no_gate_relaxation",
                "no_support_generation",
                "no_support_expansion",
                "no_dataset_mutation",
                "no_promotion",
            ],
        },
        **lifecycle_flags(),
    }
    report["exact_digest"] = digest_without_exact(report)
    write_json(output_path, report)
    return report


def validate_v202_source_pins(
    *,
    v198_report: Mapping[str, object],
    v198_load_error: str | None,
    v199_report: Mapping[str, object],
    v199_load_error: str | None,
    v200_report: Mapping[str, object],
    v200_load_error: str | None,
    v201_report: Mapping[str, object],
    v201_load_error: str | None,
    expected_v198_report_exact_digest: str,
    expected_v199_report_exact_digest: str,
    expected_v200_report_exact_digest: str,
    expected_v201_report_exact_digest: str,
) -> dict[str, object]:
    v198_exact_validation = exact_digest_validation_report(v198_report)
    v199_exact_validation = exact_digest_validation_report(v199_report)
    v200_exact_validation = exact_digest_validation_report(v200_report)
    v201_exact_validation = exact_digest_validation_report(v201_report)
    v198_source = _mapping(v198_report.get("source_pin_validation"))
    v198_source_checks = _mapping(v198_source.get("checks"))
    v199_source = _mapping(v199_report.get("source_pin_validation"))
    v200_source = _mapping(v200_report.get("source_pin_validation"))
    v201_source = _mapping(v201_report.get("source_pin_validation"))
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
        "v198_exact_digest_valid": v198_exact_validation.get("passed") is True,
        "v198_exact_digest_matches_expected": (
            v198_report.get("exact_digest") == expected_v198_report_exact_digest
        ),
        "v198_classification_matches_expected": (
            _classification(v198_report) == EXPECTED_V198_CLASSIFICATION
        ),
        "v198_route_matches_expected": _route_matches(v198_report, EXPECTED_V198_ROUTE),
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
        "v199_report_loaded": v199_load_error is None,
        "v199_report_schema_matches": (
            v199_report.get("schema_version")
            == v199.M3_CARRION_SURVIVOR_CONTINUATION_V199_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_AUDIT_SCHEMA_VERSION
        ),
        "v199_report_policy_matches": (
            v199_report.get("policy")
            == v199.M3_CARRION_SURVIVOR_CONTINUATION_V199_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_AUDIT_POLICY
        ),
        "v199_exact_digest_valid": v199_exact_validation.get("passed") is True,
        "v199_exact_digest_matches_expected": (
            v199_report.get("exact_digest") == expected_v199_report_exact_digest
        ),
        "v199_classification_matches_expected": (
            _classification(v199_report) == EXPECTED_V199_CLASSIFICATION
        ),
        "v199_route_matches_expected": _route_matches(v199_report, EXPECTED_V199_ROUTE),
        "v199_source_pin_validation_passed": v199_source.get("passed") is True,
        "v200_report_loaded": v200_load_error is None,
        "v200_report_schema_matches": (
            v200_report.get("schema_version")
            == v200.M3_CARRION_SURVIVOR_CONTINUATION_V200_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_REPAIR_DESIGN_SCHEMA_VERSION
        ),
        "v200_report_policy_matches": (
            v200_report.get("policy")
            == v200.M3_CARRION_SURVIVOR_CONTINUATION_V200_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_REPAIR_DESIGN_POLICY
        ),
        "v200_exact_digest_valid": v200_exact_validation.get("passed") is True,
        "v200_exact_digest_matches_expected": (
            v200_report.get("exact_digest") == expected_v200_report_exact_digest
        ),
        "v200_classification_matches_expected": (
            _classification(v200_report) == EXPECTED_V200_CLASSIFICATION
        ),
        "v200_route_matches_expected": _route_matches(v200_report, EXPECTED_V200_ROUTE),
        "v200_source_pin_validation_passed": v200_source.get("passed") is True,
        "v201_report_loaded": v201_load_error is None,
        "v201_report_schema_matches": (
            v201_report.get("schema_version")
            == v201.M3_CARRION_SURVIVOR_CONTINUATION_V201_HIGH_SPECIFICITY_ACTION_COMPLETE_SOURCE_CONTRACT_AUDIT_SCHEMA_VERSION
        ),
        "v201_report_policy_matches": (
            v201_report.get("policy")
            == v201.M3_CARRION_SURVIVOR_CONTINUATION_V201_HIGH_SPECIFICITY_ACTION_COMPLETE_SOURCE_CONTRACT_AUDIT_POLICY
        ),
        "v201_exact_digest_valid": v201_exact_validation.get("passed") is True,
        "v201_exact_digest_matches_expected": (
            v201_report.get("exact_digest") == expected_v201_report_exact_digest
        ),
        "v201_classification_matches_expected": (
            _classification(v201_report) == EXPECTED_V201_CLASSIFICATION
        ),
        "v201_route_matches_expected": _route_matches(v201_report, EXPECTED_V201_ROUTE),
        "v201_source_pin_validation_passed": v201_source.get("passed") is True,
        "v201_lifecycle_closed": _v201_lifecycle_closed(v201_report),
        "v198_commit_pin_matches_expected": (
            EXPECTED_V198_COMMIT == "029ed5d8c1b3e35dab52228a6d38fce30a084062"
        ),
        "v199_commit_pin_matches_expected": (
            EXPECTED_V199_COMMIT
            == "b68c4e8949a8c652dfbbb8b64b23315d563d7cc7"
        ),
        "v200_commit_pin_matches_expected": (
            EXPECTED_V200_COMMIT
            == "73e04b9049acb5efc1a1e722de485e9f2aee55cd"
        ),
        "v201_commit_pin_matches_expected": (
            EXPECTED_V201_COMMIT
            == "18c262f51467d8c9b4654232d4eb3dbfb40556d7"
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v202_source_pin_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "load_errors": {
            "v198_report": v198_load_error,
            "v199_report": v199_load_error,
            "v200_report": v200_load_error,
            "v201_report": v201_load_error,
        },
        "validation_scope": {
            "v198_direct_report_validation": True,
            "v199_direct_report_validation": True,
            "v200_direct_report_validation": True,
            "v201_direct_report_validation": True,
            "v197_validation_mode": "through_v198_source_pin_validation",
            "v197_direct_report_validation": False,
        },
        "expected": _expected_pin_block(
            expected_v198_report_exact_digest=expected_v198_report_exact_digest,
            expected_v199_report_exact_digest=expected_v199_report_exact_digest,
            expected_v200_report_exact_digest=expected_v200_report_exact_digest,
            expected_v201_report_exact_digest=expected_v201_report_exact_digest,
        ),
        "observed": {
            "v198_report_exact_digest": v198_report.get("exact_digest"),
            "v198_classification": _classification(v198_report),
            "v198_route": _selected_route(v198_report),
            "v198_v197_report_exact_digest": v198_source.get(
                "observed_v197_report_exact_digest"
            ),
            "v198_v197_route": v198_source.get("observed_v197_route"),
            "v199_report_exact_digest": v199_report.get("exact_digest"),
            "v199_classification": _classification(v199_report),
            "v199_route": _selected_route(v199_report),
            "v200_report_exact_digest": v200_report.get("exact_digest"),
            "v200_classification": _classification(v200_report),
            "v200_route": _selected_route(v200_report),
            "v201_report_exact_digest": v201_report.get("exact_digest"),
            "v201_classification": _classification(v201_report),
            "v201_route": _selected_route(v201_report),
        },
        "exact_digest_validation": {
            "v198": v198_exact_validation,
            "v199": v199_exact_validation,
            "v200": v200_exact_validation,
            "v201": v201_exact_validation,
        },
        "checks": checks,
    }


def assess_v201_required_facts(v201_report: Mapping[str, object]) -> dict[str, object]:
    audit = _mapping(v201_report.get("source_contract_audit"))
    counts = _mapping(audit.get("counts"))
    gaps = _mapping(audit.get("gap_classification"))
    route = _mapping(v201_report.get("route_decision"))
    checks = {
        "classification_matches": _classification(v201_report)
        == EXPECTED_V201_CLASSIFICATION,
        "route_matches": route.get("selected_route") == EXPECTED_V201_ROUTE
        and route.get("recommended_next_route") == EXPECTED_V201_ROUTE,
        "missing_current_valid_action_demands_match": _int(
            counts.get("total_missing_current_valid_action_demands")
        )
        == EXPECTED_V201_MISSING_CURRENT_VALID_ACTION_DEMANDS,
        "same_high_specific_action_support_zero": _int(
            counts.get("same_high_specific_action_present_for_missing_demands_count")
        )
        == EXPECTED_V201_SAME_HIGH_SPECIFIC_ACTION_SUPPORT,
        "same_high_specific_below_floor_support_zero": _int(
            counts.get("same_high_specific_below_observed_support_floor_count")
        )
        == EXPECTED_V201_SAME_HIGH_SPECIFIC_BELOW_FLOOR_SUPPORT,
        "lower_specificity_only_demands_match": _int(
            counts.get("lower_specificity_only_missing_action_count")
        )
        == EXPECTED_V201_LOWER_SPECIFICITY_ONLY_DEMANDS,
        "lower_specificity_only_share_matches": _round(
            gaps.get("lower_specificity_only_missing_action_share")
        )
        == EXPECTED_V201_LOWER_SPECIFICITY_ONLY_SHARE,
        "absent_at_same_and_known_lower_keys_match": _int(
            counts.get("missing_action_absent_at_same_and_known_lower_keys_count")
        )
        == EXPECTED_V201_ABSENT_AT_SAME_AND_KNOWN_LOWER_KEYS,
        "candidate_key_action_coverage_count_matches": _int(
            counts.get("candidate_key_action_coverage_count")
        )
        == EXPECTED_V201_CANDIDATE_KEY_ACTION_COVERAGE_COUNT,
        "artifact_feature_key_count_matches": _int(
            counts.get("artifact_feature_key_count")
        )
        == EXPECTED_V201_ARTIFACT_FEATURE_KEY_COUNT,
        "lower_specificity_grafting_non_authorizing": (
            "non_authorizing"
            in str(
                _mapping(audit.get("unsafe_non_authorizing_patterns")).get(
                    "lower_specificity_graft_or_imputation"
                )
            )
        ),
        "lifecycle_closed": _v201_lifecycle_closed(v201_report),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v202_v201_fact_assessment_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "checks": checks,
        "expected": {
            "classification": EXPECTED_V201_CLASSIFICATION,
            "route": EXPECTED_V201_ROUTE,
            "missing_current_valid_action_demands": (
                EXPECTED_V201_MISSING_CURRENT_VALID_ACTION_DEMANDS
            ),
            "same_high_specific_action_support": (
                EXPECTED_V201_SAME_HIGH_SPECIFIC_ACTION_SUPPORT
            ),
            "same_high_specific_below_floor_support": (
                EXPECTED_V201_SAME_HIGH_SPECIFIC_BELOW_FLOOR_SUPPORT
            ),
            "lower_specificity_only_demands": (
                EXPECTED_V201_LOWER_SPECIFICITY_ONLY_DEMANDS
            ),
            "lower_specificity_only_share": (
                EXPECTED_V201_LOWER_SPECIFICITY_ONLY_SHARE
            ),
            "absent_at_same_and_known_lower_specificity_keys": (
                EXPECTED_V201_ABSENT_AT_SAME_AND_KNOWN_LOWER_KEYS
            ),
            "candidate_key_action_coverage_count": (
                EXPECTED_V201_CANDIDATE_KEY_ACTION_COVERAGE_COUNT
            ),
            "artifact_feature_key_count": EXPECTED_V201_ARTIFACT_FEATURE_KEY_COUNT,
        },
        "observed": {
            "classification": _classification(v201_report),
            "route": route.get("selected_route"),
            "missing_current_valid_action_demands": counts.get(
                "total_missing_current_valid_action_demands"
            ),
            "same_high_specific_action_support": counts.get(
                "same_high_specific_action_present_for_missing_demands_count"
            ),
            "same_high_specific_below_floor_support": counts.get(
                "same_high_specific_below_observed_support_floor_count"
            ),
            "lower_specificity_only_demands": counts.get(
                "lower_specificity_only_missing_action_count"
            ),
            "lower_specificity_only_share": gaps.get(
                "lower_specificity_only_missing_action_share"
            ),
            "absent_at_same_and_known_lower_specificity_keys": counts.get(
                "missing_action_absent_at_same_and_known_lower_keys_count"
            ),
            "candidate_key_action_coverage_count": counts.get(
                "candidate_key_action_coverage_count"
            ),
            "artifact_feature_key_count": counts.get("artifact_feature_key_count"),
        },
    }


def public_masked_model_capacity_harness_contract(
    *,
    interface_expressible: bool = True,
    requires_private_runtime_features: bool = False,
) -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract_v1",
        "contract_complete_for_scaffold": (
            interface_expressible and not requires_private_runtime_features
        ),
        "runtime_implementation_in_v202": False,
        "public_input_action_mask_contract_expressible_from_existing_repo_surfaces": (
            interface_expressible
        ),
        "requires_private_or_provenance_runtime_features": (
            requires_private_runtime_features
        ),
        "existing_repo_public_surfaces": {
            "trajectory_public_observation_payload": (
                "python/evolution_sim/env/runtime/trajectory.py: observation_input"
            ),
            "trajectory_public_action_mask": (
                "python/evolution_sim/env/runtime/trajectory.py: action_mask"
            ),
            "dataset_public_observation_payload": (
                "python/evolution_sim/mind/dataset.py: observation_input"
            ),
            "dataset_public_action_mask": (
                "python/evolution_sim/mind/dataset.py: action_mask"
            ),
            "policy_public_history_context": (
                "python/evolution_sim/mind/v3_policy.py: public history/context helpers"
            ),
        },
        "allowed_runtime_inputs": {
            "public_observation_payload": (
                "current public observation_input encoded by the runtime observation contract"
            ),
            "public_currently_valid_action_mask": (
                "current action_mask from the same decision tick as the observation"
            ),
            "deterministic_public_history_or_context": (
                "history/context already available at runtime and derived only from prior public observations, public masks, and public requested/resolved actions"
            ),
        },
        "forbidden_runtime_inputs": {
            "seed_or_fixture_identity": True,
            "private_world_state": True,
            "future_outcomes_or_terminal_labels": True,
            "held_out_labels": True,
            "report_provenance_or_source_path": True,
            "support_count_oracle": True,
            "train_test_split_identity_at_inference": True,
            "lower_specificity_key_hit_as_action_authority": True,
        },
        "action_selection_invariant": {
            "policy": "current_action_mask_constrained_argmax_or_sampling_v1",
            "argmax_or_sampled_action_must_be_currently_valid": True,
            "if_predicted_action_invalid": "abstain_or_fail_closed",
            "if_current_valid_action_mask_empty": "abstain_or_fail_closed",
            "unconstrained_action_logits_authorize_no_runtime_action": True,
            "lower_specificity_grafting_is_runtime_substitute": False,
        },
        "deterministic_replay_and_seed_split_requirements": {
            "fixed_random_seeds_recorded": True,
            "train_validation_diagnostic_and_promotion_splits_named": True,
            "promotion_heldout_must_not_overlap_support_or_selection_seeds": True,
            "replay_command_and_commit_recorded": True,
            "per_seed_results_required": True,
            "aggregate_only_pass_forbidden": True,
        },
        "future_trainer_artifact_metadata_requirements": {
            "config": True,
            "training_data_digest": True,
            "code_commit": True,
            "source_report_exact_digest_pins": [
                "v197_through_v198_source_pin_validation",
                "v198",
                "v199",
                "v200",
                "v201",
                "v202_contract",
            ],
            "random_seeds": True,
            "device": True,
            "mask_contract_version": (
                "m3_carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract_v1"
            ),
            "lifecycle_flags": True,
            "training_slice_number": "future_explicit_opt_in_only",
        },
        "evaluation_gates_before_future_training_authorization": {
            "per_seed_broad_alive_birth_no_regression": True,
            "carrion_only_120_terminal_survivor_positive": True,
            "dominant_requested_action_share_cap": 0.5,
            "heuristic_action_source_count_zero": True,
            "exact_lifecycle_flags_closed_before_training": True,
            "no_aggregate_only_pass": True,
            "current_action_mask_validity_accounting": True,
            "controlled_fixture_and_broad_matrix_reported_separately": True,
        },
        "contamination_and_leakage_checks_before_future_capacity_work": {
            "no_fixture_identity_or_seed_leakage": True,
            "no_private_world_state": True,
            "no_future_outcome_or_label_at_inference": True,
            "no_support_count_oracle_at_inference": True,
            "no_report_path_or_provenance_feature_at_inference": True,
            "split_identity_not_available_to_model": True,
            "diagnostic_seeds_not_claimed_as_clean_promotion_heldout": True,
            "lower_specificity_evidence_not_reinterpreted_as_high_specific_support": True,
        },
        "lower_specificity_evidence_policy": {
            "runtime_substitute_for_high_specific_current_valid_support": "rejected",
            "grafting_or_imputation_authorization": (
                "non_authorizing_unless exact same high-specific public key and current-valid action support is proven"
            ),
            "v201_lower_specificity_only_share": EXPECTED_V201_LOWER_SPECIFICITY_ONLY_SHARE,
        },
        "lifecycle_limits": {
            "training_allowed": False,
            "slice_4_training_allowed": False,
            "training_artifact_allowed": False,
            "runtime_artifact_allowed": False,
            "runtime_integration_allowed": False,
            "default_policy_change_allowed": False,
            "support_generation_allowed": False,
            "support_expansion_allowed": False,
            "dataset_mutation_allowed": False,
            "gate_relaxation_allowed": False,
            "promotion_authorized": False,
        },
    }


def route_decision_for_v202(
    *,
    source_validation: Mapping[str, object],
    v201_facts: Mapping[str, object],
    harness_contract: Mapping[str, object],
) -> dict[str, object]:
    if source_validation.get("passed") is not True:
        route = STOP_ROUTE
    elif v201_facts.get("passed") is not True:
        route = V201_RECONCILIATION_ROUTE
    elif (
        harness_contract.get(
            "public_input_action_mask_contract_expressible_from_existing_repo_surfaces"
        )
        is not True
    ):
        route = INTERFACE_GAP_REPAIR_ROUTE
    elif harness_contract.get("requires_private_or_provenance_runtime_features") is True:
        route = LEAKAGE_CONTRACT_REPAIR_ROUTE
    elif harness_contract.get("contract_complete_for_scaffold") is True:
        route = HARNESS_SCAFFOLD_ROUTE
    else:
        route = INTERFACE_GAP_REPAIR_ROUTE
    return {
        "policy": "m3_carrion_survivor_continuation_v202_route_decision_v1",
        "recommended_next_route": route,
        "selected_route": route,
        "exactly_one_next_route_recommended": True,
        "source_pins_valid": source_validation.get("passed") is True,
        "v201_facts_valid": v201_facts.get("passed") is True,
        "public_contract_complete": harness_contract.get(
            "contract_complete_for_scaffold"
        )
        is True,
        "direct_slice_4_training_allowed": False,
        "future_explicit_slice_4_training_route_authorized": False,
        "training_allowed": False,
        "slice_4_training_consumed": False,
        "runtime_integration_allowed": False,
        "runtime_action_selection_change_allowed": False,
        "support_generation_allowed": False,
        "support_expansion_allowed": False,
        "dataset_mutation_allowed": False,
        "gate_relaxation_allowed": False,
        "promotion_authorized": False,
        "rationale": route_rationale(route),
    }


def classification_for_v202(
    *,
    source_validation: Mapping[str, object],
    route_decision: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v202_"
    if source_validation.get("passed") is not True:
        return prefix + "source_pins_invalid_routes_to_repair_no_training"
    route = str(route_decision.get("selected_route") or "")
    if route == HARNESS_SCAFFOLD_ROUTE:
        return prefix + "public_masked_model_capacity_harness_contract_ready_for_scaffold_no_training"
    if route == INTERFACE_GAP_REPAIR_ROUTE:
        return prefix + "public_masked_model_capacity_interface_gap_repair_no_training"
    if route == LEAKAGE_CONTRACT_REPAIR_ROUTE:
        return prefix + "public_feature_leakage_contract_repair_no_training"
    if route == V201_RECONCILIATION_ROUTE:
        return prefix + "v201_contract_reconciliation_no_training"
    return prefix + "closed_no_training"


def lifecycle_flags() -> dict[str, object]:
    return {
        "training_started": False,
        "training_slice_4_consumed": False,
        "training_artifact_created": False,
        "runtime_artifact_created": False,
        "runtime_integration_changed": False,
        "runtime_action_selection_changed": False,
        "default_policy_changed": False,
        "support_generated": False,
        "support_expanded": False,
        "dataset_mutated": False,
        "gate_relaxed": False,
        "promotion_authorized": False,
        "non_promoted": True,
    }


def historical_evidence_checkpoint() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v202_historical_evidence_checkpoint_v1",
        "v53_v63": (
            "scalar IQL coefficient/prior/action-distribution/global-bias/extraction tuning is closed"
        ),
        "v154_v176": (
            "tiny support-archive / nearest-neighbor scorer loop is strategically saturated"
        ),
        "v180_v186_v195": "training slices failed and are not promotion evidence",
        "v196": "low-specificity artifact coverage collapsed to eat with miss abstention",
        "v197": "low-specificity overrides were blocked without changing default runtime behavior",
        "v198": "high-specific keys existed but none were action-complete",
        "v199": "all 694 present high-specific candidate occurrences were action-incomplete",
        "v200": "selected source-contract audit before future dataset/harness work",
        "v201": (
            "lower-specificity-only evidence dominates missing current-valid "
            "actions and must not be grafted into runtime action selection"
        ),
        "anti_loop_stop_rules": [
            "do not rerun v180/v186/v195 training",
            "do not rerun scalar IQL tuning or prior blends",
            "do not rerun nearest-neighbor scorer/archive loops",
            "do not treat lower-specificity evidence as high-specific support",
            "do not impute missing current-valid high-specific actions",
            "do not propose blind support-table expansion",
            "do not authorize slice-4 training from v202",
        ],
    }


def digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)


def route_rationale(route: str) -> str:
    if route == HARNESS_SCAFFOLD_ROUTE:
        return (
            "v201 proved lower-specificity-only evidence dominates the missing "
            "current-valid action demands. v202 defines a public observation plus "
            "current-action-mask constrained capacity harness contract; the next "
            "safe step is a no-training scaffold, not training or support expansion."
        )
    if route == INTERFACE_GAP_REPAIR_ROUTE:
        return "The public observation/action-mask contract cannot yet be expressed cleanly."
    if route == LEAKAGE_CONTRACT_REPAIR_ROUTE:
        return "The proposed harness would require private, fixture, or provenance features."
    if route == V201_RECONCILIATION_ROUTE:
        return "The v201 blocker facts do not match the expected lower-specificity-only conclusion."
    if route == STOP_ROUTE:
        return "Required v198-v201 pins or v197-through-v198 validation failed."
    return "Closed no-training route."


def _load_optional_json_report(path: str | Path) -> tuple[dict[str, object], str | None]:
    try:
        return load_json_report(path), None
    except (OSError, ValueError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def _classification(report: Mapping[str, object]) -> object:
    return _mapping(report.get("classification")).get("primary")


def _selected_route(report: Mapping[str, object]) -> object:
    route = _mapping(report.get("route_decision"))
    return route.get("selected_route") or route.get("recommended_next_route")


def _route_matches(report: Mapping[str, object], expected: str) -> bool:
    route = _mapping(report.get("route_decision"))
    selected = route.get("selected_route")
    recommended = route.get("recommended_next_route")
    return (selected == expected or selected is None) and recommended == expected


def _v201_lifecycle_closed(report: Mapping[str, object]) -> bool:
    keys = (
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
    return all(report.get(key) is False for key in keys) and (
        report.get("non_promoted") is True
    )


def _expected_pin_block(
    *,
    expected_v198_report_exact_digest: str,
    expected_v199_report_exact_digest: str,
    expected_v200_report_exact_digest: str,
    expected_v201_report_exact_digest: str,
) -> dict[str, object]:
    return {
        "v197": {
            "commit": EXPECTED_V197_COMMIT,
            "report_exact_digest": EXPECTED_V197_REPORT_EXACT_DIGEST,
            "archive": EXPECTED_V197_ARCHIVE_PATH,
            "archive_sha256": EXPECTED_V197_ARCHIVE_SHA256,
            "validation_mode": "through_v198_source_pin_validation",
        },
        "v198": {
            "commit": EXPECTED_V198_COMMIT,
            "report_exact_digest": expected_v198_report_exact_digest,
            "classification": EXPECTED_V198_CLASSIFICATION,
            "route": EXPECTED_V198_ROUTE,
            "archive": EXPECTED_V198_ARCHIVE_PATH,
            "archive_sha256": EXPECTED_V198_ARCHIVE_SHA256,
        },
        "v199": {
            "commit": EXPECTED_V199_COMMIT,
            "report_exact_digest": expected_v199_report_exact_digest,
            "classification": EXPECTED_V199_CLASSIFICATION,
            "route": EXPECTED_V199_ROUTE,
            "archive": EXPECTED_V199_ARCHIVE_PATH,
            "archive_sha256": EXPECTED_V199_ARCHIVE_SHA256,
        },
        "v200": {
            "commit": EXPECTED_V200_COMMIT,
            "report_exact_digest": expected_v200_report_exact_digest,
            "classification": EXPECTED_V200_CLASSIFICATION,
            "route": EXPECTED_V200_ROUTE,
            "archive": EXPECTED_V200_ARCHIVE_PATH,
            "archive_sha256": EXPECTED_V200_ARCHIVE_SHA256,
        },
        "v201": {
            "commit": EXPECTED_V201_COMMIT,
            "report_exact_digest": expected_v201_report_exact_digest,
            "classification": EXPECTED_V201_CLASSIFICATION,
            "route": EXPECTED_V201_ROUTE,
            "archive": EXPECTED_V201_ARCHIVE_PATH,
            "archive_sha256": EXPECTED_V201_ARCHIVE_SHA256,
        },
    }
