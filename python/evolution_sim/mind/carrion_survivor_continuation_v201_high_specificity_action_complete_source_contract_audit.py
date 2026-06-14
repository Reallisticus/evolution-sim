from __future__ import annotations

from collections import Counter
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
from evolution_sim.mind.candidate_campaign import _int, _mapping, _round, write_json
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V201_HIGH_SPECIFICITY_ACTION_COMPLETE_SOURCE_CONTRACT_AUDIT_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V201_HIGH_SPECIFICITY_ACTION_COMPLETE_SOURCE_CONTRACT_AUDIT_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit_v1"
)

DEFAULT_V198_REPORT_PATH = v198.DEFAULT_OUTPUT_PATH
DEFAULT_V199_REPORT_PATH = v199.DEFAULT_OUTPUT_PATH
DEFAULT_V200_REPORT_PATH = v200.DEFAULT_OUTPUT_PATH
DEFAULT_V195_ARTIFACT_PATH = v200.DEFAULT_V195_ARTIFACT_PATH
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v201-carrion-survivor-continuation-high-specificity-action-complete-source-contract-audit.json"
)

EXPECTED_V197_COMMIT = v200.EXPECTED_V197_COMMIT
EXPECTED_V197_REPORT_EXACT_DIGEST = v200.EXPECTED_V197_REPORT_EXACT_DIGEST
EXPECTED_V197_ARCHIVE_PATH = v200.EXPECTED_V197_ARCHIVE_PATH
EXPECTED_V197_ARCHIVE_SHA256 = v200.EXPECTED_V197_ARCHIVE_SHA256
EXPECTED_V198_COMMIT = v200.EXPECTED_V198_COMMIT
EXPECTED_V198_REPORT_EXACT_DIGEST = v200.EXPECTED_V198_REPORT_EXACT_DIGEST
EXPECTED_V198_ARCHIVE_PATH = v200.EXPECTED_V198_ARCHIVE_PATH
EXPECTED_V198_ARCHIVE_SHA256 = v200.EXPECTED_V198_ARCHIVE_SHA256
EXPECTED_V199_COMMIT = v200.EXPECTED_V199_COMMIT
EXPECTED_V199_REPORT_EXACT_DIGEST = v200.EXPECTED_V199_REPORT_EXACT_DIGEST
EXPECTED_V199_ARCHIVE_PATH = v200.EXPECTED_V199_ARCHIVE_PATH
EXPECTED_V199_ARCHIVE_SHA256 = v200.EXPECTED_V199_ARCHIVE_SHA256
EXPECTED_V199_CLASSIFICATION = v200.EXPECTED_V199_CLASSIFICATION
EXPECTED_V199_ROUTE = v200.EXPECTED_V199_ROUTE
EXPECTED_V200_COMMIT = "73e04b9049acb5efc1a1e722de485e9f2aee55cd"
EXPECTED_V200_REPORT_EXACT_DIGEST = (
    "71d8959ac4bdb73a45c888e159ea032ac3a5064352c890eee9914356bbb3184c"
)
EXPECTED_V200_ARCHIVE_PATH = (
    "gdrive:evolution-sim-backups/archives/"
    "20260614T202942Z-v200-high-specificity-action-complete-contract-repair-design-qa-repair.tar.zst"
)
EXPECTED_V200_ARCHIVE_SHA256 = (
    "62cb570dd26ffc9e3c8f5062531b0de6ac955cf27eb3d01a3334e84ade60f1cd"
)
EXPECTED_V200_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v200_existing_artifact_action_incomplete_routes_to_source_contract_audit_no_training"
)
EXPECTED_V200_ROUTE = v200.SOURCE_CONTRACT_AUDIT_ROUTE
EXPECTED_V200_REPAIR_CLASS = "source_contract_audit_for_missing_current_valid_actions"
EXPECTED_V195_ARTIFACT_DIGEST = v200.EXPECTED_V195_ARTIFACT_DIGEST

EXPECTED_HIGH_SPECIFIC_CANDIDATES = v200.EXPECTED_V199_EVALUATED_COUNT
EXPECTED_KEY_ABSENT = v200.EXPECTED_V199_ABSENT_COUNT
EXPECTED_KEY_PRESENT = v200.EXPECTED_V199_PRESENT_COUNT
EXPECTED_PRESENT_INCOMPLETE = v200.EXPECTED_V199_PRESENT_INCOMPLETE_COUNT
EXPECTED_SUPPORT_FLOOR_FAILURES = v200.EXPECTED_V199_SUPPORT_FLOOR_FAILURE_COUNT
EXPECTED_MISSING_ACTION_COUNTS = {
    "attack_east": 76,
    "attack_north": 4,
    "attack_west": 80,
    "drink": 65,
    "eat": 628,
    "move_east": 339,
    "move_north": 245,
    "move_south": 469,
    "move_west": 475,
    "stay": 333,
}

STOP_ROUTE = "stop_v201_source_pins_invalid_no_training"
LINEAGE_REPAIR_ROUTE = "v202_v200_v199_lineage_repair_no_training"
INSTRUMENTATION_REPAIR_ROUTE = (
    "v202_source_contract_instrumentation_repair_no_training"
)
DATASET_CONTRACT_DESIGN_ROUTE = (
    "v202_high_specificity_action_complete_dataset_contract_design_no_training"
)
PUBLIC_MODEL_CAPACITY_ROUTE = (
    "v202_public_masked_model_capacity_harness_contract_no_training"
)
LOWER_SPECIFICITY_DOMINANCE_THRESHOLD = 0.5


def run_carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit(
    *,
    v198_report_path: str | Path = DEFAULT_V198_REPORT_PATH,
    v199_report_path: str | Path = DEFAULT_V199_REPORT_PATH,
    v200_report_path: str | Path = DEFAULT_V200_REPORT_PATH,
    v195_artifact_path: str | Path = DEFAULT_V195_ARTIFACT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    expected_v198_report_exact_digest: str = EXPECTED_V198_REPORT_EXACT_DIGEST,
    expected_v199_report_exact_digest: str = EXPECTED_V199_REPORT_EXACT_DIGEST,
    expected_v200_report_exact_digest: str = EXPECTED_V200_REPORT_EXACT_DIGEST,
    expected_v195_artifact_digest: str = EXPECTED_V195_ARTIFACT_DIGEST,
) -> dict[str, object]:
    v198_report, v198_load_error = _load_optional_json_report(v198_report_path)
    v199_report, v199_load_error = _load_optional_json_report(v199_report_path)
    v200_report, v200_load_error = _load_optional_json_report(v200_report_path)
    v195_artifact, v195_artifact_load_error = _load_optional_json_report(
        v195_artifact_path
    )
    source_validation = validate_v201_source_pins(
        v198_report=v198_report,
        v198_load_error=v198_load_error,
        v199_report=v199_report,
        v199_load_error=v199_load_error,
        v200_report=v200_report,
        v200_load_error=v200_load_error,
        v195_artifact=v195_artifact,
        v195_artifact_load_error=v195_artifact_load_error,
        expected_v198_report_exact_digest=expected_v198_report_exact_digest,
        expected_v199_report_exact_digest=expected_v199_report_exact_digest,
        expected_v200_report_exact_digest=expected_v200_report_exact_digest,
        expected_v195_artifact_digest=expected_v195_artifact_digest,
    )
    lineage_facts = assess_v201_lineage_facts(
        v199_report=v199_report,
        v200_report=v200_report,
    )
    source_audit = audit_high_specific_source_contract(
        v199_report=v199_report,
        v195_artifact=v195_artifact,
    )
    contract = action_complete_source_contract()
    route_decision = route_decision_for_v201(
        source_validation=source_validation,
        lineage_facts=lineage_facts,
        source_audit=source_audit,
    )
    classification = classification_for_v201(
        source_validation=source_validation,
        route_decision=route_decision,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V201_HIGH_SPECIFICITY_ACTION_COMPLETE_SOURCE_CONTRACT_AUDIT_SCHEMA_VERSION
        ),
        "policy": (
            M3_CARRION_SURVIVOR_CONTINUATION_V201_HIGH_SPECIFICITY_ACTION_COMPLETE_SOURCE_CONTRACT_AUDIT_POLICY
        ),
        "contract": contract,
        "historical_evidence_checkpoint": historical_evidence_checkpoint(),
        "inputs": {
            "v198_report": str(v198_report_path),
            "v199_report": str(v199_report_path),
            "v200_report": str(v200_report_path),
            "v195_artifact": str(v195_artifact_path),
            "expected_v198_report_exact_digest": expected_v198_report_exact_digest,
            "expected_v199_report_exact_digest": expected_v199_report_exact_digest,
            "expected_v200_report_exact_digest": expected_v200_report_exact_digest,
            "expected_v195_artifact_digest": expected_v195_artifact_digest,
        },
        "source_pin_validation": source_validation,
        "lineage_fact_assessment": lineage_facts,
        "source_contract_audit": source_audit,
        "action_complete_source_contract": contract,
        "route_decision": route_decision,
        "classification": {
            "primary": classification,
            "labels": [
                classification,
                "diagnostics_only",
                "high_specificity_action_complete_source_contract_audit",
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


def validate_v201_source_pins(
    *,
    v198_report: Mapping[str, object],
    v198_load_error: str | None,
    v199_report: Mapping[str, object],
    v199_load_error: str | None,
    v200_report: Mapping[str, object],
    v200_load_error: str | None,
    v195_artifact: Mapping[str, object],
    v195_artifact_load_error: str | None,
    expected_v198_report_exact_digest: str,
    expected_v199_report_exact_digest: str,
    expected_v200_report_exact_digest: str,
    expected_v195_artifact_digest: str,
) -> dict[str, object]:
    v198_exact_validation = exact_digest_validation_report(v198_report)
    v199_exact_validation = exact_digest_validation_report(v199_report)
    v200_exact_validation = exact_digest_validation_report(v200_report)
    v198_source = _mapping(v198_report.get("source_pin_validation"))
    v198_source_checks = _mapping(v198_source.get("checks"))
    v199_classification = _mapping(v199_report.get("classification"))
    v199_route = _mapping(v199_report.get("route_decision"))
    v199_source = _mapping(v199_report.get("source_pin_validation"))
    v200_classification = _mapping(v200_report.get("classification"))
    v200_route = _mapping(v200_report.get("route_decision"))
    v200_source = _mapping(v200_report.get("source_pin_validation"))
    v195_artifact_digest = stable_payload_digest(v195_artifact)
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
        "v199_report_loaded": v199_load_error is None,
        "v199_report_schema_matches": (
            v199_report.get("schema_version")
            == v199.M3_CARRION_SURVIVOR_CONTINUATION_V199_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_AUDIT_SCHEMA_VERSION
        ),
        "v199_report_policy_matches": (
            v199_report.get("policy")
            == v199.M3_CARRION_SURVIVOR_CONTINUATION_V199_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_AUDIT_POLICY
        ),
        "v199_report_exact_digest_valid": (
            v199_exact_validation.get("passed") is True
        ),
        "v199_report_exact_digest_matches_expected": (
            v199_report.get("exact_digest") == expected_v199_report_exact_digest
        ),
        "v199_classification_matches_expected": (
            v199_classification.get("primary") == EXPECTED_V199_CLASSIFICATION
        ),
        "v199_route_matches_expected": (
            v199_route.get("selected_route") == EXPECTED_V199_ROUTE
            and v199_route.get("recommended_next_route") == EXPECTED_V199_ROUTE
        ),
        "v199_source_pin_validation_passed": v199_source.get("passed") is True,
        "v199_lifecycle_closed": _lifecycle_closed(v199_report, lifecycle_keys),
        "v200_report_loaded": v200_load_error is None,
        "v200_report_schema_matches": (
            v200_report.get("schema_version")
            == v200.M3_CARRION_SURVIVOR_CONTINUATION_V200_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_REPAIR_DESIGN_SCHEMA_VERSION
        ),
        "v200_report_policy_matches": (
            v200_report.get("policy")
            == v200.M3_CARRION_SURVIVOR_CONTINUATION_V200_HIGH_SPECIFICITY_ACTION_COMPLETE_CONTRACT_REPAIR_DESIGN_POLICY
        ),
        "v200_report_exact_digest_valid": (
            v200_exact_validation.get("passed") is True
        ),
        "v200_report_exact_digest_matches_expected": (
            v200_report.get("exact_digest") == expected_v200_report_exact_digest
        ),
        "v200_classification_matches_expected": (
            v200_classification.get("primary") == EXPECTED_V200_CLASSIFICATION
        ),
        "v200_route_matches_expected": (
            v200_route.get("selected_route") == EXPECTED_V200_ROUTE
            and v200_route.get("recommended_next_route") == EXPECTED_V200_ROUTE
        ),
        "v200_source_pin_validation_passed": v200_source.get("passed") is True,
        "v200_observed_v198_report_digest_matches_expected": (
            v200_source.get("observed_v198_report_exact_digest")
            == expected_v198_report_exact_digest
        ),
        "v200_observed_v199_report_digest_matches_expected": (
            v200_source.get("observed_v199_report_exact_digest")
            == expected_v199_report_exact_digest
        ),
        "v200_observed_v198_v197_report_digest_matches_expected": (
            v200_source.get("observed_v198_v197_report_exact_digest")
            == EXPECTED_V197_REPORT_EXACT_DIGEST
        ),
        "v200_lifecycle_closed": _lifecycle_closed(v200_report, lifecycle_keys),
        "v197_commit_pin_recorded": EXPECTED_V197_COMMIT == "da8fac5",
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
        "v195_artifact_loaded": v195_artifact_load_error is None,
        "v195_artifact_digest_matches_expected": (
            v195_artifact_digest == expected_v195_artifact_digest
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v201_source_pin_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "load_errors": {
            "v198_report": v198_load_error,
            "v199_report": v199_load_error,
            "v200_report": v200_load_error,
            "v195_artifact": v195_artifact_load_error,
        },
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
        "expected_v199": {
            "commit": EXPECTED_V199_COMMIT,
            "report_exact_digest": expected_v199_report_exact_digest,
            "classification": EXPECTED_V199_CLASSIFICATION,
            "route": EXPECTED_V199_ROUTE,
            "archive": EXPECTED_V199_ARCHIVE_PATH,
            "archive_sha256": EXPECTED_V199_ARCHIVE_SHA256,
        },
        "expected_v200": {
            "commit": EXPECTED_V200_COMMIT,
            "report_exact_digest": expected_v200_report_exact_digest,
            "classification": EXPECTED_V200_CLASSIFICATION,
            "route": EXPECTED_V200_ROUTE,
            "archive": EXPECTED_V200_ARCHIVE_PATH,
            "archive_sha256": EXPECTED_V200_ARCHIVE_SHA256,
        },
        "observed": {
            "v198_report_exact_digest": v198_report.get("exact_digest"),
            "v198_v197_report_exact_digest": v198_source.get(
                "observed_v197_report_exact_digest"
            ),
            "v198_v197_route": v198_source.get("observed_v197_route"),
            "v199_report_exact_digest": v199_report.get("exact_digest"),
            "v199_classification": v199_classification.get("primary"),
            "v199_route": v199_route.get("selected_route"),
            "v200_report_exact_digest": v200_report.get("exact_digest"),
            "v200_classification": v200_classification.get("primary"),
            "v200_route": v200_route.get("selected_route"),
            "v195_artifact_digest": v195_artifact_digest,
        },
        "exact_digest_validation": {
            "v198": v198_exact_validation,
            "v199": v199_exact_validation,
            "v200": v200_exact_validation,
        },
        "checks": checks,
    }


def assess_v201_lineage_facts(
    *,
    v199_report: Mapping[str, object],
    v200_report: Mapping[str, object],
) -> dict[str, object]:
    v199_combined = _combined_v199(v199_report)
    v200_facts = _mapping(v200_report.get("v199_fact_assessment"))
    v200_design = _mapping(v200_report.get("action_complete_repair_design"))
    v199_counts = _expected_count_checks(
        v199_combined,
        require_missing_action_counts=True,
    )
    v200_counts = _expected_count_checks(
        _mapping(v200_facts.get("counts")),
        require_missing_action_counts=False,
    )
    v200_repair_matches = (
        v200_design.get("selected_repair_class") == EXPECTED_V200_REPAIR_CLASS
    )
    facts_consistent = all(v199_counts.values()) and all(v200_counts.values())
    return {
        "policy": "m3_carrion_survivor_continuation_v201_lineage_fact_assessment_v1",
        "passed": facts_consistent and v200_repair_matches,
        "facts_consistent": facts_consistent,
        "v200_selected_repair_class_matches_expected": v200_repair_matches,
        "v199_expected_count_checks": v199_counts,
        "v200_expected_count_checks": v200_counts,
        "missing_current_valid_action_counts_match_expected": (
            _counter_dict(v199_combined.get("missing_current_valid_action_counts"))
            == EXPECTED_MISSING_ACTION_COUNTS
        ),
        "expected_missing_current_valid_action_counts": dict(
            sorted(EXPECTED_MISSING_ACTION_COUNTS.items())
        ),
        "observed_missing_current_valid_action_counts": _counter_dict(
            v199_combined.get("missing_current_valid_action_counts")
        ),
    }


def audit_high_specific_source_contract(
    *,
    v199_report: Mapping[str, object],
    v195_artifact: Mapping[str, object],
) -> dict[str, object]:
    combined = _combined_v199(v199_report)
    table = _feature_action_table(v195_artifact)
    rows = _mapping(combined.get("candidate_key_action_coverage"))
    candidate_action_rows: list[dict[str, object]] = []
    missing_by_action: Counter[str] = Counter()
    same_high_specific_present_by_action: Counter[str] = Counter()
    same_high_specific_below_floor_by_action: Counter[str] = Counter()
    lower_specificity_only_by_action: Counter[str] = Counter()
    absent_everywhere_by_action: Counter[str] = Counter()
    lower_key_hit_counter: Counter[str] = Counter()
    source_gap_counter: Counter[str] = Counter()
    candidate_keys_with_missing_actions: set[str] = set()
    support_floor = 2
    for candidate_key, raw_row in sorted(rows.items()):
        row = _mapping(raw_row)
        missing_counts = _counter_dict(row.get("missing_current_valid_action_counts"))
        if not missing_counts:
            continue
        candidate_keys_with_missing_actions.add(str(candidate_key))
        lower_keys = lower_specificity_keys(str(candidate_key))
        same_key_actions = _mapping(table.get(str(candidate_key)))
        for action, missing_count in sorted(missing_counts.items()):
            same_count = _action_support_count(same_key_actions, action)
            lower_counts = {
                lower_key: _action_support_count(_mapping(table.get(lower_key)), action)
                for lower_key in lower_keys
            }
            lower_counts = {
                key: count for key, count in lower_counts.items() if count > 0
            }
            missing_by_action[action] += missing_count
            if same_count > 0:
                same_high_specific_present_by_action[action] += missing_count
            if 0 < same_count < support_floor:
                same_high_specific_below_floor_by_action[action] += missing_count
            if same_count == 0 and lower_counts:
                lower_specificity_only_by_action[action] += missing_count
                for lower_key in lower_counts:
                    lower_key_hit_counter[lower_key] += missing_count
                gap = "lower_specificity_only_non_authorizing"
            elif same_count == 0:
                absent_everywhere_by_action[action] += missing_count
                gap = "absent_at_same_high_specific_key_and_known_lower_keys"
            elif same_count < support_floor:
                gap = "same_high_specific_below_observed_support_floor"
            else:
                gap = "instrumentation_mismatch_same_key_action_present"
            source_gap_counter[gap] += missing_count
            candidate_action_rows.append(
                {
                    "candidate_key": str(candidate_key),
                    "candidate_key_digest": row.get("candidate_key_digest")
                    or stable_payload_digest(str(candidate_key)),
                    "high_specificity_category": row.get("candidate_key_category"),
                    "missing_current_valid_action": action,
                    "missing_current_valid_action_count": missing_count,
                    "same_high_specific_action_support_count": same_count,
                    "same_high_specific_below_observed_support_floor": (
                        0 < same_count < support_floor
                    ),
                    "lower_specificity_action_support_counts": dict(
                        sorted(lower_counts.items())
                    ),
                    "unsafe_lower_specificity_graft_required": bool(lower_counts)
                    and same_count == 0,
                    "source_gap": gap,
                    "scope_counts": _counter_dict(row.get("scope_counts")),
                    "seed_counts": _counter_dict(row.get("seed_counts")),
                    "valid_action_demand_counts": _counter_dict(
                        row.get("valid_action_demand_counts")
                    ),
                }
            )
    total_missing = sum(missing_by_action.values())
    lower_specificity_only_total = sum(lower_specificity_only_by_action.values())
    lower_specificity_only_share = _round(
        lower_specificity_only_total / total_missing if total_missing else 0.0
    )
    same_high_specific_present_total = sum(
        same_high_specific_present_by_action.values()
    )
    same_high_specific_below_floor_total = sum(
        same_high_specific_below_floor_by_action.values()
    )
    absent_everywhere_total = sum(absent_everywhere_by_action.values())
    source_contract_auditable = bool(rows) and bool(table)
    lower_specificity_only_dominates = (
        lower_specificity_only_share > LOWER_SPECIFICITY_DOMINANCE_THRESHOLD
    )
    high_specific_absent_confirmed = (
        total_missing > 0
        and same_high_specific_present_total == 0
        and _int(combined.get("present_but_action_incomplete_count")) > 0
    )
    return {
        "policy": "m3_carrion_survivor_continuation_v201_source_contract_audit_v1",
        "diagnostics_only": True,
        "audit_source": {
            "candidate_rows": "v199 action_complete_audit.combined.candidate_key_action_coverage",
            "artifact_table": "v195 utility_tables.feature_action_utility",
            "support_floor": support_floor,
        },
        "source_contract_auditable_from_existing_artifacts": source_contract_auditable,
        "primary_blocker": _primary_blocker(
            source_contract_auditable=source_contract_auditable,
            lower_specificity_only_dominates=lower_specificity_only_dominates,
            high_specific_absent_confirmed=high_specific_absent_confirmed,
        ),
        "gap_classification": {
            "high_specific_key_absent": _int(combined.get("key_absent_count")),
            "high_specific_key_present_but_missing_current_valid_actions": _int(
                combined.get("present_but_action_incomplete_count")
            ),
            "same_high_specific_missing_action_support_count": total_missing
            - same_high_specific_present_total,
            "same_high_specific_action_present_for_missing_demands_count": (
                same_high_specific_present_total
            ),
            "same_high_specific_below_observed_support_floor_count": (
                same_high_specific_below_floor_total
            ),
            "lower_specificity_only_missing_action_count": (
                lower_specificity_only_total
            ),
            "lower_specificity_only_missing_action_share": (
                lower_specificity_only_share
            ),
            "lower_specificity_only_evidence_dominates": (
                lower_specificity_only_dominates
            ),
            "missing_action_absent_at_same_and_known_lower_keys_count": (
                absent_everywhere_total
            ),
            "valid_action_mask_mismatch_indicated": False,
            "source_split_or_provenance_gap_indicated": False,
            "instrumentation_limit_indicated": not source_contract_auditable,
            "unsafe_lower_specificity_imputation_would_be_required": (
                lower_specificity_only_total > 0
            ),
        },
        "counts": {
            "candidate_key_action_coverage_count": len(rows),
            "candidate_keys_with_missing_current_valid_actions": len(
                candidate_keys_with_missing_actions
            ),
            "total_missing_current_valid_action_demands": total_missing,
            "same_high_specific_action_present_for_missing_demands_count": (
                same_high_specific_present_total
            ),
            "same_high_specific_below_observed_support_floor_count": (
                same_high_specific_below_floor_total
            ),
            "lower_specificity_only_missing_action_count": (
                lower_specificity_only_total
            ),
            "missing_action_absent_at_same_and_known_lower_keys_count": (
                absent_everywhere_total
            ),
            "artifact_feature_key_count": len(table),
        },
        "missing_current_valid_action_counts": dict(sorted(missing_by_action.items())),
        "same_high_specific_action_present_counts": dict(
            sorted(same_high_specific_present_by_action.items())
        ),
        "same_high_specific_below_observed_support_floor_action_counts": dict(
            sorted(same_high_specific_below_floor_by_action.items())
        ),
        "lower_specificity_only_missing_action_counts": dict(
            sorted(lower_specificity_only_by_action.items())
        ),
        "missing_action_absent_at_same_and_known_lower_keys_counts": dict(
            sorted(absent_everywhere_by_action.items())
        ),
        "source_gap_counts": dict(sorted(source_gap_counter.items())),
        "lower_specificity_key_hit_counts": dict(
            sorted(lower_key_hit_counter.items())
        ),
        "aggregate_breakdown": {
            "by_scope": _stats_breakdown(
                _mapping(combined.get("by_scope")),
                "missing_current_valid_action_counts",
            ),
            "by_seed": _stats_breakdown(
                _mapping(combined.get("by_seed")),
                "missing_current_valid_action_counts",
            ),
            "by_high_specificity_category": _stats_breakdown(
                _mapping(combined.get("by_high_specificity_category")),
                "missing_current_valid_action_counts",
            ),
        },
        "candidate_missing_action_rows": candidate_action_rows,
        "source_diagnosis": {
            "absent_source_rows": (
                "confirmed_for_same_high_specific_key_current_valid_actions"
            ),
            "aggregation_contract_gap": (
                "confirmed_artifact_key_can_exist_without_all_current_valid_actions"
            ),
            "valid_action_mask_mismatch": (
                "not_supported_by_v199_current_valid_action_demand_diagnostics"
            ),
            "source_split_or_provenance": (
                "digest_pins_pass_but_split_role_is_not_exposed_per_missing_action"
            ),
            "instrumentation_limits": (
                "candidate_key_rows_and_real_replay_provenance_are_available"
                if source_contract_auditable
                else "candidate_key_rows_or_artifact_table_missing"
            ),
            "lower_specificity_grafting": "unsafe_non_authorizing",
        },
        "unsafe_non_authorizing_patterns": {
            "lower_specificity_graft_or_imputation": (
                "non_authorizing_unless_exact_same_high_specific_public_key_and_current_valid_action_support_is_proven"
            ),
            "default_stay_fallback": "unsafe_non_authorizing",
            "mock_provenance": "unsafe_non_authorizing",
            "blind_support_generation_or_expansion": "unsafe_non_authorizing",
            "gate_relaxation": "unsafe_non_authorizing",
        },
    }


def action_complete_source_contract() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v201_action_complete_source_contract_v1",
        "before_any_future_dataset_or_harness_work_must_prove": {
            "public_observation_identity": True,
            "public_action_mask_identity": True,
            "high_specific_source_key": True,
            "current_valid_action_set": True,
            "observed_support_count_for_every_current_valid_action": True,
            "real_replay_provenance": True,
            "no_private_world_state": True,
            "no_fixture_identity_or_seed_leakage": True,
            "source_report_exact_digest_pins": True,
            "source_dataset_or_artifact_exact_digest_pins": True,
            "split_role": True,
            "closed_lifecycle_before_any_training_authorization": True,
        },
        "lower_specificity_evidence_policy": (
            "non_authorizing_unless exact same high-specific public key and "
            "current-valid action support is proven"
        ),
        "direct_slice_4_training_allowed": False,
        "future_explicit_slice_4_training_route_authorized": False,
        "training_allowed": False,
        "support_generation_allowed": False,
        "support_expansion_allowed": False,
        "runtime_integration_allowed": False,
        "gate_relaxation_allowed": False,
        "promotion_authorized": False,
    }


def route_decision_for_v201(
    *,
    source_validation: Mapping[str, object],
    lineage_facts: Mapping[str, object],
    source_audit: Mapping[str, object],
) -> dict[str, object]:
    gap = _mapping(source_audit.get("gap_classification"))
    if source_validation.get("passed") is not True:
        route = STOP_ROUTE
    elif lineage_facts.get("passed") is not True:
        route = LINEAGE_REPAIR_ROUTE
    elif source_audit.get("source_contract_auditable_from_existing_artifacts") is not True:
        route = INSTRUMENTATION_REPAIR_ROUTE
    elif gap.get("lower_specificity_only_evidence_dominates") is True:
        route = PUBLIC_MODEL_CAPACITY_ROUTE
    elif (
        _int(gap.get("same_high_specific_missing_action_support_count")) > 0
        and _int(gap.get("same_high_specific_action_present_for_missing_demands_count"))
        == 0
    ):
        route = DATASET_CONTRACT_DESIGN_ROUTE
    else:
        route = INSTRUMENTATION_REPAIR_ROUTE
    return {
        "policy": "m3_carrion_survivor_continuation_v201_route_decision_v1",
        "recommended_next_route": route,
        "selected_route": route,
        "exactly_one_next_route_recommended": True,
        "source_pins_valid": source_validation.get("passed") is True,
        "lineage_facts_consistent": lineage_facts.get("passed") is True,
        "primary_blocker": source_audit.get("primary_blocker"),
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


def classification_for_v201(
    *,
    source_validation: Mapping[str, object],
    route_decision: Mapping[str, object],
) -> str:
    prefix = "m3_carrion_survivor_continuation_v201_"
    if source_validation.get("passed") is not True:
        return prefix + "source_pins_invalid_closed_no_training"
    route = str(route_decision.get("selected_route") or "")
    if route == PUBLIC_MODEL_CAPACITY_ROUTE:
        return (
            prefix
            + "lower_specificity_only_evidence_dominates_routes_to_public_masked_model_capacity_harness_contract_no_training"
        )
    if route == DATASET_CONTRACT_DESIGN_ROUTE:
        return (
            prefix
            + "same_high_specific_action_rows_absent_routes_to_dataset_contract_design_no_training"
        )
    if route == INSTRUMENTATION_REPAIR_ROUTE:
        return prefix + "source_contract_instrumentation_repair_no_training"
    if route == LINEAGE_REPAIR_ROUTE:
        return prefix + "v200_v199_lineage_repair_no_training"
    return prefix + "closed_no_training"


def historical_evidence_checkpoint() -> dict[str, object]:
    return {
        "policy": "m3_carrion_survivor_continuation_v201_historical_evidence_checkpoint_v1",
        "v53_v63": (
            "scalar IQL coefficient/prior/action-distribution/global-bias/"
            "extraction tuning is closed"
        ),
        "v154_v176": (
            "tiny support-archive / nearest-neighbor scorer loop is strategically saturated"
        ),
        "v180_v186_v195": (
            "training slices failed and are not promotion evidence; campaign budget remains 3/10"
        ),
        "v191_v193": (
            "same-tick occupancy drift was handled as a support-evidence contract issue"
        ),
        "v196": "low-specificity artifact coverage collapsed to eat with miss abstention",
        "v197": "low-specificity overrides were blocked without changing default runtime behavior",
        "v198": "high-specific keys existed but none were action-complete",
        "v199": (
            "54536 high-specific candidates were evaluated; 694 present "
            "occurrences were all action-incomplete"
        ),
        "v200": (
            "selected a source-contract audit before any future action-complete "
            "dataset/harness work"
        ),
        "closed_lanes": [
            "v180/v186/v195 training reruns",
            "scalar IQL tuning",
            "prior blends",
            "nearest-neighbor scorer/archive loop",
            "threshold sweeps",
            "imputed valid-action confidence",
            "default-stay fallback",
            "mock provenance",
            "blind support expansion",
        ],
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


def lower_specificity_keys(candidate_key: str) -> list[str]:
    parts = candidate_key.split("|")
    if not parts:
        return []
    mask = parts[0]
    self_part = next((part for part in parts if part.startswith("self=")), None)
    coarse_part = next((part for part in parts if part.startswith("coarse=")), None)
    keys = [mask]
    if coarse_part:
        keys.append(f"{mask}|{coarse_part}")
    if self_part:
        keys.append(f"{mask}|{self_part}")
    if self_part and coarse_part:
        keys.append(f"{mask}|{self_part}|{coarse_part}")
    return list(dict.fromkeys(keys))


def digest_without_exact(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(payload)


def route_rationale(route: str) -> str:
    if route == PUBLIC_MODEL_CAPACITY_ROUTE:
        return (
            "The artifact has no same high-specific support for the missing "
            "current-valid actions, and most missing demand is supported only "
            "by lower-specificity keys. Grafting that evidence is unsafe, so "
            "the next route must specify a no-training public masked model/"
            "capacity harness contract."
        )
    if route == DATASET_CONTRACT_DESIGN_ROUTE:
        return (
            "Existing artifacts confirm absent same high-specific action rows; "
            "the next route designs the action-complete dataset contract without "
            "training or support expansion."
        )
    if route == INSTRUMENTATION_REPAIR_ROUTE:
        return "Existing artifacts do not expose enough source-contract evidence."
    if route == LINEAGE_REPAIR_ROUTE:
        return "v200/v199 lineage facts are inconsistent and must be repaired first."
    if route == STOP_ROUTE:
        return "Required v197-v200 source pins or lifecycle facts failed."
    return "Closed no-training route."


def _load_optional_json_report(path: str | Path) -> tuple[dict[str, object], str | None]:
    try:
        return load_json_report(path), None
    except (OSError, ValueError) as exc:
        return {}, f"{type(exc).__name__}: {exc}"


def _combined_v199(v199_report: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(_mapping(v199_report.get("action_complete_audit")).get("combined"))


def _feature_action_table(
    v195_artifact: Mapping[str, object],
) -> Mapping[str, object]:
    return _mapping(_mapping(v195_artifact.get("utility_tables")).get("feature_action_utility"))


def _action_support_count(actions: Mapping[str, object], action: str) -> int:
    return _int(_mapping(actions.get(action)).get("count"))


def _counter_dict(value: object) -> dict[str, int]:
    return dict(
        sorted(
            {
                str(action): _int(count)
                for action, count in _mapping(value).items()
            }.items()
        )
    )


def _stats_breakdown(
    stats_by_key: Mapping[str, object],
    counter_name: str,
) -> dict[str, dict[str, int]]:
    return {
        str(key): _counter_dict(_mapping(stats).get(counter_name))
        for key, stats in sorted(stats_by_key.items())
    }


def _expected_count_checks(
    counts: Mapping[str, object],
    *,
    require_missing_action_counts: bool,
) -> dict[str, bool]:
    missing = _counter_dict(counts.get("missing_current_valid_action_counts"))
    evaluated = _int(counts.get("high_specificity_candidate_evaluated_count"))
    absent = _int(counts.get("key_absent_count"))
    present = _int(counts.get("key_present_count"))
    incomplete = _int(counts.get("present_but_action_incomplete_count"))
    floor_failed = _int(
        counts.get("present_complete_but_observed_support_floor_failed_count")
    )
    return {
        "high_specificity_candidate_evaluated_count_matches": (
            evaluated == EXPECTED_HIGH_SPECIFIC_CANDIDATES
        ),
        "key_absent_count_matches": absent == EXPECTED_KEY_ABSENT,
        "key_present_count_matches": present == EXPECTED_KEY_PRESENT,
        "present_but_action_incomplete_count_matches": (
            incomplete == EXPECTED_PRESENT_INCOMPLETE
        ),
        "support_floor_failure_count_matches": (
            floor_failed == EXPECTED_SUPPORT_FLOOR_FAILURES
        ),
        "evaluated_equals_absent_plus_present": evaluated == absent + present,
        "present_equals_incomplete_plus_support_floor_failures": (
            present == incomplete + floor_failed
        ),
        "missing_current_valid_action_counts_match": (
            missing == EXPECTED_MISSING_ACTION_COUNTS
            if require_missing_action_counts
            else True
        ),
    }


def _lifecycle_closed(
    report: Mapping[str, object],
    lifecycle_keys: tuple[str, ...],
) -> bool:
    return all(report.get(key) is False for key in lifecycle_keys) and (
        report.get("non_promoted") is True
    )


def _primary_blocker(
    *,
    source_contract_auditable: bool,
    lower_specificity_only_dominates: bool,
    high_specific_absent_confirmed: bool,
) -> str:
    if not source_contract_auditable:
        return "source_contract_instrumentation_missing"
    if lower_specificity_only_dominates:
        return "lower_specificity_only_evidence_dominates_missing_actions"
    if high_specific_absent_confirmed:
        return "same_high_specific_current_valid_action_rows_absent"
    return "source_contract_ambiguous"
