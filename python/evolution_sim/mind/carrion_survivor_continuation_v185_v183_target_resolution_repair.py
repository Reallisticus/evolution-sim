from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path

from evolution_sim.mind import (
    carrion_survivor_continuation_v178_transition_row_dataset_audit as v178,
)
from evolution_sim.mind import (
    carrion_survivor_continuation_v184_v183_transition_row_dataset_audit as v184,
)
from evolution_sim.mind.candidate_campaign import _int, _mapping, write_json
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    exact_digest_validation_report,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    load_json_report,
)
from evolution_sim.mind.carrion_survivor_continuation_v171_replay_expansion import (
    load_jsonl_dataset,
)
from evolution_sim.mind.carrion_survivor_continuation_v179_exact_branch_transition_row_expansion import (
    transition_row_support_summary,
)
from evolution_sim.mind.carrion_survivor_continuation_v183_exact_transition_support_expansion import (
    DEFAULT_EXPANDED_TRANSITION_DATASET_OUTPUT_PATH as DEFAULT_V183_TRANSITION_DATASET_PATH,
)
from evolution_sim.mind.provenance import stable_payload_digest

M3_CARRION_SURVIVOR_CONTINUATION_V185_V183_TARGET_RESOLUTION_REPAIR_SCHEMA_VERSION = (
    v178.EXPECTED_V185_SCHEMA_VERSION
)
M3_CARRION_SURVIVOR_CONTINUATION_V185_V183_TARGET_RESOLUTION_REPAIR_POLICY = (
    v178.EXPECTED_V185_POLICY
)
M3_CARRION_SURVIVOR_CONTINUATION_V185_REPAIRED_TRANSITION_ROW_DATASET_AUDIT_SCHEMA_VERSION = (
    "m3_carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit_report_v1"
)
M3_CARRION_SURVIVOR_CONTINUATION_V185_REPAIRED_TRANSITION_ROW_DATASET_AUDIT_POLICY = (
    "diagnostics_only_m3_carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v185-carrion-survivor-continuation-v183-target-resolution-repair.json"
)
DEFAULT_REPAIRED_TRANSITION_DATASET_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v185-carrion-survivor-continuation-v183-target-resolution-repaired-compact-transition-rows.jsonl"
)
DEFAULT_REPAIRED_DATASET_AUDIT_OUTPUT_PATH = Path(
    "output/mind/"
    "mind-v3-v185-carrion-survivor-continuation-repaired-transition-row-dataset-audit.json"
)

V185_REPAIR_READY_CLASSIFICATION = v178.EXPECTED_V185_CLASSIFICATION
V185_REPAIR_SOURCE_INVALID_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v185_v183_target_resolution_repair_"
    "source_invalid_closed_no_training"
)
V185_REPAIR_CONTRACT_INVALID_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v185_v183_target_resolution_repair_"
    "target_resolution_contract_invalid_closed_no_training"
)
V185_REPAIR_SUPPORT_LIMITED_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v185_v183_target_resolution_repair_"
    "strict_filter_support_limited_closed_no_training"
)

V185_AUDIT_AUTHORIZED_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit_"
    "valid_support_ready_slice_2_training_route_authorized"
)
V185_AUDIT_SOURCE_INVALID_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit_"
    "source_invalid_closed_no_training"
)
V185_AUDIT_DATASET_CONTRACT_INVALID_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit_"
    "dataset_contract_invalid_closed_no_training"
)
V185_AUDIT_SUPPORT_LIMITED_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit_"
    "valid_support_limited_closed_no_training"
)
V185_AUDIT_RECHECK_REQUIRED_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit_"
    "valid_support_ready_default_threshold_recheck_required_no_training"
)
V185_AUDIT_DIGEST_PINS_REQUIRED_CLASSIFICATION = (
    "m3_carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit_"
    "valid_support_ready_digest_pins_required_no_training"
)


def run_carrion_survivor_continuation_v185_v183_target_resolution_repair(
    *,
    v184_report_path: str | Path = v184.DEFAULT_OUTPUT_PATH,
    v183_transition_dataset_path: str | Path = DEFAULT_V183_TRANSITION_DATASET_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    repaired_transition_dataset_output_path: str
    | Path = DEFAULT_REPAIRED_TRANSITION_DATASET_OUTPUT_PATH,
    expected_v184_report_exact_digest: str = v178.EXPECTED_V184_REPORT_EXACT_DIGEST,
    expected_v183_report_exact_digest: str = v178.EXPECTED_V183_REPORT_EXACT_DIGEST,
    expected_v183_dataset_digest: str = v178.EXPECTED_V183_DATASET_DIGEST,
) -> dict[str, object]:
    v184_report = load_json_report(v184_report_path)
    source_rows = [dict(row) for row in load_jsonl_dataset(v183_transition_dataset_path)]
    source_validation = validate_v185_sources(
        v184_report=v184_report,
        expected_v184_report_exact_digest=expected_v184_report_exact_digest,
        expected_v183_report_exact_digest=expected_v183_report_exact_digest,
        expected_v183_dataset_digest=expected_v183_dataset_digest,
    )
    source_dataset_digest = stable_payload_digest(source_rows)
    invalid_diagnostics = [
        _row_resolution_diagnostic(row, row_index=row_index)
        for row_index, row in enumerate(source_rows)
        if not _row_force_resolves_to_forced_action(row)
    ]
    repaired_rows = [
        dict(row)
        for row in source_rows
        if source_validation.get("passed") is True
        and _row_force_resolves_to_forced_action(row)
    ]
    _write_jsonl(repaired_transition_dataset_output_path, repaired_rows)
    repaired_dataset_digest = stable_payload_digest(repaired_rows)
    pre_target_audit = v178.transition_row_target_audit(source_rows)
    repaired_target_audit = v178.transition_row_target_audit(repaired_rows)
    support_summary = transition_row_support_summary(repaired_rows)
    trainable_resolution_scan = _trainable_resolution_metadata_scan(repaired_rows)
    repair_validation = _repair_validation(
        source_validation=source_validation,
        source_rows=source_rows,
        repaired_rows=repaired_rows,
        invalid_diagnostics=invalid_diagnostics,
        pre_target_audit=pre_target_audit,
        repaired_target_audit=repaired_target_audit,
        support_summary=support_summary,
        trainable_resolution_scan=trainable_resolution_scan,
    )
    classification = _repair_classification(
        source_validation=source_validation,
        repair_validation=repair_validation,
        support_summary=support_summary,
    )
    report = {
        "schema_version": (
            M3_CARRION_SURVIVOR_CONTINUATION_V185_V183_TARGET_RESOLUTION_REPAIR_SCHEMA_VERSION
        ),
        "policy": (
            M3_CARRION_SURVIVOR_CONTINUATION_V185_V183_TARGET_RESOLUTION_REPAIR_POLICY
        ),
        "contract": _repair_contract(),
        "inputs": {
            "v184_report": str(v184_report_path),
            "v183_transition_dataset": str(v183_transition_dataset_path),
            "repaired_transition_dataset_output": str(
                repaired_transition_dataset_output_path
            ),
            "expected_v184_report_exact_digest": expected_v184_report_exact_digest,
            "expected_v183_report_exact_digest": expected_v183_report_exact_digest,
            "expected_v183_dataset_digest": expected_v183_dataset_digest,
            "repair_strategy": "strict_filter_invalid_target_resolution_rows",
            "backfill_if_strict_filter_below_defaults": True,
            "backfill_required": bool(
                source_validation.get("passed") is True
                and support_summary.get("passed") is not True
            ),
            "backfill_implemented": False,
        },
        "source_validation": source_validation,
        "repair_validation": repair_validation,
        "support_summary": support_summary,
        "dataset": {
            "path": str(repaired_transition_dataset_output_path),
            "row_count": len(repaired_rows),
            "dataset_digest": repaired_dataset_digest,
            "source_dataset_path": str(v183_transition_dataset_path),
            "source_dataset_digest": source_dataset_digest,
            "source_report_path": str(v184_report_path),
            "source_report_exact_digest": v184_report.get("exact_digest"),
            "source_producer": v178.V185_SOURCE_PRODUCER,
        },
        "classification": {"primary": classification, "labels": [classification]},
        "route_recommendation": _repair_route_recommendation(
            classification=classification
        ),
        **_lifecycle_flags(diagnostic_dataset_created=bool(repaired_rows)),
    }
    report["exact_digest"] = _json_round_trip_digest(report)
    write_json(output_path, report)
    return report


def run_carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit(
    *,
    v185_report_path: str | Path = DEFAULT_OUTPUT_PATH,
    transition_dataset_path: str
    | Path = DEFAULT_REPAIRED_TRANSITION_DATASET_OUTPUT_PATH,
    output_path: str | Path = DEFAULT_REPAIRED_DATASET_AUDIT_OUTPUT_PATH,
    expected_v185_report_exact_digest: str | None = None,
    expected_dataset_digest: str | None = None,
    min_row_count: int = v178.DEFAULT_MIN_ROW_COUNT,
    min_seed_count: int = v178.DEFAULT_MIN_SEED_COUNT,
    min_branch_count: int = v178.DEFAULT_MIN_BRANCH_COUNT,
    min_forced_action_count: int = v178.DEFAULT_MIN_FORCED_ACTION_COUNT,
) -> dict[str, object]:
    audit = v178.run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
        transition_dataset_path=transition_dataset_path,
        v177_report_path=v185_report_path,
        output_path=output_path,
        expected_v177_report_exact_digest=expected_v185_report_exact_digest,
        expected_dataset_digest=expected_dataset_digest,
        min_row_count=min_row_count,
        min_seed_count=min_seed_count,
        min_branch_count=min_branch_count,
        min_forced_action_count=min_forced_action_count,
    )
    report = _repaired_audit_report(
        audit,
        v185_report_path=v185_report_path,
        transition_dataset_path=transition_dataset_path,
        expected_v185_report_exact_digest=expected_v185_report_exact_digest,
        expected_dataset_digest=expected_dataset_digest,
    )
    write_json(output_path, report)
    return report


def validate_v185_sources(
    *,
    v184_report: Mapping[str, object],
    expected_v184_report_exact_digest: str,
    expected_v183_report_exact_digest: str,
    expected_v183_dataset_digest: str,
) -> dict[str, object]:
    source = _mapping(v184_report.get("source_validation"))
    route = _mapping(v184_report.get("route_recommendation"))
    dataset = _mapping(v184_report.get("dataset"))
    target = _mapping(v184_report.get("target_audit"))
    classification = _mapping(v184_report.get("classification"))
    inputs = _mapping(v184_report.get("inputs"))
    exact = exact_digest_validation_report(v184_report)
    observed_v184_exact = str(v184_report.get("exact_digest") or "")
    observed_v183_exact = str(
        source.get("observed_source_report_exact_digest")
        or inputs.get("expected_v183_report_exact_digest")
        or ""
    )
    observed_v183_dataset_digest = str(dataset.get("dataset_digest") or "")
    checks = {
        "v184_schema_version_matches": (
            v184_report.get("schema_version")
            == v184.M3_CARRION_SURVIVOR_CONTINUATION_V184_V183_TRANSITION_ROW_DATASET_AUDIT_SCHEMA_VERSION
        ),
        "v184_policy_matches": (
            v184_report.get("policy")
            == v184.M3_CARRION_SURVIVOR_CONTINUATION_V184_V183_TRANSITION_ROW_DATASET_AUDIT_POLICY
        ),
        "v184_exact_digest_valid": exact.get("passed") is True,
        "v184_exact_digest_matches_expected": (
            observed_v184_exact == expected_v184_report_exact_digest
        ),
        "v184_classification_matches_expected": (
            classification.get("primary") == v184.V184_DATASET_CONTRACT_INVALID_CLASSIFICATION
        ),
        "v184_source_validation_passed": source.get("passed") is True,
        "v184_routes_to_v183_target_resolution_repair": (
            route.get("recommended_next_route")
            == "repair_v183_transition_rows_before_slice_2_training"
        ),
        "v184_source_producer_is_v183": (
            source.get("source_producer") == v178.V183_SOURCE_PRODUCER
        ),
        "v184_dataset_digest_matches_canonical_v183": (
            observed_v183_dataset_digest == expected_v183_dataset_digest
        ),
        "v184_v183_report_digest_matches_canonical": (
            observed_v183_exact == expected_v183_report_exact_digest
        ),
        "v184_target_audit_failed_with_expected_count": (
            target.get("passed") is False
            and _int(target.get("failure_count"), default=-1) == 14
        ),
        "v184_training_not_run": v184_report.get("training_ran") is False,
        "v184_training_artifact_not_created": (
            v184_report.get("training_artifact_created") is False
        ),
        "v184_slice_2_training_not_consumed": (
            v184_report.get("slice_2_training_consumed") is False
        ),
        "v184_runtime_action_selection_unchanged": (
            v184_report.get("runtime_action_selection_changed") is False
        ),
        "v184_promotion_not_authorized": (
            v184_report.get("promotion_authorized") is False
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "policy": "m3_carrion_survivor_continuation_v185_source_validation_v1",
        **checks,
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "expected_v184_report_exact_digest": expected_v184_report_exact_digest,
        "observed_v184_report_exact_digest": observed_v184_exact,
        "v184_exact_digest_validation": exact,
        "expected_v183_report_exact_digest": expected_v183_report_exact_digest,
        "observed_v183_report_exact_digest": observed_v183_exact,
        "expected_v183_dataset_digest": expected_v183_dataset_digest,
        "observed_v183_dataset_digest": observed_v183_dataset_digest,
        "v184_target_audit_failure_count": target.get("failure_count"),
        "v184_source_producer": source.get("source_producer"),
        "v184_route": route.get("recommended_next_route"),
    }


def _repair_validation(
    *,
    source_validation: Mapping[str, object],
    source_rows: Sequence[Mapping[str, object]],
    repaired_rows: Sequence[Mapping[str, object]],
    invalid_diagnostics: Sequence[Mapping[str, object]],
    pre_target_audit: Mapping[str, object],
    repaired_target_audit: Mapping[str, object],
    support_summary: Mapping[str, object],
    trainable_resolution_scan: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    if source_validation.get("passed") is not True:
        failures.append("source_validation_not_passed")
    if _int(pre_target_audit.get("failure_count"), default=-1) != 14:
        failures.append("pre_repair_target_failure_count_not_14")
    if len(invalid_diagnostics) != 7:
        failures.append("invalid_target_resolution_row_count_not_7")
    if repaired_target_audit.get("passed") is not True:
        failures.append("repaired_target_audit_not_passed")
    if support_summary.get("passed") is not True:
        failures.append("default_support_thresholds_not_met_after_strict_filter")
    if trainable_resolution_scan.get("passed") is not True:
        failures.append("runtime_resolution_validity_used_as_trainable_input")
    if len(repaired_rows) != len(source_rows) - len(invalid_diagnostics):
        failures.append("repaired_row_count_not_source_minus_invalid")
    all_repaired_rows_valid = all(
        _row_force_resolves_to_forced_action(row) for row in repaired_rows
    )
    if not all_repaired_rows_valid:
        failures.append("repaired_rows_do_not_force_resolve_to_forced_action")
    return {
        "policy": "m3_carrion_survivor_continuation_v185_repair_validation_v1",
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "strict_filter_used": True,
        "backfill_used": False,
        "backfill_required": support_summary.get("passed") is not True,
        "input_row_count": len(source_rows),
        "repaired_row_count": len(repaired_rows),
        "invalid_input_row_count": len(invalid_diagnostics),
        "removed_or_replaced_invalid_row_count": len(invalid_diagnostics),
        "invalid_target_resolution_rows": list(invalid_diagnostics),
        "pre_repair_target_audit": dict(pre_target_audit),
        "repaired_target_audit": dict(repaired_target_audit),
        "all_repaired_rows_force_resolve_to_forced_action": all_repaired_rows_valid,
        "required_row_target_fields": [
            "forced_action_used=true",
            "current_requested_action == forced_action",
            "current_resolved_action == forced_action",
            "current_action_valid=true",
            "current_resolution_action_valid=true",
        ],
        "runtime_resolution_validity_used_as_trainable_input": (
            trainable_resolution_scan.get("passed") is not True
        ),
        "trainable_resolution_metadata_scan": dict(trainable_resolution_scan),
    }


def _repaired_audit_report(
    audit: Mapping[str, object],
    *,
    v185_report_path: str | Path,
    transition_dataset_path: str | Path,
    expected_v185_report_exact_digest: str | None,
    expected_dataset_digest: str | None,
) -> dict[str, object]:
    report = dict(audit)
    source = dict(_mapping(report.get("source_validation")))
    authorization = dict(_mapping(report.get("training_authorization")))
    source_producer = str(source.get("source_producer") or "")
    if source_producer != v178.V185_SOURCE_PRODUCER:
        source["passed"] = False
        failures = [str(item) for item in source.get("failures") or []]
        failures.append("v185_expected_v185_repair_source_producer")
        source["failures"] = sorted(set(failures))
        source["failure_count"] = len(source["failures"])
        authorization["authorized"] = False
        authorization["training_authorized"] = False
        authorization["transition_row_training_authorized"] = False
        authorization["next_same_lane_opt_in_training_slice_authorized"] = False
        auth_failures = [str(item) for item in authorization.get("failures") or []]
        auth_failures.append("v185_expected_v185_repair_source_producer")
        authorization["failures"] = sorted(set(auth_failures))
        authorization["failure_count"] = len(authorization["failures"])
    report["source_validation"] = source
    report["training_authorization"] = authorization
    route = _repaired_audit_route_recommendation(
        _mapping(audit.get("route_recommendation")),
        source_validation=source,
        training_authorization=authorization,
    )
    classification = _repaired_audit_classification(
        v178_classification=str(
            _mapping(audit.get("classification")).get("primary") or ""
        ),
        route=route,
        source_validation=source,
    )
    report["schema_version"] = (
        M3_CARRION_SURVIVOR_CONTINUATION_V185_REPAIRED_TRANSITION_ROW_DATASET_AUDIT_SCHEMA_VERSION
    )
    report["policy"] = (
        M3_CARRION_SURVIVOR_CONTINUATION_V185_REPAIRED_TRANSITION_ROW_DATASET_AUDIT_POLICY
    )
    report["contract"] = _repaired_audit_contract(route)
    inputs = dict(_mapping(report.get("inputs")))
    inputs.update(
        {
            "source_report": str(v185_report_path),
            "v185_report": str(v185_report_path),
            "transition_dataset": str(transition_dataset_path),
            "expected_source_report_exact_digest": (
                expected_v185_report_exact_digest
            ),
            "expected_v185_report_exact_digest": expected_v185_report_exact_digest,
            "expected_dataset_digest": expected_dataset_digest,
            "source_report_must_be_v185_v183_target_resolution_repair": True,
            "future_training_route_if_authorized": (
                v178.V186_SLICE_2_TRAINING_ROUTE
                if route.get("slice_2_opt_in_training_route_authorized") is True
                else None
            ),
        }
    )
    report["inputs"] = inputs
    report["classification"] = {
        "primary": classification,
        "labels": [classification],
        "v178_style_audit_classification": _mapping(audit.get("classification")).get(
            "primary"
        ),
    }
    report["route_recommendation"] = route
    report["v178_style_audit_summary"] = {
        "schema_version": audit.get("schema_version"),
        "policy": audit.get("policy"),
        "classification": _mapping(audit.get("classification")).get("primary"),
        "exact_digest": audit.get("exact_digest"),
        "source_producer": source.get("source_producer"),
    }
    report["training_authorized"] = False
    report["training_artifact_created"] = False
    report["slice_2_training_consumed"] = False
    report["runtime_artifact_created"] = False
    report["runtime_action_selection_changed"] = False
    report["promotion_authorized"] = False
    report["exact_digest"] = _json_round_trip_digest(report)
    return report


def _repaired_audit_route_recommendation(
    v178_route: Mapping[str, object],
    *,
    source_validation: Mapping[str, object],
    training_authorization: Mapping[str, object],
) -> dict[str, object]:
    source_is_v185 = source_validation.get("source_producer") == v178.V185_SOURCE_PRODUCER
    authorized = (
        source_is_v185
        and source_validation.get("passed") is True
        and training_authorization.get("authorized") is True
        and v178_route.get("transition_row_training_authorized") is True
    )
    if authorized:
        next_route = v178.V186_SLICE_2_TRAINING_ROUTE
    elif not source_is_v185:
        next_route = "rerun_v185_audit_with_v185_repair_source_report_and_dataset"
    else:
        next_route = str(v178_route.get("recommended_next_route") or "closed")
    route = dict(v178_route)
    route.update(
        {
            "policy": (
                "m3_carrion_survivor_continuation_v185_repaired_audit_route_"
                "recommendation_v1"
            ),
            "recommended_next_route": next_route,
            "source_producer": source_validation.get("source_producer"),
            "transition_row_training_authorized": authorized,
            "training_authorized": authorized,
            "next_same_lane_opt_in_training_slice_authorized": authorized,
            "first_opt_in_training_slice_authorized": False,
            "slice_2_opt_in_training_route_authorized": authorized,
            "slice_2_training_authorized": authorized,
            "training_authorization_scope": (
                "next_same_lane_opt_in_slice_2_training" if authorized else "closed"
            ),
            "runtime_integration_authorized": False,
            "shadow_or_live_eval_authorized": False,
            "promotion_authorized": False,
        }
    )
    return route


def _repaired_audit_contract(route: Mapping[str, object]) -> dict[str, object]:
    authorized = route.get("slice_2_opt_in_training_route_authorized") is True
    return {
        "diagnostics_only": True,
        "training_allowed": False,
        "training_authorized_for_this_command": False,
        "next_same_lane_opt_in_training_slice_authorized": authorized,
        "first_opt_in_training_slice_authorized": False,
        "slice_2_opt_in_training_route_authorized": authorized,
        "slice_2_training_consumed": False,
        "training_route_if_explicitly_requested": (
            v178.V186_SLICE_2_TRAINING_ROUTE if authorized else None
        ),
        "fit_allowed": False,
        "runtime_artifact_allowed": False,
        "runtime_action_change_allowed": False,
        "shadow_or_live_eval_allowed": False,
        "promotion_allowed": False,
        "gate_relaxation_allowed": False,
        "source_report_must_be_v185_v183_target_resolution_repair": True,
        "runtime_resolution_validity_is_diagnostic_metadata_only": True,
        "public_action_masks_remain_trainable_features": True,
    }


def _repaired_audit_classification(
    *,
    v178_classification: str,
    route: Mapping[str, object],
    source_validation: Mapping[str, object],
) -> str:
    if source_validation.get("passed") is not True:
        return V185_AUDIT_SOURCE_INVALID_CLASSIFICATION
    if v178_classification.endswith("dataset_contract_invalid_closed_no_training"):
        return V185_AUDIT_DATASET_CONTRACT_INVALID_CLASSIFICATION
    if v178_classification.endswith("valid_support_limited_expand_before_training"):
        return V185_AUDIT_SUPPORT_LIMITED_CLASSIFICATION
    if v178_classification.endswith(
        "valid_support_ready_default_threshold_recheck_required_no_training"
    ):
        return V185_AUDIT_RECHECK_REQUIRED_CLASSIFICATION
    if v178_classification.endswith("valid_support_ready_digest_pins_required_no_training"):
        return V185_AUDIT_DIGEST_PINS_REQUIRED_CLASSIFICATION
    if route.get("slice_2_opt_in_training_route_authorized") is True:
        return V185_AUDIT_AUTHORIZED_CLASSIFICATION
    return V185_AUDIT_SOURCE_INVALID_CLASSIFICATION


def _repair_classification(
    *,
    source_validation: Mapping[str, object],
    repair_validation: Mapping[str, object],
    support_summary: Mapping[str, object],
) -> str:
    if source_validation.get("passed") is not True:
        return V185_REPAIR_SOURCE_INVALID_CLASSIFICATION
    if repair_validation.get("passed") is not True:
        if support_summary.get("passed") is not True:
            return V185_REPAIR_SUPPORT_LIMITED_CLASSIFICATION
        return V185_REPAIR_CONTRACT_INVALID_CLASSIFICATION
    return V185_REPAIR_READY_CLASSIFICATION


def _repair_route_recommendation(*, classification: str) -> dict[str, object]:
    ready = classification == V185_REPAIR_READY_CLASSIFICATION
    return {
        "policy": "m3_carrion_survivor_continuation_v185_route_recommendation_v1",
        "recommended_next_route": (
            "v185_repaired_transition_row_dataset_audit_before_slice_2_training"
            if ready
            else "repair_v183_transition_rows_before_slice_2_training"
        ),
        "repaired_dataset_audit_recommended": ready,
        "transition_row_training_authorized": False,
        "training_authorized": False,
        "slice_2_training_authorized": False,
        "slice_2_opt_in_training_route_authorized": False,
        "runtime_integration_authorized": False,
        "shadow_or_live_eval_authorized": False,
        "promotion_authorized": False,
    }


def _repair_contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "training_allowed": False,
        "training_authorized_for_this_command": False,
        "training_artifact_created": False,
        "slice_2_training_consumed": False,
        "fit_allowed": False,
        "runtime_artifact_allowed": False,
        "runtime_artifact_created": False,
        "runtime_integration_allowed": False,
        "runtime_action_change_allowed": False,
        "runtime_action_selection_changed": False,
        "runtime_semantics_changed": False,
        "default_runtime_behavior_changed": False,
        "shadow_or_live_eval_allowed": False,
        "promotion_allowed": False,
        "promotion_authorized": False,
        "gate_relaxation_allowed": False,
        "replay_viewer_schema_change_allowed": False,
        "repairs_v183_target_resolution_blocker": True,
        "strict_target_resolution_filter_required": True,
        "runtime_resolution_validity_is_diagnostic_metadata_only": True,
        "public_action_masks_remain_trainable_features": True,
        "repaired_dataset_audit_required_before_slice_2_training": True,
    }


def _lifecycle_flags(*, diagnostic_dataset_created: bool) -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "training_ran": False,
        "training_authorized": False,
        "training_artifact_created": False,
        "fit_ran": False,
        "scorer_retraining_ran": False,
        "scorer_retraining_authorized": False,
        "diagnostic_dataset_created": bool(diagnostic_dataset_created),
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
        "slice_2_training_consumed": False,
        "non_promoted": True,
    }


def _row_force_resolves_to_forced_action(row: Mapping[str, object]) -> bool:
    action = str(row.get("forced_action") or "")
    summary = _mapping(row.get("short_horizon_public_outcome_summary"))
    return (
        summary.get("forced_action_used") is True
        and summary.get("current_requested_action") == action
        and summary.get("current_resolved_action") == action
        and summary.get("current_action_valid") is True
        and summary.get("current_resolution_action_valid") is True
    )


def _row_resolution_diagnostic(
    row: Mapping[str, object],
    *,
    row_index: int,
) -> dict[str, object]:
    metadata = _mapping(row.get("metadata"))
    summary = _mapping(row.get("short_horizon_public_outcome_summary"))
    return {
        "row_index": int(row_index),
        "branch_id": metadata.get("branch_id"),
        "seed": metadata.get("seed"),
        "fixture": metadata.get("fixture"),
        "branch_tick": metadata.get("branch_tick"),
        "agent_id": metadata.get("agent_id"),
        "source_path": metadata.get("source_path"),
        "source_row_index": metadata.get("source_row_index"),
        "forced_action": row.get("forced_action"),
        "current_requested_action": summary.get("current_requested_action"),
        "current_resolved_action": summary.get("current_resolved_action"),
        "current_action_valid": summary.get("current_action_valid"),
        "current_resolution_action_valid": summary.get(
            "current_resolution_action_valid"
        ),
        "forced_action_used": summary.get("forced_action_used"),
    }


def _trainable_resolution_metadata_scan(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    forbidden_keys = {
        "current_action_valid",
        "current_requested_action",
        "current_resolution_action_valid",
        "current_resolved_action",
        "forced_action_used",
    }
    failures: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        _scan_forbidden_trainable_keys(
            value=_mapping(row.get("trainable_public_features")),
            row_index=row_index,
            path=("trainable_public_features",),
            forbidden_keys=forbidden_keys,
            failures=failures,
        )
    return {
        "policy": (
            "m3_carrion_survivor_continuation_v185_trainable_resolution_"
            "metadata_scan_v1"
        ),
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures[:64],
        "forbidden_runtime_resolution_metadata_keys": sorted(forbidden_keys),
        "public_action_masks_remain_trainable_features": True,
    }


def _scan_forbidden_trainable_keys(
    *,
    value: object,
    row_index: int,
    path: tuple[str, ...],
    forbidden_keys: set[str],
    failures: list[dict[str, object]],
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            next_path = (*path, key_text)
            if key_text in forbidden_keys:
                failures.append(
                    {
                        "row_index": row_index,
                        "path": ".".join(next_path),
                        "reason": "runtime_resolution_metadata_is_trainable_input",
                    }
                )
            _scan_forbidden_trainable_keys(
                value=item,
                row_index=row_index,
                path=next_path,
                forbidden_keys=forbidden_keys,
                failures=failures,
            )
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_forbidden_trainable_keys(
                value=item,
                row_index=row_index,
                path=(*path, str(index)),
                forbidden_keys=forbidden_keys,
                failures=failures,
            )


def _write_jsonl(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _json_round_trip_digest(report: Mapping[str, object]) -> str:
    payload = dict(report)
    payload.pop("exact_digest", None)
    return stable_payload_digest(
        json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    )
