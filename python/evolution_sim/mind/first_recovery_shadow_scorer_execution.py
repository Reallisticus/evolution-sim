from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V124_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V124_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_ACCEPTED_RARE_ATTACK_CONTRACT_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _counter_to_dict,
    _int,
    _mapping,
    _resolve_json_report,
)
from evolution_sim.mind.first_recovery_repaired_label_split_support_feasibility import (
    _stratified_three_way_assignments,
    _trainable_leakage,
)
from evolution_sim.mind.first_recovery_shadow_ranker import (
    DOMINANT_SELECTED_ACTION_SHARE_MAX,
    MIN_MEANINGFUL_MATERIAL_GAIN_RECALL,
)
from evolution_sim.mind.first_recovery_shadow_scorer_proposal import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V125_REPORT_PATH,
    EXPECTED_MANIFEST_ROW_COUNT,
    EXPECTED_REPAIRED_ACTION_COUNTS,
    MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_PROPOSAL_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_EXECUTION_SCHEMA_VERSION = (
    "mind_v3_first_recovery_shadow_scorer_execution_v1"
)
MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_EXECUTION_POLICY = (
    "diagnostics_only_first_recovery_v126_shadow_scorer_execution_v1"
)
MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_PREDICTION_SCHEMA_VERSION = (
    "mind_v3_first_recovery_shadow_scorer_prediction_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v126-first-recovery-shadow-scorer-execution.json"
)
DEFAULT_PREDICTIONS_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v126-first-recovery-shadow-scorer-predictions.jsonl"
)

EXPECTED_V125_CLASSIFICATION = "shadow_scorer_proposal_ready_for_review"
EXPECTED_V124_CLASSIFICATION = "accepted_rare_attack_contract_ready_for_shadow_proposal"

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "shadow_scorer_execution_diagnostics_ready_for_review",
    "shadow_scorer_execution_source_integrity_failed",
    "shadow_scorer_execution_blocked_by_metrics",
    "diagnostics_only_no_runtime_promotion",
    "readiness_rerun_blocked",
)

V125_AUTHORIZATION_FIELDS_FALSE: tuple[str, ...] = (
    "shadow_scorer_executed",
    "shadow_scorer_execution_allowed",
    "downstream_shadow_scorer_allowed",
    "training_executed",
    "trained_artifact_change_recommended",
    "model_artifact_created",
    "runtime_policy_change_recommended",
    "v113_readiness_rerun_allowed",
    "gate_change_recommended",
    "viewer_change_recommended",
    "replay_golden_change_recommended",
    "foundation_change_recommended",
    "claim_causality",
)

REQUIRED_EXECUTION_OUTPUT_SECTIONS: tuple[str, ...] = (
    "heldout_signal",
    "seed29_evaluation",
    "fixture_open_evaluation",
    "material_gain_recall",
    "action_distribution",
    "unsupported_action_audit",
    "leakage_audit",
)

REPORT_ONLY_SCORER_RULE = "candidate_action_if_action_mask_supported_else_no_contract"
MATERIAL_GAIN_PROXY_ACTIONS = frozenset({"eat"})
MAX_EXAMPLES = 16


@dataclass(frozen=True, slots=True)
class FirstRecoveryShadowScorerExecutionBuild:
    report: dict[str, object]
    prediction_rows: tuple[dict[str, object], ...]


def build_first_recovery_shadow_scorer_execution(
    *,
    v125_report: Mapping[str, object] | None = None,
    v125_report_path: str | Path | None = DEFAULT_V125_REPORT_PATH,
    v124_report: Mapping[str, object] | None = None,
    v124_report_path: str | Path | None = DEFAULT_V124_REPORT_PATH,
    v124_manifest_rows: Sequence[Mapping[str, object]] | None = None,
    v124_manifest_path: str | Path | None = DEFAULT_V124_MANIFEST_PATH,
) -> FirstRecoveryShadowScorerExecutionBuild:
    v125_payload, v125_evidence = _resolve_json_report(
        v125_report,
        v125_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_PROPOSAL_SCHEMA_VERSION,
    )
    v124_payload, v124_evidence = _resolve_json_report(
        v124_report,
        v124_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_ACCEPTED_RARE_ATTACK_CONTRACT_SCHEMA_VERSION,
    )
    manifest_rows, manifest_evidence = _resolve_jsonl_rows(
        v124_manifest_rows,
        v124_manifest_path,
        row_kind="v124_manifest",
    )
    source_reports = {
        "v125_report": v125_evidence,
        "v124_report": v124_evidence,
        "v124_manifest": manifest_evidence,
    }
    source_integrity = _source_integrity(
        source_reports=source_reports,
        v125_report=v125_payload,
        v124_report=v124_payload,
        manifest_rows=manifest_rows,
    )
    prediction_rows = (
        _prediction_rows(v124_manifest_rows=manifest_rows)
        if source_integrity["passed"]
        else tuple()
    )
    candidate_set_audit = _candidate_set_audit(
        manifest_rows=manifest_rows,
        prediction_rows=prediction_rows,
    )
    input_allowlist_audit = _input_allowlist_audit(
        v125_report=v125_payload or {},
        manifest_rows=manifest_rows,
    )
    prediction_summary = _prediction_summary(prediction_rows)
    per_split_metrics = _per_split_metrics(prediction_rows)
    heldout_signal = _heldout_current_row_signal(
        per_split_metrics,
        candidate_set_audit=candidate_set_audit,
    )
    seed29 = _seed29_evaluation(
        prediction_rows,
        candidate_set_audit=candidate_set_audit,
    )
    fixture_open = _fixture_open_evaluation(
        prediction_rows,
        candidate_set_audit=candidate_set_audit,
    )
    material_gain = _material_gain_recall(prediction_rows)
    action_distribution = _action_distribution(prediction_rows)
    unsupported = _unsupported_action_audit(prediction_rows)
    leakage = _leakage_audit(manifest_rows)
    metric_gate = _metric_gate(
        source_integrity=source_integrity,
        candidate_set_audit=candidate_set_audit,
        heldout_signal=heldout_signal,
        seed29=seed29,
        fixture_open=fixture_open,
        material_gain=material_gain,
        action_distribution=action_distribution,
        unsupported=unsupported,
        leakage=leakage,
    )
    blocker_taxonomy = _blocker_taxonomy(
        v125_report=v125_payload or {},
        metric_gate=metric_gate,
    )
    classification = _classification(
        source_integrity=source_integrity,
        metric_gate=metric_gate,
    )
    recommendation = _recommendation(classification)
    report = {
        "schema_version": MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_EXECUTION_SCHEMA_VERSION,
        "audit_policy": MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_EXECUTION_POLICY,
        "contract": _contract(),
        "source_reports": source_reports,
        "source_integrity": source_integrity,
        "scorer_contract": _scorer_contract(),
        "input_allowlist_audit": input_allowlist_audit,
        "candidate_set_audit": candidate_set_audit,
        "prediction_summary": prediction_summary,
        "per_split_metrics": per_split_metrics,
        "heldout_current_row_signal": heldout_signal,
        "seed29_evaluation": seed29,
        "fixture_open_evaluation": fixture_open,
        "material_gain_recall": material_gain,
        "action_distribution": action_distribution,
        "unsupported_action_audit": unsupported,
        "leakage_audit": leakage,
        "metric_gate": metric_gate,
        "blocker_taxonomy": blocker_taxonomy,
        "classification": classification,
        "recommendation": recommendation,
        "non_promoted": True,
    }
    return FirstRecoveryShadowScorerExecutionBuild(
        report=report,
        prediction_rows=prediction_rows,
    )


def write_first_recovery_shadow_scorer_execution_outputs(
    build: FirstRecoveryShadowScorerExecutionBuild,
    *,
    output_path: str | Path,
    predictions_output_path: str | Path,
) -> None:
    predictions = Path(predictions_output_path)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    with predictions.open("w", encoding="utf-8") as handle:
        for row in build.prediction_rows:
            json.dump(row, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")

    report = dict(build.report)
    report["prediction_summary"] = {
        **_mapping(report.get("prediction_summary")),
        "prediction_rows_path": str(predictions),
        "prediction_row_count": len(build.prediction_rows),
        "prediction_rows_digest": stable_payload_digest(list(build.prediction_rows)),
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _contract() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_EXECUTION_SCHEMA_VERSION,
        "diagnostics_only": True,
        "shadow_only_evaluation": True,
        "training_executed": False,
        "model_artifact_created": False,
        "runtime_loadable_artifact_created": False,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "gate_effect": "none",
        "viewer_effect": "none",
        "replay_golden_effect": "none",
        "foundation_effect": "none",
        "readiness_rerun_executed": False,
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "claim_causality": False,
        "contract_digest": stable_payload_digest(
            {
                "schema_version": (
                    MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_EXECUTION_SCHEMA_VERSION
                ),
                "policy": MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_EXECUTION_POLICY,
            }
        ),
    }


def _resolve_jsonl_rows(
    rows: Sequence[Mapping[str, object]] | None,
    path: str | Path | None,
    *,
    row_kind: str,
) -> tuple[tuple[dict[str, object], ...], dict[str, object]]:
    if rows is not None:
        parsed = tuple(dict(row) for row in rows)
        return parsed, {
            "loaded": True,
            "in_memory": True,
            "path": str(path) if path is not None else None,
            "row_count": len(parsed),
            "row_kind": row_kind,
        }
    if path is None:
        return (), {
            "loaded": False,
            "path": None,
            "row_count": 0,
            "row_kind": row_kind,
            "error": "missing_jsonl_path",
        }
    source = Path(path)
    if not source.exists():
        return (), {
            "loaded": False,
            "path": str(source),
            "row_count": 0,
            "row_kind": row_kind,
            "error": "missing_jsonl_file",
        }
    parsed: list[dict[str, object]] = []
    try:
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"{row_kind} row {line_number} must be an object")
                parsed.append(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return (), {
            "loaded": False,
            "path": str(source),
            "row_count": 0,
            "row_kind": row_kind,
            "error": type(exc).__name__,
            "message": str(exc),
        }
    return tuple(parsed), {
        "loaded": True,
        "path": str(source),
        "row_count": len(parsed),
        "row_kind": row_kind,
    }


def _source_integrity(
    *,
    source_reports: Mapping[str, Mapping[str, object]],
    v125_report: Mapping[str, object] | None,
    v124_report: Mapping[str, object] | None,
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    failures: list[str] = []
    for name, evidence in source_reports.items():
        if evidence.get("loaded") is not True:
            failures.append(f"missing_{name}")
        elif evidence.get("schema_matches") is False:
            failures.append(f"{name}_schema_mismatch")

    v125 = _mapping(v125_report or {})
    v124 = _mapping(v124_report or {})
    v125_source = _mapping(v125.get("source_integrity"))
    v125_recommendation = _mapping(v125.get("recommendation"))
    v125_proposal = _mapping(v125.get("proposal"))
    v125_metrics = _mapping(v125_proposal.get("planned_scorer_acceptance_metrics"))
    v124_source = _mapping(v124.get("source_integrity"))
    v124_contract_checks = _mapping(v124.get("contract_checks"))
    v124_manifest = _mapping(v124.get("manifest"))
    v124_split = _mapping(v124.get("split_support"))
    v124_join = _mapping(v124_source.get("accepted_candidate_join_validation"))
    v124_row_replay = _mapping(
        v124_source.get("accepted_candidate_row_replay_validation")
    )
    manifest_digest = stable_payload_digest(list(manifest_rows))
    branch_ids = _valid_branch_ids(manifest_rows)
    repaired_counts = _counter_to_dict(_action_counts(manifest_rows))
    leakage = _trainable_leakage(manifest_rows)
    unsupported = _unsupported_manifest_selection(manifest_rows)
    metric_requirements = _v125_metric_requirements(v125_metrics)

    if _primary(v125) != EXPECTED_V125_CLASSIFICATION:
        failures.append("v125_classification_unexpected")
    if v125_source.get("passed") is not True:
        failures.append("v125_source_integrity_not_passed")
    if v125_source.get("failures") != []:
        failures.append("v125_source_failures_not_empty_or_malformed")
    boundary = _mapping(v125_source.get("v124_contract_boundary_checks"))
    if boundary.get("passed") is not True:
        failures.append("v125_v124_contract_boundary_not_passed")
    if boundary.get("failures") != []:
        failures.append("v125_v124_contract_boundary_failures_not_empty")
    if v125_source.get("manifest_digest") != manifest_digest:
        failures.append("v125_v124_manifest_digest_mismatch")
    for field in V125_AUTHORIZATION_FIELDS_FALSE:
        if v125_recommendation.get(field) is not False:
            failures.append(f"v125_{field}_not_false")
    failures.extend(metric_requirements["failures"])

    if _primary(v124) != EXPECTED_V124_CLASSIFICATION:
        failures.append("v124_classification_unexpected")
    if v124_source.get("passed") is not True:
        failures.append("v124_source_integrity_not_passed")
    if v124_source.get("failures") != []:
        failures.append("v124_source_failures_not_empty_or_malformed")
    if v124_contract_checks.get("passed") is not True:
        failures.append("v124_contract_checks_not_passed")
    if v124_contract_checks.get("failures") != []:
        failures.append("v124_contract_failures_not_empty_or_malformed")
    if _int(v124_manifest.get("manifest_row_count")) != EXPECTED_MANIFEST_ROW_COUNT:
        failures.append("v124_report_manifest_row_count_unexpected")
    if len(manifest_rows) != EXPECTED_MANIFEST_ROW_COUNT:
        failures.append("v124_manifest_row_count_unexpected")
    if _int(v124_contract_checks.get("manifest_row_count")) != len(manifest_rows):
        failures.append("v124_contract_manifest_row_count_mismatch")
    if _int(v124_contract_checks.get("unique_branch_count")) != len(set(branch_ids)):
        failures.append("v124_contract_unique_branch_count_mismatch")
    if len(branch_ids) != len(manifest_rows):
        failures.append("v124_manifest_branch_id_missing_or_malformed")
    if len(set(branch_ids)) != len(branch_ids):
        failures.append("v124_manifest_branch_ids_not_unique")
    if v124_manifest.get("manifest_digest") != manifest_digest:
        failures.append("v124_manifest_digest_mismatch")
    if repaired_counts != EXPECTED_REPAIRED_ACTION_COUNTS:
        failures.append("v124_manifest_repaired_action_counts_unexpected")
    if _dict_of_ints(v124_contract_checks.get("repaired_action_counts")) != (
        EXPECTED_REPAIRED_ACTION_COUNTS
    ):
        failures.append("v124_contract_repaired_action_counts_unexpected")
    if v124_split.get("strict_train_validation_test_support_met") is not True:
        failures.append("v124_strict_split_support_not_true")
    if v124_join.get("passed") is not True:
        failures.append("v124_join_validation_not_passed")
    if _int(v124_join.get("mismatch_count")) != 0:
        failures.append("v124_join_validation_mismatch_nonzero")
    if v124_row_replay.get("passed") is not True:
        failures.append("v124_row_replay_validation_not_passed")
    if _int(v124_row_replay.get("failure_count")) != 0:
        failures.append("v124_row_replay_validation_failures_nonzero")

    if leakage["split_key_leak_count"]:
        failures.append("manifest_trainable_split_leakage_detected")
    if leakage["forbidden_metadata_key_count"]:
        failures.append("manifest_trainable_metadata_leakage_detected")
    if unsupported["unsupported_selection_count"]:
        failures.append("manifest_unsupported_selection_detected")

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v125_classification_primary": _primary(v125),
        "v125_source_integrity_passed": v125_source.get("passed"),
        "v125_v124_contract_boundary_checks": dict(boundary),
        "v125_metric_requirements": metric_requirements,
        "v125_authorization_checks": {
            field: v125_recommendation.get(field)
            for field in V125_AUTHORIZATION_FIELDS_FALSE
        },
        "v124_classification_primary": _primary(v124),
        "v124_source_integrity_passed": v124_source.get("passed"),
        "manifest_row_count": len(manifest_rows),
        "unique_branch_count": len(set(branch_ids)),
        "manifest_digest": manifest_digest,
        "v124_reported_manifest_digest": v124_manifest.get("manifest_digest"),
        "v125_reported_manifest_digest": v125_source.get("manifest_digest"),
        "manifest_digest_matches_v124": v124_manifest.get("manifest_digest")
        == manifest_digest,
        "manifest_digest_matches_v125": v125_source.get("manifest_digest")
        == manifest_digest,
        "repaired_action_counts": repaired_counts,
        "expected_repaired_action_counts": EXPECTED_REPAIRED_ACTION_COUNTS,
        "strict_split_support": {
            "strict_train_validation_test_support_met": v124_split.get(
                "strict_train_validation_test_support_met"
            ),
            "all_splits_contain_every_action_class": v124_split.get(
                "all_splits_contain_every_action_class"
            ),
            "minimum_per_action_support_by_split": v124_split.get(
                "minimum_per_action_support_by_split"
            ),
        },
        "accepted_candidate_join_validation": dict(v124_join),
        "accepted_candidate_row_replay_validation": dict(v124_row_replay),
        "manifest_trainable_leakage": leakage,
        "unsupported_selection_report": unsupported,
    }


def _v125_metric_requirements(metrics: Mapping[str, object]) -> dict[str, object]:
    failures: list[str] = []
    heldout = _mapping(metrics.get("heldout_current_row_signal"))
    seed29 = _mapping(metrics.get("seed29_evaluation"))
    fixture_open = _mapping(metrics.get("fixture_open_generalization"))
    action_distribution = _mapping(metrics.get("action_distribution"))
    unsupported = _mapping(metrics.get("unsupported_action_audit"))
    leakage = _mapping(metrics.get("leakage_audit"))
    material = _mapping(metrics.get("material_gain_recall"))
    required_outputs = set(_list_like(metrics.get("required_execution_outputs")))

    if heldout.get("required") is not True:
        failures.append("v125_heldout_current_row_signal_requirement_missing")
    if seed29.get("required") is not True:
        failures.append("v125_seed29_requirement_missing")
    if fixture_open.get("required") is not True:
        failures.append("v125_fixture_open_requirement_missing")
    if _number(action_distribution.get("dominant_selected_action_share_max")) != (
        DOMINANT_SELECTED_ACTION_SHARE_MAX
    ):
        failures.append("v125_dominant_action_share_cap_unexpected")
    if _number(unsupported.get("unsupported_action_rate_required")) != 0.0:
        failures.append("v125_unsupported_action_rate_requirement_unexpected")
    if leakage.get("trainable_leakage_required") != 0:
        failures.append("v125_leakage_requirement_unexpected")
    if _number(material.get("minimum_recall")) != (
        MIN_MEANINGFUL_MATERIAL_GAIN_RECALL
    ):
        failures.append("v125_material_gain_recall_floor_unexpected")
    for section in REQUIRED_EXECUTION_OUTPUT_SECTIONS:
        if section not in required_outputs:
            failures.append(f"v125_required_output_{section}_missing")
    return {
        "passed": not failures,
        "failures": failures,
        "heldout_current_row_signal_required": heldout.get("required"),
        "seed29_required": seed29.get("required"),
        "fixture_open_required": fixture_open.get("required"),
        "dominant_selected_action_share_max": action_distribution.get(
            "dominant_selected_action_share_max"
        ),
        "unsupported_action_rate_required": unsupported.get(
            "unsupported_action_rate_required"
        ),
        "trainable_leakage_required": leakage.get("trainable_leakage_required"),
        "material_gain_recall_minimum": material.get("minimum_recall"),
        "required_execution_outputs": sorted(required_outputs),
    }


def _scorer_contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "shadow_only": True,
        "report_only_deterministic_rule": REPORT_ONLY_SCORER_RULE,
        "uses_trainable_public_input_only": True,
        "uses_seed_as_input": False,
        "uses_branch_id_as_input": False,
        "uses_archive_row_id_as_input": False,
        "uses_source_or_fixture_identity_as_input": False,
        "uses_logged_action_fallback_as_input": False,
        "uses_private_world_state_as_input": False,
        "uses_audit_metadata_as_input": False,
        "uses_objective_or_outcome_fields_as_input": False,
        "split_assignment_input_to_scorer": False,
        "coefficients_serialized": False,
        "runtime_loadable_artifact_created": False,
        "training_executed": False,
    }


def _input_allowlist_audit(
    *,
    v125_report: Mapping[str, object],
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    proposal = _mapping(v125_report.get("proposal"))
    allowlist = _mapping(proposal.get("planned_trainable_input_allowlist"))
    observed = sorted(
        {
            str(key)
            for row in manifest_rows
            for key in _mapping(row.get("trainable_public_input")).keys()
        }
    )
    planned = sorted(str(key) for key in _list_like(allowlist.get("top_level_keys_observed")))
    extra = sorted(set(observed) - set(planned)) if planned else []
    missing_required = sorted(
        set(str(key) for key in _list_like(allowlist.get("required_top_level_keys")))
        - set(observed)
    )
    leakage = _trainable_leakage(manifest_rows)
    return {
        "policy": "v125_allowlist_checked_against_loaded_v124_manifest",
        "observed_top_level_keys": observed,
        "v125_planned_top_level_keys": planned,
        "required_top_level_keys_missing": missing_required,
        "observed_keys_outside_v125_planned_allowlist": extra,
        "trainable_public_input_contents_embedded_in_v126": False,
        "split_assignment_excluded_from_trainable_public_input": True,
        "audit_metadata_excluded_from_scorer_input": True,
        "leakage": leakage,
    }


def _prediction_rows(
    *,
    v124_manifest_rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    assignments = _stratified_three_way_assignments(v124_manifest_rows)
    rows: list[dict[str, object]] = []
    for index, row in enumerate(sorted(v124_manifest_rows, key=_prediction_sort_key)):
        branch_id = row.get("branch_id")
        repaired_action = row.get("repaired_action")
        trainable = _mapping(row.get("trainable_public_input"))
        predicted_action, reason = _predict_action(trainable)
        unsupported_reasons = _unsupported_prediction_reasons(
            row=row,
            predicted_action=predicted_action,
        )
        material_gain_positive = _material_gain_positive(row)
        prediction = {
            "schema_version": (
                MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_PREDICTION_SCHEMA_VERSION
            ),
            "row_index": index,
            "branch_id": branch_id,
            "branch_id_role": "non_trainable_audit_metadata",
            "split": assignments.get(str(branch_id)),
            "split_assignment_role": "evaluation_group_only_not_scorer_input",
            "repaired_action": repaired_action,
            "predicted_action": predicted_action,
            "prediction_reason": reason,
            "prediction_correct": predicted_action == repaired_action,
            "unsupported_action": bool(unsupported_reasons),
            "unsupported_reasons": unsupported_reasons,
            "material_gain_positive_proxy": material_gain_positive,
            "material_gain_proxy_policy": (
                "repaired_action_equals_eat_from_v124_manifest"
            ),
            "material_gain_recalled": bool(
                material_gain_positive and predicted_action == repaired_action
            ),
            "scorer_input": {
                "source": "trainable_public_input",
                "contents_exposed": False,
                "digest": stable_payload_digest(trainable),
                "top_level_keys": sorted(str(key) for key in trainable.keys()),
            },
            "non_trainable_audit_metadata": {
                "branch_id": branch_id,
                "current_oracle_action": row.get("current_oracle_action"),
                "current_archive_row_id": row.get("current_archive_row_id"),
                "repaired_archive_row_id": row.get("repaired_archive_row_id"),
                "selected_observation_digest": row.get("selected_observation_digest"),
                "selected_resolution_legal": row.get("selected_resolution_legal"),
                "objective_equivalence_verified": row.get(
                    "objective_equivalence_verified"
                ),
                "unique_objective_best": row.get("unique_objective_best"),
                "changed": row.get("changed"),
                "legal_tied_candidate_actions": row.get(
                    "legal_tied_candidate_actions"
                ),
                "source_metadata": _mapping(
                    row.get("non_trainable_audit_metadata")
                ),
                "audit_only_not_trainable": True,
            },
            "diagnostics_only": True,
            "training_authorized": False,
            "runtime_policy_authorized": False,
            "readiness_authorized": False,
        }
        rows.append(prediction)
    return tuple(rows)


def _predict_action(trainable: Mapping[str, object]) -> tuple[str | None, str]:
    candidate_action = trainable.get("candidate_action")
    action_mask = _mapping(trainable.get("action_mask"))
    if (
        isinstance(candidate_action, str)
        and candidate_action
        and action_mask.get(candidate_action) is True
    ):
        return candidate_action, "candidate_action_supported_by_action_mask"
    allowed = sorted(
        str(action)
        for action, allowed_flag in action_mask.items()
        if allowed_flag is True and isinstance(action, str)
    )
    if allowed:
        return allowed[0], "deterministic_first_action_mask_fallback_no_contract"
    return None, "no_supported_action_in_trainable_public_action_mask"


def _unsupported_prediction_reasons(
    *,
    row: Mapping[str, object],
    predicted_action: object,
) -> list[str]:
    reasons: list[str] = []
    trainable = _mapping(row.get("trainable_public_input"))
    action_mask = _mapping(trainable.get("action_mask"))
    if not isinstance(predicted_action, str) or not predicted_action:
        reasons.append("predicted_action_missing_or_malformed")
    elif action_mask.get(predicted_action) is not True:
        reasons.append("predicted_action_not_in_public_action_mask")
    if predicted_action != row.get("repaired_action"):
        reasons.append("predicted_action_not_supported_by_repaired_label_contract")
    if row.get("selected_resolution_legal") is not True:
        reasons.append("selected_row_resolution_not_legal")
    if row.get("objective_equivalence_verified") is not True:
        reasons.append("selected_row_objective_equivalence_not_verified")
    if row.get("unique_objective_best") is True and row.get("changed") is True:
        reasons.append("unique_objective_best_branch_changed")
    return reasons


def _prediction_summary(
    prediction_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    correct = sum(1 for row in prediction_rows if row.get("prediction_correct") is True)
    unsupported = sum(1 for row in prediction_rows if row.get("unsupported_action") is True)
    return {
        "prediction_rows_path": str(DEFAULT_PREDICTIONS_OUTPUT_PATH),
        "prediction_row_count": len(prediction_rows),
        "prediction_rows_digest": stable_payload_digest(list(prediction_rows)),
        "correct_count": correct,
        "accuracy": _ratio(correct, len(prediction_rows)),
        "unsupported_action_count": unsupported,
        "unsupported_action_rate": _ratio(unsupported, len(prediction_rows)),
        "predicted_action_counts": _counter_to_dict(
            Counter(str(row.get("predicted_action")) for row in prediction_rows)
        ),
        "repaired_action_counts": _counter_to_dict(
            Counter(str(row.get("repaired_action")) for row in prediction_rows)
        ),
    }


def _candidate_set_audit(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    prediction_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    rows_by_branch: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in manifest_rows:
        branch_id = row.get("branch_id")
        if isinstance(branch_id, str) and branch_id:
            rows_by_branch[branch_id].append(row)

    predicted_equals_candidate = 0
    candidate_equals_repaired = 0
    predicted_equals_repaired = 0
    non_selected_candidate_rows = 0
    echo_examples: list[dict[str, object]] = []
    non_selected_examples: list[dict[str, object]] = []

    for row in manifest_rows:
        trainable = _mapping(row.get("trainable_public_input"))
        candidate_action = trainable.get("candidate_action")
        repaired_action = row.get("repaired_action")
        if candidate_action == repaired_action:
            candidate_equals_repaired += 1
            if len(echo_examples) < MAX_EXAMPLES:
                echo_examples.append(
                    {
                        "branch_id": row.get("branch_id"),
                        "candidate_action": candidate_action,
                        "repaired_action": repaired_action,
                    }
                )
        else:
            non_selected_candidate_rows += 1
            if len(non_selected_examples) < MAX_EXAMPLES:
                non_selected_examples.append(
                    {
                        "branch_id": row.get("branch_id"),
                        "candidate_action": candidate_action,
                        "repaired_action": repaired_action,
                    }
                )

    first_row_by_branch = {
        branch_id: branch_rows[0]
        for branch_id, branch_rows in rows_by_branch.items()
        if branch_rows
    }
    for prediction in prediction_rows:
        branch_id = prediction.get("branch_id")
        source_row = (
            first_row_by_branch.get(branch_id)
            if isinstance(branch_id, str)
            else None
        )
        candidate_action = _mapping(
            _mapping(source_row or {}).get("trainable_public_input")
        ).get("candidate_action")
        if prediction.get("predicted_action") == candidate_action:
            predicted_equals_candidate += 1
        if prediction.get("predicted_action") == prediction.get("repaired_action"):
            predicted_equals_repaired += 1

    branches_with_multiple = sorted(
        branch_id for branch_id, rows in rows_by_branch.items() if len(rows) > 1
    )
    manifest_row_count = len(manifest_rows)
    prediction_row_count = len(prediction_rows)
    positive_only = (
        manifest_row_count > 0
        and non_selected_candidate_rows == 0
        and not branches_with_multiple
    )
    label_echo = (
        prediction_row_count > 0
        and predicted_equals_candidate == prediction_row_count
        and candidate_equals_repaired == manifest_row_count
        and predicted_equals_repaired == prediction_row_count
    )
    return {
        "manifest_row_count": manifest_row_count,
        "prediction_row_count": prediction_row_count,
        "predicted_equals_candidate_action_count": predicted_equals_candidate,
        "predicted_equals_candidate_action_share": _ratio(
            predicted_equals_candidate,
            prediction_row_count,
        ),
        "candidate_action_equals_repaired_action_count": candidate_equals_repaired,
        "candidate_action_equals_repaired_action_share": _ratio(
            candidate_equals_repaired,
            manifest_row_count,
        ),
        "predicted_equals_repaired_action_count": predicted_equals_repaired,
        "predicted_equals_repaired_action_share": _ratio(
            predicted_equals_repaired,
            prediction_row_count,
        ),
        "non_selected_candidate_row_count": non_selected_candidate_rows,
        "branches_with_multiple_candidate_rows": len(branches_with_multiple),
        "branches_with_multiple_candidate_row_examples": branches_with_multiple[
            :MAX_EXAMPLES
        ],
        "positive_only_manifest_detected": positive_only,
        "candidate_action_label_echo_detected": label_echo,
        "candidate_ranking_evidence_available": (
            non_selected_candidate_rows > 0 and bool(branches_with_multiple)
        ),
        "candidate_action_label_echo_examples": echo_examples,
        "non_selected_candidate_examples": non_selected_examples,
    }


def _per_split_metrics(
    prediction_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for row in prediction_rows:
        split = row.get("split")
        if split in grouped:
            grouped[str(split)].append(row)
    return {
        split: _prediction_metrics(rows)
        for split, rows in grouped.items()
    }


def _heldout_current_row_signal(
    per_split_metrics: Mapping[str, object],
    *,
    candidate_set_audit: Mapping[str, object],
) -> dict[str, object]:
    validation = _mapping(per_split_metrics.get("validation"))
    test = _mapping(per_split_metrics.get("test"))
    validation_pass = (
        _int(validation.get("row_count")) > 0
        and _number(validation.get("accuracy")) is not None
        and _number(validation.get("dominant_repaired_action_baseline_share")) is not None
        and float(validation["accuracy"])
        > float(validation["dominant_repaired_action_baseline_share"])
        and _int(validation.get("unsupported_action_count")) == 0
    )
    test_pass = (
        _int(test.get("row_count")) > 0
        and _number(test.get("accuracy")) is not None
        and _number(test.get("dominant_repaired_action_baseline_share")) is not None
        and float(test["accuracy"]) > float(test["dominant_repaired_action_baseline_share"])
        and _int(test.get("unsupported_action_count")) == 0
    )
    label_echo = candidate_set_audit.get("candidate_action_label_echo_detected") is True
    positive_only = candidate_set_audit.get("positive_only_manifest_detected") is True
    passed = validation_pass and test_pass and not label_echo and not positive_only
    return {
        "passed": passed,
        "label": (
            "shadow_ranker_current_row_learns_signal"
            if passed
            else "heldout_row_group_label_echo_metrics_only"
            if label_echo or positive_only
            else "shadow_ranker_current_row_no_signal"
        ),
        "validation_passed": validation_pass,
        "test_passed": test_pass,
        "row_group_echo_metrics_only": label_echo or positive_only,
        "validation_accuracy": validation.get("accuracy"),
        "validation_baseline_share": validation.get(
            "dominant_repaired_action_baseline_share"
        ),
        "test_accuracy": test.get("accuracy"),
        "test_baseline_share": test.get("dominant_repaired_action_baseline_share"),
        "current_row_signal_required": True,
    }


def _seed29_evaluation(
    prediction_rows: Sequence[Mapping[str, object]],
    *,
    candidate_set_audit: Mapping[str, object],
) -> dict[str, object]:
    rows = [
        row
        for row in prediction_rows
        if str(
            _mapping(
                _mapping(row.get("non_trainable_audit_metadata")).get(
                    "source_metadata"
                )
            ).get("seed")
        )
        == "29"
    ]
    metrics = _prediction_metrics(rows)
    label_echo = candidate_set_audit.get("candidate_action_label_echo_detected") is True
    positive_only = candidate_set_audit.get("positive_only_manifest_detected") is True
    row_group_passed = (
        _int(metrics.get("row_count")) > 0
        and _number(metrics.get("accuracy")) is not None
        and float(metrics["accuracy"]) > 0.0
        and _int(metrics.get("unsupported_action_count")) == 0
    )
    passed = row_group_passed and not label_echo and not positive_only
    return {
        "passed": passed,
        "label": (
            "shadow_ranker_seed29_passes"
            if passed
            else "seed29_row_group_label_echo_metrics_only"
            if label_echo or positive_only
            else "shadow_ranker_seed29_fails"
        ),
        "seed": 29,
        "evaluation_group_role": "heldout_grouping_only_not_scorer_input",
        "row_group_echo_metrics_only": label_echo or positive_only,
        "row_group_metrics_passed": row_group_passed,
        "metrics": metrics,
    }


def _fixture_open_evaluation(
    prediction_rows: Sequence[Mapping[str, object]],
    *,
    candidate_set_audit: Mapping[str, object],
) -> dict[str, object]:
    groups: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in prediction_rows:
        source_meta = _mapping(
            _mapping(row.get("non_trainable_audit_metadata")).get("source_metadata")
        )
        group = (
            source_meta.get("source")
            or source_meta.get("source_kind")
            or source_meta.get("source_path")
            or "unknown_source"
        )
        groups[str(group)].append(row)
    group_reports = {
        group: _prediction_metrics(rows)
        for group, rows in sorted(groups.items())
    }
    label_echo = candidate_set_audit.get("candidate_action_label_echo_detected") is True
    positive_only = candidate_set_audit.get("positive_only_manifest_detected") is True
    row_group_passed = bool(group_reports) and all(
        _int(report.get("row_count")) > 0
        and _number(report.get("accuracy")) is not None
        and float(report["accuracy"]) > 0.0
        and _int(report.get("unsupported_action_count")) == 0
        for report in group_reports.values()
    )
    passed = row_group_passed and not label_echo and not positive_only
    return {
        "passed": passed,
        "label": (
            "shadow_ranker_fixture_open_generalizes"
            if passed
            else "fixture_open_row_group_label_echo_metrics_only"
            if label_echo or positive_only
            else "shadow_ranker_fixture_open_fails"
        ),
        "source_or_fixture_identity_used_as_scorer_input": False,
        "row_group_echo_metrics_only": label_echo or positive_only,
        "row_group_metrics_passed": row_group_passed,
        "group_count": len(group_reports),
        "groups": group_reports,
    }


def _material_gain_recall(
    prediction_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    positive = [
        row
        for row in prediction_rows
        if row.get("material_gain_positive_proxy") is True
    ]
    recalled = [
        row
        for row in positive
        if row.get("material_gain_recalled") is True
    ]
    recall = _ratio(len(recalled), len(positive))
    proxy_floor_satisfied = (
        recall >= MIN_MEANINGFUL_MATERIAL_GAIN_RECALL if positive else False
    )
    return {
        "passed": False,
        "label": "material_gain_exact_label_missing",
        "minimum_recall": MIN_MEANINGFUL_MATERIAL_GAIN_RECALL,
        "recall": recall,
        "material_gain_positive_count": len(positive),
        "material_gain_recalled_count": len(recalled),
        "material_gain_proxy_policy": "repaired_action_equals_eat_from_v124_manifest",
        "material_gain_floor_satisfied_by_proxy": proxy_floor_satisfied,
        "material_gain_floor_satisfied_by_exact_label": False,
        "exact_material_gain_label_available": False,
        "claim_exact_material_gain_label": False,
    }


def _action_distribution(
    prediction_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    counts = Counter(str(row.get("predicted_action")) for row in prediction_rows)
    dominant = _dominant_action(counts)
    passed = dominant["share"] <= DOMINANT_SELECTED_ACTION_SHARE_MAX
    return {
        "passed": passed,
        "label": (
            "shadow_ranker_action_distribution_clean"
            if passed
            else "shadow_ranker_action_distribution_collapsed"
        ),
        "predicted_action_counts": _counter_to_dict(counts),
        "dominant_predicted_action": dominant,
        "dominant_selected_action_share_max": DOMINANT_SELECTED_ACTION_SHARE_MAX,
    }


def _unsupported_action_audit(
    prediction_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    examples: list[dict[str, object]] = []
    count = 0
    for row in prediction_rows:
        if row.get("unsupported_action") is not True:
            continue
        count += 1
        if len(examples) < MAX_EXAMPLES:
            examples.append(
                {
                    "branch_id": row.get("branch_id"),
                    "split": row.get("split"),
                    "repaired_action": row.get("repaired_action"),
                    "predicted_action": row.get("predicted_action"),
                    "unsupported_reasons": row.get("unsupported_reasons"),
                }
            )
    rate = _ratio(count, len(prediction_rows))
    passed = count == 0
    return {
        "passed": passed,
        "label": (
            "shadow_ranker_unsupported_action_free"
            if passed
            else "shadow_ranker_unsupported_action_selected"
        ),
        "unsupported_action_count": count,
        "unsupported_action_rate": rate,
        "unsupported_action_rate_required": 0.0,
        "examples": examples,
    }


def _leakage_audit(
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    leakage = _trainable_leakage(manifest_rows)
    count = (
        _int(leakage.get("split_key_leak_count"))
        + _int(leakage.get("forbidden_metadata_key_count"))
    )
    passed = count == 0
    return {
        "passed": passed,
        "label": (
            "shadow_ranker_leakage_free"
            if passed
            else "shadow_ranker_leakage_detected"
        ),
        "trainable_leakage_count": count,
        "trainable_leakage_required": 0,
        "details": leakage,
    }


def _metric_gate(
    *,
    source_integrity: Mapping[str, object],
    candidate_set_audit: Mapping[str, object],
    heldout_signal: Mapping[str, object],
    seed29: Mapping[str, object],
    fixture_open: Mapping[str, object],
    material_gain: Mapping[str, object],
    action_distribution: Mapping[str, object],
    unsupported: Mapping[str, object],
    leakage: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    if source_integrity.get("passed") is not True:
        failures.append("source_integrity_failed")
    if candidate_set_audit.get("positive_only_manifest_detected") is True:
        failures.append("positive_only_manifest_no_candidate_ranking_evidence")
    if candidate_set_audit.get("candidate_action_label_echo_detected") is True:
        failures.append("candidate_action_label_echo_detected")
    if heldout_signal.get("passed") is not True:
        failures.append("heldout_signal_missing_or_weak")
    if seed29.get("passed") is not True:
        failures.append("seed29_failed")
    if fixture_open.get("passed") is not True:
        failures.append("fixture_open_failed")
    if material_gain.get("exact_material_gain_label_available") is not True:
        failures.append("material_gain_exact_label_missing")
    elif material_gain.get("passed") is not True:
        failures.append("material_gain_recall_below_floor")
    if action_distribution.get("passed") is not True:
        failures.append("action_collapse_detected")
    if unsupported.get("passed") is not True:
        failures.append("unsupported_selection_detected")
    if leakage.get("passed") is not True:
        failures.append("trainable_leakage_detected")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "dominant_predicted_action_share_passed": action_distribution.get("passed"),
        "candidate_ranking_evidence_available": candidate_set_audit.get(
            "candidate_ranking_evidence_available"
        ),
        "candidate_action_label_echo_detected": candidate_set_audit.get(
            "candidate_action_label_echo_detected"
        ),
        "positive_only_manifest_detected": candidate_set_audit.get(
            "positive_only_manifest_detected"
        ),
        "unsupported_action_rate_passed": unsupported.get("passed"),
        "trainable_leakage_passed": leakage.get("passed"),
        "material_gain_recall_passed": material_gain.get("passed"),
        "material_gain_floor_satisfied_by_exact_label": material_gain.get(
            "material_gain_floor_satisfied_by_exact_label"
        ),
        "seed29_reported_and_passed": seed29.get("passed"),
        "fixture_open_reported_and_passed": fixture_open.get("passed"),
        "validation_test_reported_separately": True,
    }


def _blocker_taxonomy(
    *,
    v125_report: Mapping[str, object],
    metric_gate: Mapping[str, object],
) -> dict[str, object]:
    planned = _mapping(_mapping(v125_report.get("proposal")).get("planned_blocker_taxonomy"))
    planned_items = (
        planned
        if isinstance(planned, list)
        else _mapping(v125_report.get("proposal")).get("planned_blocker_taxonomy")
    )
    planned_labels: list[str] = []
    if isinstance(planned_items, Sequence) and not isinstance(planned_items, (str, bytes)):
        for item in planned_items:
            if isinstance(item, Mapping) and isinstance(item.get("label"), str):
                planned_labels.append(str(item["label"]))
    required = [
        "source_integrity_failed",
        "trainable_leakage_detected",
        "positive_only_manifest_no_candidate_ranking_evidence",
        "candidate_action_label_echo_detected",
        "action_collapse_detected",
        "unsupported_selection_detected",
        "heldout_signal_missing_or_weak",
        "seed29_failed",
        "fixture_open_failed",
        "material_gain_exact_label_missing",
        "material_gain_recall_below_floor",
        "runtime_or_promotion_boundary_crossed",
    ]
    return {
        "planned_labels_from_v125": sorted(set(planned_labels)),
        "required_execution_labels": required,
        "active_blockers": metric_gate.get("failures"),
    }


def _classification(
    *,
    source_integrity: Mapping[str, object],
    metric_gate: Mapping[str, object],
) -> dict[str, object]:
    if source_integrity.get("passed") is not True:
        primary = "shadow_scorer_execution_source_integrity_failed"
    elif metric_gate.get("passed") is True:
        primary = "shadow_scorer_execution_diagnostics_ready_for_review"
    else:
        primary = "shadow_scorer_execution_blocked_by_metrics"
    return {
        "primary": primary,
        "labels": _dedupe_allowed(
            [primary, "diagnostics_only_no_runtime_promotion", "readiness_rerun_blocked"]
        ),
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(classification: Mapping[str, object]) -> dict[str, object]:
    ready = (
        classification.get("primary")
        == "shadow_scorer_execution_diagnostics_ready_for_review"
    )
    return {
        "next_step": (
            "review_v126_shadow_diagnostics_without_runtime_or_readiness_planning"
            if ready
            else "resolve_v126_source_or_metric_blockers_before_any_shadow_planning"
        ),
        "summary": (
            "v126 produced report-only shadow predictions and diagnostics. It does "
            "not authorize runtime planning, training, readiness, gates, replay "
            "changes, viewer changes, or promotion."
        ),
        "shadow_scorer_execution_diagnostic_completed": ready,
        "shadow_scorer_runtime_execution_authorized": False,
        "downstream_shadow_scorer_allowed": False,
        "training_executed": False,
        "trained_artifact_change_recommended": False,
        "model_artifact_created": False,
        "runtime_policy_change_recommended": False,
        "v113_readiness_rerun_allowed": False,
        "gate_change_recommended": False,
        "viewer_change_recommended": False,
        "replay_golden_change_recommended": False,
        "foundation_change_recommended": False,
        "claim_causality": False,
    }


def _prediction_metrics(
    prediction_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    row_count = len(prediction_rows)
    correct = sum(1 for row in prediction_rows if row.get("prediction_correct") is True)
    unsupported = sum(1 for row in prediction_rows if row.get("unsupported_action") is True)
    repaired_counts = Counter(str(row.get("repaired_action")) for row in prediction_rows)
    predicted_counts = Counter(str(row.get("predicted_action")) for row in prediction_rows)
    dominant_repaired = _dominant_action(repaired_counts)
    dominant_predicted = _dominant_action(predicted_counts)
    material_positive = sum(
        1
        for row in prediction_rows
        if row.get("material_gain_positive_proxy") is True
    )
    material_recalled = sum(
        1
        for row in prediction_rows
        if row.get("material_gain_recalled") is True
    )
    return {
        "row_count": row_count,
        "correct_count": correct,
        "accuracy": _ratio(correct, row_count),
        "repaired_action_counts": _counter_to_dict(repaired_counts),
        "predicted_action_counts": _counter_to_dict(predicted_counts),
        "dominant_repaired_action_baseline": dominant_repaired,
        "dominant_repaired_action_baseline_share": dominant_repaired["share"],
        "dominant_predicted_action": dominant_predicted,
        "unsupported_action_count": unsupported,
        "unsupported_action_rate": _ratio(unsupported, row_count),
        "material_gain_positive_count": material_positive,
        "material_gain_recalled_count": material_recalled,
        "material_gain_recall": _ratio(material_recalled, material_positive),
    }


def _unsupported_manifest_selection(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    count = 0
    examples: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        reason = None
        if row.get("selected_resolution_legal") is not True:
            reason = "selected_resolution_not_legal"
        elif row.get("objective_equivalence_verified") is not True:
            reason = "objective_equivalence_not_verified"
        elif row.get("unique_objective_best") is True and row.get("changed") is True:
            reason = "unique_objective_best_changed"
        if reason is None:
            continue
        count += 1
        if len(examples) < MAX_EXAMPLES:
            examples.append(
                {
                    "row_index": index,
                    "branch_id": row.get("branch_id"),
                    "repaired_action": row.get("repaired_action"),
                    "reason": reason,
                }
            )
    return {"unsupported_selection_count": count, "examples": examples}


def _material_gain_positive(row: Mapping[str, object]) -> bool:
    return str(row.get("repaired_action")) in MATERIAL_GAIN_PROXY_ACTIONS


def _action_counts(rows: Sequence[Mapping[str, object]]) -> Counter[str]:
    return Counter(str(row.get("repaired_action")) for row in rows)


def _valid_branch_ids(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return [
        str(row.get("branch_id"))
        for row in rows
        if isinstance(row.get("branch_id"), str) and row.get("branch_id")
    ]


def _dict_of_ints(value: object) -> dict[str, int]:
    return {str(key): _int(item) for key, item in _mapping(value).items()}


def _primary(report: Mapping[str, object]) -> object:
    return _mapping(report.get("classification")).get("primary")


def _dominant_action(counts: Counter[str]) -> dict[str, object]:
    if not counts:
        return {"action": None, "count": 0, "share": 0.0, "total": 0}
    action, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    total = sum(counts.values())
    return {
        "action": action,
        "count": count,
        "share": count / total if total else 0.0,
        "total": total,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _list_like(value: object) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return []


def _dedupe_allowed(labels: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    allowed = set(ALLOWED_CLASSIFICATIONS)
    result: list[str] = []
    for label in labels:
        if label not in allowed or label in seen:
            continue
        seen.add(label)
        result.append(label)
    return result


def _prediction_sort_key(row: Mapping[str, object]) -> tuple[str, str]:
    action = str(row.get("repaired_action"))
    branch_id = str(row.get("branch_id"))
    return (action, branch_id)
