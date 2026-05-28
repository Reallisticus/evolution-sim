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
from evolution_sim.mind.first_recovery_active_coverage_archive import (
    DEFAULT_ARCHIVE_ROWS_OUTPUT_PATH as DEFAULT_V123_ARCHIVE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V123_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_branch_archive import (
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    trainable_public_input_leakage,
)
from evolution_sim.mind.first_recovery_candidate_ranker_capacity_audit import (
    ACTION_ONLY_BASELINE,
    ACTION_ORDER_BASELINE,
    DEFAULT_OUTPUT_PATH as DEFAULT_V128_REPORT_PATH,
    EXPECTED_BRANCH_COUNT,
    EXPECTED_CANDIDATE_ROW_COUNT,
    EXPECTED_NEGATIVE_ROW_COUNT,
    EXPECTED_POSITIVE_ROW_COUNT,
    MIND_V3_FIRST_RECOVERY_CANDIDATE_RANKER_CAPACITY_AUDIT_SCHEMA_VERSION,
    _action_distribution,
    _fixture_open_evaluation,
    _heldout_accuracy,
    _leakage_audit,
    _list_like,
    _material_gain_recall,
    _metrics,
    _number,
    _positive_rate_table,
    _score_lookup,
    _seed29_evaluation,
    _source_integrity as _v128_reconstructed_source_integrity,
    _unsupported_action_audit,
)
from evolution_sim.mind.first_recovery_candidate_set_shadow_execution import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V127_REPORT_PATH,
    DEFAULT_PREDICTIONS_OUTPUT_PATH as DEFAULT_V127_PREDICTIONS_PATH,
    DEFAULT_V115_ARCHIVE_ROWS_PATH,
    DEFAULT_V115_REPORT_PATH,
    EXPECTED_V124_CLASSIFICATION,
    MIND_V3_FIRST_RECOVERY_CANDIDATE_SET_SHADOW_EXECUTION_SCHEMA_VERSION,
    _candidate_groups,
    _candidate_set_audit,
    _resolve_jsonl_rows,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _counter_to_dict,
    _int,
    _mapping,
    _resolve_archive_rows,
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
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION = (
    "mind_v3_first_recovery_candidate_public_feature_surface_v1"
)
MIND_V3_FIRST_RECOVERY_CANDIDATE_PUBLIC_FEATURE_SURFACE_POLICY = (
    "diagnostics_only_first_recovery_v129_candidate_public_feature_surface_v1"
)
MIND_V3_FIRST_RECOVERY_CANDIDATE_PUBLIC_FEATURE_ROW_SCHEMA_VERSION = (
    "mind_v3_first_recovery_candidate_public_feature_row_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v129-first-recovery-candidate-public-feature-surface.json"
)
DEFAULT_FEATURE_ROWS_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v129-first-recovery-candidate-public-feature-rows.jsonl"
)

PRIMARY_PROBE = "candidate_public_feature_surface_probe"

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "candidate_public_feature_surface_source_integrity_failed",
    "candidate_public_feature_surface_missing_candidate_specific_signal",
    "candidate_public_feature_surface_ready_for_ranker_probe",
)

FORBIDDEN_FEATURE_PATH_PARTS: frozenset[str] = frozenset(
    {
        "archive_id",
        "archive_row_id",
        "branch_id",
        "fixture",
        "fixture_identity",
        "fixture_name",
        "first_action_outcome",
        "material_gain",
        "material_gain_label",
        "objective",
        "oracle_rank",
        "provenance",
        "record_index",
        "recovery_vitals_deltas",
        "repaired_action",
        "repaired_archive_row_id",
        "replay_verification",
        "resolution_action_mask",
        "resolution_legal",
        "seed",
        "source",
        "source_path",
    }
)
FORBIDDEN_FEATURE_PATH_SUBSTRINGS: tuple[str, ...] = (
    "archive_row_id",
    "first_action_outcome",
    "material_gain_label",
    "oracle_rank",
    "recovery_vitals_deltas",
    "repaired_action",
    "repaired_archive_row_id",
    "replay_verification",
    "resolution_action_mask",
    "resolution_legal",
)
OPTIONAL_CANDIDATE_PUBLIC_CONTAINERS: tuple[str, ...] = (
    "candidate_public_features",
    "candidate_target_public_state",
    "candidate_resource_public_state",
    "candidate_neighbor_public_state",
    "candidate_neighborhood_public_state",
    "candidate_action_public_semantics",
)
OPTIONAL_BY_ACTION_PUBLIC_CONTAINERS: tuple[str, ...] = (
    "candidate_public_features_by_action",
    "target_public_state_by_action",
    "resource_public_state_by_action",
    "neighbor_public_state_by_action",
    "neighborhood_public_state_by_action",
    "action_public_semantics_by_action",
)
ACTION_NAMES: tuple[str, ...] = (
    "attack_east",
    "attack_north",
    "attack_south",
    "attack_west",
    "drink",
    "eat",
    "mate",
    "move_east",
    "move_north",
    "move_south",
    "move_west",
    "stay",
)
MAX_EXAMPLES = 16


@dataclass(frozen=True, slots=True)
class FirstRecoveryCandidatePublicFeatureSurfaceBuild:
    report: dict[str, object]
    feature_rows: tuple[dict[str, object], ...]


def build_first_recovery_candidate_public_feature_surface(
    *,
    v128_report: Mapping[str, object] | None = None,
    v128_report_path: str | Path | None = DEFAULT_V128_REPORT_PATH,
    v127_report: Mapping[str, object] | None = None,
    v127_report_path: str | Path | None = DEFAULT_V127_REPORT_PATH,
    v127_prediction_rows: Sequence[Mapping[str, object]] | None = None,
    v127_predictions_path: str | Path | None = DEFAULT_V127_PREDICTIONS_PATH,
    v124_report: Mapping[str, object] | None = None,
    v124_report_path: str | Path | None = DEFAULT_V124_REPORT_PATH,
    v124_manifest_rows: Sequence[Mapping[str, object]] | None = None,
    v124_manifest_path: str | Path | None = DEFAULT_V124_MANIFEST_PATH,
    v115_report: Mapping[str, object] | None = None,
    v115_report_path: str | Path | None = DEFAULT_V115_REPORT_PATH,
    v115_archive_rows: Sequence[Mapping[str, object]] | None = None,
    v115_archive_rows_path: str | Path | None = DEFAULT_V115_ARCHIVE_ROWS_PATH,
    v123_report: Mapping[str, object] | None = None,
    v123_report_path: str | Path | None = DEFAULT_V123_REPORT_PATH,
    v123_archive_rows: Sequence[Mapping[str, object]] | None = None,
    v123_archive_rows_path: str | Path | None = DEFAULT_V123_ARCHIVE_ROWS_PATH,
) -> FirstRecoveryCandidatePublicFeatureSurfaceBuild:
    v128_payload, v128_evidence = _resolve_json_report(
        v128_report,
        v128_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_CANDIDATE_RANKER_CAPACITY_AUDIT_SCHEMA_VERSION,
    )
    v127_payload, v127_evidence = _resolve_json_report(
        v127_report,
        v127_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_CANDIDATE_SET_SHADOW_EXECUTION_SCHEMA_VERSION,
    )
    v127_predictions, v127_predictions_evidence = _resolve_jsonl_rows(
        v127_prediction_rows,
        v127_predictions_path,
        row_kind="v127_predictions",
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
    v115_payload, v115_evidence = _resolve_json_report(
        v115_report,
        v115_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    )
    v115_rows, v115_rows_evidence = _resolve_archive_rows(
        v115_archive_rows,
        v115_archive_rows_path,
    )
    v123_payload, v123_evidence = _resolve_json_report(
        v123_report,
        v123_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    )
    v123_rows, v123_rows_evidence = _resolve_archive_rows(
        v123_archive_rows,
        v123_archive_rows_path,
    )
    source_reports = {
        "v128_report": v128_evidence,
        "v127_report": v127_evidence,
        "v127_predictions": v127_predictions_evidence,
        "v124_report": v124_evidence,
        "v124_manifest": manifest_evidence,
        "v115_report": v115_evidence,
        "v115_archive_rows": v115_rows_evidence,
        "v123_report": v123_evidence,
        "v123_archive_rows": v123_rows_evidence,
    }
    candidate_rows = tuple(v115_rows) + tuple(v123_rows)
    candidate_groups = _candidate_groups(candidate_rows)
    reconstructed_audit = _candidate_set_audit(
        manifest_rows=manifest_rows,
        candidate_groups=candidate_groups,
        candidate_rows=candidate_rows,
        v115_rows=v115_rows,
        v123_rows=v123_rows,
    )
    reconstructed_v128_source = _v128_reconstructed_source_integrity(
        source_reports={
            "v127_report": v127_evidence,
            "v127_predictions": v127_predictions_evidence,
            "v124_manifest": manifest_evidence,
            "v115_report": v115_evidence,
            "v115_archive_rows": v115_rows_evidence,
            "v123_report": v123_evidence,
            "v123_archive_rows": v123_rows_evidence,
        },
        v127_report=v127_payload,
        v127_predictions=v127_predictions,
        manifest_rows=manifest_rows,
        v115_report=v115_payload,
        v115_rows=v115_rows,
        v123_report=v123_payload,
        v123_rows=v123_rows,
        candidate_rows=candidate_rows,
        reconstructed_audit=reconstructed_audit,
    )
    dataset, feature_rows = _candidate_public_feature_dataset(
        manifest_rows=manifest_rows,
        candidate_groups=candidate_groups,
    )
    feature_allowlist = _feature_allowlist_audit(dataset)
    forbidden_scan = _forbidden_feature_scan(dataset)
    candidate_signal_inventory = _candidate_signal_inventory(dataset, candidate_rows)
    within_branch_variance = _within_branch_feature_variance(
        v128_report=v128_payload,
        dataset=dataset,
        manifest_rows=manifest_rows,
    )
    source_integrity = _source_integrity(
        source_reports=source_reports,
        v128_report=v128_payload,
        v127_report=v127_payload,
        v124_report=v124_payload,
        manifest_rows=manifest_rows,
        candidate_rows=candidate_rows,
        reconstructed_audit=reconstructed_audit,
        reconstructed_v128_source=reconstructed_v128_source,
    )
    probes = (
        _run_probes(dataset)
        if source_integrity["passed"]
        and feature_allowlist["passed"]
        and forbidden_scan["passed"]
        and reconstructed_audit["candidate_sets_available"]
        else _empty_probe_reports()
    )
    comparisons = _probe_comparisons(probes)
    primary = _mapping(probes.get(PRIMARY_PROBE))
    leakage = _leakage_audit(
        manifest_rows=manifest_rows,
        candidate_rows=candidate_rows,
    )
    metric_gate = _metric_gate(
        source_integrity=source_integrity,
        feature_allowlist=feature_allowlist,
        forbidden_scan=forbidden_scan,
        candidate_signal_inventory=candidate_signal_inventory,
        within_branch_variance=within_branch_variance,
        probes=probes,
        comparisons=comparisons,
        primary=primary,
        leakage=leakage,
    )
    classification = _classification(
        source_integrity=source_integrity,
        feature_allowlist=feature_allowlist,
        forbidden_scan=forbidden_scan,
        metric_gate=metric_gate,
    )
    recommendation = _recommendation(classification)
    report = {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION
        ),
        "audit_policy": MIND_V3_FIRST_RECOVERY_CANDIDATE_PUBLIC_FEATURE_SURFACE_POLICY,
        "contract": _contract(),
        "source_reports": source_reports,
        "source_integrity": source_integrity,
        "feature_rules": _feature_rules(),
        "feature_allowlist_audit": feature_allowlist,
        "forbidden_feature_scan": forbidden_scan,
        "candidate_specific_public_signal_inventory": candidate_signal_inventory,
        "within_branch_feature_variance": within_branch_variance,
        "feature_surface_summary": _feature_surface_summary(dataset, feature_rows),
        "probe_definitions": _probe_definitions(),
        "probe_reports": probes,
        "probe_comparisons": comparisons,
        "per_split_metrics": primary.get("per_split_metrics", {}),
        "heldout_evaluation": _heldout_evaluation(probes, PRIMARY_PROBE),
        "seed29_evaluation": primary.get("seed29_evaluation", {}),
        "fixture_open_evaluation": primary.get("fixture_open_evaluation", {}),
        "action_distribution": primary.get("action_distribution", {}),
        "unsupported_action_audit": primary.get("unsupported_action_audit", {}),
        "leakage_audit": leakage,
        "material_exact_match_recall": primary.get("material_gain_recall", {}),
        "metric_gate": metric_gate,
        "classification": classification,
        "recommendation": recommendation,
        "non_promoted": True,
    }
    return FirstRecoveryCandidatePublicFeatureSurfaceBuild(
        report=report,
        feature_rows=tuple(feature_rows),
    )


def write_first_recovery_candidate_public_feature_surface_report(
    build: FirstRecoveryCandidatePublicFeatureSurfaceBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build.report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def write_first_recovery_candidate_public_feature_rows(
    build: FirstRecoveryCandidatePublicFeatureSurfaceBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in build.feature_rows:
            json.dump(row, handle, sort_keys=True, allow_nan=False)
            handle.write("\n")


def _contract() -> dict[str, object]:
    return {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_CANDIDATE_PUBLIC_FEATURE_SURFACE_SCHEMA_VERSION
        ),
        "diagnostics_only": True,
        "report_only_public_feature_surface": True,
        "reads_v115_v123_v124_v127_v128_only": True,
        "training_pipeline_changed": False,
        "training_executed": False,
        "runtime_loadable_artifact_created": False,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "readiness_rerun_executed": False,
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "gate_effect": "none",
        "viewer_effect": "none",
        "replay_golden_effect": "none",
        "foundation_effect": "none",
        "claim_causality": False,
    }


def _source_integrity(
    *,
    source_reports: Mapping[str, Mapping[str, object]],
    v128_report: Mapping[str, object] | None,
    v127_report: Mapping[str, object] | None,
    v124_report: Mapping[str, object] | None,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    reconstructed_audit: Mapping[str, object],
    reconstructed_v128_source: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    for name, evidence in source_reports.items():
        if evidence.get("loaded") is not True:
            failures.append(f"missing_{name}")
        elif evidence.get("schema_matches") is False:
            failures.append(f"{name}_schema_mismatch")

    v128 = _mapping(v128_report or {})
    v127 = _mapping(v127_report or {})
    v124 = _mapping(v124_report or {})
    manifest_digest = stable_payload_digest(list(manifest_rows))
    repaired_counts = _counter_to_dict(Counter(_row_action(row) for row in manifest_rows))
    manifest_leakage = _trainable_leakage(manifest_rows)
    candidate_leakage = trainable_public_input_leakage(candidate_rows)

    if reconstructed_v128_source.get("passed") is not True:
        failures.append("reconstructed_v128_source_integrity_failed")
    for failure in _list_like(reconstructed_v128_source.get("failures")):
        failures.append(f"reconstructed_{failure}")

    v128_source = _mapping(v128.get("source_integrity"))
    v128_variance = _v128_baseline_variance(v128)
    if _mapping(v128.get("classification")).get("primary") not in {
        "candidate_ranker_capacity_ready_for_review",
        "candidate_ranker_capacity_blocked_by_signal",
    }:
        failures.append("v128_classification_unexpected")
    if v128_source.get("passed") is not True:
        failures.append("v128_source_integrity_not_passed")
    if v128_source.get("failures") != []:
        failures.append("v128_source_failures_not_empty_or_malformed")
    if _int(v128_source.get("branch_count")) != EXPECTED_BRANCH_COUNT:
        failures.append("v128_branch_count_unexpected")
    if _int(v128_source.get("candidate_row_count")) != EXPECTED_CANDIDATE_ROW_COUNT:
        failures.append("v128_candidate_row_count_unexpected")
    if _int(v128_source.get("positive_row_count")) != EXPECTED_POSITIVE_ROW_COUNT:
        failures.append("v128_positive_row_count_unexpected")
    if _int(v128_source.get("negative_row_count")) != EXPECTED_NEGATIVE_ROW_COUNT:
        failures.append("v128_negative_row_count_unexpected")
    if v128_variance.get("only_candidate_action_and_index_vary") is not True:
        failures.append("v128_baseline_variance_not_action_only")

    v127_source = _mapping(v127.get("source_integrity"))
    if v127_source.get("passed") is not True:
        failures.append("v127_source_integrity_not_passed")
    if v127_source.get("failures") != []:
        failures.append("v127_source_failures_not_empty_or_malformed")

    v124_source = _mapping(v124.get("source_integrity"))
    v124_contract = _mapping(v124.get("contract_checks"))
    v124_manifest = _mapping(v124.get("manifest"))
    if _mapping(v124.get("classification")).get("primary") != EXPECTED_V124_CLASSIFICATION:
        failures.append("v124_classification_unexpected")
    if v124_source.get("passed") is not True:
        failures.append("v124_source_integrity_not_passed")
    if v124_source.get("failures") != []:
        failures.append("v124_source_failures_not_empty_or_malformed")
    if v124_contract.get("passed") is not True:
        failures.append("v124_contract_checks_not_passed")
    if v124_contract.get("failures") != []:
        failures.append("v124_contract_failures_not_empty_or_malformed")
    if v124_manifest.get("manifest_digest") != manifest_digest:
        failures.append("v124_manifest_digest_mismatch")
    if _int(v124_manifest.get("manifest_row_count")) != len(manifest_rows):
        failures.append("v124_manifest_row_count_mismatch")
    if len(manifest_rows) != EXPECTED_BRANCH_COUNT:
        failures.append("manifest_row_count_unexpected")

    if _int(reconstructed_audit.get("branch_count")) != EXPECTED_BRANCH_COUNT:
        failures.append("reconstructed_branch_count_unexpected")
    if _int(reconstructed_audit.get("total_candidate_rows")) != EXPECTED_CANDIDATE_ROW_COUNT:
        failures.append("reconstructed_candidate_row_count_unexpected")
    if _int(reconstructed_audit.get("positive_rows_count")) != EXPECTED_POSITIVE_ROW_COUNT:
        failures.append("reconstructed_positive_row_count_unexpected")
    if _int(reconstructed_audit.get("negative_rows_count")) != EXPECTED_NEGATIVE_ROW_COUNT:
        failures.append("reconstructed_negative_row_count_unexpected")
    reconstructed_join = _mapping(reconstructed_audit.get("repaired_archive_row_join"))
    if _int(reconstructed_join.get("exact_join_count")) != EXPECTED_BRANCH_COUNT:
        failures.append("reconstructed_exact_repaired_archive_row_join_failed")
    if _int(reconstructed_audit.get("unsupported_repaired_labels_count")) != 0:
        failures.append("reconstructed_unsupported_repaired_labels_nonzero")
    if manifest_leakage["split_key_leak_count"] or manifest_leakage["forbidden_metadata_key_count"]:
        failures.append("manifest_trainable_leakage_detected")
    if candidate_leakage.get("leak_count"):
        failures.append("candidate_trainable_leakage_detected")

    authorization = {
        "v124": _authorization_boundary(v124),
        "v127": _authorization_boundary(v127),
        "v128": _authorization_boundary(v128),
    }
    for version, summary in authorization.items():
        for failure in _list_like(summary.get("failures")):
            failures.append(f"{version}_{failure}")

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "manifest_row_count": len(manifest_rows),
        "manifest_digest": manifest_digest,
        "manifest_repaired_action_counts": repaired_counts,
        "branch_count": _int(reconstructed_audit.get("branch_count")),
        "candidate_row_count": len(candidate_rows),
        "positive_row_count": _int(reconstructed_audit.get("positive_rows_count")),
        "negative_row_count": _int(reconstructed_audit.get("negative_rows_count")),
        "exact_repaired_archive_row_joins": _int(reconstructed_join.get("exact_join_count")),
        "v128_source_integrity_passed": v128_source.get("passed"),
        "v127_source_integrity_passed": v127_source.get("passed"),
        "v124_source_integrity_passed": v124_source.get("passed"),
        "v124_manifest_digest_matches": v124_manifest.get("manifest_digest") == manifest_digest,
        "v128_baseline_variance": v128_variance,
        "candidate_set_audit": dict(reconstructed_audit),
        "reconstructed_v128_source_integrity": dict(reconstructed_v128_source),
        "manifest_trainable_leakage": manifest_leakage,
        "candidate_trainable_leakage": candidate_leakage,
        "authorization_summary": authorization,
    }


def _authorization_boundary(report: Mapping[str, object]) -> dict[str, object]:
    failures: list[str] = []
    recommendation = _mapping(report.get("recommendation"))
    contract = _mapping(report.get("contract"))
    false_fields = (
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
    none_effect_fields = (
        "runtime_policy_effect",
        "trained_artifact_effect",
        "gate_effect",
        "viewer_effect",
        "replay_golden_effect",
        "foundation_effect",
    )
    checked: dict[str, object] = {}
    for field in false_fields:
        value = recommendation.get(field, contract.get(field))
        if value is not None:
            checked[field] = value
            if value is not False:
                failures.append(f"{field}_authorized")
    for field in none_effect_fields:
        value = contract.get(field)
        if value is not None:
            checked[field] = value
            if value != "none":
                failures.append(f"{field}_not_none")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "checked_fields": checked,
        "missing_fields_treated_as_no_authorization": True,
    }


def _feature_rules() -> dict[str, object]:
    return {
        "feature_source": "candidate_row.trainable_public_input_observation_time_public_fields",
        "candidate_action_role": "categorical_candidate_being_scored",
        "candidate_action_index_used": False,
        "candidate_action_index_treatment_if_used": "not_used_by_v129_surface",
        "allowed_feature_families": [
            "candidate_action_categorical",
            "public_action_mask_derived_candidate_legality",
            "candidate_action_name_semantics",
            "public_target_self_state_before_decision",
            "public_transition_context_before_decision",
            "optional_candidate_target_resource_neighbor_public_records_when_present",
        ],
        "labels_and_outcomes_evaluation_only": [
            "repaired_action",
            "repaired_archive_row_id",
            "oracle_rank",
            "material_gain_label",
            "first_action_outcome",
            "recovery_vitals_deltas",
            "objective_fields",
            "replay_verification",
            "resolution_action_mask",
            "resolution_legal",
        ],
        "forbidden_feature_families": sorted(FORBIDDEN_FEATURE_PATH_PARTS),
        "outcome_labels_used_to_fill_missing_candidate_specific_public_data": False,
    }


def _candidate_public_feature_dataset(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_groups: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    assignments = _stratified_three_way_assignments(manifest_rows)
    dataset: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    branch_ordinals = {
        _branch_id(manifest): index
        for index, manifest in enumerate(sorted(manifest_rows, key=lambda row: str(row.get("branch_id"))))
        if _branch_id(manifest) is not None
    }
    for manifest in sorted(manifest_rows, key=lambda row: str(row.get("branch_id"))):
        branch_id = _branch_id(manifest)
        if branch_id is None:
            continue
        repaired_action = str(manifest.get("repaired_action"))
        repaired_archive_id = manifest.get("repaired_archive_row_id")
        split = assignments.get(branch_id)
        seed = _mapping(manifest.get("non_trainable_audit_metadata")).get("seed")
        fixture_group = _fixture_group(manifest)
        candidates = list(candidate_groups.get(branch_id, ()))
        for candidate_index, candidate in enumerate(candidates):
            extraction = _candidate_public_features(candidate)
            features = extraction["features"]
            dataset_row = {
                "branch_id": branch_id,
                "split": split,
                "seed": seed,
                "fixture_group": fixture_group,
                "repaired_action": repaired_action,
                "repaired_archive_row_id": repaired_archive_id,
                "archive_row_id": candidate.get("archive_row_id"),
                "candidate_action": features.get("candidate_action"),
                "is_positive": candidate.get("archive_row_id") == repaired_archive_id,
                "material_gain_label": candidate.get("material_gain_label"),
                "candidate_public_features": features,
                "extractor_source_paths": extraction["source_paths"],
                "derived_feature_paths": extraction["derived_feature_paths"],
                "optional_candidate_public_paths": extraction[
                    "optional_candidate_public_paths"
                ],
                "public_action_mask": dict(
                    _mapping(_mapping(candidate.get("trainable_public_input")).get("action_mask"))
                ),
                "action_key": (features.get("candidate_action"),),
                "candidate_public_feature_key": _candidate_public_feature_key(features),
            }
            dataset.append(dataset_row)
            feature_rows.append(
                {
                    "schema_version": (
                        MIND_V3_FIRST_RECOVERY_CANDIDATE_PUBLIC_FEATURE_ROW_SCHEMA_VERSION
                    ),
                    "candidate_public_features": features,
                    "extractor_source_paths": extraction["source_paths"],
                    "derived_feature_paths": extraction["derived_feature_paths"],
                    "non_feature_metadata": {
                        "candidate_set_ordinal": branch_ordinals.get(branch_id),
                        "candidate_ordinal_within_set": candidate_index,
                        "split_role": "evaluation_only_not_feature",
                        "split": split,
                    },
                    "diagnostics_only": True,
                    "training_authorized": False,
                    "runtime_policy_authorized": False,
                }
            )
    return tuple(dataset), tuple(feature_rows)


def _candidate_public_features(candidate: Mapping[str, object]) -> dict[str, object]:
    trainable = _mapping(candidate.get("trainable_public_input"))
    source_paths: set[str] = set()
    derived_paths: set[str] = set()
    optional_paths: set[str] = set()
    features: dict[str, object] = {}

    action = str(trainable.get("candidate_action") or candidate.get("candidate_action") or "")
    source_paths.add("trainable_public_input.candidate_action")
    features["candidate_action"] = action

    action_mask = _mapping(trainable.get("action_mask"))
    source_paths.add("trainable_public_input.action_mask")
    source_paths.add(f"trainable_public_input.action_mask.{action}")
    features["candidate_action_observation_legal"] = action_mask.get(action) is True
    features["public_legal_action_count_bucket"] = _count_bucket(
        sum(1 for value in action_mask.values() if value is True)
    )

    state = _mapping(trainable.get("target_public_state_before"))
    for key in ("energy_ratio", "hydration_ratio", "health_ratio", "age", "alive"):
        source_paths.add(f"trainable_public_input.target_public_state_before.{key}")
    features["target_energy_bucket"] = _ratio_bucket(state.get("energy_ratio"))
    features["target_hydration_bucket"] = _ratio_bucket(state.get("hydration_ratio"))
    features["target_health_bucket"] = _ratio_bucket(state.get("health_ratio"))
    features["target_age_bucket"] = _age_bucket(state.get("age"))
    features["target_alive_public"] = state.get("alive") is True

    context = _mapping(trainable.get("public_transition_context"))
    for key in (
        "records_after_animal_resource_gain",
        "ticks_after_animal_resource_gain",
    ):
        source_paths.add(f"trainable_public_input.public_transition_context.{key}")
    source_paths.add("trainable_public_input.post_carrion_first_recovery")
    features["public_records_after_resource_gain_bucket"] = _count_bucket(
        context.get("records_after_animal_resource_gain")
    )
    features["public_ticks_after_resource_gain_bucket"] = _count_bucket(
        context.get("ticks_after_animal_resource_gain")
    )
    features["post_carrion_first_recovery"] = bool(
        trainable.get("post_carrion_first_recovery")
    )

    semantics = _action_semantics(action)
    for key, value in semantics.items():
        feature_path = f"candidate_action_{key}"
        features[feature_path] = value
        derived_paths.add(f"derived_public_action_semantics.{key}")
    features["candidate_public_need_match"] = _candidate_public_need_match(
        action,
        features,
    )
    derived_paths.add("derived_public_action_semantics.public_need_match")

    for container in OPTIONAL_CANDIDATE_PUBLIC_CONTAINERS:
        if container not in trainable:
            continue
        source_paths.add(f"trainable_public_input.{container}")
        optional_paths.add(container)
        _flatten_optional_public_features(
            trainable.get(container),
            feature_prefix=container,
            source_prefix=f"trainable_public_input.{container}",
            features=features,
            source_paths=source_paths,
            optional_paths=optional_paths,
        )
    for container in OPTIONAL_BY_ACTION_PUBLIC_CONTAINERS:
        by_action = _mapping(trainable.get(container))
        if not by_action or action not in by_action:
            continue
        source_paths.add(f"trainable_public_input.{container}.{action}")
        optional_paths.add(container)
        _flatten_optional_public_features(
            by_action.get(action),
            feature_prefix=container,
            source_prefix=f"trainable_public_input.{container}.{action}",
            features=features,
            source_paths=source_paths,
            optional_paths=optional_paths,
        )

    return {
        "features": features,
        "source_paths": tuple(sorted(source_paths)),
        "derived_feature_paths": tuple(sorted(derived_paths)),
        "optional_candidate_public_paths": tuple(sorted(optional_paths)),
    }


def _flatten_optional_public_features(
    value: object,
    *,
    feature_prefix: str,
    source_prefix: str,
    features: dict[str, object],
    source_paths: set[str],
    optional_paths: set[str],
) -> None:
    if isinstance(value, Mapping):
        for key, item in sorted(value.items(), key=lambda item: str(item[0])):
            child_key = str(key)
            _flatten_optional_public_features(
                item,
                feature_prefix=f"{feature_prefix}.{child_key}",
                source_prefix=f"{source_prefix}.{child_key}",
                features=features,
                source_paths=source_paths,
                optional_paths=optional_paths,
            )
        return
    if isinstance(value, (str, int, float, bool)) or value is None:
        source_paths.add(source_prefix)
        optional_paths.add(feature_prefix)
        features[feature_prefix] = value


def _feature_allowlist_audit(dataset: Sequence[Mapping[str, object]]) -> dict[str, object]:
    feature_paths = _actual_feature_paths(dataset)
    source_paths = _actual_source_paths(dataset)
    derived_paths = _actual_derived_paths(dataset)
    denied_feature_paths = [
        path for path in feature_paths if not _allowed_feature_path(path)
    ]
    denied_source_paths = [
        path for path in source_paths if not _allowed_extractor_source_path(path)
    ]
    denied_derived_paths = [
        path for path in derived_paths if not path.startswith("derived_public_action_semantics.")
    ]
    return {
        "passed": not denied_feature_paths
        and not denied_source_paths
        and not denied_derived_paths,
        "actual_feature_paths": feature_paths,
        "actual_extractor_source_paths": source_paths,
        "actual_derived_feature_paths": derived_paths,
        "denied_feature_paths": denied_feature_paths,
        "denied_source_paths": denied_source_paths,
        "denied_derived_paths": denied_derived_paths,
        "candidate_action_index_used": "candidate_action_index" in feature_paths
        or any(path.endswith(".candidate_action_index") for path in source_paths),
        "candidate_action_index_treatment_if_used": "not_used",
        "audit_source": "actual_extractor_paths_from_candidate_public_feature_rows",
    }


def _forbidden_feature_scan(dataset: Sequence[Mapping[str, object]]) -> dict[str, object]:
    feature_paths = _actual_feature_paths(dataset)
    source_paths = _actual_source_paths(dataset)
    derived_paths = _actual_derived_paths(dataset)
    all_paths = sorted(set(feature_paths) | set(source_paths) | set(derived_paths))
    forbidden = [
        {
            "path": path,
            "matched_forbidden_parts": _forbidden_matches(path),
        }
        for path in all_paths
        if _forbidden_matches(path)
    ]
    return {
        "passed": not forbidden,
        "forbidden_feature_path_count": len(forbidden),
        "forbidden_feature_paths": forbidden[:MAX_EXAMPLES],
        "scanned_feature_path_count": len(all_paths),
        "forbidden_feature_families": sorted(FORBIDDEN_FEATURE_PATH_PARTS),
        "labels_outcomes_and_provenance_used_as_features": bool(forbidden),
    }


def _candidate_signal_inventory(
    dataset: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    optional_paths = sorted(
        {
            str(path)
            for row in dataset
            for path in _list_like(row.get("optional_candidate_public_paths"))
        }
    )
    trainable_top_level_keys = sorted(
        {
            str(key)
            for row in candidate_rows
            for key in _mapping(row.get("trainable_public_input")).keys()
        }
    )
    optional_top_level_present = sorted(
        key
        for key in trainable_top_level_keys
        if key in OPTIONAL_CANDIDATE_PUBLIC_CONTAINERS
        or key in OPTIONAL_BY_ACTION_PUBLIC_CONTAINERS
    )
    variance = _candidate_public_feature_variance(dataset)
    target_resource_neighbor = _mapping(
        variance.get("target_resource_neighborhood_variance")
    )
    target_resource_neighbor_varying = _mapping(
        target_resource_neighbor.get("varying_paths")
    )
    target_resource_neighbor_count = _int(
        target_resource_neighbor.get("varying_path_count")
    )
    return {
        "candidate_specific_action_semantics_present": True,
        "candidate_specific_action_mask_legality_present": True,
        "candidate_specific_target_resource_neighborhood_fields_present": bool(
            optional_paths
        ),
        "candidate_specific_target_resource_neighborhood_varying_path_count": len(
            target_resource_neighbor_varying
        ),
        "candidate_specific_target_resource_neighborhood_fields_vary": (
            target_resource_neighbor_count > 0
        ),
        "candidate_specific_target_resource_neighborhood_varying_paths": (
            target_resource_neighbor_varying
        ),
        "observed_optional_candidate_public_paths": optional_paths,
        "observed_candidate_trainable_top_level_keys": trainable_top_level_keys,
        "observed_optional_candidate_public_top_level_keys": optional_top_level_present,
        "current_public_records_gap": (
            "Current v115/v123 trainable_public_input rows expose candidate_action, "
            "candidate_action_index, branch-level action_mask, branch-level target "
            "public state, and branch-level public transition context. They do not "
            "expose per-candidate target/resource/neighborhood public records."
        )
        if not optional_paths
        else (
            "Optional per-candidate public records were present and extracted. "
            "Readiness still requires those fields to vary within candidate sets."
        ),
        "outcome_labels_used_to_fill_gap": False,
    }


def _within_branch_feature_variance(
    *,
    v128_report: Mapping[str, object] | None,
    dataset: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    del manifest_rows
    candidate_variance = _candidate_public_feature_variance(dataset)
    return {
        "v128_baseline": _v128_baseline_variance(_mapping(v128_report or {})),
        "candidate_public_features": candidate_variance,
    }


def _v128_baseline_variance(v128_report: Mapping[str, object]) -> dict[str, object]:
    variance = _mapping(v128_report.get("feature_variance"))
    varying = _mapping(variance.get("varying_paths"))
    varying_paths = sorted(varying)
    expected = ["candidate_action", "candidate_action_index"]
    unexpected = sorted(path for path in varying_paths if path not in expected)
    missing = sorted(path for path in expected if path not in varying)
    return {
        "source": "v128.feature_variance",
        "branch_count": variance.get("branch_count"),
        "varying_paths": varying,
        "varying_path_count": len(varying_paths),
        "expected_action_only_varying_paths": expected,
        "unexpected_varying_paths": unexpected,
        "missing_expected_varying_paths": missing,
        "only_candidate_action_and_index_vary": not unexpected and not missing,
        "candidate_action_varies_by_branch_count": variance.get(
            "candidate_action_varies_by_branch_count"
        ),
        "candidate_action_index_varies_by_branch_count": variance.get(
            "candidate_action_index_varies_by_branch_count"
        ),
    }


def _candidate_public_feature_variance(
    dataset: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in dataset:
        grouped[str(row.get("branch_id"))].append(row)
    path_branch_counts: Counter[str] = Counter()
    path_examples: defaultdict[str, list[str]] = defaultdict(list)
    for branch_id, rows in sorted(grouped.items()):
        path_values: defaultdict[str, set[str]] = defaultdict(set)
        for row in rows:
            for path, value in _mapping(row.get("candidate_public_features")).items():
                path_values[path].add(json.dumps(value, sort_keys=True))
        for path, values in path_values.items():
            if len(values) <= 1:
                continue
            path_branch_counts[path] += 1
            if len(path_examples[path]) < 4:
                path_examples[path].append(branch_id)
    varying = {
        path: {
            "branch_count": count,
            "example_branches": path_examples[path],
            "field_family": _feature_family(path),
        }
        for path, count in sorted(path_branch_counts.items())
    }
    action_identity = _variance_family(varying, "candidate_action")
    action_semantics = _variance_family(varying, "candidate_action_semantics")
    action_mask_legality = _variance_family(varying, "action_mask_derived_legality")
    target_resource_neighbor = _variance_family(
        varying,
        "candidate_target_resource_neighborhood",
    )
    return {
        "branch_count": len(grouped),
        "varying_paths": varying,
        "varying_path_count": len(varying),
        "action_identity_variance": action_identity,
        "action_semantics_variance": action_semantics,
        "action_mask_legality_variance": action_mask_legality,
        "target_resource_neighborhood_variance": target_resource_neighbor,
        "target_resource_neighborhood_varying_paths": target_resource_neighbor[
            "varying_paths"
        ],
        "target_resource_neighborhood_varying_path_count": (
            target_resource_neighbor["varying_path_count"]
        ),
        "deprecated_non_action_candidate_specific_variance": {
            "deprecated": True,
            "reason": (
                "Action-derived semantics are not true target/resource/"
                "neighborhood candidate context and must not be combined with "
                "per-candidate public context variance."
            ),
            "replacement_fields": [
                "action_identity_variance",
                "action_semantics_variance",
                "action_mask_legality_variance",
                "target_resource_neighborhood_variance",
            ],
        },
        "candidate_action_varies_by_branch_count": path_branch_counts.get(
            "candidate_action",
            0,
        ),
        "candidate_action_index_used": False,
        "candidate_action_index_varies_by_branch_count": 0,
    }


def _variance_family(
    varying: Mapping[str, object],
    family: str,
) -> dict[str, object]:
    paths = {
        path: payload
        for path, payload in sorted(varying.items())
        if _mapping(payload).get("field_family") == family
    }
    return {
        "field_family": family,
        "varying_paths": paths,
        "varying_path_count": len(paths),
    }


def _feature_surface_summary(
    dataset: Sequence[Mapping[str, object]],
    feature_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    paths = _actual_feature_paths(dataset)
    examples = [
        _mapping(row.get("candidate_public_features"))
        for row in feature_rows[:3]
    ]
    return {
        "candidate_feature_row_count": len(feature_rows),
        "feature_rows_digest": stable_payload_digest(list(feature_rows)),
        "feature_path_count": len(paths),
        "feature_paths": paths,
        "example_candidate_public_features": examples,
        "feature_rows_schema_version": (
            MIND_V3_FIRST_RECOVERY_CANDIDATE_PUBLIC_FEATURE_ROW_SCHEMA_VERSION
        ),
        "feature_rows_default_output_path": str(DEFAULT_FEATURE_ROWS_OUTPUT_PATH),
        "feature_rows_runtime_loadable": False,
    }


def _run_probes(dataset: Sequence[Mapping[str, object]]) -> dict[str, object]:
    train = [row for row in dataset if row.get("split") == "train"]
    action_prior_stats = _positive_rate_table(train, lambda row: row["action_key"])
    public_feature_stats = _positive_rate_table(
        train,
        lambda row: row["candidate_public_feature_key"],
    )
    probe_rows = {
        ACTION_ORDER_BASELINE: _predict_by_branch(dataset, _score_action_order),
        ACTION_ONLY_BASELINE: _predict_by_branch(
            dataset,
            lambda row: _score_lookup(row, action_prior_stats, row["action_key"]),
        ),
        PRIMARY_PROBE: _predict_by_branch(
            dataset,
            lambda row: _score_lookup(
                row,
                public_feature_stats,
                row["candidate_public_feature_key"],
                fallback=action_prior_stats.get(row["action_key"], 0.0),
            ),
        ),
    }
    return {name: _probe_report(name, rows) for name, rows in probe_rows.items()}


def _empty_probe_reports() -> dict[str, object]:
    return {
        name: _probe_report(name, ())
        for name in (ACTION_ORDER_BASELINE, ACTION_ONLY_BASELINE, PRIMARY_PROBE)
    }


def _probe_definitions() -> dict[str, object]:
    return {
        ACTION_ORDER_BASELINE: "Select public-observation-legal candidates by candidate_action name order.",
        ACTION_ONLY_BASELINE: (
            "Report-only train-split positive-rate table keyed only by candidate_action."
        ),
        PRIMARY_PROBE: (
            "Report-only train-split positive-rate table keyed by the extracted "
            "candidate_public_features surface, with action-prior fallback."
        ),
    }


def _probe_report(name: str, branch_predictions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    metrics = _metrics(branch_predictions)
    return {
        "name": name,
        "prediction_count": len(branch_predictions),
        "overall_metrics": metrics,
        "per_split_metrics": {
            split: _metrics([row for row in branch_predictions if row.get("split") == split])
            for split in ("train", "validation", "test")
        },
        "seed29_evaluation": _seed29_evaluation(branch_predictions),
        "fixture_open_evaluation": _fixture_open_evaluation(branch_predictions),
        "action_distribution": _action_distribution(branch_predictions),
        "unsupported_action_audit": _unsupported_action_audit(branch_predictions),
        "material_gain_recall": _material_gain_recall(branch_predictions),
    }


def _predict_by_branch(dataset: Sequence[Mapping[str, object]], score_fn) -> tuple[dict[str, object], ...]:
    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in dataset:
        grouped[str(row.get("branch_id"))].append(row)
    predictions: list[dict[str, object]] = []
    for branch_id, rows in sorted(grouped.items()):
        legal = [
            row
            for row in rows
            if _mapping(row.get("candidate_public_features")).get(
                "candidate_action_observation_legal"
            )
            is True
        ]
        selected = sorted(
            legal or rows,
            key=lambda row: (
                -score_fn(row),
                str(row.get("candidate_action")),
                stable_payload_digest(_mapping(row.get("candidate_public_features"))),
            ),
        )[0]
        positive = next((row for row in rows if row.get("is_positive") is True), rows[0])
        selected_features = _mapping(selected.get("candidate_public_features"))
        predictions.append(
            {
                "branch_id": branch_id,
                "split": positive.get("split"),
                "seed": positive.get("seed"),
                "fixture_group": positive.get("fixture_group"),
                "repaired_action": positive.get("repaired_action"),
                "predicted_action": selected.get("candidate_action"),
                "prediction_correct": selected.get("is_positive") is True,
                "unsupported_action": selected_features.get(
                    "candidate_action_observation_legal"
                )
                is not True,
                "repaired_label_material_gain_positive": (
                    positive.get("material_gain_label") is True
                ),
                "repaired_label_material_gain_exact_match_recalled": (
                    selected.get("is_positive") is True
                    and positive.get("material_gain_label") is True
                ),
                "candidate_set_has_material_gain_candidate": any(
                    row.get("material_gain_label") is True for row in rows
                ),
                "selected_material_gain_candidate": (
                    selected.get("material_gain_label") is True
                ),
            }
        )
    return tuple(predictions)


def _score_action_order(row: Mapping[str, object]) -> float:
    action_mask = _mapping(row.get("public_action_mask"))
    actions = sorted(str(action) for action, legal in action_mask.items() if legal is True)
    try:
        return 1.0 - (actions.index(str(row.get("candidate_action"))) / max(1, len(actions)))
    except ValueError:
        return 0.0


def _probe_comparisons(probes: Mapping[str, object]) -> dict[str, object]:
    primary = _mapping(_mapping(probes.get(PRIMARY_PROBE)).get("overall_metrics"))
    action_only = _mapping(_mapping(probes.get(ACTION_ONLY_BASELINE)).get("overall_metrics"))
    action_order = _mapping(_mapping(probes.get(ACTION_ORDER_BASELINE)).get("overall_metrics"))
    heldout_primary = _heldout_accuracy(probes, PRIMARY_PROBE)
    heldout_action_only = _heldout_accuracy(probes, ACTION_ONLY_BASELINE)
    heldout_action_order = _heldout_accuracy(probes, ACTION_ORDER_BASELINE)
    return {
        "primary_probe": PRIMARY_PROBE,
        "action_only_baseline": ACTION_ONLY_BASELINE,
        "action_order_baseline": ACTION_ORDER_BASELINE,
        "overall_accuracy_delta_vs_action_only": _number(primary.get("accuracy"), 0.0)
        - _number(action_only.get("accuracy"), 0.0),
        "overall_accuracy_delta_vs_action_order": _number(primary.get("accuracy"), 0.0)
        - _number(action_order.get("accuracy"), 0.0),
        "heldout_accuracy": heldout_primary,
        "heldout_action_only_accuracy": heldout_action_only,
        "heldout_action_order_accuracy": heldout_action_order,
        "heldout_delta_vs_action_only": heldout_primary - heldout_action_only,
        "heldout_delta_vs_action_order": heldout_primary - heldout_action_order,
        "heldout_signal_beats_action_only_baseline": heldout_primary > heldout_action_only,
        "heldout_signal_beats_action_order_baseline": heldout_primary > heldout_action_order,
        "per_probe_vs_action_only": {
            name: _probe_delta_vs_action_only(probes, name, action_only)
            for name in sorted(probes)
        },
    }


def _probe_delta_vs_action_only(
    probes: Mapping[str, object],
    name: str,
    action_only_metrics: Mapping[str, object],
) -> dict[str, object]:
    metrics = _mapping(_mapping(probes.get(name)).get("overall_metrics"))
    heldout = _heldout_accuracy(probes, name)
    action_only_heldout = _heldout_accuracy(probes, ACTION_ONLY_BASELINE)
    return {
        "accuracy": _number(metrics.get("accuracy"), 0.0),
        "accuracy_delta_vs_action_only": _number(metrics.get("accuracy"), 0.0)
        - _number(action_only_metrics.get("accuracy"), 0.0),
        "heldout_accuracy": heldout,
        "heldout_delta_vs_action_only": heldout - action_only_heldout,
    }


def _heldout_evaluation(probes: Mapping[str, object], name: str) -> dict[str, object]:
    return {
        "probe": name,
        "heldout_accuracy": _heldout_accuracy(probes, name),
        "action_only_baseline_accuracy": _heldout_accuracy(probes, ACTION_ONLY_BASELINE),
        "action_order_baseline_accuracy": _heldout_accuracy(probes, ACTION_ORDER_BASELINE),
    }


def _metric_gate(
    *,
    source_integrity: Mapping[str, object],
    feature_allowlist: Mapping[str, object],
    forbidden_scan: Mapping[str, object],
    candidate_signal_inventory: Mapping[str, object],
    within_branch_variance: Mapping[str, object],
    probes: Mapping[str, object],
    comparisons: Mapping[str, object],
    primary: Mapping[str, object],
    leakage: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    if source_integrity.get("passed") is not True:
        failures.append("source_integrity_failed")
    if feature_allowlist.get("passed") is not True:
        failures.append("feature_allowlist_failed")
    if forbidden_scan.get("passed") is not True:
        failures.append("forbidden_feature_scan_failed")
    if leakage.get("passed") is not True:
        failures.append("leakage_failed")
    candidate_variance = _mapping(
        _mapping(within_branch_variance.get("candidate_public_features"))
    )
    target_resource_variance = _mapping(
        candidate_variance.get("target_resource_neighborhood_variance")
    )
    target_resource_varying_count = _int(
        target_resource_variance.get("varying_path_count")
    )
    if (
        candidate_signal_inventory.get(
            "candidate_specific_target_resource_neighborhood_fields_present"
        )
        is not True
    ):
        failures.append("candidate_specific_target_resource_neighborhood_signal_missing")
    if target_resource_varying_count <= 0:
        failures.append("target_resource_neighborhood_features_do_not_vary")
    if comparisons.get("heldout_signal_beats_action_only_baseline") is not True:
        failures.append("heldout_signal_not_above_action_only_baseline")
    if comparisons.get("heldout_signal_beats_action_order_baseline") is not True:
        failures.append("heldout_signal_not_above_action_order_baseline")
    if _mapping(primary.get("seed29_evaluation")).get("passed") is not True:
        failures.append("seed29_failed")
    if _mapping(primary.get("fixture_open_evaluation")).get("passed") is not True:
        failures.append("fixture_open_failed")
    action = _mapping(primary.get("action_distribution"))
    if action.get("passed") is not True:
        failures.append("action_collapse_detected")
    material = _mapping(primary.get("material_gain_recall"))
    if material.get("passed") is not True:
        failures.append("material_exact_match_recall_below_floor")
    unsupported = _mapping(primary.get("unsupported_action_audit"))
    if unsupported.get("passed") is not True:
        failures.append("unsupported_selection_detected")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "primary_probe": PRIMARY_PROBE,
        "source_integrity_passed": source_integrity.get("passed"),
        "feature_allowlist_passed": feature_allowlist.get("passed"),
        "forbidden_feature_scan_passed": forbidden_scan.get("passed"),
        "heldout_signal_beats_action_only_baseline": comparisons.get(
            "heldout_signal_beats_action_only_baseline"
        ),
        "heldout_signal_beats_action_order_baseline": comparisons.get(
            "heldout_signal_beats_action_order_baseline"
        ),
        "seed29_passed": _mapping(primary.get("seed29_evaluation")).get("passed"),
        "fixture_open_passed": _mapping(primary.get("fixture_open_evaluation")).get("passed"),
        "dominant_predicted_action_share_passed": action.get("passed"),
        "unsupported_action_rate_passed": unsupported.get("passed"),
        "leakage_passed": leakage.get("passed"),
        "material_exact_match_recall_passed": material.get("passed"),
        "candidate_specific_target_resource_neighborhood_fields_present": (
            candidate_signal_inventory.get(
                "candidate_specific_target_resource_neighborhood_fields_present"
            )
        ),
        "target_resource_neighborhood_varying_path_count": (
            target_resource_varying_count
        ),
        "target_resource_neighborhood_variance_required": True,
    }


def _classification(
    source_integrity: Mapping[str, object],
    feature_allowlist: Mapping[str, object],
    forbidden_scan: Mapping[str, object],
    metric_gate: Mapping[str, object],
) -> dict[str, object]:
    if (
        source_integrity.get("passed") is not True
        or feature_allowlist.get("passed") is not True
        or forbidden_scan.get("passed") is not True
    ):
        primary = "candidate_public_feature_surface_source_integrity_failed"
    elif metric_gate.get("passed") is True:
        primary = "candidate_public_feature_surface_ready_for_ranker_probe"
    else:
        primary = "candidate_public_feature_surface_missing_candidate_specific_signal"
    return {
        "primary": primary,
        "labels": [primary],
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(classification: Mapping[str, object]) -> dict[str, object]:
    ready = (
        classification.get("primary")
        == "candidate_public_feature_surface_ready_for_ranker_probe"
    )
    return {
        "next_step": (
            "review_candidate_public_feature_surface_before_any_ranker_probe"
            if ready
            else "add_observation_time_candidate_target_resource_or_neighbor_public_signal_before_ranker_probe"
        ),
        "summary": (
            "v129 is diagnostics-only. It extracts and audits a report-only "
            "candidate-conditioned public feature surface; it does not train, "
            "authorize readiness, authorize downstream shadow scoring, or change runtime."
        ),
        "candidate_public_feature_surface_ready_for_ranker_probe": ready,
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "training_executed": False,
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "model_artifact_created": False,
        "gate_change_recommended": False,
        "viewer_change_recommended": False,
        "replay_golden_change_recommended": False,
        "foundation_change_recommended": False,
        "claim_causality": False,
    }


def _candidate_public_feature_key(features: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (str(key), json.dumps(value, sort_keys=True))
        for key, value in sorted(features.items(), key=lambda item: str(item[0]))
    )


def _action_semantics(action: str) -> dict[str, object]:
    direction = "none"
    for candidate in ("north", "east", "south", "west"):
        if action.endswith(f"_{candidate}"):
            direction = candidate
            break
    family = "other"
    if action.startswith("move_"):
        family = "movement"
    elif action.startswith("attack_"):
        family = "attack"
    elif action in {"eat", "drink"}:
        family = "resource"
    elif action == "mate":
        family = "reproduction"
    elif action.startswith("signal_"):
        family = "signal"
    elif action == "stay":
        family = "idle"
    return {
        "family": family,
        "direction": direction,
        "is_movement": family == "movement",
        "is_attack": family == "attack",
        "is_resource_use": family == "resource",
        "is_signal": family == "signal",
        "is_stay": action == "stay",
        "requires_directional_neighbor": family in {"movement", "attack"},
    }


def _candidate_public_need_match(
    action: str,
    features: Mapping[str, object],
) -> str:
    energy = features.get("target_energy_bucket")
    hydration = features.get("target_hydration_bucket")
    health = features.get("target_health_bucket")
    if action == "eat" and energy in {"very_low", "low"}:
        return "energy_need"
    if action == "drink" and hydration in {"very_low", "low"}:
        return "hydration_need"
    if action == "stay" and health in {"very_low", "low"}:
        return "health_conservation_need"
    if action.startswith("move_") and hydration == "high" and energy != "very_low":
        return "mobility_plausible"
    if action.startswith("attack_") and energy != "very_low" and health != "very_low":
        return "attack_plausible"
    return "no_public_need_match"


def _actual_feature_paths(dataset: Sequence[Mapping[str, object]]) -> list[str]:
    return sorted(
        {
            str(path)
            for row in dataset
            for path in _mapping(row.get("candidate_public_features")).keys()
        }
    )


def _actual_source_paths(dataset: Sequence[Mapping[str, object]]) -> list[str]:
    return sorted(
        {
            str(path)
            for row in dataset
            for path in _list_like(row.get("extractor_source_paths"))
        }
    )


def _actual_derived_paths(dataset: Sequence[Mapping[str, object]]) -> list[str]:
    return sorted(
        {
            str(path)
            for row in dataset
            for path in _list_like(row.get("derived_feature_paths"))
        }
    )


def _allowed_feature_path(path: str) -> bool:
    if path == "candidate_action":
        return True
    if path.startswith("candidate_action_"):
        return True
    if path in {
        "candidate_public_need_match",
        "post_carrion_first_recovery",
        "public_legal_action_count_bucket",
        "public_records_after_resource_gain_bucket",
        "public_ticks_after_resource_gain_bucket",
        "target_age_bucket",
        "target_alive_public",
        "target_energy_bucket",
        "target_health_bucket",
        "target_hydration_bucket",
    }:
        return True
    return path.startswith(OPTIONAL_CANDIDATE_PUBLIC_CONTAINERS) or path.startswith(
        OPTIONAL_BY_ACTION_PUBLIC_CONTAINERS
    )


def _allowed_extractor_source_path(path: str) -> bool:
    prefixes = (
        "trainable_public_input.candidate_action",
        "trainable_public_input.action_mask",
        "trainable_public_input.target_public_state_before",
        "trainable_public_input.public_transition_context",
        "trainable_public_input.post_carrion_first_recovery",
    ) + tuple(
        f"trainable_public_input.{container}"
        for container in OPTIONAL_CANDIDATE_PUBLIC_CONTAINERS
    ) + tuple(
        f"trainable_public_input.{container}"
        for container in OPTIONAL_BY_ACTION_PUBLIC_CONTAINERS
    )
    return path.startswith(prefixes)


def _forbidden_matches(path: str) -> list[str]:
    lowered = path.lower()
    tokens = set(_path_tokens(lowered))
    matches = sorted(FORBIDDEN_FEATURE_PATH_PARTS & tokens)
    for substring in FORBIDDEN_FEATURE_PATH_SUBSTRINGS:
        if substring in lowered and substring not in matches:
            matches.append(substring)
    return sorted(matches)


def _path_tokens(path: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for character in path:
        if character.isalnum():
            current.append(character)
            continue
        if current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    tokens.extend(path.replace("-", "_").split("_"))
    return [token for token in tokens if token]


def _feature_family(path: str) -> str:
    if path == "candidate_action":
        return "candidate_action"
    if path == "candidate_action_observation_legal":
        return "action_mask_derived_legality"
    if path.startswith("candidate_action_") or path == "candidate_public_need_match":
        return "candidate_action_semantics"
    if path.startswith(OPTIONAL_CANDIDATE_PUBLIC_CONTAINERS) or path.startswith(
        OPTIONAL_BY_ACTION_PUBLIC_CONTAINERS
    ):
        return "candidate_target_resource_neighborhood"
    return "branch_public_context"


def _branch_id(row: Mapping[str, object]) -> str | None:
    value = row.get("branch_id")
    return value if isinstance(value, str) and value else None


def _row_action(row: Mapping[str, object]) -> str:
    return str(row.get("repaired_action") or row.get("candidate_action") or "")


def _fixture_group(manifest: Mapping[str, object]) -> str:
    meta = _mapping(manifest.get("non_trainable_audit_metadata"))
    source = str(meta.get("source") or meta.get("source_kind") or meta.get("source_path") or "")
    if "active-coverage" in source or source == "open_mind_v3":
        return "open_mind_v3"
    if "fixture_carrion_only" in source or "carrion_only" in source:
        return "fixture_carrion_only"
    return source or "unknown_source"


def _ratio_bucket(value: object) -> str:
    number = _number(value, None)
    if number is None:
        return "missing"
    if number < 0.25:
        return "very_low"
    if number < 0.5:
        return "low"
    if number < 0.75:
        return "mid"
    return "high"


def _age_bucket(value: object) -> str:
    number = _number(value, None)
    if number is None:
        return "missing"
    if number < 20:
        return "young"
    if number < 50:
        return "adult"
    return "old"


def _count_bucket(value: object) -> str:
    number = _number(value, None)
    if number is None:
        return "missing"
    if number <= 0:
        return "zero"
    if number <= 4:
        return "short"
    if number <= 12:
        return "mid"
    return "long"
