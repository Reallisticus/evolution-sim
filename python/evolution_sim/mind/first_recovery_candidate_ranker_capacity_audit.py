from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V124_MANIFEST_PATH,
)
from evolution_sim.mind.first_recovery_active_coverage_archive import (
    DEFAULT_ARCHIVE_ROWS_OUTPUT_PATH as DEFAULT_V123_ARCHIVE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V123_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_branch_archive import (
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    trainable_public_input_leakage,
)
from evolution_sim.mind.first_recovery_candidate_set_shadow_execution import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V127_REPORT_PATH,
    DEFAULT_PREDICTIONS_OUTPUT_PATH as DEFAULT_V127_PREDICTIONS_PATH,
    DEFAULT_V115_ARCHIVE_ROWS_PATH,
    DEFAULT_V115_REPORT_PATH,
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

MIND_V3_FIRST_RECOVERY_CANDIDATE_RANKER_CAPACITY_AUDIT_SCHEMA_VERSION = (
    "mind_v3_first_recovery_candidate_ranker_capacity_audit_v1"
)
MIND_V3_FIRST_RECOVERY_CANDIDATE_RANKER_CAPACITY_AUDIT_POLICY = (
    "diagnostics_only_first_recovery_v128_candidate_ranker_capacity_audit_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v128-first-recovery-candidate-ranker-capacity-audit.json"
)

EXPECTED_V127_CLASSIFICATION = "candidate_set_shadow_execution_blocked_by_metrics"
EXPECTED_BRANCH_COUNT = 108
EXPECTED_CANDIDATE_ROW_COUNT = 542
EXPECTED_POSITIVE_ROW_COUNT = 108
EXPECTED_NEGATIVE_ROW_COUNT = 434

PRIMARY_PROBE = "public_input_interaction_probe"
ACTION_ONLY_BASELINE = "action_prior_baseline"
ACTION_ORDER_BASELINE = "action_order_baseline"

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "candidate_ranker_capacity_ready_for_review",
    "candidate_ranker_capacity_source_integrity_failed",
    "candidate_ranker_capacity_blocked_by_signal",
    "diagnostics_only_no_runtime_promotion",
    "readiness_rerun_blocked",
)

FORBIDDEN_FEATURE_FAMILIES: tuple[str, ...] = (
    "seed",
    "branch_id",
    "source",
    "source_kind",
    "source_path",
    "record_index",
    "logged_action",
    "archive_row_id",
    "replay_verification",
    "oracle_rank",
    "material_gain",
    "terminal_alive_delta",
    "birth_delta",
    "death_delta",
    "death_reduction_delta",
    "first_action_outcome",
    "digest",
    "path",
    "provenance",
)
MAX_EXAMPLES = 16


@dataclass(frozen=True, slots=True)
class FirstRecoveryCandidateRankerCapacityAuditBuild:
    report: dict[str, object]


def build_first_recovery_candidate_ranker_capacity_audit(
    *,
    v127_report: Mapping[str, object] | None = None,
    v127_report_path: str | Path | None = DEFAULT_V127_REPORT_PATH,
    v127_prediction_rows: Sequence[Mapping[str, object]] | None = None,
    v127_predictions_path: str | Path | None = DEFAULT_V127_PREDICTIONS_PATH,
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
) -> FirstRecoveryCandidateRankerCapacityAuditBuild:
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
        "v127_report": v127_evidence,
        "v127_predictions": v127_predictions_evidence,
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
    dataset = _candidate_dataset(
        manifest_rows=manifest_rows,
        candidate_groups=candidate_groups,
    )
    source_integrity = _source_integrity(
        source_reports=source_reports,
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
    feature_variance = _feature_variance(candidate_groups, manifest_rows)
    feature_rules = _feature_rules()
    input_allowlist_audit = _input_allowlist_audit()
    probes = (
        _run_probes(dataset)
        if source_integrity["passed"] and reconstructed_audit["candidate_sets_available"]
        else _empty_probe_reports()
    )
    comparisons = _probe_comparisons(probes)
    primary = _mapping(probes.get(PRIMARY_PROBE))
    metric_gate = _metric_gate(
        source_integrity=source_integrity,
        feature_variance=feature_variance,
        probes=probes,
        comparisons=comparisons,
        primary=primary,
    )
    classification = _classification(source_integrity, metric_gate)
    recommendation = _recommendation(classification)
    report = {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_CANDIDATE_RANKER_CAPACITY_AUDIT_SCHEMA_VERSION
        ),
        "audit_policy": MIND_V3_FIRST_RECOVERY_CANDIDATE_RANKER_CAPACITY_AUDIT_POLICY,
        "contract": _contract(),
        "source_reports": source_reports,
        "source_integrity": source_integrity,
        "feature_rules": feature_rules,
        "input_allowlist_audit": input_allowlist_audit,
        "feature_variance": feature_variance,
        "probe_definitions": _probe_definitions(),
        "probe_reports": probes,
        "probe_comparisons": comparisons,
        "per_split_metrics": primary.get("per_split_metrics", {}),
        "seed29_evaluation": primary.get("seed29_evaluation", {}),
        "fixture_open_evaluation": primary.get("fixture_open_evaluation", {}),
        "material_gain_recall": primary.get("material_gain_recall", {}),
        "action_distribution": primary.get("action_distribution", {}),
        "unsupported_action_audit": primary.get("unsupported_action_audit", {}),
        "leakage_audit": _leakage_audit(
            manifest_rows=manifest_rows,
            candidate_rows=candidate_rows,
        ),
        "metric_gate": metric_gate,
        "classification": classification,
        "recommendation": recommendation,
        "non_promoted": True,
    }
    return FirstRecoveryCandidateRankerCapacityAuditBuild(report=report)


def write_first_recovery_candidate_ranker_capacity_audit_report(
    build: FirstRecoveryCandidateRankerCapacityAuditBuild,
    *,
    output_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(build.report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _contract() -> dict[str, object]:
    return {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_CANDIDATE_RANKER_CAPACITY_AUDIT_SCHEMA_VERSION
        ),
        "diagnostics_only": True,
        "report_only_capacity_probe": True,
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
    v127_report: Mapping[str, object] | None,
    v127_predictions: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    v115_report: Mapping[str, object] | None,
    v115_rows: Sequence[Mapping[str, object]],
    v123_report: Mapping[str, object] | None,
    v123_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    reconstructed_audit: Mapping[str, object],
) -> dict[str, object]:
    del v115_report, v123_report
    failures: list[str] = []
    for name, evidence in source_reports.items():
        if evidence.get("loaded") is not True:
            failures.append(f"missing_{name}")
        elif evidence.get("schema_matches") is False:
            failures.append(f"{name}_schema_mismatch")
    report = _mapping(v127_report or {})
    v127_source = _mapping(report.get("source_integrity"))
    v127_audit = _mapping(report.get("candidate_set_audit"))
    v127_leakage = _mapping(report.get("leakage_audit"))
    v127_unsupported = _mapping(report.get("unsupported_action_audit"))
    v127_predictions_digest = stable_payload_digest(list(v127_predictions))
    prediction_summary = _mapping(report.get("prediction_summary"))
    manifest_leakage = _trainable_leakage(manifest_rows)
    candidate_leakage = trainable_public_input_leakage(candidate_rows)

    if _mapping(report.get("classification")).get("primary") != EXPECTED_V127_CLASSIFICATION:
        failures.append("v127_classification_unexpected")
    if v127_source.get("passed") is not True:
        failures.append("v127_source_integrity_not_passed")
    if v127_source.get("failures") != []:
        failures.append("v127_source_failures_not_empty_or_malformed")
    if _int(v127_audit.get("branch_count")) != EXPECTED_BRANCH_COUNT:
        failures.append("v127_branch_count_unexpected")
    if _int(v127_audit.get("total_candidate_rows")) != EXPECTED_CANDIDATE_ROW_COUNT:
        failures.append("v127_candidate_row_count_unexpected")
    if _int(v127_audit.get("positive_rows_count")) != EXPECTED_POSITIVE_ROW_COUNT:
        failures.append("v127_positive_row_count_unexpected")
    if _int(v127_audit.get("negative_rows_count")) != EXPECTED_NEGATIVE_ROW_COUNT:
        failures.append("v127_negative_row_count_unexpected")
    join = _mapping(v127_audit.get("repaired_archive_row_join"))
    if _int(join.get("exact_join_count")) != EXPECTED_BRANCH_COUNT or join.get("passed") is not True:
        failures.append("v127_exact_repaired_archive_row_join_failed")
    if _int(v127_leakage.get("trainable_leakage_count")) != 0:
        failures.append("v127_trainable_leakage_nonzero")
    if _int(v127_audit.get("unsupported_repaired_labels_count")) != 0:
        failures.append("v127_unsupported_repaired_labels_nonzero")
    if _int(v127_unsupported.get("unsupported_action_count")) != 0:
        failures.append("v127_unsupported_action_count_nonzero")
    if _int(prediction_summary.get("prediction_row_count")) != len(v127_predictions):
        failures.append("v127_prediction_row_count_mismatch")
    if len(v127_predictions) != EXPECTED_BRANCH_COUNT:
        failures.append("v127_prediction_row_count_unexpected")
    if (
        prediction_summary.get("prediction_rows_digest") is not None
        and prediction_summary.get("prediction_rows_digest") != v127_predictions_digest
    ):
        failures.append("v127_prediction_digest_mismatch")

    if len(manifest_rows) != EXPECTED_BRANCH_COUNT:
        failures.append("manifest_row_count_unexpected")
    if len(candidate_rows) != EXPECTED_CANDIDATE_ROW_COUNT:
        failures.append("candidate_row_count_unexpected")
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

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v127_source_integrity_passed": v127_source.get("passed"),
        "branch_count": _int(reconstructed_audit.get("branch_count")),
        "candidate_row_count": len(candidate_rows),
        "positive_row_count": _int(reconstructed_audit.get("positive_rows_count")),
        "negative_row_count": _int(reconstructed_audit.get("negative_rows_count")),
        "exact_repaired_archive_row_joins": _int(reconstructed_join.get("exact_join_count")),
        "trainable_leakage_count": _int(candidate_leakage.get("leak_count")),
        "unsupported_repaired_labels_count": _int(
            reconstructed_audit.get("unsupported_repaired_labels_count")
        ),
        "v115_candidate_row_count": len(v115_rows),
        "v123_candidate_row_count": len(v123_rows),
        "v127_prediction_row_count": len(v127_predictions),
        "expected_v127_prediction_row_count": EXPECTED_BRANCH_COUNT,
        "v127_prediction_rows_digest": v127_predictions_digest,
        "candidate_set_audit": dict(reconstructed_audit),
        "manifest_trainable_leakage": manifest_leakage,
        "candidate_trainable_leakage": candidate_leakage,
    }


def _candidate_dataset(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_groups: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[dict[str, object], ...]:
    assignments = _stratified_three_way_assignments(manifest_rows)
    rows: list[dict[str, object]] = []
    for manifest in sorted(manifest_rows, key=lambda row: str(row.get("branch_id"))):
        branch_id = _branch_id(manifest)
        if branch_id is None:
            continue
        repaired_action = str(manifest.get("repaired_action"))
        repaired_archive_id = manifest.get("repaired_archive_row_id")
        split = assignments.get(branch_id)
        for candidate in candidate_groups.get(branch_id, ()):
            trainable = _mapping(candidate.get("trainable_public_input"))
            rows.append(
                {
                    "branch_id": branch_id,
                    "split": split,
                    "seed": _mapping(manifest.get("non_trainable_audit_metadata")).get("seed"),
                    "fixture_group": _fixture_group(manifest),
                    "repaired_action": repaired_action,
                    "repaired_archive_row_id": repaired_archive_id,
                    "archive_row_id": candidate.get("archive_row_id"),
                    "candidate_action": candidate.get("candidate_action"),
                    "is_positive": candidate.get("archive_row_id") == repaired_archive_id,
                    "resolution_legal": candidate.get("resolution_legal"),
                    "observation_legal": candidate.get("observation_legal"),
                    "material_gain_label": candidate.get("material_gain_label"),
                    "trainable_public_input": trainable,
                    "action_key": _action_key(trainable),
                    "interaction_key": _interaction_key(trainable),
                }
            )
    return tuple(rows)


def _feature_rules() -> dict[str, object]:
    return {
        "feature_source": "candidate_row.trainable_public_input_only",
        "candidate_action_role": "candidate_being_scored",
        "candidate_action_index_used": False,
        "candidate_action_index_treatment_if_used": "categorical_only_not_ordinal",
        "labels_and_outcomes_evaluation_only": [
            "repaired_action",
            "repaired_archive_row_id",
            "oracle_rank",
            "material_gain_label",
            "objective_fields",
            "replay_results",
        ],
        "forbidden_feature_families": list(FORBIDDEN_FEATURE_FAMILIES),
    }


def _input_allowlist_audit() -> dict[str, object]:
    used_paths = [
        "candidate_action",
        "post_carrion_first_recovery",
        "public_transition_context.records_after_animal_resource_gain",
        "public_transition_context.ticks_after_animal_resource_gain",
        "target_public_state_before.age",
        "target_public_state_before.energy_ratio",
        "target_public_state_before.health_ratio",
        "target_public_state_before.hydration_ratio",
    ]
    return {
        "passed": True,
        "scorer_input_source": "candidate_row.trainable_public_input_only",
        "used_feature_paths": used_paths,
        "candidate_action_index_used": False,
        "candidate_action_index_treatment_if_used": "categorical_only_not_ordinal",
        "forbidden_feature_families": list(FORBIDDEN_FEATURE_FAMILIES),
        "forbidden_feature_paths_used": [],
        "forbidden_feature_path_count": 0,
        "split_assignment_role": "evaluation_grouping_only_not_scorer_input",
        "audit_metadata_role": "evaluation_grouping_only_not_scorer_input",
    }


def _feature_variance(
    candidate_groups: Mapping[str, Sequence[Mapping[str, object]]],
    manifest_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    branch_ids = {_branch_id(row) for row in manifest_rows if _branch_id(row)}
    path_branch_counts: Counter[str] = Counter()
    path_examples: defaultdict[str, list[str]] = defaultdict(list)
    for branch_id in sorted(str(branch) for branch in branch_ids):
        path_values: defaultdict[str, set[str]] = defaultdict(set)
        for row in candidate_groups.get(branch_id, ()):
            for path, value in _flatten_trainable(_mapping(row.get("trainable_public_input"))).items():
                path_values[path].add(value)
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
    non_action_varying = {
        path: payload
        for path, payload in varying.items()
        if payload["field_family"] == "non_action_public"
    }
    return {
        "branch_count": len(branch_ids),
        "varying_paths": varying,
        "varying_path_count": len(varying),
        "non_action_public_varying_paths": non_action_varying,
        "non_action_public_varying_path_count": len(non_action_varying),
        "non_action_public_fields_branch_constant": not non_action_varying,
        "candidate_action_varies_by_branch_count": path_branch_counts.get("candidate_action", 0),
        "candidate_action_index_varies_by_branch_count": path_branch_counts.get("candidate_action_index", 0),
    }


def _run_probes(dataset: Sequence[Mapping[str, object]]) -> dict[str, object]:
    train = [row for row in dataset if row.get("split") == "train"]
    action_prior_stats = _positive_rate_table(train, lambda row: row["action_key"])
    interaction_stats = _positive_rate_table(train, lambda row: row["interaction_key"])
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
                interaction_stats,
                row["interaction_key"],
                fallback=action_prior_stats.get(row["action_key"], 0.0),
            ),
        ),
    }
    return {
        name: _probe_report(name, rows)
        for name, rows in probe_rows.items()
    }


def _empty_probe_reports() -> dict[str, object]:
    return {
        name: _probe_report(name, ())
        for name in (ACTION_ORDER_BASELINE, ACTION_ONLY_BASELINE, PRIMARY_PROBE)
    }


def _probe_definitions() -> dict[str, object]:
    return {
        ACTION_ORDER_BASELINE: "Select legal candidates by candidate_action name order.",
        ACTION_ONLY_BASELINE: (
            "Report-only train-split positive-rate table keyed only by candidate_action."
        ),
        PRIMARY_PROBE: (
            "Report-only positive-rate interaction keyed by candidate_action and "
            "public state/context bins, with action-prior fallback."
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
        "material_gain_recall": _material_gain_recall(branch_predictions),
        "action_distribution": _action_distribution(branch_predictions),
        "unsupported_action_audit": _unsupported_action_audit(branch_predictions),
    }


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


def _metric_gate(
    *,
    source_integrity: Mapping[str, object],
    feature_variance: Mapping[str, object],
    probes: Mapping[str, object],
    comparisons: Mapping[str, object],
    primary: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    if source_integrity.get("passed") is not True:
        failures.append("source_integrity_failed")
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
        failures.append("material_gain_recall_below_floor")
    unsupported = _mapping(primary.get("unsupported_action_audit"))
    if unsupported.get("passed") is not True:
        failures.append("unsupported_selection_detected")
    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "primary_probe": PRIMARY_PROBE,
        "source_integrity_passed": source_integrity.get("passed"),
        "heldout_signal_beats_action_only_baseline": comparisons.get(
            "heldout_signal_beats_action_only_baseline"
        ),
        "heldout_signal_beats_action_order_baseline": comparisons.get(
            "heldout_signal_beats_action_order_baseline"
        ),
        "seed29_passed": _mapping(primary.get("seed29_evaluation")).get("passed"),
        "fixture_open_passed": _mapping(primary.get("fixture_open_evaluation")).get("passed"),
        "dominant_predicted_action_share_passed": action.get("passed"),
        "material_gain_recall_passed": material.get("passed"),
        "unsupported_action_rate_passed": unsupported.get("passed"),
        "leakage_passed": source_integrity.get("trainable_leakage_count") == 0,
        "non_action_public_fields_branch_constant": feature_variance.get(
            "non_action_public_fields_branch_constant"
        ),
    }


def _classification(
    source_integrity: Mapping[str, object],
    metric_gate: Mapping[str, object],
) -> dict[str, object]:
    if source_integrity.get("passed") is not True:
        primary = "candidate_ranker_capacity_source_integrity_failed"
    elif metric_gate.get("passed") is True:
        primary = "candidate_ranker_capacity_ready_for_review"
    else:
        primary = "candidate_ranker_capacity_blocked_by_signal"
    return {
        "primary": primary,
        "labels": _dedupe_allowed(
            [primary, "diagnostics_only_no_runtime_promotion", "readiness_rerun_blocked"]
        ),
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(classification: Mapping[str, object]) -> dict[str, object]:
    ready = classification.get("primary") == "candidate_ranker_capacity_ready_for_review"
    return {
        "next_step": (
            "review_candidate_ranker_capacity_before_any_shadow_scorer_proposal"
            if ready
            else "candidate_ranker_surface_needs_non_action_public_signal_or_new_probe_design"
        ),
        "summary": (
            "v128 is diagnostics-only. It does not train, serialize a runtime "
            "artifact, authorize readiness, or authorize downstream shadow scoring."
        ),
        "capacity_ready_for_review": ready,
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


def _positive_rate_table(rows: Sequence[Mapping[str, object]], key_fn) -> dict[object, float]:
    totals: Counter[object] = Counter()
    positives: Counter[object] = Counter()
    for row in rows:
        key = key_fn(row)
        totals[key] += 1
        if row.get("is_positive") is True:
            positives[key] += 1
    return {
        key: positives[key] / totals[key]
        for key in totals
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
            if row.get("resolution_legal") is True and row.get("observation_legal") is True
        ]
        selected = sorted(
            legal or rows,
            key=lambda row: (-score_fn(row), str(row.get("candidate_action")), str(row.get("archive_row_id"))),
        )[0]
        positive = next((row for row in rows if row.get("is_positive") is True), rows[0])
        predictions.append(
            {
                "branch_id": branch_id,
                "split": positive.get("split"),
                "seed": positive.get("seed"),
                "fixture_group": positive.get("fixture_group"),
                "repaired_action": positive.get("repaired_action"),
                "predicted_action": selected.get("candidate_action"),
                "prediction_correct": selected.get("is_positive") is True,
                "unsupported_action": not (
                    selected.get("resolution_legal") is True
                    and selected.get("observation_legal") is True
                ),
                "repaired_label_material_gain_positive": positive.get("material_gain_label") is True,
                "repaired_label_material_gain_exact_match_recalled": (
                    selected.get("is_positive") is True
                    and positive.get("material_gain_label") is True
                ),
                "candidate_set_has_material_gain_candidate": any(
                    row.get("material_gain_label") is True for row in rows
                ),
                "selected_material_gain_candidate": selected.get("material_gain_label") is True,
            }
        )
    return tuple(predictions)


def _score_action_order(row: Mapping[str, object]) -> float:
    actions = sorted(str(candidate) for candidate in _mapping(_mapping(row.get("trainable_public_input")).get("action_mask")).keys())
    try:
        return 1.0 - (actions.index(str(row.get("candidate_action"))) / max(1, len(actions)))
    except ValueError:
        return 0.0


def _score_lookup(
    row: Mapping[str, object],
    table: Mapping[object, float],
    key: object,
    *,
    fallback: float = 0.0,
) -> float:
    return float(table.get(key, fallback))


def _metrics(predictions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    total = len(predictions)
    correct = sum(1 for row in predictions if row.get("prediction_correct") is True)
    unsupported = sum(1 for row in predictions if row.get("unsupported_action") is True)
    labels = Counter(str(row.get("repaired_action")) for row in predictions)
    predicted = Counter(str(row.get("predicted_action")) for row in predictions)
    material_positive = sum(
        1 for row in predictions if row.get("repaired_label_material_gain_positive")
    )
    material_recalled = sum(
        1
        for row in predictions
        if row.get("repaired_label_material_gain_exact_match_recalled")
    )
    material_any = sum(1 for row in predictions if row.get("candidate_set_has_material_gain_candidate"))
    selected_material = sum(1 for row in predictions if row.get("selected_material_gain_candidate"))
    return {
        "row_count": total,
        "correct_count": correct,
        "accuracy": _ratio(correct, total),
        "dominant_label": _dominant(labels),
        "dominant_label_share": _dominant(labels)["share"],
        "dominant_predicted_action": _dominant(predicted),
        "repaired_action_counts": _counter_to_dict(labels),
        "predicted_action_counts": _counter_to_dict(predicted),
        "unsupported_action_count": unsupported,
        "unsupported_action_rate": _ratio(unsupported, total),
        "repaired_label_material_gain_positive_count": material_positive,
        "repaired_label_material_gain_exact_match_recalled_count": material_recalled,
        "repaired_label_material_gain_exact_match_recall": _ratio(
            material_recalled,
            material_positive,
        ),
        "candidate_sets_with_any_material_gain_candidate": material_any,
        "selected_material_positive_candidate_count": selected_material,
        "selected_material_positive_candidate_share": _ratio(selected_material, material_any),
    }


def _seed29_evaluation(predictions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    rows = [row for row in predictions if str(row.get("seed")) == "29"]
    metrics = _metrics(rows)
    passed = _group_passes(metrics, require_material=True)
    return {"passed": passed, "seed": 29, "metrics": metrics}


def _fixture_open_evaluation(predictions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    groups: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in predictions:
        groups[str(row.get("fixture_group"))].append(row)
    reports = {name: _metrics(rows) for name, rows in sorted(groups.items())}
    return {
        "passed": bool(reports) and all(_group_passes(report) for report in reports.values()),
        "groups": reports,
        "group_count": len(reports),
    }


def _material_gain_recall(predictions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    metrics = _metrics(predictions)
    recall = metrics["repaired_label_material_gain_exact_match_recall"]
    passed = (
        metrics["repaired_label_material_gain_positive_count"] > 0
        and recall >= MIN_MEANINGFUL_MATERIAL_GAIN_RECALL
    )
    return {
        "passed": passed,
        "metric_name": "repaired_label_material_gain_exact_match_recall",
        "minimum_recall": MIN_MEANINGFUL_MATERIAL_GAIN_RECALL,
        "repaired_label_material_gain_exact_match_recall": recall,
        "repaired_label_material_gain_positive_count": metrics[
            "repaired_label_material_gain_positive_count"
        ],
        "repaired_label_material_gain_exact_match_recalled_count": metrics[
            "repaired_label_material_gain_exact_match_recalled_count"
        ],
        "general_material_candidate_selection": {
            "candidate_sets_with_any_material_gain_candidate": metrics[
                "candidate_sets_with_any_material_gain_candidate"
            ],
            "selected_material_positive_candidate_count": metrics[
                "selected_material_positive_candidate_count"
            ],
            "selected_material_positive_candidate_share": metrics[
                "selected_material_positive_candidate_share"
            ],
        },
    }


def _action_distribution(predictions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    counts = Counter(str(row.get("predicted_action")) for row in predictions)
    dominant = _dominant(counts)
    return {
        "passed": dominant["share"] <= DOMINANT_SELECTED_ACTION_SHARE_MAX,
        "predicted_action_counts": _counter_to_dict(counts),
        "dominant_predicted_action": dominant,
        "dominant_selected_action_share_max": DOMINANT_SELECTED_ACTION_SHARE_MAX,
    }


def _unsupported_action_audit(predictions: Sequence[Mapping[str, object]]) -> dict[str, object]:
    count = sum(1 for row in predictions if row.get("unsupported_action") is True)
    return {
        "passed": count == 0,
        "unsupported_action_count": count,
        "unsupported_action_rate": _ratio(count, len(predictions)),
    }


def _leakage_audit(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    manifest_leakage = _trainable_leakage(manifest_rows)
    candidate_leakage = trainable_public_input_leakage(candidate_rows)
    count = (
        _int(manifest_leakage.get("split_key_leak_count"))
        + _int(manifest_leakage.get("forbidden_metadata_key_count"))
        + _int(candidate_leakage.get("leak_count"))
    )
    return {
        "passed": count == 0,
        "trainable_leakage_count": count,
        "manifest_trainable_leakage": manifest_leakage,
        "candidate_trainable_leakage": candidate_leakage,
    }


def _group_passes(metrics: Mapping[str, object], *, require_material: bool = False) -> bool:
    material_positive = _int(metrics.get("repaired_label_material_gain_positive_count"))
    material_ok = (
        not require_material
        or material_positive == 0
        or _number(metrics.get("repaired_label_material_gain_exact_match_recall"), 0.0)
        >= MIN_MEANINGFUL_MATERIAL_GAIN_RECALL
    )
    return (
        _int(metrics.get("row_count")) > 0
        and _int(metrics.get("unsupported_action_count")) == 0
        and _number(metrics.get("accuracy"), 0.0)
        > _number(metrics.get("dominant_label_share"), 0.0)
        and _number(_mapping(metrics.get("dominant_predicted_action")).get("share"), 1.0)
        <= DOMINANT_SELECTED_ACTION_SHARE_MAX
        and material_ok
    )


def _action_key(trainable: Mapping[str, object]) -> tuple[str]:
    return (str(trainable.get("candidate_action")),)


def _interaction_key(trainable: Mapping[str, object]) -> tuple[object, ...]:
    state = _mapping(trainable.get("target_public_state_before"))
    context = _mapping(trainable.get("public_transition_context"))
    return (
        trainable.get("candidate_action"),
        _ratio_bucket(state.get("energy_ratio")),
        _ratio_bucket(state.get("hydration_ratio")),
        _ratio_bucket(state.get("health_ratio")),
        _age_bucket(state.get("age")),
        _count_bucket(context.get("records_after_animal_resource_gain")),
        _count_bucket(context.get("ticks_after_animal_resource_gain")),
        bool(trainable.get("post_carrion_first_recovery")),
    )


def _flatten_trainable(value: object, prefix: str = "") -> dict[str, str]:
    if isinstance(value, Mapping):
        result: dict[str, str] = {}
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten_trainable(item, child))
        return result
    return {prefix: json.dumps(value, sort_keys=True)}


def _feature_family(path: str) -> str:
    if path in {"candidate_action", "candidate_action_index"}:
        return "candidate_action"
    if path.startswith("action_mask"):
        return "action_mask"
    return "non_action_public"


def _fixture_group(manifest: Mapping[str, object]) -> str:
    meta = _mapping(manifest.get("non_trainable_audit_metadata"))
    return str(meta.get("source") or meta.get("source_kind") or meta.get("source_path") or "unknown_source")


def _branch_id(row: Mapping[str, object]) -> str | None:
    value = row.get("branch_id")
    return value if isinstance(value, str) and value else None


def _ratio_bucket(value: object) -> str:
    number = _number(value, None)
    if number is None:
        return "missing"
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


def _heldout_accuracy(probes: Mapping[str, object], name: str) -> float:
    per_split = _mapping(_mapping(probes.get(name)).get("per_split_metrics"))
    validation = _mapping(per_split.get("validation"))
    test = _mapping(per_split.get("test"))
    total = _int(validation.get("row_count")) + _int(test.get("row_count"))
    correct = _int(validation.get("correct_count")) + _int(test.get("correct_count"))
    return _ratio(correct, total)


def _dominant(counts: Counter[str]) -> dict[str, object]:
    if not counts:
        return {"action": None, "count": 0, "share": 0.0, "total": 0}
    action, count = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    total = sum(counts.values())
    return {"action": action, "count": count, "share": _ratio(count, total), "total": total}


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _number(value: object, default: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
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
