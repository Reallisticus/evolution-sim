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
from evolution_sim.mind.first_recovery_shadow_scorer_execution import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V126_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_EXECUTION_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_shadow_scorer_proposal import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V125_REPORT_PATH,
    EXPECTED_MANIFEST_ROW_COUNT,
    EXPECTED_REPAIRED_ACTION_COUNTS,
    MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_PROPOSAL_SCHEMA_VERSION,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_CANDIDATE_SET_SHADOW_EXECUTION_SCHEMA_VERSION = (
    "mind_v3_first_recovery_candidate_set_shadow_execution_v1"
)
MIND_V3_FIRST_RECOVERY_CANDIDATE_SET_SHADOW_EXECUTION_POLICY = (
    "diagnostics_only_first_recovery_v127_candidate_set_shadow_execution_v1"
)
MIND_V3_FIRST_RECOVERY_CANDIDATE_SET_PREDICTION_SCHEMA_VERSION = (
    "mind_v3_first_recovery_candidate_set_prediction_v1"
)

DEFAULT_V115_REPORT_PATH = Path(
    "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.json"
)
DEFAULT_V115_ARCHIVE_ROWS_PATH = Path(
    "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.jsonl.gz"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v127-first-recovery-candidate-set-shadow-execution.json"
)
DEFAULT_PREDICTIONS_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v127-first-recovery-candidate-set-predictions.jsonl"
)

EXPECTED_V126_CLASSIFICATION = "shadow_scorer_execution_blocked_by_metrics"
EXPECTED_V125_CLASSIFICATION = "shadow_scorer_proposal_ready_for_review"
EXPECTED_V124_CLASSIFICATION = "accepted_rare_attack_contract_ready_for_shadow_proposal"
EXPECTED_V126_ECHO_FAILURES = {
    "positive_only_manifest_no_candidate_ranking_evidence",
    "candidate_action_label_echo_detected",
}

PRIMARY_SCORER = "action_order_baseline"
ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "candidate_set_shadow_execution_ready_for_review",
    "candidate_set_shadow_execution_source_integrity_failed",
    "candidate_set_shadow_execution_blocked_by_metrics",
    "candidate_set_shadow_execution_blocked_by_missing_candidate_sets",
    "diagnostics_only_no_runtime_promotion",
    "readiness_rerun_blocked",
)
AUTHORIZATION_FIELDS_FALSE: tuple[str, ...] = (
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
AUTHORIZATION_EFFECT_FIELDS_NONE: tuple[str, ...] = (
    "runtime_policy_effect",
    "trained_artifact_effect",
    "gate_effect",
    "viewer_effect",
    "replay_golden_effect",
    "foundation_effect",
)
MAX_EXAMPLES = 16


@dataclass(frozen=True, slots=True)
class FirstRecoveryCandidateSetShadowExecutionBuild:
    report: dict[str, object]
    prediction_rows: tuple[dict[str, object], ...]


def build_first_recovery_candidate_set_shadow_execution(
    *,
    v126_report: Mapping[str, object] | None = None,
    v126_report_path: str | Path | None = DEFAULT_V126_REPORT_PATH,
    v125_report: Mapping[str, object] | None = None,
    v125_report_path: str | Path | None = DEFAULT_V125_REPORT_PATH,
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
) -> FirstRecoveryCandidateSetShadowExecutionBuild:
    v126_payload, v126_evidence = _resolve_json_report(
        v126_report,
        v126_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_SHADOW_SCORER_EXECUTION_SCHEMA_VERSION,
    )
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
        "v126_report": v126_evidence,
        "v125_report": v125_evidence,
        "v124_report": v124_evidence,
        "v124_manifest": manifest_evidence,
        "v115_report": v115_evidence,
        "v115_archive_rows": v115_rows_evidence,
        "v123_report": v123_evidence,
        "v123_archive_rows": v123_rows_evidence,
    }
    candidate_rows = tuple(v115_rows) + tuple(v123_rows)
    candidate_groups = _candidate_groups(candidate_rows)
    candidate_set_audit = _candidate_set_audit(
        manifest_rows=manifest_rows,
        candidate_groups=candidate_groups,
        candidate_rows=candidate_rows,
        v115_rows=v115_rows,
        v123_rows=v123_rows,
    )
    source_integrity = _source_integrity(
        source_reports=source_reports,
        v126_report=v126_payload,
        v125_report=v125_payload,
        v124_report=v124_payload,
        v115_report=v115_payload,
        v115_rows=v115_rows,
        v123_report=v123_payload,
        v123_rows=v123_rows,
        manifest_rows=manifest_rows,
        candidate_rows=candidate_rows,
        candidate_set_audit=candidate_set_audit,
    )
    prediction_rows = (
        _prediction_rows(
            manifest_rows=manifest_rows,
            candidate_groups=candidate_groups,
        )
        if source_integrity["passed"]
        and candidate_set_audit["candidate_sets_available"]
        else tuple()
    )
    input_allowlist_audit = _input_allowlist_audit(
        manifest_rows=manifest_rows,
        candidate_rows=candidate_rows,
    )
    prediction_summary = _prediction_summary(prediction_rows)
    per_split_metrics = _per_split_metrics(prediction_rows)
    seed29 = _seed29_evaluation(prediction_rows)
    fixture_open = _fixture_open_evaluation(prediction_rows)
    material_gain = _material_gain_recall(prediction_rows)
    action_distribution = _action_distribution(prediction_rows)
    unsupported = _unsupported_action_audit(prediction_rows)
    leakage = _leakage_audit(
        manifest_rows=manifest_rows,
        candidate_rows=candidate_rows,
    )
    metric_gate = _metric_gate(
        source_integrity=source_integrity,
        candidate_set_audit=candidate_set_audit,
        prediction_summary=prediction_summary,
        seed29=seed29,
        fixture_open=fixture_open,
        material_gain=material_gain,
        action_distribution=action_distribution,
        unsupported=unsupported,
        leakage=leakage,
    )
    classification = _classification(
        source_integrity=source_integrity,
        candidate_set_audit=candidate_set_audit,
        metric_gate=metric_gate,
    )
    recommendation = _recommendation(classification)
    report = {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_CANDIDATE_SET_SHADOW_EXECUTION_SCHEMA_VERSION
        ),
        "audit_policy": MIND_V3_FIRST_RECOVERY_CANDIDATE_SET_SHADOW_EXECUTION_POLICY,
        "contract": _contract(),
        "source_reports": source_reports,
        "source_integrity": source_integrity,
        "candidate_set_audit": candidate_set_audit,
        "scorer_contract": _scorer_contract(),
        "input_allowlist_audit": input_allowlist_audit,
        "prediction_summary": prediction_summary,
        "per_split_metrics": per_split_metrics,
        "seed29_evaluation": seed29,
        "fixture_open_evaluation": fixture_open,
        "material_gain_recall": material_gain,
        "action_distribution": action_distribution,
        "unsupported_action_audit": unsupported,
        "leakage_audit": leakage,
        "metric_gate": metric_gate,
        "recommendation": recommendation,
        "classification": classification,
        "non_promoted": True,
    }
    return FirstRecoveryCandidateSetShadowExecutionBuild(
        report=report,
        prediction_rows=prediction_rows,
    )


def write_first_recovery_candidate_set_shadow_execution_outputs(
    build: FirstRecoveryCandidateSetShadowExecutionBuild,
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
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_CANDIDATE_SET_SHADOW_EXECUTION_SCHEMA_VERSION
        ),
        "diagnostics_only": True,
        "candidate_set_shadow_only": True,
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
                    MIND_V3_FIRST_RECOVERY_CANDIDATE_SET_SHADOW_EXECUTION_SCHEMA_VERSION
                ),
                "policy": MIND_V3_FIRST_RECOVERY_CANDIDATE_SET_SHADOW_EXECUTION_POLICY,
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
    v126_report: Mapping[str, object] | None,
    v125_report: Mapping[str, object] | None,
    v124_report: Mapping[str, object] | None,
    v115_report: Mapping[str, object] | None,
    v115_rows: Sequence[Mapping[str, object]],
    v123_report: Mapping[str, object] | None,
    v123_rows: Sequence[Mapping[str, object]],
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
    candidate_set_audit: Mapping[str, object],
) -> dict[str, object]:
    failures: list[str] = []
    for name, evidence in source_reports.items():
        if evidence.get("loaded") is not True:
            failures.append(f"missing_{name}")
        elif evidence.get("schema_matches") is False:
            failures.append(f"{name}_schema_mismatch")

    v126 = _mapping(v126_report or {})
    v125 = _mapping(v125_report or {})
    v124 = _mapping(v124_report or {})
    v115 = _mapping(v115_report or {})
    v123 = _mapping(v123_report or {})
    manifest_digest = stable_payload_digest(list(manifest_rows))
    manifest_branch_ids = _valid_manifest_branch_ids(manifest_rows)
    repaired_counts = _counter_to_dict(Counter(_row_action(row) for row in manifest_rows))
    manifest_leakage = _trainable_leakage(manifest_rows)
    candidate_leakage = trainable_public_input_leakage(candidate_rows)
    candidate_row_integrity = _candidate_row_integrity(candidate_rows)
    archive_lineage = {
        "v115": _archive_lineage_check(
            version="v115",
            report=v115,
            evidence=source_reports["v115_archive_rows"],
            rows=v115_rows,
        ),
        "v123": _archive_lineage_check(
            version="v123",
            report=v123,
            evidence=source_reports["v123_archive_rows"],
            rows=v123_rows,
        ),
    }
    authorization_summary = {
        version: _authorization_summary(version, report)
        for version, report in (("v126", v126), ("v125", v125), ("v124", v124))
    }

    v126_metric_failures = set(
        str(item)
        for item in _list_like(_mapping(v126.get("metric_gate")).get("failures"))
    )
    if _primary(v126) != EXPECTED_V126_CLASSIFICATION:
        failures.append("v126_classification_unexpected")
    if _mapping(v126.get("source_integrity")).get("passed") is not True:
        failures.append("v126_source_integrity_not_passed")
    if not EXPECTED_V126_ECHO_FAILURES <= v126_metric_failures:
        failures.append("v126_positive_only_metric_failures_missing")

    if _primary(v125) != EXPECTED_V125_CLASSIFICATION:
        failures.append("v125_classification_unexpected")
    v125_source = _mapping(v125.get("source_integrity"))
    if v125_source.get("passed") is not True:
        failures.append("v125_source_integrity_not_passed")
    if v125_source.get("failures") != []:
        failures.append("v125_source_failures_not_empty_or_malformed")
    if v125_source.get("manifest_digest") != manifest_digest:
        failures.append("v125_manifest_digest_mismatch")

    if _primary(v124) != EXPECTED_V124_CLASSIFICATION:
        failures.append("v124_classification_unexpected")
    v124_source = _mapping(v124.get("source_integrity"))
    v124_contract = _mapping(v124.get("contract_checks"))
    v124_manifest = _mapping(v124.get("manifest"))
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
    if len(manifest_rows) != EXPECTED_MANIFEST_ROW_COUNT:
        failures.append("v124_manifest_row_count_unexpected")
    if len(manifest_branch_ids) != len(manifest_rows):
        failures.append("v124_manifest_branch_id_missing_or_malformed")
    if len(set(manifest_branch_ids)) != len(manifest_branch_ids):
        failures.append("v124_manifest_branch_ids_not_unique")
    if repaired_counts != EXPECTED_REPAIRED_ACTION_COUNTS:
        failures.append("v124_repaired_action_counts_unexpected")

    v115_summary = _mapping(v115.get("branch_archive_summary"))
    if v115_summary.get("replay_verified") is not True:
        failures.append("v115_replay_not_verified")
    if _int(v115_summary.get("archive_row_count")) != len(v115_rows):
        failures.append("v115_archive_row_count_mismatch")
    if _int(v115_summary.get("branch_result_count")) != len(_archive_branch_ids(v115_rows)):
        failures.append("v115_branch_count_mismatch")
    if _int(v115_summary.get("heuristic_action_source_count")) != 0:
        failures.append("v115_heuristic_action_source_nonzero")

    v123_summary = _mapping(v123.get("branch_archive_summary"))
    v123_replay = _mapping(v123.get("replay_verification"))
    if v123.get("replay_verified") is not True or v123_replay.get("replay_verified") is not True:
        failures.append("v123_replay_not_verified")
    if _int(v123_summary.get("archive_row_count")) != len(v123_rows):
        failures.append("v123_archive_row_count_mismatch")
    if _int(v123_summary.get("branch_result_count")) != len(_archive_branch_ids(v123_rows)):
        failures.append("v123_branch_count_mismatch")
    if _int(v123_summary.get("heuristic_action_source_count")) != 0:
        failures.append("v123_heuristic_action_source_nonzero")
    if _mapping(v123.get("source_integrity")).get("passed") is not True:
        failures.append("v123_source_integrity_not_passed")
    if _mapping(v123.get("source_integrity")).get("failures") != []:
        failures.append("v123_source_failures_not_empty_or_malformed")

    if manifest_leakage["split_key_leak_count"]:
        failures.append("manifest_trainable_split_leakage_detected")
    if manifest_leakage["forbidden_metadata_key_count"]:
        failures.append("manifest_trainable_metadata_leakage_detected")
    if candidate_leakage.get("leak_count"):
        failures.append("candidate_trainable_leakage_detected")
    failures.extend(candidate_row_integrity["failure_labels"])

    if candidate_set_audit.get("missing_candidate_set_branch_count") != 0:
        failures.append("candidate_set_branches_missing")
    if candidate_set_audit.get("repaired_action_missing_from_candidate_set_count") != 0:
        failures.append("repaired_action_missing_from_candidate_set")
    if candidate_set_audit.get("unsupported_repaired_labels_count") != 0:
        failures.append("unsupported_repaired_labels_in_candidate_set")
    repaired_join = _mapping(candidate_set_audit.get("repaired_archive_row_join"))
    for label, count_key in (
        (
            "repaired_archive_row_id_missing_or_malformed",
            "missing_or_malformed_count",
        ),
        ("repaired_archive_row_id_not_in_candidate_set", "not_found_count"),
        (
            "repaired_archive_row_id_duplicate_in_candidate_set",
            "duplicate_match_count",
        ),
        ("repaired_archive_row_id_branch_mismatch", "branch_mismatch_count"),
        ("repaired_archive_row_id_action_mismatch", "action_mismatch_count"),
    ):
        if _int(repaired_join.get(count_key)) != 0:
            failures.append(label)

    v126_source = _mapping(v126.get("source_integrity"))
    if v126_source.get("manifest_digest") != manifest_digest:
        failures.append("v126_manifest_digest_mismatch")
    for field in ("manifest_digest_matches_v124", "manifest_digest_matches_v125"):
        if field in v126_source and v126_source.get(field) is not True:
            failures.append(f"v126_{field}_not_true")

    for lineage in archive_lineage.values():
        failures.extend(_list_like(lineage.get("failures")))

    for summary in authorization_summary.values():
        failures.extend(_list_like(summary.get("failures")))

    return {
        "passed": not failures,
        "failures": sorted(set(failures)),
        "v126_classification_primary": _primary(v126),
        "v126_metric_failures": sorted(v126_metric_failures),
        "v125_classification_primary": _primary(v125),
        "v124_classification_primary": _primary(v124),
        "manifest_row_count": len(manifest_rows),
        "manifest_digest": manifest_digest,
        "manifest_digest_matches_v124": v124_manifest.get("manifest_digest")
        == manifest_digest,
        "manifest_digest_matches_v125": v125_source.get("manifest_digest")
        == manifest_digest,
        "manifest_digest_matches_v126": v126_source.get("manifest_digest")
        == manifest_digest,
        "unique_manifest_branch_count": len(set(manifest_branch_ids)),
        "repaired_action_counts": repaired_counts,
        "v115_archive_row_count": len(v115_rows),
        "v115_branch_count": len(_archive_branch_ids(v115_rows)),
        "v115_replay_verified": v115_summary.get("replay_verified"),
        "v123_archive_row_count": len(v123_rows),
        "v123_branch_count": len(_archive_branch_ids(v123_rows)),
        "v123_replay_verified": v123_replay.get("replay_verified"),
        "archive_lineage_checks": archive_lineage,
        "candidate_row_integrity": candidate_row_integrity,
        "repaired_archive_row_join": repaired_join,
        "manifest_trainable_leakage": manifest_leakage,
        "candidate_trainable_leakage": candidate_leakage,
        "authorization_summary": authorization_summary,
    }


def _candidate_set_audit(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_groups: Mapping[str, Sequence[Mapping[str, object]]],
    candidate_rows: Sequence[Mapping[str, object]],
    v115_rows: Sequence[Mapping[str, object]],
    v123_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    missing: list[str] = []
    unsupported: list[dict[str, object]] = []
    repaired_missing: list[dict[str, object]] = []
    branch_sizes: Counter[int] = Counter()
    repaired_archive_join = _repaired_archive_row_join(
        manifest_rows=manifest_rows,
        candidate_rows=candidate_rows,
    )
    positive_rows = 0
    negative_rows = 0
    repaired_present = 0
    branches_with_multiple = 0
    for manifest in manifest_rows:
        branch_id = _branch_id(manifest)
        if branch_id is None:
            continue
        candidates = list(candidate_groups.get(branch_id, ()))
        if not candidates:
            missing.append(branch_id)
            continue
        branch_sizes[len(candidates)] += 1
        if len(candidates) > 1:
            branches_with_multiple += 1
        repaired_action = _row_action(manifest)
        positives = [
            row
            for row in candidates
            if str(row.get("candidate_action")) == repaired_action
        ]
        if positives:
            repaired_present += 1
            positive_rows += len(positives)
            negative_rows += max(0, len(candidates) - len(positives))
            selected = positives[0]
            if (
                selected.get("resolution_legal") is not True
                or selected.get("observation_legal") is not True
            ):
                unsupported.append(
                    {
                        "branch_id": branch_id,
                        "repaired_action": repaired_action,
                        "archive_row_id": selected.get("archive_row_id"),
                        "resolution_legal": selected.get("resolution_legal"),
                        "observation_legal": selected.get("observation_legal"),
                    }
                )
        else:
            negative_rows += len(candidates)
            repaired_missing.append(
                {
                    "branch_id": branch_id,
                    "repaired_action": repaired_action,
                    "candidate_actions": sorted(
                        str(row.get("candidate_action")) for row in candidates
                    ),
                }
            )
    branch_count = len({_branch_id(row) for row in manifest_rows if _branch_id(row)})
    total_candidate_rows = sum(
        len(candidate_groups.get(branch_id, ()))
        for branch_id in {
            _branch_id(row) for row in manifest_rows if _branch_id(row)
        }
    )
    ranking_evidence = (
        branch_count > 0
        and not missing
        and branches_with_multiple == branch_count
        and positive_rows >= branch_count
        and negative_rows > 0
        and not repaired_missing
        and not unsupported
    )
    return {
        "branch_count": branch_count,
        "total_candidate_rows": total_candidate_rows,
        "candidate_rows_by_source_archive": {
            "v115": len(v115_rows),
            "v123": len(v123_rows),
        },
        "candidate_rows_per_branch_distribution": {
            str(size): count for size, count in sorted(branch_sizes.items())
        },
        "branches_with_multiple_candidates": branches_with_multiple,
        "positive_rows_count": positive_rows,
        "negative_rows_count": negative_rows,
        "missing_candidate_set_branch_count": len(missing),
        "missing_candidate_set_branches": missing[:MAX_EXAMPLES],
        "repaired_action_present_in_candidate_set_count": repaired_present,
        "repaired_action_missing_from_candidate_set_count": len(repaired_missing),
        "repaired_action_missing_examples": repaired_missing[:MAX_EXAMPLES],
        "unsupported_repaired_labels_count": len(unsupported),
        "unsupported_repaired_label_examples": unsupported[:MAX_EXAMPLES],
        "repaired_archive_row_join": repaired_archive_join,
        "candidate_ranking_evidence_available": ranking_evidence,
        "candidate_sets_available": not missing,
    }


def _repaired_archive_row_join(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    by_archive_id: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in candidate_rows:
        archive_row_id = row.get("archive_row_id")
        if isinstance(archive_row_id, str) and archive_row_id:
            by_archive_id[archive_row_id].append(row)
    counts: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    exact_join_count = 0
    for index, manifest in enumerate(manifest_rows):
        branch_id = _branch_id(manifest)
        repaired_action = _row_action(manifest)
        repaired_archive_row_id = manifest.get("repaired_archive_row_id")
        if not isinstance(repaired_archive_row_id, str) or not repaired_archive_row_id:
            _record_join_issue(
                counts,
                examples,
                "repaired_archive_row_id_missing_or_malformed",
                index,
                manifest,
                None,
            )
            continue
        matches = by_archive_id.get(repaired_archive_row_id, [])
        if not matches:
            _record_join_issue(
                counts,
                examples,
                "repaired_archive_row_id_not_in_candidate_set",
                index,
                manifest,
                None,
            )
            continue
        if len(matches) != 1:
            _record_join_issue(
                counts,
                examples,
                "repaired_archive_row_id_duplicate_in_candidate_set",
                index,
                manifest,
                matches[0],
            )
            continue
        candidate = matches[0]
        candidate_branch_id = _candidate_branch_id(candidate)
        candidate_action = candidate.get("candidate_action")
        branch_ok = candidate_branch_id == branch_id
        action_ok = candidate_action == repaired_action
        if not branch_ok:
            _record_join_issue(
                counts,
                examples,
                "repaired_archive_row_id_branch_mismatch",
                index,
                manifest,
                candidate,
            )
        if not action_ok:
            _record_join_issue(
                counts,
                examples,
                "repaired_archive_row_id_action_mismatch",
                index,
                manifest,
                candidate,
            )
        if branch_ok and action_ok:
            exact_join_count += 1
    return {
        "exact_join_count": exact_join_count,
        "expected_join_count": len(manifest_rows),
        "missing_or_malformed_count": counts[
            "repaired_archive_row_id_missing_or_malformed"
        ],
        "not_found_count": counts["repaired_archive_row_id_not_in_candidate_set"],
        "duplicate_match_count": counts[
            "repaired_archive_row_id_duplicate_in_candidate_set"
        ],
        "branch_mismatch_count": counts["repaired_archive_row_id_branch_mismatch"],
        "action_mismatch_count": counts["repaired_archive_row_id_action_mismatch"],
        "failure_counts": _counter_to_dict(counts),
        "examples": examples,
        "passed": exact_join_count == len(manifest_rows) and not counts,
    }


def _record_join_issue(
    counts: Counter[str],
    examples: list[dict[str, object]],
    label: str,
    index: int,
    manifest: Mapping[str, object],
    candidate: Mapping[str, object] | None,
) -> None:
    counts[label] += 1
    if len(examples) >= MAX_EXAMPLES:
        return
    candidate_payload = _mapping(candidate or {})
    examples.append(
        {
            "row_index": index,
            "failure": label,
            "branch_id": manifest.get("branch_id"),
            "repaired_action": manifest.get("repaired_action"),
            "repaired_archive_row_id": manifest.get("repaired_archive_row_id"),
            "candidate_branch_id": _candidate_branch_id(candidate_payload),
            "candidate_action": candidate_payload.get("candidate_action"),
            "candidate_archive_row_id": candidate_payload.get("archive_row_id"),
        }
    )


def _prediction_rows(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_groups: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[dict[str, object], ...]:
    assignments = _stratified_three_way_assignments(manifest_rows)
    rows: list[dict[str, object]] = []
    for index, manifest in enumerate(sorted(manifest_rows, key=_manifest_sort_key)):
        branch_id = _branch_id(manifest)
        if branch_id is None:
            continue
        candidates = list(candidate_groups.get(branch_id, ()))
        selected_by_baseline = {
            "action_order_baseline": _select_action_order(candidates),
            "dominant_action_stay_baseline": _select_preferred_action(candidates, "stay"),
            "public_need_rule_baseline": _select_public_need_rule(candidates),
        }
        primary = selected_by_baseline[PRIMARY_SCORER]
        repaired_action = _row_action(manifest)
        prediction = {
            "schema_version": (
                MIND_V3_FIRST_RECOVERY_CANDIDATE_SET_PREDICTION_SCHEMA_VERSION
            ),
            "row_index": index,
            "branch_id": branch_id,
            "branch_id_role": "non_trainable_audit_metadata",
            "split": assignments.get(branch_id),
            "split_assignment_role": "evaluation_group_only_not_scorer_input",
            "repaired_action": repaired_action,
            "primary_scorer": PRIMARY_SCORER,
            "predicted_action": _candidate_action(primary),
            "predicted_archive_row_id": _archive_row_id(primary),
            "prediction_correct": _candidate_action(primary) == repaired_action,
            "unsupported_action": not _prediction_supported(primary),
            "repaired_label_material_gain_positive": _material_gain_positive_for_label(
                candidates,
                repaired_action,
            ),
            "repaired_label_material_gain_exact_match_recalled": bool(
                _material_gain_positive_for_label(candidates, repaired_action)
                and _candidate_action(primary) == repaired_action
            ),
            "candidate_set_has_material_gain_candidate": any(
                candidate.get("material_gain_label") is True for candidate in candidates
            ),
            "selected_material_gain_candidate": bool(
                primary and primary.get("material_gain_label") is True
            ),
            "baseline_predictions": {
                name: _baseline_prediction_payload(row, repaired_action)
                for name, row in selected_by_baseline.items()
            },
            "candidate_set": [
                _candidate_payload(candidate, repaired_action)
                for candidate in sorted(candidates, key=_candidate_sort_key)
            ],
            "scorer_input_boundary": {
                "candidate_action_is_candidate_being_scored": True,
                "branch_id_used_as_input": False,
                "source_or_seed_used_as_input": False,
                "outcome_or_objective_used_as_input": False,
                "trainable_public_input_contents_exposed": False,
            },
            "non_trainable_audit_metadata": {
                "current_oracle_action": manifest.get("current_oracle_action"),
                "current_archive_row_id": manifest.get("current_archive_row_id"),
                "repaired_archive_row_id": manifest.get("repaired_archive_row_id"),
                "selected_observation_digest": manifest.get(
                    "selected_observation_digest"
                ),
                "source_metadata": _mapping(
                    manifest.get("non_trainable_audit_metadata")
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


def _candidate_payload(
    candidate: Mapping[str, object],
    repaired_action: str,
) -> dict[str, object]:
    trainable = _mapping(candidate.get("trainable_public_input"))
    return {
        "archive_row_id": candidate.get("archive_row_id"),
        "candidate_action": candidate.get("candidate_action"),
        "is_repaired_label": str(candidate.get("candidate_action")) == repaired_action,
        "resolution_legal": candidate.get("resolution_legal"),
        "observation_legal": candidate.get("observation_legal"),
        "material_gain_label": candidate.get("material_gain_label"),
        "oracle_rank": candidate.get("oracle_rank"),
        "trainable_public_input": {
            "contents_exposed": False,
            "digest": stable_payload_digest(trainable),
            "top_level_keys": sorted(str(key) for key in trainable.keys()),
        },
    }


def _source_to_archive(source: str, rows: Sequence[Mapping[str, object]]) -> dict[str, str]:
    return {str(row.get("archive_row_id")): source for row in rows}


def _input_allowlist_audit(
    *,
    manifest_rows: Sequence[Mapping[str, object]],
    candidate_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    observed = sorted(
        {
            str(key)
            for row in candidate_rows
            for key in _mapping(row.get("trainable_public_input")).keys()
        }
    )
    manifest_leakage = _trainable_leakage(manifest_rows)
    candidate_leakage = trainable_public_input_leakage(candidate_rows)
    return {
        "policy": "candidate_rows_trainable_public_input_only",
        "observed_candidate_trainable_top_level_keys": observed,
        "candidate_action_role": "candidate_being_scored_not_final_label_copy",
        "split_assignment_excluded_from_scorer_input": True,
        "branch_seed_source_metadata_excluded_from_scorer_input": True,
        "outcome_and_objective_fields_excluded_from_scorer_input": True,
        "trainable_public_input_contents_embedded_in_v127": False,
        "manifest_trainable_leakage": manifest_leakage,
        "candidate_trainable_leakage": candidate_leakage,
    }


def _scorer_contract() -> dict[str, object]:
    return {
        "diagnostics_only": True,
        "candidate_set_shadow_only": True,
        "primary_scorer": PRIMARY_SCORER,
        "primary_scorer_description": (
            "Ranks legal candidate rows by deterministic candidate-action order. "
            "This is a report-only baseline, not a trained or runtime scorer."
        ),
        "candidate_action_role": "candidate_being_scored_not_copied_final_label",
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
        "baselines_reported": [
            "action_order_baseline",
            "dominant_action_stay_baseline",
            "public_need_rule_baseline",
        ],
        "coefficients_serialized": False,
        "runtime_loadable_artifact_created": False,
        "training_executed": False,
    }


def _prediction_summary(
    prediction_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    primary = _prediction_metrics(prediction_rows)
    baseline_metrics = {
        name: _baseline_metrics(prediction_rows, name)
        for name in (
            "action_order_baseline",
            "dominant_action_stay_baseline",
            "public_need_rule_baseline",
        )
    }
    return {
        "prediction_rows_path": str(DEFAULT_PREDICTIONS_OUTPUT_PATH),
        "prediction_row_count": len(prediction_rows),
        "prediction_rows_digest": stable_payload_digest(list(prediction_rows)),
        "primary_scorer": PRIMARY_SCORER,
        **primary,
        "baseline_metrics": baseline_metrics,
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
    return {split: _prediction_metrics(rows) for split, rows in grouped.items()}


def _seed29_evaluation(
    prediction_rows: Sequence[Mapping[str, object]],
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
    passed = (
        _int(metrics.get("row_count")) > 0
        and _int(metrics.get("unsupported_action_count")) == 0
        and _number(metrics.get("accuracy")) is not None
        and float(metrics["accuracy"]) > float(metrics["dominant_label_share"])
        and _number(_mapping(metrics.get("dominant_predicted_action")).get("share"))
        is not None
        and float(_mapping(metrics.get("dominant_predicted_action"))["share"])
        <= DOMINANT_SELECTED_ACTION_SHARE_MAX
        and (
            _int(metrics.get("repaired_label_material_gain_positive_count")) == 0
            or (
                _number(
                    metrics.get("repaired_label_material_gain_exact_match_recall")
                )
                is not None
                and float(
                    metrics["repaired_label_material_gain_exact_match_recall"]
                )
                >= MIN_MEANINGFUL_MATERIAL_GAIN_RECALL
            )
        )
    )
    return {
        "passed": passed,
        "label": (
            "candidate_set_seed29_passes"
            if passed
            else "candidate_set_seed29_failed"
        ),
        "seed": 29,
        "evaluation_group_role": "heldout_grouping_only_not_scorer_input",
        "metrics": metrics,
        "dominant_selected_action_share_max": DOMINANT_SELECTED_ACTION_SHARE_MAX,
        "material_gain_recall_floor": MIN_MEANINGFUL_MATERIAL_GAIN_RECALL,
    }


def _fixture_open_evaluation(
    prediction_rows: Sequence[Mapping[str, object]],
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
        group: _prediction_metrics(rows) for group, rows in sorted(groups.items())
    }
    passed = bool(group_reports) and all(
        _int(report.get("row_count")) > 0
        and _int(report.get("unsupported_action_count")) == 0
        and _number(report.get("accuracy")) is not None
        and float(report["accuracy"]) > float(report["dominant_label_share"])
        and _number(_mapping(report.get("dominant_predicted_action")).get("share"))
        is not None
        and float(_mapping(report.get("dominant_predicted_action"))["share"])
        <= DOMINANT_SELECTED_ACTION_SHARE_MAX
        for report in group_reports.values()
    )
    return {
        "passed": passed,
        "label": (
            "candidate_set_fixture_open_passes"
            if passed
            else "candidate_set_fixture_open_failed"
        ),
        "source_or_fixture_identity_used_as_scorer_input": False,
        "group_pass_criteria": {
            "unsupported_action_count": 0,
            "accuracy_above_dominant_label_baseline": True,
            "dominant_predicted_action_share_max": DOMINANT_SELECTED_ACTION_SHARE_MAX,
        },
        "group_count": len(group_reports),
        "groups": group_reports,
    }


def _material_gain_recall(
    prediction_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    positives = [
        row
        for row in prediction_rows
        if row.get("repaired_label_material_gain_positive")
    ]
    recalled = [
        row
        for row in positives
        if row.get("repaired_label_material_gain_exact_match_recalled")
    ]
    recall = _ratio(len(recalled), len(positives))
    candidate_sets_with_material = [
        row
        for row in prediction_rows
        if row.get("candidate_set_has_material_gain_candidate")
    ]
    selected_material = [
        row
        for row in candidate_sets_with_material
        if row.get("selected_material_gain_candidate")
    ]
    selected_material_recall = _ratio(
        len(selected_material),
        len(candidate_sets_with_material),
    )
    passed = bool(positives) and recall >= MIN_MEANINGFUL_MATERIAL_GAIN_RECALL
    return {
        "passed": passed,
        "label": (
            "repaired_label_material_gain_exact_match_recall_passes"
            if passed
            else "repaired_label_material_gain_exact_match_recall_below_floor"
        ),
        "metric_name": "repaired_label_material_gain_exact_match_recall",
        "minimum_recall": MIN_MEANINGFUL_MATERIAL_GAIN_RECALL,
        "repaired_label_material_gain_exact_match_recall": recall,
        "repaired_label_material_gain_positive_count": len(positives),
        "repaired_label_material_gain_exact_match_recalled_count": len(recalled),
        "generic_material_gain_recall_claimed": False,
        "general_material_candidate_selection": {
            "candidate_sets_with_any_material_gain_candidate": len(
                candidate_sets_with_material
            ),
            "selected_material_positive_candidate_count": len(selected_material),
            "selected_material_positive_candidate_share": selected_material_recall,
        },
        "claim_exact_material_gain_label": True,
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
            "candidate_set_action_distribution_clean"
            if passed
            else "candidate_set_action_distribution_collapsed"
        ),
        "predicted_action_counts": _counter_to_dict(counts),
        "dominant_predicted_action": dominant,
        "dominant_selected_action_share_max": DOMINANT_SELECTED_ACTION_SHARE_MAX,
    }


def _unsupported_action_audit(
    prediction_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    count = 0
    examples: list[dict[str, object]] = []
    for row in prediction_rows:
        if row.get("unsupported_action") is not True:
            continue
        count += 1
        if len(examples) < MAX_EXAMPLES:
            examples.append(
                {
                    "branch_id": row.get("branch_id"),
                    "predicted_action": row.get("predicted_action"),
                    "repaired_action": row.get("repaired_action"),
                    "predicted_archive_row_id": row.get("predicted_archive_row_id"),
                }
            )
    return {
        "passed": count == 0,
        "unsupported_action_count": count,
        "unsupported_action_rate": _ratio(count, len(prediction_rows)),
        "unsupported_action_rate_required": 0.0,
        "examples": examples,
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
        "trainable_leakage_required": 0,
        "manifest_trainable_leakage": manifest_leakage,
        "candidate_trainable_leakage": candidate_leakage,
    }


def _metric_gate(
    *,
    source_integrity: Mapping[str, object],
    candidate_set_audit: Mapping[str, object],
    prediction_summary: Mapping[str, object],
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
    if candidate_set_audit.get("candidate_sets_available") is not True:
        failures.append("candidate_sets_missing")
    if candidate_set_audit.get("candidate_ranking_evidence_available") is not True:
        failures.append("candidate_ranking_evidence_missing")
    if prediction_summary.get("accuracy", 0.0) <= prediction_summary.get(
        "dominant_label_share",
        0.0,
    ):
        failures.append("candidate_ranking_signal_missing_or_weak")
    if seed29.get("passed") is not True:
        failures.append("seed29_failed")
    if fixture_open.get("passed") is not True:
        failures.append("fixture_open_failed")
    if material_gain.get("passed") is not True:
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
        "candidate_ranking_evidence_available": candidate_set_audit.get(
            "candidate_ranking_evidence_available"
        ),
        "candidate_sets_available": candidate_set_audit.get("candidate_sets_available"),
        "dominant_predicted_action_share_passed": action_distribution.get("passed"),
        "unsupported_action_rate_passed": unsupported.get("passed"),
        "trainable_leakage_passed": leakage.get("passed"),
        "material_gain_recall_passed": material_gain.get("passed"),
        "material_gain_metric_name": material_gain.get("metric_name"),
        "seed29_reported_and_passed": seed29.get("passed"),
        "fixture_open_reported_and_passed": fixture_open.get("passed"),
    }


def _classification(
    *,
    source_integrity: Mapping[str, object],
    candidate_set_audit: Mapping[str, object],
    metric_gate: Mapping[str, object],
) -> dict[str, object]:
    if candidate_set_audit.get("candidate_sets_available") is not True:
        primary = "candidate_set_shadow_execution_blocked_by_missing_candidate_sets"
    elif source_integrity.get("passed") is not True:
        primary = "candidate_set_shadow_execution_source_integrity_failed"
    elif metric_gate.get("passed") is True:
        primary = "candidate_set_shadow_execution_ready_for_review"
    else:
        primary = "candidate_set_shadow_execution_blocked_by_metrics"
    return {
        "primary": primary,
        "labels": _dedupe_allowed(
            [primary, "diagnostics_only_no_runtime_promotion", "readiness_rerun_blocked"]
        ),
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(classification: Mapping[str, object]) -> dict[str, object]:
    ready = classification.get("primary") == "candidate_set_shadow_execution_ready_for_review"
    return {
        "next_step": (
            "review_v127_candidate_set_shadow_diagnostic_before_any_new_scorer_design"
            if ready
            else "design_non_tautological_candidate_ranker_before_any_shadow_or_runtime_planning"
        ),
        "summary": (
            "v127 builds a diagnostics-only candidate-set evaluation surface. It "
            "does not train, serialize a runtime artifact, authorize readiness, or "
            "promote any shadow scorer."
        ),
        "candidate_set_artifact_ready_for_next_scorer_design": (
            classification.get("primary")
            in {
                "candidate_set_shadow_execution_ready_for_review",
                "candidate_set_shadow_execution_blocked_by_metrics",
            }
        ),
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
    label_counts = Counter(str(row.get("repaired_action")) for row in prediction_rows)
    predicted_counts = Counter(str(row.get("predicted_action")) for row in prediction_rows)
    dominant_label = _dominant_action(label_counts)
    dominant_predicted = _dominant_action(predicted_counts)
    material_positive = sum(
        1
        for row in prediction_rows
        if row.get("repaired_label_material_gain_positive")
    )
    material_recalled = sum(1 for row in prediction_rows if row.get("material_gain_recalled"))
    repaired_material_recalled = sum(
        1
        for row in prediction_rows
        if row.get("repaired_label_material_gain_exact_match_recalled")
    )
    candidate_sets_with_material = sum(
        1
        for row in prediction_rows
        if row.get("candidate_set_has_material_gain_candidate")
    )
    selected_material = sum(
        1 for row in prediction_rows if row.get("selected_material_gain_candidate")
    )
    return {
        "row_count": row_count,
        "correct_count": correct,
        "accuracy": _ratio(correct, row_count),
        "repaired_action_counts": _counter_to_dict(label_counts),
        "predicted_action_counts": _counter_to_dict(predicted_counts),
        "dominant_label": dominant_label,
        "dominant_label_share": dominant_label["share"],
        "dominant_predicted_action": dominant_predicted,
        "unsupported_action_count": unsupported,
        "unsupported_action_rate": _ratio(unsupported, row_count),
        "repaired_label_material_gain_positive_count": material_positive,
        "repaired_label_material_gain_exact_match_recalled_count": (
            repaired_material_recalled
        ),
        "repaired_label_material_gain_exact_match_recall": _ratio(
            repaired_material_recalled,
            material_positive,
        ),
        "candidate_sets_with_any_material_gain_candidate": candidate_sets_with_material,
        "selected_material_positive_candidate_count": selected_material,
        "selected_material_positive_candidate_share": _ratio(
            selected_material,
            candidate_sets_with_material,
        ),
    }


def _baseline_metrics(
    prediction_rows: Sequence[Mapping[str, object]],
    baseline_name: str,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for row in prediction_rows:
        baseline = _mapping(_mapping(row.get("baseline_predictions")).get(baseline_name))
        rows.append(
            {
                "repaired_action": row.get("repaired_action"),
                "predicted_action": baseline.get("predicted_action"),
                "prediction_correct": baseline.get("prediction_correct"),
                "unsupported_action": baseline.get("unsupported_action"),
                "repaired_label_material_gain_positive": row.get(
                    "repaired_label_material_gain_positive"
                ),
                "repaired_label_material_gain_exact_match_recalled": bool(
                    row.get("repaired_label_material_gain_positive")
                    and baseline.get("prediction_correct") is True
                ),
                "candidate_set_has_material_gain_candidate": row.get(
                    "candidate_set_has_material_gain_candidate"
                ),
                "selected_material_gain_candidate": False,
            }
        )
    return _prediction_metrics(rows)


def _candidate_groups(
    candidate_rows: Sequence[Mapping[str, object]],
) -> dict[str, tuple[Mapping[str, object], ...]]:
    grouped: defaultdict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in candidate_rows:
        branch_id = _candidate_branch_id(row)
        if branch_id is not None:
            grouped[branch_id].append(row)
    return {
        branch_id: tuple(sorted(rows, key=_candidate_sort_key))
        for branch_id, rows in grouped.items()
    }


def _candidate_row_integrity(
    candidate_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    for index, row in enumerate(candidate_rows):
        branch_id = _candidate_branch_id(row)
        archive_row_id = row.get("archive_row_id")
        candidate_action = row.get("candidate_action")
        if branch_id is None:
            _record_integrity_failure(
                counts,
                examples,
                "candidate_row_branch_id_missing_or_malformed",
                index,
                row,
            )
        if not isinstance(archive_row_id, str) or not archive_row_id:
            _record_integrity_failure(
                counts,
                examples,
                "candidate_row_archive_row_id_missing_or_malformed",
                index,
                row,
            )
        elif branch_id is not None and not archive_row_id.startswith(
            f"{branch_id}::action::"
        ):
            _record_integrity_failure(
                counts,
                examples,
                "candidate_row_archive_row_id_branch_prefix_mismatch",
                index,
                row,
            )
        elif isinstance(candidate_action, str) and (
            archive_row_id.rsplit("::action::", 1)[-1] != candidate_action
        ):
            _record_integrity_failure(
                counts,
                examples,
                "candidate_row_archive_row_id_action_suffix_mismatch",
                index,
                row,
            )
        if row.get("replay_verification_result") is not True:
            _record_integrity_failure(
                counts,
                examples,
                "candidate_row_replay_verification_not_true",
                index,
                row,
            )
        if not _valid_sha256_digest(row.get("replay_verification_digest")):
            _record_integrity_failure(
                counts,
                examples,
                "candidate_row_replay_digest_missing_or_malformed",
                index,
                row,
            )
        if not isinstance(row.get("trainable_public_input"), Mapping):
            _record_integrity_failure(
                counts,
                examples,
                "candidate_row_trainable_public_input_missing_or_malformed",
                index,
                row,
            )
    return {
        "passed": not counts,
        "failure_labels": sorted(counts),
        "failure_count": sum(counts.values()),
        "failure_counts": _counter_to_dict(counts),
        "examples": examples,
    }


def _record_integrity_failure(
    counts: Counter[str],
    examples: list[dict[str, object]],
    label: str,
    index: int,
    row: Mapping[str, object],
) -> None:
    counts[label] += 1
    if len(examples) >= MAX_EXAMPLES:
        return
    examples.append(
        {
            "row_index": index,
            "branch_id": _candidate_branch_id(row),
            "archive_row_id": row.get("archive_row_id"),
            "candidate_action": row.get("candidate_action"),
            "failure": label,
        }
    )


def _select_action_order(
    candidates: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    legal = _legal_candidates(candidates)
    return sorted(legal, key=_candidate_sort_key)[0] if legal else None


def _select_preferred_action(
    candidates: Sequence[Mapping[str, object]],
    action: str,
) -> Mapping[str, object] | None:
    legal = _legal_candidates(candidates)
    preferred = [row for row in legal if row.get("candidate_action") == action]
    return sorted(preferred, key=_candidate_sort_key)[0] if preferred else _select_action_order(legal)


def _select_public_need_rule(
    candidates: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    legal = _legal_candidates(candidates)
    if not legal:
        return None
    state = _mapping(_mapping(legal[0].get("trainable_public_input")).get("target_public_state_before"))
    energy = _number(state.get("energy_ratio"))
    hydration = _number(state.get("hydration_ratio"))
    if energy is not None and energy < 0.65:
        selected = _select_preferred_action(legal, "eat")
        if selected is not None and selected.get("candidate_action") == "eat":
            return selected
    if hydration is not None and hydration < 0.65:
        selected = _select_preferred_action(legal, "drink")
        if selected is not None and selected.get("candidate_action") == "drink":
            return selected
    return _select_preferred_action(legal, "stay")


def _legal_candidates(
    candidates: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    return [
        row
        for row in candidates
        if row.get("resolution_legal") is True and row.get("observation_legal") is True
    ]


def _baseline_prediction_payload(
    candidate: Mapping[str, object] | None,
    repaired_action: str,
) -> dict[str, object]:
    predicted = _candidate_action(candidate)
    return {
        "predicted_action": predicted,
        "predicted_archive_row_id": _archive_row_id(candidate),
        "prediction_correct": predicted == repaired_action,
        "unsupported_action": not _prediction_supported(candidate),
    }


def _prediction_supported(candidate: Mapping[str, object] | None) -> bool:
    return bool(
        candidate
        and candidate.get("resolution_legal") is True
        and candidate.get("observation_legal") is True
    )


def _material_gain_positive_for_label(
    candidates: Sequence[Mapping[str, object]],
    repaired_action: str,
) -> bool:
    return any(
        str(row.get("candidate_action")) == repaired_action
        and row.get("material_gain_label") is True
        for row in candidates
    )


def _archive_lineage_check(
    *,
    version: str,
    report: Mapping[str, object],
    evidence: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    summary = _mapping(report.get("branch_archive_summary"))
    reported_digest = (
        summary.get("archive_rows_sha256")
        or report.get("archive_rows_sha256")
        or report.get("archive_rows_digest")
    )
    reported_path = report.get("archive_rows_path")
    loaded_digest = evidence.get("file_sha256") or stable_payload_digest(list(rows))
    loaded_path = evidence.get("path")
    failures: list[str] = []
    if reported_digest is not None and reported_digest != loaded_digest:
        failures.append(f"{version}_archive_rows_digest_mismatch")
    if (
        reported_path is not None
        and loaded_path is not None
        and str(reported_path) != str(loaded_path)
    ):
        failures.append(f"{version}_archive_rows_path_mismatch")
    return {
        "passed": not failures,
        "failures": failures,
        "reported_archive_rows_digest": reported_digest,
        "loaded_archive_rows_digest": loaded_digest,
        "archive_rows_digest_matches": (
            reported_digest == loaded_digest if reported_digest is not None else None
        ),
        "reported_archive_rows_path": reported_path,
        "loaded_archive_rows_path": loaded_path,
        "archive_rows_path_matches": (
            str(reported_path) == str(loaded_path)
            if reported_path is not None and loaded_path is not None
            else None
        ),
    }


def _authorization_summary(
    version: str,
    report: Mapping[str, object],
) -> dict[str, object]:
    recommendation = _mapping(report.get("recommendation"))
    contract = _mapping(report.get("contract"))
    failures: list[str] = []
    present_false_or_none: dict[str, object] = {}
    legacy_missing: list[str] = []
    for field in AUTHORIZATION_FIELDS_FALSE:
        value_present = field in recommendation or field in contract
        value = recommendation.get(field) if field in recommendation else contract.get(field)
        if not value_present:
            legacy_missing.append(field)
            continue
        if value not in (False, None):
            failures.append(f"{version}_{field}_not_false")
        else:
            present_false_or_none[field] = False if value is None else value
    present_effects: dict[str, object] = {}
    for field in AUTHORIZATION_EFFECT_FIELDS_NONE:
        value_present = field in recommendation or field in contract
        value = recommendation.get(field) if field in recommendation else contract.get(field)
        if not value_present:
            legacy_missing.append(field)
            continue
        if value not in ("none", None):
            failures.append(f"{version}_{field}_not_none")
        else:
            present_effects[field] = "none" if value is None else value
    return {
        "passed": not failures,
        "failures": failures,
        "present_false_or_none_authorization_fields": present_false_or_none,
        "present_none_or_none_effect_fields": present_effects,
        "legacy_missing_authorization_fields": sorted(legacy_missing),
        "legacy_missing_fields_normalized_to_no_authorization": bool(legacy_missing),
    }


def _archive_branch_ids(rows: Sequence[Mapping[str, object]]) -> set[str]:
    return {
        branch_id
        for row in rows
        if (branch_id := _candidate_branch_id(row)) is not None
    }


def _valid_manifest_branch_ids(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return [
        branch_id
        for row in rows
        if (branch_id := _branch_id(row)) is not None
    ]


def _branch_id(row: Mapping[str, object]) -> str | None:
    value = row.get("branch_id")
    return value if isinstance(value, str) and value else None


def _candidate_branch_id(row: Mapping[str, object]) -> str | None:
    value = _mapping(row.get("provenance")).get("branch_id")
    return value if isinstance(value, str) and value else None


def _row_action(row: Mapping[str, object]) -> str:
    return str(row.get("repaired_action"))


def _candidate_action(row: Mapping[str, object] | None) -> str | None:
    value = _mapping(row or {}).get("candidate_action")
    return value if isinstance(value, str) and value else None


def _archive_row_id(row: Mapping[str, object] | None) -> str | None:
    value = _mapping(row or {}).get("archive_row_id")
    return value if isinstance(value, str) and value else None


def _candidate_sort_key(row: Mapping[str, object]) -> tuple[str, str]:
    return (str(row.get("candidate_action")), str(row.get("archive_row_id")))


def _manifest_sort_key(row: Mapping[str, object]) -> tuple[str, str]:
    return (str(row.get("repaired_action")), str(row.get("branch_id")))


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


def _valid_sha256_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


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
