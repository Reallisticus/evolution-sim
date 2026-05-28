from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.mind.first_recovery_branch_archive import (
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    archive_rows_sha256,
    trainable_public_input_leakage,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _counter_to_dict,
    _int,
    _list,
    _mapping,
    _number,
    _resolve_archive_rows,
    _resolve_json_report,
    _round,
    _share,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_ARCHIVE_BLOCKER_DIAGNOSTIC_SCHEMA_VERSION = (
    "mind_v3_first_recovery_archive_blocker_diagnostic_v1"
)
MIND_V3_FIRST_RECOVERY_ARCHIVE_BLOCKER_DIAGNOSTIC_POLICY = (
    "diagnostics_only_first_recovery_v115_archive_blocker_analysis_v1"
)

DEFAULT_ARCHIVE_REPORT_PATH = Path(
    "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.json"
)
DEFAULT_ARCHIVE_ROWS_PATH = Path(
    "output/mind/mind-v3-v115-expanded-v109-first-recovery-branch-archive.jsonl.gz"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v116-first-recovery-archive-blocker-diagnostic.json"
)

EXPECTED_V115_SELECTED_TARGET_COUNT = 106
EXPECTED_V115_SKIPPED_TARGET_COUNT = 0
EXPECTED_V115_MATERIALIZED_BRANCH_POINT_COUNT = 106
EXPECTED_V115_ARCHIVE_ROW_COUNT = 530
EXPECTED_V115_ACTION_RUN_COUNT = 530
EXPECTED_V115_UNIQUE_BRANCH_ID_COUNT = 106
EXPECTED_V115_BRANCH_SIZE_DISTRIBUTION = {
    "3": 10,
    "4": 24,
    "5": 32,
    "6": 36,
    "7": 4,
}
EXPECTED_V115_DOMINANT_ORACLE_ACTION = "stay"
EXPECTED_V115_DOMINANT_ORACLE_ACTION_COUNT = 64
EXPECTED_V115_DOMINANT_ORACLE_ACTION_SHARE = 0.603774
EXPECTED_V115_RESOLUTION_INVALID_COUNT = 4
EXPECTED_V115_RECOMMENDATION = (
    "do_not_start_v110_until_archive_support_blockers_are_resolved"
)

DOMINANT_ACTION_SHARE_MAX = 0.50
OBJECTIVE_TIE_PROXY_SHARE_MIN = 0.25
MAX_EXAMPLES = 12

EXPECTED_RESOLUTION_INVALID_PUBLIC_CAUSES = frozenset(
    {
        "resolution_invalid_occupancy_race",
        "resolution_invalid_blocked_route",
        "resolution_invalid_depleted_resource",
        "resolution_invalid_other_public_path",
    }
)

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "diagnostics_only_no_runtime_promotion",
    "missing_evidence_inconclusive",
    "v115_archive_verified",
    "v115_archive_mismatch",
    "trainable_signal_leakage_detected",
    "stay_oracle_dominance_detected",
    "stay_dominance_branch_outcome_correlated",
    "objective_scoring_artifact_risk_present",
    "data_support_artifact_not_primary",
    "resolution_invalid_public_expected_drift",
    "resolution_invalid_unexplained",
    "downstream_shadow_scorer_blocked",
    "downstream_shadow_scorer_allowed",
    "readiness_rerun_blocked",
)


class FirstRecoveryArchiveBlockerDiagnosticError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FirstRecoveryArchiveBlockerDiagnosticBuild:
    report: dict[str, object]


@dataclass(frozen=True, slots=True)
class _BranchGrouping:
    groups: tuple[tuple[str, tuple[dict[str, object], ...]], ...]
    missing_branch_id_rows: tuple[dict[str, object], ...]


def build_first_recovery_archive_blocker_diagnostic(
    *,
    archive_report: Mapping[str, object] | None = None,
    archive_report_path: str | Path | None = DEFAULT_ARCHIVE_REPORT_PATH,
    archive_rows: Sequence[Mapping[str, object]] | None = None,
    archive_rows_path: str | Path | None = DEFAULT_ARCHIVE_ROWS_PATH,
    enforce_expected_v115_facts: bool | None = None,
) -> FirstRecoveryArchiveBlockerDiagnosticBuild:
    contract = _contract()
    archive_payload, archive_evidence = _resolve_json_report(
        archive_report,
        archive_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    )
    rows, rows_evidence = _resolve_archive_rows(archive_rows, archive_rows_path)
    branch_grouping = _branch_groups(rows)
    groups = branch_grouping.groups
    enforce_facts = (
        _mapping(archive_evidence).get("in_memory") is not True
        if enforce_expected_v115_facts is None
        else bool(enforce_expected_v115_facts)
    )
    source_reports = _source_reports(
        archive_report=archive_payload,
        archive_evidence=archive_evidence,
        archive_rows=rows,
        rows_evidence=rows_evidence,
        group_count=len(groups),
    )
    source_verification = _source_archive_verification(
        archive_report=archive_payload,
        archive_rows=rows,
        branch_grouping=branch_grouping,
        enforce_expected_v115_facts=enforce_facts,
    )
    stay_dominance = _stay_dominance_analysis(groups, source_verification)
    resolution_invalid = _resolution_invalid_analysis(rows)
    missing_evidence = _missing_evidence(
        source_reports=source_reports,
        source_verification=source_verification,
    )
    classification = _classification(
        missing_evidence=missing_evidence,
        source_verification=source_verification,
        stay_dominance_analysis=stay_dominance,
        resolution_invalid_analysis=resolution_invalid,
    )
    recommendation = _recommendation(
        classification=classification,
        source_verification=source_verification,
        stay_dominance_analysis=stay_dominance,
        resolution_invalid_analysis=resolution_invalid,
    )
    report = {
        "schema_version": (
            MIND_V3_FIRST_RECOVERY_ARCHIVE_BLOCKER_DIAGNOSTIC_SCHEMA_VERSION
        ),
        "audit_policy": MIND_V3_FIRST_RECOVERY_ARCHIVE_BLOCKER_DIAGNOSTIC_POLICY,
        "contract": contract,
        "source_reports": source_reports,
        "source_archive_verification": source_verification,
        "stay_dominance_analysis": stay_dominance,
        "resolution_invalid_analysis": resolution_invalid,
        "classification": classification,
        "recommendation": recommendation,
        "non_promoted": True,
    }
    return FirstRecoveryArchiveBlockerDiagnosticBuild(report=report)


def write_first_recovery_archive_blocker_diagnostic_report(
    build: FirstRecoveryArchiveBlockerDiagnosticBuild,
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
            MIND_V3_FIRST_RECOVERY_ARCHIVE_BLOCKER_DIAGNOSTIC_SCHEMA_VERSION
        ),
        "diagnostics_only": True,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "gate_effect": "none",
        "replay_golden_effect": "none",
        "summary_only_effect": "none",
        "observation_field_change": False,
        "viewer_effect": "none",
        "archive_replay_executed": False,
        "readiness_rerun_executed": False,
        "runtime_policy_implemented": False,
        "trained_artifact_emitted": False,
        "private_world_state_input": False,
        "private_world_state_serialized": False,
        "identity_fields_used_for_diagnostic_grouping_only": True,
        "branch_identity_trainable_signal": False,
        "fixture_identity_trainable_signal": False,
        "seed_identity_trainable_signal": False,
        "source_identity_trainable_signal": False,
        "logged_action_trainable_signal": False,
        "diagnostic_policy": (
            "read_only_v115_archive_blocker_analysis_over_serialized_public_rows"
        ),
        "contract_digest": stable_payload_digest(
            {
                "schema_version": (
                    MIND_V3_FIRST_RECOVERY_ARCHIVE_BLOCKER_DIAGNOSTIC_SCHEMA_VERSION
                ),
                "audit_policy": (
                    MIND_V3_FIRST_RECOVERY_ARCHIVE_BLOCKER_DIAGNOSTIC_POLICY
                ),
            }
        ),
    }


def _source_reports(
    *,
    archive_report: Mapping[str, object] | None,
    archive_evidence: Mapping[str, object],
    archive_rows: Sequence[Mapping[str, object]],
    rows_evidence: Mapping[str, object],
    group_count: int,
) -> dict[str, object]:
    return {
        "archive_report": archive_evidence,
        "archive_rows": rows_evidence,
        "archive_schema_version": _mapping(archive_report or {}).get("schema_version"),
        "archive_row_count": len(archive_rows),
        "archive_branch_target_group_count": int(group_count),
        "source_report_policy": "inputs_are_read_only_diagnostics_sources",
    }


def _source_archive_verification(
    *,
    archive_report: Mapping[str, object] | None,
    archive_rows: Sequence[Mapping[str, object]],
    branch_grouping: _BranchGrouping,
    enforce_expected_v115_facts: bool,
) -> dict[str, object]:
    groups = branch_grouping.groups
    report = _mapping(archive_report or {})
    reconstruction = _mapping(report.get("row_reconstruction"))
    selection = _mapping(report.get("target_selection"))
    summary = _mapping(report.get("branch_archive_summary"))
    legality = _mapping(report.get("legality_summary"))
    oracle = _mapping(report.get("oracle_label_summary"))
    recommendation = _mapping(report.get("research_recommendation"))
    branch_shape = _branch_shape_verification(
        branch_grouping=branch_grouping,
        oracle=oracle,
    )
    leakage = trainable_public_input_leakage(archive_rows)
    row_replay_false_count = sum(
        1 for row in archive_rows if row.get("replay_verification_result") is False
    )
    reconstructed_count_report = _int_or_none(
        reconstruction.get("reconstructed_first_recovery_row_count")
    )
    expected_constructible_count_report = _int_or_none(
        reconstruction.get("expected_constructible_first_recovery_row_count")
    )
    full_reconstruction_count_summary = _int_or_none(
        summary.get("full_reconstruction_row_count")
    )
    selected_target_count_report = _int_or_none(selection.get("selected_target_count"))
    selected_row_count_summary = _int_or_none(summary.get("selected_row_count"))
    skipped_target_count_report = _int_or_none(selection.get("skipped_target_count"))
    reconstructed_count = (
        reconstructed_count_report
        if reconstructed_count_report is not None
        else (
            expected_constructible_count_report
            if expected_constructible_count_report is not None
            else (full_reconstruction_count_summary or 0)
        )
    )
    selected_target_count = (
        selected_target_count_report
        if selected_target_count_report is not None
        else (selected_row_count_summary or 0)
    )
    skipped_target_count = (
        skipped_target_count_report
        if skipped_target_count_report is not None
        else max(0, reconstructed_count - selected_target_count)
    )
    source_value_consistency = _source_value_consistency(
        reconstructed_count_report=reconstructed_count_report,
        expected_constructible_count_report=expected_constructible_count_report,
        full_reconstruction_count_summary=full_reconstruction_count_summary,
        selected_target_count_report=selected_target_count_report,
        selected_row_count_summary=selected_row_count_summary,
        skipped_target_count_report=skipped_target_count_report,
        unique_branch_id_count=len(groups),
        archive_row_count_summary=_int_or_none(summary.get("archive_row_count")),
        archive_row_count_jsonl=len(archive_rows),
        action_run_count_summary=_int_or_none(summary.get("action_run_count")),
    )
    facts = {
        "selected_target_count": selected_target_count,
        "skipped_target_count": skipped_target_count,
        "materialized_branch_point_count": _int(
            summary.get("materialized_branch_point_count")
        ),
        "materialization_failure_count": _int(
            summary.get("materialization_failure_count")
        ),
        "branch_result_count": _int(summary.get("branch_result_count")),
        "action_run_count": _int(summary.get("action_run_count")),
        "archive_row_count_report": _int(summary.get("archive_row_count")),
        "archive_row_count_jsonl": len(archive_rows),
        "unique_branch_id_count": len(groups),
        "replay_verified": summary.get("replay_verified") is True,
        "row_replay_verification_false_count": row_replay_false_count,
        "heuristic_action_source_count": _int(
            summary.get("heuristic_action_source_count")
        ),
        "trainable_leakage_detected": leakage.get("leakage_detected") is True,
        "trainable_leak_count": _int(leakage.get("leak_count")),
        "dominant_oracle_action": oracle.get("dominant_oracle_action"),
        "dominant_oracle_action_count": _int(oracle.get("dominant_oracle_action_count")),
        "dominant_oracle_action_share": _number(
            oracle.get("dominant_oracle_action_share")
        ),
        "resolution_invalid_count": _int(legality.get("resolution_invalid_count")),
        "report_recommendation": recommendation.get("recommendation"),
    }
    expected_checks = {
        "reconstructed_first_recovery_row_count_report": (
            reconstructed_count_report == EXPECTED_V115_SELECTED_TARGET_COUNT
        ),
        "selected_target_count_report": (
            selected_target_count_report == EXPECTED_V115_SELECTED_TARGET_COUNT
        ),
        "selected_row_count_summary": (
            selected_row_count_summary == EXPECTED_V115_SELECTED_TARGET_COUNT
        ),
        "selected_target_count": (
            facts["selected_target_count"] == EXPECTED_V115_SELECTED_TARGET_COUNT
        ),
        "skipped_target_count": (
            facts["skipped_target_count"] == EXPECTED_V115_SKIPPED_TARGET_COUNT
        ),
        "materialized_branch_point_count": (
            facts["materialized_branch_point_count"]
            == EXPECTED_V115_MATERIALIZED_BRANCH_POINT_COUNT
        ),
        "branch_result_count": facts["branch_result_count"] == EXPECTED_V115_UNIQUE_BRANCH_ID_COUNT,
        "materialization_failure_count": facts["materialization_failure_count"] == 0,
        "archive_row_count_report": (
            facts["archive_row_count_report"] == EXPECTED_V115_ARCHIVE_ROW_COUNT
        ),
        "archive_row_count_jsonl": (
            facts["archive_row_count_jsonl"] == EXPECTED_V115_ARCHIVE_ROW_COUNT
        ),
        "action_run_count": (
            facts["action_run_count"] == EXPECTED_V115_ACTION_RUN_COUNT
        ),
        "unique_branch_id_count": (
            facts["unique_branch_id_count"] == EXPECTED_V115_UNIQUE_BRANCH_ID_COUNT
        ),
        "replay_verified": facts["replay_verified"] is True,
        "row_replay_verification_false_count": (
            facts["row_replay_verification_false_count"] == 0
        ),
        "heuristic_action_source_count": facts["heuristic_action_source_count"] == 0,
        "trainable_leak_count": facts["trainable_leak_count"] == 0,
        "dominant_oracle_action": (
            facts["dominant_oracle_action"] == EXPECTED_V115_DOMINANT_ORACLE_ACTION
        ),
        "dominant_oracle_action_count": (
            facts["dominant_oracle_action_count"]
            == EXPECTED_V115_DOMINANT_ORACLE_ACTION_COUNT
        ),
        "dominant_oracle_action_share": (
            abs(
                float(facts["dominant_oracle_action_share"])
                - EXPECTED_V115_DOMINANT_ORACLE_ACTION_SHARE
            )
            <= 0.000001
        ),
        "resolution_invalid_count": (
            facts["resolution_invalid_count"] == EXPECTED_V115_RESOLUTION_INVALID_COUNT
        ),
        "report_recommendation": (
            facts["report_recommendation"] == EXPECTED_V115_RECOMMENDATION
        ),
    }
    integrity_checks = {
        "expected_branch_size_distribution": (
            branch_shape["branch_size_distribution_matches_expected"] is True
        ),
        "no_missing_branch_id_rows": (
            branch_shape["missing_branch_id_row_count"] == 0
        ),
        "no_duplicate_candidate_actions_within_branch": (
            branch_shape["duplicate_candidate_action_branch_count"] == 0
        ),
        "logged_action_present_in_each_branch": (
            branch_shape["missing_logged_action_branch_count"] == 0
        ),
        "exactly_one_rank1_row_per_branch": (
            branch_shape["rank1_integrity_passed"] is True
        ),
        "serialized_oracle_best_action_consistency": (
            branch_shape["serialized_oracle_best_action_mismatch_count"] == 0
        ),
        "branch_best_oracle_action_counts_match_report": (
            branch_shape["branch_best_oracle_action_counts_match_report"] is True
        ),
        "source_value_consistency": (
            source_value_consistency["consistency_failure_count"] == 0
        ),
    }
    integrity_failures = sorted(
        key for key, passed in integrity_checks.items() if passed is not True
    )
    observed_fact_failures = sorted(
        key for key, passed in expected_checks.items() if passed is not True
    )
    fact_failures = observed_fact_failures if enforce_expected_v115_facts else []
    return {
        "archive_schema_version": report.get("schema_version"),
        "reconstructed_first_recovery_row_count": reconstructed_count,
        **facts,
        "oracle_action_counts_report": _int_counter(oracle.get("oracle_action_counts")),
        "branch_shape": branch_shape,
        "source_value_consistency": source_value_consistency,
        "resolution_category_counts_report": _int_counter(
            legality.get("resolution_category_counts")
        ),
        "archive_rows_sha256": archive_rows_sha256(archive_rows),
        "trainable_public_input_leakage": leakage,
        "expected_v115_facts": {
            "selected_target_count": EXPECTED_V115_SELECTED_TARGET_COUNT,
            "skipped_target_count": EXPECTED_V115_SKIPPED_TARGET_COUNT,
            "materialized_branch_point_count": (
                EXPECTED_V115_MATERIALIZED_BRANCH_POINT_COUNT
            ),
            "branch_result_count": EXPECTED_V115_UNIQUE_BRANCH_ID_COUNT,
            "branch_size_distribution": dict(EXPECTED_V115_BRANCH_SIZE_DISTRIBUTION),
            "archive_row_count": EXPECTED_V115_ARCHIVE_ROW_COUNT,
            "action_run_count": EXPECTED_V115_ACTION_RUN_COUNT,
            "unique_branch_id_count": EXPECTED_V115_UNIQUE_BRANCH_ID_COUNT,
            "dominant_oracle_action": EXPECTED_V115_DOMINANT_ORACLE_ACTION,
            "dominant_oracle_action_count": (
                EXPECTED_V115_DOMINANT_ORACLE_ACTION_COUNT
            ),
            "dominant_oracle_action_share": (
                EXPECTED_V115_DOMINANT_ORACLE_ACTION_SHARE
            ),
            "resolution_invalid_count": EXPECTED_V115_RESOLUTION_INVALID_COUNT,
            "report_recommendation": EXPECTED_V115_RECOMMENDATION,
        },
        "expected_v115_checks": expected_checks,
        "expected_v115_fact_checks_enforced": bool(enforce_expected_v115_facts),
        "expected_v115_observed_fact_failures": observed_fact_failures,
        "expected_v115_fact_failures": fact_failures,
        "integrity_checks": integrity_checks,
        "integrity_failures": integrity_failures,
        "verification_passed": not fact_failures and not integrity_failures,
    }


def _branch_shape_verification(
    *,
    branch_grouping: _BranchGrouping,
    oracle: Mapping[str, object],
) -> dict[str, object]:
    groups = branch_grouping.groups
    branch_size_distribution = Counter(str(len(rows)) for _, rows in groups)
    duplicate_examples: list[dict[str, object]] = []
    missing_logged_examples: list[dict[str, object]] = []
    no_rank1_examples: list[dict[str, object]] = []
    multiple_rank1_examples: list[dict[str, object]] = []
    oracle_best_mismatch_examples: list[dict[str, object]] = []
    duplicate_branch_count = 0
    missing_logged_branch_count = 0
    no_rank1_branch_count = 0
    multiple_rank1_branch_count = 0
    oracle_best_mismatch_count = 0
    best_action_counts: Counter[str] = Counter()
    missing_branch_id_examples = [
        _row_assignment_example(row)
        for row in branch_grouping.missing_branch_id_rows[:MAX_EXAMPLES]
    ]
    for branch_id, rows in groups:
        candidate_counts = Counter(str(row.get("candidate_action")) for row in rows)
        duplicate_actions = [
            action for action, count in sorted(candidate_counts.items()) if count > 1
        ]
        if duplicate_actions:
            duplicate_branch_count += 1
            if len(duplicate_examples) < MAX_EXAMPLES:
                duplicate_examples.append(
                    {
                        "branch_id": branch_id,
                        "duplicate_candidate_actions": duplicate_actions,
                        "candidate_action_counts": _counter_to_dict(candidate_counts),
                    }
                )
        provenance = _mapping(rows[0].get("provenance")) if rows else {}
        logged_action = provenance.get("logged_action")
        if not isinstance(logged_action, str) or logged_action not in candidate_counts:
            missing_logged_branch_count += 1
            if len(missing_logged_examples) < MAX_EXAMPLES:
                missing_logged_examples.append(
                    {
                        "branch_id": branch_id,
                        "logged_action": logged_action,
                        "candidate_actions": sorted(candidate_counts),
                    }
                )
        rank1_rows = [
            row for row in rows if _int(row.get("oracle_rank"), default=-1) == 1
        ]
        if len(rank1_rows) == 0:
            no_rank1_branch_count += 1
            if len(no_rank1_examples) < MAX_EXAMPLES:
                no_rank1_examples.append(
                    {
                        "branch_id": branch_id,
                        "candidate_actions": sorted(candidate_counts),
                        "oracle_ranks": [
                            row.get("oracle_rank")
                            for row in sorted(rows, key=_row_sort_key)
                        ],
                    }
                )
            continue
        if len(rank1_rows) > 1:
            multiple_rank1_branch_count += 1
            if len(multiple_rank1_examples) < MAX_EXAMPLES:
                multiple_rank1_examples.append(
                    {
                        "branch_id": branch_id,
                        "rank1_candidate_actions": sorted(
                            str(row.get("candidate_action")) for row in rank1_rows
                        ),
                    }
                )
            continue
        best_row = dict(rank1_rows[0])
        best_action = str(best_row.get("candidate_action"))
        best_action_counts[best_action] += 1
        serialized_best_actions = sorted(
            {
                str(row.get("oracle_best_action"))
                for row in rows
                if row.get("oracle_best_action") is not None
            }
        )
        if serialized_best_actions and serialized_best_actions != [best_action]:
            oracle_best_mismatch_count += 1
            if len(oracle_best_mismatch_examples) < MAX_EXAMPLES:
                oracle_best_mismatch_examples.append(
                    {
                        "branch_id": branch_id,
                        "rank1_candidate_action": best_action,
                        "serialized_oracle_best_actions_present": serialized_best_actions,
                    }
                )
    report_counts = _int_counter(oracle.get("oracle_action_counts"))
    best_counts = _counter_to_dict(best_action_counts)
    count_deltas = {
        key: int(best_counts.get(key, 0)) - int(report_counts.get(key, 0))
        for key in sorted(set(best_counts) | set(report_counts))
        if int(best_counts.get(key, 0)) != int(report_counts.get(key, 0))
    }
    failures = []
    if len(groups) != EXPECTED_V115_UNIQUE_BRANCH_ID_COUNT:
        failures.append("unique_branch_count")
    if _counter_to_dict(branch_size_distribution) != EXPECTED_V115_BRANCH_SIZE_DISTRIBUTION:
        failures.append("branch_size_distribution")
    if branch_grouping.missing_branch_id_rows:
        failures.append("missing_branch_id_rows")
    if duplicate_branch_count:
        failures.append("duplicate_candidate_actions")
    if missing_logged_branch_count:
        failures.append("missing_logged_action")
    if no_rank1_branch_count:
        failures.append("no_rank1_row")
    if multiple_rank1_branch_count:
        failures.append("multiple_rank1_rows")
    if oracle_best_mismatch_count:
        failures.append("serialized_oracle_best_action_mismatch")
    if count_deltas:
        failures.append("branch_best_oracle_action_counts_mismatch")
    branch_size_counts = _counter_to_dict(branch_size_distribution)
    return {
        "unique_branch_count": len(groups),
        "expected_unique_branch_count": EXPECTED_V115_UNIQUE_BRANCH_ID_COUNT,
        "unique_branch_count_matches_expected": (
            len(groups) == EXPECTED_V115_UNIQUE_BRANCH_ID_COUNT
        ),
        "branch_size_distribution": branch_size_counts,
        "expected_branch_size_distribution": dict(EXPECTED_V115_BRANCH_SIZE_DISTRIBUTION),
        "branch_size_distribution_matches_expected": (
            branch_size_counts == EXPECTED_V115_BRANCH_SIZE_DISTRIBUTION
        ),
        "missing_branch_id_row_count": len(branch_grouping.missing_branch_id_rows),
        "missing_branch_id_row_examples": missing_branch_id_examples,
        "duplicate_candidate_action_branch_count": duplicate_branch_count,
        "duplicate_candidate_action_examples": duplicate_examples,
        "missing_logged_action_branch_count": missing_logged_branch_count,
        "missing_logged_action_examples": missing_logged_examples,
        "no_rank1_branch_count": no_rank1_branch_count,
        "no_rank1_examples": no_rank1_examples,
        "multiple_rank1_branch_count": multiple_rank1_branch_count,
        "multiple_rank1_examples": multiple_rank1_examples,
        "rank1_integrity_passed": (
            no_rank1_branch_count == 0 and multiple_rank1_branch_count == 0
        ),
        "serialized_oracle_best_action_mismatch_count": oracle_best_mismatch_count,
        "serialized_oracle_best_action_mismatch_examples": (
            oracle_best_mismatch_examples
        ),
        "branch_best_oracle_action_counts_from_rows": best_counts,
        "oracle_action_counts_report": report_counts,
        "branch_best_oracle_action_counts_match_report": not count_deltas,
        "branch_best_oracle_action_count_deltas": count_deltas,
        "failure_count": len(failures),
        "failures": failures,
        "logged_action_candidate_policy": (
            "diagnostic archive rows should include the logged action among "
            "candidate actions for each branch"
        ),
    }


def _source_value_consistency(
    *,
    reconstructed_count_report: int | None,
    expected_constructible_count_report: int | None,
    full_reconstruction_count_summary: int | None,
    selected_target_count_report: int | None,
    selected_row_count_summary: int | None,
    skipped_target_count_report: int | None,
    unique_branch_id_count: int,
    archive_row_count_summary: int | None,
    archive_row_count_jsonl: int,
    action_run_count_summary: int | None,
) -> dict[str, object]:
    checks = {
        "reconstructed_first_recovery_row_count": _consistency_check(
            {
                "row_reconstruction": reconstructed_count_report,
                "expected_constructible": expected_constructible_count_report,
                "branch_archive_summary": full_reconstruction_count_summary,
                "archive_rows_grouped": unique_branch_id_count,
            },
            expected=EXPECTED_V115_SELECTED_TARGET_COUNT,
        ),
        "selected_target_count": _consistency_check(
            {
                "target_selection": selected_target_count_report,
                "branch_archive_summary": selected_row_count_summary,
                "archive_rows_grouped": unique_branch_id_count,
            },
            expected=EXPECTED_V115_SELECTED_TARGET_COUNT,
        ),
        "skipped_target_count": _consistency_check(
            {"target_selection": skipped_target_count_report},
            expected=EXPECTED_V115_SKIPPED_TARGET_COUNT,
        ),
        "archive_row_count": _consistency_check(
            {
                "branch_archive_summary": archive_row_count_summary,
                "archive_rows_jsonl": archive_row_count_jsonl,
            },
            expected=EXPECTED_V115_ARCHIVE_ROW_COUNT,
        ),
        "action_run_count": _consistency_check(
            {"branch_archive_summary": action_run_count_summary},
            expected=EXPECTED_V115_ACTION_RUN_COUNT,
        ),
    }
    failures = [
        name for name, check in sorted(checks.items()) if check["consistent"] is not True
    ]
    return {
        "checks": checks,
        "consistency_failure_count": len(failures),
        "consistency_failures": failures,
        "policy": "preserve_report_and_row_source_values_instead_of_silent_fallback",
    }


def _consistency_check(
    values: Mapping[str, int | None],
    *,
    expected: int,
) -> dict[str, object]:
    missing = [name for name, value in sorted(values.items()) if value is None]
    present = {
        name: int(value)
        for name, value in sorted(values.items())
        if value is not None
    }
    mismatched = [
        name for name, value in present.items() if int(value) != int(expected)
    ]
    consistent = not missing and not mismatched and len(set(present.values())) <= 1
    return {
        "values": {name: value for name, value in sorted(values.items())},
        "expected": expected,
        "missing_sources": missing,
        "mismatched_sources": mismatched,
        "consistent": consistent,
    }


def _stay_dominance_analysis(
    groups: Sequence[tuple[str, tuple[dict[str, object], ...]]],
    source_verification: Mapping[str, object],
) -> dict[str, object]:
    best_rows = [_best_row(rows) for _, rows in groups if rows]
    action_counts = Counter(str(row.get("candidate_action")) for row in best_rows)
    dominant_action, dominant_count = _dominant(action_counts)
    branch_count = len(best_rows)
    stay_rows = [row for row in best_rows if row.get("candidate_action") == "stay"]
    non_stay_rows = [row for row in best_rows if row.get("candidate_action") != "stay"]
    tie_proxy_branch_count = 0
    stay_tie_proxy_count = 0
    for _, rows in groups:
        if not rows:
            continue
        best = _best_row(rows)
        best_key = _delta_key(best)
        best_id = best.get("archive_row_id")
        tied = any(
            row.get("archive_row_id") != best_id and _delta_key(row) == best_key
            for row in rows
        )
        if tied:
            tie_proxy_branch_count += 1
            if best.get("candidate_action") == "stay":
                stay_tie_proxy_count += 1
    stay_share = _share(len(stay_rows), branch_count)
    stay_tie_share = _share(stay_tie_proxy_count, len(stay_rows))
    source_verified = source_verification.get("verification_passed") is True
    data_support_artifact_not_primary = (
        source_verified
        and _int(source_verification.get("selected_target_count"))
        == _int(source_verification.get("reconstructed_first_recovery_row_count"))
        and _int(source_verification.get("skipped_target_count")) == 0
    )
    stay_dominant = (
        dominant_action == "stay"
        and dominant_count > 0
        and _share(dominant_count, branch_count) > DOMINANT_ACTION_SHARE_MAX
    )
    objective_artifact_risk = (
        stay_dominant
        and (
            stay_tie_share >= OBJECTIVE_TIE_PROXY_SHARE_MIN
            or _share(
                sum(1 for row in stay_rows if row.get("material_gain_label") is not True),
                len(stay_rows),
            )
            >= DOMINANT_ACTION_SHARE_MAX
        )
    )
    if stay_dominant and objective_artifact_risk:
        answer = "stay_dominance_correlated_with_branch_outcomes_but_tie_artifact_risk"
    elif stay_dominant:
        answer = "stay_dominance_correlated_with_branch_outcomes"
    elif branch_count:
        answer = "stay_not_dominant"
    else:
        answer = "missing_evidence_inconclusive"
    return {
        "answer": answer,
        "branch_count": branch_count,
        "oracle_action_counts": _counter_to_dict(action_counts),
        "oracle_action_shares": {
            action: _share(count, branch_count)
            for action, count in sorted(action_counts.items())
        },
        "oracle_action_entropy": _entropy(action_counts),
        "dominant_oracle_action": dominant_action,
        "dominant_oracle_action_count": dominant_count,
        "dominant_oracle_action_share": _share(dominant_count, branch_count),
        "stay_branch_count": len(stay_rows),
        "non_stay_branch_count": len(non_stay_rows),
        "stay_branch_share": stay_share,
        "stay_dominance_detected": stay_dominant,
        "branch_outcome_signal_present": bool(source_verified and branch_count),
        "branch_outcome_signal_interpretation": (
            "oracle ranks are serialized branch-replay outcomes; this supports "
            "correlation under the v109 objective, not runtime causality"
        ),
        "objective_scoring_artifact_risk": objective_artifact_risk,
        "objective_scoring_artifact_reason": (
            "many stay-best branches have no material gain and/or tie the serialized "
            "delta proxy, so the v109 lexicographic oracle tie break may favor stay"
        ),
        "data_support_artifact_not_primary": data_support_artifact_not_primary,
        "data_support_reason": (
            "v115 selected all reconstructed rows with zero skipped targets"
            if data_support_artifact_not_primary
            else "v115 source verification did not prove exhaustive support"
        ),
        "stay_vs_non_stay_counts": {
            "by_seed": _counts_by_dimension(best_rows, _seed_key),
            "by_source": _counts_by_dimension(best_rows, _source_key),
            "by_logged_action": _counts_by_dimension(best_rows, _logged_action_key),
            "by_tick_bucket": _counts_by_dimension(best_rows, _tick_bucket_key),
        },
        "outcome_deltas": {
            "stay": _outcome_aggregate(stay_rows),
            "non_stay": _outcome_aggregate(non_stay_rows),
        },
        "target_public_state_before_bins": _vitals_bins(best_rows),
        "tie_proxy": {
            "policy": (
                "serialized_delta_key_match_without_private_world_or_raw_branch_runs"
            ),
            "branch_count_with_any_best_tie_proxy": tie_proxy_branch_count,
            "stay_best_branch_count_with_tie_proxy": stay_tie_proxy_count,
            "stay_best_tie_proxy_share": stay_tie_share,
        },
        "examples": {
            "top_stay_branches": [
                _branch_example(row)
                for row in sorted(stay_rows, key=_example_sort_key, reverse=True)[
                    :MAX_EXAMPLES
                ]
            ],
            "strongest_non_stay_branches": [
                _branch_example(row)
                for row in sorted(non_stay_rows, key=_example_sort_key, reverse=True)[
                    :MAX_EXAMPLES
                ]
            ],
        },
    }


def _resolution_invalid_analysis(
    archive_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    invalid_rows = [
        dict(row)
        for row in archive_rows
        if row.get("resolution_legal") is False
        or row.get("resolution_invalid_public_cause") is not None
    ]
    expected: list[dict[str, object]] = []
    unexplained: list[dict[str, object]] = []
    for row in invalid_rows:
        cause = row.get("resolution_invalid_public_cause")
        first = _mapping(row.get("first_action_outcome"))
        is_expected = (
            isinstance(cause, str)
            and cause in EXPECTED_RESOLUTION_INVALID_PUBLIC_CAUSES
            and first.get("observation_legal") is True
            and first.get("resolution_legal") is False
            and first.get("invalid_reason") is not None
        )
        if is_expected:
            expected.append(row)
        else:
            unexplained.append(row)
    if unexplained:
        answer = "resolution_invalid_unexplained"
    elif invalid_rows:
        answer = "resolution_invalid_public_expected_drift"
    else:
        answer = "resolution_valid"
    return {
        "answer": answer,
        "invalid_count": len(invalid_rows),
        "expected_public_drift_count": len(expected),
        "unexplained_count": len(unexplained),
        "public_cause_counts": _counter_to_dict(
            Counter(str(row.get("resolution_invalid_public_cause")) for row in invalid_rows)
        ),
        "expected_public_drift_causes": sorted(EXPECTED_RESOLUTION_INVALID_PUBLIC_CAUSES),
        "examples": [_resolution_invalid_example(row) for row in invalid_rows[:MAX_EXAMPLES]],
        "expected_public_drift_examples": [
            _resolution_invalid_example(row) for row in expected[:MAX_EXAMPLES]
        ],
        "unexplained_examples": [
            _resolution_invalid_example(row) for row in unexplained[:MAX_EXAMPLES]
        ],
        "classification_note": (
            "public expected drift means observation-time legality changed by "
            "resolution time in serialized public masks; it is still not a "
            "runtime policy recommendation"
        ),
        "downstream_resolution_policy_note": (
            "downstream diagnostics must keep resolution_legal visible and must "
            "not treat resolution-invalid candidate rows as executable actions"
        ),
    }


def _classification(
    *,
    missing_evidence: Sequence[str],
    source_verification: Mapping[str, object],
    stay_dominance_analysis: Mapping[str, object],
    resolution_invalid_analysis: Mapping[str, object],
) -> dict[str, object]:
    labels: list[str] = ["diagnostics_only_no_runtime_promotion"]
    if source_verification.get("trainable_leakage_detected") is True:
        labels.append("trainable_signal_leakage_detected")
        labels.append("downstream_shadow_scorer_blocked")
        labels.append("readiness_rerun_blocked")
        primary = "trainable_signal_leakage_detected"
    elif missing_evidence:
        labels.append("missing_evidence_inconclusive")
        labels.append("downstream_shadow_scorer_blocked")
        labels.append("readiness_rerun_blocked")
        primary = "missing_evidence_inconclusive"
    elif source_verification.get("verification_passed") is not True:
        labels.append("v115_archive_mismatch")
        labels.append("downstream_shadow_scorer_blocked")
        labels.append("readiness_rerun_blocked")
        primary = "v115_archive_mismatch"
    else:
        labels.append("v115_archive_verified")
        if stay_dominance_analysis.get("data_support_artifact_not_primary") is True:
            labels.append("data_support_artifact_not_primary")
        if stay_dominance_analysis.get("stay_dominance_detected") is True:
            labels.append("stay_oracle_dominance_detected")
            labels.append("stay_dominance_branch_outcome_correlated")
        if stay_dominance_analysis.get("objective_scoring_artifact_risk") is True:
            labels.append("objective_scoring_artifact_risk_present")
        if _int(resolution_invalid_analysis.get("unexplained_count")) > 0:
            labels.append("resolution_invalid_unexplained")
        elif _int(resolution_invalid_analysis.get("invalid_count")) > 0:
            labels.append("resolution_invalid_public_expected_drift")
        blocked = (
            stay_dominance_analysis.get("stay_dominance_detected") is True
            or _int(resolution_invalid_analysis.get("unexplained_count")) > 0
        )
        labels.append(
            "downstream_shadow_scorer_blocked"
            if blocked
            else "downstream_shadow_scorer_allowed"
        )
        labels.append("readiness_rerun_blocked")
        primary = (
            "stay_oracle_dominance_detected"
            if stay_dominance_analysis.get("stay_dominance_detected") is True
            else (
                "resolution_invalid_unexplained"
                if _int(resolution_invalid_analysis.get("unexplained_count")) > 0
                else "v115_archive_verified"
            )
        )
    return {
        "primary": primary,
        "labels": _dedupe_allowed(labels),
        "missing_evidence": sorted(set(str(item) for item in missing_evidence)),
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(
    *,
    classification: Mapping[str, object],
    source_verification: Mapping[str, object],
    stay_dominance_analysis: Mapping[str, object],
    resolution_invalid_analysis: Mapping[str, object],
) -> dict[str, object]:
    labels = set(str(item) for item in _list(classification.get("labels")))
    downstream_blocked = "downstream_shadow_scorer_blocked" in labels
    if "trainable_signal_leakage_detected" in labels:
        next_step = "stop_until_trainable_signal_leakage_is_removed"
    elif "missing_evidence_inconclusive" in labels or "v115_archive_mismatch" in labels:
        next_step = "repair_or_regenerate_v115_archive_before_downstream_diagnostics"
    elif "stay_oracle_dominance_detected" in labels:
        next_step = "diagnose_stay_oracle_objective_tie_break_before_shadow_scorer_or_readiness"
    elif "resolution_invalid_unexplained" in labels:
        next_step = "explain_resolution_invalid_rows_before_shadow_scorer_or_readiness"
    else:
        next_step = "v115_archive_blockers_clear_for_separate_shadow_scorer_diagnostic"
    return {
        "next_step": next_step,
        "summary": _recommendation_summary(
            downstream_blocked=downstream_blocked,
            stay_dominance_analysis=stay_dominance_analysis,
            resolution_invalid_analysis=resolution_invalid_analysis,
            source_verification=source_verification,
        ),
        "v115_usable_for_downstream_shadow_scorer": not downstream_blocked,
        "v113_readiness_rerun_allowed": False,
        "readiness_rerun_reason": (
            "this diagnostic never authorizes a v113 readiness rerun; run a separate "
            "shadow-scorer support diagnostic first even if archive blockers clear"
        ),
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_change_recommended": False,
        "observation_field_change_recommended": False,
        "viewer_change_recommended": False,
        "claim_causality": False,
        "causality_note": (
            "v116 diagnoses correlation and artifact risk in the serialized v115 "
            "archive; it does not prove causal runtime behavior"
        ),
    }


def _recommendation_summary(
    *,
    downstream_blocked: bool,
    stay_dominance_analysis: Mapping[str, object],
    resolution_invalid_analysis: Mapping[str, object],
    source_verification: Mapping[str, object],
) -> str:
    if source_verification.get("verification_passed") is not True:
        return "v115 source facts do not match the expected exhaustive archive boundary."
    if stay_dominance_analysis.get("stay_dominance_detected") is True:
        return (
            "v115 is exhaustive and leakage-free, but the branch-level oracle still "
            "collapses to stay; treat this as branch-outcome correlation with "
            "objective/tie-break artifact risk, not training-ready supervision."
        )
    if _int(resolution_invalid_analysis.get("unexplained_count")) > 0:
        return "resolution-invalid rows remain unexplained, so downstream use is blocked."
    if downstream_blocked:
        return "v115 remains blocked by unresolved diagnostics."
    return "v115 archive blocker diagnostics clear; readiness still requires a separate slice."


def _missing_evidence(
    *,
    source_reports: Mapping[str, object],
    source_verification: Mapping[str, object],
) -> list[str]:
    missing: list[str] = []
    for key in ("archive_report", "archive_rows"):
        evidence = _mapping(source_reports.get(key))
        if evidence.get("loaded") is not True:
            missing.append(key)
        elif key == "archive_report" and evidence.get("schema_matches") is not True:
            missing.append("archive_report_schema")
    if not _int(source_verification.get("archive_row_count_jsonl")):
        missing.append("archive_rows")
    return sorted(set(missing))


def _branch_groups(rows: Sequence[Mapping[str, object]]) -> _BranchGrouping:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    missing: list[dict[str, object]] = []
    for row in rows:
        branch_id = _mapping(row.get("provenance")).get("branch_id")
        if isinstance(branch_id, str) and branch_id:
            grouped[branch_id].append(dict(row))
        else:
            missing.append(dict(row))
    groups = tuple(
        (branch_id, tuple(sorted(group_rows, key=_row_sort_key)))
        for branch_id, group_rows in sorted(grouped.items())
    )
    return _BranchGrouping(groups=groups, missing_branch_id_rows=tuple(missing))


def _row_assignment_example(row: Mapping[str, object]) -> dict[str, object]:
    provenance = _mapping(row.get("provenance"))
    return {
        "archive_row_id": row.get("archive_row_id"),
        "candidate_action": row.get("candidate_action"),
        "seed": row.get("seed"),
        "source_kind": row.get("source_kind"),
        "tick": row.get("tick"),
        "provenance_keys": sorted(str(key) for key in provenance),
    }


def _best_row(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return dict(sorted(rows, key=_oracle_rank_sort_key)[0])


def _row_sort_key(row: Mapping[str, object]) -> tuple[int, str]:
    return (_int(row.get("oracle_rank"), default=10**9), str(row.get("candidate_action")))


def _oracle_rank_sort_key(row: Mapping[str, object]) -> tuple[int, str]:
    return (_int(row.get("oracle_rank"), default=10**9), str(row.get("candidate_action")))


def _dominant(counts: Mapping[str, int] | Counter[str]) -> tuple[str | None, int]:
    if not counts:
        return None, 0
    key, count = sorted(counts.items(), key=lambda item: (-int(item[1]), item[0]))[0]
    return str(key), int(count)


def _entropy(counts: Mapping[str, int] | Counter[str]) -> float:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        count = int(value)
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return _round(entropy)


def _counts_by_dimension(
    rows: Sequence[Mapping[str, object]],
    key_func,
) -> dict[str, dict[str, object]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = str(key_func(row))
        bucket = "stay" if row.get("candidate_action") == "stay" else "non_stay"
        counts[key][bucket] += 1
    return {
        key: {
            "stay": counter["stay"],
            "non_stay": counter["non_stay"],
            "total": counter["stay"] + counter["non_stay"],
            "stay_share": _share(counter["stay"], counter["stay"] + counter["non_stay"]),
        }
        for key, counter in sorted(counts.items())
    }


def _outcome_aggregate(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    numeric_fields = (
        "terminal_alive_delta",
        "birth_delta",
        "death_delta",
        "death_reduction_delta",
    )
    recovery_delta = [
        _number(_mapping(row.get("recovery_vitals_deltas")).get("target_recovery_score_delta"))
        for row in rows
    ]
    payload: dict[str, object] = {
        "branch_count": len(rows),
        "material_gain_count": sum(1 for row in rows if row.get("material_gain_label") is True),
        "material_gain_share": _share(
            sum(1 for row in rows if row.get("material_gain_label") is True),
            len(rows),
        ),
        "resolution_invalid_count": sum(
            1 for row in rows if row.get("resolution_legal") is False
        ),
    }
    for field in numeric_fields:
        values = [_number(row.get(field)) for row in rows]
        payload[field] = _numeric_summary(values)
    payload["target_recovery_score_delta"] = _numeric_summary(recovery_delta)
    return payload


def _numeric_summary(values: Sequence[float]) -> dict[str, object]:
    parsed = [float(value) for value in values]
    total = sum(parsed)
    return {
        "sum": _round(total),
        "mean": _round(total / len(parsed)) if parsed else 0.0,
        "positive_count": sum(1 for value in parsed if value > 0.0),
        "zero_count": sum(1 for value in parsed if value == 0.0),
        "negative_count": sum(1 for value in parsed if value < 0.0),
    }


def _vitals_bins(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    fields = ("energy_ratio", "hydration_ratio", "health_ratio")
    bins: dict[str, dict[str, Counter[str]]] = {
        field: {"stay": Counter(), "non_stay": Counter()} for field in fields
    }
    for row in rows:
        group = "stay" if row.get("candidate_action") == "stay" else "non_stay"
        vitals = _mapping(
            _mapping(row.get("trainable_public_input")).get("target_public_state_before")
        )
        for field in fields:
            bins[field][group][_ratio_bin(vitals.get(field))] += 1
    return {
        field: {
            "stay": _counter_to_dict(groups["stay"]),
            "non_stay": _counter_to_dict(groups["non_stay"]),
        }
        for field, groups in sorted(bins.items())
    }


def _ratio_bin(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "missing"
    parsed = float(value)
    if not math.isfinite(parsed):
        return "missing"
    if parsed < 0.25:
        return "0.00-0.25"
    if parsed < 0.50:
        return "0.25-0.50"
    if parsed < 0.75:
        return "0.50-0.75"
    return "0.75-1.00"


def _branch_example(row: Mapping[str, object]) -> dict[str, object]:
    provenance = _mapping(row.get("provenance"))
    first = _mapping(row.get("first_action_outcome"))
    return {
        "archive_row_id": row.get("archive_row_id"),
        "branch_id": provenance.get("branch_id"),
        "seed": row.get("seed"),
        "source_kind": row.get("source_kind"),
        "tick": row.get("tick"),
        "candidate_action": row.get("candidate_action"),
        "logged_action": provenance.get("logged_action"),
        "oracle_rank": row.get("oracle_rank"),
        "material_gain_label": row.get("material_gain_label"),
        "branch_outcome_delta_score": _round(_delta_score(row)),
        "terminal_alive_delta": row.get("terminal_alive_delta"),
        "birth_delta": row.get("birth_delta"),
        "death_delta": row.get("death_delta"),
        "death_reduction_delta": row.get("death_reduction_delta"),
        "target_recovery_score_delta": _mapping(row.get("recovery_vitals_deltas")).get(
            "target_recovery_score_delta"
        ),
        "target_public_state_before": _mapping(
            _mapping(row.get("trainable_public_input")).get("target_public_state_before")
        ),
        "first_action_outcome": {
            "observation_legal": first.get("observation_legal"),
            "resolution_legal": first.get("resolution_legal"),
            "resolved_action": first.get("resolved_action"),
            "moved": first.get("moved"),
            "ate": first.get("ate"),
            "drank": first.get("drank"),
            "energy_ratio_delta": first.get("energy_ratio_delta"),
            "hydration_ratio_delta": first.get("hydration_ratio_delta"),
            "health_ratio_delta": first.get("health_ratio_delta"),
            "invalid_reason": first.get("invalid_reason"),
        },
    }


def _resolution_invalid_example(row: Mapping[str, object]) -> dict[str, object]:
    provenance = _mapping(row.get("provenance"))
    first = _mapping(row.get("first_action_outcome"))
    return {
        "archive_row_id": row.get("archive_row_id"),
        "branch_id": provenance.get("branch_id"),
        "seed": row.get("seed"),
        "source_kind": row.get("source_kind"),
        "tick": row.get("tick"),
        "candidate_action": row.get("candidate_action"),
        "logged_action": provenance.get("logged_action"),
        "public_cause": row.get("resolution_invalid_public_cause"),
        "first_action_outcome": {
            "observation_legal": first.get("observation_legal"),
            "resolution_legal": first.get("resolution_legal"),
            "action_valid": first.get("action_valid"),
            "resolution_action_valid": first.get("resolution_action_valid"),
            "invalid_reason": first.get("invalid_reason"),
            "resolved_action": first.get("resolved_action"),
            "moved": first.get("moved"),
            "ate": first.get("ate"),
            "drank": first.get("drank"),
        },
        "terminal_alive_delta": row.get("terminal_alive_delta"),
        "birth_delta": row.get("birth_delta"),
        "death_delta": row.get("death_delta"),
        "death_reduction_delta": row.get("death_reduction_delta"),
        "recovery_vitals_deltas": row.get("recovery_vitals_deltas"),
    }


def _example_sort_key(row: Mapping[str, object]) -> tuple[float, int, int, str]:
    return (
        _delta_score(row),
        int(row.get("material_gain_label") is True),
        -_int(row.get("oracle_rank"), default=10**9),
        str(row.get("archive_row_id")),
    )


def _delta_score(row: Mapping[str, object]) -> float:
    recovery = _number(
        _mapping(row.get("recovery_vitals_deltas")).get("target_recovery_score_delta")
    )
    return (
        100.0 * _number(row.get("terminal_alive_delta"))
        + 10.0 * _number(row.get("birth_delta"))
        + 5.0 * _number(row.get("death_reduction_delta"))
        + recovery
        + (1.0 if row.get("material_gain_label") is True else 0.0)
    )


def _delta_key(row: Mapping[str, object]) -> tuple[float, float, float, float, bool]:
    return (
        _number(row.get("terminal_alive_delta")),
        _number(row.get("birth_delta")),
        _number(row.get("death_reduction_delta")),
        _number(
            _mapping(row.get("recovery_vitals_deltas")).get(
                "target_recovery_score_delta"
            )
        ),
        row.get("material_gain_label") is True,
    )


def _seed_key(row: Mapping[str, object]) -> str:
    provenance = _mapping(row.get("provenance"))
    return str(provenance.get("seed", row.get("seed", "missing")))


def _source_key(row: Mapping[str, object]) -> str:
    provenance = _mapping(row.get("provenance"))
    return str(provenance.get("source_kind", row.get("source_kind", "missing")))


def _logged_action_key(row: Mapping[str, object]) -> str:
    return str(_mapping(row.get("provenance")).get("logged_action", "missing"))


def _tick_bucket_key(row: Mapping[str, object]) -> str:
    tick = _int(row.get("tick"), default=-1)
    if tick < 0:
        return "missing"
    if tick < 10:
        return "000-009"
    if tick < 20:
        return "010-019"
    if tick < 40:
        return "020-039"
    if tick < 80:
        return "040-079"
    return "080+"


def _int_counter(value: object) -> dict[str, int]:
    mapping = _mapping(value)
    return {
        str(key): _int(item)
        for key, item in sorted(mapping.items(), key=lambda pair: str(pair[0]))
        if _int(item) > 0
    }


def _int_or_none(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    parsed = _int(value, default=-10**12)
    return None if parsed == -10**12 else parsed


def _dedupe_allowed(labels: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    allowed = set(ALLOWED_CLASSIFICATIONS)
    deduped: list[str] = []
    for label in labels:
        if label not in allowed or label in seen:
            continue
        seen.add(label)
        deduped.append(label)
    return deduped
