from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.mind.first_recovery_archive_blocker_diagnostic import (
    DEFAULT_ARCHIVE_REPORT_PATH,
    DEFAULT_ARCHIVE_ROWS_PATH,
    build_first_recovery_archive_blocker_diagnostic,
)
from evolution_sim.mind.first_recovery_branch_archive import (
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_public_signal_audit import (
    _counter_to_dict,
    _int,
    _list,
    _mapping,
    _resolve_archive_rows,
    _resolve_json_report,
    _share,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_FIRST_RECOVERY_ORACLE_TIE_BREAK_AUDIT_SCHEMA_VERSION = (
    "mind_v3_first_recovery_oracle_tie_break_audit_v1"
)
MIND_V3_FIRST_RECOVERY_ORACLE_TIE_BREAK_AUDIT_POLICY = (
    "diagnostics_only_first_recovery_v117_oracle_tie_break_audit_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v117-first-recovery-oracle-tie-break-audit.json"
)

DOMINANT_ACTION_SHARE_MAX = 0.50
MAX_EXAMPLES = 16

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "diagnostics_only_no_runtime_promotion",
    "missing_evidence_inconclusive",
    "source_archive_verification_failed",
    "objective_support_stay_dominance",
    "tie_break_artifact_likely",
    "objective_tie_break_inconclusive",
    "objective_input_field_integrity",
    "readiness_rerun_blocked",
)


class FirstRecoveryOracleTieBreakAuditError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class FirstRecoveryOracleTieBreakAuditBuild:
    report: dict[str, object]


@dataclass(frozen=True, slots=True)
class _BranchGroup:
    branch_id: str
    rows: tuple[dict[str, object], ...]


def build_first_recovery_oracle_tie_break_audit(
    *,
    archive_report: Mapping[str, object] | None = None,
    archive_report_path: str | Path | None = DEFAULT_ARCHIVE_REPORT_PATH,
    archive_rows: Sequence[Mapping[str, object]] | None = None,
    archive_rows_path: str | Path | None = DEFAULT_ARCHIVE_ROWS_PATH,
    strict_source_verification: bool | None = None,
) -> FirstRecoveryOracleTieBreakAuditBuild:
    contract = _contract()
    archive_payload, archive_evidence = _resolve_json_report(
        archive_report,
        archive_report_path,
        expected_schema=MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    )
    rows, rows_evidence = _resolve_archive_rows(archive_rows, archive_rows_path)
    strict = (
        _mapping(archive_evidence).get("in_memory") is not True
        if strict_source_verification is None
        else bool(strict_source_verification)
    )
    objective_input_integrity = _objective_input_integrity(rows)
    source_verification = _source_archive_verification(
        archive_report=archive_payload,
        archive_report_path=archive_report_path,
        archive_rows=rows,
        archive_rows_path=archive_rows_path,
        strict_source_verification=strict,
        objective_input_integrity=objective_input_integrity,
    )
    source_reports = _source_reports(
        archive_report=archive_payload,
        archive_evidence=archive_evidence,
        rows=rows,
        rows_evidence=rows_evidence,
        strict_source_verification=strict,
    )
    objective_analysis = (
        _objective_tie_break_analysis(rows)
        if objective_input_integrity.get("passed") is True
        else _blocked_objective_tie_break_analysis(
            rows,
            objective_input_integrity=objective_input_integrity,
        )
    )
    missing_evidence = _missing_evidence(
        source_reports=source_reports,
        rows=rows,
    )
    classification = _classification(
        missing_evidence=missing_evidence,
        source_verification=source_verification,
        strict_source_verification=strict,
        objective_input_integrity=objective_input_integrity,
        objective_tie_break_analysis=objective_analysis,
    )
    recommendation = _recommendation(
        classification=classification,
        objective_tie_break_analysis=objective_analysis,
    )
    report = {
        "schema_version": MIND_V3_FIRST_RECOVERY_ORACLE_TIE_BREAK_AUDIT_SCHEMA_VERSION,
        "audit_policy": MIND_V3_FIRST_RECOVERY_ORACLE_TIE_BREAK_AUDIT_POLICY,
        "contract": contract,
        "source_reports": source_reports,
        "source_archive_verification": source_verification,
        "objective_input_integrity": objective_input_integrity,
        "objective_tie_break_analysis": objective_analysis,
        "classification": classification,
        "recommendation": recommendation,
        "non_promoted": True,
    }
    return FirstRecoveryOracleTieBreakAuditBuild(report=report)


def write_first_recovery_oracle_tie_break_audit_report(
    build: FirstRecoveryOracleTieBreakAuditBuild,
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
        "schema_version": MIND_V3_FIRST_RECOVERY_ORACLE_TIE_BREAK_AUDIT_SCHEMA_VERSION,
        "diagnostics_only": True,
        "runtime_policy_effect": "none",
        "trained_artifact_effect": "none",
        "gate_effect": "none",
        "replay_golden_effect": "none",
        "summary_only_effect": "none",
        "observation_field_change": False,
        "viewer_effect": "none",
        "archive_replay_executed": False,
        "training_executed": False,
        "readiness_rerun_executed": False,
        "runtime_policy_implemented": False,
        "claim_causality": False,
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "diagnostic_policy": (
            "read_only_serialized_v115_branch_rows_objective_tie_break_audit"
        ),
        "contract_digest": stable_payload_digest(
            {
                "schema_version": (
                    MIND_V3_FIRST_RECOVERY_ORACLE_TIE_BREAK_AUDIT_SCHEMA_VERSION
                ),
                "audit_policy": MIND_V3_FIRST_RECOVERY_ORACLE_TIE_BREAK_AUDIT_POLICY,
            }
        ),
    }


def _source_archive_verification(
    *,
    archive_report: Mapping[str, object] | None,
    archive_report_path: str | Path | None,
    archive_rows: Sequence[Mapping[str, object]],
    archive_rows_path: str | Path | None,
    strict_source_verification: bool,
    objective_input_integrity: Mapping[str, object],
) -> dict[str, object]:
    if objective_input_integrity.get("passed") is not True:
        return {
            "verification_passed": False,
            "verification_skipped": True,
            "skip_reason": "objective_input_field_integrity",
            "strict_source_verification": bool(strict_source_verification),
            "expected_v115_fact_failures": [],
            "integrity_failures": ["objective_input_field_integrity"],
            "v116_classification_primary": None,
        }
    build = build_first_recovery_archive_blocker_diagnostic(
        archive_report=archive_report,
        archive_report_path=archive_report_path,
        archive_rows=archive_rows,
        archive_rows_path=archive_rows_path,
        enforce_expected_v115_facts=strict_source_verification,
    )
    verification = dict(_mapping(build.report.get("source_archive_verification")))
    verification["strict_source_verification"] = bool(strict_source_verification)
    verification["v116_classification_primary"] = _mapping(
        build.report.get("classification")
    ).get("primary")
    return verification


def _source_reports(
    *,
    archive_report: Mapping[str, object] | None,
    archive_evidence: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    rows_evidence: Mapping[str, object],
    strict_source_verification: bool,
) -> dict[str, object]:
    return {
        "archive_report": archive_evidence,
        "archive_rows": rows_evidence,
        "archive_schema_version": _mapping(archive_report or {}).get("schema_version"),
        "archive_row_count": len(rows),
        "strict_source_verification": bool(strict_source_verification),
        "source_report_policy": "inputs_are_read_only_diagnostics_sources",
    }


def _objective_input_integrity(
    archive_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    missing_field_count = 0
    malformed_field_count = 0
    non_finite_numeric_value_count = 0
    malformed_or_missing_nested_mapping_count = 0
    failed_row_indexes: set[int] = set()
    examples: dict[str, list[dict[str, object]]] = {
        "missing_fields": [],
        "malformed_fields": [],
        "non_finite_numeric_values": [],
        "malformed_or_missing_nested_mappings": [],
    }
    for index, row in enumerate(archive_rows):
        biology = row.get("biology_homeostasis_labels")
        if not isinstance(biology, Mapping):
            malformed_or_missing_nested_mapping_count += 1
            failed_row_indexes.add(index)
            _append_integrity_example(
                examples["malformed_or_missing_nested_mappings"],
                row=row,
                row_index=index,
                field="biology_homeostasis_labels",
                issue=(
                    "missing_nested_mapping"
                    if "biology_homeostasis_labels" not in row
                    else "malformed_nested_mapping"
                ),
                value=biology,
            )
        elif "target_alive" not in biology:
            missing_field_count += 1
            failed_row_indexes.add(index)
            _append_integrity_example(
                examples["missing_fields"],
                row=row,
                row_index=index,
                field="biology_homeostasis_labels.target_alive",
                issue="missing_field",
            )
        elif type(biology.get("target_alive")) is not bool:
            malformed_field_count += 1
            failed_row_indexes.add(index)
            _append_integrity_example(
                examples["malformed_fields"],
                row=row,
                row_index=index,
                field="biology_homeostasis_labels.target_alive",
                issue="expected_bool",
                value=biology.get("target_alive"),
            )

        for field in (
            "terminal_alive_delta",
            "birth_delta",
            "death_reduction_delta",
        ):
            issue = _numeric_integrity_issue(row, field)
            if issue is None:
                continue
            failed_row_indexes.add(index)
            if issue == "missing_field":
                missing_field_count += 1
                _append_integrity_example(
                    examples["missing_fields"],
                    row=row,
                    row_index=index,
                    field=field,
                    issue=issue,
                )
            elif issue == "non_finite_numeric_value":
                non_finite_numeric_value_count += 1
                _append_integrity_example(
                    examples["non_finite_numeric_values"],
                    row=row,
                    row_index=index,
                    field=field,
                    issue=issue,
                    value=row.get(field),
                )
            else:
                malformed_field_count += 1
                _append_integrity_example(
                    examples["malformed_fields"],
                    row=row,
                    row_index=index,
                    field=field,
                    issue=issue,
                    value=row.get(field),
                )

        vitals = row.get("recovery_vitals_deltas")
        if not isinstance(vitals, Mapping):
            malformed_or_missing_nested_mapping_count += 1
            failed_row_indexes.add(index)
            _append_integrity_example(
                examples["malformed_or_missing_nested_mappings"],
                row=row,
                row_index=index,
                field="recovery_vitals_deltas",
                issue=(
                    "missing_nested_mapping"
                    if "recovery_vitals_deltas" not in row
                    else "malformed_nested_mapping"
                ),
                value=vitals,
            )
        else:
            field = "target_recovery_score_delta"
            issue = _numeric_integrity_issue(vitals, field)
            if issue is not None:
                failed_row_indexes.add(index)
                path = f"recovery_vitals_deltas.{field}"
                if issue == "missing_field":
                    missing_field_count += 1
                    _append_integrity_example(
                        examples["missing_fields"],
                        row=row,
                        row_index=index,
                        field=path,
                        issue=issue,
                    )
                elif issue == "non_finite_numeric_value":
                    non_finite_numeric_value_count += 1
                    _append_integrity_example(
                        examples["non_finite_numeric_values"],
                        row=row,
                        row_index=index,
                        field=path,
                        issue=issue,
                        value=vitals.get(field),
                    )
                else:
                    malformed_field_count += 1
                    _append_integrity_example(
                        examples["malformed_fields"],
                        row=row,
                        row_index=index,
                        field=path,
                        issue=issue,
                        value=vitals.get(field),
                    )
    total_failure_count = (
        missing_field_count
        + malformed_field_count
        + non_finite_numeric_value_count
        + malformed_or_missing_nested_mapping_count
    )
    return {
        "checked_row_count": len(archive_rows),
        "valid_row_count": len(archive_rows) - len(failed_row_indexes),
        "failed_row_count": len(failed_row_indexes),
        "missing_field_count": missing_field_count,
        "malformed_field_count": malformed_field_count,
        "non_finite_numeric_value_count": non_finite_numeric_value_count,
        "malformed_or_missing_nested_mapping_count": (
            malformed_or_missing_nested_mapping_count
        ),
        "total_failure_count": total_failure_count,
        "passed": total_failure_count == 0,
        "integrity_failures": (
            [] if total_failure_count == 0 else ["objective_input_field_integrity"]
        ),
        "field_requirements": {
            "biology_homeostasis_labels.target_alive": (
                "required explicit bool"
            ),
            "terminal_alive_delta": "required finite int_or_float_non_bool",
            "birth_delta": "required finite int_or_float_non_bool",
            "recovery_vitals_deltas.target_recovery_score_delta": (
                "required finite int_or_float_non_bool"
            ),
            "death_reduction_delta": "required finite int_or_float_non_bool",
        },
        "examples": examples,
    }


def _numeric_integrity_issue(
    payload: Mapping[str, object],
    field: str,
) -> str | None:
    if field not in payload:
        return "missing_field"
    value = payload.get(field)
    if type(value) not in (int, float):
        return "expected_finite_number"
    if not math.isfinite(float(value)):
        return "non_finite_numeric_value"
    return None


def _append_integrity_example(
    examples: list[dict[str, object]],
    *,
    row: Mapping[str, object],
    row_index: int,
    field: str,
    issue: str,
    value: object = None,
) -> None:
    if len(examples) >= MAX_EXAMPLES:
        return
    example = {
        "row_index": row_index,
        "archive_row_id": row.get("archive_row_id"),
        "branch_id": _mapping(row.get("provenance")).get("branch_id"),
        "candidate_action": row.get("candidate_action"),
        "field": field,
        "issue": issue,
    }
    if value is not None:
        example["value_repr"] = repr(value)
    examples.append(example)


def _objective_tie_break_analysis(
    archive_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    groups = _branch_groups(archive_rows)
    current_counts: Counter[str] = Counter()
    unique_counts: Counter[str] = Counter()
    tied_current_counts: Counter[str] = Counter()
    tie_size_distribution: Counter[str] = Counter()
    current_resolution_counts: Counter[str] = Counter()
    objective_best_resolution_counts: Counter[str] = Counter()
    rank1_key_not_serialized_max_examples: list[dict[str, object]] = []
    branch_examples: list[dict[str, object]] = []
    alternatives = _new_alternative_accumulators()
    unique_count = 0
    tied_count = 0
    rank1_key_not_serialized_max_count = 0
    for group in groups:
        rank1_rows = [
            row for row in group.rows if _int(row.get("oracle_rank"), default=-1) == 1
        ]
        if len(rank1_rows) != 1:
            continue
        current = rank1_rows[0]
        current_action = str(current.get("candidate_action"))
        current_counts[current_action] += 1
        current_resolution_counts[_resolution_bucket(current)] += 1
        rank1_key = _serialized_objective_key(current)
        equivalent = [
            row for row in group.rows if _serialized_objective_key(row) == rank1_key
        ]
        tie_size_distribution[str(len(equivalent))] += 1
        for row in equivalent:
            objective_best_resolution_counts[_resolution_bucket(row)] += 1
        if len(equivalent) == 1:
            unique_count += 1
            unique_counts[current_action] += 1
        else:
            tied_count += 1
            tied_current_counts[current_action] += 1
        max_key = max(_serialized_objective_key(row) for row in group.rows)
        if rank1_key != max_key:
            rank1_key_not_serialized_max_count += 1
            if len(rank1_key_not_serialized_max_examples) < MAX_EXAMPLES:
                rank1_key_not_serialized_max_examples.append(
                    {
                        "branch_id": group.branch_id,
                        "current_rank1_action": current_action,
                        "rank1_serialized_objective_key": list(rank1_key),
                        "max_serialized_objective_key": list(max_key),
                    }
                )
        selections = _tie_neutral_selections(group, equivalent)
        for name, row in selections.items():
            _record_alternative(alternatives[name], row)
        if len(branch_examples) < MAX_EXAMPLES and len(equivalent) > 1:
            branch_examples.append(
                {
                    "branch_id": group.branch_id,
                    "current_rank1_action": current_action,
                    "rank1_tie_size": len(equivalent),
                    "rank1_serialized_objective_key": list(rank1_key),
                    "rank1_equivalent_actions": sorted(
                        str(row.get("candidate_action")) for row in equivalent
                    ),
                    "resolution_legal_actions_in_tie": sorted(
                        str(row.get("candidate_action"))
                        for row in equivalent
                        if row.get("resolution_legal") is True
                    ),
                }
            )
    branch_count = len(groups)
    alternative_reports = {
        name: _finalize_alternative(payload, branch_count=branch_count)
        for name, payload in alternatives.items()
    }
    current_stay_count = current_counts["stay"]
    current_stay_share = _share(current_stay_count, branch_count)
    unique_stay_count = unique_counts["stay"]
    unique_stay_share = _share(unique_stay_count, unique_count)
    tie_neutral_stay_counts = {
        name: _int(report.get("stay_count"))
        for name, report in alternative_reports.items()
    }
    tie_neutral_stay_shares = {
        name: report.get("stay_share")
        for name, report in alternative_reports.items()
    }
    return {
        "answer": _analysis_answer(
            branch_count=branch_count,
            current_stay_share=current_stay_share,
            unique_stay_share=unique_stay_share,
            tie_neutral_stay_counts=tie_neutral_stay_counts,
        ),
        "objective_input_policy": _objective_input_policy(),
        "branch_count": branch_count,
        "current_serialized_oracle_action_counts": _counter_to_dict(current_counts),
        "current_serialized_oracle_stay_count": current_stay_count,
        "current_serialized_oracle_stay_share": current_stay_share,
        "unique_objective_best_branch_count": unique_count,
        "unique_objective_best_action_counts": _counter_to_dict(unique_counts),
        "unique_objective_best_stay_count": unique_stay_count,
        "unique_objective_best_stay_share": unique_stay_share,
        "multiple_objective_best_branch_count": tied_count,
        "tied_branch_current_serialized_oracle_action_counts": _counter_to_dict(
            tied_current_counts
        ),
        "rank1_tie_size_distribution": _counter_to_dict(tie_size_distribution),
        "rank1_key_not_serialized_max_count": rank1_key_not_serialized_max_count,
        "rank1_key_not_serialized_max_examples": (
            rank1_key_not_serialized_max_examples
        ),
        "resolution_separation": {
            "current_rank1_resolution_counts": _counter_to_dict(current_resolution_counts),
            "rank1_equivalent_candidate_resolution_counts": _counter_to_dict(
                objective_best_resolution_counts
            ),
            "policy": (
                "resolution-invalid rows are diagnostic evidence only and are not "
                "executable policy recommendations"
            ),
        },
        "tie_neutral_alternatives": alternative_reports,
        "tie_neutral_stay_counts": tie_neutral_stay_counts,
        "tie_neutral_stay_shares": tie_neutral_stay_shares,
        "examples": branch_examples,
    }


def _blocked_objective_tie_break_analysis(
    archive_rows: Sequence[Mapping[str, object]],
    *,
    objective_input_integrity: Mapping[str, object],
) -> dict[str, object]:
    branch_count = len(_branch_groups(archive_rows))
    return {
        "answer": "objective_tie_break_inconclusive",
        "objective_input_policy": _objective_input_policy(),
        "analysis_blocked": True,
        "blocked_reason": "objective_input_field_integrity",
        "blocked_failure_count": _int(
            objective_input_integrity.get("total_failure_count")
        ),
        "branch_count": branch_count,
        "current_serialized_oracle_action_counts": {},
        "current_serialized_oracle_stay_count": 0,
        "current_serialized_oracle_stay_share": _share(0, branch_count),
        "unique_objective_best_branch_count": 0,
        "unique_objective_best_action_counts": {},
        "unique_objective_best_stay_count": 0,
        "unique_objective_best_stay_share": _share(0, 0),
        "multiple_objective_best_branch_count": 0,
        "tied_branch_current_serialized_oracle_action_counts": {},
        "rank1_tie_size_distribution": {},
        "rank1_key_not_serialized_max_count": 0,
        "rank1_key_not_serialized_max_examples": [],
        "resolution_separation": {
            "current_rank1_resolution_counts": {},
            "rank1_equivalent_candidate_resolution_counts": {},
            "policy": (
                "resolution-invalid rows are diagnostic evidence only and are not "
                "executable policy recommendations"
            ),
        },
        "tie_neutral_alternatives": {},
        "tie_neutral_stay_counts": {},
        "tie_neutral_stay_shares": {},
        "examples": [],
    }


def _objective_input_policy() -> dict[str, object]:
    return {
        "scope": "serialized_v115_archive_rows_only",
        "comparison_key_order": [
            "biology_homeostasis_labels.target_alive",
            "terminal_alive_delta",
            "birth_delta",
            "target_recovery_score_delta",
            "death_reduction_delta",
        ],
        "excluded_from_key": [
            "candidate_action",
            "oracle_rank",
            "resolution_legal",
            "material_gain_label",
        ],
        "caution": (
            "v115 rows do not serialize every raw branch-run oracle input; "
            "rank-1 equivalence is exact over serialized objective fields only"
        ),
    }


def _new_alternative_accumulators() -> dict[str, dict[str, object]]:
    names = (
        "action_name",
        "prefer_non_stay_resolution_legal",
        "prefer_material_gain_resolution_legal",
        "deterministic_hash",
    )
    return {
        name: {
            "action_counts": Counter(),
            "selected_resolution_invalid_count": 0,
            "examples": [],
        }
        for name in names
    }


def _tie_neutral_selections(
    group: _BranchGroup,
    equivalent: Sequence[Mapping[str, object]],
) -> dict[str, Mapping[str, object]]:
    if not equivalent:
        return {}
    return {
        "action_name": sorted(equivalent, key=_action_name_sort_key)[0],
        "prefer_non_stay_resolution_legal": _prefer_non_stay_resolution_legal(
            equivalent
        ),
        "prefer_material_gain_resolution_legal": _prefer_material_gain_resolution_legal(
            equivalent
        ),
        "deterministic_hash": sorted(
            equivalent,
            key=lambda row: _stable_hash_key(group.branch_id, row),
        )[0],
    }


def _record_alternative(
    payload: dict[str, object],
    row: Mapping[str, object],
) -> None:
    action = str(row.get("candidate_action"))
    action_counts = payload["action_counts"]
    assert isinstance(action_counts, Counter)
    action_counts[action] += 1
    if row.get("resolution_legal") is not True:
        payload["selected_resolution_invalid_count"] = (
            _int(payload.get("selected_resolution_invalid_count")) + 1
        )
    examples = payload["examples"]
    if isinstance(examples, list) and len(examples) < MAX_EXAMPLES:
        examples.append(
            {
                "archive_row_id": row.get("archive_row_id"),
                "branch_id": _mapping(row.get("provenance")).get("branch_id"),
                "candidate_action": action,
                "resolution_legal": row.get("resolution_legal"),
                "oracle_rank": row.get("oracle_rank"),
            }
        )


def _finalize_alternative(
    payload: Mapping[str, object],
    *,
    branch_count: int,
) -> dict[str, object]:
    action_counts = payload.get("action_counts")
    counts = action_counts if isinstance(action_counts, Counter) else Counter()
    stay_count = counts["stay"]
    return {
        "action_counts": _counter_to_dict(counts),
        "stay_count": stay_count,
        "stay_share": _share(stay_count, branch_count),
        "selected_resolution_invalid_count": _int(
            payload.get("selected_resolution_invalid_count")
        ),
        "examples": _list(payload.get("examples")),
    }


def _analysis_answer(
    *,
    branch_count: int,
    current_stay_share: float,
    unique_stay_share: float,
    tie_neutral_stay_counts: Mapping[str, int],
) -> str:
    if branch_count <= 0:
        return "missing_evidence_inconclusive"
    if unique_stay_share > DOMINANT_ACTION_SHARE_MAX:
        return "objective_support_stay_dominance"
    neutralized = [
        count for count in tie_neutral_stay_counts.values()
        if _share(int(count), branch_count) <= DOMINANT_ACTION_SHARE_MAX
    ]
    if (
        current_stay_share > DOMINANT_ACTION_SHARE_MAX
        and len(neutralized) >= max(1, math.ceil(len(tie_neutral_stay_counts) / 2))
    ):
        return "tie_break_artifact_likely"
    return "objective_tie_break_inconclusive"


def _classification(
    *,
    missing_evidence: Sequence[str],
    source_verification: Mapping[str, object],
    strict_source_verification: bool,
    objective_input_integrity: Mapping[str, object],
    objective_tie_break_analysis: Mapping[str, object],
) -> dict[str, object]:
    labels = ["diagnostics_only_no_runtime_promotion", "readiness_rerun_blocked"]
    if missing_evidence:
        primary = "missing_evidence_inconclusive"
        labels.insert(0, primary)
    elif (
        strict_source_verification
        and source_verification.get("verification_passed") is not True
    ):
        primary = "source_archive_verification_failed"
        labels.insert(0, primary)
    elif objective_input_integrity.get("passed") is not True:
        primary = "objective_tie_break_inconclusive"
        labels.insert(0, "objective_input_field_integrity")
        labels.insert(0, primary)
    else:
        answer = str(objective_tie_break_analysis.get("answer"))
        primary = answer if answer in ALLOWED_CLASSIFICATIONS else (
            "objective_tie_break_inconclusive"
        )
        labels.insert(0, primary)
    return {
        "primary": primary,
        "labels": _dedupe_allowed(labels),
        "missing_evidence": sorted(set(str(item) for item in missing_evidence)),
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(
    *,
    classification: Mapping[str, object],
    objective_tie_break_analysis: Mapping[str, object],
) -> dict[str, object]:
    primary = classification.get("primary")
    return {
        "next_step": (
            "treat_stay_dominance_as_tie_break_artifact_before_any_shadow_or_readiness_work"
            if primary == "tie_break_artifact_likely"
            else "keep_v113_readiness_blocked_until_objective_tie_break_evidence_is_clear"
        ),
        "summary": _recommendation_summary(
            primary=primary,
            objective_tie_break_analysis=objective_tie_break_analysis,
        ),
        "downstream_shadow_scorer_allowed": False,
        "v113_readiness_rerun_allowed": False,
        "runtime_policy_change_recommended": False,
        "trained_artifact_change_recommended": False,
        "gate_change_recommended": False,
        "observation_field_change_recommended": False,
        "viewer_change_recommended": False,
        "claim_causality": False,
    }


def _recommendation_summary(
    *,
    primary: object,
    objective_tie_break_analysis: Mapping[str, object],
) -> str:
    current = _int(
        objective_tie_break_analysis.get("current_serialized_oracle_stay_count")
    )
    neutral = _mapping(objective_tie_break_analysis.get("tie_neutral_stay_counts"))
    if primary == "tie_break_artifact_likely":
        return (
            f"Serialized oracle stay dominance ({current} branches) collapses under "
            f"tie-neutral alternatives ({dict(neutral)}), so v117 supports a "
            "tie-break artifact interpretation without proving runtime causality."
        )
    if primary == "objective_support_stay_dominance":
        return (
            "Stay remains dominant among branches with a unique serialized "
            "objective-best candidate; keep readiness blocked and do not promote."
        )
    return (
        "The serialized objective/tie-break evidence is inconclusive; keep v113 "
        "readiness blocked."
    )


def _missing_evidence(
    *,
    source_reports: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> list[str]:
    missing: list[str] = []
    report_evidence = _mapping(source_reports.get("archive_report"))
    rows_evidence = _mapping(source_reports.get("archive_rows"))
    if report_evidence.get("loaded") is not True:
        missing.append("archive_report")
    elif report_evidence.get("schema_matches") is not True:
        missing.append("archive_report_schema")
    if rows_evidence.get("loaded") is not True or not rows:
        missing.append("archive_rows")
    return sorted(set(missing))


def _branch_groups(rows: Sequence[Mapping[str, object]]) -> tuple[_BranchGroup, ...]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for index, row in enumerate(rows):
        branch_id = _mapping(row.get("provenance")).get("branch_id")
        if not isinstance(branch_id, str) or not branch_id:
            continue
        item = dict(row)
        item["_archive_input_index"] = index
        grouped[branch_id].append(item)
    return tuple(
        _BranchGroup(
            branch_id=branch_id,
            rows=tuple(sorted(group_rows, key=_row_sort_key)),
        )
        for branch_id, group_rows in sorted(grouped.items())
    )


def _serialized_objective_key(
    row: Mapping[str, object],
) -> tuple[int, float, float, float, float]:
    biology = row.get("biology_homeostasis_labels")
    if not isinstance(biology, Mapping):
        raise FirstRecoveryOracleTieBreakAuditError(
            "biology_homeostasis_labels must be a mapping"
        )
    target_alive = biology.get("target_alive")
    if type(target_alive) is not bool:
        raise FirstRecoveryOracleTieBreakAuditError(
            "biology_homeostasis_labels.target_alive must be bool"
        )
    vitals = row.get("recovery_vitals_deltas")
    if not isinstance(vitals, Mapping):
        raise FirstRecoveryOracleTieBreakAuditError(
            "recovery_vitals_deltas must be a mapping"
        )
    return (
        int(target_alive),
        _strict_objective_number(row, "terminal_alive_delta"),
        _strict_objective_number(row, "birth_delta"),
        _strict_objective_number(vitals, "target_recovery_score_delta"),
        _strict_objective_number(row, "death_reduction_delta"),
    )


def _strict_objective_number(
    payload: Mapping[str, object],
    field: str,
) -> float:
    value = payload.get(field)
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise FirstRecoveryOracleTieBreakAuditError(
            f"{field} must be a finite int or float"
        )
    return float(value)


def _prefer_non_stay_resolution_legal(
    rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    legal = [row for row in rows if row.get("resolution_legal") is True]
    pool = legal or list(rows)
    non_stay = [row for row in pool if row.get("candidate_action") != "stay"]
    return sorted(non_stay or pool, key=_action_name_sort_key)[0]


def _prefer_material_gain_resolution_legal(
    rows: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    legal = [row for row in rows if row.get("resolution_legal") is True]
    pool = legal or list(rows)
    material = [row for row in pool if row.get("material_gain_label") is True]
    return sorted(material or pool, key=_action_name_sort_key)[0]


def _stable_hash_key(branch_id: str, row: Mapping[str, object]) -> str:
    index = _int(row.get("_archive_input_index"), default=-1)
    return hashlib.sha256(f"{branch_id}|{index}".encode("utf-8")).hexdigest()


def _action_name_sort_key(row: Mapping[str, object]) -> tuple[str, str]:
    return (str(row.get("candidate_action")), str(row.get("archive_row_id")))


def _row_sort_key(row: Mapping[str, object]) -> tuple[int, str]:
    return (
        _int(row.get("oracle_rank"), default=10**9),
        str(row.get("candidate_action")),
    )


def _resolution_bucket(row: Mapping[str, object]) -> str:
    return "resolution_legal" if row.get("resolution_legal") is True else (
        "resolution_invalid"
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
