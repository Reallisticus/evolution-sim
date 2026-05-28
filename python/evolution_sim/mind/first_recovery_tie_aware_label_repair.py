from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evolution_sim.mind.first_recovery_archive_blocker_diagnostic import (
    DEFAULT_ARCHIVE_REPORT_PATH,
    DEFAULT_ARCHIVE_ROWS_PATH,
)
from evolution_sim.mind.first_recovery_branch_archive import (
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
)
from evolution_sim.mind.first_recovery_oracle_tie_break_audit import (
    build_first_recovery_oracle_tie_break_audit,
    _serialized_objective_key,
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

MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_SCHEMA_VERSION = (
    "mind_v3_first_recovery_tie_aware_label_repair_v1"
)
MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_POLICY = (
    "diagnostics_only_first_recovery_v118_tie_aware_label_repair_v1"
)

DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v118-first-recovery-tie-aware-label-repair.json"
)

DOMINANT_ACTION_SHARE_MAX = 0.50
MAX_EXAMPLES = 16

POLICY_NAMES: tuple[str, ...] = (
    "min_change_resolution_legal",
    "prefer_material_gain_resolution_legal",
    "prefer_non_stay_resolution_legal",
    "action_balance_resolution_legal",
)

ALLOWED_CLASSIFICATIONS: tuple[str, ...] = (
    "diagnostics_only_no_runtime_promotion",
    "missing_evidence_inconclusive",
    "source_archive_verification_failed",
    "objective_input_field_integrity",
    "tie_aware_repair_clears_action_collapse",
    "tie_aware_repair_still_action_collapsed",
    "tie_aware_repair_inconclusive",
    "readiness_rerun_blocked",
)


@dataclass(frozen=True, slots=True)
class FirstRecoveryTieAwareLabelRepairBuild:
    report: dict[str, object]


@dataclass(frozen=True, slots=True)
class _BranchGroup:
    branch_id: str
    rows: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _BranchRepairContext:
    branch_id: str
    rows: tuple[dict[str, object], ...]
    current: dict[str, object]
    rank1_key: tuple[int, float, float, float, float]
    equivalent: tuple[dict[str, object], ...]
    legal_equivalent: tuple[dict[str, object], ...]
    unique_objective_best: bool


def build_first_recovery_tie_aware_label_repair(
    *,
    archive_report: Mapping[str, object] | None = None,
    archive_report_path: str | Path | None = DEFAULT_ARCHIVE_REPORT_PATH,
    archive_rows: Sequence[Mapping[str, object]] | None = None,
    archive_rows_path: str | Path | None = DEFAULT_ARCHIVE_ROWS_PATH,
    strict_source_verification: bool | None = None,
) -> FirstRecoveryTieAwareLabelRepairBuild:
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
    v117 = build_first_recovery_oracle_tie_break_audit(
        archive_report=archive_payload,
        archive_report_path=archive_report_path,
        archive_rows=rows,
        archive_rows_path=archive_rows_path,
        strict_source_verification=strict,
    ).report
    source_reports = _source_reports(
        archive_report=archive_payload,
        archive_evidence=archive_evidence,
        rows=rows,
        rows_evidence=rows_evidence,
        strict_source_verification=strict,
    )
    source_verification = dict(_mapping(v117.get("source_archive_verification")))
    objective_input_integrity = dict(_mapping(v117.get("objective_input_integrity")))
    missing_evidence = _missing_evidence(source_reports=source_reports, rows=rows)
    repair = _tie_aware_label_repair_analysis(
        rows,
        source_verification=source_verification,
        objective_input_integrity=objective_input_integrity,
        missing_evidence=missing_evidence,
        strict_source_verification=strict,
    )
    classification = _classification(
        missing_evidence=missing_evidence,
        source_verification=source_verification,
        objective_input_integrity=objective_input_integrity,
        repair=repair,
        strict_source_verification=strict,
    )
    recommendation = _recommendation(
        classification=classification,
        repair=repair,
    )
    report = {
        "schema_version": MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_SCHEMA_VERSION,
        "audit_policy": MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_POLICY,
        "contract": _contract(),
        "source_reports": source_reports,
        "source_archive_verification": source_verification,
        "objective_input_integrity": objective_input_integrity,
        "v117_reference": _v117_reference(v117),
        "tie_aware_label_repair": repair,
        "classification": classification,
        "recommendation": recommendation,
        "non_promoted": True,
    }
    return FirstRecoveryTieAwareLabelRepairBuild(report=report)


def write_first_recovery_tie_aware_label_repair_report(
    build: FirstRecoveryTieAwareLabelRepairBuild,
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
        "schema_version": MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_SCHEMA_VERSION,
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
        "shadow_scorer_implemented": False,
        "objective_values_changed": False,
        "claim_causality": False,
        "v113_readiness_rerun_allowed": False,
        "downstream_shadow_scorer_allowed": False,
        "diagnostic_policy": (
            "read_only_serialized_v115_branch_rows_tie_aware_label_repair_proposal"
        ),
        "contract_digest": stable_payload_digest(
            {
                "schema_version": (
                    MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_SCHEMA_VERSION
                ),
                "audit_policy": MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_POLICY,
            }
        ),
    }


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


def _v117_reference(v117: Mapping[str, object]) -> dict[str, object]:
    analysis = _mapping(v117.get("objective_tie_break_analysis"))
    classification = _mapping(v117.get("classification"))
    recommendation = _mapping(v117.get("recommendation"))
    return {
        "schema_version": v117.get("schema_version"),
        "classification_primary": classification.get("primary"),
        "current_serialized_oracle_action_counts": analysis.get(
            "current_serialized_oracle_action_counts"
        ),
        "current_serialized_oracle_stay_count": analysis.get(
            "current_serialized_oracle_stay_count"
        ),
        "unique_objective_best_stay_count": analysis.get(
            "unique_objective_best_stay_count"
        ),
        "tie_neutral_stay_counts": analysis.get("tie_neutral_stay_counts"),
        "claim_causality": recommendation.get("claim_causality"),
        "v113_readiness_rerun_allowed": recommendation.get(
            "v113_readiness_rerun_allowed"
        ),
        "downstream_shadow_scorer_allowed": recommendation.get(
            "downstream_shadow_scorer_allowed"
        ),
    }


def _tie_aware_label_repair_analysis(
    archive_rows: Sequence[Mapping[str, object]],
    *,
    source_verification: Mapping[str, object],
    objective_input_integrity: Mapping[str, object],
    missing_evidence: Sequence[str],
    strict_source_verification: bool,
) -> dict[str, object]:
    blocked_reasons: list[str] = []
    if missing_evidence:
        blocked_reasons.append("missing_evidence")
    if objective_input_integrity.get("passed") is not True:
        blocked_reasons.append("objective_input_field_integrity")
    if (
        strict_source_verification
        and source_verification.get("verification_passed") is not True
    ):
        blocked_reasons.append("source_archive_verification_failed")
    if blocked_reasons:
        return _blocked_repair_analysis(archive_rows, blocked_reasons=blocked_reasons)

    contexts, integrity_failures, integrity_examples = _branch_repair_contexts(
        archive_rows
    )
    if integrity_failures:
        return _blocked_repair_analysis(
            archive_rows,
            blocked_reasons=integrity_failures,
            branch_integrity_examples=integrity_examples,
        )

    policy_reports = {
        name: _policy_report(name, contexts) for name in POLICY_NAMES
    }
    valid_policy_names = [
        name for name, report in policy_reports.items()
        if report.get("policy_valid") is True
    ]
    clearing_policy_names = [
        name for name in valid_policy_names
        if float(_mapping(policy_reports[name]).get("dominant_action_share", 1.0))
        <= DOMINANT_ACTION_SHARE_MAX
    ]
    best_valid_policy = _best_policy_name(policy_reports, valid_policy_names)
    best_clearing_policy = _best_policy_name(policy_reports, clearing_policy_names)
    current_counts = Counter(
        str(context.current.get("candidate_action")) for context in contexts
    )
    return {
        "analysis_blocked": False,
        "repair_policy": {
            "objective_values_changed": False,
            "repair_scope": "only_candidates_tied_on_v117_serialized_objective_key",
            "executable_candidate_filter": "resolution_legal_true",
            "unique_objective_best_policy": "preserve_current_rank1_label",
        },
        "branch_count": len(contexts),
        "current_serialized_oracle_action_counts": _counter_to_dict(current_counts),
        "current_dominant_action": _dominant_action(current_counts)[0],
        "current_dominant_action_share": _dominant_action_share(current_counts),
        "unique_objective_best_branch_count": sum(
            1 for context in contexts if context.unique_objective_best
        ),
        "tied_objective_best_branch_count": sum(
            1 for context in contexts if not context.unique_objective_best
        ),
        "policies": policy_reports,
        "valid_policy_names": valid_policy_names,
        "clearing_policy_names": clearing_policy_names,
        "best_valid_policy": best_valid_policy,
        "best_clearing_policy": best_clearing_policy,
    }


def _blocked_repair_analysis(
    archive_rows: Sequence[Mapping[str, object]],
    *,
    blocked_reasons: Sequence[str],
    branch_integrity_examples: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    return {
        "analysis_blocked": True,
        "blocked_reasons": sorted(set(str(reason) for reason in blocked_reasons)),
        "branch_count": len(_branch_groups(archive_rows)),
        "repair_policy": {
            "objective_values_changed": False,
            "repair_scope": "only_candidates_tied_on_v117_serialized_objective_key",
            "executable_candidate_filter": "resolution_legal_true",
            "unique_objective_best_policy": "preserve_current_rank1_label",
        },
        "branch_integrity_examples": _list(branch_integrity_examples),
        "policies": {},
        "valid_policy_names": [],
        "clearing_policy_names": [],
        "best_valid_policy": None,
        "best_clearing_policy": None,
    }


def _branch_repair_contexts(
    archive_rows: Sequence[Mapping[str, object]],
) -> tuple[list[_BranchRepairContext], list[str], list[dict[str, object]]]:
    contexts: list[_BranchRepairContext] = []
    failures: list[str] = []
    examples: list[dict[str, object]] = []
    for group in _branch_groups(archive_rows):
        rank1_rows = [
            row for row in group.rows if _int(row.get("oracle_rank"), default=-1) == 1
        ]
        if len(rank1_rows) != 1:
            failures.append("rank1_row_integrity")
            if len(examples) < MAX_EXAMPLES:
                examples.append(
                    {
                        "branch_id": group.branch_id,
                        "rank1_row_count": len(rank1_rows),
                    }
                )
            continue
        current = rank1_rows[0]
        rank1_key = _serialized_objective_key(current)
        equivalent = tuple(
            row for row in group.rows
            if _serialized_objective_key(row) == rank1_key
        )
        legal_equivalent = tuple(
            row for row in equivalent if row.get("resolution_legal") is True
        )
        contexts.append(
            _BranchRepairContext(
                branch_id=group.branch_id,
                rows=group.rows,
                current=current,
                rank1_key=rank1_key,
                equivalent=equivalent,
                legal_equivalent=legal_equivalent,
                unique_objective_best=len(equivalent) == 1,
            )
        )
    return contexts, sorted(set(failures)), examples


def _policy_report(
    policy_name: str,
    contexts: Sequence[_BranchRepairContext],
) -> dict[str, object]:
    if policy_name == "action_balance_resolution_legal":
        selections = _action_balance_selections(contexts)
    else:
        selections = {
            context.branch_id: _select_policy_row(policy_name, context)
            for context in contexts
        }
    return _summarize_policy_report(policy_name, contexts, selections)


def _select_policy_row(
    policy_name: str,
    context: _BranchRepairContext,
) -> dict[str, object] | None:
    if context.unique_objective_best:
        return context.current if context.current.get("resolution_legal") is True else None
    legal = context.legal_equivalent
    if not legal:
        return None
    current_action = str(context.current.get("candidate_action"))
    if (
        policy_name == "min_change_resolution_legal"
        and context.current.get("resolution_legal") is True
    ):
        return context.current
    if policy_name == "prefer_material_gain_resolution_legal":
        material = [row for row in legal if row.get("material_gain_label") is True]
        return sorted(material or list(legal), key=_row_choice_key)[0]
    if policy_name == "prefer_non_stay_resolution_legal":
        non_stay = [row for row in legal if row.get("candidate_action") != "stay"]
        return sorted(non_stay or list(legal), key=_row_choice_key)[0]
    if policy_name == "min_change_resolution_legal":
        return sorted(legal, key=_row_choice_key)[0]
    raise ValueError(f"unknown repair policy: {policy_name}")


def _action_balance_selections(
    contexts: Sequence[_BranchRepairContext],
) -> dict[str, dict[str, object] | None]:
    counts: Counter[str] = Counter()
    selections: dict[str, dict[str, object] | None] = {}
    unique = [context for context in contexts if context.unique_objective_best]
    tied = [context for context in contexts if not context.unique_objective_best]
    for context in sorted(unique, key=lambda item: item.branch_id):
        row = (
            context.current
            if context.current.get("resolution_legal") is True
            else None
        )
        selections[context.branch_id] = row
        if row is not None:
            counts[str(row.get("candidate_action"))] += 1
    for context in sorted(tied, key=lambda item: item.branch_id):
        legal = context.legal_equivalent
        if not legal:
            selections[context.branch_id] = None
            continue
        row = sorted(
            legal,
            key=lambda candidate: (
                counts[str(candidate.get("candidate_action"))],
                str(candidate.get("candidate_action")),
                str(candidate.get("archive_row_id")),
            ),
        )[0]
        selections[context.branch_id] = row
        counts[str(row.get("candidate_action"))] += 1
    return selections


def _summarize_policy_report(
    policy_name: str,
    contexts: Sequence[_BranchRepairContext],
    selections: Mapping[str, Mapping[str, object] | None],
) -> dict[str, object]:
    repaired_counts: Counter[str] = Counter()
    changed_current_counts: Counter[str] = Counter()
    changed_repaired_counts: Counter[str] = Counter()
    changed_transition_counts: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    changed_count = 0
    unique_changed_count = 0
    tied_changed_count = 0
    resolution_invalid_selected_count = 0
    objective_equivalence_violation_count = 0
    unselected_branch_count = 0
    for context in contexts:
        selected = selections.get(context.branch_id)
        current_action = str(context.current.get("candidate_action"))
        if selected is None:
            unselected_branch_count += 1
            continue
        repaired_action = str(selected.get("candidate_action"))
        repaired_counts[repaired_action] += 1
        changed = repaired_action != current_action
        if changed:
            changed_count += 1
            changed_current_counts[current_action] += 1
            changed_repaired_counts[repaired_action] += 1
            changed_transition_counts[f"{current_action}->{repaired_action}"] += 1
            if context.unique_objective_best:
                unique_changed_count += 1
            else:
                tied_changed_count += 1
            if len(examples) < MAX_EXAMPLES:
                examples.append(_changed_example(context, selected))
        if selected.get("resolution_legal") is not True:
            resolution_invalid_selected_count += 1
        if _serialized_objective_key(selected) != context.rank1_key:
            objective_equivalence_violation_count += 1
    dominant_action, dominant_count = _dominant_action(repaired_counts)
    branch_count = len(contexts)
    policy_valid = (
        unique_changed_count == 0
        and resolution_invalid_selected_count == 0
        and objective_equivalence_violation_count == 0
        and unselected_branch_count == 0
    )
    return {
        "policy_name": policy_name,
        "policy_valid": policy_valid,
        "repaired_action_counts": _counter_to_dict(repaired_counts),
        "dominant_action": dominant_action,
        "dominant_action_count": dominant_count,
        "dominant_action_share": _dominant_action_share(repaired_counts),
        "changed_branch_count": changed_count,
        "changed_branch_share": _share(changed_count, branch_count),
        "changed_branches_by_current_action": _counter_to_dict(
            changed_current_counts
        ),
        "changed_branches_by_repaired_action": _counter_to_dict(
            changed_repaired_counts
        ),
        "changed_branch_transition_counts": _counter_to_dict(
            changed_transition_counts
        ),
        "unique_best_branches_changed_count": unique_changed_count,
        "tied_branches_changed_count": tied_changed_count,
        "resolution_invalid_selected_count": resolution_invalid_selected_count,
        "objective_equivalence_violation_count": (
            objective_equivalence_violation_count
        ),
        "unselected_branch_count": unselected_branch_count,
        "objective_values_changed": False,
        "examples": examples,
    }


def _changed_example(
    context: _BranchRepairContext,
    selected: Mapping[str, object],
) -> dict[str, object]:
    return {
        "branch_id": context.branch_id,
        "current_archive_row_id": context.current.get("archive_row_id"),
        "repaired_archive_row_id": selected.get("archive_row_id"),
        "current_action": context.current.get("candidate_action"),
        "repaired_action": selected.get("candidate_action"),
        "rank1_tie_size": len(context.equivalent),
        "legal_tied_actions": sorted(
            str(row.get("candidate_action")) for row in context.legal_equivalent
        ),
        "repaired_material_gain_label": selected.get("material_gain_label"),
        "serialized_objective_key": list(context.rank1_key),
    }


def _best_policy_name(
    policy_reports: Mapping[str, Mapping[str, object]],
    names: Sequence[str],
) -> str | None:
    if not names:
        return None
    return sorted(
        names,
        key=lambda name: (
            float(_mapping(policy_reports[name]).get("dominant_action_share", 1.0)),
            _int(_mapping(policy_reports[name]).get("changed_branch_count")),
            name,
        ),
    )[0]


def _classification(
    *,
    missing_evidence: Sequence[str],
    source_verification: Mapping[str, object],
    objective_input_integrity: Mapping[str, object],
    repair: Mapping[str, object],
    strict_source_verification: bool,
) -> dict[str, object]:
    labels = ["diagnostics_only_no_runtime_promotion", "readiness_rerun_blocked"]
    if missing_evidence:
        primary = "tie_aware_repair_inconclusive"
        labels[:0] = ["missing_evidence_inconclusive", primary]
    elif objective_input_integrity.get("passed") is not True:
        primary = "tie_aware_repair_inconclusive"
        labels[:0] = ["objective_input_field_integrity", primary]
    elif (
        strict_source_verification
        and source_verification.get("verification_passed") is not True
    ):
        primary = "tie_aware_repair_inconclusive"
        labels[:0] = ["source_archive_verification_failed", primary]
    elif repair.get("analysis_blocked") is True:
        primary = "tie_aware_repair_inconclusive"
        labels.insert(0, primary)
    elif _list(repair.get("clearing_policy_names")):
        primary = "tie_aware_repair_clears_action_collapse"
        labels.insert(0, primary)
    elif _list(repair.get("valid_policy_names")):
        primary = "tie_aware_repair_still_action_collapsed"
        labels.insert(0, primary)
    else:
        primary = "tie_aware_repair_inconclusive"
        labels.insert(0, primary)
    return {
        "primary": primary,
        "labels": _dedupe_allowed(labels),
        "allowed_classifications": list(ALLOWED_CLASSIFICATIONS),
    }


def _recommendation(
    *,
    classification: Mapping[str, object],
    repair: Mapping[str, object],
) -> dict[str, object]:
    primary = classification.get("primary")
    return {
        "next_step": (
            "record_tie_aware_label_repair_as_diagnostic_only_before_any_shadow_or_readiness_work"
            if primary == "tie_aware_repair_clears_action_collapse"
            else "keep_v113_readiness_blocked_until_label_repair_evidence_is_actionable"
        ),
        "summary": _recommendation_summary(primary=primary, repair=repair),
        "best_valid_policy": repair.get("best_valid_policy"),
        "best_clearing_policy": repair.get("best_clearing_policy"),
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
    repair: Mapping[str, object],
) -> str:
    if primary == "tie_aware_repair_clears_action_collapse":
        best_name = repair.get("best_clearing_policy") or repair.get(
            "best_valid_policy"
        )
        best = _mapping(_mapping(repair.get("policies")).get(str(best_name)))
        return (
            f"Tie-aware legal label repair policy {best_name} clears action "
            f"collapse with dominant action {best.get('dominant_action')} at "
            f"{best.get('dominant_action_share')}; this is diagnostics-only and "
            "does not authorize readiness or shadow scoring."
        )
    if primary == "tie_aware_repair_still_action_collapsed":
        return (
            "All valid objective-preserving legal repair policies remain action "
            "collapsed; keep v113 readiness blocked."
        )
    return (
        "Tie-aware label repair evidence is inconclusive; keep v113 readiness "
        "and downstream shadow scoring blocked."
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


def _row_sort_key(row: Mapping[str, object]) -> tuple[int, str]:
    return (
        _int(row.get("oracle_rank"), default=10**9),
        str(row.get("candidate_action")),
    )


def _row_choice_key(row: Mapping[str, object]) -> tuple[str, str]:
    return (str(row.get("candidate_action")), str(row.get("archive_row_id")))


def _dominant_action(counts: Counter[str]) -> tuple[str | None, int]:
    if not counts:
        return None, 0
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]


def _dominant_action_share(counts: Counter[str]) -> float:
    total = sum(counts.values())
    _action, count = _dominant_action(counts)
    return _share(count, total)


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
