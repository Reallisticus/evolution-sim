from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.mind.branch_utility_risk_audit import (
    _field_summary,
    _utility_comparison,
)
from evolution_sim.mind.broad_branch_residual_oracle_audit import (
    MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.broad_transfer_residual_audit import V98_STRICT_EXCLUDED_SEEDS
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_V100_BROAD_BRANCH_RESIDUAL_CONSTRAINED_AUDIT_SCHEMA_VERSION = (
    "mind_v3_v100_broad_branch_residual_constrained_audit_v1"
)
MIND_V3_V100_BROAD_BRANCH_RESIDUAL_CONSTRAINED_POLICY = (
    "v100_broad_branch_residual_diversity_constrained_assignment_v1"
)
V100_MAX_DOMINANT_ACTION_SHARE = 0.50
V100_MIN_COMPARISON_COUNT = 10
V100_MIN_SOURCE_SEEDS = 8
V100_MIN_SAFE_NON_LOGGED_OVERRIDE_COUNT = 8


class BroadBranchResidualConstrainedAuditError(ValueError):
    pass


def load_json_report(source: str | Path | Mapping[str, object]) -> dict[str, object]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BroadBranchResidualConstrainedAuditError(
            f"failed to read report: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BroadBranchResidualConstrainedAuditError(
            f"report is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise BroadBranchResidualConstrainedAuditError("report must be an object")
    return payload


def build_broad_branch_residual_constrained_audit_report(
    *,
    v99_broad_branch_residual_oracle_report: Mapping[str, object],
) -> dict[str, object]:
    _validate_v99_report(v99_broad_branch_residual_oracle_report)
    candidate_rows = _candidate_rows(v99_broad_branch_residual_oracle_report)
    baseline = _baseline_assignment(candidate_rows)
    constrained = _greedy_constrained_assignment(
        candidate_rows,
        max_action_count=_max_action_count(len(candidate_rows)),
    )
    rule_reports = [
        _assignment_report(
            "v99_unconstrained_target_local_oracle_baseline",
            baseline,
            acceptance_candidate=False,
        ),
        _assignment_report(
            "greedy_diversity_constrained_broad_residual_v1",
            constrained,
            acceptance_candidate=True,
        ),
    ]
    coverage = _coverage_report(
        v99_broad_branch_residual_oracle_report,
        candidate_rows=candidate_rows,
    )
    floors = {
        "source_v99_replay_verified": True,
        "source_v99_heuristic_action_source_count": 0,
        "source_v99_unsupported_candidate_action_count": 0,
        "no_strict_seed_leakage": True,
        "comparison_count": V100_MIN_COMPARISON_COUNT,
        "source_seed_count": V100_MIN_SOURCE_SEEDS,
        "dominant_predicted_action_share_max": V100_MAX_DOMINANT_ACTION_SHARE,
        "target_alive_delta_negative_count": 0,
        "mean_target_local_score_delta_gt": 0.0,
        "mean_terminal_alive_delta_min": 0.0,
        "mean_birth_delta_min": 0.0,
        "safe_non_logged_override_count": V100_MIN_SAFE_NON_LOGGED_OVERRIDE_COUNT,
        "runtime_policy_trained": False,
    }
    acceptance = _acceptance(
        coverage=coverage,
        rule_reports=rule_reports,
        floors=floors,
    )
    accepted = bool(
        acceptance["v100_broad_branch_residual_constrained_diagnostic_accepted"]
    )
    contract = {
        "schema_version": MIND_V3_V100_BROAD_BRANCH_RESIDUAL_CONSTRAINED_AUDIT_SCHEMA_VERSION,
        "policy": MIND_V3_V100_BROAD_BRANCH_RESIDUAL_CONSTRAINED_POLICY,
        "diagnostic_only": True,
        "runtime_policy_trained": False,
        "promotion_run_executed": False,
        "source_schema_version": (
            v99_broad_branch_residual_oracle_report.get("schema_version")
        ),
        "assignment_policy": (
            "greedy_replacement_from_replay_verified_candidate_outcomes_v1"
        ),
        "uses_replay_backed_candidate_outcomes": True,
        "uses_exact_reexecution": False,
        "feature_contract": {
            "uses_private_world_state_as_runtime_input": False,
            "uses_fixture_identity": False,
            "uses_seed_id_as_runtime_feature": False,
            "uses_branch_id_as_runtime_feature": False,
            "uses_logged_action_as_runtime_fallback": False,
            "runtime_policy_artifact_emitted": False,
        },
        "support_floors": floors,
    }
    return {
        "schema_version": MIND_V3_V100_BROAD_BRANCH_RESIDUAL_CONSTRAINED_AUDIT_SCHEMA_VERSION,
        "policy": MIND_V3_V100_BROAD_BRANCH_RESIDUAL_CONSTRAINED_POLICY,
        "contract": contract,
        "provenance": {
            "source_v99_digest": stable_payload_digest(
                {
                    "schema_version": v99_broad_branch_residual_oracle_report.get(
                        "schema_version"
                    ),
                    "aggregate": v99_broad_branch_residual_oracle_report.get(
                        "aggregate"
                    ),
                    "branch_points": v99_broad_branch_residual_oracle_report.get(
                        "branch_points"
                    ),
                    "branch_results": v99_broad_branch_residual_oracle_report.get(
                        "branch_results"
                    ),
                }
            ),
            "contract_digest": stable_payload_digest(contract),
        },
        "coverage": coverage,
        "rule_reports": rule_reports,
        "broad_branch_residual_constrained_support_probe": {
            "policy": MIND_V3_V100_BROAD_BRANCH_RESIDUAL_CONSTRAINED_POLICY,
            "accuracy": 1.0 if accepted else 0.0,
            "support_accuracy_floor": 1.0,
            "materially_supports_v101_residual_distillation": accepted,
            "runtime_policy_status": (
                "v101_distillation_allowed"
                if accepted
                else "rejected_no_runtime_policy"
            ),
            "blocker_count": acceptance["blocker_count"],
            "best_rule": acceptance.get("best_rule_for_diagnostics"),
        },
        "acceptance": acceptance,
        "v100_broad_branch_residual_constrained_diagnostic_accepted": accepted,
        "v101_residual_distillation_allowed": bool(
            acceptance["v101_residual_distillation_allowed"]
        ),
        "blocker_count": int(acceptance["blocker_count"]),
    }


def write_broad_branch_residual_constrained_audit_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _candidate_rows(
    report: Mapping[str, object],
) -> list[dict[str, object]]:
    branch_points = {
        str(point.get("branch_id", "")): point
        for point in _list_of_mappings(report.get("branch_points"))
    }
    rows: list[dict[str, object]] = []
    for result in _list_of_mappings(report.get("branch_results")):
        branch_id = str(result.get("branch_id", ""))
        point = _mapping(branch_points.get(branch_id))
        logged_action = str(result.get("logged_action", ""))
        logged_run = _action_run(_list_of_mappings(result.get("action_runs")), logged_action)
        if logged_run is None:
            continue
        row_context = {
            "branch_id": branch_id,
            "seed": result.get("seed"),
            "logged_action": logged_action,
            "before": point.get("before", {}),
            "target_local_action": None,
        }
        candidates = []
        for run in _list_of_mappings(result.get("action_runs")):
            action = str(run.get("forced_action", ""))
            comparison = _utility_comparison(
                row_context,
                rule=MIND_V3_V100_BROAD_BRANCH_RESIDUAL_CONSTRAINED_POLICY,
                predicted_action=action,
                predicted_run=run,
                logged_run=logged_run,
            )
            candidate = {
                **comparison,
                "action": action,
                "target_local_score": comparison["predicted_target_local_score"],
                "target_local_score_delta": comparison["target_local_score_delta"],
                "terminal_alive_delta": comparison["terminal_alive_delta"],
                "birth_delta": comparison["birth_delta"],
                "target_alive_delta": comparison["target_alive_delta"],
                "forced_action_supported": bool(run.get("forced_action_supported")),
                "forced_action_used": bool(run.get("forced_action_used")),
            }
            candidates.append(candidate)
        rows.append(
            {
                "branch_id": branch_id,
                "seed": int(result.get("seed", 0)),
                "branch_tick": int(result.get("branch_tick", 0)),
                "agent_id": int(result.get("agent_id", 0)),
                "logged_action": logged_action,
                "baseline_action": str(result.get("target_local_oracle_action", "")),
                "candidates": sorted(
                    candidates,
                    key=lambda item: _candidate_sort_key(item),
                    reverse=True,
                ),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            int(row.get("seed", 0)),
            int(row.get("branch_tick", 0)),
            int(row.get("agent_id", 0)),
            str(row.get("branch_id", "")),
        ),
    )


def _baseline_assignment(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    assignment = []
    for row in rows:
        baseline_action = str(row.get("baseline_action", ""))
        candidates = _list_of_mappings(row.get("candidates"))
        selected = _candidate_by_action(candidates, baseline_action) or (
            candidates[0] if candidates else None
        )
        if selected is not None:
            assignment.append(_assignment_item(row, selected))
    return assignment


def _greedy_constrained_assignment(
    rows: Sequence[Mapping[str, object]],
    *,
    max_action_count: int,
) -> list[dict[str, object]]:
    assignment = _baseline_assignment(rows)
    row_by_id = {str(row.get("branch_id", "")): row for row in rows}
    counts = Counter(str(item.get("predicted_action", "")) for item in assignment)
    while counts and max(counts.values()) > max_action_count:
        over_action, _ = max(sorted(counts.items()), key=lambda item: (item[1], item[0]))
        replacement = _best_replacement(
            assignment,
            row_by_id=row_by_id,
            counts=counts,
            over_action=over_action,
            max_action_count=max_action_count,
        )
        if replacement is None:
            break
        index, item = replacement
        old_action = str(assignment[index].get("predicted_action", ""))
        counts[old_action] -= 1
        if counts[old_action] <= 0:
            del counts[old_action]
        new_action = str(item.get("predicted_action", ""))
        counts[new_action] += 1
        assignment[index] = item
    return assignment


def _best_replacement(
    assignment: Sequence[Mapping[str, object]],
    *,
    row_by_id: Mapping[str, Mapping[str, object]],
    counts: Counter[str],
    over_action: str,
    max_action_count: int,
) -> tuple[int, dict[str, object]] | None:
    choices: list[tuple[float, float, str, int, dict[str, object]]] = []
    for index, current in enumerate(assignment):
        if str(current.get("predicted_action", "")) != over_action:
            continue
        row = _mapping(row_by_id.get(str(current.get("branch_id", ""))))
        for candidate in _list_of_mappings(row.get("candidates")):
            action = str(candidate.get("action", ""))
            if action == over_action:
                continue
            if counts[action] >= max_action_count:
                continue
            if not _candidate_is_safe(candidate):
                continue
            item = _assignment_item(row, candidate)
            loss = _float(current.get("target_local_score_delta")) - _float(
                item.get("target_local_score_delta")
            )
            choices.append(
                (
                    loss,
                    -_float(item.get("target_local_score_delta")),
                    str(item.get("branch_id", "")),
                    index,
                    item,
                )
            )
    if not choices:
        return None
    choices.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return choices[0][3], choices[0][4]


def _assignment_report(
    rule: str,
    assignment: Sequence[Mapping[str, object]],
    *,
    acceptance_candidate: bool,
) -> dict[str, object]:
    action_counts = Counter(str(item.get("predicted_action", "")) for item in assignment)
    dominant = _dominant_count_share(action_counts)
    comparisons = [dict(item) for item in assignment]
    safe_overrides = [
        item
        for item in assignment
        if item.get("safe_non_logged_override") is True
    ]
    return {
        "rule": rule,
        "acceptance_candidate": bool(acceptance_candidate),
        "comparison_count": len(assignment),
        "action_counts": dict(sorted(action_counts.items())),
        "dominant_predicted_action": dominant["key"],
        "dominant_predicted_action_count": dominant["count"],
        "dominant_predicted_action_share": dominant["share"],
        "target_alive_delta_negative_count": sum(
            1 for item in assignment if _float(item.get("target_alive_delta")) < 0.0
        ),
        "target_local_score_delta_summary": _field_summary(
            comparisons,
            "target_local_score_delta",
        ),
        "terminal_alive_delta_summary": _field_summary(
            comparisons,
            "terminal_alive_delta",
        ),
        "birth_delta_summary": _field_summary(comparisons, "birth_delta"),
        "safe_non_logged_override_count": len(safe_overrides),
        "safe_non_logged_override_share": _safe_rate(
            len(safe_overrides),
            len(assignment),
        ),
        "assignment": comparisons,
    }


def _coverage_report(
    report: Mapping[str, object],
    *,
    candidate_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    aggregate = _mapping(report.get("aggregate"))
    source_seeds = sorted({int(row.get("seed", 0)) for row in candidate_rows})
    return {
        "source_schema_version": report.get("schema_version"),
        "source_replay_verified": bool(aggregate.get("replay_verified")),
        "source_heuristic_action_source_count": int(
            aggregate.get("heuristic_action_source_count", 0)
        ),
        "source_unsupported_candidate_action_count": int(
            aggregate.get("unsupported_candidate_action_count", 0)
        ),
        "source_seed_count": len(source_seeds),
        "source_seeds": source_seeds,
        "strict_seed_leak_count": len(
            set(source_seeds) & set(V98_STRICT_EXCLUDED_SEEDS)
        ),
        "candidate_row_count": len(candidate_rows),
    }


def _acceptance(
    *,
    coverage: Mapping[str, object],
    rule_reports: Sequence[Mapping[str, object]],
    floors: Mapping[str, object],
) -> dict[str, object]:
    candidate_reports = [
        report
        for report in rule_reports
        if report.get("acceptance_candidate") is True
    ]
    accepted_reports = [
        report
        for report in candidate_reports
        if not _rule_blockers(report, coverage=coverage, floors=floors)
    ]
    best_rule = max(
        candidate_reports,
        key=lambda report: (
            -len(_rule_blockers(report, coverage=coverage, floors=floors)),
            _float(_mapping(report.get("target_local_score_delta_summary")).get("mean")),
            -_float(report.get("dominant_predicted_action_share")),
            str(report.get("rule", "")),
        ),
        default=None,
    )
    blockers = (
        []
        if accepted_reports
        else _rule_blockers(best_rule or {}, coverage=coverage, floors=floors)
    )
    accepted = bool(accepted_reports)
    return {
        "v100_broad_branch_residual_constrained_diagnostic_accepted": accepted,
        "v101_residual_distillation_allowed": accepted,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "accepted_rules": [str(report.get("rule", "")) for report in accepted_reports],
        "best_rule_for_diagnostics": _best_rule_summary(best_rule)
        if best_rule is not None
        else None,
        "recommendation": (
            "Proceed to v101 support-gated residual distillation from constrained "
            "broad branch labels; keep this diagnostic separate from promotion."
            if accepted
            else "Do not distill a broad residual from this constrained diagnostic."
        ),
    }


def _rule_blockers(
    report: Mapping[str, object],
    *,
    coverage: Mapping[str, object],
    floors: Mapping[str, object],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []

    def block(reason: str, field: str, observed: object, required: object, comparator: str) -> None:
        blockers.append(
            {
                "reason": reason,
                "field": field,
                "observed": observed,
                "required": required,
                "comparator": comparator,
            }
        )

    if coverage.get("source_replay_verified") is not True:
        block("source_replay_not_verified", "source_replay_verified", False, True, "eq")
    if _int(coverage.get("source_heuristic_action_source_count")) != 0:
        block(
            "source_heuristic_action_source_count_nonzero",
            "source_heuristic_action_source_count",
            coverage.get("source_heuristic_action_source_count"),
            0,
            "eq",
        )
    if _int(coverage.get("source_unsupported_candidate_action_count")) != 0:
        block(
            "source_unsupported_candidate_action_count_nonzero",
            "source_unsupported_candidate_action_count",
            coverage.get("source_unsupported_candidate_action_count"),
            0,
            "eq",
        )
    if _int(coverage.get("strict_seed_leak_count")) != 0:
        block(
            "strict_seed_leakage",
            "strict_seed_leak_count",
            coverage.get("strict_seed_leak_count"),
            0,
            "eq",
        )
    if _int(report.get("comparison_count")) < _int(floors.get("comparison_count")):
        block(
            "insufficient_comparison_count",
            "comparison_count",
            report.get("comparison_count"),
            floors.get("comparison_count"),
            "ge",
        )
    if _int(coverage.get("source_seed_count")) < _int(floors.get("source_seed_count")):
        block(
            "insufficient_source_seed_count",
            "source_seed_count",
            coverage.get("source_seed_count"),
            floors.get("source_seed_count"),
            "ge",
        )
    if _float(report.get("dominant_predicted_action_share")) > _float(
        floors.get("dominant_predicted_action_share_max")
    ):
        block(
            "dominant_predicted_action_share_above_cap",
            "dominant_predicted_action_share",
            report.get("dominant_predicted_action_share"),
            floors.get("dominant_predicted_action_share_max"),
            "le",
        )
    if _int(report.get("target_alive_delta_negative_count")) != 0:
        block(
            "target_alive_delta_negative",
            "target_alive_delta_negative_count",
            report.get("target_alive_delta_negative_count"),
            0,
            "eq",
        )
    target = _mapping(report.get("target_local_score_delta_summary"))
    if _float(target.get("mean")) <= 0.0:
        block(
            "mean_target_local_score_delta_not_positive",
            "target_local_score_delta_summary.mean",
            target.get("mean"),
            floors.get("mean_target_local_score_delta_gt"),
            "gt",
        )
    terminal = _mapping(report.get("terminal_alive_delta_summary"))
    if _float(terminal.get("mean")) < 0.0:
        block(
            "mean_terminal_alive_delta_negative",
            "terminal_alive_delta_summary.mean",
            terminal.get("mean"),
            floors.get("mean_terminal_alive_delta_min"),
            "ge",
        )
    births = _mapping(report.get("birth_delta_summary"))
    if _float(births.get("mean")) < 0.0:
        block(
            "mean_birth_delta_negative",
            "birth_delta_summary.mean",
            births.get("mean"),
            floors.get("mean_birth_delta_min"),
            "ge",
        )
    if _int(report.get("safe_non_logged_override_count")) < _int(
        floors.get("safe_non_logged_override_count")
    ):
        block(
            "insufficient_safe_non_logged_override_count",
            "safe_non_logged_override_count",
            report.get("safe_non_logged_override_count"),
            floors.get("safe_non_logged_override_count"),
            "ge",
        )
    return blockers


def _assignment_item(
    row: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    action = str(candidate.get("action", ""))
    logged = str(row.get("logged_action", ""))
    return {
        "branch_id": row.get("branch_id"),
        "seed": row.get("seed"),
        "branch_tick": row.get("branch_tick"),
        "agent_id": row.get("agent_id"),
        "logged_action": logged,
        "predicted_action": action,
        "target_local_score_delta": candidate.get("target_local_score_delta"),
        "terminal_alive_delta": candidate.get("terminal_alive_delta"),
        "birth_delta": candidate.get("birth_delta"),
        "target_alive_delta": candidate.get("target_alive_delta"),
        "safe_non_logged_override": bool(
            action != logged
            and _float(candidate.get("target_local_score_delta")) > 0.0
            and _float(candidate.get("target_alive_delta")) >= 0.0
            and _float(candidate.get("terminal_alive_delta")) >= 0.0
            and _float(candidate.get("birth_delta")) >= 0.0
        ),
    }


def _candidate_sort_key(candidate: Mapping[str, object]) -> tuple[float, float, float, str]:
    return (
        _float(candidate.get("target_local_score_delta")),
        _float(candidate.get("terminal_alive_delta")),
        _float(candidate.get("birth_delta")),
        str(candidate.get("action", "")),
    )


def _candidate_is_safe(candidate: Mapping[str, object]) -> bool:
    return (
        candidate.get("forced_action_supported") is True
        and candidate.get("forced_action_used") is True
        and _float(candidate.get("target_alive_delta")) >= 0.0
        and _float(candidate.get("terminal_alive_delta")) >= 0.0
        and _float(candidate.get("birth_delta")) >= 0.0
    )


def _candidate_by_action(
    candidates: Sequence[Mapping[str, object]],
    action: str,
) -> Mapping[str, object] | None:
    return next(
        (candidate for candidate in candidates if str(candidate.get("action", "")) == action),
        None,
    )


def _action_run(
    runs: Sequence[Mapping[str, object]],
    action: str,
) -> Mapping[str, object] | None:
    return next((run for run in runs if str(run.get("forced_action", "")) == action), None)


def _best_rule_summary(report: Mapping[str, object]) -> dict[str, object]:
    target = _mapping(report.get("target_local_score_delta_summary"))
    terminal = _mapping(report.get("terminal_alive_delta_summary"))
    births = _mapping(report.get("birth_delta_summary"))
    return {
        "rule": report.get("rule"),
        "comparison_count": report.get("comparison_count"),
        "dominant_predicted_action": report.get("dominant_predicted_action"),
        "dominant_predicted_action_share": report.get(
            "dominant_predicted_action_share"
        ),
        "target_alive_delta_negative_count": report.get(
            "target_alive_delta_negative_count"
        ),
        "mean_target_local_score_delta": target.get("mean"),
        "mean_terminal_alive_delta": terminal.get("mean"),
        "mean_birth_delta": births.get("mean"),
        "safe_non_logged_override_count": report.get("safe_non_logged_override_count"),
        "safe_non_logged_override_share": report.get("safe_non_logged_override_share"),
        "blocker_count": len(
            _rule_blockers(
                report,
                coverage={
                    "source_replay_verified": True,
                    "source_heuristic_action_source_count": 0,
                    "source_unsupported_candidate_action_count": 0,
                    "strict_seed_leak_count": 0,
                    "source_seed_count": V100_MIN_SOURCE_SEEDS,
                },
                floors={
                    "comparison_count": V100_MIN_COMPARISON_COUNT,
                    "source_seed_count": V100_MIN_SOURCE_SEEDS,
                    "dominant_predicted_action_share_max": V100_MAX_DOMINANT_ACTION_SHARE,
                    "safe_non_logged_override_count": V100_MIN_SAFE_NON_LOGGED_OVERRIDE_COUNT,
                    "mean_target_local_score_delta_gt": 0.0,
                    "mean_terminal_alive_delta_min": 0.0,
                    "mean_birth_delta_min": 0.0,
                },
            )
        ),
    }


def _validate_v99_report(report: Mapping[str, object]) -> None:
    if (
        report.get("schema_version")
        != MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_AUDIT_SCHEMA_VERSION
    ):
        raise BroadBranchResidualConstrainedAuditError(
            "source report has unsupported v99 schema_version"
        )


def _max_action_count(count: int) -> int:
    return int(math.floor(float(count) * V100_MAX_DOMINANT_ACTION_SHARE))


def _dominant_count_share(counts: Counter[str]) -> dict[str, object]:
    total = sum(int(count) for count in counts.values())
    if total <= 0:
        return {"key": None, "count": 0, "share": 0.0}
    key, count = max(sorted(counts.items()), key=lambda item: (item[1], item[0]))
    return {"key": key, "count": int(count), "share": _round(count / float(total))}


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list_of_mappings(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    return 0


def _float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    denom = float(denominator)
    if denom <= 0.0:
        return 0.0
    return _round(float(numerator) / denom)


def _round(value: float) -> float:
    return round(float(value), 6)


def _open_output(path: Path) -> TextIO:
    return path.open("w", encoding="utf-8")
