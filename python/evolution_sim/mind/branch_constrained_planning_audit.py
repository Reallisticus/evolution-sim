from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.branch_action_oracle_labels import (
    MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION,
)
from evolution_sim.mind.branch_mode_objective_audit import (
    _action_option_mode,
    _list_of_mappings,
    _mapping,
    _safe_rate,
    _target_terminal_projection,
)
from evolution_sim.mind.branch_sequence_continuation_scorer import (
    MIND_V3_BRANCH_SEQUENCE_CONTINUATION_SCORER_SCHEMA_VERSION,
    V94_SEED_41_TICK_113_BRANCH_ID,
    V94_SEED_41_TICK_114_BRANCH_ID,
    _select_action_for_rule as _v94_select_action_for_rule,
)
from evolution_sim.mind.branch_utility_risk_audit import (
    _candidate_actions,
    _field_summary,
    _float,
    _int,
    _utility_comparison,
    _utility_rows,
)
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_BRANCH_CONSTRAINED_PLANNING_AUDIT_SCHEMA_VERSION = (
    "mind_v3_branch_constrained_planning_audit_v1"
)
V95_STRICT_EVAL_SEEDS: tuple[int, ...] = (13, 19, 29, 37, 41, 43)
V95_MIN_STRICT_COMPARISON_COUNT = 48
V95_MAX_DOMINANT_PREDICTED_ACTION_SHARE = 0.50
V95_MAX_TARGET_ALIVE_NEGATIVE_COUNT = 0
V95_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA = 0.0
V95_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA = 0.0
V95_MIN_MEAN_TERMINAL_ALIVE_DELTA = 0.0
V95_MIN_MEAN_BIRTH_DELTA = 0.0
V95_BASELINE_RULE = "v94_sequence_prefix_nearest_neighbor_k5_baseline"
V95_BEAM_WIDTH = 4096

_PLANNER_RULES = (
    "greedy_constrained_planner_v1",
    "beam_global_constrained_planner_v1",
    "per_seed_constrained_planner_v1",
    "diversity_regularized_planner_v1",
)


class BranchConstrainedPlanningAuditError(ValueError):
    pass


def build_branch_constrained_planning_audit_report(
    *,
    strict_branch_action_oracle_labels: Mapping[str, object],
    branch_sequence_continuation_scorer_report: Mapping[str, object],
) -> dict[str, object]:
    _validate_inputs(
        strict_branch_action_oracle_labels=strict_branch_action_oracle_labels,
        branch_sequence_continuation_scorer_report=(
            branch_sequence_continuation_scorer_report
        ),
    )
    labels = _list_of_mappings(
        strict_branch_action_oracle_labels.get("labels"),
        "strict.labels",
    )
    rows = _utility_rows(labels)
    candidate_predictions = _v94_candidate_predictions(
        branch_sequence_continuation_scorer_report
    )
    candidate_outcomes = _candidate_outcomes(rows)
    baseline_actions = _v94_baseline_actions(
        rows=rows,
        candidate_predictions=candidate_predictions,
    )
    baseline_assignment = _assignment_from_actions(
        rows=rows,
        candidate_outcomes=candidate_outcomes,
        actions=baseline_actions,
    )
    max_action_count = int(
        math.floor(
            float(len(rows)) * V95_MAX_DOMINANT_PREDICTED_ACTION_SHARE
        )
    )
    planner_assignments = {
        V95_BASELINE_RULE: baseline_assignment,
        "greedy_constrained_planner_v1": _greedy_constrained_assignment(
            baseline_assignment,
            candidate_outcomes=candidate_outcomes,
            max_action_count=max_action_count,
        ),
        "beam_global_constrained_planner_v1": _beam_global_assignment(
            rows,
            candidate_outcomes=candidate_outcomes,
            max_action_count=max_action_count,
            diversity_penalty=0.0,
        ),
        "per_seed_constrained_planner_v1": _per_seed_constrained_assignment(
            baseline_assignment,
            candidate_outcomes=candidate_outcomes,
        ),
        "diversity_regularized_planner_v1": _beam_global_assignment(
            rows,
            candidate_outcomes=candidate_outcomes,
            max_action_count=max_action_count,
            diversity_penalty=2.0,
        ),
    }
    planner_reports = [
        _planner_report(
            rule,
            assignment,
            baseline_assignment=baseline_assignment,
            max_action_count=max_action_count,
        )
        for rule, assignment in planner_assignments.items()
    ]
    coverage = _coverage_report(
        strict_labels=strict_branch_action_oracle_labels,
        rows=rows,
        candidate_outcomes=candidate_outcomes,
        candidate_predictions=candidate_predictions,
    )
    acceptance = _acceptance(coverage=coverage, planner_reports=planner_reports)
    contract = {
        "schema_version": MIND_V3_BRANCH_CONSTRAINED_PLANNING_AUDIT_SCHEMA_VERSION,
        "strict_label_schema_version": strict_branch_action_oracle_labels.get(
            "schema_version"
        ),
        "source_sequence_scorer_schema_version": (
            branch_sequence_continuation_scorer_report.get("schema_version")
        ),
        "runtime_policy_trained": False,
        "runtime_ready": False,
        "diagnostic_only": True,
        "planner_policy": "v95_simulator_in_loop_constrained_branch_planning_v1",
        "strict_eval_seeds": list(V95_STRICT_EVAL_SEEDS),
        "uses_replay_backed_candidate_outcomes": True,
        "uses_exact_reexecution": False,
        "feature_contract": {
            "uses_private_world_state_as_runtime_input": False,
            "uses_fixture_identity_as_runtime_input": False,
            "uses_seed_id_as_runtime_feature": False,
            "uses_branch_id_as_runtime_feature": False,
            "uses_logged_action_as_runtime_fallback": False,
            "runtime_policy_artifact_emitted": False,
        },
        "objective_order": [
            "target alive first",
            "target-local score",
            "terminal alive",
            "births",
            "action diversity constraint",
        ],
        "planner_rules": [V95_BASELINE_RULE, *_PLANNER_RULES],
        "acceptance_candidate_rules": list(_PLANNER_RULES),
        "baseline_rules_not_eligible_for_acceptance": [V95_BASELINE_RULE],
        "support_floors": {
            "replay_verified": True,
            "heuristic_action_source_count": 0,
            "strict_eval_comparison_count": V95_MIN_STRICT_COMPARISON_COUNT,
            "dominant_predicted_action_share_max": (
                V95_MAX_DOMINANT_PREDICTED_ACTION_SHARE
            ),
            "target_alive_delta_negative_count": (
                V95_MAX_TARGET_ALIVE_NEGATIVE_COUNT
            ),
            "global_mean_target_local_score_delta_gt": (
                V95_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA
            ),
            "per_seed_target_local_score_delta_min": (
                V95_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA
            ),
            "mean_terminal_alive_delta_min": V95_MIN_MEAN_TERMINAL_ALIVE_DELTA,
            "mean_birth_delta_min": V95_MIN_MEAN_BIRTH_DELTA,
            "seed_41_tick_113_catastrophe_avoided": True,
            "seed_41_tick_114_catastrophe_avoided": True,
            "accepted_result_runtime_ready": False,
        },
    }
    accepted = bool(acceptance.get("v95_constrained_planning_diagnostic_accepted"))
    return {
        "schema_version": MIND_V3_BRANCH_CONSTRAINED_PLANNING_AUDIT_SCHEMA_VERSION,
        "contract": contract,
        "provenance": {
            "strict_label_digest": stable_payload_digest(
                {
                    "schema_version": strict_branch_action_oracle_labels.get(
                        "schema_version"
                    ),
                    "aggregate": strict_branch_action_oracle_labels.get("aggregate"),
                    "acceptance": strict_branch_action_oracle_labels.get(
                        "acceptance"
                    ),
                    "labels": labels,
                }
            ),
            "sequence_scorer_digest": stable_payload_digest(
                {
                    "schema_version": branch_sequence_continuation_scorer_report.get(
                        "schema_version"
                    ),
                    "coverage": branch_sequence_continuation_scorer_report.get(
                        "coverage"
                    ),
                    "acceptance": branch_sequence_continuation_scorer_report.get(
                        "acceptance"
                    ),
                    "candidate_sequence_predictions": (
                        branch_sequence_continuation_scorer_report.get(
                            "candidate_sequence_predictions"
                        )
                    ),
                }
            ),
            "contract_digest": stable_payload_digest(contract),
        },
        "coverage": coverage,
        "planner_reports": planner_reports,
        "minimum_utility_cost_to_meet_action_cap": (
            _minimum_utility_cost_to_meet_cap(planner_reports)
        ),
        "constrained_planning_support_probe": {
            "policy": "v95_simulator_in_loop_constrained_planning_v1",
            "accuracy": 1.0 if accepted else 0.0,
            "support_accuracy_floor": 1.0,
            "materially_supports_constrained_planning": accepted,
            "runtime_policy_status": "diagnostic_only_no_runtime_policy_trained",
        },
        "acceptance": acceptance,
    }


def load_json_report(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    try:
        with _open_input(resolved) as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise BranchConstrainedPlanningAuditError(
            f"failed to read report: {resolved}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BranchConstrainedPlanningAuditError(
            f"report is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise BranchConstrainedPlanningAuditError("report must be a JSON object")
    return payload


def write_branch_constrained_planning_audit_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _validate_inputs(
    *,
    strict_branch_action_oracle_labels: Mapping[str, object],
    branch_sequence_continuation_scorer_report: Mapping[str, object],
) -> None:
    if (
        strict_branch_action_oracle_labels.get("schema_version")
        != MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION
    ):
        raise BranchConstrainedPlanningAuditError(
            "strict branch action oracle labels have unsupported schema_version"
        )
    if (
        branch_sequence_continuation_scorer_report.get("schema_version")
        != MIND_V3_BRANCH_SEQUENCE_CONTINUATION_SCORER_SCHEMA_VERSION
    ):
        raise BranchConstrainedPlanningAuditError(
            "sequence continuation scorer report has unsupported schema_version"
        )


def _v94_candidate_predictions(
    report: Mapping[str, object],
) -> dict[str, list[Mapping[str, object]]]:
    prediction_payload = _mapping(report.get("candidate_sequence_predictions"))
    scores = _list_of_mappings(
        prediction_payload.get("candidate_scores"),
        "candidate_sequence_predictions.candidate_scores",
    )
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for item in scores:
        grouped[str(item.get("branch_id", ""))].append(item)
    return {key: sorted(value, key=lambda item: str(item.get("action", ""))) for key, value in grouped.items()}


def _v94_baseline_actions(
    *,
    rows: Sequence[Mapping[str, object]],
    candidate_predictions: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, str]:
    actions = {}
    for row in rows:
        branch_id = str(row.get("branch_id", ""))
        action = _v94_select_action_for_rule(
            row,
            rule="sequence_prefix_nearest_neighbor_k5",
            candidate_predictions=list(candidate_predictions.get(branch_id, [])),
            v93_baseline_actions={},
        )
        if action is not None:
            actions[branch_id] = action
    return actions


def _candidate_outcomes(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    outcomes: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        branch_id = str(row.get("branch_id", ""))
        action_values = _candidate_actions(row)
        logged_run = _action_value(action_values, str(row.get("logged_action", "")))
        if logged_run is None:
            outcomes[branch_id] = []
            continue
        items = []
        for candidate in action_values:
            action = str(candidate.get("action", ""))
            comparison = _utility_comparison(
                row,
                rule="candidate_action_outcome",
                predicted_action=action,
                predicted_run=candidate,
                logged_run=logged_run,
            )
            target = _target_terminal_projection(candidate)
            comparison["predicted_target_alive"] = target.get("alive") is True
            comparison["branch_target_local_score"] = comparison.get(
                "predicted_target_local_score"
            )
            comparison["planner_score"] = _planner_score(comparison)
            items.append(comparison)
        outcomes[branch_id] = sorted(
            items,
            key=lambda item: str(item.get("predicted_action", "")),
        )
    return outcomes


def _assignment_from_actions(
    *,
    rows: Sequence[Mapping[str, object]],
    candidate_outcomes: Mapping[str, Sequence[Mapping[str, object]]],
    actions: Mapping[str, str],
) -> list[dict[str, object]]:
    assignment = []
    for row in rows:
        branch_id = str(row.get("branch_id", ""))
        action = actions.get(branch_id)
        candidate = _candidate_outcome(candidate_outcomes.get(branch_id, []), action)
        if candidate is not None:
            assignment.append(dict(candidate))
    return sorted(assignment, key=lambda item: str(item.get("branch_id", "")))


def _greedy_constrained_assignment(
    baseline_assignment: Sequence[Mapping[str, object]],
    *,
    candidate_outcomes: Mapping[str, Sequence[Mapping[str, object]]],
    max_action_count: int,
) -> list[dict[str, object]]:
    selected = {str(item.get("branch_id", "")): dict(item) for item in baseline_assignment}
    seen_assignments: set[tuple[tuple[str, str], ...]] = set()
    while True:
        assignment_key = _assignment_key(selected)
        if assignment_key in seen_assignments:
            break
        seen_assignments.add(assignment_key)
        counts = Counter(str(item.get("predicted_action", "")) for item in selected.values())
        dominant_action, dominant_count = _dominant_count(counts)
        if dominant_action is None or dominant_count <= max_action_count:
            break
        replacements = _replacement_options(
            selected,
            candidate_outcomes=candidate_outcomes,
            from_action=dominant_action,
        )
        if not replacements:
            break
        selected[replacements[0]["branch_id"]] = dict(replacements[0]["replacement"])
    return sorted(selected.values(), key=lambda item: str(item.get("branch_id", "")))


def _per_seed_constrained_assignment(
    baseline_assignment: Sequence[Mapping[str, object]],
    *,
    candidate_outcomes: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    selected = {str(item.get("branch_id", "")): dict(item) for item in baseline_assignment}
    by_seed: dict[str, list[str]] = defaultdict(list)
    for item in selected.values():
        by_seed[str(item.get("seed"))].append(str(item.get("branch_id", "")))
    for branch_ids in by_seed.values():
        max_seed_count = max(
            1,
            int(
                math.floor(
                    float(len(branch_ids)) * V95_MAX_DOMINANT_PREDICTED_ACTION_SHARE
                )
            ),
        )
        seen_assignments: set[tuple[tuple[str, str], ...]] = set()
        while True:
            subset = {branch_id: selected[branch_id] for branch_id in branch_ids}
            assignment_key = _assignment_key(subset)
            if assignment_key in seen_assignments:
                break
            seen_assignments.add(assignment_key)
            counts = Counter(
                str(selected[branch_id].get("predicted_action", ""))
                for branch_id in branch_ids
            )
            dominant_action, dominant_count = _dominant_count(counts)
            if dominant_action is None or dominant_count <= max_seed_count:
                break
            replacements = _replacement_options(
                subset,
                candidate_outcomes=candidate_outcomes,
                from_action=dominant_action,
            )
            if not replacements:
                break
            selected[replacements[0]["branch_id"]] = dict(replacements[0]["replacement"])
    return sorted(selected.values(), key=lambda item: str(item.get("branch_id", "")))


def _replacement_options(
    selected: Mapping[str, Mapping[str, object]],
    *,
    candidate_outcomes: Mapping[str, Sequence[Mapping[str, object]]],
    from_action: str,
) -> list[dict[str, object]]:
    options = []
    for branch_id, current in selected.items():
        if str(current.get("predicted_action", "")) != from_action:
            continue
        for candidate in candidate_outcomes.get(branch_id, []):
            if str(candidate.get("predicted_action", "")) == from_action:
                continue
            if _float(candidate.get("target_alive_delta")) < 0.0:
                continue
            options.append(
                {
                    "branch_id": branch_id,
                    "current": dict(current),
                    "replacement": dict(candidate),
                    "target_local_delta_change": _round(
                        _float(candidate.get("target_local_score_delta"))
                        - _float(current.get("target_local_score_delta"))
                    ),
                    "terminal_alive_delta_change": _round(
                        _float(candidate.get("terminal_alive_delta"))
                        - _float(current.get("terminal_alive_delta"))
                    ),
                    "birth_delta_change": _round(
                        _float(candidate.get("birth_delta"))
                        - _float(current.get("birth_delta"))
                    ),
                    "target_alive_delta_change": _round(
                        _float(candidate.get("target_alive_delta"))
                        - _float(current.get("target_alive_delta"))
                    ),
                }
            )
    return sorted(
        options,
        key=lambda item: (
            _float(item.get("target_alive_delta_change")),
            _float(item.get("target_local_delta_change")),
            _float(item.get("terminal_alive_delta_change")),
            _float(item.get("birth_delta_change")),
            str(_mapping(item.get("replacement")).get("predicted_action", "")),
            str(item.get("branch_id", "")),
        ),
        reverse=True,
    )


def _assignment_key(
    selected: Mapping[str, Mapping[str, object]],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (
                str(branch_id),
                str(item.get("predicted_action", "")),
            )
            for branch_id, item in selected.items()
        )
    )


def _beam_global_assignment(
    rows: Sequence[Mapping[str, object]],
    *,
    candidate_outcomes: Mapping[str, Sequence[Mapping[str, object]]],
    max_action_count: int,
    diversity_penalty: float,
) -> list[dict[str, object]]:
    states: list[tuple[tuple[float, float, float, float], Counter[str], list[dict[str, object]]]] = [
        ((0.0, 0.0, 0.0, 0.0), Counter(), [])
    ]
    ordered_rows = sorted(rows, key=lambda row: str(row.get("branch_id", "")))
    for row in ordered_rows:
        branch_id = str(row.get("branch_id", ""))
        candidates = [
            dict(item)
            for item in candidate_outcomes.get(branch_id, [])
            if _float(item.get("target_alive_delta")) >= 0.0
        ]
        if not candidates:
            candidates = [dict(item) for item in candidate_outcomes.get(branch_id, [])]
        next_states = []
        for score, counts, picked in states:
            for candidate in candidates:
                action = str(candidate.get("predicted_action", ""))
                if counts[action] >= max_action_count:
                    continue
                updated_counts = Counter(counts)
                updated_counts[action] += 1
                penalty = diversity_penalty * float(updated_counts[action] - 1)
                updated_score = (
                    score[0] + _float(candidate.get("target_alive_delta")),
                    score[1]
                    + _float(candidate.get("target_local_score_delta"))
                    - penalty,
                    score[2] + _float(candidate.get("terminal_alive_delta")),
                    score[3] + _float(candidate.get("birth_delta")),
                )
                next_states.append(
                    (updated_score, updated_counts, [*picked, candidate])
                )
        if not next_states:
            return []
        next_states.sort(
            key=lambda state: (
                state[0],
                -max(state[1].values(), default=0),
                stable_payload_digest(
                    [
                        item.get("branch_id")
                        for item in state[2]
                    ]
                ),
            ),
            reverse=True,
        )
        states = next_states[:V95_BEAM_WIDTH]
    feasible = [
        state
        for state in states
        if _assignment_clears_structural_constraints(
            state[2],
            max_action_count=max_action_count,
        )
    ]
    candidates = feasible or states
    best = max(
        candidates,
        key=lambda state: (
            _assignment_acceptance_score(state[2], max_action_count=max_action_count),
            state[0],
            -max(state[1].values(), default=0),
        ),
    )
    return sorted(best[2], key=lambda item: str(item.get("branch_id", "")))


def _assignment_clears_structural_constraints(
    assignment: Sequence[Mapping[str, object]],
    *,
    max_action_count: int,
) -> bool:
    if not assignment:
        return False
    counts = Counter(str(item.get("predicted_action", "")) for item in assignment)
    if max(counts.values(), default=0) > max_action_count:
        return False
    if any(_float(item.get("target_alive_delta")) < 0.0 for item in assignment):
        return False
    return True


def _assignment_acceptance_score(
    assignment: Sequence[Mapping[str, object]],
    *,
    max_action_count: int,
) -> tuple[float, ...]:
    counts = Counter(str(item.get("predicted_action", "")) for item in assignment)
    return (
        1.0 if max(counts.values(), default=0) <= max_action_count else 0.0,
        -float(sum(1 for item in assignment if _float(item.get("target_alive_delta")) < 0.0)),
        _mean([_float(item.get("target_local_score_delta")) for item in assignment]),
        _mean([_float(item.get("terminal_alive_delta")) for item in assignment]),
        _mean([_float(item.get("birth_delta")) for item in assignment]),
        -_safe_rate(max(counts.values(), default=0), len(assignment)),
    )


def _planner_report(
    rule: str,
    assignment: Sequence[Mapping[str, object]],
    *,
    baseline_assignment: Sequence[Mapping[str, object]],
    max_action_count: int,
) -> dict[str, object]:
    baseline_by_branch = {
        str(item.get("branch_id", "")): item for item in baseline_assignment
    }
    comparisons = [dict(item, rule=rule) for item in assignment]
    action_counts = Counter(str(item.get("predicted_action")) for item in comparisons)
    mode_counts = Counter(str(item.get("predicted_mode")) for item in comparisons)
    dominant_action, dominant_action_count = _dominant_count(action_counts)
    dominant_mode, dominant_mode_count = _dominant_count(mode_counts)
    replacements = _replacements_vs_baseline(comparisons, baseline_by_branch)
    utility_delta_vs_baseline = _round(
        sum(_float(item.get("target_local_score_delta")) for item in comparisons)
        - sum(
            _float(item.get("target_local_score_delta"))
            for item in baseline_assignment
        )
    )
    return {
        "rule": rule,
        "rule_class": "baseline" if rule == V95_BASELINE_RULE else "planner",
        "eligible_for_acceptance": rule in _PLANNER_RULES,
        "runtime_ready": False,
        "comparison_count": len(comparisons),
        "max_allowed_action_count": int(max_action_count),
        "predicted_action_counts": dict(sorted(action_counts.items())),
        "predicted_mode_counts": dict(sorted(mode_counts.items())),
        "dominant_predicted_action": dominant_action,
        "dominant_predicted_action_count": dominant_action_count,
        "dominant_predicted_action_share": _safe_rate(
            dominant_action_count,
            len(comparisons),
        ),
        "dominant_predicted_mode": dominant_mode,
        "dominant_predicted_mode_count": dominant_mode_count,
        "dominant_predicted_mode_share": _safe_rate(
            dominant_mode_count,
            len(comparisons),
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
        "target_alive_delta_summary": _field_summary(
            comparisons,
            "target_alive_delta",
        ),
        "per_seed_delta_means": _per_seed_delta_means(comparisons),
        "seed_41_sequence_cases": _seed_41_sequence_cases(comparisons),
        "replacement_summary_vs_v94_baseline": {
            "changed_action_count": len(replacements),
            "eat_replacement_count": sum(
                1 for item in replacements if item.get("from_action") == "eat"
            ),
            "target_local_sum_delta_vs_baseline": utility_delta_vs_baseline,
            "target_local_utility_cost_vs_baseline": _round(
                max(0.0, -utility_delta_vs_baseline)
            ),
            "replacements": replacements,
        },
        "negative_comparisons": sorted(
            [
                dict(item)
                for item in comparisons
                if _float(item.get("target_local_score_delta")) < 0.0
            ],
            key=lambda item: (
                _float(item.get("target_local_score_delta")),
                str(item.get("branch_id", "")),
            ),
        ),
    }


def _coverage_report(
    *,
    strict_labels: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    candidate_outcomes: Mapping[str, Sequence[Mapping[str, object]]],
    candidate_predictions: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    aggregate = _mapping(strict_labels.get("aggregate"))
    return {
        "strict_eval_label_count": len(rows),
        "strict_eval_labels_by_seed": dict(
            sorted(Counter(str(row.get("seed")) for row in rows).items())
        ),
        "strict_candidate_outcome_count": sum(
            len(items) for items in candidate_outcomes.values()
        ),
        "v94_candidate_prediction_count": sum(
            len(items) for items in candidate_predictions.values()
        ),
        "strict_replay_verified_all_labels": bool(
            aggregate.get("replay_verified_all_labels", False)
        ),
        "strict_heuristic_action_source_count": _heuristic_count(aggregate),
        "strict_unsupported_oracle_action_count": int(
            aggregate.get("unsupported_oracle_action_count", 0)
        ),
        "strict_unsupported_logged_action_count": int(
            aggregate.get("unsupported_logged_action_count", 0)
        ),
    }


def _acceptance(
    *,
    coverage: Mapping[str, object],
    planner_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    blockers_by_rule = []
    for report in planner_reports:
        if report.get("eligible_for_acceptance") is not True:
            continue
        blockers_by_rule.append(
            {
                "rule": report.get("rule"),
                "blockers": _rule_blockers(report, coverage),
            }
        )
    accepted_rules = [
        str(item.get("rule"))
        for item in blockers_by_rule
        if not item.get("blockers")
    ]
    best = _best_rule_for_diagnostics(planner_reports, blockers_by_rule)
    best_blockers = next(
        (
            list(_list_of_mappings(item.get("blockers"), "blockers"))
            for item in blockers_by_rule
            if item.get("rule") == best.get("rule")
        ),
        [],
    )
    accepted = bool(accepted_rules)
    return {
        "v95_constrained_planning_diagnostic_accepted": accepted,
        "v96_distillation_runtime_feasibility_allowed": accepted,
        "runtime_policy_trained": False,
        "runtime_ready": False,
        "accepted_rules": accepted_rules,
        "best_rule_for_diagnostics": best,
        "strict_blockers": best_blockers,
        "blockers_by_rule": blockers_by_rule,
    }


def _rule_blockers(
    report: Mapping[str, object],
    coverage: Mapping[str, object],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    if coverage.get("strict_replay_verified_all_labels") is not True:
        blockers.append({"reason": "strict_replay_verified_all_labels_false"})
    for field in (
        "strict_heuristic_action_source_count",
        "strict_unsupported_oracle_action_count",
    ):
        if int(coverage.get(field, 0)) != 0:
            blockers.append(
                {
                    "reason": f"{field}_nonzero",
                    "observed": int(coverage.get(field, 0)),
                    "required": 0,
                }
            )
    if int(report.get("comparison_count", 0)) < V95_MIN_STRICT_COMPARISON_COUNT:
        blockers.append(
            {
                "reason": "strict_eval_comparison_count_below_floor",
                "observed": int(report.get("comparison_count", 0)),
                "required_min": V95_MIN_STRICT_COMPARISON_COUNT,
            }
        )
    if (
        _float(report.get("dominant_predicted_action_share"))
        > V95_MAX_DOMINANT_PREDICTED_ACTION_SHARE
    ):
        blockers.append(
            {
                "reason": "dominant_predicted_action_share_above_cap",
                "observed": report.get("dominant_predicted_action_share"),
                "required_max": V95_MAX_DOMINANT_PREDICTED_ACTION_SHARE,
            }
        )
    target_alive = _mapping(report.get("target_alive_delta_summary"))
    if int(target_alive.get("negative_count", 0)) != V95_MAX_TARGET_ALIVE_NEGATIVE_COUNT:
        blockers.append(
            {
                "reason": "target_alive_delta_negative_count_nonzero",
                "observed": int(target_alive.get("negative_count", 0)),
                "required": V95_MAX_TARGET_ALIVE_NEGATIVE_COUNT,
            }
        )
    target_local = _mapping(report.get("target_local_score_delta_summary"))
    if _float(target_local.get("mean")) <= V95_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA:
        blockers.append(
            {
                "reason": "mean_target_local_score_delta_not_positive",
                "observed": target_local.get("mean"),
                "required_gt": V95_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA,
            }
        )
    negative_seed_means = {
        seed: _mapping(value).get("target_local_score_delta")
        for seed, value in sorted(_mapping(report.get("per_seed_delta_means")).items())
        if _float(_mapping(value).get("target_local_score_delta"))
        < V95_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA
    }
    if negative_seed_means:
        blockers.append(
            {
                "reason": "per_seed_target_local_score_delta_negative",
                "observed": negative_seed_means,
                "required_min": V95_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA,
            }
        )
    terminal_alive = _mapping(report.get("terminal_alive_delta_summary"))
    if _float(terminal_alive.get("mean")) < V95_MIN_MEAN_TERMINAL_ALIVE_DELTA:
        blockers.append(
            {
                "reason": "mean_terminal_alive_delta_negative",
                "observed": terminal_alive.get("mean"),
                "required_min": V95_MIN_MEAN_TERMINAL_ALIVE_DELTA,
            }
        )
    births = _mapping(report.get("birth_delta_summary"))
    if _float(births.get("mean")) < V95_MIN_MEAN_BIRTH_DELTA:
        blockers.append(
            {
                "reason": "mean_birth_delta_negative",
                "observed": births.get("mean"),
                "required_min": V95_MIN_MEAN_BIRTH_DELTA,
            }
        )
    seed_41 = _mapping(report.get("seed_41_sequence_cases"))
    for key in ("tick_113", "tick_114"):
        case = _mapping(seed_41.get(key))
        if case.get("avoided") is not True:
            blockers.append(
                {
                    "reason": f"seed_41_{key}_catastrophe_not_avoided",
                    "observed": case,
                    "required": "target_alive_delta >= 0",
                }
            )
    if report.get("runtime_ready") is not False:
        blockers.append({"reason": "accepted_result_presented_as_runtime_ready"})
    return blockers


def _best_rule_for_diagnostics(
    planner_reports: Sequence[Mapping[str, object]],
    blockers_by_rule: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    blocker_count_by_rule = {
        str(item.get("rule")): len(_list_of_mappings(item.get("blockers"), "blockers"))
        for item in blockers_by_rule
    }
    candidates = [
        report
        for report in planner_reports
        if report.get("eligible_for_acceptance") is True
    ]
    if not candidates:
        return {}
    best = max(
        candidates,
        key=lambda report: (
            -blocker_count_by_rule.get(str(report.get("rule")), 999),
            -_float(report.get("dominant_predicted_action_share")),
            -int(
                _mapping(report.get("target_alive_delta_summary")).get(
                    "negative_count",
                    0,
                )
            ),
            _float(
                _mapping(report.get("target_local_score_delta_summary")).get("mean")
            ),
            _float(_mapping(report.get("terminal_alive_delta_summary")).get("mean")),
            _float(_mapping(report.get("birth_delta_summary")).get("mean")),
            -_float(
                _mapping(report.get("replacement_summary_vs_v94_baseline")).get(
                    "target_local_utility_cost_vs_baseline"
                )
            ),
            str(report.get("rule")),
        ),
    )
    seed_41 = _mapping(best.get("seed_41_sequence_cases"))
    replacement = _mapping(best.get("replacement_summary_vs_v94_baseline"))
    return {
        "rule": best.get("rule"),
        "blocker_count": blocker_count_by_rule.get(str(best.get("rule")), 0),
        "dominant_predicted_action": best.get("dominant_predicted_action"),
        "dominant_predicted_action_share": best.get("dominant_predicted_action_share"),
        "mean_target_local_score_delta": _mapping(
            best.get("target_local_score_delta_summary")
        ).get("mean"),
        "target_alive_delta_negative_count": _mapping(
            best.get("target_alive_delta_summary")
        ).get("negative_count"),
        "mean_terminal_alive_delta": _mapping(
            best.get("terminal_alive_delta_summary")
        ).get("mean"),
        "mean_birth_delta": _mapping(best.get("birth_delta_summary")).get("mean"),
        "seed_41_tick_113_avoided": _mapping(seed_41.get("tick_113")).get("avoided"),
        "seed_41_tick_114_avoided": _mapping(seed_41.get("tick_114")).get("avoided"),
        "changed_action_count_vs_v94": replacement.get("changed_action_count"),
        "eat_replacement_count_vs_v94": replacement.get("eat_replacement_count"),
        "target_local_utility_cost_vs_v94": replacement.get(
            "target_local_utility_cost_vs_baseline"
        ),
        "target_local_sum_delta_vs_v94": replacement.get(
            "target_local_sum_delta_vs_baseline"
        ),
        "runtime_ready": best.get("runtime_ready"),
    }


def _minimum_utility_cost_to_meet_cap(
    planner_reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    baseline = next(
        (
            report
            for report in planner_reports
            if report.get("rule") == V95_BASELINE_RULE
        ),
        None,
    )
    baseline_counts = _mapping(_mapping(baseline).get("predicted_action_counts"))
    baseline_action = _mapping(baseline).get("dominant_predicted_action")
    baseline_action_count = _int(baseline_counts.get(str(baseline_action)))
    baseline_comparison_count = _int(_mapping(baseline).get("comparison_count"))
    max_action_count = int(
        math.floor(
            float(baseline_comparison_count)
            * V95_MAX_DOMINANT_PREDICTED_ACTION_SHARE
        )
    )
    required_reduction = max(0, baseline_action_count - max_action_count)
    eligible = [
        report
        for report in planner_reports
        if report.get("eligible_for_acceptance") is True
        and _float(report.get("dominant_predicted_action_share"))
        <= V95_MAX_DOMINANT_PREDICTED_ACTION_SHARE
    ]
    if not eligible:
        return {"available": False}
    best = min(
        eligible,
        key=lambda report: (
            abs(
                _int(
                    _mapping(report.get("replacement_summary_vs_v94_baseline")).get(
                        "eat_replacement_count"
                    )
                )
                - required_reduction
            ),
            _float(
                _mapping(report.get("replacement_summary_vs_v94_baseline")).get(
                    "target_local_utility_cost_vs_baseline"
                )
            ),
            -_float(
                _mapping(report.get("target_local_score_delta_summary")).get("mean")
            ),
            str(report.get("rule")),
        ),
    )
    replacement = _mapping(best.get("replacement_summary_vs_v94_baseline"))
    return {
        "available": True,
        "baseline_dominant_action": baseline_action,
        "baseline_dominant_action_count": baseline_action_count,
        "max_allowed_action_count": max_action_count,
        "required_reduction": required_reduction,
        "rule": best.get("rule"),
        "target_local_utility_cost_vs_v94": replacement.get(
            "target_local_utility_cost_vs_baseline"
        ),
        "target_local_sum_delta_vs_v94": replacement.get(
            "target_local_sum_delta_vs_baseline"
        ),
        "changed_action_count": replacement.get("changed_action_count"),
        "eat_replacement_count": replacement.get("eat_replacement_count"),
        "replacements": replacement.get("replacements"),
    }


def _replacements_vs_baseline(
    comparisons: Sequence[Mapping[str, object]],
    baseline_by_branch: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    replacements = []
    for item in comparisons:
        branch_id = str(item.get("branch_id", ""))
        baseline = _mapping(baseline_by_branch.get(branch_id))
        from_action = str(baseline.get("predicted_action", ""))
        to_action = str(item.get("predicted_action", ""))
        if not from_action or from_action == to_action:
            continue
        replacements.append(
            {
                "branch_id": branch_id,
                "seed": item.get("seed"),
                "from_action": from_action,
                "to_action": to_action,
                "from_mode": _action_option_mode(from_action),
                "to_mode": _action_option_mode(to_action),
                "target_local_delta_change": _round(
                    _float(item.get("target_local_score_delta"))
                    - _float(baseline.get("target_local_score_delta"))
                ),
                "terminal_alive_delta_change": _round(
                    _float(item.get("terminal_alive_delta"))
                    - _float(baseline.get("terminal_alive_delta"))
                ),
                "birth_delta_change": _round(
                    _float(item.get("birth_delta"))
                    - _float(baseline.get("birth_delta"))
                ),
                "target_alive_delta_change": _round(
                    _float(item.get("target_alive_delta"))
                    - _float(baseline.get("target_alive_delta"))
                ),
                "baseline_target_local_score_delta": baseline.get(
                    "target_local_score_delta"
                ),
                "replacement_target_local_score_delta": item.get(
                    "target_local_score_delta"
                ),
            }
        )
    return sorted(
        replacements,
        key=lambda item: (
            str(item.get("from_action", "")),
            str(item.get("branch_id", "")),
            str(item.get("to_action", "")),
        ),
    )


def _per_seed_delta_means(
    comparisons: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, float]]:
    fields = (
        "target_local_score_delta",
        "terminal_alive_delta",
        "birth_delta",
        "target_alive_delta",
    )
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for item in comparisons:
        grouped[str(item.get("seed"))].append(item)
    return {
        seed: {
            field: _round(_mean([_float(item.get(field)) for item in items]))
            for field in fields
        }
        for seed, items in sorted(grouped.items())
    }


def _seed_41_sequence_cases(
    comparisons: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "tick_113": _seed_41_case(comparisons, V94_SEED_41_TICK_113_BRANCH_ID),
        "tick_114": _seed_41_case(comparisons, V94_SEED_41_TICK_114_BRANCH_ID),
    }


def _seed_41_case(
    comparisons: Sequence[Mapping[str, object]],
    branch_id: str,
) -> dict[str, object]:
    matching = [
        dict(item) for item in comparisons if str(item.get("branch_id", "")) == branch_id
    ]
    return {
        "branch_id": branch_id,
        "comparison_found": bool(matching),
        "avoided": bool(matching)
        and all(_float(item.get("target_alive_delta")) >= 0.0 for item in matching),
        "comparisons": matching,
    }


def _planner_score(comparison: Mapping[str, object]) -> float:
    return _round(
        (1.0 if comparison.get("predicted_target_alive") is True else 0.0) * 10000.0
        + _float(comparison.get("target_alive_delta")) * 1000.0
        + _float(comparison.get("target_local_score_delta"))
        + _float(comparison.get("terminal_alive_delta")) * 10.0
        + _float(comparison.get("birth_delta")) * 5.0
    )


def _action_value(
    action_values: Sequence[Mapping[str, object]],
    action: str,
) -> Mapping[str, object] | None:
    return next(
        (item for item in action_values if str(item.get("action", "")) == action),
        None,
    )


def _candidate_outcome(
    outcomes: Sequence[Mapping[str, object]],
    action: str | None,
) -> Mapping[str, object] | None:
    if action is None:
        return None
    return next(
        (item for item in outcomes if str(item.get("predicted_action", "")) == action),
        None,
    )


def _dominant_count(counts: Counter[str]) -> tuple[str | None, int]:
    if not counts:
        return None, 0
    item = max(sorted(counts.items()), key=lambda value: (value[1], value[0]))
    return item[0], int(item[1])


def _heuristic_count(aggregate: Mapping[str, object]) -> int:
    return int(
        aggregate.get("heuristic_action_source_count", 0)
        or (
            0
            if aggregate.get("zero_heuristic_all_labels") is True
            else aggregate.get("label_count", 0)
        )
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _round(value: float) -> float:
    return round(float(value), 6)


def _open_input(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _open_output(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8")
    return path.open("w", encoding="utf-8")
