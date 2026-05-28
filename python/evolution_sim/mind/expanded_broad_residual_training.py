from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.branch_mode_objective_audit import _action_option_mode
from evolution_sim.mind.broad_branch_residual_constrained_audit import (
    MIND_V3_V100_BROAD_BRANCH_RESIDUAL_CONSTRAINED_AUDIT_SCHEMA_VERSION,
    _candidate_rows,
)
from evolution_sim.mind.broad_branch_residual_distillation_example import (
    _distillation_example_artifact,
    _training_evaluation,
    _training_rows,
    load_json_report,
    validate_broad_residual_distillation_example_artifact,
    score_broad_residual_distillation_example_artifact,
)
from evolution_sim.mind.broad_branch_residual_oracle_audit import (
    MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.broad_transfer_residual_audit import V98_STRICT_EXCLUDED_SEEDS
from evolution_sim.mind.provenance import stable_payload_digest

MIND_V3_V102_EXPANDED_BROAD_RESIDUAL_TRAINING_SCHEMA_VERSION = (
    "mind_v3_v102_expanded_broad_residual_training_v1"
)
MIND_V3_V102_EXPANDED_BROAD_RESIDUAL_TRAINING_POLICY = (
    "v102_expanded_policy_visible_broad_residual_training_v1"
)
V102_MIN_TRAINING_ROWS = 80
V102_MIN_SOURCE_SEEDS = 10
V102_MAX_DOMINANT_TEACHER_ACTION_SHARE = 0.50
V102_MIN_TEACHER_MODE_COUNT = 4
V102_MIN_REPOSITION_LABEL_SHARE = 0.20
V102_MAX_LOO_DOMINANT_PREDICTED_ACTION_SHARE = 0.50
V102_ACTION_PRIOR_BALANCE_PENALTY = 2.0
V102_REQUIRED_BRANCH_CONTEXTS: dict[str, tuple[str, ...]] = {
    "plant_food": ("energy", "plant_food"),
    "hydration": ("hydration",),
    "movement": ("movement",),
    "reproduction_readiness": ("reproduction_readiness",),
    "pre_death": ("pre_death",),
    "recovery": ("recovery",),
    "animal_resource": ("animal_resource",),
}


class ExpandedBroadResidualTrainingError(ValueError):
    pass


def build_expanded_broad_residual_training_report(
    *,
    v99_expanded_oracle_report: Mapping[str, object],
    v100_constrained_report: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    _validate_source_reports(
        v99_report=v99_expanded_oracle_report,
        v100_report=v100_constrained_report,
    )
    rows = _training_rows(
        v99_report=v99_expanded_oracle_report,
        v100_report=v100_constrained_report,
    )
    artifact = _distillation_example_artifact(rows)
    validate_broad_residual_distillation_example_artifact(artifact)
    training_eval = _training_evaluation(artifact, rows)
    reloaded = json.loads(json.dumps(artifact, sort_keys=True, allow_nan=False))
    reload_eval = _training_evaluation(reloaded, rows)
    reload_identical = training_eval["predicted_actions"] == reload_eval[
        "predicted_actions"
    ]
    coverage = _coverage(
        rows=rows,
        v99_report=v99_expanded_oracle_report,
    )
    loo_eval = _leave_one_source_seed_out_evaluation(
        rows=rows,
        v99_report=v99_expanded_oracle_report,
    )
    floors = {
        "training_row_count": V102_MIN_TRAINING_ROWS,
        "source_seed_count": V102_MIN_SOURCE_SEEDS,
        "strict_seed_leakage": 0,
        "replay_verified": True,
        "unsupported_candidate_action_count": 0,
        "unsupported_predicted_action_count": 0,
        "dominant_teacher_action_share_max": V102_MAX_DOMINANT_TEACHER_ACTION_SHARE,
        "represented_teacher_mode_count": V102_MIN_TEACHER_MODE_COUNT,
        "reposition_label_share": V102_MIN_REPOSITION_LABEL_SHARE,
        "required_branch_contexts": sorted(V102_REQUIRED_BRANCH_CONTEXTS),
        "target_alive_delta_negative_count": 0,
        "mean_target_local_score_delta_gt": 0.0,
        "mean_terminal_alive_delta_min": 0.0,
        "mean_birth_delta_min": 0.0,
        "artifact_reload_identical_choices": True,
        "loo_dominant_predicted_action_share_max": (
            V102_MAX_LOO_DOMINANT_PREDICTED_ACTION_SHARE
        ),
        "loo_mean_replay_target_local_score_delta_gt": 0.0,
        "runtime_policy_trained": False,
    }
    acceptance = _acceptance(
        coverage=coverage,
        training_eval=training_eval,
        loo_eval=loo_eval,
        reload_identical=reload_identical,
        floors=floors,
    )
    accepted = bool(acceptance["v102_expanded_broad_residual_training_accepted"])
    contract = {
        "schema_version": (
            MIND_V3_V102_EXPANDED_BROAD_RESIDUAL_TRAINING_SCHEMA_VERSION
        ),
        "policy": MIND_V3_V102_EXPANDED_BROAD_RESIDUAL_TRAINING_POLICY,
        "diagnostic_only": True,
        "runtime_policy_trained": False,
        "promotion_run_executed": False,
        "strict_excluded_seeds": list(V98_STRICT_EXCLUDED_SEEDS),
        "source_oracle_schema_version": v99_expanded_oracle_report.get(
            "schema_version"
        ),
        "source_constrained_schema_version": v100_constrained_report.get(
            "schema_version"
        ),
        "training_contract": {
            "default_policy": "linear_mind_v3",
            "label_source": "expanded_v100_diversity_constrained_assignment",
            "features": [
                "policy-visible observation_input",
                "action_mask",
                "public_history_trace",
                "candidate action identity",
            ],
            "uses_private_world_state": False,
            "uses_fixture_identity": False,
            "uses_seed_id_as_runtime_feature": False,
            "uses_branch_id_as_runtime_feature": False,
            "uses_logged_action_as_runtime_fallback": False,
            "uses_global_batch_quota_at_runtime": False,
            "uses_planner_outcome_tables_at_runtime": False,
        },
        "support_floors": floors,
    }
    support_probe = {
        "policy": MIND_V3_V102_EXPANDED_BROAD_RESIDUAL_TRAINING_POLICY,
        "accuracy": 1.0 if accepted else 0.0,
        "support_accuracy_floor": 1.0,
        "materially_supports_v103_residual_runtime": accepted,
        "runtime_policy_status": (
            "v103_runtime_feasibility_allowed"
            if accepted
            else "rejected_no_runtime_policy"
        ),
        "training_row_count": coverage["training_row_count"],
        "source_seed_count": coverage["source_seed_count"],
        "dominant_teacher_action_share": coverage[
            "dominant_teacher_action_share"
        ],
        "loo_dominant_predicted_action_share": loo_eval[
            "dominant_predicted_action_share"
        ],
        "loo_mean_replay_target_local_score_delta": loo_eval[
            "target_local_score_delta_mean"
        ],
        "blocker_count": acceptance["blocker_count"],
    }
    report = {
        "schema_version": (
            MIND_V3_V102_EXPANDED_BROAD_RESIDUAL_TRAINING_SCHEMA_VERSION
        ),
        "policy": MIND_V3_V102_EXPANDED_BROAD_RESIDUAL_TRAINING_POLICY,
        "contract": contract,
        "provenance": {
            "source_v99_digest": stable_payload_digest(
                {
                    "schema_version": v99_expanded_oracle_report.get(
                        "schema_version"
                    ),
                    "aggregate": v99_expanded_oracle_report.get("aggregate"),
                    "branch_points": v99_expanded_oracle_report.get(
                        "branch_points"
                    ),
                    "branch_results": v99_expanded_oracle_report.get(
                        "branch_results"
                    ),
                }
            ),
            "source_v100_digest": stable_payload_digest(
                {
                    "schema_version": v100_constrained_report.get(
                        "schema_version"
                    ),
                    "coverage": v100_constrained_report.get("coverage"),
                    "acceptance": v100_constrained_report.get("acceptance"),
                    "rule_reports": v100_constrained_report.get("rule_reports"),
                }
            ),
            "artifact_digest": stable_payload_digest(artifact),
            "contract_digest": stable_payload_digest(contract),
        },
        "coverage": coverage,
        "training_evaluation": training_eval,
        "reload_evaluation": {
            "artifact_reload_identical_choices": reload_identical,
            "training_accuracy": reload_eval["training_accuracy"],
            "predicted_action_counts": reload_eval["predicted_action_counts"],
        },
        "leave_one_source_seed_out_evaluation": loo_eval,
        "expanded_training_artifact": artifact,
        "expanded_broad_residual_training_support_probe": support_probe,
        "acceptance": acceptance,
        "v102_expanded_broad_residual_training_accepted": accepted,
        "v103_support_gated_residual_runtime_allowed": bool(
            acceptance["v103_support_gated_residual_runtime_allowed"]
        ),
        "runtime_promotion_allowed": False,
        "blocker_count": int(acceptance["blocker_count"]),
    }
    return report, artifact


def write_expanded_broad_residual_training_report(
    report: Mapping[str, object],
    *,
    output_path: str | Path,
    artifact: Mapping[str, object] | None = None,
    artifact_output_path: str | Path | None = None,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    if artifact is not None and artifact_output_path is not None:
        artifact_path = Path(artifact_output_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with _open_output(artifact_path) as handle:
            json.dump(artifact, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")


def _coverage(
    *,
    rows: Sequence[Mapping[str, object]],
    v99_report: Mapping[str, object],
) -> dict[str, object]:
    source_seeds = sorted(
        {
            int(_mapping(row.get("provenance")).get("source_seed", 0))
            for row in rows
        }
    )
    teacher_action_counts = Counter(str(row.get("teacher_action", "")) for row in rows)
    teacher_mode_counts = Counter(str(row.get("teacher_mode", "")) for row in rows)
    category_counts = Counter(
        str(category)
        for point in _list_of_mappings(v99_report.get("branch_points"))
        for category in _list(point.get("categories"))
    )
    logged_action_counts = Counter(
        str(point.get("logged_action", ""))
        for point in _list_of_mappings(v99_report.get("branch_points"))
    )
    aggregate = _mapping(v99_report.get("aggregate"))
    dominant = _dominant_count_share(teacher_action_counts)
    reposition_count = int(teacher_mode_counts.get("reposition", 0))
    return {
        "training_row_count": len(rows),
        "source_seed_count": len(source_seeds),
        "source_seeds": source_seeds,
        "strict_seed_leak_count": len(set(source_seeds) & set(V98_STRICT_EXCLUDED_SEEDS)),
        "source_replay_verified": bool(aggregate.get("replay_verified")),
        "source_heuristic_action_source_count": int(
            aggregate.get("heuristic_action_source_count", 0)
        ),
        "source_unsupported_candidate_action_count": int(
            aggregate.get("unsupported_candidate_action_count", 0)
        ),
        "label_action_legal_count": sum(
            1
            for row in rows
            if row.get("teacher_action") in _list(row.get("legal_actions"))
        ),
        "teacher_action_counts": dict(sorted(teacher_action_counts.items())),
        "teacher_mode_counts": dict(sorted(teacher_mode_counts.items())),
        "represented_teacher_mode_count": len(
            {mode for mode, count in teacher_mode_counts.items() if count > 0}
        ),
        "dominant_teacher_action": dominant["key"],
        "dominant_teacher_action_count": dominant["count"],
        "dominant_teacher_action_share": dominant["share"],
        "movement_reposition_label_count": reposition_count,
        "movement_reposition_label_share": _safe_rate(reposition_count, len(rows)),
        "branch_category_counts": dict(sorted(category_counts.items())),
        "required_branch_context_counts": {
            context: sum(int(category_counts.get(category, 0)) for category in aliases)
            for context, aliases in sorted(V102_REQUIRED_BRANCH_CONTEXTS.items())
        },
        "missing_required_branch_contexts": [
            context
            for context, aliases in sorted(V102_REQUIRED_BRANCH_CONTEXTS.items())
            if sum(int(category_counts.get(category, 0)) for category in aliases) <= 0
        ],
        "branch_logged_action_counts": dict(sorted(logged_action_counts.items())),
        "override_label_count": sum(1 for row in rows if row.get("override_label") is True),
        "safe_non_logged_override_count": sum(
            1 for row in rows if row.get("safe_non_logged_override") is True
        ),
        "target_alive_delta_negative_count": sum(
            1 for row in rows if _float(row.get("target_alive_delta")) < 0.0
        ),
        "target_local_score_delta_mean": _round(
            _mean([_float(row.get("target_local_score_delta")) for row in rows])
        ),
        "terminal_alive_delta_mean": _round(
            _mean([_float(row.get("terminal_alive_delta")) for row in rows])
        ),
        "birth_delta_mean": _round(
            _mean([_float(row.get("birth_delta")) for row in rows])
        ),
    }


def _leave_one_source_seed_out_evaluation(
    *,
    rows: Sequence[Mapping[str, object]],
    v99_report: Mapping[str, object],
) -> dict[str, object]:
    rule_reports = [
        _loo_rule_evaluation(
            rows=rows,
            v99_report=v99_report,
            rule="nearest_support_v1",
            action_prior_penalty_scale=0.0,
            acceptance_candidate=False,
        ),
        _loo_rule_evaluation(
            rows=rows,
            v99_report=v99_report,
            rule="action_prior_balanced_nearest_support_v1",
            action_prior_penalty_scale=V102_ACTION_PRIOR_BALANCE_PENALTY,
            acceptance_candidate=True,
        ),
    ]
    best = max(
        rule_reports,
        key=lambda report: (
            -int(report.get("blocker_count", 0)),
            _float(report.get("target_local_score_delta_mean")),
            -_float(report.get("dominant_predicted_action_share")),
            1 if report.get("acceptance_candidate") is True else 0,
            str(report.get("rule", "")),
        ),
    )
    return {
        **best,
        "selected_rule": best.get("rule"),
        "rule_reports": rule_reports,
    }


def _loo_rule_evaluation(
    *,
    rows: Sequence[Mapping[str, object]],
    v99_report: Mapping[str, object],
    rule: str,
    action_prior_penalty_scale: float,
    acceptance_candidate: bool,
) -> dict[str, object]:
    candidate_rows = _candidate_rows(v99_report)
    candidates_by_branch = {
        str(row.get("branch_id", "")): _list_of_mappings(row.get("candidates"))
        for row in candidate_rows
    }
    source_seeds = sorted(
        {
            int(_mapping(row.get("provenance")).get("source_seed", 0))
            for row in rows
        }
    )
    comparisons: list[dict[str, object]] = []
    unsupported_count = 0
    for held_out_seed in source_seeds:
        train_rows = [
            row
            for row in rows
            if int(_mapping(row.get("provenance")).get("source_seed", 0))
            != held_out_seed
        ]
        heldout_rows = [
            row
            for row in rows
            if int(_mapping(row.get("provenance")).get("source_seed", 0))
            == held_out_seed
        ]
        artifact = _distillation_example_artifact(train_rows)
        validate_broad_residual_distillation_example_artifact(artifact)
        support = _list_of_mappings(artifact.get("support_examples"))
        train_action_counts = Counter(
            str(row.get("teacher_action", "")) for row in train_rows
        )
        train_count = max(len(train_rows), 1)
        for row in heldout_rows:
            policy_state = _mapping(row.get("policy_state"))
            scored = _score_with_action_prior_balance(
                artifact=artifact,
                support=support,
                train_action_counts=train_action_counts,
                train_count=train_count,
                action_prior_penalty_scale=action_prior_penalty_scale,
                observation_input=_mapping(policy_state.get("observation_input")),
                action_mask=_mapping(policy_state.get("action_mask")),
                public_history_trace=_list_of_mappings(
                    policy_state.get("public_history_trace")
                ),
            )
            predicted_action = scored.get("selected_action")
            provenance = _mapping(row.get("provenance"))
            branch_id = str(provenance.get("branch_id", ""))
            candidates = candidates_by_branch.get(branch_id, [])
            candidate = _candidate_by_action(candidates, str(predicted_action))
            unsupported = (
                predicted_action not in ACTION_NAMES
                or scored.get("unsupported_predicted_action") is True
                or candidate is None
            )
            if unsupported:
                unsupported_count += 1
            comparisons.append(
                {
                    "held_out_seed": held_out_seed,
                    "row_index": row.get("row_index"),
                    "branch_tick": provenance.get("branch_tick"),
                    "agent_id": provenance.get("agent_id"),
                    "teacher_action": row.get("teacher_action"),
                    "teacher_mode": row.get("teacher_mode"),
                    "predicted_action": predicted_action,
                    "predicted_mode": _action_option_mode(str(predicted_action)),
                    "exact_action_correct": predicted_action == row.get("teacher_action"),
                    "unsupported_predicted_action": unsupported,
                    "target_local_score_delta": _round(
                        _float(_mapping(candidate).get("target_local_score_delta"))
                    )
                    if candidate is not None
                    else 0.0,
                    "terminal_alive_delta": _round(
                        _float(_mapping(candidate).get("terminal_alive_delta"))
                    )
                    if candidate is not None
                    else 0.0,
                    "birth_delta": _round(_float(_mapping(candidate).get("birth_delta")))
                    if candidate is not None
                    else 0.0,
                    "target_alive_delta": _round(
                        _float(_mapping(candidate).get("target_alive_delta"))
                    )
                    if candidate is not None
                    else 0.0,
                }
            )
    predicted_counts = Counter(
        str(item.get("predicted_action", "")) for item in comparisons
    )
    predicted_mode_counts = Counter(
        str(item.get("predicted_mode", "")) for item in comparisons
    )
    exact_correct = sum(
        1 for item in comparisons if item.get("exact_action_correct") is True
    )
    by_seed = []
    for seed in source_seeds:
        seed_rows = [
            item for item in comparisons if int(item.get("held_out_seed", 0)) == seed
        ]
        by_seed.append(
            {
                "held_out_seed": seed,
                "comparison_count": len(seed_rows),
                "exact_action_accuracy": _safe_rate(
                    sum(
                        1
                        for item in seed_rows
                        if item.get("exact_action_correct") is True
                    ),
                    len(seed_rows),
                ),
                "target_local_score_delta_mean": _round(
                    _mean(
                        [
                            _float(item.get("target_local_score_delta"))
                            for item in seed_rows
                        ]
                    )
                ),
                "terminal_alive_delta_mean": _round(
                    _mean([_float(item.get("terminal_alive_delta")) for item in seed_rows])
                ),
                "birth_delta_mean": _round(
                    _mean([_float(item.get("birth_delta")) for item in seed_rows])
                ),
                "target_alive_delta_negative_count": sum(
                    1 for item in seed_rows if _float(item.get("target_alive_delta")) < 0.0
                ),
            }
        )
    dominant = _dominant_count_share(predicted_counts)
    return {
        "rule": rule,
        "acceptance_candidate": bool(acceptance_candidate),
        "action_prior_penalty_scale": _round(action_prior_penalty_scale),
        "split_policy": "leave_one_source_seed_out_v1",
        "comparison_count": len(comparisons),
        "source_seed_count": len(source_seeds),
        "exact_action_correct_count": exact_correct,
        "exact_action_accuracy": _safe_rate(exact_correct, len(comparisons)),
        "unsupported_predicted_action_count": unsupported_count,
        "predicted_action_counts": dict(sorted(predicted_counts.items())),
        "predicted_mode_counts": dict(sorted(predicted_mode_counts.items())),
        "dominant_predicted_action": dominant["key"],
        "dominant_predicted_action_count": dominant["count"],
        "dominant_predicted_action_share": dominant["share"],
        "target_alive_delta_negative_count": sum(
            1 for item in comparisons if _float(item.get("target_alive_delta")) < 0.0
        ),
        "target_local_score_delta_mean": _round(
            _mean([_float(item.get("target_local_score_delta")) for item in comparisons])
        ),
        "terminal_alive_delta_mean": _round(
            _mean([_float(item.get("terminal_alive_delta")) for item in comparisons])
        ),
        "birth_delta_mean": _round(
            _mean([_float(item.get("birth_delta")) for item in comparisons])
        ),
        "by_seed": by_seed,
        "blocker_count": _loo_rule_blocker_count(
            dominant_predicted_action_share=dominant["share"],
            target_local_score_delta_mean=_mean(
                [_float(item.get("target_local_score_delta")) for item in comparisons]
            ),
            terminal_alive_delta_mean=_mean(
                [_float(item.get("terminal_alive_delta")) for item in comparisons]
            ),
            birth_delta_mean=_mean([_float(item.get("birth_delta")) for item in comparisons]),
            unsupported_predicted_action_count=unsupported_count,
        ),
        "examples": comparisons[:64],
    }


def _score_with_action_prior_balance(
    *,
    artifact: Mapping[str, object],
    support: Sequence[Mapping[str, object]],
    train_action_counts: Mapping[str, int],
    train_count: int,
    action_prior_penalty_scale: float,
    observation_input: Mapping[str, object],
    action_mask: Mapping[str, object],
    public_history_trace: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if action_prior_penalty_scale <= 0.0:
        return score_broad_residual_distillation_example_artifact(
            artifact=artifact,
            observation_input=observation_input,
            action_mask=action_mask,
            public_history_trace=public_history_trace,
        )
    from evolution_sim.mind.broad_branch_residual_distillation_example import (
        _nearest_support,
        _legal_actions,
    )
    from evolution_sim.mind.v3_planner_distilled import (
        candidate_feature_vector,
        planner_distilled_runtime_row,
    )

    row = planner_distilled_runtime_row(
        observation_input=observation_input,
        action_mask=action_mask,
        public_history_trace=public_history_trace,
    )
    candidate_scores: list[dict[str, object]] = []
    for action in _legal_actions(action_mask):
        features = candidate_feature_vector(row, action)
        if not features:
            continue
        distance, weight = _nearest_support(features, support, action=action)
        if math.isinf(distance):
            score = float("-inf")
        else:
            action_prior_share = _float(train_action_counts.get(action)) / float(
                max(train_count, 1)
            )
            score = (
                -distance
                + 0.0001 * weight
                - action_prior_penalty_scale * action_prior_share
            )
        candidate_scores.append(
            {
                "action": action,
                "mode": _action_option_mode(action),
                "nearest_support_distance": _round(distance)
                if math.isfinite(distance)
                else None,
                "support_weight": _round(weight),
                "score": _round(score) if math.isfinite(score) else None,
            }
        )
    if not candidate_scores:
        return {
            "selected_action": None,
            "candidate_scores": [],
            "unsupported_predicted_action": True,
        }
    selected = max(
        candidate_scores,
        key=lambda item: (
            _float(item.get("score"), default=float("-inf")),
            str(item.get("action", "")),
        ),
    )
    return {
        "selected_action": selected.get("action"),
        "selected_mode": selected.get("mode"),
        "candidate_scores": candidate_scores,
        "unsupported_predicted_action": selected.get("action") not in _list(
            [action for action in ACTION_NAMES if bool(action_mask.get(action, False))]
        ),
    }


def _loo_rule_blocker_count(
    *,
    dominant_predicted_action_share: float,
    target_local_score_delta_mean: float,
    terminal_alive_delta_mean: float,
    birth_delta_mean: float,
    unsupported_predicted_action_count: int,
) -> int:
    blockers = 0
    if unsupported_predicted_action_count != 0:
        blockers += 1
    if dominant_predicted_action_share > V102_MAX_LOO_DOMINANT_PREDICTED_ACTION_SHARE:
        blockers += 1
    if target_local_score_delta_mean <= 0.0:
        blockers += 1
    if terminal_alive_delta_mean < 0.0:
        blockers += 1
    if birth_delta_mean < 0.0:
        blockers += 1
    return blockers


def _acceptance(
    *,
    coverage: Mapping[str, object],
    training_eval: Mapping[str, object],
    loo_eval: Mapping[str, object],
    reload_identical: bool,
    floors: Mapping[str, object],
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []

    def block(
        reason: str,
        field: str,
        observed: object,
        required: object,
        comparator: str,
    ) -> None:
        blockers.append(
            {
                "reason": reason,
                "field": field,
                "observed": observed,
                "required": required,
                "comparator": comparator,
            }
        )

    if _int(coverage.get("training_row_count")) < _int(floors.get("training_row_count")):
        block("insufficient_training_rows", "training_row_count", coverage.get("training_row_count"), floors.get("training_row_count"), "ge")
    if _int(coverage.get("source_seed_count")) < _int(floors.get("source_seed_count")):
        block("insufficient_source_seed_count", "source_seed_count", coverage.get("source_seed_count"), floors.get("source_seed_count"), "ge")
    if _int(coverage.get("strict_seed_leak_count")) != 0:
        block("strict_seed_leakage", "strict_seed_leak_count", coverage.get("strict_seed_leak_count"), 0, "eq")
    if coverage.get("source_replay_verified") is not True:
        block("source_replay_not_verified", "source_replay_verified", coverage.get("source_replay_verified"), True, "eq")
    if _int(coverage.get("source_unsupported_candidate_action_count")) != 0:
        block("unsupported_candidate_action", "source_unsupported_candidate_action_count", coverage.get("source_unsupported_candidate_action_count"), 0, "eq")
    if _int(training_eval.get("unsupported_predicted_action_count")) != 0:
        block("unsupported_training_prediction", "training_eval.unsupported_predicted_action_count", training_eval.get("unsupported_predicted_action_count"), 0, "eq")
    if _float(coverage.get("dominant_teacher_action_share")) > _float(floors.get("dominant_teacher_action_share_max")):
        block("dominant_teacher_action_share_above_cap", "dominant_teacher_action_share", coverage.get("dominant_teacher_action_share"), floors.get("dominant_teacher_action_share_max"), "le")
    if _int(coverage.get("represented_teacher_mode_count")) < _int(floors.get("represented_teacher_mode_count")):
        block("insufficient_teacher_mode_count", "represented_teacher_mode_count", coverage.get("represented_teacher_mode_count"), floors.get("represented_teacher_mode_count"), "ge")
    if _float(coverage.get("movement_reposition_label_share")) < _float(floors.get("reposition_label_share")):
        block("insufficient_reposition_label_share", "movement_reposition_label_share", coverage.get("movement_reposition_label_share"), floors.get("reposition_label_share"), "ge")
    missing_contexts = _list(coverage.get("missing_required_branch_contexts"))
    if missing_contexts:
        block("missing_required_branch_contexts", "missing_required_branch_contexts", missing_contexts, floors.get("required_branch_contexts"), "empty")
    if _int(coverage.get("target_alive_delta_negative_count")) != 0:
        block("target_alive_delta_negative", "target_alive_delta_negative_count", coverage.get("target_alive_delta_negative_count"), 0, "eq")
    if _float(coverage.get("target_local_score_delta_mean")) <= 0.0:
        block("mean_target_local_score_delta_not_positive", "target_local_score_delta_mean", coverage.get("target_local_score_delta_mean"), 0.0, "gt")
    if _float(coverage.get("terminal_alive_delta_mean")) < 0.0:
        block("mean_terminal_alive_delta_negative", "terminal_alive_delta_mean", coverage.get("terminal_alive_delta_mean"), 0.0, "ge")
    if _float(coverage.get("birth_delta_mean")) < 0.0:
        block("mean_birth_delta_negative", "birth_delta_mean", coverage.get("birth_delta_mean"), 0.0, "ge")
    if not reload_identical:
        block("artifact_reload_changed_choices", "artifact_reload_identical_choices", reload_identical, True, "eq")
    if _int(loo_eval.get("unsupported_predicted_action_count")) != 0:
        block("loo_unsupported_predicted_action", "loo.unsupported_predicted_action_count", loo_eval.get("unsupported_predicted_action_count"), 0, "eq")
    if _float(loo_eval.get("dominant_predicted_action_share")) > _float(floors.get("loo_dominant_predicted_action_share_max")):
        block("loo_dominant_predicted_action_share_above_cap", "loo.dominant_predicted_action_share", loo_eval.get("dominant_predicted_action_share"), floors.get("loo_dominant_predicted_action_share_max"), "le")
    if _float(loo_eval.get("target_local_score_delta_mean")) <= 0.0:
        block("loo_mean_target_local_score_delta_not_positive", "loo.target_local_score_delta_mean", loo_eval.get("target_local_score_delta_mean"), 0.0, "gt")
    if _float(loo_eval.get("terminal_alive_delta_mean")) < 0.0:
        block("loo_mean_terminal_alive_delta_negative", "loo.terminal_alive_delta_mean", loo_eval.get("terminal_alive_delta_mean"), 0.0, "ge")
    if _float(loo_eval.get("birth_delta_mean")) < 0.0:
        block("loo_mean_birth_delta_negative", "loo.birth_delta_mean", loo_eval.get("birth_delta_mean"), 0.0, "ge")
    accepted = not blockers
    return {
        "v102_expanded_broad_residual_training_accepted": accepted,
        "v103_support_gated_residual_runtime_allowed": accepted,
        "runtime_promotion_allowed": False,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "recommendation": (
            "Proceed to v103 opt-in support-gated residual runtime feasibility."
            if accepted
            else "Do not train or wire v103 runtime from this dataset."
        ),
    }


def _validate_source_reports(
    *,
    v99_report: Mapping[str, object],
    v100_report: Mapping[str, object],
) -> None:
    if (
        v99_report.get("schema_version")
        != MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_AUDIT_SCHEMA_VERSION
    ):
        raise ExpandedBroadResidualTrainingError(
            "v99 expanded oracle report has unsupported schema_version"
        )
    if (
        v100_report.get("schema_version")
        != MIND_V3_V100_BROAD_BRANCH_RESIDUAL_CONSTRAINED_AUDIT_SCHEMA_VERSION
    ):
        raise ExpandedBroadResidualTrainingError(
            "v100 constrained report has unsupported schema_version"
        )
    if (
        v100_report.get("v100_broad_branch_residual_constrained_diagnostic_accepted")
        is not True
    ):
        raise ExpandedBroadResidualTrainingError(
            "constrained report must be accepted before v102"
        )


def _candidate_by_action(
    candidates: Sequence[Mapping[str, object]],
    action: str,
) -> Mapping[str, object] | None:
    for candidate in candidates:
        if str(candidate.get("action", "")) == action:
            return candidate
    return None


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


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return int(value)
    return 0


def _float(value: object, *, default: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    parsed = float(value)
    return parsed if math.isfinite(parsed) else default


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    denom = float(denominator)
    if denom <= 0.0:
        return 0.0
    return _round(float(numerator) / denom)


def _round(value: float) -> float:
    return round(float(value), 6)


def _open_output(path: Path) -> TextIO:
    return path.open("w", encoding="utf-8")
