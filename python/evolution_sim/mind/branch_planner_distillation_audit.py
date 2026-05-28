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
    _squared_distance,
)
from evolution_sim.mind.branch_constrained_planning_audit import (
    MIND_V3_BRANCH_CONSTRAINED_PLANNING_AUDIT_SCHEMA_VERSION,
    V95_MAX_DOMINANT_PREDICTED_ACTION_SHARE,
    _beam_global_assignment,
    _candidate_outcomes,
)
from evolution_sim.mind.branch_mode_objective_audit import (
    _action_option_mode,
    _list_of_mappings,
    _mapping,
    _safe_rate,
)
from evolution_sim.mind.branch_sequence_continuation_scorer import (
    MIND_V3_BRANCH_SEQUENCE_CONTINUATION_SCORER_SCHEMA_VERSION,
    V94_K,
    V94_SEED_41_TICK_113_BRANCH_ID,
    V94_SEED_41_TICK_114_BRANCH_ID,
    _predicted_sequence_stats,
    _sequence_training_examples,
)
from evolution_sim.mind.branch_utility_risk_audit import (
    _candidate_actions,
    _candidate_feature_vector,
    _field_summary,
    _float,
    _int,
    _utility_rows,
)
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.v3_planner_distilled import (
    MindV3PlannerDistilledArtifactError,
    artifact_has_forbidden_example_keys as _runtime_artifact_has_forbidden_example_keys,
    score_distilled_planner_artifact as _runtime_score_distilled_planner_artifact,
    validate_mind_v3_planner_distilled_artifact,
)

MIND_V3_PLANNER_DISTILLATION_AUDIT_SCHEMA_VERSION = (
    "mind_v3_planner_distillation_runtime_feasibility_v1"
)
MIND_V3_PLANNER_DISTILLED_ARTIFACT_SCHEMA_VERSION = (
    "mind_v3_planner_distilled_action_scorer_artifact_v1"
)
V96_STRICT_EVAL_SEEDS: tuple[int, ...] = (13, 19, 29, 37, 41, 43)
V96_MIN_STRICT_COMPARISON_COUNT = 48
V96_MAX_DOMINANT_PREDICTED_ACTION_SHARE = 0.50
V96_MAX_DOMINANT_PREDICTED_MODE_SHARE = 0.75
V96_MAX_TARGET_ALIVE_NEGATIVE_COUNT = 0
V96_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA = 0.0
V96_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA = 0.0
V96_MIN_MEAN_TERMINAL_ALIVE_DELTA = 0.0
V96_MIN_MEAN_BIRTH_DELTA = 0.0
V96_TEACHER_DIVERSITY_PENALTY = 2.0
V96_ANTI_COLLAPSE_STRENGTH = 2.0
V96_TEACHER_IMITATION_K = 5
V96_TEACHER_IMITATION_WEIGHT = 0.05
V96_ACCEPTED_RULE = "distilled_sequence_cvar_action_prior_scorer_v1"
V96_INVALID_STRICT_CONTROL_RULE = "invalid_train_on_strict_teacher_control"


class BranchPlannerDistillationAuditError(ValueError):
    pass


def build_branch_planner_distillation_audit_report(
    *,
    support_branch_action_oracle_labels: Mapping[str, object],
    strict_branch_action_oracle_labels: Mapping[str, object],
    branch_sequence_continuation_scorer_report: Mapping[str, object],
    branch_constrained_planning_audit_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    _validate_inputs(
        support_branch_action_oracle_labels=support_branch_action_oracle_labels,
        strict_branch_action_oracle_labels=strict_branch_action_oracle_labels,
        branch_sequence_continuation_scorer_report=(
            branch_sequence_continuation_scorer_report
        ),
        branch_constrained_planning_audit_report=(
            branch_constrained_planning_audit_report
        ),
    )
    support_labels = _list_of_mappings(
        support_branch_action_oracle_labels.get("labels"),
        "support.labels",
    )
    strict_labels = _list_of_mappings(
        strict_branch_action_oracle_labels.get("labels"),
        "strict.labels",
    )
    support_rows = _utility_rows(support_labels)
    strict_rows = _utility_rows(strict_labels)
    support_seeds = {int(row.get("seed", -1)) for row in support_rows}
    strict_seed_overlap = sorted(support_seeds & set(V96_STRICT_EVAL_SEEDS))
    if strict_seed_overlap:
        raise BranchPlannerDistillationAuditError(
            "support training rows include strict held-out seeds: "
            f"{strict_seed_overlap}"
        )

    support_candidate_outcomes = _candidate_outcomes(support_rows)
    strict_candidate_outcomes = _candidate_outcomes(strict_rows)
    support_teacher_assignment = _support_teacher_assignment(
        support_rows,
        candidate_outcomes=support_candidate_outcomes,
    )
    support_sequence_examples = _sequence_training_examples(support_rows)
    support_loo_predictions = _sequence_predictions(
        rows=support_rows,
        support_examples=support_sequence_examples,
        leave_one_source_seed_out=True,
    )
    strict_predictions = _sequence_predictions(
        rows=strict_rows,
        support_examples=support_sequence_examples,
        leave_one_source_seed_out=False,
    )
    support_baseline_actions = _select_sequence_cvar_actions(
        rows=support_rows,
        candidate_predictions=support_loo_predictions,
        action_penalties={},
        teacher_imitation_examples=[],
        teacher_imitation_weight=0.0,
    )
    distilled_artifact = _fit_distilled_artifact(
        support_rows=support_rows,
        support_sequence_examples=support_sequence_examples,
        support_teacher_assignment=support_teacher_assignment,
        support_baseline_actions=support_baseline_actions,
        support_candidate_outcomes=support_candidate_outcomes,
    )
    reloaded_artifact = json.loads(json.dumps(distilled_artifact, sort_keys=True))
    support_teacher_report = _teacher_report(
        support_rows=support_rows,
        assignment=support_teacher_assignment,
        support_baseline_actions=support_baseline_actions,
    )
    support_fit_report = _support_fit_report(
        support_rows=support_rows,
        candidate_outcomes=support_candidate_outcomes,
        support_teacher_assignment=support_teacher_assignment,
        original_artifact=distilled_artifact,
        reloaded_artifact=reloaded_artifact,
    )
    strict_report = _evaluate_artifact_rule(
        rows=strict_rows,
        candidate_outcomes=strict_candidate_outcomes,
        artifact=reloaded_artifact,
        rule=V96_ACCEPTED_RULE,
    )
    v94_baseline_report = _v94_baseline_report(
        branch_sequence_continuation_scorer_report
    )
    v95_teacher_summary = _v95_teacher_summary(branch_constrained_planning_audit_report)
    invalid_control = _invalid_strict_teacher_control(
        strict_rows=strict_rows,
        strict_candidate_outcomes=strict_candidate_outcomes,
    )
    runtime_reload = _runtime_reload_check(
        rows=strict_rows,
        original_artifact=distilled_artifact,
        reloaded_artifact=reloaded_artifact,
    )
    coverage = _coverage_report(
        support_labels=support_branch_action_oracle_labels,
        strict_labels=strict_branch_action_oracle_labels,
        support_rows=support_rows,
        strict_rows=strict_rows,
        support_loo_predictions=support_loo_predictions,
        strict_predictions=strict_predictions,
        runtime_reload=runtime_reload,
        artifact=distilled_artifact,
    )
    acceptance = _acceptance(coverage=coverage, strict_report=strict_report)
    contract = {
        "schema_version": MIND_V3_PLANNER_DISTILLATION_AUDIT_SCHEMA_VERSION,
        "source_support_label_schema_version": (
            support_branch_action_oracle_labels.get("schema_version")
        ),
        "strict_eval_label_schema_version": strict_branch_action_oracle_labels.get(
            "schema_version"
        ),
        "source_sequence_scorer_schema_version": (
            branch_sequence_continuation_scorer_report.get("schema_version")
        ),
        "source_constrained_planning_schema_version": (
            branch_constrained_planning_audit_report.get("schema_version")
            if branch_constrained_planning_audit_report is not None
            else None
        ),
        "runtime_policy_trained": False,
        "runtime_feasibility_artifact_serialized": True,
        "runtime_feasibility_accepted_is_not_promotion": True,
        "strict_eval_seeds": list(V96_STRICT_EVAL_SEEDS),
        "split_policy": "train_non_strict_support_eval_strict_carrion_v1",
        "teacher_policy": (
            "support_only_diversity_regularized_constrained_planner_v1"
        ),
        "student_policy": V96_ACCEPTED_RULE,
        "feature_contract": {
            "schema_version": "v96_policy_visible_sequence_prefix_features_v1",
            "uses_private_world_state": False,
            "uses_fixture_identity": False,
            "uses_seed_id_as_runtime_feature": False,
            "uses_branch_id_as_runtime_feature": False,
            "uses_logged_action_as_runtime_fallback": False,
            "uses_strict_eval_label_identity": False,
            "requires_planner_outcome_tables_at_inference": False,
            "requires_global_batch_assignment_or_quota_at_inference": False,
            "features": [
                "decoded policy observation compact state",
                "action mask",
                "same-agent public history trace",
                "candidate action identity and movement direction",
                "candidate action support flag",
            ],
        },
        "target_contract": {
            "teacher_labels_from_strict_eval_rows": False,
            "teacher_labels_from_non_strict_support_rows": True,
            "utility_weighted_teacher_imitation": True,
            "anti_collapse_regularization_source": (
                "support baseline action share minus support teacher action share"
            ),
        },
        "support_floors": {
            "no_strict_held_out_leakage": True,
            "replay_verified": True,
            "heuristic_source_count": 0,
            "unsupported_predicted_logged_or_oracle_action_count": 0,
            "strict_comparison_count": V96_MIN_STRICT_COMPARISON_COUNT,
            "dominant_predicted_action_share_max": (
                V96_MAX_DOMINANT_PREDICTED_ACTION_SHARE
            ),
            "dominant_predicted_mode_share_max": (
                V96_MAX_DOMINANT_PREDICTED_MODE_SHARE
            ),
            "target_alive_delta_negative_count": (
                V96_MAX_TARGET_ALIVE_NEGATIVE_COUNT
            ),
            "global_mean_target_local_score_delta_gt": (
                V96_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA
            ),
            "per_seed_target_local_mean_min": (
                V96_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA
            ),
            "mean_terminal_alive_delta_min": V96_MIN_MEAN_TERMINAL_ALIVE_DELTA,
            "mean_birth_delta_min": V96_MIN_MEAN_BIRTH_DELTA,
            "seed_41_tick_113_catastrophe_avoided": True,
            "seed_41_tick_114_catastrophe_avoided": True,
            "artifact_reload_tested": True,
        },
    }
    accepted = bool(acceptance.get("v96_runtime_feasibility_accepted"))
    return {
        "schema_version": MIND_V3_PLANNER_DISTILLATION_AUDIT_SCHEMA_VERSION,
        "contract": contract,
        "provenance": {
            "support_label_digest": stable_payload_digest(
                {
                    "schema_version": support_branch_action_oracle_labels.get(
                        "schema_version"
                    ),
                    "aggregate": support_branch_action_oracle_labels.get(
                        "aggregate"
                    ),
                    "labels": support_labels,
                }
            ),
            "strict_label_digest": stable_payload_digest(
                {
                    "schema_version": strict_branch_action_oracle_labels.get(
                        "schema_version"
                    ),
                    "aggregate": strict_branch_action_oracle_labels.get("aggregate"),
                    "labels": strict_labels,
                }
            ),
            "source_sequence_scorer_digest": stable_payload_digest(
                {
                    "schema_version": branch_sequence_continuation_scorer_report.get(
                        "schema_version"
                    ),
                    "coverage": branch_sequence_continuation_scorer_report.get(
                        "coverage"
                    ),
                    "sequence_continuation_dataset": (
                        branch_sequence_continuation_scorer_report.get(
                            "sequence_continuation_dataset"
                        )
                    ),
                }
            ),
            "source_constrained_planning_digest": (
                stable_payload_digest(
                    {
                        "schema_version": branch_constrained_planning_audit_report.get(
                            "schema_version"
                        ),
                        "coverage": branch_constrained_planning_audit_report.get(
                            "coverage"
                        ),
                        "acceptance": branch_constrained_planning_audit_report.get(
                            "acceptance"
                        ),
                    }
                )
                if branch_constrained_planning_audit_report is not None
                else None
            ),
            "distilled_artifact_digest": stable_payload_digest(distilled_artifact),
            "contract_digest": stable_payload_digest(contract),
        },
        "coverage": coverage,
        "support_teacher_labels": support_teacher_report,
        "support_fit_diagnostics": support_fit_report,
        "distilled_artifact": distilled_artifact,
        "runtime_reload_check": runtime_reload,
        "strict_branch_replay_evaluation": strict_report,
        "comparisons": {
            "v94_sequence_baseline": v94_baseline_report,
            "v95_teacher_upper_bound": v95_teacher_summary,
            "v96_distilled_runtime_feasible_scorer": strict_report,
            "invalid_train_on_strict_labels_control": invalid_control,
        },
        "planner_distillation_runtime_feasibility_support_probe": {
            "policy": "v96_planner_distillation_runtime_feasibility_v1",
            "accuracy": 1.0 if accepted else 0.0,
            "support_accuracy_floor": 1.0,
            "materially_supports_runtime_feasible_planner_distillation": accepted,
            "runtime_policy_status": (
                "runtime_feasibility_only_not_promoted"
                if accepted
                else "diagnostic_failed_no_runtime_training"
            ),
        },
        "acceptance": acceptance,
    }


def load_json_report(path: str | Path) -> dict[str, object]:
    resolved = Path(path)
    try:
        with _open_input(resolved) as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise BranchPlannerDistillationAuditError(
            f"failed to read report: {resolved}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BranchPlannerDistillationAuditError(
            f"report is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise BranchPlannerDistillationAuditError("report must be a JSON object")
    return payload


def write_branch_planner_distillation_audit_report(
    report: Mapping[str, object],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_output(path) as handle:
        json.dump(report, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def score_distilled_planner_artifact(
    *,
    row: Mapping[str, object],
    artifact: Mapping[str, object],
) -> dict[str, object]:
    return _runtime_score_distilled_planner_artifact(row=row, artifact=artifact)


def _validate_inputs(
    *,
    support_branch_action_oracle_labels: Mapping[str, object],
    strict_branch_action_oracle_labels: Mapping[str, object],
    branch_sequence_continuation_scorer_report: Mapping[str, object],
    branch_constrained_planning_audit_report: Mapping[str, object] | None,
) -> None:
    if (
        support_branch_action_oracle_labels.get("schema_version")
        != MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION
    ):
        raise BranchPlannerDistillationAuditError(
            "support branch action oracle labels have unsupported schema_version"
        )
    if (
        strict_branch_action_oracle_labels.get("schema_version")
        != MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION
    ):
        raise BranchPlannerDistillationAuditError(
            "strict branch action oracle labels have unsupported schema_version"
        )
    if (
        branch_sequence_continuation_scorer_report.get("schema_version")
        != MIND_V3_BRANCH_SEQUENCE_CONTINUATION_SCORER_SCHEMA_VERSION
    ):
        raise BranchPlannerDistillationAuditError(
            "sequence continuation scorer report has unsupported schema_version"
        )
    if (
        branch_constrained_planning_audit_report is not None
        and branch_constrained_planning_audit_report.get("schema_version")
        != MIND_V3_BRANCH_CONSTRAINED_PLANNING_AUDIT_SCHEMA_VERSION
    ):
        raise BranchPlannerDistillationAuditError(
            "constrained planning audit report has unsupported schema_version"
        )


def _support_teacher_assignment(
    rows: Sequence[Mapping[str, object]],
    *,
    candidate_outcomes: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    return _beam_global_assignment(
        rows,
        candidate_outcomes=candidate_outcomes,
        max_action_count=int(
            math.floor(float(len(rows)) * V95_MAX_DOMINANT_PREDICTED_ACTION_SHARE)
        ),
        diversity_penalty=V96_TEACHER_DIVERSITY_PENALTY,
    )


def _sequence_predictions(
    *,
    rows: Sequence[Mapping[str, object]],
    support_examples: Sequence[Mapping[str, object]],
    leave_one_source_seed_out: bool,
) -> dict[str, list[dict[str, object]]]:
    predictions: dict[str, list[dict[str, object]]] = {}
    strict_seed_set = set(V96_STRICT_EVAL_SEEDS)
    for row in rows:
        branch_id = str(row.get("branch_id", ""))
        row_seed = row.get("seed")
        training = [
            item
            for item in support_examples
            if not leave_one_source_seed_out or item.get("seed") != row_seed
        ]
        branch_predictions = []
        for candidate in _candidate_actions(row):
            action = str(candidate.get("action", ""))
            features = _candidate_feature_vector(row, action)
            if not features:
                continue
            neighbors = _nearest_sequence_examples(features, training, k=V94_K)
            if not neighbors:
                continue
            branch_predictions.append(
                {
                    "branch_id": branch_id,
                    "seed": row_seed,
                    "action": action,
                    "mode": _action_option_mode(action),
                    "neighbor_count": len(neighbors),
                    "neighbor_seed_counts": dict(
                        sorted(
                            Counter(
                                str(item[1].get("seed")) for item in neighbors
                            ).items()
                        )
                    ),
                    "uses_strict_seed_training": any(
                        item[1].get("seed") in strict_seed_set
                        for item in neighbors
                    ),
                    "predicted": _predicted_sequence_stats(neighbors),
                }
            )
        predictions[branch_id] = sorted(
            branch_predictions,
            key=lambda item: str(item.get("action", "")),
        )
    return predictions


def _nearest_sequence_examples(
    features: Sequence[float],
    examples: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> list[tuple[float, Mapping[str, object]]]:
    neighbors = [
        (_squared_distance(features, _tuple(item.get("features"))), item)
        for item in examples
        if item.get("features")
    ]
    neighbors.sort(
        key=lambda item: (
            item[0],
            str(item[1].get("branch_id", "")),
            str(item[1].get("action", "")),
        )
    )
    return neighbors[: max(1, int(k))]


def _select_sequence_cvar_actions(
    *,
    rows: Sequence[Mapping[str, object]],
    candidate_predictions: Mapping[str, Sequence[Mapping[str, object]]],
    action_penalties: Mapping[str, float],
    teacher_imitation_examples: Sequence[Mapping[str, object]],
    teacher_imitation_weight: float,
) -> dict[str, str]:
    actions = {}
    for row in rows:
        branch_id = str(row.get("branch_id", ""))
        selected = _select_action_from_prediction_scores(
            row=row,
            candidate_predictions=list(candidate_predictions.get(branch_id, [])),
            action_penalties=action_penalties,
            teacher_imitation_examples=teacher_imitation_examples,
            teacher_imitation_weight=teacher_imitation_weight,
        )
        if selected is not None:
            actions[branch_id] = selected
    return actions


def _select_action_from_prediction_scores(
    *,
    row: Mapping[str, object],
    candidate_predictions: Sequence[Mapping[str, object]],
    action_penalties: Mapping[str, float],
    teacher_imitation_examples: Sequence[Mapping[str, object]],
    teacher_imitation_weight: float,
) -> str | None:
    if not candidate_predictions:
        return None
    scores = []
    for item in candidate_predictions:
        action = str(item.get("action", ""))
        predicted = _mapping(item.get("predicted"))
        sequence_cvar = _float(predicted.get("target_local_sequence_score_cvar_25"))
        imitation = _teacher_imitation_margin(
            _candidate_feature_vector(row, action),
            teacher_imitation_examples,
            k=V96_TEACHER_IMITATION_K,
        )
        score = (
            sequence_cvar
            + float(teacher_imitation_weight) * imitation
            - _float(action_penalties.get(action))
        )
        scores.append((_round(score), _round(sequence_cvar), action))
    selected = max(scores, key=lambda item: (item[0], item[1], item[2]))
    return selected[2]


def _fit_distilled_artifact(
    *,
    support_rows: Sequence[Mapping[str, object]],
    support_sequence_examples: Sequence[Mapping[str, object]],
    support_teacher_assignment: Sequence[Mapping[str, object]],
    support_baseline_actions: Mapping[str, str],
    support_candidate_outcomes: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    teacher_by_branch = {
        str(item.get("branch_id", "")): item for item in support_teacher_assignment
    }
    teacher_counts = Counter(
        str(item.get("predicted_action", "")) for item in support_teacher_assignment
    )
    baseline_counts = Counter(str(action) for action in support_baseline_actions.values())
    total = float(max(1, len(support_rows)))
    action_penalties = {
        action: _round(
            V96_ANTI_COLLAPSE_STRENGTH
            * max(0.0, (baseline_counts[action] - teacher_counts[action]) / total)
        )
        for action in ACTION_NAMES
    }
    teacher_examples = _teacher_imitation_examples(
        support_rows=support_rows,
        teacher_by_branch=teacher_by_branch,
        support_candidate_outcomes=support_candidate_outcomes,
    )
    sequence_examples = [
        {
            "example_index": index,
            "action": str(item.get("action", "")),
            "mode": str(item.get("mode", "")),
            "features": _round_sequence(_tuple(item.get("features"))),
            "metrics": _mapping(item.get("metrics")),
        }
        for index, item in enumerate(
            sorted(
                support_sequence_examples,
                key=lambda value: (
                    str(value.get("seed", "")),
                    str(value.get("branch_id", "")),
                    str(value.get("action", "")),
                ),
            )
        )
    ]
    return {
        "schema_version": MIND_V3_PLANNER_DISTILLED_ARTIFACT_SCHEMA_VERSION,
        "model_type": "deterministic_sequence_cvar_action_prior_scorer_v1",
        "runtime_feasibility_only": True,
        "promotion_ready": False,
        "training_split_policy": "non_strict_support_only_v1",
        "strict_eval_seeds_excluded_from_training": list(V96_STRICT_EVAL_SEEDS),
        "runtime_forbidden_inputs": [
            "seed_id",
            "branch_id",
            "fixture_identity",
            "logged_action",
            "strict_eval_label_identity",
            "planner_candidate_outcome_table",
            "global_batch_action_quota",
            "private_simulation_world_state",
        ],
        "inference_contract": {
            "one_row_one_agent_local_decision": True,
            "requires_action_mask": True,
            "requires_policy_visible_features_only": True,
            "requires_planner_outcome_tables": False,
            "requires_global_batch_assignment": False,
            "uses_heuristic_fallback": False,
        },
        "scoring_policy": {
            "sequence_neighbor_count": V94_K,
            "teacher_imitation_neighbor_count": V96_TEACHER_IMITATION_K,
            "teacher_imitation_weight": V96_TEACHER_IMITATION_WEIGHT,
            "sequence_score_field": "target_local_sequence_score_cvar_25",
            "anti_collapse_strength": V96_ANTI_COLLAPSE_STRENGTH,
            "final_score": (
                "sequence_cvar + teacher_imitation_weight * "
                "utility_weighted_teacher_margin - learned_action_penalty"
            ),
        },
        "learned_action_penalties": dict(sorted(action_penalties.items())),
        "action_penalty_derivation": {
            "support_teacher_action_counts": dict(sorted(teacher_counts.items())),
            "support_baseline_action_counts": dict(sorted(baseline_counts.items())),
            "policy": (
                "anti_collapse_strength * max(0, "
                "baseline_action_share - teacher_action_share)"
            ),
        },
        "sequence_support_examples": sequence_examples,
        "teacher_imitation_examples": teacher_examples,
    }


def _teacher_imitation_examples(
    *,
    support_rows: Sequence[Mapping[str, object]],
    teacher_by_branch: Mapping[str, Mapping[str, object]],
    support_candidate_outcomes: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    examples = []
    for row in sorted(support_rows, key=lambda item: str(item.get("branch_id", ""))):
        branch_id = str(row.get("branch_id", ""))
        teacher = _mapping(teacher_by_branch.get(branch_id))
        teacher_action = str(teacher.get("predicted_action", ""))
        teacher_utility = max(0.0, _float(teacher.get("target_local_score_delta")))
        utility_weight = 1.0 + min(5.0, teacher_utility / 100.0)
        for candidate in _candidate_actions(row):
            action = str(candidate.get("action", ""))
            features = _candidate_feature_vector(row, action)
            if not features:
                continue
            candidate_outcome = _candidate_outcome(
                support_candidate_outcomes.get(branch_id, []),
                action,
            )
            examples.append(
                {
                    "example_index": len(examples),
                    "action": action,
                    "mode": _action_option_mode(action),
                    "features": _round_sequence(features),
                    "teacher_selected": action == teacher_action,
                    "utility_weight": _round(utility_weight),
                    "candidate_target_local_delta": (
                        _mapping(candidate_outcome).get("target_local_score_delta")
                    ),
                }
            )
    return examples


def _artifact_candidate_scores(
    *,
    row: Mapping[str, object],
    artifact: Mapping[str, object],
) -> list[dict[str, object]]:
    sequence_examples = _list_of_mappings(
        artifact.get("sequence_support_examples"),
        "artifact.sequence_support_examples",
    )
    teacher_examples = _list_of_mappings(
        artifact.get("teacher_imitation_examples"),
        "artifact.teacher_imitation_examples",
    )
    action_penalties = _mapping(artifact.get("learned_action_penalties"))
    scoring = _mapping(artifact.get("scoring_policy"))
    teacher_weight = _float(scoring.get("teacher_imitation_weight"))
    scores = []
    for candidate in _candidate_actions(row):
        action = str(candidate.get("action", ""))
        features = _candidate_feature_vector(row, action)
        if not features:
            continue
        sequence_neighbors = _nearest_artifact_examples(
            features,
            sequence_examples,
            k=_int(scoring.get("sequence_neighbor_count")) or V94_K,
        )
        if not sequence_neighbors:
            continue
        sequence_predicted = _predicted_sequence_stats(sequence_neighbors)
        sequence_cvar = _float(
            sequence_predicted.get("target_local_sequence_score_cvar_25")
        )
        imitation_margin = _teacher_imitation_margin(
            features,
            teacher_examples,
            k=_int(scoring.get("teacher_imitation_neighbor_count"))
            or V96_TEACHER_IMITATION_K,
        )
        action_penalty = _float(action_penalties.get(action))
        final_score = sequence_cvar + teacher_weight * imitation_margin - action_penalty
        scores.append(
            {
                "action": action,
                "mode": _action_option_mode(action),
                "sequence_cvar_score": _round(sequence_cvar),
                "utility_weighted_teacher_margin": _round(imitation_margin),
                "learned_action_penalty": _round(action_penalty),
                "final_score": _round(final_score),
                "sequence_neighbor_count": len(sequence_neighbors),
            }
        )
    return sorted(scores, key=lambda item: str(item.get("action", "")))


def _nearest_artifact_examples(
    features: Sequence[float],
    examples: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> list[tuple[float, Mapping[str, object]]]:
    neighbors = [
        (_squared_distance(features, _tuple(item.get("features"))), item)
        for item in examples
        if item.get("features")
    ]
    neighbors.sort(
        key=lambda item: (
            item[0],
            _int(item[1].get("example_index")),
            str(item[1].get("action", "")),
        )
    )
    return neighbors[: max(1, int(k))]


def _teacher_imitation_margin(
    features: Sequence[float],
    examples: Sequence[Mapping[str, object]],
    *,
    k: int,
) -> float:
    if not examples:
        return 0.0
    neighbors = _nearest_artifact_examples(features, examples, k=k)
    weighted_total = 0.0
    weight_sum = 0.0
    for distance, item in neighbors:
        distance_weight = 1.0 / (1.0 + float(distance))
        utility_weight = max(0.0, _float(item.get("utility_weight")))
        signed = 1.0 if item.get("teacher_selected") is True else -1.0
        weight = distance_weight * utility_weight
        weighted_total += weight * signed
        weight_sum += weight
    return _round(weighted_total / weight_sum) if weight_sum else 0.0


def _support_fit_report(
    *,
    support_rows: Sequence[Mapping[str, object]],
    candidate_outcomes: Mapping[str, Sequence[Mapping[str, object]]],
    support_teacher_assignment: Sequence[Mapping[str, object]],
    original_artifact: Mapping[str, object],
    reloaded_artifact: Mapping[str, object],
) -> dict[str, object]:
    original = _evaluate_artifact_rule(
        rows=support_rows,
        candidate_outcomes=candidate_outcomes,
        artifact=original_artifact,
        rule="support_fit_original_artifact",
    )
    reloaded = _evaluate_artifact_rule(
        rows=support_rows,
        candidate_outcomes=candidate_outcomes,
        artifact=reloaded_artifact,
        rule="support_fit_reloaded_artifact",
    )
    teacher_by_branch = {
        str(item.get("branch_id", "")): str(item.get("predicted_action", ""))
        for item in support_teacher_assignment
    }
    predictions = {
        str(item.get("branch_id", "")): str(item.get("predicted_action", ""))
        for item in _list_of_mappings(original.get("comparisons"), "comparisons")
    }
    correct = sum(
        1
        for branch_id, action in predictions.items()
        if teacher_by_branch.get(branch_id) == action
    )
    return {
        "support_comparison_count": original.get("comparison_count"),
        "support_teacher_exact_action_accuracy": _safe_rate(
            correct,
            len(support_rows),
        ),
        "original_and_reloaded_actions_match": (
            original.get("predicted_action_counts")
            == reloaded.get("predicted_action_counts")
            and [
                item.get("predicted_action")
                for item in _list_of_mappings(original.get("comparisons"), "comparisons")
            ]
            == [
                item.get("predicted_action")
                for item in _list_of_mappings(reloaded.get("comparisons"), "comparisons")
            ]
        ),
        "support_rule_report": _strip_comparisons(original),
    }


def _evaluate_artifact_rule(
    *,
    rows: Sequence[Mapping[str, object]],
    candidate_outcomes: Mapping[str, Sequence[Mapping[str, object]]],
    artifact: Mapping[str, object],
    rule: str,
) -> dict[str, object]:
    comparisons = []
    missing = 0
    unsupported = 0
    runtime_scores = []
    for row in rows:
        scored = score_distilled_planner_artifact(row=row, artifact=artifact)
        action = scored.get("selected_action")
        runtime_scores.append(
            {
                "branch_id": row.get("branch_id"),
                "seed": row.get("seed"),
                "selected_action": action,
                "selected_mode": scored.get("selected_mode"),
                "candidate_scores": scored.get("candidate_scores"),
            }
        )
        if not action:
            missing += 1
            continue
        if action not in ACTION_NAMES:
            unsupported += 1
            continue
        branch_id = str(row.get("branch_id", ""))
        predicted_run = _candidate_outcome(candidate_outcomes.get(branch_id, []), str(action))
        if predicted_run is None:
            missing += 1
            continue
        comparison = dict(predicted_run)
        comparison["rule"] = rule
        comparisons.append(comparison)
    return _summarize_rule(
        rule,
        comparisons,
        missing_branch_outcome_count=missing,
        unsupported_predicted_action_count=unsupported,
        runtime_scores=runtime_scores,
        eligible_for_acceptance=rule == V96_ACCEPTED_RULE,
    )


def _summarize_rule(
    rule: str,
    comparisons: Sequence[Mapping[str, object]],
    *,
    missing_branch_outcome_count: int,
    unsupported_predicted_action_count: int,
    runtime_scores: Sequence[Mapping[str, object]] | None = None,
    eligible_for_acceptance: bool,
) -> dict[str, object]:
    action_counts = Counter(str(item.get("predicted_action")) for item in comparisons)
    mode_counts = Counter(str(item.get("predicted_mode")) for item in comparisons)
    dominant_action, dominant_action_count = _dominant_count(action_counts)
    dominant_mode, dominant_mode_count = _dominant_count(mode_counts)
    return {
        "rule": rule,
        "rule_class": (
            "runtime_feasible_distilled_scorer"
            if rule == V96_ACCEPTED_RULE
            else "diagnostic_comparison"
        ),
        "eligible_for_acceptance": eligible_for_acceptance,
        "runtime_ready_for_feasibility": rule == V96_ACCEPTED_RULE,
        "promotion_ready": False,
        "uses_logged_action_runtime_fallback": False,
        "uses_seed_branch_fixture_or_hidden_state_runtime_features": False,
        "uses_planner_outcome_tables_at_inference": False,
        "uses_global_batch_quota_at_inference": False,
        "heuristic_action_source_count": 0,
        "comparison_count": len(comparisons),
        "missing_branch_outcome_count": int(missing_branch_outcome_count),
        "unsupported_predicted_action_count": int(unsupported_predicted_action_count),
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
        "worst_examples": sorted(
            [dict(item) for item in comparisons],
            key=lambda item: (
                _float(item.get("target_local_score_delta")),
                str(item.get("branch_id", "")),
            ),
        )[:16],
        "comparisons": sorted(
            [dict(item) for item in comparisons],
            key=lambda item: str(item.get("branch_id", "")),
        ),
        "runtime_scores": list(runtime_scores or []),
    }


def _coverage_report(
    *,
    support_labels: Mapping[str, object],
    strict_labels: Mapping[str, object],
    support_rows: Sequence[Mapping[str, object]],
    strict_rows: Sequence[Mapping[str, object]],
    support_loo_predictions: Mapping[str, Sequence[Mapping[str, object]]],
    strict_predictions: Mapping[str, Sequence[Mapping[str, object]]],
    runtime_reload: Mapping[str, object],
    artifact: Mapping[str, object],
) -> dict[str, object]:
    support_aggregate = _mapping(support_labels.get("aggregate"))
    strict_aggregate = _mapping(strict_labels.get("aggregate"))
    support_seeds = {int(row.get("seed", -1)) for row in support_rows}
    strict_seeds = {int(row.get("seed", -1)) for row in strict_rows}
    return {
        "support_label_count": len(support_rows),
        "support_labels_by_seed": dict(
            sorted(Counter(str(row.get("seed")) for row in support_rows).items())
        ),
        "support_source_seeds": sorted(support_seeds),
        "strict_eval_label_count": len(strict_rows),
        "strict_eval_labels_by_seed": dict(
            sorted(Counter(str(row.get("seed")) for row in strict_rows).items())
        ),
        "strict_eval_seeds": sorted(strict_seeds),
        "support_strict_seed_overlap": sorted(support_seeds & strict_seeds),
        "support_strict_contract_seed_overlap": sorted(
            support_seeds & set(V96_STRICT_EVAL_SEEDS)
        ),
        "strict_candidate_action_count": sum(
            len(_candidate_actions(row)) for row in strict_rows
        ),
        "support_candidate_action_count": sum(
            len(_candidate_actions(row)) for row in support_rows
        ),
        "support_replay_verified_all_labels": bool(
            support_aggregate.get("replay_verified_all_labels", False)
        ),
        "strict_replay_verified_all_labels": bool(
            strict_aggregate.get("replay_verified_all_labels", False)
        ),
        "support_heuristic_action_source_count": _heuristic_count(
            support_aggregate
        ),
        "strict_heuristic_action_source_count": _heuristic_count(strict_aggregate),
        "support_unsupported_oracle_action_count": int(
            support_aggregate.get("unsupported_oracle_action_count", 0)
        ),
        "strict_unsupported_oracle_action_count": int(
            strict_aggregate.get("unsupported_oracle_action_count", 0)
        ),
        "strict_unsupported_logged_action_count": int(
            strict_aggregate.get("unsupported_logged_action_count", 0)
        ),
        "support_loo_strict_seed_training_leak_count": sum(
            1
            for items in support_loo_predictions.values()
            for item in items
            if item.get("uses_strict_seed_training") is True
        ),
        "strict_eval_strict_seed_training_leak_count": sum(
            1
            for items in strict_predictions.values()
            for item in items
            if item.get("uses_strict_seed_training") is True
        ),
        "artifact_reload_actions_match": bool(runtime_reload.get("actions_match")),
        "artifact_reload_score_digest_match": bool(
            runtime_reload.get("score_digest_match")
        ),
        "artifact_uses_forbidden_example_keys": _artifact_has_forbidden_example_keys(
            artifact
        ),
        "inference_requires_planner_outcome_tables": bool(
            _mapping(artifact.get("inference_contract")).get(
                "requires_planner_outcome_tables",
                True,
            )
        ),
        "inference_requires_global_batch_assignment": bool(
            _mapping(artifact.get("inference_contract")).get(
                "requires_global_batch_assignment",
                True,
            )
        ),
        "inference_uses_heuristic_fallback": bool(
            _mapping(artifact.get("inference_contract")).get(
                "uses_heuristic_fallback",
                True,
            )
        ),
    }


def _acceptance(
    *,
    coverage: Mapping[str, object],
    strict_report: Mapping[str, object],
) -> dict[str, object]:
    blockers = _rule_blockers(strict_report, coverage)
    accepted = not blockers
    best = {
        "rule": strict_report.get("rule"),
        "blocker_count": len(blockers),
        "dominant_predicted_action": strict_report.get("dominant_predicted_action"),
        "dominant_predicted_action_share": strict_report.get(
            "dominant_predicted_action_share"
        ),
        "dominant_predicted_mode": strict_report.get("dominant_predicted_mode"),
        "dominant_predicted_mode_share": strict_report.get(
            "dominant_predicted_mode_share"
        ),
        "target_alive_delta_negative_count": _mapping(
            strict_report.get("target_alive_delta_summary")
        ).get("negative_count"),
        "mean_target_local_score_delta": _mapping(
            strict_report.get("target_local_score_delta_summary")
        ).get("mean"),
        "mean_terminal_alive_delta": _mapping(
            strict_report.get("terminal_alive_delta_summary")
        ).get("mean"),
        "mean_birth_delta": _mapping(strict_report.get("birth_delta_summary")).get(
            "mean"
        ),
        "seed_41_tick_113_avoided": _mapping(
            _mapping(strict_report.get("seed_41_sequence_cases")).get("tick_113")
        ).get("avoided"),
        "seed_41_tick_114_avoided": _mapping(
            _mapping(strict_report.get("seed_41_sequence_cases")).get("tick_114")
        ).get("avoided"),
    }
    return {
        "v96_runtime_feasibility_accepted": accepted,
        "v97_full_strict_promotion_run_allowed": accepted,
        "runtime_policy_trained": False,
        "runtime_feasibility_artifact_serialized": True,
        "promotion_ready": False,
        "accepted_rules": [V96_ACCEPTED_RULE] if accepted else [],
        "best_rule_for_diagnostics": best,
        "strict_blockers": blockers,
    }


def _rule_blockers(
    report: Mapping[str, object],
    coverage: Mapping[str, object],
) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    for field in (
        "support_replay_verified_all_labels",
        "strict_replay_verified_all_labels",
        "artifact_reload_actions_match",
        "artifact_reload_score_digest_match",
    ):
        if coverage.get(field) is not True:
            blockers.append({"reason": f"{field}_false"})
    for field in (
        "support_heuristic_action_source_count",
        "strict_heuristic_action_source_count",
        "support_unsupported_oracle_action_count",
        "strict_unsupported_oracle_action_count",
        "strict_unsupported_logged_action_count",
        "support_loo_strict_seed_training_leak_count",
        "strict_eval_strict_seed_training_leak_count",
    ):
        if int(coverage.get(field, 0)) != 0:
            blockers.append(
                {
                    "reason": f"{field}_nonzero",
                    "observed": int(coverage.get(field, 0)),
                    "required": 0,
                }
            )
    if coverage.get("support_strict_contract_seed_overlap"):
        blockers.append(
            {
                "reason": "support_strict_contract_seed_overlap_nonempty",
                "observed": coverage.get("support_strict_contract_seed_overlap"),
                "required": [],
            }
        )
    for field in (
        "artifact_uses_forbidden_example_keys",
        "inference_requires_planner_outcome_tables",
        "inference_requires_global_batch_assignment",
        "inference_uses_heuristic_fallback",
    ):
        if coverage.get(field):
            blockers.append({"reason": f"{field}_true"})
    if int(report.get("unsupported_predicted_action_count", 0)) != 0:
        blockers.append(
            {
                "reason": "unsupported_predicted_action_count_nonzero",
                "observed": int(report.get("unsupported_predicted_action_count", 0)),
                "required": 0,
            }
        )
    if int(report.get("comparison_count", 0)) < V96_MIN_STRICT_COMPARISON_COUNT:
        blockers.append(
            {
                "reason": "strict_comparison_count_below_floor",
                "observed": int(report.get("comparison_count", 0)),
                "required_min": V96_MIN_STRICT_COMPARISON_COUNT,
            }
        )
    if (
        _float(report.get("dominant_predicted_action_share"))
        > V96_MAX_DOMINANT_PREDICTED_ACTION_SHARE
    ):
        blockers.append(
            {
                "reason": "dominant_predicted_action_share_above_cap",
                "observed": report.get("dominant_predicted_action_share"),
                "required_max": V96_MAX_DOMINANT_PREDICTED_ACTION_SHARE,
            }
        )
    if (
        _float(report.get("dominant_predicted_mode_share"))
        > V96_MAX_DOMINANT_PREDICTED_MODE_SHARE
    ):
        blockers.append(
            {
                "reason": "dominant_predicted_mode_share_above_cap",
                "observed": report.get("dominant_predicted_mode_share"),
                "required_max": V96_MAX_DOMINANT_PREDICTED_MODE_SHARE,
            }
        )
    target_alive = _mapping(report.get("target_alive_delta_summary"))
    if (
        int(target_alive.get("negative_count", 0))
        != V96_MAX_TARGET_ALIVE_NEGATIVE_COUNT
    ):
        blockers.append(
            {
                "reason": "target_alive_delta_negative_count_nonzero",
                "observed": int(target_alive.get("negative_count", 0)),
                "required": V96_MAX_TARGET_ALIVE_NEGATIVE_COUNT,
            }
        )
    target_local = _mapping(report.get("target_local_score_delta_summary"))
    if _float(target_local.get("mean")) <= V96_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA:
        blockers.append(
            {
                "reason": "mean_target_local_score_delta_not_positive",
                "observed": target_local.get("mean"),
                "required_gt": V96_MIN_MEAN_TARGET_LOCAL_SCORE_DELTA,
            }
        )
    negative_seed_means = {
        seed: _mapping(value).get("target_local_score_delta")
        for seed, value in sorted(_mapping(report.get("per_seed_delta_means")).items())
        if _float(_mapping(value).get("target_local_score_delta"))
        < V96_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA
    }
    if negative_seed_means:
        blockers.append(
            {
                "reason": "per_seed_target_local_score_delta_negative",
                "observed": negative_seed_means,
                "required_min": V96_MIN_PER_SEED_TARGET_LOCAL_SCORE_DELTA,
            }
        )
    terminal_alive = _mapping(report.get("terminal_alive_delta_summary"))
    if _float(terminal_alive.get("mean")) < V96_MIN_MEAN_TERMINAL_ALIVE_DELTA:
        blockers.append(
            {
                "reason": "mean_terminal_alive_delta_negative",
                "observed": terminal_alive.get("mean"),
                "required_min": V96_MIN_MEAN_TERMINAL_ALIVE_DELTA,
            }
        )
    births = _mapping(report.get("birth_delta_summary"))
    if _float(births.get("mean")) < V96_MIN_MEAN_BIRTH_DELTA:
        blockers.append(
            {
                "reason": "mean_birth_delta_negative",
                "observed": births.get("mean"),
                "required_min": V96_MIN_MEAN_BIRTH_DELTA,
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
    for field in (
        "uses_logged_action_runtime_fallback",
        "uses_seed_branch_fixture_or_hidden_state_runtime_features",
        "uses_planner_outcome_tables_at_inference",
        "uses_global_batch_quota_at_inference",
    ):
        if report.get(field) is not False:
            blockers.append({"reason": f"{field}_true"})
    if int(report.get("heuristic_action_source_count", 0)) != 0:
        blockers.append({"reason": "heuristic_action_source_count_nonzero"})
    return blockers


def _teacher_report(
    *,
    support_rows: Sequence[Mapping[str, object]],
    assignment: Sequence[Mapping[str, object]],
    support_baseline_actions: Mapping[str, str],
) -> dict[str, object]:
    action_counts = Counter(str(item.get("predicted_action", "")) for item in assignment)
    mode_counts = Counter(str(item.get("predicted_mode", "")) for item in assignment)
    baseline_counts = Counter(str(action) for action in support_baseline_actions.values())
    dominant_action, dominant_action_count = _dominant_count(action_counts)
    return {
        "policy": "support_only_diversity_regularized_constrained_planner_v1",
        "uses_strict_eval_rows": False,
        "uses_replay_backed_candidate_outcomes_for_training_labels": True,
        "label_count": len(assignment),
        "source_seed_counts": dict(
            sorted(Counter(str(row.get("seed")) for row in support_rows).items())
        ),
        "teacher_action_counts": dict(sorted(action_counts.items())),
        "teacher_mode_counts": dict(sorted(mode_counts.items())),
        "baseline_sequence_action_counts_before_penalty": dict(
            sorted(baseline_counts.items())
        ),
        "dominant_teacher_action": dominant_action,
        "dominant_teacher_action_share": _safe_rate(
            dominant_action_count,
            len(assignment),
        ),
        "target_local_score_delta_summary": _field_summary(
            assignment,
            "target_local_score_delta",
        ),
        "terminal_alive_delta_summary": _field_summary(
            assignment,
            "terminal_alive_delta",
        ),
        "birth_delta_summary": _field_summary(assignment, "birth_delta"),
        "target_alive_delta_summary": _field_summary(assignment, "target_alive_delta"),
    }


def _runtime_reload_check(
    *,
    rows: Sequence[Mapping[str, object]],
    original_artifact: Mapping[str, object],
    reloaded_artifact: Mapping[str, object],
) -> dict[str, object]:
    original_scores = [
        score_distilled_planner_artifact(row=row, artifact=original_artifact)
        for row in rows
    ]
    reloaded_scores = [
        score_distilled_planner_artifact(row=row, artifact=reloaded_artifact)
        for row in rows
    ]
    original_actions = [item.get("selected_action") for item in original_scores]
    reloaded_actions = [item.get("selected_action") for item in reloaded_scores]
    return {
        "policy": "json_roundtrip_reload_v1",
        "actions_match": original_actions == reloaded_actions,
        "score_digest_match": stable_payload_digest(original_scores)
        == stable_payload_digest(reloaded_scores),
        "comparison_count": len(rows),
    }


def _v94_baseline_report(
    branch_sequence_continuation_scorer_report: Mapping[str, object],
) -> dict[str, object]:
    reports = _list_of_mappings(
        branch_sequence_continuation_scorer_report.get("decision_rule_reports"),
        "decision_rule_reports",
    )
    report = next(
        (
            item
            for item in reports
            if item.get("rule") == "sequence_prefix_nearest_neighbor_k5"
        ),
        {},
    )
    return _strip_comparisons(report)


def _v95_teacher_summary(
    branch_constrained_planning_audit_report: Mapping[str, object] | None,
) -> dict[str, object]:
    if branch_constrained_planning_audit_report is None:
        return {"available": False}
    acceptance = _mapping(branch_constrained_planning_audit_report.get("acceptance"))
    best = _mapping(acceptance.get("best_rule_for_diagnostics"))
    return {
        "available": True,
        "diagnostic_only_not_runtime_ready": True,
        "best_rule": best.get("rule"),
        "dominant_predicted_action_share": best.get(
            "dominant_predicted_action_share"
        ),
        "target_alive_delta_negative_count": best.get(
            "target_alive_delta_negative_count"
        ),
        "mean_target_local_score_delta": best.get("mean_target_local_score_delta"),
        "mean_terminal_alive_delta": best.get("mean_terminal_alive_delta"),
        "mean_birth_delta": best.get("mean_birth_delta"),
        "seed_41_tick_113_avoided": best.get("seed_41_tick_113_avoided"),
        "seed_41_tick_114_avoided": best.get("seed_41_tick_114_avoided"),
    }


def _invalid_strict_teacher_control(
    *,
    strict_rows: Sequence[Mapping[str, object]],
    strict_candidate_outcomes: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, object]:
    assignment = _beam_global_assignment(
        strict_rows,
        candidate_outcomes=strict_candidate_outcomes,
        max_action_count=int(
            math.floor(
                float(len(strict_rows)) * V96_MAX_DOMINANT_PREDICTED_ACTION_SHARE
            )
        ),
        diversity_penalty=V96_TEACHER_DIVERSITY_PENALTY,
    )
    report = _summarize_rule(
        V96_INVALID_STRICT_CONTROL_RULE,
        assignment,
        missing_branch_outcome_count=0,
        unsupported_predicted_action_count=0,
        runtime_scores=None,
        eligible_for_acceptance=False,
    )
    stripped = _strip_comparisons(report)
    stripped["invalid_reason"] = (
        "uses strict held-out branch outcome labels as teacher labels"
    )
    stripped["counts_as_accepted_result"] = False
    return stripped


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


def _candidate_outcome(
    outcomes: Sequence[Mapping[str, object]],
    action: str,
) -> Mapping[str, object] | None:
    return next(
        (item for item in outcomes if str(item.get("predicted_action", "")) == action),
        None,
    )


def _validate_artifact(artifact: Mapping[str, object]) -> None:
    try:
        validate_mind_v3_planner_distilled_artifact(artifact)
    except MindV3PlannerDistilledArtifactError as exc:
        raise BranchPlannerDistillationAuditError(str(exc)) from exc


def _artifact_has_forbidden_example_keys(artifact: Mapping[str, object]) -> bool:
    return _runtime_artifact_has_forbidden_example_keys(artifact)


def _strip_comparisons(report: Mapping[str, object]) -> dict[str, object]:
    stripped = dict(report)
    stripped.pop("comparisons", None)
    stripped.pop("runtime_scores", None)
    return stripped


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


def _tuple(value: object) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)):
        return ()
    return tuple(float(item) for item in value)


def _round_sequence(values: Sequence[float]) -> list[float]:
    return [_round(float(value)) for value in values]


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
