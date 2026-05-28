from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence, TextIO

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.broad_branch_residual_constrained_audit import (
    MIND_V3_V100_BROAD_BRANCH_RESIDUAL_CONSTRAINED_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.broad_branch_residual_oracle_audit import (
    MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_AUDIT_SCHEMA_VERSION,
)
from evolution_sim.mind.broad_transfer_residual_audit import V98_STRICT_EXCLUDED_SEEDS
from evolution_sim.mind.branch_mode_objective_audit import _action_option_mode
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.v3_planner_distilled import (
    candidate_feature_vector,
    planner_distilled_runtime_row,
)

MIND_V3_V101_BROAD_BRANCH_RESIDUAL_DISTILLATION_EXAMPLE_SCHEMA_VERSION = (
    "mind_v3_v101_broad_branch_residual_distillation_example_v1"
)
MIND_V3_V101_BROAD_BRANCH_RESIDUAL_EXAMPLE_ARTIFACT_SCHEMA_VERSION = (
    "mind_v3_v101_broad_branch_residual_example_artifact_v1"
)
MIND_V3_V101_BROAD_BRANCH_RESIDUAL_DISTILLATION_POLICY = (
    "v101_policy_visible_broad_residual_distillation_example_v1"
)
V101_MIN_TRAINING_ROWS = 10
V101_MIN_SOURCE_SEEDS = 8
V101_MAX_DOMINANT_TEACHER_ACTION_SHARE = 0.50
V101_MIN_SAFE_NON_LOGGED_OVERRIDES = 8


class BroadBranchResidualDistillationExampleError(ValueError):
    pass


def load_json_report(source: str | Path | Mapping[str, object]) -> dict[str, object]:
    if isinstance(source, Mapping):
        return dict(source)
    path = Path(source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BroadBranchResidualDistillationExampleError(
            f"failed to read report: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise BroadBranchResidualDistillationExampleError(
            f"report is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise BroadBranchResidualDistillationExampleError("report must be an object")
    return payload


def load_broad_residual_distillation_example_artifact(
    source: str | Path | Mapping[str, object],
) -> dict[str, object]:
    payload = load_json_report(source)
    artifact = payload.get("distilled_example_artifact")
    if not isinstance(artifact, Mapping):
        artifact = payload
    validate_broad_residual_distillation_example_artifact(artifact)
    return dict(artifact)


def build_broad_branch_residual_distillation_example_report(
    *,
    v99_broad_branch_residual_oracle_report: Mapping[str, object],
    v100_broad_branch_residual_constrained_report: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    _validate_source_reports(
        v99_report=v99_broad_branch_residual_oracle_report,
        v100_report=v100_broad_branch_residual_constrained_report,
    )
    rows = _training_rows(
        v99_report=v99_broad_branch_residual_oracle_report,
        v100_report=v100_broad_branch_residual_constrained_report,
    )
    artifact = _distillation_example_artifact(rows)
    validate_broad_residual_distillation_example_artifact(artifact)
    training_eval = _training_evaluation(artifact, rows)
    reloaded = json.loads(json.dumps(artifact, sort_keys=True, allow_nan=False))
    reload_eval = _training_evaluation(reloaded, rows)
    reload_identical = training_eval["predicted_actions"] == reload_eval[
        "predicted_actions"
    ]
    coverage = _coverage(rows)
    floors = {
        "source_v100_accepted": True,
        "no_strict_seed_leakage": True,
        "training_row_count": V101_MIN_TRAINING_ROWS,
        "source_seed_count": V101_MIN_SOURCE_SEEDS,
        "label_action_legal_count": len(rows),
        "dominant_teacher_action_share_max": (
            V101_MAX_DOMINANT_TEACHER_ACTION_SHARE
        ),
        "safe_non_logged_override_count": V101_MIN_SAFE_NON_LOGGED_OVERRIDES,
        "training_accuracy": 1.0,
        "artifact_reload_identical_choices": True,
        "runtime_policy_trained": False,
    }
    acceptance = _acceptance(
        coverage=coverage,
        training_eval=training_eval,
        reload_identical=reload_identical,
        floors=floors,
    )
    accepted = bool(
        acceptance["v101_broad_residual_distillation_example_accepted"]
    )
    contract = {
        "schema_version": (
            MIND_V3_V101_BROAD_BRANCH_RESIDUAL_DISTILLATION_EXAMPLE_SCHEMA_VERSION
        ),
        "policy": MIND_V3_V101_BROAD_BRANCH_RESIDUAL_DISTILLATION_POLICY,
        "diagnostic_only": True,
        "runtime_policy_trained": False,
        "promotion_run_executed": False,
        "source_v99_schema_version": (
            v99_broad_branch_residual_oracle_report.get("schema_version")
        ),
        "source_v100_schema_version": (
            v100_broad_branch_residual_constrained_report.get("schema_version")
        ),
        "training_contract": {
            "default_policy": "linear_mind_v3",
            "label_source": "v100_diversity_constrained_broad_branch_assignment",
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
    report = {
        "schema_version": (
            MIND_V3_V101_BROAD_BRANCH_RESIDUAL_DISTILLATION_EXAMPLE_SCHEMA_VERSION
        ),
        "policy": MIND_V3_V101_BROAD_BRANCH_RESIDUAL_DISTILLATION_POLICY,
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
                }
            ),
            "source_v100_digest": stable_payload_digest(
                {
                    "schema_version": v100_broad_branch_residual_constrained_report.get(
                        "schema_version"
                    ),
                    "acceptance": v100_broad_branch_residual_constrained_report.get(
                        "acceptance"
                    ),
                    "rule_reports": v100_broad_branch_residual_constrained_report.get(
                        "rule_reports"
                    ),
                }
            ),
            "artifact_digest": stable_payload_digest(artifact),
            "contract_digest": stable_payload_digest(contract),
        },
        "coverage": coverage,
        "training_rows": rows,
        "distilled_example_artifact": artifact,
        "training_evaluation": training_eval,
        "reload_evaluation": {
            "artifact_reload_identical_choices": reload_identical,
            "training_accuracy": reload_eval["training_accuracy"],
            "predicted_action_counts": reload_eval["predicted_action_counts"],
        },
        "broad_branch_residual_distillation_example_support_probe": {
            "policy": MIND_V3_V101_BROAD_BRANCH_RESIDUAL_DISTILLATION_POLICY,
            "accuracy": 1.0 if accepted else 0.0,
            "support_accuracy_floor": 1.0,
            "materially_supports_v102_expanded_training": accepted,
            "runtime_policy_status": "training_example_only_no_runtime_promotion",
            "training_row_count": coverage["training_row_count"],
            "source_seed_count": coverage["source_seed_count"],
            "dominant_teacher_action_share": coverage[
                "dominant_teacher_action_share"
            ],
            "training_accuracy": training_eval["training_accuracy"],
            "blocker_count": acceptance["blocker_count"],
        },
        "acceptance": acceptance,
        "v101_broad_residual_distillation_example_accepted": accepted,
        "v102_expanded_training_allowed": bool(
            acceptance["v102_expanded_training_allowed"]
        ),
        "runtime_promotion_allowed": False,
        "blocker_count": int(acceptance["blocker_count"]),
    }
    return report, artifact


def validate_broad_residual_distillation_example_artifact(
    artifact: Mapping[str, object],
) -> None:
    if (
        artifact.get("schema_version")
        != MIND_V3_V101_BROAD_BRANCH_RESIDUAL_EXAMPLE_ARTIFACT_SCHEMA_VERSION
    ):
        raise BroadBranchResidualDistillationExampleError(
            "artifact has unsupported schema_version"
        )
    inference = _mapping(artifact.get("inference_contract"))
    required = {
        "one_row_one_agent_local_decision": True,
        "requires_action_mask": True,
        "requires_policy_visible_features_only": True,
        "requires_linear_default_action": True,
        "requires_global_batch_assignment": False,
        "requires_planner_outcome_tables": False,
        "uses_heuristic_fallback": False,
    }
    for key, expected in required.items():
        if inference.get(key) is not expected:
            raise BroadBranchResidualDistillationExampleError(
                f"artifact violates inference_contract.{key}"
            )
    if _artifact_has_forbidden_keys(artifact):
        raise BroadBranchResidualDistillationExampleError(
            "artifact contains forbidden runtime/provenance keys"
        )
    examples = _list_of_mappings(artifact.get("examples"))
    support = _list_of_mappings(artifact.get("support_examples"))
    if not examples:
        raise BroadBranchResidualDistillationExampleError(
            "artifact must include training examples"
        )
    if not support:
        raise BroadBranchResidualDistillationExampleError(
            "artifact must include support examples"
        )


def score_broad_residual_distillation_example_artifact(
    *,
    artifact: Mapping[str, object],
    observation_input: Mapping[str, object],
    action_mask: Mapping[str, object],
    public_history_trace: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    validate_broad_residual_distillation_example_artifact(artifact)
    row = planner_distilled_runtime_row(
        observation_input=observation_input,
        action_mask=action_mask,
        public_history_trace=public_history_trace,
    )
    support = _list_of_mappings(artifact.get("support_examples"))
    candidate_scores: list[dict[str, object]] = []
    for action in _legal_actions(action_mask):
        features = candidate_feature_vector(row, action)
        if not features:
            continue
        distance, weight = _nearest_support(
            features,
            support,
            action=action,
        )
        if math.isinf(distance):
            score = float("-inf")
        else:
            score = -distance + 0.0001 * weight
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
        "unsupported_predicted_action": selected.get("action") not in _legal_actions(
            action_mask
        ),
    }


def write_broad_branch_residual_distillation_example_report(
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


def _training_rows(
    *,
    v99_report: Mapping[str, object],
    v100_report: Mapping[str, object],
) -> list[dict[str, object]]:
    branch_points = {
        str(point.get("branch_id", "")): point
        for point in _list_of_mappings(v99_report.get("branch_points"))
    }
    assignment = _accepted_assignment(v100_report)
    rows: list[dict[str, object]] = []
    for index, item in enumerate(assignment):
        branch_id = str(item.get("branch_id", ""))
        point = _mapping(branch_points.get(branch_id))
        observation_input = _mapping(point.get("observation_input"))
        if not observation_input:
            raise BroadBranchResidualDistillationExampleError(
                "v99 branch point is missing observation_input; regenerate v99"
            )
        action_mask = _complete_action_mask(_mapping(point.get("action_mask")))
        public_history = [
            dict(history_item)
            for history_item in _list_of_mappings(point.get("public_history_trace"))
        ]
        teacher_action = str(item.get("predicted_action", ""))
        linear_action = str(item.get("logged_action", ""))
        if teacher_action not in ACTION_NAMES or not bool(
            action_mask.get(teacher_action, False)
        ):
            raise BroadBranchResidualDistillationExampleError(
                f"teacher action is unsupported for branch {branch_id}: "
                f"{teacher_action}"
            )
        runtime_row = planner_distilled_runtime_row(
            observation_input=observation_input,
            action_mask=action_mask,
            public_history_trace=public_history,
        )
        candidate_vectors = {
            action: list(candidate_feature_vector(runtime_row, action))
            for action in _legal_actions(action_mask)
        }
        if not candidate_vectors.get(teacher_action):
            raise BroadBranchResidualDistillationExampleError(
                f"teacher action has no feature vector: {teacher_action}"
            )
        target_delta = _float(item.get("target_local_score_delta"))
        rows.append(
            {
                "row_index": index,
                "provenance": {
                    "branch_id": branch_id,
                    "source_seed": int(item.get("seed", 0)),
                    "branch_tick": int(item.get("branch_tick", 0)),
                    "agent_id": int(item.get("agent_id", 0)),
                },
                "policy_state": {
                    "observation_input": dict(observation_input),
                    "action_mask": action_mask,
                    "public_history_trace": public_history,
                },
                "linear_action": linear_action,
                "teacher_action": teacher_action,
                "teacher_mode": _action_option_mode(teacher_action),
                "override_label": teacher_action != linear_action,
                "safe_non_logged_override": bool(
                    item.get("safe_non_logged_override")
                ),
                "target_local_score_delta": _round(target_delta),
                "terminal_alive_delta": _round(
                    _float(item.get("terminal_alive_delta"))
                ),
                "birth_delta": _round(_float(item.get("birth_delta"))),
                "target_alive_delta": _round(
                    _float(item.get("target_alive_delta"))
                ),
                "example_weight": _round(1.0 + min(max(target_delta, 0.0) / 50.0, 4.0)),
                "legal_actions": _legal_actions(action_mask),
                "candidate_feature_vectors": candidate_vectors,
            }
        )
    return rows


def _distillation_example_artifact(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    examples = []
    support = []
    for row in rows:
        policy_state = _mapping(row.get("policy_state"))
        teacher_action = str(row.get("teacher_action", ""))
        vectors = _mapping(row.get("candidate_feature_vectors"))
        example = {
            "example_index": int(row.get("row_index", 0)),
            "linear_action": str(row.get("linear_action", "")),
            "teacher_action": teacher_action,
            "teacher_mode": str(row.get("teacher_mode", "")),
            "override_label": bool(row.get("override_label")),
            "safe_non_logged_override": bool(row.get("safe_non_logged_override")),
            "target_local_score_delta": row.get("target_local_score_delta"),
            "terminal_alive_delta": row.get("terminal_alive_delta"),
            "birth_delta": row.get("birth_delta"),
            "target_alive_delta": row.get("target_alive_delta"),
            "example_weight": row.get("example_weight"),
            "policy_state": {
                "observation_input": dict(
                    _mapping(policy_state.get("observation_input"))
                ),
                "action_mask": dict(_mapping(policy_state.get("action_mask"))),
                "public_history_trace": [
                    dict(item)
                    for item in _list_of_mappings(
                        policy_state.get("public_history_trace")
                    )
                ],
            },
            "candidate_feature_vectors": {
                str(action): list(vector)
                for action, vector in sorted(vectors.items())
                if isinstance(vector, list)
            },
        }
        examples.append(example)
        support.append(
            {
                "example_index": example["example_index"],
                "action": teacher_action,
                "mode": example["teacher_mode"],
                "feature_vector": list(vectors.get(teacher_action, [])),
                "weight": row.get("example_weight"),
            }
        )
    action_counts = Counter(str(row.get("teacher_action", "")) for row in rows)
    return {
        "schema_version": (
            MIND_V3_V101_BROAD_BRANCH_RESIDUAL_EXAMPLE_ARTIFACT_SCHEMA_VERSION
        ),
        "training_policy": MIND_V3_V101_BROAD_BRANCH_RESIDUAL_DISTILLATION_POLICY,
        "diagnostic_example_only": True,
        "runtime_ready": False,
        "promotion_ready": False,
        "scorer_policy": "nearest_teacher_action_support_example_v1",
        "inference_contract": {
            "one_row_one_agent_local_decision": True,
            "requires_action_mask": True,
            "requires_policy_visible_features_only": True,
            "requires_linear_default_action": True,
            "requires_global_batch_assignment": False,
            "requires_planner_outcome_tables": False,
            "uses_heuristic_fallback": False,
        },
        "training_row_count": len(rows),
        "teacher_action_counts": dict(sorted(action_counts.items())),
        "examples": examples,
        "support_examples": support,
    }


def _training_evaluation(
    artifact: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    predicted_actions = []
    examples = []
    unsupported = 0
    for row in rows:
        policy_state = _mapping(row.get("policy_state"))
        scored = score_broad_residual_distillation_example_artifact(
            artifact=artifact,
            observation_input=_mapping(policy_state.get("observation_input")),
            action_mask=_mapping(policy_state.get("action_mask")),
            public_history_trace=_list_of_mappings(
                policy_state.get("public_history_trace")
            ),
        )
        predicted = scored.get("selected_action")
        predicted_actions.append(predicted)
        if scored.get("unsupported_predicted_action") is True:
            unsupported += 1
        examples.append(
            {
                "row_index": row.get("row_index"),
                "teacher_action": row.get("teacher_action"),
                "predicted_action": predicted,
                "correct": predicted == row.get("teacher_action"),
            }
        )
    correct = sum(
        1
        for row, predicted in zip(rows, predicted_actions)
        if predicted == row.get("teacher_action")
    )
    counts = Counter(str(action) for action in predicted_actions)
    dominant = _dominant_count_share(counts)
    return {
        "evaluated_row_count": len(rows),
        "correct_count": correct,
        "training_accuracy": _safe_rate(correct, len(rows)),
        "predicted_actions": list(predicted_actions),
        "predicted_action_counts": dict(sorted(counts.items())),
        "dominant_predicted_action": dominant["key"],
        "dominant_predicted_action_share": dominant["share"],
        "unsupported_predicted_action_count": unsupported,
        "examples": examples,
    }


def _coverage(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    seeds = sorted(
        {
            int(_mapping(row.get("provenance")).get("source_seed", 0))
            for row in rows
        }
    )
    teacher_counts = Counter(str(row.get("teacher_action", "")) for row in rows)
    mode_counts = Counter(str(row.get("teacher_mode", "")) for row in rows)
    dominant = _dominant_count_share(teacher_counts)
    return {
        "training_row_count": len(rows),
        "source_seed_count": len(seeds),
        "source_seeds": seeds,
        "strict_seed_leak_count": len(set(seeds) & set(V98_STRICT_EXCLUDED_SEEDS)),
        "label_action_legal_count": sum(
            1
            for row in rows
            if row.get("teacher_action") in row.get("legal_actions", [])
        ),
        "override_label_count": sum(1 for row in rows if row.get("override_label") is True),
        "safe_non_logged_override_count": sum(
            1 for row in rows if row.get("safe_non_logged_override") is True
        ),
        "teacher_action_counts": dict(sorted(teacher_counts.items())),
        "teacher_mode_counts": dict(sorted(mode_counts.items())),
        "dominant_teacher_action": dominant["key"],
        "dominant_teacher_action_share": dominant["share"],
        "target_local_score_delta_mean": _round(
            _mean([_float(row.get("target_local_score_delta")) for row in rows])
        ),
        "terminal_alive_delta_mean": _round(
            _mean([_float(row.get("terminal_alive_delta")) for row in rows])
        ),
        "birth_delta_mean": _round(
            _mean([_float(row.get("birth_delta")) for row in rows])
        ),
        "target_alive_delta_negative_count": sum(
            1 for row in rows if _float(row.get("target_alive_delta")) < 0.0
        ),
    }


def _acceptance(
    *,
    coverage: Mapping[str, object],
    training_eval: Mapping[str, object],
    reload_identical: bool,
    floors: Mapping[str, object],
) -> dict[str, object]:
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

    if _int(coverage.get("strict_seed_leak_count")) != 0:
        block("strict_seed_leakage", "strict_seed_leak_count", coverage.get("strict_seed_leak_count"), 0, "eq")
    if _int(coverage.get("training_row_count")) < _int(floors.get("training_row_count")):
        block("insufficient_training_rows", "training_row_count", coverage.get("training_row_count"), floors.get("training_row_count"), "ge")
    if _int(coverage.get("source_seed_count")) < _int(floors.get("source_seed_count")):
        block("insufficient_source_seed_count", "source_seed_count", coverage.get("source_seed_count"), floors.get("source_seed_count"), "ge")
    if _int(coverage.get("label_action_legal_count")) != _int(floors.get("label_action_legal_count")):
        block("illegal_teacher_action", "label_action_legal_count", coverage.get("label_action_legal_count"), floors.get("label_action_legal_count"), "eq")
    if _float(coverage.get("dominant_teacher_action_share")) > _float(
        floors.get("dominant_teacher_action_share_max")
    ):
        block("dominant_teacher_action_share_above_cap", "dominant_teacher_action_share", coverage.get("dominant_teacher_action_share"), floors.get("dominant_teacher_action_share_max"), "le")
    if _int(coverage.get("safe_non_logged_override_count")) < _int(
        floors.get("safe_non_logged_override_count")
    ):
        block("insufficient_safe_non_logged_overrides", "safe_non_logged_override_count", coverage.get("safe_non_logged_override_count"), floors.get("safe_non_logged_override_count"), "ge")
    if _float(training_eval.get("training_accuracy")) < _float(
        floors.get("training_accuracy")
    ):
        block("training_accuracy_below_floor", "training_accuracy", training_eval.get("training_accuracy"), floors.get("training_accuracy"), "ge")
    if _int(training_eval.get("unsupported_predicted_action_count")) != 0:
        block("unsupported_predicted_action", "unsupported_predicted_action_count", training_eval.get("unsupported_predicted_action_count"), 0, "eq")
    if not reload_identical:
        block("artifact_reload_changed_choices", "artifact_reload_identical_choices", reload_identical, True, "eq")
    accepted = not blockers
    return {
        "v101_broad_residual_distillation_example_accepted": accepted,
        "v102_expanded_training_allowed": accepted,
        "runtime_promotion_allowed": False,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "recommendation": (
            "Use this as the minimal clean training contract, then expand v102 "
            "branch support before any runtime rollout."
            if accepted
            else "Do not train from this example until blockers are fixed."
        ),
    }


def _accepted_assignment(report: Mapping[str, object]) -> list[Mapping[str, object]]:
    acceptance = _mapping(report.get("acceptance"))
    accepted_rules = {
        str(rule) for rule in acceptance.get("accepted_rules", []) if str(rule)
    }
    for rule in _list_of_mappings(report.get("rule_reports")):
        if str(rule.get("rule", "")) in accepted_rules:
            assignment = _list_of_mappings(rule.get("assignment"))
            if assignment:
                return assignment
    raise BroadBranchResidualDistillationExampleError(
        "v100 report does not include an accepted assignment"
    )


def _validate_source_reports(
    *,
    v99_report: Mapping[str, object],
    v100_report: Mapping[str, object],
) -> None:
    if (
        v99_report.get("schema_version")
        != MIND_V3_V99_BROAD_BRANCH_RESIDUAL_ORACLE_AUDIT_SCHEMA_VERSION
    ):
        raise BroadBranchResidualDistillationExampleError(
            "v99 report has unsupported schema_version"
        )
    if (
        v100_report.get("schema_version")
        != MIND_V3_V100_BROAD_BRANCH_RESIDUAL_CONSTRAINED_AUDIT_SCHEMA_VERSION
    ):
        raise BroadBranchResidualDistillationExampleError(
            "v100 report has unsupported schema_version"
        )
    if (
        v100_report.get("v100_broad_branch_residual_constrained_diagnostic_accepted")
        is not True
    ):
        raise BroadBranchResidualDistillationExampleError(
            "v100 constrained diagnostic must be accepted before v101"
        )


def _artifact_has_forbidden_keys(artifact: Mapping[str, object]) -> bool:
    forbidden = {
        "seed",
        "source_seed",
        "branch_id",
        "fixture",
        "fixture_identity",
        "agent_id",
        "logged_action",
        "strict_eval_label_identity",
        "planner_candidate_outcome_table",
        "global_batch_action_quota",
        "private_simulation_world_state",
    }
    return _contains_forbidden_keys(artifact, forbidden)


def _contains_forbidden_keys(value: object, forbidden: set[str]) -> bool:
    if isinstance(value, Mapping):
        if set(value) & forbidden:
            return True
        return any(_contains_forbidden_keys(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_keys(item, forbidden) for item in value)
    return False


def _nearest_support(
    features: Sequence[float],
    examples: Sequence[Mapping[str, object]],
    *,
    action: str,
) -> tuple[float, float]:
    best_distance = float("inf")
    best_weight = 0.0
    for example in examples:
        if str(example.get("action", "")) != action:
            continue
        vector = _float_list(example.get("feature_vector"))
        if not vector:
            continue
        distance = _squared_distance(features, vector)
        if distance < best_distance:
            best_distance = distance
            best_weight = _float(example.get("weight"))
    return best_distance, best_weight


def _squared_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        return float("inf")
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right))


def _complete_action_mask(raw: Mapping[str, object]) -> dict[str, bool]:
    return {action: bool(raw.get(action, False)) for action in ACTION_NAMES}


def _legal_actions(action_mask: Mapping[str, object]) -> list[str]:
    return [action for action in ACTION_NAMES if bool(action_mask.get(action, False))]


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


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    parsed = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return []
        parsed.append(float(item))
    return parsed


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
