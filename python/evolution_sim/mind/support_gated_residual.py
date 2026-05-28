from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.v3_planner_distilled import (
    candidate_feature_vector,
    planner_distilled_runtime_row,
)

MIND_V3_V103_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION = (
    "mind_v3_v103_support_gated_residual_runtime_artifact_v1"
)
MIND_V3_V104_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION = (
    "mind_v3_v104_action_conditioned_support_gated_residual_runtime_artifact_v1"
)
MIND_V3_V103_SUPPORT_GATED_RESIDUAL_POLICY = (
    "mind_v3_v103_support_gated_residual_runtime_v1"
)
MIND_V3_V104_SUPPORT_GATED_RESIDUAL_POLICY = (
    "mind_v3_v104_action_conditioned_support_gated_residual_runtime_v1"
)
MIND_V3_V103_BRANCH_REPLAY_FEASIBILITY_SCHEMA_VERSION = (
    "mind_v3_v103_branch_replay_feasibility_v1"
)
MIND_V3_V104_BRANCH_REPLAY_FEASIBILITY_SCHEMA_VERSION = (
    "mind_v3_v104_branch_replay_feasibility_v1"
)
MIND_V3_V103_RUNTIME_DIAGNOSTICS_POLICY = (
    "mind_v3_v103_support_gated_residual_runtime_diagnostics_v1"
)
MIND_V3_V104_RUNTIME_DIAGNOSTICS_POLICY = (
    "mind_v3_v104_action_conditioned_support_gated_residual_runtime_diagnostics_v1"
)
MIND_V3_V103_THRESHOLD_POLICY = (
    "v102_loo_p75_distance_p75_margin_support_gate_v1"
)
MIND_V3_V104_ACTION_CONDITIONED_THRESHOLD_POLICY = (
    "v102_loo_action_conditioned_p75_distance_p75_margin_support_gate_v1"
)
MIND_V3_V102_EXPANDED_BROAD_RESIDUAL_TRAINING_SCHEMA_VERSION = (
    "mind_v3_v102_expanded_broad_residual_training_v1"
)
V103_ACTION_PRIOR_BALANCE_PENALTY = 2.0
V103_DISTANCE_THRESHOLD_QUANTILE = 0.75
V103_MARGIN_THRESHOLD_QUANTILE = 0.75
V103_MAX_DOMINANT_APPLIED_OVERRIDE_ACTION_SHARE = 0.50
SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSIONS = frozenset(
    {
        MIND_V3_V103_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION,
        MIND_V3_V104_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION,
    }
)
SUPPORT_GATED_RESIDUAL_POLICIES = frozenset(
    {
        MIND_V3_V103_SUPPORT_GATED_RESIDUAL_POLICY,
        MIND_V3_V104_SUPPORT_GATED_RESIDUAL_POLICY,
    }
)


class SupportGatedResidualError(ValueError):
    pass


def load_support_gated_residual_artifact(
    source: str | Path | Mapping[str, object],
) -> dict[str, object]:
    if isinstance(source, Mapping):
        payload = dict(source)
    else:
        path = Path(source)
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise SupportGatedResidualError(
                f"failed to read support-gated residual artifact: {path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise SupportGatedResidualError(
                f"support-gated residual artifact is not valid JSON: {exc.msg}"
            ) from exc
        if not isinstance(loaded, dict):
            raise SupportGatedResidualError(
                "support-gated residual artifact payload must be a JSON object"
            )
        payload = loaded
    embedded = payload.get("support_gated_residual_artifact")
    if isinstance(embedded, Mapping):
        payload = dict(embedded)
    validate_support_gated_residual_artifact(payload)
    return dict(payload)


def validate_support_gated_residual_artifact(
    artifact: Mapping[str, object],
) -> None:
    schema_version = artifact.get("schema_version")
    if schema_version not in SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSIONS:
        raise SupportGatedResidualError(
            "support-gated residual artifact has unsupported schema_version"
        )
    if artifact.get("policy") not in SUPPORT_GATED_RESIDUAL_POLICIES:
        raise SupportGatedResidualError(
            "support-gated residual artifact has unsupported policy"
        )
    if artifact.get("runtime_ready") is not True:
        raise SupportGatedResidualError("support-gated residual artifact is not runtime_ready")
    if artifact.get("promotion_ready") is not False:
        raise SupportGatedResidualError("support-gated residual artifact must not be promotion_ready")
    if artifact.get("runtime_promotion_allowed") is not False:
        raise SupportGatedResidualError(
            "support-gated residual artifact must not allow runtime promotion"
        )
    inference = _mapping(artifact.get("inference_contract"))
    required_contract = {
        "one_row_one_agent_local_decision": True,
        "requires_action_mask": True,
        "requires_policy_visible_features_only": True,
        "requires_linear_default_action": True,
        "requires_planner_outcome_tables": False,
        "requires_global_batch_assignment": False,
        "uses_heuristic_fallback": False,
        "uses_seed_id_as_runtime_feature": False,
        "uses_branch_id_as_runtime_feature": False,
        "uses_fixture_id_as_runtime_feature": False,
        "uses_logged_action_as_runtime_fallback": False,
        "uses_private_simulator_state": False,
    }
    for key, expected in required_contract.items():
        if inference.get(key) is not expected:
            raise SupportGatedResidualError(
                f"support-gated residual artifact violates inference_contract.{key}"
            )
    gate = _mapping(artifact.get("support_gate"))
    if schema_version == MIND_V3_V104_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION:
        action_thresholds = _mapping(gate.get("action_thresholds"))
        if not action_thresholds:
            raise SupportGatedResidualError(
                "v104 support_gate.action_thresholds must not be empty"
            )
        for action, thresholds in action_thresholds.items():
            if str(action) not in ACTION_NAMES:
                raise SupportGatedResidualError(
                    f"v104 support_gate.action_thresholds has unsupported action: {action}"
                )
            _validate_gate_thresholds(
                _mapping(thresholds),
                prefix=f"support_gate.action_thresholds[{action!r}]",
            )
    else:
        _validate_gate_thresholds(gate, prefix="support_gate")
    if _contains_forbidden_runtime_keys(artifact):
        raise SupportGatedResidualError(
            "support-gated residual artifact contains forbidden runtime/provenance keys"
        )
    support = _list_of_mappings(artifact.get("support_examples"))
    if not support:
        raise SupportGatedResidualError(
            "support-gated residual artifact must include support_examples"
        )
    for index, example in enumerate(support):
        action = str(example.get("action", ""))
        if action not in ACTION_NAMES:
            raise SupportGatedResidualError(
                f"support example {index} has unsupported action: {action}"
            )
        vector = _float_list(example.get("feature_vector"))
        if not vector:
            raise SupportGatedResidualError(
                f"support example {index} must include a feature_vector"
            )


def _validate_gate_thresholds(
    thresholds: Mapping[str, object],
    *,
    prefix: str,
) -> None:
    for key in ("nearest_support_distance_threshold", "residual_score_margin_threshold"):
        value = _float(thresholds.get(key), default=float("nan"))
        if not math.isfinite(value) or value < 0.0:
            raise SupportGatedResidualError(f"{prefix}.{key} must be finite and non-negative")


def build_support_gated_residual_runtime_artifact(
    *,
    v102_report: Mapping[str, object],
    v102_artifact: Mapping[str, object],
    v99_report: Mapping[str, object],
    v100_report: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    if (
        v102_report.get("schema_version")
        != MIND_V3_V102_EXPANDED_BROAD_RESIDUAL_TRAINING_SCHEMA_VERSION
    ):
        raise SupportGatedResidualError("v102 report has unsupported schema_version")
    if v102_report.get("v103_support_gated_residual_runtime_allowed") is not True:
        raise SupportGatedResidualError("v102 report does not allow v103 runtime feasibility")
    from evolution_sim.mind.broad_branch_residual_distillation_example import (
        _training_rows,
        validate_broad_residual_distillation_example_artifact,
    )

    validate_broad_residual_distillation_example_artifact(v102_artifact)
    rows = _training_rows(v99_report=v99_report, v100_report=v100_report)
    threshold_report = _threshold_report(rows=rows)
    support_examples = [
        _support_example_for_runtime(example)
        for example in _list_of_mappings(v102_artifact.get("support_examples"))
    ]
    action_counts = Counter(str(example.get("action", "")) for example in support_examples)
    contract = {
        "one_row_one_agent_local_decision": True,
        "requires_action_mask": True,
        "requires_policy_visible_features_only": True,
        "requires_linear_default_action": True,
        "requires_planner_outcome_tables": False,
        "requires_global_batch_assignment": False,
        "uses_heuristic_fallback": False,
        "uses_seed_id_as_runtime_feature": False,
        "uses_branch_id_as_runtime_feature": False,
        "uses_fixture_id_as_runtime_feature": False,
        "uses_logged_action_as_runtime_fallback": False,
        "uses_private_simulator_state": False,
    }
    artifact = {
        "schema_version": (
            MIND_V3_V103_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION
        ),
        "policy": MIND_V3_V103_SUPPORT_GATED_RESIDUAL_POLICY,
        "default_action_policy": "linear_mind_v3",
        "runtime_ready": True,
        "promotion_ready": False,
        "runtime_promotion_allowed": False,
        "training_policy": v102_artifact.get("training_policy"),
        "source_v102_schema_version": v102_report.get("schema_version"),
        "source_v102_report_digest": stable_payload_digest(
            {
                "schema_version": v102_report.get("schema_version"),
                "coverage": v102_report.get("coverage"),
                "acceptance": v102_report.get("acceptance"),
                "leave_one_source_seed_out_evaluation": v102_report.get(
                    "leave_one_source_seed_out_evaluation"
                ),
            }
        ),
        "scorer_rule": "action_prior_balanced_nearest_support_v1",
        "action_prior_penalty_scale": V103_ACTION_PRIOR_BALANCE_PENALTY,
        "support_gate": {
            "policy": MIND_V3_V103_THRESHOLD_POLICY,
            "threshold_source": "v102_leave_one_source_seed_out_distribution_v1",
            "legal_action_required": True,
            "action_support_required": True,
            "distance_threshold_inclusive": True,
            "margin_threshold_inclusive": True,
            "nearest_support_distance_threshold": threshold_report[
                "nearest_support_distance_threshold"
            ],
            "residual_score_margin_threshold": threshold_report[
                "residual_score_margin_threshold"
            ],
            "distance_quantile": V103_DISTANCE_THRESHOLD_QUANTILE,
            "margin_quantile": V103_MARGIN_THRESHOLD_QUANTILE,
        },
        "inference_contract": contract,
        "training_row_count": int(v102_artifact.get("training_row_count", len(rows))),
        "support_action_counts": dict(sorted(action_counts.items())),
        "teacher_action_counts": dict(
            sorted(
                {
                    str(key): int(value)
                    for key, value in _mapping(
                        v102_artifact.get("teacher_action_counts")
                    ).items()
                    if isinstance(value, int) and not isinstance(value, bool)
                }.items()
            )
        ),
        "support_examples": support_examples,
        "threshold_diagnostics": threshold_report,
    }
    validate_support_gated_residual_artifact(artifact)
    return artifact, threshold_report


def build_action_conditioned_support_gated_residual_runtime_artifact(
    *,
    v102_report: Mapping[str, object],
    v102_artifact: Mapping[str, object],
    v99_report: Mapping[str, object],
    v100_report: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    if (
        v102_report.get("schema_version")
        != MIND_V3_V102_EXPANDED_BROAD_RESIDUAL_TRAINING_SCHEMA_VERSION
    ):
        raise SupportGatedResidualError("v102 report has unsupported schema_version")
    if v102_report.get("v103_support_gated_residual_runtime_allowed") is not True:
        raise SupportGatedResidualError(
            "v102 report does not allow support-gated residual runtime feasibility"
        )
    from evolution_sim.mind.broad_branch_residual_distillation_example import (
        _training_rows,
        validate_broad_residual_distillation_example_artifact,
    )

    validate_broad_residual_distillation_example_artifact(v102_artifact)
    rows = _training_rows(v99_report=v99_report, v100_report=v100_report)
    threshold_report = _action_conditioned_threshold_report(rows=rows)
    support_examples = [
        _support_example_for_runtime(example)
        for example in _list_of_mappings(v102_artifact.get("support_examples"))
    ]
    action_counts = Counter(str(example.get("action", "")) for example in support_examples)
    contract = {
        "one_row_one_agent_local_decision": True,
        "requires_action_mask": True,
        "requires_policy_visible_features_only": True,
        "requires_linear_default_action": True,
        "requires_planner_outcome_tables": False,
        "requires_global_batch_assignment": False,
        "uses_heuristic_fallback": False,
        "uses_seed_id_as_runtime_feature": False,
        "uses_branch_id_as_runtime_feature": False,
        "uses_fixture_id_as_runtime_feature": False,
        "uses_logged_action_as_runtime_fallback": False,
        "uses_private_simulator_state": False,
    }
    artifact = {
        "schema_version": (
            MIND_V3_V104_SUPPORT_GATED_RESIDUAL_ARTIFACT_SCHEMA_VERSION
        ),
        "policy": MIND_V3_V104_SUPPORT_GATED_RESIDUAL_POLICY,
        "default_action_policy": "linear_mind_v3",
        "runtime_ready": True,
        "promotion_ready": False,
        "runtime_promotion_allowed": False,
        "training_policy": v102_artifact.get("training_policy"),
        "source_v102_schema_version": v102_report.get("schema_version"),
        "source_v102_report_digest": stable_payload_digest(
            {
                "schema_version": v102_report.get("schema_version"),
                "coverage": v102_report.get("coverage"),
                "acceptance": v102_report.get("acceptance"),
                "leave_one_source_seed_out_evaluation": v102_report.get(
                    "leave_one_source_seed_out_evaluation"
                ),
            }
        ),
        "scorer_rule": "action_prior_balanced_nearest_support_v1",
        "action_prior_penalty_scale": V103_ACTION_PRIOR_BALANCE_PENALTY,
        "support_gate": {
            "policy": MIND_V3_V104_ACTION_CONDITIONED_THRESHOLD_POLICY,
            "threshold_source": "v102_leave_one_source_seed_out_by_action_distribution_v1",
            "legal_action_required": True,
            "action_support_required": True,
            "distance_threshold_inclusive": True,
            "margin_threshold_inclusive": True,
            "threshold_scope": "selected_action",
            "distance_quantile": V103_DISTANCE_THRESHOLD_QUANTILE,
            "margin_quantile": V103_MARGIN_THRESHOLD_QUANTILE,
            "action_thresholds": threshold_report["action_thresholds"],
        },
        "inference_contract": contract,
        "training_row_count": int(v102_artifact.get("training_row_count", len(rows))),
        "support_action_counts": dict(sorted(action_counts.items())),
        "teacher_action_counts": dict(
            sorted(
                {
                    str(key): int(value)
                    for key, value in _mapping(
                        v102_artifact.get("teacher_action_counts")
                    ).items()
                    if isinstance(value, int) and not isinstance(value, bool)
                }.items()
            )
        ),
        "support_examples": support_examples,
        "threshold_diagnostics": threshold_report,
    }
    validate_support_gated_residual_artifact(artifact)
    return artifact, threshold_report


def score_support_gated_residual_artifact(
    *,
    artifact: Mapping[str, object],
    observation_input: Mapping[str, object],
    action_mask: Mapping[str, object],
    linear_action: str,
    public_history_trace: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    validate_support_gated_residual_artifact(artifact)
    gate = _mapping(artifact.get("support_gate"))
    return _score_support_gated_residual(
        support=_list_of_mappings(artifact.get("support_examples")),
        action_counts=_int_counter(artifact.get("support_action_counts")),
        train_count=max(_int(artifact.get("training_row_count")), 1),
        action_prior_penalty_scale=_float(
            artifact.get("action_prior_penalty_scale"),
            default=V103_ACTION_PRIOR_BALANCE_PENALTY,
        ),
        distance_threshold=_float(gate.get("nearest_support_distance_threshold")),
        margin_threshold=_float(gate.get("residual_score_margin_threshold")),
        action_thresholds=_mapping(gate.get("action_thresholds")),
        observation_input=observation_input,
        action_mask=action_mask,
        linear_action=linear_action,
        public_history_trace=public_history_trace,
    )


def build_branch_replay_feasibility_report(
    *,
    runtime_artifact: Mapping[str, object],
    v99_report: Mapping[str, object],
    v100_report: Mapping[str, object],
) -> dict[str, object]:
    validate_support_gated_residual_artifact(runtime_artifact)
    from evolution_sim.mind.broad_branch_residual_constrained_audit import (
        _candidate_rows,
    )
    from evolution_sim.mind.broad_branch_residual_distillation_example import (
        _training_rows,
    )

    rows = _training_rows(v99_report=v99_report, v100_report=v100_report)
    candidates_by_branch = {
        str(row.get("branch_id", "")): _list_of_mappings(row.get("candidates"))
        for row in _candidate_rows(v99_report)
    }
    comparisons = []
    for row in rows:
        policy_state = _mapping(row.get("policy_state"))
        provenance = _mapping(row.get("provenance"))
        linear_action = str(row.get("linear_action", ""))
        scored = score_support_gated_residual_artifact(
            artifact=runtime_artifact,
            observation_input=_mapping(policy_state.get("observation_input")),
            action_mask=_mapping(policy_state.get("action_mask")),
            public_history_trace=_list_of_mappings(
                policy_state.get("public_history_trace")
            ),
            linear_action=linear_action,
        )
        proposed_action = scored.get("selected_action")
        branch_id = str(provenance.get("branch_id", ""))
        candidate = _candidate_by_action(
            candidates_by_branch.get(branch_id, []),
            str(proposed_action),
        )
        override_allowed = bool(scored.get("override_allowed"))
        unsupported_candidate = override_allowed and candidate is None
        applied = override_allowed and candidate is not None
        target_delta = _float(_mapping(candidate).get("target_local_score_delta")) if applied else 0.0
        terminal_delta = _float(_mapping(candidate).get("terminal_alive_delta")) if applied else 0.0
        birth_delta = _float(_mapping(candidate).get("birth_delta")) if applied else 0.0
        target_alive_delta = _float(_mapping(candidate).get("target_alive_delta")) if applied else 0.0
        comparisons.append(
            {
                "row_index": row.get("row_index"),
                "source_seed": provenance.get("source_seed"),
                "branch_tick": provenance.get("branch_tick"),
                "linear_action": linear_action,
                "teacher_action": row.get("teacher_action"),
                "proposed_action": proposed_action,
                "applied_override_action": proposed_action if applied else None,
                "override_proposed": bool(scored.get("override_proposed")),
                "override_allowed": override_allowed,
                "override_applied": applied,
                "abstention_reason": scored.get("abstention_reason"),
                "nearest_support_distance": scored.get("nearest_support_distance"),
                "score_margin": scored.get("score_margin"),
                "unsupported_proposed_action": bool(
                    scored.get("unsupported_proposed_action")
                ),
                "unsupported_candidate_action": unsupported_candidate,
                "target_local_score_delta": _round(target_delta),
                "terminal_alive_delta": _round(terminal_delta),
                "birth_delta": _round(birth_delta),
                "target_alive_delta": _round(target_alive_delta),
            }
        )
    summary = _branch_replay_summary(comparisons)
    gate = _branch_replay_gate(summary=summary, source_report=v99_report)
    return {
        "schema_version": MIND_V3_V103_BRANCH_REPLAY_FEASIBILITY_SCHEMA_VERSION,
        "policy": runtime_artifact.get("policy"),
        "runtime_artifact_schema_version": runtime_artifact.get("schema_version"),
        "runtime_promotion_allowed": False,
        "source_replay_verified": bool(
            _mapping(v99_report.get("aggregate")).get("replay_verified")
        ),
        "source_unsupported_candidate_action_count": int(
            _mapping(v99_report.get("aggregate")).get(
                "unsupported_candidate_action_count", 0
            )
        ),
        "summary": summary,
        "safety_gate": gate,
        "comparisons": comparisons[:128],
        "v103_branch_replay_feasibility_passed": bool(gate["passed"]),
    }


def build_v104_branch_replay_feasibility_report(
    *,
    runtime_artifact: Mapping[str, object],
    v99_report: Mapping[str, object],
    v100_report: Mapping[str, object],
) -> dict[str, object]:
    report = build_branch_replay_feasibility_report(
        runtime_artifact=runtime_artifact,
        v99_report=v99_report,
        v100_report=v100_report,
    )
    report["schema_version"] = MIND_V3_V104_BRANCH_REPLAY_FEASIBILITY_SCHEMA_VERSION
    report["v104_branch_replay_feasibility_passed"] = bool(
        report.pop("v103_branch_replay_feasibility_passed")
    )
    return report


def support_gated_residual_runtime_diagnostics(
    decision_diagnostics: Sequence[Mapping[str, object] | None],
    *,
    run_count: int = 1,
) -> dict[str, object]:
    total = len(decision_diagnostics)
    decision_count = 0
    proposed_override_count = 0
    gate_accepted_override_count = 0
    applied_override_count = 0
    shadowed_override_count = 0
    unsupported_count = 0
    proposed_counts: Counter[str] = Counter()
    proposed_override_counts: Counter[str] = Counter()
    gate_accepted_counts: Counter[str] = Counter()
    applied_counts: Counter[str] = Counter()
    shadowed_counts: Counter[str] = Counter()
    abstention_reasons: Counter[str] = Counter()
    distances = []
    margins = []
    accepted_distances = []
    accepted_margins = []
    observed_policies: Counter[str] = Counter()
    for diagnostic in decision_diagnostics:
        if not isinstance(diagnostic, Mapping):
            continue
        policy = diagnostic.get("support_residual_policy")
        if policy not in SUPPORT_GATED_RESIDUAL_POLICIES:
            continue
        observed_policies.update([str(policy)])
        decision_count += 1
        proposed = _diagnostic_action(diagnostic, "support_residual_proposed_action")
        if proposed is not None:
            proposed_counts.update([proposed])
        if diagnostic.get("support_residual_override_proposed") is True:
            proposed_override_count += 1
            if proposed is not None:
                proposed_override_counts.update([proposed])
        if diagnostic.get("support_residual_override_allowed") is True:
            gate_accepted_override_count += 1
            if proposed is not None:
                gate_accepted_counts.update([proposed])
        if diagnostic.get("support_residual_override_applied") is True:
            applied_override_count += 1
            if proposed is not None:
                applied_counts.update([proposed])
        if diagnostic.get("support_residual_shadowed") is True:
            shadowed_override_count += 1
            if proposed is not None:
                shadowed_counts.update([proposed])
        if diagnostic.get("support_residual_unsupported_proposed_action") is True:
            unsupported_count += 1
        reason = diagnostic.get("support_residual_abstention_reason")
        if isinstance(reason, str) and reason:
            abstention_reasons.update([reason])
        distance = _optional_float(diagnostic.get("support_residual_nearest_support_distance"))
        margin = _optional_float(diagnostic.get("support_residual_score_margin"))
        if distance is not None:
            distances.append(distance)
        if margin is not None:
            margins.append(margin)
        if diagnostic.get("support_residual_override_allowed") is True:
            if distance is not None:
                accepted_distances.append(distance)
            if margin is not None:
                accepted_margins.append(margin)
    gate_dominant = _dominant_count_share(gate_accepted_counts)
    proposed_dominant = _dominant_count_share(proposed_override_counts)
    applied_dominant = _dominant_count_share(applied_counts)
    diagnostics_policy = (
        MIND_V3_V104_RUNTIME_DIAGNOSTICS_POLICY
        if observed_policies.get(MIND_V3_V104_SUPPORT_GATED_RESIDUAL_POLICY, 0) > 0
        else MIND_V3_V103_RUNTIME_DIAGNOSTICS_POLICY
    )
    return {
        "policy": diagnostics_policy,
        "run_count": int(run_count),
        "total_decision_count": total,
        "decision_count": decision_count,
        "proposed_override_count": proposed_override_count,
        "proposed_override_share": _safe_rate(proposed_override_count, decision_count),
        "gate_accepted_override_count": gate_accepted_override_count,
        "gate_accepted_override_share": _safe_rate(
            gate_accepted_override_count,
            decision_count,
        ),
        "applied_override_count": applied_override_count,
        "applied_override_share": _safe_rate(applied_override_count, decision_count),
        "shadowed_override_count": shadowed_override_count,
        "shadowed_override_share": _safe_rate(shadowed_override_count, decision_count),
        "unsupported_proposed_action_count": unsupported_count,
        "proposed_action_counts": dict(sorted(proposed_counts.items())),
        "proposed_override_action_counts": dict(sorted(proposed_override_counts.items())),
        "gate_accepted_override_action_counts": dict(sorted(gate_accepted_counts.items())),
        "applied_override_action_counts": dict(sorted(applied_counts.items())),
        "shadowed_override_action_counts": dict(sorted(shadowed_counts.items())),
        "dominant_proposed_override_action": proposed_dominant["key"],
        "dominant_proposed_override_action_share": proposed_dominant["share"],
        "dominant_gate_accepted_override_action": gate_dominant["key"],
        "dominant_gate_accepted_override_action_share": gate_dominant["share"],
        "dominant_applied_override_action": applied_dominant["key"],
        "dominant_applied_override_action_share": applied_dominant["share"],
        "abstention_reason_counts": dict(sorted(abstention_reasons.items())),
        "nearest_support_distance_stats": _float_summary(distances),
        "score_margin_stats": _float_summary(margins),
        "gate_accepted_nearest_support_distance_stats": _float_summary(
            accepted_distances
        ),
        "gate_accepted_score_margin_stats": _float_summary(accepted_margins),
    }


def aggregate_support_gated_residual_runtime_diagnostics(
    runs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    diagnostics = []
    observed_policy = MIND_V3_V103_RUNTIME_DIAGNOSTICS_POLICY
    for run in runs:
        payload = run.get("support_residual_diagnostics")
        if isinstance(payload, Mapping):
            diagnostics.append(payload)
            if payload.get("policy") == MIND_V3_V104_RUNTIME_DIAGNOSTICS_POLICY:
                observed_policy = MIND_V3_V104_RUNTIME_DIAGNOSTICS_POLICY
    total_decision_count = sum(int(item.get("total_decision_count", 0)) for item in diagnostics)
    decision_count = sum(int(item.get("decision_count", 0)) for item in diagnostics)
    proposed_override_count = sum(int(item.get("proposed_override_count", 0)) for item in diagnostics)
    gate_accepted_override_count = sum(int(item.get("gate_accepted_override_count", 0)) for item in diagnostics)
    applied_override_count = sum(int(item.get("applied_override_count", 0)) for item in diagnostics)
    shadowed_override_count = sum(int(item.get("shadowed_override_count", 0)) for item in diagnostics)
    unsupported_count = sum(int(item.get("unsupported_proposed_action_count", 0)) for item in diagnostics)
    proposed_counts: Counter[str] = Counter()
    proposed_override_counts: Counter[str] = Counter()
    gate_accepted_counts: Counter[str] = Counter()
    applied_counts: Counter[str] = Counter()
    shadowed_counts: Counter[str] = Counter()
    abstention_reasons: Counter[str] = Counter()
    for item in diagnostics:
        proposed_counts.update(_int_counter(item.get("proposed_action_counts")))
        proposed_override_counts.update(_int_counter(item.get("proposed_override_action_counts")))
        gate_accepted_counts.update(_int_counter(item.get("gate_accepted_override_action_counts")))
        applied_counts.update(_int_counter(item.get("applied_override_action_counts")))
        shadowed_counts.update(_int_counter(item.get("shadowed_override_action_counts")))
        abstention_reasons.update(_int_counter(item.get("abstention_reason_counts")))
    proposed_dominant = _dominant_count_share(proposed_override_counts)
    gate_dominant = _dominant_count_share(gate_accepted_counts)
    applied_dominant = _dominant_count_share(applied_counts)
    return {
        "policy": observed_policy,
        "run_count": len(runs),
        "total_decision_count": total_decision_count,
        "decision_count": decision_count,
        "proposed_override_count": proposed_override_count,
        "proposed_override_share": _safe_rate(proposed_override_count, decision_count),
        "gate_accepted_override_count": gate_accepted_override_count,
        "gate_accepted_override_share": _safe_rate(
            gate_accepted_override_count,
            decision_count,
        ),
        "applied_override_count": applied_override_count,
        "applied_override_share": _safe_rate(applied_override_count, decision_count),
        "shadowed_override_count": shadowed_override_count,
        "shadowed_override_share": _safe_rate(shadowed_override_count, decision_count),
        "unsupported_proposed_action_count": unsupported_count,
        "proposed_action_counts": dict(sorted(proposed_counts.items())),
        "proposed_override_action_counts": dict(sorted(proposed_override_counts.items())),
        "gate_accepted_override_action_counts": dict(sorted(gate_accepted_counts.items())),
        "applied_override_action_counts": dict(sorted(applied_counts.items())),
        "shadowed_override_action_counts": dict(sorted(shadowed_counts.items())),
        "dominant_proposed_override_action": proposed_dominant["key"],
        "dominant_proposed_override_action_share": proposed_dominant["share"],
        "dominant_gate_accepted_override_action": gate_dominant["key"],
        "dominant_gate_accepted_override_action_share": gate_dominant["share"],
        "dominant_applied_override_action": applied_dominant["key"],
        "dominant_applied_override_action_share": applied_dominant["share"],
        "abstention_reason_counts": dict(sorted(abstention_reasons.items())),
    }


def _threshold_report(
    *,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    records = _leave_one_source_seed_out_score_records(rows)
    distances = [
        _float(item.get("nearest_support_distance"))
        for item in records
        if _optional_float(item.get("nearest_support_distance")) is not None
    ]
    margins = [
        _float(item.get("score_margin"))
        for item in records
        if _optional_float(item.get("score_margin")) is not None
    ]
    distance_threshold = _round(_quantile(distances, V103_DISTANCE_THRESHOLD_QUANTILE))
    margin_threshold = _round(_quantile(margins, V103_MARGIN_THRESHOLD_QUANTILE))
    proposed_counts = Counter(str(item.get("selected_action", "")) for item in records)
    changed_count = sum(1 for item in records if item.get("override_proposed") is True)
    return {
        "policy": MIND_V3_V103_THRESHOLD_POLICY,
        "threshold_source": "v102_leave_one_source_seed_out_distribution_v1",
        "record_count": len(records),
        "nearest_support_distance_threshold": distance_threshold,
        "residual_score_margin_threshold": margin_threshold,
        "nearest_support_distance_distribution": _float_summary(distances),
        "score_margin_distribution": _float_summary(margins),
        "loo_selected_action_counts": dict(sorted(proposed_counts.items())),
        "loo_proposed_override_count": changed_count,
        "loo_proposed_override_share": _safe_rate(changed_count, len(records)),
        "action_prior_penalty_scale": V103_ACTION_PRIOR_BALANCE_PENALTY,
    }


def _action_conditioned_threshold_report(
    *,
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    records = _leave_one_source_seed_out_score_records(rows)
    by_action: dict[str, list[Mapping[str, object]]] = {}
    for item in records:
        action = str(item.get("selected_action", ""))
        if action in ACTION_NAMES:
            by_action.setdefault(action, []).append(item)
    action_thresholds: dict[str, dict[str, object]] = {}
    for action in ACTION_NAMES:
        action_records = by_action.get(action, [])
        if not action_records:
            continue
        distances = [
            _float(item.get("nearest_support_distance"))
            for item in action_records
            if _optional_float(item.get("nearest_support_distance")) is not None
        ]
        margins = [
            _float(item.get("score_margin"))
            for item in action_records
            if _optional_float(item.get("score_margin")) is not None
        ]
        changed_count = sum(
            1 for item in action_records if item.get("override_proposed") is True
        )
        action_thresholds[action] = {
            "nearest_support_distance_threshold": _round(
                _quantile(distances, V103_DISTANCE_THRESHOLD_QUANTILE)
            ),
            "residual_score_margin_threshold": _round(
                _quantile(margins, V103_MARGIN_THRESHOLD_QUANTILE)
            ),
            "distance_quantile": V103_DISTANCE_THRESHOLD_QUANTILE,
            "margin_quantile": V103_MARGIN_THRESHOLD_QUANTILE,
            "calibration_record_count": len(action_records),
            "calibration_override_proposed_count": changed_count,
            "calibration_override_proposed_share": _safe_rate(
                changed_count,
                len(action_records),
            ),
            "nearest_support_distance_distribution": _float_summary(distances),
            "score_margin_distribution": _float_summary(margins),
        }
    selected_counts = Counter(str(item.get("selected_action", "")) for item in records)
    changed_count = sum(1 for item in records if item.get("override_proposed") is True)
    return {
        "policy": MIND_V3_V104_ACTION_CONDITIONED_THRESHOLD_POLICY,
        "threshold_source": "v102_leave_one_source_seed_out_by_action_distribution_v1",
        "record_count": len(records),
        "action_thresholds": dict(sorted(action_thresholds.items())),
        "loo_selected_action_counts": dict(sorted(selected_counts.items())),
        "loo_proposed_override_count": changed_count,
        "loo_proposed_override_share": _safe_rate(changed_count, len(records)),
        "distance_quantile": V103_DISTANCE_THRESHOLD_QUANTILE,
        "margin_quantile": V103_MARGIN_THRESHOLD_QUANTILE,
        "action_prior_penalty_scale": V103_ACTION_PRIOR_BALANCE_PENALTY,
    }


def _leave_one_source_seed_out_score_records(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    from evolution_sim.mind.broad_branch_residual_distillation_example import (
        _distillation_example_artifact,
    )

    records = []
    source_seeds = sorted(
        {
            int(_mapping(row.get("provenance")).get("source_seed", 0))
            for row in rows
        }
    )
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
        support_artifact = _distillation_example_artifact(train_rows)
        support = _list_of_mappings(support_artifact.get("support_examples"))
        action_counts = Counter(str(row.get("teacher_action", "")) for row in train_rows)
        for row in heldout_rows:
            policy_state = _mapping(row.get("policy_state"))
            records.append(
                {
                    "held_out_seed": held_out_seed,
                    **_score_support_gated_residual(
                        support=support,
                        action_counts=action_counts,
                        train_count=max(len(train_rows), 1),
                        action_prior_penalty_scale=V103_ACTION_PRIOR_BALANCE_PENALTY,
                        distance_threshold=float("inf"),
                        margin_threshold=0.0,
                        action_thresholds={},
                        observation_input=_mapping(
                            policy_state.get("observation_input")
                        ),
                        action_mask=_mapping(policy_state.get("action_mask")),
                        public_history_trace=_list_of_mappings(
                            policy_state.get("public_history_trace")
                        ),
                        linear_action=str(row.get("linear_action", "")),
                    ),
                }
            )
    return records


def _score_support_gated_residual(
    *,
    support: Sequence[Mapping[str, object]],
    action_counts: Mapping[str, int],
    train_count: int,
    action_prior_penalty_scale: float,
    distance_threshold: float,
    margin_threshold: float,
    action_thresholds: Mapping[str, object],
    observation_input: Mapping[str, object],
    action_mask: Mapping[str, object],
    linear_action: str,
    public_history_trace: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    row = planner_distilled_runtime_row(
        observation_input=observation_input,
        action_mask=action_mask,
        public_history_trace=public_history_trace,
    )
    support_counts = Counter(str(example.get("action", "")) for example in support)
    candidate_scores = []
    for action in _legal_actions(action_mask):
        features = candidate_feature_vector(row, action)
        if not features:
            continue
        distance, weight = _nearest_support(features, support, action=action)
        supported = support_counts[action] > 0 and math.isfinite(distance)
        if supported:
            action_prior_share = _float(action_counts.get(action)) / float(
                max(train_count, 1)
            )
            score = (
                -distance
                + 0.0001 * weight
                - action_prior_penalty_scale * action_prior_share
            )
        else:
            score = float("-inf")
        candidate_scores.append(
            {
                "action": action,
                "mode": _action_option_mode(action),
                "legal": True,
                "support_count": int(support_counts[action]),
                "nearest_support_distance": _round(distance)
                if math.isfinite(distance)
                else None,
                "support_weight": _round(weight),
                "score": _round(score) if math.isfinite(score) else None,
            }
        )
    finite_candidates = [
        item for item in candidate_scores if _optional_float(item.get("score")) is not None
    ]
    if not finite_candidates:
        return {
            "selected_action": None,
            "selected_mode": None,
            "selected_score": None,
            "nearest_support_distance": None,
            "support_weight": 0.0,
            "score_margin": 0.0,
            "candidate_scores": candidate_scores,
            "candidate_scores_top": [],
            "override_proposed": False,
            "override_allowed": False,
            "support_gate_passed": False,
            "unsupported_proposed_action": True,
            "abstention_reason": "no_supported_legal_action",
            "public_history_steps": len(public_history_trace),
        }
    ordered = sorted(
        finite_candidates,
        key=lambda item: (
            _float(item.get("score"), default=float("-inf")),
            str(item.get("action", "")),
        ),
        reverse=True,
    )
    selected = ordered[0]
    selected_action = str(selected.get("action", ""))
    selected_distance = _optional_float(selected.get("nearest_support_distance"))
    selected_score = _optional_float(selected.get("score"))
    score_margin = (
        _round(_float(ordered[0].get("score")) - _float(ordered[1].get("score")))
        if len(ordered) >= 2
        else 0.0
    )
    legal_gate = selected_action in _legal_actions(action_mask)
    action_support_gate = int(selected.get("support_count", 0)) > 0
    selected_thresholds = _selected_action_thresholds(
        action_thresholds=action_thresholds,
        action=selected_action,
        default_distance_threshold=distance_threshold,
        default_margin_threshold=margin_threshold,
    )
    selected_distance_threshold = selected_thresholds["distance_threshold"]
    selected_margin_threshold = selected_thresholds["margin_threshold"]
    distance_gate = (
        selected_distance is not None
        and selected_distance <= float(selected_distance_threshold)
    )
    margin_gate = score_margin >= float(selected_margin_threshold)
    support_gate_passed = legal_gate and action_support_gate and distance_gate and margin_gate
    override_proposed = selected_action != linear_action
    override_allowed = support_gate_passed and override_proposed
    unsupported = not legal_gate or not action_support_gate
    return {
        "selected_action": selected_action,
        "selected_mode": selected.get("mode"),
        "selected_score": selected_score,
        "nearest_support_distance": selected_distance,
        "support_weight": selected.get("support_weight"),
        "score_margin": score_margin,
        "distance_threshold": _round(selected_distance_threshold),
        "margin_threshold": _round(selected_margin_threshold),
        "threshold_scope": selected_thresholds["scope"],
        "candidate_scores": candidate_scores,
        "candidate_scores_top": [
            _candidate_score_summary(item) for item in ordered[:5]
        ],
        "linear_action": linear_action,
        "override_proposed": override_proposed,
        "override_allowed": override_allowed,
        "support_gate_passed": support_gate_passed,
        "legal_action_gate_passed": legal_gate,
        "action_support_gate_passed": action_support_gate,
        "distance_gate_passed": distance_gate,
        "margin_gate_passed": margin_gate,
        "unsupported_proposed_action": unsupported,
        "abstention_reason": _abstention_reason(
            legal_gate=legal_gate,
            action_support_gate=action_support_gate,
            distance_gate=distance_gate,
            margin_gate=margin_gate,
            override_proposed=override_proposed,
        ),
        "public_history_steps": len(public_history_trace),
    }


def _branch_replay_summary(
    comparisons: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    applied = [
        item for item in comparisons if item.get("override_applied") is True
    ]
    proposed = [
        item for item in comparisons if item.get("override_proposed") is True
    ]
    row_count = len(comparisons)
    applied_counts = Counter(str(item.get("applied_override_action")) for item in applied)
    proposed_counts = Counter(str(item.get("proposed_action")) for item in proposed)
    applied_dominant = _dominant_count_share(applied_counts)
    by_seed = []
    source_seeds = sorted(
        {
            int(item.get("source_seed", 0))
            for item in comparisons
            if isinstance(item.get("source_seed"), int)
        }
    )
    for seed in source_seeds:
        seed_rows = [
            item for item in comparisons if int(item.get("source_seed", 0)) == seed
        ]
        by_seed.append(
            {
                "source_seed": seed,
                "row_count": len(seed_rows),
                "applied_override_count": sum(
                    1 for item in seed_rows if item.get("override_applied") is True
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
                    _mean(
                        [
                            _float(item.get("terminal_alive_delta"))
                            for item in seed_rows
                        ]
                    )
                ),
                "birth_delta_mean": _round(
                    _mean([_float(item.get("birth_delta")) for item in seed_rows])
                ),
                "target_alive_delta_negative_count": sum(
                    1
                    for item in seed_rows
                    if _float(item.get("target_alive_delta")) < 0.0
                ),
            }
        )
    return {
        "row_count": row_count,
        "proposed_override_count": len(proposed),
        "proposed_override_share": _safe_rate(len(proposed), row_count),
        "applied_override_count": len(applied),
        "applied_override_share": _safe_rate(len(applied), row_count),
        "abstention_count": row_count - len(applied),
        "abstention_share": _safe_rate(row_count - len(applied), row_count),
        "proposed_override_action_counts": dict(sorted(proposed_counts.items())),
        "applied_override_action_counts": dict(sorted(applied_counts.items())),
        "dominant_applied_override_action": applied_dominant["key"],
        "dominant_applied_override_action_count": applied_dominant["count"],
        "dominant_applied_override_action_share": applied_dominant["share"],
        "unsupported_proposed_action_count": sum(
            1 for item in comparisons if item.get("unsupported_proposed_action") is True
        ),
        "unsupported_candidate_action_count": sum(
            1 for item in comparisons if item.get("unsupported_candidate_action") is True
        ),
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
        "nearest_support_distance_stats": _float_summary(
            [
                _float(item.get("nearest_support_distance"))
                for item in comparisons
                if _optional_float(item.get("nearest_support_distance")) is not None
            ]
        ),
        "score_margin_stats": _float_summary(
            [
                _float(item.get("score_margin"))
                for item in comparisons
                if _optional_float(item.get("score_margin")) is not None
            ]
        ),
        "per_source_seed": by_seed,
    }


def _branch_replay_gate(
    *,
    summary: Mapping[str, object],
    source_report: Mapping[str, object],
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

    aggregate = _mapping(source_report.get("aggregate"))
    if aggregate.get("replay_verified") is not True:
        block("source_replay_not_verified", "source_replay_verified", aggregate.get("replay_verified"), True, "eq")
    if _int(aggregate.get("unsupported_candidate_action_count")) != 0:
        block("source_unsupported_candidate_action", "source_unsupported_candidate_action_count", aggregate.get("unsupported_candidate_action_count"), 0, "eq")
    if _int(summary.get("unsupported_proposed_action_count")) != 0:
        block("unsupported_proposed_action", "unsupported_proposed_action_count", summary.get("unsupported_proposed_action_count"), 0, "eq")
    if _int(summary.get("unsupported_candidate_action_count")) != 0:
        block("unsupported_candidate_action", "unsupported_candidate_action_count", summary.get("unsupported_candidate_action_count"), 0, "eq")
    if _int(summary.get("applied_override_count")) <= 0:
        block("no_applied_overrides", "applied_override_count", summary.get("applied_override_count"), 0, "gt")
    if _int(summary.get("target_alive_delta_negative_count")) != 0:
        block("target_alive_delta_negative", "target_alive_delta_negative_count", summary.get("target_alive_delta_negative_count"), 0, "eq")
    if _float(summary.get("target_local_score_delta_mean")) <= 0.0:
        block("mean_target_local_score_delta_not_positive", "target_local_score_delta_mean", summary.get("target_local_score_delta_mean"), 0.0, "gt")
    if _float(summary.get("terminal_alive_delta_mean")) < 0.0:
        block("mean_terminal_alive_delta_negative", "terminal_alive_delta_mean", summary.get("terminal_alive_delta_mean"), 0.0, "ge")
    if _float(summary.get("birth_delta_mean")) < 0.0:
        block("mean_birth_delta_negative", "birth_delta_mean", summary.get("birth_delta_mean"), 0.0, "ge")
    if (
        _float(summary.get("dominant_applied_override_action_share"))
        > V103_MAX_DOMINANT_APPLIED_OVERRIDE_ACTION_SHARE
    ):
        block(
            "dominant_applied_override_action_share_above_cap",
            "dominant_applied_override_action_share",
            summary.get("dominant_applied_override_action_share"),
            V103_MAX_DOMINANT_APPLIED_OVERRIDE_ACTION_SHARE,
            "le",
        )
    for item in _list_of_mappings(summary.get("per_source_seed")):
        if _float(item.get("target_local_score_delta_mean")) < 0.0:
            block(
                "per_source_seed_target_local_mean_negative",
                f"per_source_seed[{item.get('source_seed')}].target_local_score_delta_mean",
                item.get("target_local_score_delta_mean"),
                0.0,
                "ge",
            )
    return {
        "passed": not blockers,
        "blocker_count": len(blockers),
        "blockers": blockers,
        "runtime_promotion_allowed": False,
    }


def _support_example_for_runtime(example: Mapping[str, object]) -> dict[str, object]:
    return {
        "example_index": int(example.get("example_index", 0)),
        "action": str(example.get("action", "")),
        "mode": str(example.get("mode", "")),
        "feature_vector": _float_list(example.get("feature_vector")),
        "weight": _round(_float(example.get("weight"))),
    }


def _candidate_by_action(
    candidates: Sequence[Mapping[str, object]],
    action: str,
) -> Mapping[str, object] | None:
    for candidate in candidates:
        if str(candidate.get("action", "")) == action:
            return candidate
    return None


def _action_option_mode(action: str) -> str:
    if action == "eat":
        return "exploit_resource"
    if action == "drink":
        return "recover_hydration"
    if action.startswith("move_"):
        return "reposition"
    if action == "mate":
        return "reproduction"
    if action.startswith("attack_"):
        return "hunt"
    return "conserve"


def _candidate_score_summary(item: Mapping[str, object]) -> dict[str, object]:
    return {
        "action": item.get("action"),
        "mode": item.get("mode"),
        "support_count": item.get("support_count"),
        "nearest_support_distance": item.get("nearest_support_distance"),
        "support_weight": item.get("support_weight"),
        "score": item.get("score"),
    }


def _selected_action_thresholds(
    *,
    action_thresholds: Mapping[str, object],
    action: str,
    default_distance_threshold: float,
    default_margin_threshold: float,
) -> dict[str, object]:
    thresholds = _mapping(action_thresholds.get(action))
    if thresholds:
        return {
            "scope": "selected_action",
            "distance_threshold": _float(
                thresholds.get("nearest_support_distance_threshold"),
                default=float("inf"),
            ),
            "margin_threshold": _float(
                thresholds.get("residual_score_margin_threshold"),
                default=float("inf"),
            ),
        }
    return {
        "scope": "global",
        "distance_threshold": float(default_distance_threshold),
        "margin_threshold": float(default_margin_threshold),
    }


def _abstention_reason(
    *,
    legal_gate: bool,
    action_support_gate: bool,
    distance_gate: bool,
    margin_gate: bool,
    override_proposed: bool,
) -> str | None:
    if not legal_gate:
        return "illegal_action"
    if not action_support_gate:
        return "unsupported_action"
    if not distance_gate:
        return "distance_above_threshold"
    if not margin_gate:
        return "margin_below_threshold"
    if not override_proposed:
        return "matched_linear_action"
    return None


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


def _contains_forbidden_runtime_keys(value: object) -> bool:
    forbidden = {
        "seed",
        "seed_id",
        "source_seed",
        "branch_id",
        "fixture",
        "fixture_id",
        "fixture_identity",
        "agent_id",
        "logged_action",
        "strict_eval_label_identity",
        "planner_candidate_outcome_table",
        "global_batch_action_quota",
        "private_simulator_state",
        "private_simulation_world_state",
    }
    if isinstance(value, Mapping):
        if set(value) & forbidden:
            return True
        return any(_contains_forbidden_runtime_keys(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_runtime_keys(item) for item in value)
    return False


def _legal_actions(action_mask: Mapping[str, object]) -> list[str]:
    return [action for action in ACTION_NAMES if bool(action_mask.get(action, False))]


def _float_summary(values: Sequence[float]) -> dict[str, object]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": 0.0,
            "p25": None,
            "p50": None,
            "p75": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": _round(ordered[0]),
        "max": _round(ordered[-1]),
        "mean": _round(_mean(ordered)),
        "p25": _round(_quantile(ordered, 0.25)),
        "p50": _round(_quantile(ordered, 0.50)),
        "p75": _round(_quantile(ordered, 0.75)),
    }


def _quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(quantile))) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


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


def _int_counter(value: object) -> Counter[str]:
    counter: Counter[str] = Counter()
    if not isinstance(value, Mapping):
        return counter
    for key, count in value.items():
        if isinstance(count, int) and not isinstance(count, bool):
            counter[str(key)] = int(count)
    return counter


def _diagnostic_action(diagnostic: Mapping[str, object], key: str) -> str | None:
    value = diagnostic.get(key)
    if not isinstance(value, str) or not value or value == "none":
        return None
    return value


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


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _float_list(value: object) -> list[float]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return []
        parsed = float(item)
        if not math.isfinite(parsed):
            return []
        result.append(parsed)
    return result


def _mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def _safe_rate(numerator: int | float, denominator: int | float) -> float:
    denom = float(denominator)
    if denom <= 0.0:
        return 0.0
    return _round(float(numerator) / denom)


def _round(value: float) -> float:
    return round(float(value), 6)
