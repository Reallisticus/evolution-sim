from __future__ import annotations

from collections.abc import Mapping, Sequence

from evolution_sim.mind.dataset import (
    TrajectoryTransition,
    build_trajectory_transitions,
)

VIABILITY_SUPPRESSION_COMPONENT = "hard_guard_or_delegate_suppression"
VIABILITY_COMPONENT_NAMES: tuple[str, ...] = (
    "death_or_survival_horizon_risk",
    "energy_floor_risk",
    "hydration_floor_risk",
    "health_floor_risk",
    "invalid_action",
    VIABILITY_SUPPRESSION_COMPONENT,
)
VIABILITY_HEAD_POLICY = "multi_component_constraint_viability_head_v1"
VIABILITY_ACTION_HEAD_POLICY = (
    "action_conditioned_multi_component_constraint_viability_head_v1"
)
VIABILITY_ACTION_SUPERVISION_POLICY = (
    "logged_action_components_plus_learned_action_suppression_v2"
)
VIABILITY_TARGET_POLICY = (
    "survival_floor_invalid_suppression_constraint_target_v0"
)
VIABILITY_REPRODUCTION_DIAGNOSTIC_POLICY = (
    "reproduction_viability_diagnostic_count_v0"
)
VIABILITY_RUNTIME_DECISION_POLICY = "diagnostics_only_not_used_for_runtime_v0"
VIABILITY_SURVIVAL_HORIZON_TICKS = 8
VIABILITY_FLOOR_RISK_RATIO = 0.15
VIABILITY_HEALTH_FLOOR_RISK_RATIO = 0.2
HEURISTIC_GUARD_POLICY = "observation_heuristic_safety_floor_v1"
HEURISTIC_DELEGATE_POLICY = "observation_heuristic_confidence_delegate_v1"


def build_viability_component_targets(
    records: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, bool], ...], tuple[bool, ...], tuple[int, ...]]:
    record_dicts = [dict(record) for record in records]
    transitions = build_trajectory_transitions(record_dicts)
    if len(transitions) != len(record_dicts):
        return (), (), ()
    survival_horizon_risks, survival_horizons = _survival_horizon_risks(
        record_dicts,
        transitions,
    )
    component_targets: list[dict[str, bool]] = []
    for index, (record, transition) in enumerate(
        zip(record_dicts, transitions, strict=True)
    ):
        component_targets.append(
            _viability_constraint_components(
                record,
                transition,
                survival_horizon_risk=survival_horizon_risks[index],
            )
        )
    aggregate_targets = [
        any(components.values())
        for components in component_targets
    ]
    return (
        tuple(component_targets),
        tuple(aggregate_targets),
        tuple(survival_horizons),
    )


def reproduction_viable(record: Mapping[str, object]) -> bool:
    outcome = _mapping(record.get("outcome"))
    reward = _mapping(record.get("reward"))
    components = _mapping(reward.get("components"))
    return (
        outcome.get("reproduced") is True
        or outcome.get("reproduction_ready_after") is True
        or _number(components.get("reproduction_readiness")) > 0.0
        or _number(components.get("reproduction_success")) > 0.0
    )


def _survival_horizon_risks(
    records: Sequence[Mapping[str, object]],
    transitions: Sequence[TrajectoryTransition],
) -> tuple[list[bool], list[int]]:
    risk_flags = [False for _ in transitions]
    observed_horizons = [0 for _ in transitions]
    transition_indices_by_agent: dict[tuple[str, int], list[int]] = {}
    for index, transition in enumerate(transitions):
        transition_indices_by_agent.setdefault(
            (transition.episode_id, transition.agent_id),
            [],
        ).append(index)

    for indices in transition_indices_by_agent.values():
        for position, index in enumerate(indices):
            observed = 0
            risk = _record_death(records[index])
            if not risk:
                for future_index in indices[
                    position + 1 : position + 1 + VIABILITY_SURVIVAL_HORIZON_TICKS
                ]:
                    observed += 1
                    if _record_death(records[future_index]):
                        risk = True
                        break
            risk_flags[index] = risk
            observed_horizons[index] = observed
    return risk_flags, observed_horizons


def _viability_constraint_components(
    record: Mapping[str, object],
    transition: TrajectoryTransition,
    *,
    survival_horizon_risk: bool,
) -> dict[str, bool]:
    after = _mapping(record.get("after"))
    hard_guard_or_delegate = _transition_suppressed_by_fallback(transition)
    return {
        "death_or_survival_horizon_risk": _record_death(record)
        or survival_horizon_risk,
        "energy_floor_risk": _ratio_at_or_below(
            after,
            "energy_ratio",
            VIABILITY_FLOOR_RISK_RATIO,
        ),
        "hydration_floor_risk": _ratio_at_or_below(
            after,
            "hydration_ratio",
            VIABILITY_FLOOR_RISK_RATIO,
        ),
        "health_floor_risk": _ratio_at_or_below(
            after,
            "health_ratio",
            VIABILITY_HEALTH_FLOOR_RISK_RATIO,
        ),
        "invalid_action": _invalid_action(record),
        "hard_guard_or_delegate_suppression": hard_guard_or_delegate,
    }


def _transition_suppressed_by_fallback(transition: TrajectoryTransition) -> bool:
    diagnostics = transition.policy_decision_diagnostics
    return (
        HEURISTIC_GUARD_POLICY in transition.action_source
        or HEURISTIC_DELEGATE_POLICY in transition.action_source
        or _diagnostic_bool(diagnostics or {}, "guard_used")
        or _diagnostic_bool(diagnostics or {}, "heuristic_delegate_used")
    )


def _record_death(record: Mapping[str, object]) -> bool:
    after = _mapping(record.get("after"))
    outcome = _mapping(record.get("outcome"))
    passive = _mapping(outcome.get("passive"))
    return (
        after.get("alive") is False
        or outcome.get("died") is True
        or passive.get("died_after_action") is True
        or passive.get("died_before_action") is True
        or passive.get("killed") is True
    )


def _invalid_action(record: Mapping[str, object]) -> bool:
    outcome = _mapping(record.get("outcome"))
    return (
        record.get("action_valid") is False
        or record.get("resolution_action_valid") is False
        or outcome.get("observation_action_valid") is False
        or outcome.get("invalid_reason") is not None
    )


def _ratio_at_or_below(
    payload: Mapping[str, object],
    field: str,
    threshold: float,
) -> bool:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return float(value) <= threshold


def _diagnostic_bool(
    diagnostic: Mapping[str, object],
    key: str,
) -> bool:
    return diagnostic.get(key) is True


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _mapping(payload: object) -> Mapping[str, object]:
    return payload if isinstance(payload, Mapping) else {}
