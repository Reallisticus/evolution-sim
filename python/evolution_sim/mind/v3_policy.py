from __future__ import annotations

from collections.abc import Mapping, Sequence
from random import Random

from evolution_sim.env.runtime.observations import (
    NAVIGATION_INPUT_FIELDS,
    NAVIGATION_TARGETS,
    PATCH_CELL_COUNT,
    PATCH_INPUT_FIELDS,
    SELF_INPUT_FIELDS,
    decode_observation_input,
    encode_observation_input,
)
from evolution_sim.env.runtime.policy import ActionDecision
from evolution_sim.mind.evolution import (
    MIND_V3_POLICY_ID,
    MIND_V3_POLICY_VERSION,
    MIND_V3_REWARD_UPDATE_POLICY,
    adapt_mind_v3_metadata,
    founder_mind_v3_metadata,
    inherit_mind_v3_metadata,
    score_mind_v3_metadata,
)

MIND_V3_ELIGIBILITY_TRACE_POLICY = (
    "policy_valid_requested_action_horizon_eligibility_trace_v3"
)
MIND_V3_ELIGIBILITY_TRACE_LENGTH = 12
MIND_V3_ELIGIBILITY_TRACE_DECAY = 0.84
MIND_V3_REWARD_SIGNAL_POLICY = (
    "balanced_bottleneck_visible_navigation_carrion_readiness_signal_v8"
)
MIND_V3_FOUNDER_TEMPLATE_ASSIGNMENT_POLICY = (
    "contextual_trophic_founder_template_assignment_v1"
)
MIND_V3_REWARD_COMPONENT_SIGNAL_WEIGHTS = {
    "survival_continuation": 0.65,
    "energy_stability": 0.2,
    "hydration_stability": 0.2,
    "health_preservation": 0.35,
    "resource_acquisition": 0.12,
    "reproduction_readiness": 2.0,
    "reproduction_success": 1.75,
    "invalid_action_penalty": 1.0,
    "movement_cost": 0.2,
}
MIND_V3_REPRODUCTION_READINESS_GOALS = {
    "energy_ratio": 0.88,
    "hydration_ratio": 0.72,
    "health_ratio": 0.72,
}
MIND_V3_REPRODUCTION_READINESS_PROGRESS_WEIGHTS = {
    "energy_ratio": 1.8,
    "hydration_ratio": 1.1,
    "health_ratio": 1.1,
}
MIND_V3_REPRODUCTION_READINESS_REGRESSION_WEIGHTS = {
    "energy_ratio": 1.1,
    "hydration_ratio": 0.9,
    "health_ratio": 0.95,
}
MIND_V3_BALANCED_READINESS_PROGRESS_WEIGHT = 1.6
MIND_V3_BALANCED_READINESS_REGRESSION_WEIGHT = 1.35
MIND_V3_LIMITING_READINESS_PROGRESS_WEIGHT = 0.8
MIND_V3_LIMITING_READINESS_REGRESSION_WEIGHT = 0.9
MIND_V3_NO_GAIN_EAT_BASE_PENALTY = 0.24
MIND_V3_NO_GAIN_EAT_STREAK_PENALTY = 0.07
MIND_V3_NO_GAIN_EAT_STREAK_CAP = 4
MIND_V3_HYDRATION_LIMITING_EAT_PENALTY = 0.14
MIND_V3_INEFFECTIVE_EAT_PENALTY = 0.06
MIND_V3_NO_GAIN_DRINK_PENALTY = 0.08
MIND_V3_USEFUL_EAT_SIGNAL_CAP = 0.34
MIND_V3_USEFUL_ANIMAL_RESOURCE_EAT_SIGNAL_CAP = 0.16
MIND_V3_USEFUL_DRINK_SIGNAL_CAP = 0.22
MIND_V3_USEFUL_MOVEMENT_SIGNAL_CAP = 0.1
MIND_V3_USEFUL_NAVIGATION_MOVEMENT_SIGNAL_CAP = 0.08
MIND_V3_NAVIGATION_MOVEMENT_SIGNAL_SCALE = 0.18
MIND_V3_NAVIGATION_MOVEMENT_DIRECTIONS = {
    "move_north": (0.0, -1.0),
    "move_south": (0.0, 1.0),
    "move_east": (1.0, 0.0),
    "move_west": (-1.0, 0.0),
}


class MindV3EvolutionPolicy:
    policy_id = MIND_V3_POLICY_ID
    policy_version = MIND_V3_POLICY_VERSION

    def __init__(
        self,
        *,
        seed: int,
        founder_template_metadata: (
            Mapping[str, object] | Sequence[Mapping[str, object]] | None
        ) = None,
    ) -> None:
        self._rng = Random(seed)
        self._founder_template_pool = _founder_template_pool(
            founder_template_metadata
        )
        self._agent_metadata: dict[int, dict[str, object]] = {}
        self._eligibility_traces: dict[int, list[tuple[str, list[float]]]] = {}
        self._no_gain_eat_streaks: dict[int, int] = {}
        self.controller_update_count = 0

    def register_agent_mind(
        self,
        *,
        agent_id: int,
        metadata: Mapping[str, object],
    ) -> None:
        self._agent_metadata[int(agent_id)] = dict(metadata)

    def agent_mind_metadata(self, *, agent_id: int) -> dict[str, object]:
        return dict(self._agent_metadata[int(agent_id)])

    def controller_population_metadata(self) -> dict[int, dict[str, object]]:
        return {
            int(agent_id): dict(metadata)
            for agent_id, metadata in self._agent_metadata.items()
        }

    def founder_metadata(self, *, agent_id: int) -> dict[str, object]:
        if self._founder_template_pool:
            template = self._select_founder_template(
                agent_id=agent_id,
                trophic_role=None,
                meat_mode=None,
            )
            metadata = inherit_mind_v3_metadata(
                primary_parent_metadata=template,
                secondary_parent_metadata=None,
                child_agent_id=agent_id,
                rng=self._rng,
            )
            _record_founder_template_assignment(
                metadata,
                template=template,
                trophic_role=None,
                meat_mode=None,
            )
        else:
            metadata = founder_mind_v3_metadata(agent_id=agent_id, rng=self._rng)
        self.register_agent_mind(agent_id=agent_id, metadata=metadata)
        return metadata

    def contextual_founder_metadata(
        self,
        *,
        agent_id: int,
        trophic_role: str | None = None,
        meat_mode: str | None = None,
    ) -> dict[str, object]:
        if not self._founder_template_pool:
            return self.founder_metadata(agent_id=agent_id)
        template = self._select_founder_template(
            agent_id=agent_id,
            trophic_role=trophic_role,
            meat_mode=meat_mode,
        )
        metadata = inherit_mind_v3_metadata(
            primary_parent_metadata=template,
            secondary_parent_metadata=None,
            child_agent_id=agent_id,
            rng=self._rng,
        )
        _record_founder_template_assignment(
            metadata,
            template=template,
            trophic_role=trophic_role,
            meat_mode=meat_mode,
        )
        self.register_agent_mind(agent_id=agent_id, metadata=metadata)
        return metadata

    def _select_founder_template(
        self,
        *,
        agent_id: int,
        trophic_role: str | None,
        meat_mode: str | None,
    ) -> dict[str, object]:
        if not self._founder_template_pool:
            raise ValueError("founder template pool is empty")
        preferred_profiles = _preferred_template_profiles(
            trophic_role=trophic_role,
            meat_mode=meat_mode,
        )
        for profile in preferred_profiles:
            matches = [
                template
                for template in self._founder_template_pool
                if str(template.get("specialization_profile", "")) == profile
            ]
            if matches:
                return dict(matches[abs(int(agent_id)) % len(matches)])
        return dict(
            self._founder_template_pool[
                abs(int(agent_id)) % len(self._founder_template_pool)
            ]
        )

    def child_metadata(
        self,
        *,
        child_agent_id: int,
        primary_parent_id: int,
        secondary_parent_id: int | None,
    ) -> dict[str, object]:
        metadata = inherit_mind_v3_metadata(
            primary_parent_metadata=self._agent_metadata.get(primary_parent_id, {}),
            secondary_parent_metadata=(
                self._agent_metadata.get(secondary_parent_id, {})
                if secondary_parent_id is not None
                else None
            ),
            child_agent_id=child_agent_id,
            rng=self._rng,
        )
        self.register_agent_mind(agent_id=child_agent_id, metadata=metadata)
        return metadata

    def decide(
        self,
        observation: dict[str, object],
        action_mask: dict[str, bool],
    ) -> ActionDecision:
        agent_id = _agent_id(observation)
        metadata = self._agent_metadata.get(agent_id)
        if metadata is None:
            self_state = observation.get("self")
            self_payload = self_state if isinstance(self_state, Mapping) else {}
            metadata = self.contextual_founder_metadata(
                agent_id=agent_id,
                trophic_role=_optional_string(self_payload.get("trophic_role")),
                meat_mode=_optional_string(self_payload.get("meat_mode")),
            )
        scores = score_mind_v3_metadata(
            metadata=metadata,
            observation_input=_observation_values(observation),
            action_mask=action_mask,
        )
        requested_action, score = _best_action(scores, action_mask)
        return ActionDecision(
            requested_action=requested_action,
            source=MIND_V3_POLICY_VERSION,
            policy_id=self.policy_id,
            policy_version=self.policy_version,
            diagnostics={
                "runtime_mode": "mind-v3-autonomous-evolution",
                "heuristic_free": True,
                "agent_id": agent_id,
                "score": score,
                "legal_action_count": sum(
                    1 for allowed in action_mask.values() if allowed
                ),
            },
        )

    def observe_transition(self, record: dict[str, object]) -> dict[str, object] | None:
        passive_terminal_feedback = False
        if record.get("action_source") == "passive":
            if not _record_has_terminal_feedback(record):
                return None
            passive_terminal_feedback = True
        policy_id = record.get("policy_id")
        if policy_id is not None and policy_id != self.policy_id:
            return None
        agent_id = _record_agent_id(record)
        if agent_id is None:
            return None
        metadata = self._agent_metadata.get(agent_id)
        if metadata is None:
            return None
        action = _record_action(record)
        if action is None and not passive_terminal_feedback:
            return None
        requested_action = _record_requested_action(record)
        resolved_action = _record_resolved_action(record)
        reward_total = _record_reward_total(record)
        reward_signal_components = _record_reward_signal_components(
            record,
            reward_total=reward_total,
            no_gain_eat_streak=self._no_gain_eat_streaks.get(agent_id, 0),
        )
        reward_signal = float(reward_signal_components["reward_signal"])
        observation_values = _observation_values(record)
        trace_items = self._eligibility_traces.setdefault(agent_id, [])
        if not passive_terminal_feedback:
            if action is None:
                return None
            trace_items.append((action, observation_values))
            if len(trace_items) > MIND_V3_ELIGIBILITY_TRACE_LENGTH:
                del trace_items[0 : len(trace_items) - MIND_V3_ELIGIBILITY_TRACE_LENGTH]
        elif not trace_items:
            return None
        updated = metadata
        credited_actions: list[str] = []
        for age, (credit_action, credit_observation_values) in enumerate(
            reversed(trace_items)
        ):
            updated = adapt_mind_v3_metadata(
                metadata=updated,
                observation_input=credit_observation_values,
                action=credit_action,
                reward_signal=reward_signal
                * (MIND_V3_ELIGIBILITY_TRACE_DECAY ** age),
            )
            credited_actions.append(credit_action)
        self._agent_metadata[agent_id] = updated
        if float(reward_signal_components.get("no_gain_eat", 0.0)) > 0.0:
            self._no_gain_eat_streaks[agent_id] = (
                self._no_gain_eat_streaks.get(agent_id, 0) + 1
            )
        else:
            self._no_gain_eat_streaks[agent_id] = 0
        self.controller_update_count += 1
        return {
            "schema_version": "mind_v3_controller_update_trace_v1",
            "policy": MIND_V3_REWARD_UPDATE_POLICY,
            "credit_assignment": MIND_V3_ELIGIBILITY_TRACE_POLICY,
            "update_index": self.controller_update_count,
            "agent_id": agent_id,
            "action": action if action is not None else (resolved_action or "stay"),
            "requested_action": requested_action or "stay",
            "resolved_action": resolved_action or "stay",
            "credited_actions": credited_actions,
            "reward_total": reward_total,
            "reward_signal": reward_signal,
            "reward_signal_policy": MIND_V3_REWARD_SIGNAL_POLICY,
            "reward_signal_components": reward_signal_components,
            "terminal_feedback": passive_terminal_feedback,
            "trace_appended_action": not passive_terminal_feedback,
            "heuristic_free": True,
        }


def _founder_template_pool(
    founder_template_metadata: (
        Mapping[str, object] | Sequence[Mapping[str, object]] | None
    ),
) -> list[dict[str, object]]:
    if founder_template_metadata is None:
        return []
    if isinstance(founder_template_metadata, Mapping):
        return [dict(founder_template_metadata)]
    pool: list[dict[str, object]] = []
    for template in founder_template_metadata:
        if isinstance(template, Mapping):
            pool.append(dict(template))
    return pool


def _preferred_template_profiles(
    *,
    trophic_role: str | None,
    meat_mode: str | None,
) -> tuple[str, ...]:
    role = trophic_role or "unknown"
    mode = meat_mode or "unknown"
    if mode == "scavenger":
        return (
            "scavenger",
            "predator_scavenger",
            "hydration_seeker",
            "disperser",
            "forager",
            "reproducer",
        )
    if mode == "hunter":
        return (
            "predator_scavenger",
            "scavenger",
            "disperser",
            "hydration_seeker",
            "forager",
            "reproducer",
        )
    if mode == "mixed" or role == "omnivore":
        return (
            "forager",
            "predator_scavenger",
            "scavenger",
            "hydration_seeker",
            "disperser",
            "reproducer",
        )
    if role == "herbivore" or mode == "none":
        return (
            "forager",
            "hydration_seeker",
            "disperser",
            "reproducer",
            "scavenger",
            "predator_scavenger",
        )
    return ()


def _record_founder_template_assignment(
    metadata: dict[str, object],
    *,
    template: Mapping[str, object],
    trophic_role: str | None,
    meat_mode: str | None,
) -> None:
    metadata["founder_template_assignment_policy"] = (
        MIND_V3_FOUNDER_TEMPLATE_ASSIGNMENT_POLICY
    )
    metadata["founder_template_source_profile"] = str(
        template.get("specialization_profile", "unknown")
    )
    metadata["founder_template_context"] = {
        "trophic_role": trophic_role,
        "meat_mode": meat_mode,
    }


def _optional_string(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def _agent_id(observation: Mapping[str, object]) -> int:
    metadata = observation.get("metadata")
    if not isinstance(metadata, Mapping):
        return -1
    return int(metadata.get("agent_id", -1))


def _record_agent_id(record: Mapping[str, object]) -> int | None:
    value = record.get("agent_id")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _record_action(record: Mapping[str, object]) -> str | None:
    requested = _record_requested_action(record)
    if requested is not None and _record_requested_action_valid(record, requested):
        return requested
    return _record_resolved_action(record)


def _record_requested_action(record: Mapping[str, object]) -> str | None:
    action = record.get("requested_action")
    return str(action) if isinstance(action, str) and action else None


def _record_resolved_action(record: Mapping[str, object]) -> str | None:
    action = record.get("resolved_action", record.get("requested_action"))
    return str(action) if isinstance(action, str) and action else None


def _record_requested_action_valid(
    record: Mapping[str, object],
    action: str,
) -> bool:
    explicit_valid = record.get("action_valid")
    if isinstance(explicit_valid, bool):
        return explicit_valid
    action_mask = record.get("action_mask")
    if isinstance(action_mask, Mapping):
        return bool(action_mask.get(action, False))
    return True


def _record_reward_total(record: Mapping[str, object]) -> float:
    reward = record.get("reward")
    if not isinstance(reward, Mapping):
        return 0.0
    total = reward.get("total", 0.0)
    return (
        float(total)
        if isinstance(total, (int, float)) and not isinstance(total, bool)
        else 0.0
    )


def _record_reward_signal(
    record: Mapping[str, object],
    *,
    reward_total: float,
) -> float:
    return float(
        _record_reward_signal_components(
            record,
            reward_total=reward_total,
            no_gain_eat_streak=0,
        )["reward_signal"]
    )


def _record_reward_signal_components(
    record: Mapping[str, object],
    *,
    reward_total: float,
    no_gain_eat_streak: int,
) -> dict[str, float | str]:
    component_signal = _record_component_reward_signal(
        record,
        reward_total=reward_total,
    )
    readiness_deltas = _record_readiness_deltas(record)
    bottleneck = _record_readiness_bottleneck(record)
    readiness_delta_signal = _readiness_delta_signal(
        readiness_deltas,
        bottleneck=bottleneck,
    )
    action_outcome_signal, no_gain_eat = _record_action_outcome_signal(
        record,
        readiness_deltas=readiness_deltas,
        bottleneck=bottleneck,
        no_gain_eat_streak=no_gain_eat_streak,
    )
    terminal_signal = _record_terminal_signal(record)
    raw_signal = (
        0.4 * component_signal
        + readiness_delta_signal
        + action_outcome_signal
        + terminal_signal
    )
    return {
        "policy": MIND_V3_REWARD_SIGNAL_POLICY,
        "component_reward_signal": _round(component_signal),
        "readiness_delta_signal": _round(readiness_delta_signal),
        "action_outcome_signal": _round(action_outcome_signal),
        "terminal_signal": _round(terminal_signal),
        "limiting_readiness_field": str(bottleneck.get("field", "unknown")),
        "limiting_readiness_delta": _round(
            float(bottleneck.get("limiting_delta", 0.0))
        ),
        "balanced_core_readiness_delta": _round(
            float(bottleneck.get("balanced_delta", 0.0))
        ),
        "before_balanced_core_readiness": _round(
            float(bottleneck.get("before_balanced", 0.0))
        ),
        "after_balanced_core_readiness": _round(
            float(bottleneck.get("after_balanced", 0.0))
        ),
        "no_gain_eat": 1.0 if no_gain_eat else 0.0,
        "no_gain_eat_streak": float(no_gain_eat_streak),
        "raw_signal": _round(raw_signal),
        "reward_signal": _clamp_reward_signal(raw_signal),
    }


def _record_component_reward_signal(
    record: Mapping[str, object],
    *,
    reward_total: float,
) -> float:
    reward = record.get("reward")
    if not isinstance(reward, Mapping):
        return _clamp_reward_signal(reward_total)
    components = reward.get("components")
    if not isinstance(components, Mapping):
        return _clamp_reward_signal(reward_total)
    signal = 0.0
    component_seen = False
    for name, weight in MIND_V3_REWARD_COMPONENT_SIGNAL_WEIGHTS.items():
        value = components.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        component_seen = True
        signal += float(weight) * float(value)
    if not component_seen:
        return _clamp_reward_signal(reward_total)
    return _clamp_reward_signal(signal)


def _record_readiness_delta_signal(record: Mapping[str, object]) -> float:
    return _readiness_delta_signal(
        _record_readiness_deltas(record),
        bottleneck=_record_readiness_bottleneck(record),
    )


def _record_readiness_deltas(record: Mapping[str, object]) -> dict[str, float]:
    before = record.get("before")
    after = record.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return {}
    deltas: dict[str, float] = {}
    for field, goal in MIND_V3_REPRODUCTION_READINESS_GOALS.items():
        before_value = _number(before.get(field))
        after_value = _number(after.get(field))
        if before_value is None or after_value is None:
            continue
        before_gap = max(0.0, goal - before_value)
        after_gap = max(0.0, goal - after_value)
        deltas[field] = (before_gap - after_gap) / max(goal, 1e-9)
    return deltas


def _record_readiness_bottleneck(
    record: Mapping[str, object],
) -> dict[str, float | str]:
    before = record.get("before")
    after = record.get("after")
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return {
            "field": "unknown",
            "before_balanced": 0.0,
            "after_balanced": 0.0,
            "balanced_delta": 0.0,
            "limiting_delta": 0.0,
        }
    before_progress: dict[str, float] = {}
    after_progress: dict[str, float] = {}
    for field, goal in MIND_V3_REPRODUCTION_READINESS_GOALS.items():
        before_value = _number(before.get(field))
        after_value = _number(after.get(field))
        if before_value is None or after_value is None:
            continue
        before_progress[field] = _readiness_progress(before_value, goal)
        after_progress[field] = _readiness_progress(after_value, goal)
    if not before_progress:
        return {
            "field": "unknown",
            "before_balanced": 0.0,
            "after_balanced": 0.0,
            "balanced_delta": 0.0,
            "limiting_delta": 0.0,
        }
    limiting_field = min(
        before_progress,
        key=lambda field: (before_progress[field], field),
    )
    before_balanced = min(before_progress.values())
    after_balanced = min(
        after_progress.get(field, before_progress[field])
        for field in before_progress
    )
    limiting_delta = (
        after_progress.get(limiting_field, before_progress[limiting_field])
        - before_progress[limiting_field]
    )
    return {
        "field": limiting_field,
        "before_balanced": before_balanced,
        "after_balanced": after_balanced,
        "balanced_delta": after_balanced - before_balanced,
        "limiting_delta": limiting_delta,
    }


def _readiness_progress(value: float, goal: float) -> float:
    return max(0.0, min(1.0, float(value) / max(float(goal), 1e-9)))


def _readiness_delta_signal(
    readiness_deltas: Mapping[str, float],
    *,
    bottleneck: Mapping[str, float | str],
) -> float:
    signal = 0.0
    limiting_field = str(bottleneck.get("field", "unknown"))
    for field in MIND_V3_REPRODUCTION_READINESS_GOALS:
        delta = float(readiness_deltas.get(field, 0.0))
        if delta >= 0.0:
            positive_weight_scale = 1.0 if field == limiting_field else 0.35
            signal += (
                MIND_V3_REPRODUCTION_READINESS_PROGRESS_WEIGHTS[field]
                * positive_weight_scale
                * delta
            )
        else:
            signal += (
                MIND_V3_REPRODUCTION_READINESS_REGRESSION_WEIGHTS[field]
                * delta
            )
    balanced_delta = float(bottleneck.get("balanced_delta", 0.0))
    if balanced_delta >= 0.0:
        signal += MIND_V3_BALANCED_READINESS_PROGRESS_WEIGHT * balanced_delta
    else:
        signal += MIND_V3_BALANCED_READINESS_REGRESSION_WEIGHT * balanced_delta
    limiting_delta = float(bottleneck.get("limiting_delta", 0.0))
    if limiting_delta >= 0.0:
        signal += MIND_V3_LIMITING_READINESS_PROGRESS_WEIGHT * limiting_delta
    else:
        signal += MIND_V3_LIMITING_READINESS_REGRESSION_WEIGHT * limiting_delta
    return signal


def _record_action_outcome_signal(
    record: Mapping[str, object],
    *,
    readiness_deltas: Mapping[str, float],
    bottleneck: Mapping[str, float | str],
    no_gain_eat_streak: int,
) -> tuple[float, bool]:
    requested_action = _record_requested_action(record)
    resolved_action = _record_resolved_action(record)
    action = requested_action or resolved_action or "stay"
    positive_total = sum(max(0.0, float(value)) for value in readiness_deltas.values())
    energy_progress = max(0.0, float(readiness_deltas.get("energy_ratio", 0.0)))
    hydration_progress = max(
        0.0,
        float(readiness_deltas.get("hydration_ratio", 0.0)),
    )
    health_progress = max(0.0, float(readiness_deltas.get("health_ratio", 0.0)))
    limiting_field = str(bottleneck.get("field", "unknown"))
    limiting_progress = max(0.0, float(bottleneck.get("limiting_delta", 0.0)))
    balanced_progress = max(0.0, float(bottleneck.get("balanced_delta", 0.0)))
    outcome = record.get("outcome")
    outcome_payload = outcome if isinstance(outcome, Mapping) else {}
    feeding = outcome_payload.get("feeding")
    feeding_payload = feeding if isinstance(feeding, Mapping) else {}
    drinking = outcome_payload.get("drinking")
    drinking_payload = drinking if isinstance(drinking, Mapping) else {}
    resource_gain = _number(outcome_payload.get("resource_gain")) or 0.0

    if action == "eat":
        food_source = str(feeding_payload.get("food_source", ""))
        feeding_gain = _number(feeding_payload.get("gained_energy")) or 0.0
        observed_gain = max(resource_gain, feeding_gain)
        ate = bool(feeding_payload.get("ate", False)) or observed_gain > 0.0
        animal_resource_eat = (
            food_source in {"carcass", "fresh_kill"} and observed_gain > 0.0
        )
        useful_progress = limiting_progress + 0.5 * balanced_progress
        if limiting_field == "energy_ratio":
            useful_progress += 0.5 * energy_progress
        elif limiting_field == "hydration_ratio":
            useful_progress += 0.5 * hydration_progress
        elif limiting_field == "health_ratio":
            useful_progress += 0.5 * health_progress
        if ate and useful_progress > 0.0:
            animal_resource_signal = 0.0
            if animal_resource_eat:
                animal_resource_signal = min(
                    MIND_V3_USEFUL_ANIMAL_RESOURCE_EAT_SIGNAL_CAP,
                    0.35 * energy_progress
                    + 0.2 * hydration_progress
                    + 0.2 * health_progress,
                )
            return (
                min(
                    MIND_V3_USEFUL_EAT_SIGNAL_CAP
                    + MIND_V3_USEFUL_ANIMAL_RESOURCE_EAT_SIGNAL_CAP,
                    0.45 * useful_progress + animal_resource_signal,
                ),
                False,
            )
        if (
            ate
            and limiting_field == "hydration_ratio"
            and hydration_progress <= 0.0
            and balanced_progress <= 0.0
        ):
            return (-MIND_V3_HYDRATION_LIMITING_EAT_PENALTY, False)
        if ate and observed_gain <= 0.0 and positive_total <= 0.0:
            penalty = MIND_V3_NO_GAIN_EAT_BASE_PENALTY + (
                MIND_V3_NO_GAIN_EAT_STREAK_PENALTY
                * min(
                    MIND_V3_NO_GAIN_EAT_STREAK_CAP,
                    max(0, int(no_gain_eat_streak)),
                )
            )
            return -penalty, True
        if ate:
            return (-MIND_V3_INEFFECTIVE_EAT_PENALTY, False)
        penalty = MIND_V3_NO_GAIN_EAT_BASE_PENALTY + (
            MIND_V3_NO_GAIN_EAT_STREAK_PENALTY
            * min(
                MIND_V3_NO_GAIN_EAT_STREAK_CAP,
                max(0, int(no_gain_eat_streak)),
            )
        )
        return -penalty, True

    if action == "drink":
        useful_progress = limiting_progress + 0.5 * balanced_progress
        if limiting_field == "hydration_ratio":
            useful_progress += 0.5 * hydration_progress
        if bool(drinking_payload.get("drank", False)) and useful_progress > 0.0:
            return (
                min(MIND_V3_USEFUL_DRINK_SIGNAL_CAP, 0.65 * useful_progress),
                False,
            )
        if bool(drinking_payload.get("drank", False)):
            return (-MIND_V3_NO_GAIN_DRINK_PENALTY, False)
        return (-0.12, False)

    if action.startswith("move_"):
        useful_progress = limiting_progress + 0.5 * balanced_progress
        navigation_signal = _record_navigation_movement_signal(
            record,
            action=action,
            bottleneck=bottleneck,
        )
        if bool(record.get("moved", False)) and useful_progress > 0.0:
            immediate_signal = min(
                MIND_V3_USEFUL_MOVEMENT_SIGNAL_CAP,
                0.25 * max(useful_progress, 0.25 * positive_total),
            )
            return (
                min(
                    MIND_V3_USEFUL_MOVEMENT_SIGNAL_CAP,
                    immediate_signal + navigation_signal,
                ),
                False,
            )
        if bool(record.get("moved", False)) and navigation_signal > 0.0:
            return (navigation_signal, False)
        return (0.0, False)

    return (0.0, False)


def _record_navigation_movement_signal(
    record: Mapping[str, object],
    *,
    action: str,
    bottleneck: Mapping[str, float | str],
) -> float:
    action_direction = MIND_V3_NAVIGATION_MOVEMENT_DIRECTIONS.get(action)
    if action_direction is None:
        return 0.0
    if not isinstance(record.get("observation_input"), Mapping):
        return 0.0
    values = _observation_values(record)
    if not values:
        return 0.0
    energy = _vector_self_feature(values, "energy_ratio", default=1.0)
    hydration = _vector_self_feature(values, "hydration_ratio", default=1.0)
    matched_diet = _vector_self_feature(values, "matched_diet_ratio")
    trophic_role = _vector_self_feature(values, "trophic_role_code")
    meat_mode = _vector_self_feature(values, "meat_mode_code")
    vegetation = _vector_self_feature(values, "tile_vegetation")
    local_plant = max(_vector_center_patch_feature(values, "food"), vegetation)
    local_animal = max(
        _vector_center_patch_feature(values, "fresh_kill_energy"),
        _vector_center_patch_feature(values, "carcass_energy"),
    )
    hunger = max(0.0, 0.72 - energy) / 0.72
    thirst = max(0.0, 0.72 - hydration) / 0.72
    if hunger <= 0.0 and thirst <= 0.0:
        return 0.0
    plant_preference = max(0.0, min(1.0, 1.0 - meat_mode + local_plant * 0.5))
    meat_preference = max(
        0.0,
        min(
            1.0,
            meat_mode * 1.8
            + trophic_role * 0.25
            + local_animal * 0.5
            + max(0.0, 0.65 - matched_diet) * 0.3,
        ),
    )
    prey_preference = max(0.0, min(1.0, meat_mode * 1.3 + trophic_role * 0.35))
    limiting_field = str(bottleneck.get("field", "unknown"))
    water_weight = thirst * (1.35 if limiting_field == "hydration_ratio" else 0.7)
    energy_weight = 1.25 if limiting_field == "energy_ratio" else 0.85
    target_signals = [
        water_weight
        * _navigation_movement_alignment(values, "water", action_direction),
        hunger
        * plant_preference
        * energy_weight
        * _navigation_movement_alignment(values, "plant", action_direction),
        hunger
        * meat_preference
        * energy_weight
        * _navigation_movement_alignment(values, "carrion", action_direction),
        hunger
        * prey_preference
        * energy_weight
        * 0.75
        * _navigation_movement_alignment(values, "prey", action_direction),
    ]
    best_signal = max(target_signals)
    if best_signal <= 0.0:
        return 0.0
    return min(
        MIND_V3_USEFUL_NAVIGATION_MOVEMENT_SIGNAL_CAP,
        MIND_V3_NAVIGATION_MOVEMENT_SIGNAL_SCALE * best_signal,
    )


def _navigation_movement_alignment(
    values: list[float],
    target: str,
    action_direction: tuple[float, float],
) -> float:
    target_dx, target_dy = _vector_navigation_vector(values, target)
    action_dx, action_dy = action_direction
    return max(0.0, action_dx * target_dx + action_dy * target_dy)


def _record_terminal_signal(record: Mapping[str, object]) -> float:
    outcome = record.get("outcome")
    if not isinstance(outcome, Mapping):
        return 0.0
    signal = 0.0
    if outcome.get("reproduction_ready_after") is True:
        signal += 0.45
    if outcome.get("reproduced") is True:
        signal += 1.0
    if outcome.get("died") is True:
        signal -= 0.85
    return signal


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _clamp_reward_signal(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _round(value: float) -> float:
    return round(float(value), 4)


def _record_has_terminal_feedback(record: Mapping[str, object]) -> bool:
    outcome = record.get("outcome")
    if not isinstance(outcome, Mapping):
        return False
    if outcome.get("died") is True:
        return True
    passive = outcome.get("passive")
    if not isinstance(passive, Mapping):
        return False
    return (
        passive.get("killed") is True
        or passive.get("died_before_action") is True
        or passive.get("died_after_action") is True
    )


def _observation_values(observation: Mapping[str, object]) -> list[float]:
    payload = observation.get("observation_input")
    if isinstance(payload, Mapping):
        values = payload.get("values")
        if isinstance(values, list):
            return [
                float(value)
                for value in values
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            ]
        return decode_observation_input(dict(payload))
    return decode_observation_input(encode_observation_input(dict(observation)))


def _vector_self_feature(
    values: list[float],
    field: str,
    *,
    default: float = 0.0,
) -> float:
    try:
        index = SELF_INPUT_FIELDS.index(field)
    except ValueError:
        return default
    if index >= len(values):
        return default
    return max(0.0, min(1.0, float(values[index])))


def _vector_center_patch_feature(
    values: list[float],
    field: str,
    *,
    default: float = 0.0,
) -> float:
    try:
        field_index = PATCH_INPUT_FIELDS.index(field)
    except ValueError:
        return default
    center_cell_index = PATCH_CELL_COUNT // 2
    index = (
        len(SELF_INPUT_FIELDS)
        + center_cell_index * len(PATCH_INPUT_FIELDS)
        + field_index
    )
    if index >= len(values):
        return default
    return max(0.0, min(1.0, float(values[index])))


def _vector_navigation_feature(
    values: list[float],
    target: str,
    field: str,
    *,
    default: float = 0.0,
) -> float:
    try:
        target_index = NAVIGATION_TARGETS.index(target)
        field_index = NAVIGATION_INPUT_FIELDS.index(field)
    except ValueError:
        return default
    navigation_start = len(SELF_INPUT_FIELDS) + (
        PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
    )
    index = (
        navigation_start
        + target_index * len(NAVIGATION_INPUT_FIELDS)
        + field_index
    )
    if index >= len(values):
        return default
    return max(-1.0, min(1.0, float(values[index])))


def _vector_navigation_vector(
    values: list[float],
    target: str,
) -> tuple[float, float]:
    strength = max(0.0, _vector_navigation_feature(values, target, "strength"))
    distance = max(
        0.0,
        min(1.0, _vector_navigation_feature(values, target, "distance")),
    )
    falloff = max(0.0, 1.0 - distance * 0.35)
    scale = strength * falloff
    return (
        _vector_navigation_feature(values, target, "dx") * scale,
        _vector_navigation_feature(values, target, "dy") * scale,
    )


def _best_action(
    scores: Mapping[str, float],
    action_mask: Mapping[str, bool],
) -> tuple[str, float]:
    best_action = "stay"
    best_score = float("-inf")
    for action in sorted(action_mask):
        if not bool(action_mask[action]):
            continue
        score = float(scores.get(action, 0.0))
        if score > best_score:
            best_action = action
            best_score = score
    if best_score == float("-inf"):
        return "stay", 0.0
    return best_action, best_score
