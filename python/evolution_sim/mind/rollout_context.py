from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evolution_sim.env.runtime.action_contract import ACTION_NAMES, MOVEMENT_ACTIONS

MIND_V3_ROLLOUT_CONTEXT_SCHEMA_VERSION = "mind_v3_rollout_context_v1"
MIND_V3_ROLLOUT_CONTEXT_UPDATE_TRACE_SCHEMA_VERSION = (
    "mind_v3_rollout_context_update_trace_v1"
)
MIND_V3_ROLLOUT_CONTEXT_FEATURE_POLICY = (
    "previous_trajectory_rows_policy_owned_context_v1"
)
ANIMAL_RESOURCE_FOOD_SOURCES = frozenset(("carcass", "fresh_kill"))


@dataclass(frozen=True, slots=True)
class RolloutContextConfig:
    recent_window: int = 3
    recovery_phase_ticks: int = 12
    ticks_since_cap: int = 16
    no_gain_eat_streak_cap: int = 5

    def __post_init__(self) -> None:
        if self.recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if self.recovery_phase_ticks <= 0:
            raise ValueError("recovery_phase_ticks must be positive")
        if self.ticks_since_cap <= 0:
            raise ValueError("ticks_since_cap must be positive")
        if self.no_gain_eat_streak_cap <= 0:
            raise ValueError("no_gain_eat_streak_cap must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": MIND_V3_ROLLOUT_CONTEXT_SCHEMA_VERSION,
            "policy": MIND_V3_ROLLOUT_CONTEXT_FEATURE_POLICY,
            "recent_window": self.recent_window,
            "recovery_phase_ticks": self.recovery_phase_ticks,
            "ticks_since_cap": self.ticks_since_cap,
            "no_gain_eat_streak_cap": self.no_gain_eat_streak_cap,
            "vector_size": rollout_context_vector_size(),
        }


class RolloutContextState:
    def __init__(self, config: RolloutContextConfig | None = None):
        self.config = config or RolloutContextConfig()
        self._recent_requested: deque[str] = deque(maxlen=self.config.recent_window)
        self._recent_resolved: deque[str] = deque(maxlen=self.config.recent_window)
        self._recent_moved: deque[int] = deque(maxlen=self.config.recent_window)
        self._recent_drank: deque[int] = deque(maxlen=self.config.recent_window)
        self._recent_ate: deque[int] = deque(maxlen=self.config.recent_window)
        self._recent_resource_gain: deque[float] = deque(
            maxlen=self.config.recent_window
        )
        self._recent_energy_delta: deque[float] = deque(maxlen=self.config.recent_window)
        self._recent_hydration_delta: deque[float] = deque(
            maxlen=self.config.recent_window
        )
        self._recent_health_delta: deque[float] = deque(maxlen=self.config.recent_window)
        self.no_gain_eat_streak = 0
        self.ticks_since_drink: int | None = None
        self.ticks_since_animal_resource_gain: int | None = None
        self.recovery_phase_remaining = 0
        self.previous_tick: int | None = None

    def snapshot(self) -> dict[str, object]:
        return {
            "schema_version": MIND_V3_ROLLOUT_CONTEXT_SCHEMA_VERSION,
            "recent_requested_actions": list(self._recent_requested),
            "recent_resolved_actions": list(self._recent_resolved),
            "recent_moved_flags": list(self._recent_moved),
            "recent_drank_flags": list(self._recent_drank),
            "recent_ate_flags": list(self._recent_ate),
            "recent_resource_gain": [_round(value) for value in self._recent_resource_gain],
            "recent_energy_delta": [_round(value) for value in self._recent_energy_delta],
            "recent_hydration_delta": [
                _round(value) for value in self._recent_hydration_delta
            ],
            "recent_health_delta": [_round(value) for value in self._recent_health_delta],
            "no_gain_eat_streak": self.no_gain_eat_streak,
            "ticks_since_drink": self.ticks_since_drink,
            "ticks_since_animal_resource_gain": self.ticks_since_animal_resource_gain,
            "post_carrion_contact": self.post_carrion_contact,
            "recovery_phase_remaining": self.recovery_phase_remaining,
        }

    @property
    def post_carrion_contact(self) -> bool:
        return (
            self.ticks_since_animal_resource_gain is not None
            and self.ticks_since_animal_resource_gain <= self.config.recovery_phase_ticks
        )

    def context_key(self) -> str:
        snapshot = self.snapshot()
        requested = _recent_family_token(snapshot["recent_requested_actions"])
        resolved = _recent_family_token(snapshot["recent_resolved_actions"])
        flags = (
            f"m{_sum_token(snapshot['recent_moved_flags'])}"
            f"d{_sum_token(snapshot['recent_drank_flags'])}"
            f"a{_sum_token(snapshot['recent_ate_flags'])}"
        )
        deltas = (
            f"e{_delta_bucket(_sum_number_list(snapshot['recent_energy_delta']))}"
            f"h{_delta_bucket(_sum_number_list(snapshot['recent_hydration_delta']))}"
            f"hp{_delta_bucket(_sum_number_list(snapshot['recent_health_delta']))}"
        )
        return (
            f"{MIND_V3_ROLLOUT_CONTEXT_SCHEMA_VERSION}"
            f"|rq={requested}|rs={resolved}|{flags}|g={_gain_bucket(snapshot)}"
            f"|{deltas}|nge={_streak_bucket(self.no_gain_eat_streak)}"
            f"|td={_ticks_bucket(self.ticks_since_drink, self.config.ticks_since_cap)}"
            "|ta="
            f"{_ticks_bucket(self.ticks_since_animal_resource_gain, self.config.ticks_since_cap)}"
            f"|pc={int(self.post_carrion_contact)}"
            f"|rr={_remaining_bucket(self.recovery_phase_remaining, self.config.recovery_phase_ticks)}"
        )

    def coarse_context_key(self) -> str:
        return (
            f"{MIND_V3_ROLLOUT_CONTEXT_SCHEMA_VERSION}"
            f"|phase={_phase_token(self)}"
            f"|last={_last_family(self._recent_resolved)}"
            f"|drank={_sum_token(self._recent_drank)}"
            f"|ate={_sum_token(self._recent_ate)}"
            f"|moved={_sum_token(self._recent_moved)}"
            f"|nge={_streak_bucket(self.no_gain_eat_streak)}"
        )

    def values(self) -> list[float]:
        window = float(self.config.recent_window)
        values: list[float] = []
        requested_counts = _action_counts(self._recent_requested)
        resolved_counts = _action_counts(self._recent_resolved)
        values.extend(requested_counts[action] / window for action in ACTION_NAMES)
        values.extend(resolved_counts[action] / window for action in ACTION_NAMES)
        values.append(sum(self._recent_moved) / window)
        values.append(sum(self._recent_drank) / window)
        values.append(sum(self._recent_ate) / window)
        values.append(_clip_unit(sum(self._recent_resource_gain)))
        values.append(_clip_signed_unit(sum(self._recent_energy_delta)))
        values.append(_clip_signed_unit(sum(self._recent_hydration_delta)))
        values.append(_clip_signed_unit(sum(self._recent_health_delta)))
        values.append(
            min(
                self.config.no_gain_eat_streak_cap,
                self.no_gain_eat_streak,
            )
            / float(self.config.no_gain_eat_streak_cap)
        )
        values.extend(_ticks_values(self.ticks_since_drink, self.config.ticks_since_cap))
        values.extend(
            _ticks_values(
                self.ticks_since_animal_resource_gain,
                self.config.ticks_since_cap,
            )
        )
        values.append(1.0 if self.post_carrion_contact else 0.0)
        values.append(
            min(self.config.recovery_phase_ticks, self.recovery_phase_remaining)
            / float(self.config.recovery_phase_ticks)
        )
        return [_round(value) for value in values]

    def update_from_record(self, record: Mapping[str, object]) -> dict[str, object]:
        previous = self.snapshot()
        event = rollout_context_event_from_record(record)
        tick = event["tick"]
        delta_ticks = _delta_ticks(self.previous_tick, tick)
        requested_action = str(event["requested_action"])
        resolved_action = str(event["resolved_action"])
        resource_gain = float(event["resource_gain"])
        drank = bool(event["drank"])
        animal_resource_gain = bool(event["animal_resource_gain"])

        self._recent_requested.append(requested_action)
        self._recent_resolved.append(resolved_action)
        self._recent_moved.append(1 if bool(event["moved"]) else 0)
        self._recent_drank.append(1 if drank else 0)
        self._recent_ate.append(1 if bool(event["ate"]) else 0)
        self._recent_resource_gain.append(resource_gain)
        self._recent_energy_delta.append(float(event["energy_delta"]))
        self._recent_hydration_delta.append(float(event["hydration_delta"]))
        self._recent_health_delta.append(float(event["health_delta"]))

        if requested_action == "eat" or resolved_action == "eat":
            self.no_gain_eat_streak = (
                0
                if resource_gain > 0.0
                else min(
                    self.config.no_gain_eat_streak_cap,
                    self.no_gain_eat_streak + 1,
                )
            )
        elif resource_gain > 0.0:
            self.no_gain_eat_streak = 0

        if drank:
            self.ticks_since_drink = 0
        elif self.ticks_since_drink is not None:
            self.ticks_since_drink = min(
                self.config.ticks_since_cap,
                self.ticks_since_drink + delta_ticks,
            )

        if animal_resource_gain:
            self.ticks_since_animal_resource_gain = 0
            self.recovery_phase_remaining = self.config.recovery_phase_ticks
        else:
            if self.ticks_since_animal_resource_gain is not None:
                self.ticks_since_animal_resource_gain = min(
                    self.config.ticks_since_cap,
                    self.ticks_since_animal_resource_gain + delta_ticks,
                )
            self.recovery_phase_remaining = max(
                0,
                self.recovery_phase_remaining - delta_ticks,
            )

        self.previous_tick = tick
        return {
            "schema_version": MIND_V3_ROLLOUT_CONTEXT_UPDATE_TRACE_SCHEMA_VERSION,
            "policy": MIND_V3_ROLLOUT_CONTEXT_FEATURE_POLICY,
            "agent_id": event["agent_id"],
            "tick": tick,
            "requested_action": requested_action,
            "resolved_action": resolved_action,
            "previous_context": previous,
            "updated_context": self.snapshot(),
        }


def rollout_context_feature_contract(
    config: RolloutContextConfig | None = None,
) -> dict[str, object]:
    resolved = config or RolloutContextConfig()
    return {
        **resolved.to_dict(),
        "row_scope": "per_agent_previous_rows_before_current_decision",
        "allowed_trajectory_fields": [
            "tick",
            "agent_id",
            "requested_action",
            "resolved_action",
            "moved",
            "before.energy_ratio",
            "before.hydration_ratio",
            "before.health_ratio",
            "after.energy_ratio",
            "after.hydration_ratio",
            "after.health_ratio",
            "outcome.feeding.ate",
            "outcome.feeding.food_source",
            "outcome.drinking.drank",
            "outcome.resource_gain",
        ],
        "excluded_runtime_inputs": [
            "fixture identity",
            "private world state",
            "future trajectory rows",
            "heuristic action recommendation",
        ],
        "update_order": (
            "snapshot is read before the current row, then updated after the "
            "current row is finalized"
        ),
        "numeric_vector_fields": rollout_context_vector_fields(),
    }


def rollout_context_vector_fields() -> list[str]:
    fields = [f"recent_requested_count:{action}" for action in ACTION_NAMES]
    fields.extend(f"recent_resolved_count:{action}" for action in ACTION_NAMES)
    fields.extend(
        [
            "recent_moved_rate",
            "recent_drank_rate",
            "recent_ate_rate",
            "recent_resource_gain_sum",
            "recent_energy_delta_sum",
            "recent_hydration_delta_sum",
            "recent_health_delta_sum",
            "no_gain_eat_streak",
            "ticks_since_drink_seen",
            "ticks_since_drink_norm",
            "ticks_since_animal_resource_gain_seen",
            "ticks_since_animal_resource_gain_norm",
            "post_carrion_contact",
            "recovery_phase_remaining_norm",
        ]
    )
    return fields


def rollout_context_vector_size() -> int:
    return len(rollout_context_vector_fields())


def rollout_context_event_from_record(
    record: Mapping[str, object],
) -> dict[str, object]:
    outcome = _mapping(record.get("outcome"))
    before = _mapping(record.get("before"))
    after = _mapping(record.get("after"))
    feeding = _mapping(outcome.get("feeding"))
    drinking = _mapping(outcome.get("drinking"))
    tick = _int(record.get("tick"), default=0)
    requested_action = _action(record.get("requested_action"))
    resolved_action = _action(record.get("resolved_action"))
    resource_gain = _number(outcome.get("resource_gain"), default=0.0)
    food_source = feeding.get("food_source")
    return {
        "tick": tick,
        "agent_id": _int(record.get("agent_id"), default=-1),
        "requested_action": requested_action,
        "resolved_action": resolved_action,
        "moved": _moved(record, outcome),
        "drank": bool(drinking.get("drank", False)),
        "ate": bool(feeding.get("ate", False)),
        "resource_gain": _round(max(0.0, resource_gain)),
        "animal_resource_gain": (
            resource_gain > 0.0
            and isinstance(food_source, str)
            and food_source in ANIMAL_RESOURCE_FOOD_SOURCES
        ),
        "energy_delta": _ratio_delta(before, after, "energy_ratio"),
        "hydration_delta": _ratio_delta(before, after, "hydration_ratio"),
        "health_delta": _ratio_delta(before, after, "health_ratio"),
    }


def _action(value: object) -> str:
    return str(value) if value in ACTION_NAMES else "stay"


def _moved(record: Mapping[str, object], outcome: Mapping[str, object]) -> bool:
    moved = record.get("moved")
    if isinstance(moved, bool):
        return moved
    movement = _mapping(outcome.get("movement"))
    return bool(movement.get("moved", False))


def _ratio_delta(
    before: Mapping[str, object],
    after: Mapping[str, object],
    field: str,
) -> float:
    return _round(_number(after.get(field), default=0.0) - _number(before.get(field), default=0.0))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _number(value: object, *, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    parsed = float(value)
    return parsed if math.isfinite(parsed) else default


def _int(value: object, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return int(value)


def _round(value: float) -> float:
    return round(float(value), 6)


def _delta_ticks(previous_tick: int | None, tick: int) -> int:
    if previous_tick is None:
        return 1
    return max(1, tick - previous_tick)


def _action_counts(actions: Sequence[str]) -> dict[str, int]:
    counts = {action: 0 for action in ACTION_NAMES}
    for action in actions:
        if action in counts:
            counts[action] += 1
    return counts


def _recent_family_token(actions: object) -> str:
    if not isinstance(actions, Sequence):
        return "none"
    families = [_action_family(str(action)) for action in actions if str(action)]
    return ",".join(families[-3:]) if families else "none"


def _last_family(actions: Sequence[str]) -> str:
    if not actions:
        return "none"
    return _action_family(str(actions[-1]))


def _action_family(action: str) -> str:
    if action in MOVEMENT_ACTIONS or action.startswith("move_"):
        return "move"
    if action.startswith("attack_"):
        return "attack"
    if action.startswith("signal_"):
        return "signal"
    if action == "mate":
        return "mate"
    if action in ("eat", "drink", "stay"):
        return action
    return "other"


def _sum_token(values: object) -> int:
    if not isinstance(values, Sequence):
        return 0
    return min(9, sum(1 for value in values if bool(value)))


def _sum_number_list(values: object) -> float:
    if not isinstance(values, Sequence):
        return 0.0
    total = 0.0
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        total += float(value)
    return total


def _gain_bucket(snapshot: Mapping[str, object]) -> str:
    gain = _sum_number_list(snapshot.get("recent_resource_gain"))
    if gain <= 0.0:
        return "0"
    if gain < 0.1:
        return "lo"
    if gain < 0.35:
        return "mid"
    return "hi"


def _delta_bucket(value: float) -> str:
    if value <= -0.15:
        return "neg2"
    if value < -0.01:
        return "neg1"
    if value <= 0.01:
        return "flat"
    if value < 0.15:
        return "pos1"
    return "pos2"


def _streak_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 3:
        return "2_3"
    return "4p"


def _ticks_bucket(value: int | None, cap: int) -> str:
    if value is None:
        return "none"
    clipped = min(cap, max(0, int(value)))
    if clipped == 0:
        return "0"
    if clipped <= 2:
        return "1_2"
    if clipped <= 5:
        return "3_5"
    if clipped < cap:
        return "6p"
    return "cap"


def _remaining_bucket(value: int, cap: int) -> str:
    if value <= 0:
        return "0"
    ratio = value / float(cap)
    if ratio >= 0.67:
        return "hi"
    if ratio >= 0.34:
        return "mid"
    return "lo"


def _phase_token(state: RolloutContextState) -> str:
    if state.recovery_phase_remaining > 0:
        return "recovery"
    if state.post_carrion_contact:
        return "post_carrion"
    return "neutral"


def _clip_unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _clip_signed_unit(value: float) -> float:
    return min(1.0, max(-1.0, float(value)))


def _ticks_values(value: int | None, cap: int) -> list[float]:
    if value is None:
        return [0.0, 1.0]
    return [1.0, min(cap, max(0, int(value))) / float(cap)]
