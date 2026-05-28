from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from evolution_sim.env.runtime.action_contract import ACTION_NAMES, MOVEMENT_ACTIONS
from evolution_sim.env.runtime.observations import (
    NAVIGATION_INPUT_FIELDS,
    NAVIGATION_TARGETS,
    PATCH_CELL_COUNT,
    PATCH_INPUT_FIELDS,
    SELF_INPUT_FIELDS,
    decode_observation_input,
)

MIND_V3_RECOVERY_CONTEXT_SCHEMA_VERSION = "mind_v3_recovery_context_v1"
MIND_V3_RECOVERY_CONTEXT_UPDATE_TRACE_SCHEMA_VERSION = (
    "mind_v3_recovery_context_update_trace_v1"
)
MIND_V3_RECOVERY_CONTEXT_FEATURE_POLICY = (
    "previous_public_post_animal_resource_recovery_context_v1"
)
MIND_V3_RECOVERY_CONTEXT_TICKS_CAP = 16
MIND_V3_RECOVERY_CONTEXT_NO_GAIN_EAT_STREAK_CAP = 5


class RecoveryContextState:
    def __init__(self) -> None:
        self.previous_public_observation_summary: dict[str, float] | None = None

    def snapshot(self) -> dict[str, object]:
        previous_available = self.previous_public_observation_summary is not None
        return {
            "schema_version": MIND_V3_RECOVERY_CONTEXT_SCHEMA_VERSION,
            "previous_observation_input_available": previous_available,
            "previous_public_observation_summary_available": previous_available,
            "previous_public_observation_summary_fields": [
                "navigation.water.distance"
            ],
        }

    def values(
        self,
        *,
        rollout_context_snapshot: Mapping[str, object],
        current_observation_values: Sequence[object] | None = None,
        current_observation_input: Mapping[str, object] | None = None,
        action_mask: Mapping[str, bool],
    ) -> list[float]:
        return recovery_context_values(
            rollout_context_snapshot=rollout_context_snapshot,
            current_observation_values=current_observation_values,
            current_observation_input=current_observation_input,
            previous_public_observation_summary=(
                self.previous_public_observation_summary
            ),
            action_mask=action_mask,
        )

    def update_from_record(
        self,
        record: Mapping[str, object],
        *,
        observation_values: Sequence[object] | None = None,
    ) -> dict[str, object]:
        previous = self.snapshot()
        values = (
            _coerced_observation_values(observation_values)
            if observation_values is not None
            else recovery_context_decoded_observation_values(
                record.get("observation_input")
            )
        )
        self.previous_public_observation_summary = (
            _public_observation_summary(values) if values else None
        )
        return {
            "schema_version": MIND_V3_RECOVERY_CONTEXT_UPDATE_TRACE_SCHEMA_VERSION,
            "policy": MIND_V3_RECOVERY_CONTEXT_FEATURE_POLICY,
            "agent_id": _int(record.get("agent_id"), default=-1),
            "tick": _int(record.get("tick"), default=0),
            "previous_context": previous,
            "updated_context": self.snapshot(),
        }


def recovery_context_feature_contract() -> dict[str, object]:
    return {
        "schema_version": MIND_V3_RECOVERY_CONTEXT_SCHEMA_VERSION,
        "policy": MIND_V3_RECOVERY_CONTEXT_FEATURE_POLICY,
        "vector_size": recovery_context_vector_size(),
        "row_scope": "per_agent_previous_rows_before_current_decision",
        "current_public_inputs": [
            "observation_input",
            "action_mask",
        ],
        "previous_public_inputs": [
            "same_agent_finalized_trajectory_rows",
            "previous_record.observation_input",
            "previous_record.requested_action",
            "previous_record.resolved_action",
            "previous_record.moved",
            "previous_record.outcome.resource_gain",
            "previous_record.outcome.feeding",
            "previous_record.outcome.feeding.food_source",
            "previous_record.outcome.drinking",
        ],
        "previous_finalized_public_outcome_dependencies": [
            "outcome.resource_gain",
            "outcome.feeding.food_source",
            "outcome.drinking.drank",
            "resolved_action",
            "moved",
        ],
        "derived_rollout_snapshot_dependencies": [
            "post_carrion_contact",
            "ticks_since_drink",
            "no_gain_eat_streak",
            "recent_resolved_actions",
            "recent_moved_flags",
        ],
        "outcome_timing_contract": (
            "previous_finalized_public_rows_only_not_current_or_future_outcomes"
        ),
        "decoded_observation_source": "mind_observation_v3_encoded_input",
        "missing_prior_observation_policy": "zero_features",
        "malformed_context_vector_policy": "zero_features",
        "excluded_runtime_inputs": [
            "private_world_state",
            "future_rows_or_outcomes",
            "scenario_identity",
            "seed_identity",
            "source_path_identity",
            "branch_identity",
            "heuristic_recommendation",
            "logged_action_fallback",
            "controller_private_diagnostics",
        ],
        "numeric_vector_fields": recovery_context_vector_fields(),
    }


def recovery_context_vector_fields() -> list[str]:
    return [
        "post_carrion_contact",
        "post_carrion_contact_x_hydration_debt",
        "post_carrion_contact_x_energy_debt",
        "post_carrion_contact_x_water_distance",
        "post_carrion_contact_x_water_strength",
        "post_carrion_contact_x_water_approach_progress",
        "post_carrion_contact_x_ticks_since_drink_seen",
        "post_carrion_contact_x_ticks_since_drink_norm",
        "post_carrion_contact_x_drink_missing",
        "post_carrion_contact_x_no_gain_eat_streak",
        "recent_resolved_stay_or_no_progress_movement_rate",
        "post_carrion_contact_x_drink_available",
        "post_carrion_contact_x_eat_available",
        "post_carrion_contact_x_movement_available",
    ]


def recovery_context_vector_size() -> int:
    return len(recovery_context_vector_fields())


def recovery_context_values(
    *,
    rollout_context_snapshot: Mapping[str, object],
    action_mask: Mapping[str, bool],
    current_observation_values: Sequence[object] | None = None,
    current_observation_input: Mapping[str, object] | None = None,
    previous_public_observation_summary: Mapping[str, object] | None = None,
    previous_observation_values: Sequence[object] | None = None,
    previous_observation_input: Mapping[str, object] | None = None,
) -> list[float]:
    fields = recovery_context_vector_fields()
    zero = [0.0] * len(fields)
    current_values = (
        _coerced_observation_values(current_observation_values)
        if current_observation_values is not None
        else recovery_context_decoded_observation_values(current_observation_input)
    )
    if not current_values:
        return zero
    previous_summary = _coerced_public_observation_summary(
        previous_public_observation_summary
    )
    if previous_summary is None and previous_observation_values is not None:
        previous_values = _coerced_observation_values(previous_observation_values)
        previous_summary = (
            _public_observation_summary(previous_values) if previous_values else None
        )
    if previous_summary is None and isinstance(previous_observation_input, Mapping):
        previous_values = recovery_context_decoded_observation_values(
            previous_observation_input
        )
        previous_summary = (
            _public_observation_summary(previous_values) if previous_values else None
        )
    post_carrion = (
        1.0 if bool(rollout_context_snapshot.get("post_carrion_contact")) else 0.0
    )
    energy = _self_feature(current_values, "energy_ratio", default=1.0)
    hydration = _self_feature(current_values, "hydration_ratio", default=1.0)
    water_distance = _navigation_feature(current_values, "water", "distance")
    water_strength = _navigation_feature(current_values, "water", "strength")
    previous_water_distance = (
        previous_summary["water_distance"]
        if previous_summary is not None
        else water_distance
    )
    ticks_since_drink = _optional_nonnegative_int(
        rollout_context_snapshot.get("ticks_since_drink")
    )
    drink_seen = 1.0 if ticks_since_drink is not None else 0.0
    drink_missing = 1.0 if ticks_since_drink is None else 0.0
    ticks_norm = (
        0.0
        if ticks_since_drink is None
        else min(MIND_V3_RECOVERY_CONTEXT_TICKS_CAP, ticks_since_drink)
        / float(MIND_V3_RECOVERY_CONTEXT_TICKS_CAP)
    )
    no_gain_eat = min(
        MIND_V3_RECOVERY_CONTEXT_NO_GAIN_EAT_STREAK_CAP,
        max(0, _int(rollout_context_snapshot.get("no_gain_eat_streak"), default=0)),
    ) / float(MIND_V3_RECOVERY_CONTEXT_NO_GAIN_EAT_STREAK_CAP)
    values = [
        post_carrion,
        post_carrion * max(0.0, 1.0 - hydration),
        post_carrion * max(0.0, 1.0 - energy),
        post_carrion * water_distance,
        post_carrion * water_strength,
        post_carrion * _clip_signed(previous_water_distance - water_distance),
        post_carrion * drink_seen,
        post_carrion * ticks_norm,
        post_carrion * drink_missing,
        post_carrion * no_gain_eat,
        _recent_stay_or_no_progress_rate(rollout_context_snapshot),
        post_carrion * (1.0 if bool(action_mask.get("drink", False)) else 0.0),
        post_carrion * (1.0 if bool(action_mask.get("eat", False)) else 0.0),
        post_carrion * (
            1.0
            if any(bool(action_mask.get(action, False)) for action in MOVEMENT_ACTIONS)
            else 0.0
        ),
    ]
    if len(values) != len(fields):
        return zero
    return [_round(_clip_signed(value)) for value in values]


def recovery_context_decoded_observation_values(payload: object) -> list[float]:
    if not isinstance(payload, Mapping):
        return []
    raw_values = payload.get("values")
    if isinstance(raw_values, list):
        return _coerced_observation_values(raw_values)
    try:
        return decode_observation_input(dict(payload))
    except (ValueError, TypeError):
        return []


def _coerced_observation_values(values: object) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return []
    coerced = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return []
        coerced.append(_finite_number(value))
    return coerced


def _public_observation_summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "water_distance": _navigation_feature(values, "water", "distance"),
    }


def _coerced_public_observation_summary(
    summary: Mapping[str, object] | None,
) -> dict[str, float] | None:
    if not isinstance(summary, Mapping):
        return None
    value = summary.get("water_distance")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        return None
    return {"water_distance": _clip_unit(parsed)}


def _self_feature(
    values: Sequence[float],
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
    return _clip_unit(_finite_number(values[index]))


def _navigation_feature(
    values: Sequence[float],
    target: str,
    field: str,
) -> float:
    try:
        target_index = NAVIGATION_TARGETS.index(target)
        field_index = NAVIGATION_INPUT_FIELDS.index(field)
    except ValueError:
        return 0.0
    offset = (
        len(SELF_INPUT_FIELDS)
        + PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
        + target_index * len(NAVIGATION_INPUT_FIELDS)
        + field_index
    )
    if offset >= len(values):
        return 0.0
    if field == "distance" or field == "strength":
        return _clip_unit(_finite_number(values[offset]))
    return _clip_signed(_finite_number(values[offset]))


def _recent_stay_or_no_progress_rate(
    snapshot: Mapping[str, object],
) -> float:
    resolved = snapshot.get("recent_resolved_actions")
    moved = snapshot.get("recent_moved_flags")
    if not isinstance(resolved, Sequence) or isinstance(resolved, (str, bytes)):
        return 0.0
    moved_values = (
        list(moved)
        if isinstance(moved, Sequence) and not isinstance(moved, (str, bytes))
        else []
    )
    count = 0
    total = 0
    for index, action in enumerate(resolved):
        action_name = str(action)
        moved_flag = bool(moved_values[index]) if index < len(moved_values) else False
        if action_name == "stay" or (
            action_name in MOVEMENT_ACTIONS and not moved_flag
        ):
            count += 1
        total += 1
    return _round(count / max(1, total))


def _optional_nonnegative_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return max(0, int(value))


def _int(value: object, *, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return int(value)


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    parsed = float(value)
    return parsed if math.isfinite(parsed) else 0.0


def _clip_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _clip_signed(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _round(value: float) -> float:
    return round(float(value), 6)
