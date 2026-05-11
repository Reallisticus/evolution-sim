from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from random import Random

from evolution_sim.env.runtime.action_contract import ACTION_NAMES
from evolution_sim.env.runtime.observations import (
    NAVIGATION_INPUT_FIELDS,
    NAVIGATION_TARGETS,
    PATCH_CELL_COUNT,
    PATCH_INPUT_FIELDS,
    SELF_INPUT_FIELDS,
)

MIND_V3_CONTROLLER_SCHEMA_VERSION = "mind_v3_controller_metadata_v1"
MIND_V3_POLICY_ID = "mind_v3_autonomous_evolution_policy"
MIND_V3_POLICY_VERSION = "mind_v3_autonomous_evolution_policy_v1"
MIND_V3_LEGACY_HIDDEN_UNITS = 8
MIND_V3_LOCAL_NAVIGATION_HIDDEN_UNITS = 16
MIND_V3_HIDDEN_UNITS = 24
MIND_V3_MUTATION_SIGMA = 0.035
MIND_V3_WEIGHT_LIMIT = 1.5
MIND_V3_REWARD_UPDATE_POLICY = "bounded_reward_modulated_controller_update_v1"
MIND_V3_REWARD_LEARNING_RATE = 0.025
MIND_V3_LEGACY_CONTROLLER_ARCHITECTURE = (
    "fixed_random_feature_projection_linear_action_head_v1"
)
MIND_V3_HOMEOSTATIC_CONTROLLER_ARCHITECTURE = (
    "homeostatic_feature_projection_linear_action_head_v2"
)
MIND_V3_LOCAL_NAVIGATION_CONTROLLER_ARCHITECTURE = (
    "local_navigation_feature_projection_linear_action_head_v3"
)
MIND_V3_CONTROLLER_ARCHITECTURE = (
    "need_gated_local_navigation_feature_projection_linear_action_head_v4"
)
MIND_V3_FOUNDER_PRIOR_POLICY = "diverse_need_gated_navigation_action_prior_v3"
MIND_V3_SPECIALIZATION_PROFILES: tuple[str, ...] = (
    "forager",
    "hydration_seeker",
    "disperser",
    "reproducer",
    "scavenger",
    "predator_scavenger",
)
MIND_V3_SELF_FIELD_INDEX = {
    field: index for index, field in enumerate(SELF_INPUT_FIELDS)
}
MIND_V3_HOMEOSTATIC_FEATURE_FIELDS: tuple[str, ...] = (
    "energy_ratio",
    "hydration_ratio",
    "health_ratio",
    "injury_load",
    "reproduction_ready",
    "matched_diet_ratio",
    "trophic_role_code",
    "meat_mode_code",
    "season_code",
    "water_access_reason_code",
    "hydrology_support_adjacent_to_water",
    "hydrology_support_wetland",
    "hydrology_support_flooded",
    "refuge_score",
    "hazard_type_code",
    "hazard_level",
    "tile_vegetation",
    "sexual_reproduction_unlocked",
    "reproductive_signal",
    "communication_signal",
)
MIND_V3_CONTEXT_FEATURE_FIELDS: tuple[str, ...] = (
    *MIND_V3_HOMEOSTATIC_FEATURE_FIELDS,
    "local_patch.food",
    "local_patch.fresh_kill_energy",
    "local_patch.carcass_energy",
    "local_patch.hazard_type_code",
    "local_patch.hazard_level",
    "navigation.water",
    "navigation.plant",
    "navigation.carrion",
    "navigation.prey",
)
MIND_V3_NEED_GATED_FEATURE_FIELDS: tuple[str, ...] = (
    "thirst_x_navigation.water.dx",
    "thirst_x_navigation.water.dy",
    "plant_hunger_x_navigation.plant.dx",
    "plant_hunger_x_navigation.plant.dy",
    "meat_hunger_x_navigation.carrion.dx",
    "meat_hunger_x_navigation.carrion.dy",
    "predatory_hunger_x_navigation.prey.dx",
    "predatory_hunger_x_navigation.prey.dy",
)
MIND_V3_MOVEMENT_ACTIONS = {
    "move_north",
    "move_south",
    "move_east",
    "move_west",
}
MIND_V3_ATTACK_ACTIONS = {
    "attack_north",
    "attack_south",
    "attack_east",
    "attack_west",
}


def mind_v3_parameter_count(*, architecture: str | None = None) -> int:
    hidden_units = _hidden_units_for_architecture(
        architecture or MIND_V3_CONTROLLER_ARCHITECTURE
    )
    return len(ACTION_NAMES) * hidden_units + len(ACTION_NAMES)


def load_mind_v3_controller_metadata(path: str | Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Mind v3 controller metadata file must contain an object")
    if payload.get("schema_version") == MIND_V3_CONTROLLER_SCHEMA_VERSION:
        return _validated_metadata(payload)
    best_candidate = payload.get("best_candidate")
    if isinstance(best_candidate, dict):
        metadata = best_candidate.get("controller_metadata")
        if isinstance(metadata, dict):
            return _validated_metadata(metadata)
    raise ValueError(
        "Mind v3 controller metadata file must be raw controller metadata "
        "or a mind_v3_evolution_search_v1 report with best_candidate metadata"
    )


def load_mind_v3_founder_template(
    path: str | Path,
) -> dict[str, object] | list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Mind v3 founder template file must contain an object")
    if payload.get("schema_version") == MIND_V3_CONTROLLER_SCHEMA_VERSION:
        return _validated_metadata(payload)
    best_candidate = payload.get("best_candidate")
    if isinstance(best_candidate, dict):
        pool = best_candidate.get("founder_template_pool")
        if isinstance(pool, list) and pool:
            templates: list[dict[str, object]] = []
            for index, metadata in enumerate(pool):
                if not isinstance(metadata, dict):
                    raise ValueError(
                        "Mind v3 founder_template_pool entries must be objects "
                        f"(entry {index})"
                    )
                templates.append(_validated_metadata(metadata))
            return templates
        metadata = best_candidate.get("controller_metadata")
        if isinstance(metadata, dict):
            return _validated_metadata(metadata)
    raise ValueError(
        "Mind v3 founder template file must be raw controller metadata "
        "or a mind_v3_evolution_search_v1 report with best_candidate metadata"
    )


def founder_mind_v3_metadata(
    *,
    agent_id: int,
    rng: Random,
    specialization_profile: str | None = None,
) -> dict[str, object]:
    local = Random((agent_id + 1) * 1_000_003 + int(rng.random() * 1_000_000_000))
    if specialization_profile is None:
        specialization_profile = MIND_V3_SPECIALIZATION_PROFILES[
            local.randrange(len(MIND_V3_SPECIALIZATION_PROFILES))
        ]
    elif specialization_profile not in MIND_V3_SPECIALIZATION_PROFILES:
        raise ValueError(
            f"unsupported Mind v3 specialization_profile {specialization_profile!r}"
        )
    return {
        "schema_version": MIND_V3_CONTROLLER_SCHEMA_VERSION,
        "inherited_state": True,
        "state_size": mind_v3_parameter_count(),
        "architecture": MIND_V3_CONTROLLER_ARCHITECTURE,
        "mutation_policy": "gaussian_head_mutation_v1",
        "founder_prior_policy": MIND_V3_FOUNDER_PRIOR_POLICY,
        "specialization_profile": specialization_profile,
        "parent_schema_versions": [],
        "action_head_weights": {
            action: [
                _round(
                    local.gauss(
                        _founder_action_weight_prior(
                            action,
                            unit,
                            specialization_profile,
                        ),
                        0.06,
                    )
                )
                for unit in range(MIND_V3_HIDDEN_UNITS)
            ]
            for action in ACTION_NAMES
        },
        "action_head_bias": {
            action: _round(
                local.gauss(
                    _founder_action_bias_prior(action, specialization_profile),
                    0.035,
                )
            )
            for action in ACTION_NAMES
        },
    }


def _validated_metadata(metadata: dict[str, object]) -> dict[str, object]:
    if metadata.get("schema_version") != MIND_V3_CONTROLLER_SCHEMA_VERSION:
        raise ValueError("Mind v3 controller metadata has stale schema_version")
    architecture = str(metadata.get("architecture", ""))
    if int(metadata.get("state_size", -1)) != mind_v3_parameter_count(
        architecture=architecture
    ):
        raise ValueError("Mind v3 controller metadata has invalid state_size")
    if architecture not in {
        MIND_V3_CONTROLLER_ARCHITECTURE,
        MIND_V3_LOCAL_NAVIGATION_CONTROLLER_ARCHITECTURE,
        MIND_V3_HOMEOSTATIC_CONTROLLER_ARCHITECTURE,
        MIND_V3_LEGACY_CONTROLLER_ARCHITECTURE,
    }:
        raise ValueError("Mind v3 controller metadata has unsupported architecture")
    validated = dict(metadata)
    validated["action_head_weights"] = _strict_weights(metadata)
    validated["action_head_bias"] = _strict_bias(metadata)
    return validated


def inherit_mind_v3_metadata(
    *,
    primary_parent_metadata: Mapping[str, object],
    secondary_parent_metadata: Mapping[str, object] | None,
    child_agent_id: int,
    rng: Random,
) -> dict[str, object]:
    architecture = _controller_architecture(primary_parent_metadata)
    hidden_units = _hidden_units_for_architecture(architecture)
    primary_weights = _weights(primary_parent_metadata)
    primary_bias = _bias(primary_parent_metadata)
    secondary_weights = (
        _weights(secondary_parent_metadata)
        if secondary_parent_metadata is not None
        else None
    )
    if secondary_weights is not None and any(
        len(values) != hidden_units for values in secondary_weights.values()
    ):
        secondary_weights = None
    secondary_bias = (
        _bias(secondary_parent_metadata)
        if secondary_parent_metadata is not None
        else None
    )
    weights: dict[str, list[float]] = {}
    bias: dict[str, float] = {}
    for action in ACTION_NAMES:
        weights[action] = []
        for index, value in enumerate(primary_weights[action]):
            base = value
            if secondary_weights is not None and rng.random() < 0.5:
                base = secondary_weights[action][index]
            weights[action].append(_mutated(base, rng))
        base_bias = primary_bias[action]
        if secondary_bias is not None and rng.random() < 0.5:
            base_bias = secondary_bias[action]
        bias[action] = _mutated(base_bias, rng)
    parent_versions = [str(primary_parent_metadata.get("schema_version", "unknown"))]
    if secondary_parent_metadata is not None:
        parent_versions.append(
            str(secondary_parent_metadata.get("schema_version", "unknown"))
        )
    return {
        "schema_version": MIND_V3_CONTROLLER_SCHEMA_VERSION,
        "inherited_state": True,
        "state_size": mind_v3_parameter_count(architecture=architecture),
        "architecture": architecture,
        "mutation_policy": "gaussian_head_mutation_v1",
        "child_agent_id": int(child_agent_id),
        "parent_schema_versions": parent_versions,
        "founder_prior_policy": primary_parent_metadata.get("founder_prior_policy"),
        "specialization_profile": primary_parent_metadata.get(
            "specialization_profile"
        ),
        "action_head_weights": weights,
        "action_head_bias": bias,
    }


def score_mind_v3_metadata(
    *,
    metadata: Mapping[str, object],
    observation_input: list[float],
    action_mask: Mapping[str, bool],
) -> dict[str, float]:
    hidden = _hidden_features(metadata, observation_input)
    weights = _weights(metadata)
    bias = _bias(metadata)
    scores: dict[str, float] = {}
    for action in ACTION_NAMES:
        if not bool(action_mask.get(action, False)):
            continue
        scores[action] = _round(
            bias[action]
            + sum(
                weights[action][index] * hidden[index]
                for index in range(len(hidden))
            )
        )
    return scores


def adapt_mind_v3_metadata(
    *,
    metadata: Mapping[str, object],
    observation_input: list[float],
    action: str,
    reward_signal: float,
) -> dict[str, object]:
    weights = _weights(metadata)
    bias = _bias(metadata)
    if action not in weights:
        return dict(metadata)
    hidden = _hidden_features(metadata, observation_input)
    signal = max(-1.0, min(1.0, float(reward_signal)))
    for index, value in enumerate(weights[action]):
        weights[action][index] = _round(
            max(
                -MIND_V3_WEIGHT_LIMIT,
                min(
                    MIND_V3_WEIGHT_LIMIT,
                    value + MIND_V3_REWARD_LEARNING_RATE * signal * hidden[index],
                ),
            )
        )
    bias[action] = _round(
        max(
            -MIND_V3_WEIGHT_LIMIT,
            min(
                MIND_V3_WEIGHT_LIMIT,
                bias[action] + MIND_V3_REWARD_LEARNING_RATE * signal,
            ),
        )
    )
    updated = dict(metadata)
    updated.update(
        {
            "schema_version": MIND_V3_CONTROLLER_SCHEMA_VERSION,
            "inherited_state": True,
            "state_size": mind_v3_parameter_count(
                architecture=_controller_architecture(metadata)
            ),
            "architecture": _controller_architecture(metadata),
            "last_update_policy": MIND_V3_REWARD_UPDATE_POLICY,
            "action_head_weights": weights,
            "action_head_bias": bias,
        }
    )
    return updated


def _weights(metadata: Mapping[str, object] | None) -> dict[str, list[float]]:
    hidden_units = _hidden_unit_count(metadata)
    if (
        metadata is None
        or metadata.get("schema_version") != MIND_V3_CONTROLLER_SCHEMA_VERSION
    ):
        return _zero_weights(hidden_units)
    raw = metadata.get("action_head_weights")
    if not isinstance(raw, Mapping):
        return _zero_weights(hidden_units)
    parsed = _zero_weights(hidden_units)
    for action in ACTION_NAMES:
        values = raw.get(action)
        if isinstance(values, list) and len(values) == hidden_units:
            parsed[action] = [_finite_float(value) for value in values]
    return parsed


def _bias(metadata: Mapping[str, object] | None) -> dict[str, float]:
    if (
        metadata is None
        or metadata.get("schema_version") != MIND_V3_CONTROLLER_SCHEMA_VERSION
    ):
        return {action: 0.0 for action in ACTION_NAMES}
    raw = metadata.get("action_head_bias")
    if not isinstance(raw, Mapping):
        return {action: 0.0 for action in ACTION_NAMES}
    return {action: _finite_float(raw.get(action, 0.0)) for action in ACTION_NAMES}


def _strict_weights(metadata: Mapping[str, object]) -> dict[str, list[float]]:
    hidden_units = _hidden_unit_count(metadata)
    raw = metadata.get("action_head_weights")
    if not isinstance(raw, Mapping):
        raise ValueError("Mind v3 controller metadata is missing action_head_weights")
    parsed: dict[str, list[float]] = {}
    for action in ACTION_NAMES:
        values = raw.get(action)
        if not isinstance(values, list):
            raise ValueError(
                f"Mind v3 controller metadata action_head_weights.{action} must be a list"
            )
        if len(values) != hidden_units:
            raise ValueError(
                f"Mind v3 controller metadata action_head_weights.{action} has invalid shape"
            )
        parsed[action] = [
            _strict_weight_value(
                value,
                field=f"action_head_weights.{action}[{index}]",
            )
            for index, value in enumerate(values)
        ]
    return parsed


def _strict_bias(metadata: Mapping[str, object]) -> dict[str, float]:
    raw = metadata.get("action_head_bias")
    if not isinstance(raw, Mapping):
        raise ValueError("Mind v3 controller metadata is missing action_head_bias")
    parsed: dict[str, float] = {}
    for action in ACTION_NAMES:
        parsed[action] = _strict_weight_value(
            raw.get(action),
            field=f"action_head_bias.{action}",
        )
    return parsed


def _strict_weight_value(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Mind v3 controller metadata {field} must be finite")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Mind v3 controller metadata {field} must be finite")
    if parsed < -MIND_V3_WEIGHT_LIMIT or parsed > MIND_V3_WEIGHT_LIMIT:
        raise ValueError(
            f"Mind v3 controller metadata {field} exceeds weight bounds"
        )
    return _round(parsed)


def _zero_weights(hidden_units: int) -> dict[str, list[float]]:
    return {action: [0.0] * hidden_units for action in ACTION_NAMES}


def _mutated(value: float, rng: Random) -> float:
    return _round(
        max(
            -MIND_V3_WEIGHT_LIMIT,
            min(MIND_V3_WEIGHT_LIMIT, value + rng.gauss(0.0, MIND_V3_MUTATION_SIGMA)),
        )
    )


def _controller_architecture(metadata: Mapping[str, object] | None) -> str:
    if metadata is None:
        return MIND_V3_CONTROLLER_ARCHITECTURE
    architecture = metadata.get("architecture")
    if architecture == MIND_V3_LEGACY_CONTROLLER_ARCHITECTURE:
        return MIND_V3_LEGACY_CONTROLLER_ARCHITECTURE
    if architecture == MIND_V3_HOMEOSTATIC_CONTROLLER_ARCHITECTURE:
        return MIND_V3_HOMEOSTATIC_CONTROLLER_ARCHITECTURE
    if architecture == MIND_V3_LOCAL_NAVIGATION_CONTROLLER_ARCHITECTURE:
        return MIND_V3_LOCAL_NAVIGATION_CONTROLLER_ARCHITECTURE
    return MIND_V3_CONTROLLER_ARCHITECTURE


def _hidden_units_for_architecture(architecture: str) -> int:
    if architecture in {
        MIND_V3_LEGACY_CONTROLLER_ARCHITECTURE,
        MIND_V3_HOMEOSTATIC_CONTROLLER_ARCHITECTURE,
    }:
        return MIND_V3_LEGACY_HIDDEN_UNITS
    if architecture == MIND_V3_LOCAL_NAVIGATION_CONTROLLER_ARCHITECTURE:
        return MIND_V3_LOCAL_NAVIGATION_HIDDEN_UNITS
    return MIND_V3_HIDDEN_UNITS


def _hidden_unit_count(metadata: Mapping[str, object] | None) -> int:
    return _hidden_units_for_architecture(_controller_architecture(metadata))


def _hidden_features(
    metadata: Mapping[str, object],
    observation_input: list[float],
) -> list[float]:
    architecture = _controller_architecture(metadata)
    if architecture == MIND_V3_LEGACY_CONTROLLER_ARCHITECTURE:
        return _fixed_hidden_features(observation_input)
    if architecture == MIND_V3_HOMEOSTATIC_CONTROLLER_ARCHITECTURE:
        return _homeostatic_hidden_features(observation_input)
    if architecture == MIND_V3_LOCAL_NAVIGATION_CONTROLLER_ARCHITECTURE:
        return _local_navigation_hidden_features(observation_input)
    return _need_gated_local_navigation_hidden_features(observation_input)


def _homeostatic_hidden_features(observation_input: list[float]) -> list[float]:
    values = [_finite_float(value) for value in observation_input]
    energy = _self_feature(values, "energy_ratio", default=1.0)
    hydration = _self_feature(values, "hydration_ratio", default=1.0)
    health = _self_feature(values, "health_ratio", default=1.0)
    injury = _self_feature(values, "injury_load")
    reproduction_ready = _self_feature(values, "reproduction_ready")
    matched_diet = _self_feature(values, "matched_diet_ratio")
    trophic_role = _self_feature(values, "trophic_role_code")
    meat_mode = _self_feature(values, "meat_mode_code")
    season = _self_feature(values, "season_code")
    water_access = _self_feature(values, "water_access_reason_code")
    adjacent_water = _self_feature(values, "hydrology_support_adjacent_to_water")
    wetland = _self_feature(values, "hydrology_support_wetland")
    flooded = _self_feature(values, "hydrology_support_flooded")
    refuge = _self_feature(values, "refuge_score")
    hazard_type = _self_feature(values, "hazard_type_code")
    hazard_level = _self_feature(values, "hazard_level")
    vegetation = _self_feature(values, "tile_vegetation")
    sexual_unlocked = _self_feature(values, "sexual_reproduction_unlocked")
    reproductive_signal = _self_feature(values, "reproductive_signal")
    communication_signal = _self_feature(values, "communication_signal")
    water_context = max(water_access, adjacent_water, wetland, flooded)
    hazard = max(hazard_type, hazard_level)
    signal_context = max(reproductive_signal, communication_signal)
    return [
        _round(math.tanh((0.65 - energy) * 3.0)),
        _round(math.tanh((0.65 - hydration) * 3.0)),
        _round(math.tanh((0.60 - health) * 2.0 + injury * 1.5 + hazard * 1.2)),
        _round(
            math.tanh(
                reproduction_ready * 2.0
                + sexual_unlocked
                + reproductive_signal
                - 1.0
            )
        ),
        _round(
            math.tanh(
                matched_diet * 1.5
                + vegetation * 1.2
                + (1.0 - energy) * 0.6
                - 1.0
            )
        ),
        _round(math.tanh(water_context * 2.0 + (1.0 - hydration) * 0.8 - 1.0)),
        _round(
            math.tanh(
                (1.0 - refuge) * 0.8
                + hazard * 1.2
                + (1.0 - vegetation) * 0.4
                + (1.0 - water_context) * 0.2
                - 0.7
            )
        ),
        _round(
            math.tanh(
                (trophic_role * 2.0 - 1.0)
                + (meat_mode * 2.0 - 1.0)
                + season * 0.5
                + signal_context * 0.5
            )
        ),
    ]


def _local_navigation_hidden_features(observation_input: list[float]) -> list[float]:
    values = [_finite_float(value) for value in observation_input]
    energy = _self_feature(values, "energy_ratio", default=1.0)
    hydration = _self_feature(values, "hydration_ratio", default=1.0)
    health = _self_feature(values, "health_ratio", default=1.0)
    injury = _self_feature(values, "injury_load")
    reproduction_ready = _self_feature(values, "reproduction_ready")
    matched_diet = _self_feature(values, "matched_diet_ratio")
    trophic_role = _self_feature(values, "trophic_role_code")
    meat_mode = _self_feature(values, "meat_mode_code")
    season = _self_feature(values, "season_code")
    water_access = _self_feature(values, "water_access_reason_code")
    adjacent_water = _self_feature(values, "hydrology_support_adjacent_to_water")
    wetland = _self_feature(values, "hydrology_support_wetland")
    flooded = _self_feature(values, "hydrology_support_flooded")
    refuge = _self_feature(values, "refuge_score")
    hazard_type = _self_feature(values, "hazard_type_code")
    hazard_level = _self_feature(values, "hazard_level")
    vegetation = _self_feature(values, "tile_vegetation")
    sexual_unlocked = _self_feature(values, "sexual_reproduction_unlocked")
    reproductive_signal = _self_feature(values, "reproductive_signal")
    communication_signal = _self_feature(values, "communication_signal")
    center_food = _center_patch_feature(values, "food")
    center_fresh_kill = _center_patch_feature(values, "fresh_kill_energy")
    center_carcass = _center_patch_feature(values, "carcass_energy")
    center_hazard = max(
        _center_patch_feature(values, "hazard_type_code"),
        _center_patch_feature(values, "hazard_level"),
    )
    water_context = max(water_access, adjacent_water, wetland, flooded)
    hazard = max(hazard_type, hazard_level, center_hazard)
    signal_context = max(reproductive_signal, communication_signal)
    local_plant = max(center_food, vegetation)
    local_animal = max(center_fresh_kill, center_carcass)
    hunger = max(0.0, 0.68 - energy)
    thirst = max(0.0, 0.68 - hydration)
    water_dx, water_dy = _navigation_vector(values, "water")
    plant_dx, plant_dy = _navigation_vector(values, "plant")
    carrion_dx, carrion_dy = _navigation_vector(values, "carrion")
    prey_dx, prey_dy = _navigation_vector(values, "prey")
    movement_pressure = max(
        abs(water_dx),
        abs(water_dy),
        abs(plant_dx),
        abs(plant_dy),
        abs(carrion_dx),
        abs(carrion_dy),
        abs(prey_dx),
        abs(prey_dy),
    )
    return [
        _round(math.tanh(hunger * 4.2 + local_plant * 0.7 + local_animal * 1.1)),
        _round(math.tanh(thirst * 4.2 + water_context * 1.0)),
        _round(math.tanh((0.60 - health) * 2.0 + injury * 1.5 + hazard * 1.2)),
        _round(
            math.tanh(
                reproduction_ready * 2.0
                + sexual_unlocked
                + reproductive_signal
                - 1.0
            )
        ),
        _round(math.tanh(local_plant * 2.4 + matched_diet * 0.8 + hunger - 0.6)),
        _round(math.tanh(local_animal * 2.8 + meat_mode * 0.7 + hunger - 0.55)),
        water_dx,
        water_dy,
        plant_dx,
        plant_dy,
        carrion_dx,
        carrion_dy,
        prey_dx,
        prey_dy,
        _round(math.tanh(movement_pressure * 2.0 + hazard * 0.7 - 0.6)),
        _round(
            math.tanh(
                refuge * 1.4
                + water_context * 0.7
                + vegetation * 0.5
                + signal_context * 0.25
                + season * 0.15
                + trophic_role * 0.15
                - hazard * 1.2
                - 0.6
            )
        ),
    ]


def _need_gated_local_navigation_hidden_features(
    observation_input: list[float],
) -> list[float]:
    values = [_finite_float(value) for value in observation_input]
    base = _local_navigation_hidden_features(values)
    energy = _self_feature(values, "energy_ratio", default=1.0)
    hydration = _self_feature(values, "hydration_ratio", default=1.0)
    matched_diet = _self_feature(values, "matched_diet_ratio")
    trophic_role = _self_feature(values, "trophic_role_code")
    meat_mode = _self_feature(values, "meat_mode_code")
    vegetation = _self_feature(values, "tile_vegetation")
    local_plant = max(_center_patch_feature(values, "food"), vegetation)
    local_animal = max(
        _center_patch_feature(values, "fresh_kill_energy"),
        _center_patch_feature(values, "carcass_energy"),
    )
    hunger = max(0.0, 0.72 - energy) / 0.72
    thirst = max(0.0, 0.72 - hydration) / 0.72
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
    water_dx, water_dy = _navigation_vector(values, "water")
    plant_dx, plant_dy = _navigation_vector(values, "plant")
    carrion_dx, carrion_dy = _navigation_vector(values, "carrion")
    prey_dx, prey_dy = _navigation_vector(values, "prey")
    return [
        *base,
        _round(thirst * water_dx),
        _round(thirst * water_dy),
        _round(hunger * plant_preference * plant_dx),
        _round(hunger * plant_preference * plant_dy),
        _round(hunger * meat_preference * carrion_dx),
        _round(hunger * meat_preference * carrion_dy),
        _round(hunger * prey_preference * prey_dx),
        _round(hunger * prey_preference * prey_dy),
    ]


def _fixed_hidden_features(observation_input: list[float]) -> list[float]:
    values = [_finite_float(value) for value in observation_input]
    if not values:
        values = [0.0]
    hidden: list[float] = []
    for unit in range(MIND_V3_LEGACY_HIDDEN_UNITS):
        total = 0.0
        for offset in range(12):
            index = (unit * 37 + offset * 53) % len(values)
            total += values[index] * _projection_weight(unit, offset)
        hidden.append(_round(math.tanh(total / 4.0)))
    return hidden


def _projection_weight(unit: int, offset: int) -> float:
    return ((unit * 17 + offset * 31) % 23 - 11) / 11.0


def _founder_action_weight_prior(
    action: str,
    unit: int,
    specialization_profile: str,
) -> float:
    prior = 0.0
    directional_units = _directional_prior_units(action)
    if action == "eat":
        if unit == 0:
            prior += 0.16
        if unit in {4, 5}:
            prior += 0.12
    elif action == "drink":
        if unit == 1:
            prior += 0.18
        if unit in {6, 7}:
            prior += 0.12
    elif action == "mate":
        if unit == 3:
            prior += 0.20
        if unit in {0, 1, 2}:
            prior -= 0.04
    elif action in MIND_V3_MOVEMENT_ACTIONS:
        if unit == 14:
            prior += 0.10
        prior += directional_units.get(unit, 0.0)
    elif action in MIND_V3_ATTACK_ACTIONS:
        if unit in {0, 5, 12, 13}:
            prior += 0.07
        if unit == 2:
            prior -= 0.04
        prior += directional_units.get(unit, 0.0) * 0.8
    elif action.startswith("signal_"):
        if unit == 3:
            prior += 0.03
        if unit == 15:
            prior += 0.02
    if specialization_profile == "forager":
        if action == "eat" and unit in {0, 4, 5}:
            prior += 0.08
        if action in MIND_V3_MOVEMENT_ACTIONS and unit in {8, 9}:
            prior += 0.04
        if action in MIND_V3_ATTACK_ACTIONS:
            prior -= 0.02
    elif specialization_profile == "hydration_seeker":
        if action == "drink" and unit in {1, 6, 7}:
            prior += 0.09
        if action in MIND_V3_MOVEMENT_ACTIONS and unit in {6, 7}:
            prior += 0.05
    elif specialization_profile == "disperser":
        if action in MIND_V3_MOVEMENT_ACTIONS and unit in {8, 9, 10, 11, 14}:
            prior += 0.08
        if action == "stay" and unit == 14:
            prior -= 0.05
    elif specialization_profile == "reproducer":
        if action == "mate" and unit == 3:
            prior += 0.12
        if action.startswith("signal_") and unit in {3, 15}:
            prior += 0.05
    elif specialization_profile == "scavenger":
        if action == "eat" and unit in {0, 5}:
            prior += 0.11
        if action == "drink" and unit in {1, 6, 7}:
            prior += 0.04
        if action in MIND_V3_MOVEMENT_ACTIONS and unit in {10, 11, 14}:
            prior += 0.11
        if action in MIND_V3_ATTACK_ACTIONS and unit in {0, 5, 12, 13}:
            prior -= 0.08
    elif specialization_profile == "predator_scavenger":
        if action in MIND_V3_ATTACK_ACTIONS and unit in {0, 5, 12, 13}:
            prior += 0.10
        if action in MIND_V3_MOVEMENT_ACTIONS and unit in {10, 11, 12, 13}:
            prior += 0.05
        if action == "eat" and unit in {0, 5}:
            prior += 0.05
    return max(-MIND_V3_WEIGHT_LIMIT, min(MIND_V3_WEIGHT_LIMIT, prior))


def _directional_prior_units(action: str) -> dict[int, float]:
    if action.endswith("_east"):
        return {
            6: 0.06,
            8: 0.05,
            10: 0.07,
            12: 0.08,
            16: 0.11,
            18: 0.08,
            20: 0.1,
            22: 0.09,
        }
    if action.endswith("_west"):
        return {
            6: -0.06,
            8: -0.05,
            10: -0.07,
            12: -0.08,
            16: -0.11,
            18: -0.08,
            20: -0.1,
            22: -0.09,
        }
    if action.endswith("_south"):
        return {
            7: 0.06,
            9: 0.05,
            11: 0.07,
            13: 0.08,
            17: 0.11,
            19: 0.08,
            21: 0.1,
            23: 0.09,
        }
    if action.endswith("_north"):
        return {
            7: -0.06,
            9: -0.05,
            11: -0.07,
            13: -0.08,
            17: -0.11,
            19: -0.08,
            21: -0.1,
            23: -0.09,
        }
    return {}


def _founder_action_bias_prior(action: str, specialization_profile: str) -> float:
    if action.startswith("signal_"):
        return 0.02 if specialization_profile == "reproducer" else -0.08
    if action in MIND_V3_ATTACK_ACTIONS:
        if specialization_profile == "scavenger":
            return -0.08
        return 0.02 if specialization_profile == "predator_scavenger" else -0.03
    if action in MIND_V3_MOVEMENT_ACTIONS:
        if specialization_profile == "scavenger":
            return 0.02
        return 0.04 if specialization_profile == "disperser" else 0.0
    if action == "mate":
        return 0.04 if specialization_profile == "reproducer" else -0.03
    if action == "drink":
        if specialization_profile == "hydration_seeker":
            return 0.03
        return 0.02 if specialization_profile == "scavenger" else 0.01
    if action == "eat":
        return (
            0.035
            if specialization_profile == "scavenger"
            else (
                0.03
                if specialization_profile in {"forager", "predator_scavenger"}
                else 0.01
            )
        )
    return 0.0


def _self_feature(values: list[float], field: str, *, default: float = 0.0) -> float:
    index = MIND_V3_SELF_FIELD_INDEX[field]
    if index >= len(values):
        return default
    return max(0.0, min(1.0, float(values[index])))


def _center_patch_feature(
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


def _navigation_feature(
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


def _navigation_vector(values: list[float], target: str) -> tuple[float, float]:
    strength = max(0.0, _navigation_feature(values, target, "strength"))
    distance = max(0.0, min(1.0, _navigation_feature(values, target, "distance")))
    falloff = max(0.0, 1.0 - distance * 0.35)
    scale = strength * falloff
    return (
        _round(_navigation_feature(values, target, "dx") * scale),
        _round(_navigation_feature(values, target, "dy") * scale),
    )


def _finite_float(value: object) -> float:
    number = (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else 0.0
    )
    if not math.isfinite(number):
        return 0.0
    return max(-MIND_V3_WEIGHT_LIMIT, min(MIND_V3_WEIGHT_LIMIT, number))


def _round(value: float) -> float:
    return round(float(value), 6)
