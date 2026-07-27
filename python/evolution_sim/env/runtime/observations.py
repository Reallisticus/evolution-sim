from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
import struct
from typing import Any, Callable
import zlib

from evolution_sim.env.runtime.action_contract import action_contract, action_names
from evolution_sim.env.runtime.mating import (
    ASEXUAL_REPRODUCTION_MODE,
    PROTO_X_EXPRESSION,
    PROTO_Y_EXPRESSION,
    PROTO_Z_EXPRESSION,
    SEXUAL_EXPRESSION,
    X_EXPRESSION,
    Y_EXPRESSION,
    Z_EXPRESSION,
)
from evolution_sim.env.runtime.signals import (
    COMMUNICATION_AGGREGATE_PROJECTION,
    COMMUNICATION_SIGNAL_FIELD,
    REPRODUCTIVE_SIGNAL_FIELD,
    SIGNAL_FIELD_NAMES,
    communication_signal_emission_enabled,
    communication_token_field_names,
    parse_communication_token_field_name,
    signal_contract,
)
from evolution_sim.env.runtime.state import (
    MIND_INHERITANCE_PLACEHOLDER_VERSION,
    Agent,
    Tile,
    TrophicProfile,
)

OBSERVATION_SCHEMA_VERSION = "mind_observation_v3"
OBSERVATION_ENCODER_VERSION = "mind_observation_encoder_v2"
TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION = "mind_observation_v5"
TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION = (
    "mind_observation_encoder_v4"
)
OBSERVATION_INPUT_DTYPE = "float32"
OBSERVATION_STORAGE_DTYPE = "int16"
OBSERVATION_STORAGE_ENCODING = "zlib_base64_little_endian_int16"
OBSERVATION_QUANTIZATION_SCALE = 32767.0
OBSERVATION_INPUT_VALUE_RANGE: tuple[float, float] = (-1.0, 1.0)
LOCAL_PATCH_RADIUS = 2
NAVIGATION_RADIUS = 10
METADATA_FIELDS: tuple[str, ...] = ("agent_id",)
SELF_FIELDS: tuple[str, ...] = (
    "energy_ratio",
    "hydration_ratio",
    "health_ratio",
    "injury_load",
    "age_norm",
    "reproduction_ready",
    "matched_diet_ratio",
    "trophic_role",
    "meat_mode",
    "season",
    "water_access_reason",
    "hydrology_support_code",
    "refuge_score",
    "hazard_type",
    "hazard_level",
    "tile_vegetation",
    "tile_recovery_debt",
    "reproductive_stage",
    "reproductive_expression",
    "sexual_reproduction_unlocked",
    "reproductive_signal",
    "communication_signal",
    "mind_inheritance_available",
)
PATCH_FIELDS: tuple[str, ...] = (
    "dx",
    "dy",
    "in_bounds",
    "terrain",
    "occupant",
    "same_lineage",
    "water_access_reason",
    "food",
    "vegetation",
    "recovery_debt",
    "fresh_kill_energy",
    "carcass_energy",
    "hazard_type",
    "hazard_level",
    "ecology_state",
    "prey_biomass",
    "carrion_signal",
    "predator_risk",
    "reproductive_signal",
    "communication_signal",
)
NAVIGATION_TARGETS: tuple[str, ...] = ("water", "plant", "carrion", "prey")
NAVIGATION_FIELDS: tuple[str, ...] = ("dx", "dy", "distance", "strength")
SELF_INPUT_FIELDS: tuple[str, ...] = (
    "energy_ratio",
    "hydration_ratio",
    "health_ratio",
    "injury_load",
    "age_norm",
    "reproduction_ready",
    "matched_diet_ratio",
    "trophic_role_code",
    "meat_mode_code",
    "season_code",
    "water_access_reason_code",
    "hydrology_support_is_land",
    "hydrology_support_adjacent_to_water",
    "hydrology_support_wetland",
    "hydrology_support_flooded",
    "refuge_score",
    "hazard_type_code",
    "hazard_level",
    "tile_vegetation",
    "tile_recovery_debt",
    "reproductive_stage_code",
    "reproductive_expression_code",
    "sexual_reproduction_unlocked",
    "reproductive_signal",
    "communication_signal",
    "mind_inheritance_available",
)
PATCH_INPUT_FIELDS: tuple[str, ...] = (
    "dx",
    "dy",
    "in_bounds",
    "terrain_code",
    "occupant_code",
    "same_lineage",
    "water_access_reason_code",
    "food",
    "vegetation",
    "recovery_debt",
    "fresh_kill_energy",
    "carcass_energy",
    "hazard_type_code",
    "hazard_level",
    "ecology_state_code",
    "prey_biomass",
    "carrion_signal",
    "predator_risk",
    "reproductive_signal",
    "communication_signal",
)
NAVIGATION_INPUT_FIELDS: tuple[str, ...] = (
    "dx",
    "dy",
    "distance",
    "strength",
)
SEASON_VOCAB: tuple[str, ...] = ("wet", "dry")
TERRAIN_VOCAB: tuple[str, ...] = (
    "out_of_bounds",
    "plain",
    "forest",
    "wetland",
    "rocky",
    "water",
)
OCCUPANT_VOCAB: tuple[str, ...] = ("none", "self", "agent")
TROPHIC_ROLE_VOCAB: tuple[str, ...] = ("none", "herbivore", "omnivore", "carnivore")
MEAT_MODE_VOCAB: tuple[str, ...] = ("none", "scavenger", "hunter", "mixed")
REPRODUCTIVE_STAGE_VOCAB: tuple[str, ...] = (
    "stage0_asexual",
    "stage1_facultative_sex",
    "stage2_proto_roles",
    "stage3_x_y_z",
    "stage4_hybridization",
)
REPRODUCTIVE_EXPRESSION_VOCAB: tuple[str, ...] = (
    ASEXUAL_REPRODUCTION_MODE,
    SEXUAL_EXPRESSION,
    PROTO_X_EXPRESSION,
    PROTO_Y_EXPRESSION,
    PROTO_Z_EXPRESSION,
    X_EXPRESSION,
    Y_EXPRESSION,
    Z_EXPRESSION,
)
WATER_ACCESS_REASON_VOCAB: tuple[str, ...] = (
    "none",
    "adjacent_water",
    "wetland",
    "flooded",
)
HAZARD_TYPE_VOCAB: tuple[str, ...] = ("none", "exposure", "instability")
ECOLOGY_STATE_VOCAB: tuple[str, ...] = (
    "none",
    "stable",
    "lush",
    "recovering",
    "depleted",
)
ENUM_VOCABS: dict[str, tuple[str, ...]] = {
    "season": SEASON_VOCAB,
    "terrain": TERRAIN_VOCAB,
    "occupant": OCCUPANT_VOCAB,
    "trophic_role": TROPHIC_ROLE_VOCAB,
    "meat_mode": MEAT_MODE_VOCAB,
    "reproductive_stage": REPRODUCTIVE_STAGE_VOCAB,
    "reproductive_expression": REPRODUCTIVE_EXPRESSION_VOCAB,
    "water_access_reason": WATER_ACCESS_REASON_VOCAB,
    "hazard_type": HAZARD_TYPE_VOCAB,
    "ecology_state": ECOLOGY_STATE_VOCAB,
}
PATCH_CELL_COUNT = (LOCAL_PATCH_RADIUS * 2 + 1) ** 2
OBSERVATION_INPUT_VECTOR_SIZE = len(SELF_INPUT_FIELDS) + (
    PATCH_CELL_COUNT * len(PATCH_INPUT_FIELDS)
) + (
    len(NAVIGATION_TARGETS) * len(NAVIGATION_INPUT_FIELDS)
)


def observation_schema_version(signal_config: Any | None = None) -> str:
    if (
        signal_config is not None
        and communication_signal_emission_enabled(signal_config)
    ):
        return TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION
    return OBSERVATION_SCHEMA_VERSION


def observation_encoder_version(signal_config: Any | None = None) -> str:
    if (
        signal_config is not None
        and communication_signal_emission_enabled(signal_config)
    ):
        return TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION
    return OBSERVATION_ENCODER_VERSION


def observation_input_vector_size(signal_config: Any | None = None) -> int:
    token_count = len(communication_token_field_names(signal_config))
    return OBSERVATION_INPUT_VECTOR_SIZE + token_count * (1 + PATCH_CELL_COUNT)


@dataclass(frozen=True, slots=True)
class ObservationContext:
    width: int
    height: int
    max_age: int
    grid: Sequence[Sequence[Tile]]
    agents: Mapping[int, Agent]
    climate_state: dict[str, object]
    biotic_state: Any
    signal_state: Any
    action_mask: dict[str, bool]
    movement_actions: tuple[tuple[str, int, int], ...]
    profile_for: Callable[[Agent], TrophicProfile]
    energy_ratio: Callable[[Agent], float]
    hydration_ratio: Callable[[Agent], float]
    health_ratio: Callable[[Agent], float]
    is_reproduction_ready: Callable[[Agent], bool]
    matched_diet_ratio: Callable[[Agent, TrophicProfile], float]
    water_access_reason: Callable[[int, int], str]
    hydrology_support_code: Callable[[int, int], int]
    refuge_score: Callable[[int, int], float]
    hazard_at: Callable[[int, int], tuple[str, float]]
    ecology_state_at: Callable[[int, int], str]
    in_bounds: Callable[[int, int], bool]
    prey_vulnerability: Callable[[Agent], float]


def observation_contract(signal_config: Any | None = None) -> dict[str, object]:
    token_fields = communication_token_field_names(signal_config)
    schema_version = observation_schema_version(signal_config)
    encoder_version = observation_encoder_version(signal_config)
    vector_size = observation_input_vector_size(signal_config)
    contract: dict[str, object] = {
        "schema_version": schema_version,
        "local_patch_radius": LOCAL_PATCH_RADIUS,
        "metadata_fields": list(METADATA_FIELDS),
        "metadata_policy_excluded": True,
        "self_fields": [*SELF_FIELDS, *token_fields],
        "patch_fields": [*PATCH_FIELDS, *token_fields],
        "navigation_radius": NAVIGATION_RADIUS,
        "navigation_targets": list(NAVIGATION_TARGETS),
        "navigation_fields": list(NAVIGATION_FIELDS),
        "action_names": list(action_names(signal_config)),
        "action_contract": action_contract(signal_config),
        "signal_contract": signal_contract(signal_config),
        "mind_inheritance_placeholder": {
            "schema_version": MIND_INHERITANCE_PLACEHOLDER_VERSION,
            "policy_visible": False,
        },
        "enum_vocabs": {
            name: list(values) for name, values in sorted(ENUM_VOCABS.items())
        },
        "policy_input": {
            "semantic_role": "raw_encoded_observation_tensor",
            "compatibility_role": "historical_policy_input_key_compatibility",
            "encoder_version": encoder_version,
            "decoded_dtype": OBSERVATION_INPUT_DTYPE,
            "storage_dtype": OBSERVATION_STORAGE_DTYPE,
            "storage_encoding": OBSERVATION_STORAGE_ENCODING,
            "shape": [vector_size],
            "value_range": list(OBSERVATION_INPUT_VALUE_RANGE),
            "self_input_fields": [*SELF_INPUT_FIELDS, *token_fields],
            "patch_input_fields": [*PATCH_INPUT_FIELDS, *token_fields],
            "patch_cell_count": PATCH_CELL_COUNT,
            "patch_order": "row_major_dy_then_dx_centered",
            "navigation_input_fields": list(NAVIGATION_INPUT_FIELDS),
            "navigation_target_order": list(NAVIGATION_TARGETS),
            "categorical_encoding": "normalized_ordinal_code",
            "nonnegative_signal_encoding": "x/(1+x)",
            "quantization_scale": OBSERVATION_QUANTIZATION_SCALE,
            "contains_controller_private_diagnostics": True,
            "controller_private_diagnostic_fields": [
                "self.mind_inheritance_available"
            ],
            "promotion_eligible_direct_policy_input": False,
            "safe_projection_required_for_mind_v3_promotion": True,
            "promotion_safe_projection_examples": [
                "mind_ecological_policy_input_v1",
                "architecture_specific_safe_feature_selection_v1",
            ],
            "promotion_policy_input_guidance": (
                "Mind v3 promotion paths must use a safe projection such as "
                "ecological policy input or architecture-specific safe feature "
                "selection that excludes controller-private diagnostics."
            ),
        },
        "privileged_world_state": False,
    }
    if token_fields:
        contract["communication_token_channels"] = {
            "policy_visible": True,
            "spatial": True,
            "field_order": list(token_fields),
            "token_order": list(range(len(token_fields))),
            "simulator_assigned_meanings": False,
            "profile_provenance_policy_visible": False,
            "aggregate_communication_field_retained": True,
            "aggregate_projection": COMMUNICATION_AGGREGATE_PROJECTION,
        }
    return contract


def _resolve_observation_context(
    world: Any,
    agent: Agent,
    *,
    observation_context: ObservationContext | None = None,
) -> ObservationContext:
    if observation_context is not None:
        return observation_context
    raise ValueError("observation_context is required")


def build_observation(
    world: Any,
    agent: Agent,
    *,
    observation_context: ObservationContext | None = None,
) -> dict[str, object]:
    context = _resolve_observation_context(
        world,
        agent,
        observation_context=observation_context,
    )
    climate_state = context.climate_state
    profile = context.profile_for(agent)
    tile = context.grid[agent.y][agent.x]
    hazard_type, hazard_level = context.hazard_at(agent.x, agent.y)
    signal_state = context.signal_state
    token_fields = _signal_state_communication_token_fields(signal_state)
    self_state: dict[str, object] = {
        "energy_ratio": _round(context.energy_ratio(agent)),
        "hydration_ratio": _round(context.hydration_ratio(agent)),
        "health_ratio": _round(context.health_ratio(agent)),
        "injury_load": _round(agent.injury_load),
        "age_norm": _round(agent.age / max(context.max_age, 1)),
        "reproduction_ready": bool(context.is_reproduction_ready(agent)),
        "matched_diet_ratio": _round(
            context.matched_diet_ratio(agent, profile)
        ),
        "trophic_role": profile.role,
        "meat_mode": profile.meat_mode,
        "season": str(climate_state["season"]),
        "water_access_reason": context.water_access_reason(agent.x, agent.y),
        "hydrology_support_code": context.hydrology_support_code(agent.x, agent.y),
        "refuge_score": _round(context.refuge_score(agent.x, agent.y)),
        "hazard_type": hazard_type,
        "hazard_level": _round(hazard_level),
        "tile_vegetation": _round(tile.vegetation),
        "tile_recovery_debt": _round(tile.recovery_debt),
        "reproductive_stage": agent.reproductive_stage,
        "reproductive_expression": agent.reproductive_expression,
        "sexual_reproduction_unlocked": agent.reproductive_stage != "stage0_asexual",
        REPRODUCTIVE_SIGNAL_FIELD: _round(
            signal_state.reproductive_signal[agent.y][agent.x]
        ),
        COMMUNICATION_SIGNAL_FIELD: _round(
            _signal_value_for_observer(
                signal_state,
                COMMUNICATION_SIGNAL_FIELD,
                observer_agent_id=agent.agent_id,
                x=agent.x,
                y=agent.y,
            )
        ),
        "mind_inheritance_available": bool(
            agent.mind_inheritance_metadata.get("inherited_state", False)
        ),
    }
    self_state.update(
        {
            field_name: _round(
                _signal_value_for_observer(
                    signal_state,
                    field_name,
                    observer_agent_id=agent.agent_id,
                    x=agent.x,
                    y=agent.y,
                )
            )
            for field_name in token_fields
        }
    )
    return {
        "schema_version": (
            TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION
            if token_fields
            else OBSERVATION_SCHEMA_VERSION
        ),
        "metadata": {
            "agent_id": agent.agent_id,
        },
        "self": self_state,
        "local_patch": [
            _patch_cell(context, agent, dx, dy)
            for dy in range(-LOCAL_PATCH_RADIUS, LOCAL_PATCH_RADIUS + 1)
            for dx in range(-LOCAL_PATCH_RADIUS, LOCAL_PATCH_RADIUS + 1)
        ],
        "navigation": _navigation_targets(context, agent),
        "action_mask": dict(context.action_mask),
    }


def observation_digest(observation: dict[str, object]) -> str:
    payload = json.dumps(observation, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _observation_communication_token_fields(
    observation: dict[str, object],
) -> tuple[str, ...]:
    self_state = observation.get("self")
    patch = observation.get("local_patch")
    if not isinstance(self_state, dict):
        raise ValueError("observation self section must be a mapping")
    if not isinstance(patch, list) or len(patch) != PATCH_CELL_COUNT:
        raise ValueError(
            f"observation local_patch must contain {PATCH_CELL_COUNT} cells"
        )
    token_fields_by_id: dict[int, str] = {}
    for field_name in self_state:
        token_id = parse_communication_token_field_name(str(field_name))
        if token_id is not None:
            token_fields_by_id[token_id] = str(field_name)
    token_fields = tuple(
        token_fields_by_id[token_id] for token_id in sorted(token_fields_by_id)
    )
    if token_fields and tuple(sorted(token_fields_by_id)) != tuple(
        range(len(token_fields))
    ):
        raise ValueError(
            "communication token observation fields must use contiguous token ids"
        )
    expected_fields = set(token_fields)
    for cell in patch:
        if not isinstance(cell, dict):
            raise ValueError("observation local_patch cells must be mappings")
        cell_token_fields = {
            str(field_name)
            for field_name in cell
            if parse_communication_token_field_name(str(field_name)) is not None
        }
        if cell_token_fields != expected_fields:
            raise ValueError(
                "communication token observation fields must match across "
                "self and local_patch"
            )
    return token_fields


def encode_observation_input(observation: dict[str, object]) -> dict[str, object]:
    """Encode the policy-visible observation as a compact, versioned tensor payload."""
    schema_version = observation.get("schema_version")
    if schema_version not in {
        OBSERVATION_SCHEMA_VERSION,
        TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
    }:
        raise ValueError("observation schema_version is missing or stale")
    token_fields = _observation_communication_token_fields(observation)
    if schema_version == OBSERVATION_SCHEMA_VERSION and token_fields:
        raise ValueError(
            "mind_observation_v3 cannot contain communication token channels"
        )
    if (
        schema_version == TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION
        and not token_fields
    ):
        raise ValueError(
            f"{TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION} requires "
            "communication token channels"
        )
    expected_size = (
        OBSERVATION_INPUT_VECTOR_SIZE
        + len(token_fields) * (1 + PATCH_CELL_COUNT)
    )
    values = _observation_input_values(
        observation,
        communication_token_fields=token_fields,
    )
    if len(values) != expected_size:
        raise ValueError(
            f"encoded observation has {len(values)} values; expected "
            f"{expected_size}"
        )
    packed = _pack_quantized_values(values)
    data = base64.b64encode(zlib.compress(packed, level=6)).decode("ascii")
    payload: dict[str, object] = {
        "schema_version": schema_version,
        "encoder_version": (
            TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION
            if token_fields
            else OBSERVATION_ENCODER_VERSION
        ),
        "decoded_dtype": OBSERVATION_INPUT_DTYPE,
        "storage_dtype": OBSERVATION_STORAGE_DTYPE,
        "storage_encoding": OBSERVATION_STORAGE_ENCODING,
        "shape": [expected_size],
        "value_range": list(OBSERVATION_INPUT_VALUE_RANGE),
        "data": data,
    }
    if token_fields:
        payload["communication_token_count"] = len(token_fields)
        payload["communication_token_field_order"] = list(token_fields)
    return payload


def decode_observation_input(payload: dict[str, object]) -> list[float]:
    _validate_observation_input_header(payload)
    shape = payload["shape"]
    expected_size = int(shape[0])  # type: ignore[index]
    data = payload.get("data")
    if not isinstance(data, str):
        raise ValueError("observation input data must be a base64 string")
    try:
        packed = zlib.decompress(base64.b64decode(data.encode("ascii")))
    except (ValueError, zlib.error) as exc:
        raise ValueError("observation input data is not valid compressed base64") from exc
    expected_bytes = expected_size * 2
    if len(packed) != expected_bytes:
        raise ValueError(
            f"observation input byte length {len(packed)} does not match "
            f"expected {expected_bytes}"
        )
    if expected_size == 0:
        return []
    values = [
        _round(float(value) / OBSERVATION_QUANTIZATION_SCALE)
        for value in struct.unpack(f"<{expected_size}h", packed)
    ]
    _validate_decoded_values(values)
    return values


def validate_observation_input_payload(payload: dict[str, object]) -> list[str]:
    try:
        decode_observation_input(payload)
    except ValueError as exc:
        return [str(exc)]
    return []


def _patch_cell(
    context: ObservationContext,
    agent: Agent,
    dx: int,
    dy: int,
) -> dict[str, object]:
    x = agent.x + dx
    y = agent.y + dy
    token_fields = _signal_state_communication_token_fields(context.signal_state)
    if not context.in_bounds(x, y):
        cell: dict[str, object] = {
            "dx": dx,
            "dy": dy,
            "in_bounds": False,
            "terrain": "out_of_bounds",
            "occupant": "none",
            "same_lineage": False,
            "water_access_reason": "none",
            "food": 0.0,
            "vegetation": 0.0,
            "recovery_debt": 0.0,
            "fresh_kill_energy": 0.0,
            "carcass_energy": 0.0,
            "hazard_type": "none",
            "hazard_level": 0.0,
            "ecology_state": "none",
            "prey_biomass": 0.0,
            "carrion_signal": 0.0,
            "predator_risk": 0.0,
            REPRODUCTIVE_SIGNAL_FIELD: 0.0,
            COMMUNICATION_SIGNAL_FIELD: 0.0,
        }
        cell.update({field_name: 0.0 for field_name in token_fields})
        return cell

    tile = context.grid[y][x]
    occupant = "none"
    same_lineage = False
    if tile.occupant_id is not None:
        occupant_agent = context.agents.get(tile.occupant_id)
        if occupant_agent is not None and occupant_agent.alive:
            same_lineage = occupant_agent.lineage_id == agent.lineage_id
            occupant = "self" if occupant_agent.agent_id == agent.agent_id else "agent"
    hazard_type, hazard_level = context.hazard_at(x, y)
    biotic_state = context.biotic_state
    signal_state = context.signal_state
    cell = {
        "dx": dx,
        "dy": dy,
        "in_bounds": True,
        "terrain": tile.terrain,
        "occupant": occupant,
        "same_lineage": same_lineage,
        "water_access_reason": context.water_access_reason(x, y),
        "food": _round(tile.food),
        "vegetation": _round(tile.vegetation),
        "recovery_debt": _round(tile.recovery_debt),
        "fresh_kill_energy": _round(tile.fresh_kill_energy),
        "carcass_energy": _round(tile.carcass_energy),
        "hazard_type": hazard_type,
        "hazard_level": _round(hazard_level),
        "ecology_state": (
            context.ecology_state_at(x, y) if tile.terrain != "water" else "none"
        ),
        "prey_biomass": _round(biotic_state.prey_biomass[y][x]),
        "carrion_signal": _round(biotic_state.carrion[y][x]),
        "predator_risk": _round(biotic_state.predator_risk[y][x]),
        REPRODUCTIVE_SIGNAL_FIELD: _round(signal_state.reproductive_signal[y][x]),
        COMMUNICATION_SIGNAL_FIELD: _round(
            _signal_value_for_observer(
                signal_state,
                COMMUNICATION_SIGNAL_FIELD,
                observer_agent_id=agent.agent_id,
                x=x,
                y=y,
            )
        ),
    }
    cell.update(
        {
            field_name: _round(
                _signal_value_for_observer(
                    signal_state,
                    field_name,
                    observer_agent_id=agent.agent_id,
                    x=x,
                    y=y,
                )
            )
            for field_name in token_fields
        }
    )
    return cell


def _signal_value_for_observer(
    signal_state: Any,
    field_name: str,
    *,
    observer_agent_id: int,
    x: int,
    y: int,
) -> float:
    receiver_projection = getattr(
        signal_state,
        "field_value_for_receiver",
        None,
    )
    if callable(receiver_projection):
        return float(
            receiver_projection(
                field_name,
                x=x,
                y=y,
                receiver_agent_id=observer_agent_id,
            )
        )
    return float(signal_state.field(field_name)[y][x])


def _signal_state_communication_token_fields(
    signal_state: Any,
) -> tuple[str, ...]:
    field_names = getattr(signal_state, "field_names", None)
    if not callable(field_names):
        return ()
    return tuple(
        name for name in field_names() if name not in SIGNAL_FIELD_NAMES
    )


def _navigation_targets(
    context: ObservationContext,
    agent: Agent,
) -> dict[str, dict[str, object]]:
    best: dict[str, tuple[float, int, int, int, float]] = {}
    carrion_signal_best: dict[str, tuple[float, int, int, int, float]] = {}
    prey_signal_best: dict[str, tuple[float, int, int, int, float]] = {}
    biotic_state = context.biotic_state
    for dy in range(-NAVIGATION_RADIUS, NAVIGATION_RADIUS + 1):
        span = NAVIGATION_RADIUS - abs(dy)
        for dx in range(-span, span + 1):
            distance = abs(dx) + abs(dy)
            x = agent.x + dx
            y = agent.y + dy
            if not context.in_bounds(x, y):
                continue
            tile = context.grid[y][x]
            if tile.terrain == "water":
                continue
            occupant = tile.occupant_id
            occupied_by_other = occupant is not None and occupant != agent.agent_id
            water_strength = (
                1.0 if context.water_access_reason(x, y) != "none" else 0.0
            )
            plant_strength = max(0.0, float(tile.food))
            carrion_resource_strength = max(0.0, float(tile.fresh_kill_energy)) + max(
                0.0,
                float(tile.carcass_energy),
            )
            carrion_signal_strength = max(0.0, float(biotic_state.carrion[y][x]))
            prey_strength = max(0.0, float(biotic_state.prey_biomass[y][x]))
            prey_resource_strength = _prey_resource_strength(context, agent, occupant)
            if water_strength > 0:
                _consider_navigation_target(
                    best,
                    "water",
                    dx,
                    dy,
                    distance,
                    water_strength,
                )
            if not occupied_by_other:
                _consider_navigation_target(best, "plant", dx, dy, distance, plant_strength)
            if carrion_resource_strength > 0:
                path_target = _navigation_first_step_to_tile(
                    context,
                    agent,
                    target_x=x,
                    target_y=y,
                    max_distance=NAVIGATION_RADIUS,
                )
                target_dx, target_dy, target_distance = (
                    path_target if path_target is not None else (dx, dy, distance)
                )
                _consider_navigation_target(
                    best,
                    "carrion",
                    target_dx,
                    target_dy,
                    target_distance,
                    carrion_resource_strength + carrion_signal_strength * 0.1,
                )
            elif not occupied_by_other:
                _consider_navigation_target(
                    carrion_signal_best,
                    "carrion",
                    dx,
                    dy,
                    distance,
                    carrion_signal_strength,
                )
            if prey_resource_strength > 0:
                _consider_navigation_target(
                    best,
                    "prey",
                    dx,
                    dy,
                    distance,
                    prey_resource_strength + prey_strength * 0.1,
                )
            else:
                _consider_navigation_target(
                    prey_signal_best,
                    "prey",
                    dx,
                    dy,
                    distance,
                    prey_strength,
                )
    return {
        target: _navigation_payload(
            best.get(target)
            or (carrion_signal_best.get(target) if target == "carrion" else None)
            or (prey_signal_best.get(target) if target == "prey" else None)
        )
        for target in NAVIGATION_TARGETS
    }


def _prey_resource_strength(
    context: ObservationContext,
    agent: Agent,
    occupant_id: int | None,
) -> float:
    if occupant_id is None or occupant_id == agent.agent_id:
        return 0.0
    target = context.agents.get(occupant_id)
    if target is None or not target.alive:
        return 0.0
    target_profile = context.profile_for(target)
    vulnerability = context.prey_vulnerability(target)
    if target_profile.role == "herbivore":
        return 1.0 + vulnerability * 0.45
    if target_profile.role == "omnivore":
        return max(0.0, vulnerability - 1.08) * 0.72
    return 0.0


def _navigation_first_step_to_tile(
    context: ObservationContext,
    agent: Agent,
    *,
    target_x: int,
    target_y: int,
    max_distance: int,
) -> tuple[int, int, int] | None:
    if target_x == agent.x and target_y == agent.y:
        return (0, 0, 0)

    visited = {(agent.x, agent.y)}
    frontier = deque([(agent.x, agent.y, 0, 0, 0)])
    while frontier:
        x, y, distance, first_dx, first_dy = frontier.popleft()
        if distance >= max_distance:
            continue
        for _, dx, dy in _ordered_movement_actions_toward(
            context,
            x,
            y,
            target_x,
            target_y,
        ):
            nx = x + dx
            ny = y + dy
            if (nx, ny) in visited:
                continue
            if not _navigation_path_tile_open(context, agent, nx, ny):
                continue
            step_dx = first_dx if distance > 0 else dx
            step_dy = first_dy if distance > 0 else dy
            next_distance = distance + 1
            if nx == target_x and ny == target_y:
                return (step_dx, step_dy, next_distance)
            visited.add((nx, ny))
            frontier.append((nx, ny, next_distance, step_dx, step_dy))
    return None


def _ordered_movement_actions_toward(
    context: ObservationContext,
    x: int,
    y: int,
    target_x: int,
    target_y: int,
) -> list[tuple[str, int, int]]:
    actions = list(context.movement_actions)
    return sorted(
        actions,
        key=lambda action: (
            abs(target_x - (x + action[1])) + abs(target_y - (y + action[2])),
            action[0],
        ),
    )


def _navigation_path_tile_open(
    context: ObservationContext,
    agent: Agent,
    x: int,
    y: int,
) -> bool:
    if not context.in_bounds(x, y):
        return False
    tile = context.grid[y][x]
    if tile.terrain == "water":
        return False
    return tile.occupant_id is None or tile.occupant_id == agent.agent_id


def _consider_navigation_target(
    best: dict[str, tuple[float, int, int, int, float]],
    target: str,
    dx: int,
    dy: int,
    distance: int,
    strength: float,
) -> None:
    if strength <= 1e-9:
        return
    score = strength - distance * 0.045
    current = best.get(target)
    if current is None or score > current[0]:
        best[target] = (score, dx, dy, distance, strength)


def _navigation_payload(
    item: tuple[float, int, int, int, float] | None,
) -> dict[str, object]:
    if item is None:
        return {"dx": 0, "dy": 0, "distance": 0, "strength": 0.0}
    _, dx, dy, distance, strength = item
    return {
        "dx": dx,
        "dy": dy,
        "distance": distance,
        "strength": _round(strength),
    }


def _round(value: float) -> float:
    return round(float(value), 4)


def _observation_input_values(
    observation: dict[str, object],
    *,
    communication_token_fields: tuple[str, ...] = (),
) -> list[float]:
    self_state = observation.get("self")
    patch = observation.get("local_patch")
    navigation = observation.get("navigation")
    if not isinstance(self_state, dict):
        raise ValueError("observation self section must be a mapping")
    if not isinstance(patch, list) or len(patch) != PATCH_CELL_COUNT:
        raise ValueError(
            f"observation local_patch must contain {PATCH_CELL_COUNT} cells"
        )
    if not isinstance(navigation, dict):
        raise ValueError("observation navigation section must be a mapping")
    values = _self_input_values(
        self_state,
        communication_token_fields=communication_token_fields,
    )
    for cell in patch:
        if not isinstance(cell, dict):
            raise ValueError("observation local_patch cells must be mappings")
        values.extend(
            _patch_input_values(
                cell,
                communication_token_fields=communication_token_fields,
            )
        )
    for target in NAVIGATION_TARGETS:
        payload = navigation.get(target)
        if not isinstance(payload, dict):
            raise ValueError(f"observation navigation target {target} must be a mapping")
        values.extend(_navigation_input_values(payload))
    _validate_decoded_values(values)
    return values


def _self_input_values(
    self_state: dict[str, object],
    *,
    communication_token_fields: tuple[str, ...] = (),
) -> list[float]:
    support_values = _hydrology_support_values(self_state["hydrology_support_code"])
    values = [
        _unit_value(self_state["energy_ratio"]),
        _unit_value(self_state["hydration_ratio"]),
        _unit_value(self_state["health_ratio"]),
        _unit_value(self_state["injury_load"]),
        _unit_value(self_state["age_norm"]),
        _bool_value(self_state["reproduction_ready"]),
        _unit_value(self_state["matched_diet_ratio"]),
        _enum_value(self_state["trophic_role"], TROPHIC_ROLE_VOCAB),
        _enum_value(self_state["meat_mode"], MEAT_MODE_VOCAB),
        _enum_value(self_state["season"], SEASON_VOCAB),
        _enum_value(self_state["water_access_reason"], WATER_ACCESS_REASON_VOCAB),
        *support_values,
        _unit_value(self_state["refuge_score"]),
        _enum_value(self_state["hazard_type"], HAZARD_TYPE_VOCAB),
        _unit_value(self_state["hazard_level"]),
        _unit_value(self_state["tile_vegetation"]),
        _unit_value(self_state["tile_recovery_debt"]),
        _enum_value(self_state["reproductive_stage"], REPRODUCTIVE_STAGE_VOCAB),
        _enum_value(
            self_state["reproductive_expression"],
            REPRODUCTIVE_EXPRESSION_VOCAB,
        ),
        _bool_value(self_state["sexual_reproduction_unlocked"]),
        _nonnegative_signal_value(self_state[REPRODUCTIVE_SIGNAL_FIELD]),
        _nonnegative_signal_value(self_state[COMMUNICATION_SIGNAL_FIELD]),
        _bool_value(self_state["mind_inheritance_available"]),
    ]
    values.extend(
        _nonnegative_signal_value(self_state[field_name])
        for field_name in communication_token_fields
    )
    return values


def _patch_input_values(
    cell: dict[str, object],
    *,
    communication_token_fields: tuple[str, ...] = (),
) -> list[float]:
    values = [
        _offset_value(cell["dx"]),
        _offset_value(cell["dy"]),
        _bool_value(cell["in_bounds"]),
        _enum_value(cell["terrain"], TERRAIN_VOCAB),
        _enum_value(cell["occupant"], OCCUPANT_VOCAB),
        _bool_value(cell["same_lineage"]),
        _enum_value(cell["water_access_reason"], WATER_ACCESS_REASON_VOCAB),
        _nonnegative_signal_value(cell["food"]),
        _unit_value(cell["vegetation"]),
        _unit_value(cell["recovery_debt"]),
        _nonnegative_signal_value(cell["fresh_kill_energy"]),
        _nonnegative_signal_value(cell["carcass_energy"]),
        _enum_value(cell["hazard_type"], HAZARD_TYPE_VOCAB),
        _unit_value(cell["hazard_level"]),
        _enum_value(cell["ecology_state"], ECOLOGY_STATE_VOCAB),
        _nonnegative_signal_value(cell["prey_biomass"]),
        _nonnegative_signal_value(cell["carrion_signal"]),
        _nonnegative_signal_value(cell["predator_risk"]),
        _nonnegative_signal_value(cell[REPRODUCTIVE_SIGNAL_FIELD]),
        _nonnegative_signal_value(cell[COMMUNICATION_SIGNAL_FIELD]),
    ]
    values.extend(
        _nonnegative_signal_value(cell[field_name])
        for field_name in communication_token_fields
    )
    return values


def _navigation_input_values(payload: dict[str, object]) -> list[float]:
    return [
        _signed_radius_value(payload["dx"], NAVIGATION_RADIUS),
        _signed_radius_value(payload["dy"], NAVIGATION_RADIUS),
        _unit_value(float(payload["distance"]) / max(NAVIGATION_RADIUS, 1)),
        _nonnegative_signal_value(payload["strength"]),
    ]


def _pack_quantized_values(values: list[float]) -> bytes:
    quantized: list[int] = []
    for value in values:
        if not isfinite(value):
            raise ValueError("observation input values must be finite")
        low, high = OBSERVATION_INPUT_VALUE_RANGE
        if value < low or value > high:
            raise ValueError(
                f"observation input value {value} is outside range [{low}, {high}]"
            )
        quantized.append(int(round(value * OBSERVATION_QUANTIZATION_SCALE)))
    return struct.pack(f"<{len(quantized)}h", *quantized)


def _validate_observation_input_header(payload: dict[str, object]) -> None:
    schema_version = payload.get("schema_version")
    if schema_version not in {
        OBSERVATION_SCHEMA_VERSION,
        TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION,
    }:
        raise ValueError("observation input schema_version is missing or stale")
    token_count = 0
    expected_encoder_version = OBSERVATION_ENCODER_VERSION
    if schema_version == TOKENIZED_COMMUNICATION_OBSERVATION_SCHEMA_VERSION:
        token_count_value = payload.get("communication_token_count")
        if (
            isinstance(token_count_value, bool)
            or not isinstance(token_count_value, int)
            or token_count_value <= 0
        ):
            raise ValueError(
                "observation input communication_token_count is missing or stale"
            )
        token_count = token_count_value
        expected_fields = [
            f"{COMMUNICATION_SIGNAL_FIELD}_token_{token_id}"
            for token_id in range(token_count)
        ]
        if payload.get("communication_token_field_order") != expected_fields:
            raise ValueError(
                "observation input communication_token_field_order is missing "
                "or stale"
            )
        expected_encoder_version = (
            TOKENIZED_COMMUNICATION_OBSERVATION_ENCODER_VERSION
        )
    elif (
        "communication_token_count" in payload
        or "communication_token_field_order" in payload
    ):
        raise ValueError(
            "mind_observation_v3 input cannot declare communication token channels"
        )
    if payload.get("encoder_version") != expected_encoder_version:
        raise ValueError("observation input encoder_version is missing or stale")
    if payload.get("decoded_dtype") != OBSERVATION_INPUT_DTYPE:
        raise ValueError("observation input decoded_dtype is missing or stale")
    if payload.get("storage_dtype") != OBSERVATION_STORAGE_DTYPE:
        raise ValueError("observation input storage_dtype is missing or stale")
    if payload.get("storage_encoding") != OBSERVATION_STORAGE_ENCODING:
        raise ValueError("observation input storage_encoding is missing or stale")
    if payload.get("value_range") != list(OBSERVATION_INPUT_VALUE_RANGE):
        raise ValueError("observation input value_range is missing or stale")
    shape = payload.get("shape")
    expected_size = (
        OBSERVATION_INPUT_VECTOR_SIZE
        + token_count * (1 + PATCH_CELL_COUNT)
    )
    if shape != [expected_size]:
        raise ValueError("observation input shape is missing or stale")


def _validate_decoded_values(values: list[float]) -> None:
    low, high = OBSERVATION_INPUT_VALUE_RANGE
    for value in values:
        if not isfinite(value):
            raise ValueError("observation input values must be finite")
        if value < low or value > high:
            raise ValueError(
                f"observation input value {value} is outside range [{low}, {high}]"
            )


def _unit_value(value: object) -> float:
    number = _finite_number(value)
    return _round(min(1.0, max(0.0, number)))


def _nonnegative_signal_value(value: object) -> float:
    number = max(0.0, _finite_number(value))
    return _round(number / (1.0 + number))


def _offset_value(value: object) -> float:
    if LOCAL_PATCH_RADIUS <= 0:
        return 0.0
    number = _finite_number(value)
    return _round(min(1.0, max(-1.0, number / LOCAL_PATCH_RADIUS)))


def _signed_radius_value(value: object, radius: int) -> float:
    if radius <= 0:
        return 0.0
    number = _finite_number(value)
    return _round(min(1.0, max(-1.0, number / radius)))


def _bool_value(value: object) -> float:
    return 1.0 if bool(value) else 0.0


def _enum_value(value: object, vocab: tuple[str, ...]) -> float:
    label = str(value)
    try:
        index = vocab.index(label)
    except ValueError as exc:
        raise ValueError(f"unknown observation enum value {label!r}") from exc
    if len(vocab) <= 1:
        return 0.0
    return _round(index / (len(vocab) - 1))


def _hydrology_support_values(value: object) -> list[float]:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("hydrology_support_code must be an integer")
    if value < 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        1.0,
        1.0 if value & 1 else 0.0,
        1.0 if value & 2 else 0.0,
        1.0 if value & 4 else 0.0,
    ]


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("observation input numeric values must be finite numbers")
    number = float(value)
    if not isfinite(number):
        raise ValueError("observation input numeric values must be finite numbers")
    return number
