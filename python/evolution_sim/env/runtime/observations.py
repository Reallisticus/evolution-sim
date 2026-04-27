from __future__ import annotations

import hashlib
import json
from typing import Any

from evolution_sim.env.runtime.action_space import ACTION_NAMES, build_action_mask
from evolution_sim.env.runtime.state import Agent

OBSERVATION_SCHEMA_VERSION = "mind_observation_v1"
LOCAL_PATCH_RADIUS = 2
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
)


def observation_contract() -> dict[str, object]:
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "local_patch_radius": LOCAL_PATCH_RADIUS,
        "self_fields": list(SELF_FIELDS),
        "patch_fields": list(PATCH_FIELDS),
        "action_names": list(ACTION_NAMES),
        "privileged_world_state": False,
    }


def build_observation(world: Any, agent: Agent) -> dict[str, object]:
    climate_state = world._climate_state()
    profile = world._trophic_profile(agent)
    tile = world.grid[agent.y][agent.x]
    hazard_type, hazard_level = world._hazard_at(agent.x, agent.y)
    biotic_state = world._current_biotic_state()
    return {
        "schema_version": OBSERVATION_SCHEMA_VERSION,
        "agent_id": agent.agent_id,
        "self": {
            "energy_ratio": _round(world._energy_ratio(agent)),
            "hydration_ratio": _round(world._hydration_ratio(agent)),
            "health_ratio": _round(world._health_ratio(agent)),
            "injury_load": _round(agent.injury_load),
            "age_norm": _round(agent.age / max(world.config.max_age, 1)),
            "reproduction_ready": bool(world._is_reproduction_ready(agent)),
            "matched_diet_ratio": _round(world._matched_diet_ratio(agent, profile)),
            "trophic_role": profile.role,
            "meat_mode": profile.meat_mode,
            "season": str(climate_state["season"]),
            "water_access_reason": world._water_access_reason(agent.x, agent.y),
            "hydrology_support_code": world._hydrology_support_code(agent.x, agent.y),
            "refuge_score": _round(world._refuge_score(agent.x, agent.y)),
            "hazard_type": hazard_type,
            "hazard_level": _round(hazard_level),
            "tile_vegetation": _round(tile.vegetation),
            "tile_recovery_debt": _round(tile.recovery_debt),
        },
        "local_patch": [
            _patch_cell(world, agent, dx, dy, biotic_state)
            for dy in range(-LOCAL_PATCH_RADIUS, LOCAL_PATCH_RADIUS + 1)
            for dx in range(-LOCAL_PATCH_RADIUS, LOCAL_PATCH_RADIUS + 1)
        ],
        "action_mask": build_action_mask(world, agent),
    }


def observation_digest(observation: dict[str, object]) -> str:
    payload = json.dumps(observation, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _patch_cell(
    world: Any,
    agent: Agent,
    dx: int,
    dy: int,
    biotic_state: Any,
) -> dict[str, object]:
    x = agent.x + dx
    y = agent.y + dy
    if not world._in_bounds(x, y):
        return {
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
        }

    tile = world.grid[y][x]
    occupant = "none"
    same_lineage = False
    if tile.occupant_id is not None:
        occupant_agent = world.agents.get(tile.occupant_id)
        if occupant_agent is not None and occupant_agent.alive:
            same_lineage = occupant_agent.lineage_id == agent.lineage_id
            occupant = "self" if occupant_agent.agent_id == agent.agent_id else "agent"
    hazard_type, hazard_level = world._hazard_at(x, y)
    return {
        "dx": dx,
        "dy": dy,
        "in_bounds": True,
        "terrain": tile.terrain,
        "occupant": occupant,
        "same_lineage": same_lineage,
        "water_access_reason": world._water_access_reason(x, y),
        "food": _round(tile.food),
        "vegetation": _round(tile.vegetation),
        "recovery_debt": _round(tile.recovery_debt),
        "fresh_kill_energy": _round(tile.fresh_kill_energy),
        "carcass_energy": _round(tile.carcass_energy),
        "hazard_type": hazard_type,
        "hazard_level": _round(hazard_level),
        "ecology_state": world._ecology_state_at(x, y) if tile.terrain != "water" else "none",
        "prey_biomass": _round(biotic_state.prey_biomass[y][x]),
        "carrion_signal": _round(biotic_state.carrion[y][x]),
        "predator_risk": _round(biotic_state.predator_risk[y][x]),
    }


def _round(value: float) -> float:
    return round(float(value), 4)
