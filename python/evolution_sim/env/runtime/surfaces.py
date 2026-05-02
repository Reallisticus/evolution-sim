from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from evolution_sim.env.runtime.state import Agent


HABITAT_STATE_CODES = {"stable": 0, "bloom": 1, "flooded": 2, "parched": 3}
ECOLOGY_STATE_CODES = {"stable": 0, "lush": 1, "recovering": 2, "depleted": 3}
HAZARD_TYPE_CODES = {"none": 0, "exposure": 1, "instability": 2}
NON_LAND_ECOLOGY_CODE = -1
HYDROLOGY_REASON_CODES = {"none": 0, "adjacent_water": 1, "wetland": 2, "flooded": 3}
HYDROLOGY_SUPPORT_FLAGS = {"adjacent_to_water": 1, "wetland": 2, "flooded": 4}
SOFT_REFUGE_CODES = {"none": 0, "canopy_refuge": 1}


@dataclass(frozen=True, slots=True)
class FrameSurfaceContext:
    climate_state: dict[str, object]
    habitat_snapshot: tuple[list[list[str]], dict[str, int]]
    hydrology_snapshot: tuple[Any, Any, Any, Any, Any]
    refuge_snapshot: tuple[Any, Any, Any, Any]
    ecology_snapshot: tuple[Any, Any, Any]
    hazard_snapshot: tuple[Any, Any, Any, Any]
    biotic_field_snapshot: tuple[Any, Any]
    signal_field_snapshot: tuple[Any, Any]
    fresh_kill_snapshot: tuple[Any, Any]
    carcass_snapshot: tuple[Any, Any, Any]
    energy_ratio: Callable[[Agent], float]
    hydration_ratio: Callable[[Agent], float]
    health_ratio: Callable[[Agent], float]
    agent_energy_drain_modifier: Callable[[Agent, str], float]
    agent_hydration_drain_modifier: Callable[[Agent, str], float]
    is_reproduction_ready: Callable[[Agent], bool]
    trophic_role: Callable[[Agent], str]
    meat_mode: Callable[[Agent], str]
    refuge_score: Callable[[int, int], float]
    matched_diet_ratio: Callable[[Agent], float]


def _resolve_surface_context(
    world: Any,
    *,
    surface_context: FrameSurfaceContext | None = None,
) -> FrameSurfaceContext:
    if surface_context is not None:
        return surface_context
    return world._frame_surface_context()


def materialize_frame_surfaces(
    world: Any,
    *,
    surface_context: FrameSurfaceContext | None = None,
) -> dict[str, object]:
    context = _resolve_surface_context(world, surface_context=surface_context)
    habitat_states, habitat_counts = context.habitat_snapshot
    (
        hydrology_primary_codes,
        hydrology_support_codes,
        hydrology_primary_counts,
        hydrology_support_counts,
        hydrology_primary_stats,
    ) = context.hydrology_snapshot
    refuge_codes, refuge_score_codes, refuge_counts, refuge_stats = context.refuge_snapshot
    ecology_codes, ecology_counts, ecology_stats = context.ecology_snapshot
    hazard_type_codes, hazard_level_codes, hazard_counts, hazard_stats = (
        context.hazard_snapshot
    )
    biotic_fields, biotic_field_stats = context.biotic_field_snapshot
    signal_fields, signal_field_stats = context.signal_field_snapshot
    fresh_kill_energy_codes, fresh_kill_stats = context.fresh_kill_snapshot
    carcass_energy_codes, carcass_freshness_codes, carcass_stats = (
        context.carcass_snapshot
    )

    hydrology_reason_by_code = {code: reason for reason, code in HYDROLOGY_REASON_CODES.items()}
    refuge_reason_by_code = {code: reason for reason, code in SOFT_REFUGE_CODES.items()}
    ecology_state_by_code = {code: state for state, code in ECOLOGY_STATE_CODES.items()}
    hazard_type_by_code = {code: hazard_type for hazard_type, code in HAZARD_TYPE_CODES.items()}

    return {
        "climate_state": context.climate_state,
        "habitat_states": habitat_states,
        "habitat_codes": [
            [HABITAT_STATE_CODES[state] for state in row] for row in habitat_states
        ],
        "habitat_counts": habitat_counts,
        "hydrology_primary_codes": hydrology_primary_codes,
        "hydrology_support_codes": hydrology_support_codes,
        "hydrology_primary_counts": hydrology_primary_counts,
        "hydrology_support_counts": hydrology_support_counts,
        "hydrology_primary_stats": hydrology_primary_stats,
        "water_reasons": [
            [
                hydrology_reason_by_code.get(code, "none")
                if code != NON_LAND_ECOLOGY_CODE
                else "none"
                for code in row
            ]
            for row in hydrology_primary_codes
        ],
        "refuge_codes": refuge_codes,
        "refuge_score_codes": refuge_score_codes,
        "refuge_counts": refuge_counts,
        "refuge_stats": refuge_stats,
        "soft_refuge_reasons": [
            [
                refuge_reason_by_code.get(code, "none")
                if code != NON_LAND_ECOLOGY_CODE
                else "none"
                for code in row
            ]
            for row in refuge_codes
        ],
        "ecology_codes": ecology_codes,
        "ecology_counts": ecology_counts,
        "ecology_stats": ecology_stats,
        "ecology_states": [
            [
                ecology_state_by_code.get(code, "stable")
                if code != NON_LAND_ECOLOGY_CODE
                else "stable"
                for code in row
            ]
            for row in ecology_codes
        ],
        "hazard_type_codes": hazard_type_codes,
        "hazard_level_codes": hazard_level_codes,
        "hazard_counts": hazard_counts,
        "hazard_stats": hazard_stats,
        "hazard_types": [
            [
                hazard_type_by_code.get(code, "none")
                if code != NON_LAND_ECOLOGY_CODE
                else "none"
                for code in row
            ]
            for row in hazard_type_codes
        ],
        "biotic_fields": biotic_fields,
        "biotic_field_stats": biotic_field_stats,
        "signal_fields": signal_fields,
        "signal_field_stats": signal_field_stats,
        "fresh_kill_energy_codes": fresh_kill_energy_codes,
        "fresh_kill_stats": fresh_kill_stats,
        "carcass_energy_codes": carcass_energy_codes,
        "carcass_freshness_codes": carcass_freshness_codes,
        "carcass_stats": carcass_stats,
    }


def build_agent_frame_telemetry(
    world: Any,
    alive: list[Agent],
    *,
    season: str,
    surfaces: dict[str, object],
    surface_context: FrameSurfaceContext | None = None,
) -> dict[int, dict[str, object]]:
    context = _resolve_surface_context(world, surface_context=surface_context)
    telemetry: dict[int, dict[str, object]] = {}
    water_reasons = surfaces["water_reasons"]
    soft_refuge_reasons = surfaces["soft_refuge_reasons"]
    ecology_states = surfaces["ecology_states"]
    habitat_states = surfaces["habitat_states"]
    hazard_types = surfaces["hazard_types"]
    hydrology_support_codes = surfaces["hydrology_support_codes"]
    for agent in alive:
        support_code = hydrology_support_codes[agent.y][agent.x]
        telemetry[agent.agent_id] = {
            "energy_ratio": context.energy_ratio(agent),
            "hydration_ratio": context.hydration_ratio(agent),
            "health_ratio": context.health_ratio(agent),
            "energy_modifier": context.agent_energy_drain_modifier(agent, season),
            "hydration_modifier": context.agent_hydration_drain_modifier(agent, season),
            "reproduction_ready": context.is_reproduction_ready(agent),
            "trophic_role": context.trophic_role(agent),
            "meat_mode": context.meat_mode(agent),
            "water_reason": water_reasons[agent.y][agent.x],
            "soft_refuge_reason": soft_refuge_reasons[agent.y][agent.x],
            "hydrology_support_code": support_code,
            "refuge_score": context.refuge_score(agent.x, agent.y),
            "matched_diet_ratio": context.matched_diet_ratio(agent),
            "habitat_state": habitat_states[agent.y][agent.x],
            "ecology_state": ecology_states[agent.y][agent.x],
            "hazard_type": hazard_types[agent.y][agent.x],
            "has_water_access": water_reasons[agent.y][agent.x] != "none",
            "shoreline_support": bool(
                support_code & HYDROLOGY_SUPPORT_FLAGS["adjacent_to_water"]
            ),
            "wetland_support": bool(support_code & HYDROLOGY_SUPPORT_FLAGS["wetland"]),
            "flooded_support": bool(support_code & HYDROLOGY_SUPPORT_FLAGS["flooded"]),
        }
    return telemetry
