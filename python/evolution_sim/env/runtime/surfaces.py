from __future__ import annotations

from typing import Any

from evolution_sim.env.runtime.state import Agent


HABITAT_STATE_CODES = {"stable": 0, "bloom": 1, "flooded": 2, "parched": 3}
ECOLOGY_STATE_CODES = {"stable": 0, "lush": 1, "recovering": 2, "depleted": 3}
HAZARD_TYPE_CODES = {"none": 0, "exposure": 1, "instability": 2}
NON_LAND_ECOLOGY_CODE = -1
HYDROLOGY_REASON_CODES = {"none": 0, "adjacent_water": 1, "wetland": 2, "flooded": 3}
HYDROLOGY_SUPPORT_FLAGS = {"adjacent_to_water": 1, "wetland": 2, "flooded": 4}
SOFT_REFUGE_CODES = {"none": 0, "canopy_refuge": 1}


def materialize_frame_surfaces(world: Any) -> dict[str, object]:
    habitat_states, habitat_counts = world._habitat_state_grid()
    (
        hydrology_primary_codes,
        hydrology_support_codes,
        hydrology_primary_counts,
        hydrology_support_counts,
        hydrology_primary_stats,
    ) = world._hydrology_snapshot()
    refuge_codes, refuge_score_codes, refuge_counts, refuge_stats = world._refuge_snapshot()
    ecology_codes, ecology_counts, ecology_stats = world._ecology_snapshot()
    hazard_type_codes, hazard_level_codes, hazard_counts, hazard_stats = world._hazard_snapshot()
    biotic_fields, biotic_field_stats = world._biotic_field_snapshot()
    fresh_kill_energy_codes, fresh_kill_stats = world._fresh_kill_snapshot()
    carcass_energy_codes, carcass_freshness_codes, carcass_stats = world._carcass_snapshot()

    hydrology_reason_by_code = {code: reason for reason, code in HYDROLOGY_REASON_CODES.items()}
    refuge_reason_by_code = {code: reason for reason, code in SOFT_REFUGE_CODES.items()}
    ecology_state_by_code = {code: state for state, code in ECOLOGY_STATE_CODES.items()}
    hazard_type_by_code = {code: hazard_type for hazard_type, code in HAZARD_TYPE_CODES.items()}

    return {
        "climate_state": world._climate_state(),
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
) -> dict[int, dict[str, object]]:
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
            "energy_ratio": world._energy_ratio(agent),
            "hydration_ratio": world._hydration_ratio(agent),
            "health_ratio": world._health_ratio(agent),
            "energy_modifier": world._agent_energy_drain_modifier(agent, season),
            "hydration_modifier": world._agent_hydration_drain_modifier(agent, season),
            "reproduction_ready": world._is_reproduction_ready(agent),
            "trophic_role": world._trophic_role(agent),
            "meat_mode": world._meat_mode(agent),
            "water_reason": water_reasons[agent.y][agent.x],
            "soft_refuge_reason": soft_refuge_reasons[agent.y][agent.x],
            "hydrology_support_code": support_code,
            "refuge_score": world._refuge_score(agent.x, agent.y),
            "matched_diet_ratio": world._matched_diet_ratio(agent),
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
