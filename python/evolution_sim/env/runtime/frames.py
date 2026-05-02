from __future__ import annotations

from typing import Any

import evolution_sim.env.runtime.reproduction as runtime_reproduction
import evolution_sim.env.runtime.signals as runtime_signals
from evolution_sim.env.runtime.reporting import (
    finalize_diet_totals,
    finalize_grouped_diet_totals,
)


def _combat_stats(world: Any) -> dict[str, object]:
    return {
        "attack_attempts": len(world.tick_attack_events),
        "successful_attacks": sum(
            1 for event in world.tick_attack_events if event["success"]
        ),
        "kills": sum(1 for event in world.tick_attack_events if event["kill"]),
        "damage_dealt": round(
            sum(
                float(event["damage"])
                for event in world.tick_attack_events
                if event["success"]
            ),
            4,
        ),
        "attack_damage_taken": round(
            sum(
                float(event["amount"])
                for event in world.tick_damage_events
                if event["source"] == "attack"
            ),
            4,
        ),
        "hazard_damage_taken": round(
            sum(
                float(event["amount"])
                for event in world.tick_damage_events
                if str(event["source"]).startswith("hazard_")
            ),
            4,
        ),
        "fresh_kill_consumption_events": len(world.tick_fresh_kill_events),
        "fresh_kill_energy_consumed": round(
            sum(float(event["energy"]) for event in world.tick_fresh_kill_events),
            4,
        ),
        "fresh_kill_gained_energy": round(
            sum(float(event["gained_energy"]) for event in world.tick_fresh_kill_events),
            4,
        ),
        "carcass_consumption_events": len(world.tick_carcass_events),
        "carcass_energy_consumed": round(
            sum(float(event["energy"]) for event in world.tick_carcass_events),
            4,
        ),
        "carcass_gained_energy": round(
            sum(float(event["gained_energy"]) for event in world.tick_carcass_events),
            4,
        ),
    }


def _fresh_kill_flow(world: Any) -> dict[str, object]:
    return {
        "deposition_events": len(world.tick_fresh_kill_deposit_events),
        "fresh_kill_energy_deposited": round(
            sum(
                float(event["deposited_energy"])
                for event in world.tick_fresh_kill_deposit_events
            ),
            4,
        ),
        "fresh_kill_energy_converted_to_carcass": round(
            world.tick_fresh_kill_to_carcass_energy,
            4,
        ),
        "consumption_events": len(world.tick_fresh_kill_events),
        "fresh_kill_energy_consumed": round(
            sum(float(event["energy"]) for event in world.tick_fresh_kill_events),
            4,
        ),
        "fresh_kill_gained_energy": round(
            sum(float(event["gained_energy"]) for event in world.tick_fresh_kill_events),
            4,
        ),
    }


def _carcass_flow(world: Any) -> dict[str, object]:
    return {
        "deposition_events": len(world.tick_carcass_deposit_events),
        "carcass_energy_deposited": round(
            sum(
                float(event["deposited_energy"])
                for event in world.tick_carcass_deposit_events
            ),
            4,
        ),
        "carcass_energy_decayed": round(world.tick_carcass_energy_decayed, 4),
        "consumption_events": len(world.tick_carcass_events),
        "carcass_energy_consumed": round(
            sum(float(event["energy"]) for event in world.tick_carcass_events),
            4,
        ),
        "carcass_gained_energy": round(
            sum(float(event["gained_energy"]) for event in world.tick_carcass_events),
            4,
        ),
    }


def _empty_diet_totals() -> dict[str, float]:
    return {
        "plant_events": 0,
        "plant_energy": 0.0,
        "fresh_kill_events": 0,
        "fresh_kill_energy": 0.0,
        "carcass_events": 0,
        "carcass_energy": 0.0,
    }


def _empty_grouped_diet_totals(
    groups: list[str] | dict[str, int],
) -> dict[str, dict[str, float]]:
    return {group: _empty_diet_totals() for group in groups}


def _accumulate_diet_totals(
    totals: dict[str, float],
    food_source: str,
    gained_energy: float,
) -> None:
    if food_source not in {"plant", "fresh_kill", "carcass"}:
        raise ValueError(f"Unsupported food source: {food_source}")
    totals[f"{food_source}_events"] += 1
    totals[f"{food_source}_energy"] += gained_energy


def _diet_stats(
    world: Any,
    *,
    trophic_role_codes: dict[str, int],
    meat_mode_codes: dict[str, int],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    diet_totals = _empty_diet_totals()
    diet_by_trophic_role = _empty_grouped_diet_totals(
        [role for role in trophic_role_codes if role != "none"]
    )
    diet_by_meat_mode = _empty_grouped_diet_totals(meat_mode_codes)
    for event in world.tick_feeding_events:
        food_source = str(event["food_source"])
        gained_energy = float(event["gained_energy"])
        trophic_role = str(event["trophic_role"])
        meat_mode = str(event["meat_mode"])
        _accumulate_diet_totals(diet_totals, food_source, gained_energy)
        _accumulate_diet_totals(
            diet_by_trophic_role[trophic_role],
            food_source,
            gained_energy,
        )
        _accumulate_diet_totals(
            diet_by_meat_mode[meat_mode],
            food_source,
            gained_energy,
        )
    return (
        finalize_diet_totals(diet_totals),
        finalize_grouped_diet_totals(diet_by_trophic_role),
        finalize_grouped_diet_totals(diet_by_meat_mode),
    )


def _agent_rows(
    world: Any,
    alive: list[Any],
    *,
    agent_telemetry: dict[int, dict[str, object]],
    species_map: dict[int, int],
    ecotype_map: dict[int, int],
) -> list[list[object]]:
    return [
        [
            agent.agent_id,
            agent.x,
            agent.y,
            round(agent.energy, 4),
            round(float(agent_telemetry[agent.agent_id]["energy_ratio"]), 4),
            round(agent.hydration, 4),
            round(float(agent_telemetry[agent.agent_id]["hydration_ratio"]), 4),
            round(agent.health, 4),
            round(float(agent_telemetry[agent.agent_id]["health_ratio"]), 4),
            round(agent.injury_load, 4),
            agent.age,
            round(float(agent_telemetry[agent.agent_id]["energy_modifier"]), 4),
            round(float(agent_telemetry[agent.agent_id]["hydration_modifier"]), 4),
            round(world.grid[agent.y][agent.x].vegetation, 4),
            round(world.grid[agent.y][agent.x].recovery_debt, 4),
            int(agent_telemetry[agent.agent_id]["reproduction_ready"]),
            agent_telemetry[agent.agent_id]["trophic_role"],
            agent_telemetry[agent.agent_id]["meat_mode"],
            agent.last_damage_source,
            agent_telemetry[agent.agent_id]["water_reason"],
            agent_telemetry[agent.agent_id]["soft_refuge_reason"],
            agent_telemetry[agent.agent_id]["hydrology_support_code"],
            round(float(agent_telemetry[agent.agent_id]["refuge_score"]), 4),
            round(float(agent_telemetry[agent.agent_id]["matched_diet_ratio"]), 4),
            ecotype_map.get(agent.agent_id, 0),
            species_map.get(agent.agent_id, 0),
        ]
        for agent in alive
    ]


def _population_trophic_counts_from_telemetry(
    alive: list[Any],
    agent_telemetry: dict[int, dict[str, object]],
    *,
    trophic_role_codes: dict[str, int],
    meat_mode_codes: dict[str, int],
) -> tuple[dict[str, int], dict[str, int]]:
    trophic_role_counts = {
        role: 0 for role in trophic_role_codes if role != "none"
    }
    meat_mode_counts = {mode: 0 for mode in meat_mode_codes}
    for agent in alive:
        telemetry = agent_telemetry[agent.agent_id]
        trophic_role_counts[str(telemetry["trophic_role"])] += 1
        meat_mode_counts[str(telemetry["meat_mode"])] += 1
    return trophic_role_counts, meat_mode_counts


def _agent_category_grid(
    alive: list[Any],
    agent_telemetry: dict[int, dict[str, object]],
    *,
    width: int,
    height: int,
    telemetry_field: str,
    codes: dict[str, int],
) -> list[list[int]]:
    grid = [[codes["none"] for _ in range(width)] for _ in range(height)]
    for agent in alive:
        telemetry_value = str(agent_telemetry[agent.agent_id][telemetry_field])
        grid[agent.y][agent.x] = codes[telemetry_value]
    return grid


def build_frame_payload(
    *,
    tick: int,
    season: str,
    surfaces: dict[str, object],
    signal_flow: dict[str, object],
    signal_emissions: dict[str, object],
    fresh_kill_patches: list[dict[str, object]],
    fresh_kill_flow: dict[str, object],
    carcass_patches: list[dict[str, object]],
    carcass_flow: dict[str, object],
    combat_stats: dict[str, object],
    diet_stats: dict[str, object],
    diet_by_trophic_role: dict[str, object],
    diet_by_meat_mode: dict[str, object],
    trophic_role_counts: dict[str, int],
    meat_mode_counts: dict[str, int],
    reproduction_stats: dict[str, object],
    trophic_role_grid: list[list[int]],
    meat_mode_grid: list[list[int]],
    alive_count: int,
    births_this_tick: int,
    deaths_this_tick: int,
    trait_means: dict[str, float],
    species_metrics: dict[str, dict[str, object]],
    ecotype_metrics: dict[str, dict[str, object]],
    species_counts: list[list[int]],
    ecotype_counts: list[list[int]],
    agent_rows: list[list[object]],
) -> dict[str, object]:
    return {
        "tick": tick,
        "season": season,
        "field_state": surfaces["climate_state"],
        "biotic_fields": surfaces["biotic_fields"],
        "biotic_field_stats": surfaces["biotic_field_stats"],
        "signal_fields": surfaces["signal_fields"],
        "signal_field_stats": surfaces["signal_field_stats"],
        "signal_flow": signal_flow,
        "signal_emissions": signal_emissions,
        "habitat_state_counts": surfaces["habitat_counts"],
        "habitat_state_codes": surfaces["habitat_codes"],
        "hydrology_primary_counts": surfaces["hydrology_primary_counts"],
        "hydrology_primary_codes": surfaces["hydrology_primary_codes"],
        "hydrology_primary_stats": surfaces["hydrology_primary_stats"],
        "hydrology_support_counts": surfaces["hydrology_support_counts"],
        "hydrology_support_codes": surfaces["hydrology_support_codes"],
        "refuge_counts": surfaces["refuge_counts"],
        "refuge_codes": surfaces["refuge_codes"],
        "refuge_score_codes": surfaces["refuge_score_codes"],
        "refuge_stats": surfaces["refuge_stats"],
        "hazard_counts": surfaces["hazard_counts"],
        "hazard_type_codes": surfaces["hazard_type_codes"],
        "hazard_level_codes": surfaces["hazard_level_codes"],
        "hazard_stats": surfaces["hazard_stats"],
        "fresh_kill_energy_codes": surfaces["fresh_kill_energy_codes"],
        "fresh_kill_stats": surfaces["fresh_kill_stats"],
        "fresh_kill_patches": fresh_kill_patches,
        "fresh_kill_flow": fresh_kill_flow,
        "carcass_energy_codes": surfaces["carcass_energy_codes"],
        "carcass_freshness_codes": surfaces["carcass_freshness_codes"],
        "carcass_stats": surfaces["carcass_stats"],
        "carcass_patches": carcass_patches,
        "carcass_flow": carcass_flow,
        "combat_stats": combat_stats,
        "diet_stats": diet_stats,
        "diet_by_trophic_role": diet_by_trophic_role,
        "diet_by_meat_mode": diet_by_meat_mode,
        "trophic_role_counts": trophic_role_counts,
        "meat_mode_counts": meat_mode_counts,
        "reproduction_stats": reproduction_stats,
        "trophic_role_codes": trophic_role_grid,
        "meat_mode_codes": meat_mode_grid,
        "ecology_state_counts": surfaces["ecology_counts"],
        "ecology_state_codes": surfaces["ecology_codes"],
        "ecology_stats": surfaces["ecology_stats"],
        "alive_agents": alive_count,
        "births": births_this_tick,
        "deaths": deaths_this_tick,
        "trait_means": trait_means,
        "species_metrics": species_metrics,
        "ecotype_metrics": ecotype_metrics,
        "species_counts": species_counts,
        "ecotype_counts": ecotype_counts,
        "agents": agent_rows,
    }


def capture_frame(
    world: Any,
    *,
    births_this_tick: int,
    deaths_this_tick: int,
    trophic_role_codes: dict[str, int],
    meat_mode_codes: dict[str, int],
) -> None:
    (
        alive,
        species_map,
        species_records,
        ecotype_map,
        ecotype_records,
    ) = world._refresh_population_snapshots()
    trait_means = world._trait_means(alive)
    surfaces = world._materialize_frame_surfaces()
    season = world._season_state()["name"]
    agent_telemetry = world._build_agent_frame_telemetry(
        alive,
        season=season,
        surfaces=surfaces,
    )
    species_metrics = world._build_species_metrics(
        alive,
        species_map,
        world.agent_last_species_map,
        agent_telemetry=agent_telemetry,
    )
    ecotype_metrics = world._build_species_metrics(
        alive,
        ecotype_map,
        world.agent_last_ecotype_map,
        agent_telemetry=agent_telemetry,
    )
    fresh_kill_patches = world._fresh_kill_patch_summaries()
    carcass_patches = world._carcass_patch_summaries()
    combat_stats = _combat_stats(world)
    fresh_kill_flow = _fresh_kill_flow(world)
    carcass_flow = _carcass_flow(world)
    diet_stats, frame_diet_by_trophic_role, frame_diet_by_meat_mode = _diet_stats(
        world,
        trophic_role_codes=trophic_role_codes,
        meat_mode_codes=meat_mode_codes,
    )
    trophic_role_counts, meat_mode_counts = _population_trophic_counts_from_telemetry(
        alive,
        agent_telemetry,
        trophic_role_codes=trophic_role_codes,
        meat_mode_codes=meat_mode_codes,
    )
    reproduction_stats = runtime_reproduction.build_frame_reproduction_stats(
        world,
        alive,
        trophic_role_codes=trophic_role_codes,
        meat_mode_codes=meat_mode_codes,
    )
    world.run_fresh_kill_totals["fresh_kill_tiles"] = surfaces["fresh_kill_stats"][
        "fresh_kill_tiles"
    ]
    world.run_fresh_kill_totals["total_fresh_kill_energy"] = surfaces[
        "fresh_kill_stats"
    ]["total_fresh_kill_energy"]
    world.run_carcass_totals["carcass_tiles"] = surfaces["carcass_stats"]["carcass_tiles"]
    world.run_carcass_totals["total_carcass_energy"] = surfaces["carcass_stats"][
        "total_carcass_energy"
    ]
    frame = build_frame_payload(
        tick=world.tick,
        season=season,
        surfaces=surfaces,
        signal_flow=runtime_signals.finalize_signal_totals(world.tick_signal_totals),
        signal_emissions=runtime_signals.signal_emission_debug_snapshot(world),
        fresh_kill_patches=fresh_kill_patches,
        fresh_kill_flow=fresh_kill_flow,
        carcass_patches=carcass_patches,
        carcass_flow=carcass_flow,
        combat_stats=combat_stats,
        diet_stats=diet_stats,
        diet_by_trophic_role=frame_diet_by_trophic_role,
        diet_by_meat_mode=frame_diet_by_meat_mode,
        trophic_role_counts=trophic_role_counts,
        meat_mode_counts=meat_mode_counts,
        reproduction_stats=reproduction_stats,
        trophic_role_grid=_agent_category_grid(
            alive,
            agent_telemetry,
            width=world.config.width,
            height=world.config.height,
            telemetry_field="trophic_role",
            codes=trophic_role_codes,
        ),
        meat_mode_grid=_agent_category_grid(
            alive,
            agent_telemetry,
            width=world.config.width,
            height=world.config.height,
            telemetry_field="meat_mode",
            codes=meat_mode_codes,
        ),
        alive_count=len(alive),
        births_this_tick=births_this_tick,
        deaths_this_tick=deaths_this_tick,
        trait_means=trait_means,
        species_metrics=species_metrics,
        ecotype_metrics=ecotype_metrics,
        species_counts=[
            [record.species_id, record.member_count] for record in species_records
        ],
        ecotype_counts=[
            [record.species_id, record.member_count] for record in ecotype_records
        ],
        agent_rows=_agent_rows(
            world,
            alive,
            agent_telemetry=agent_telemetry,
            species_map=species_map,
            ecotype_map=ecotype_map,
        ),
    )
    world.viewer_frames.append(frame)
    world.agent_last_species_map = species_map.copy()
    world.agent_last_ecotype_map = ecotype_map.copy()
