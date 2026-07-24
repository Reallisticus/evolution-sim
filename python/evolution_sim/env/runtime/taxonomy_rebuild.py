from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from evolution_sim.env.runtime.leaderboards import (
    build_species_metric_leaderboards,
    build_taxonomy_top_species,
)
from evolution_sim.env.runtime.reporting_contracts import (
    BIOTIC_FIELD_NAMES,
    DIET_SERIES_METRICS,
    ECOLOGY_STATES,
    HABITAT_STATES,
    HAZARD_TYPES,
    HYDROLOGY_REASONS,
    MEAT_MODE_SERIES,
    REFUGE_REASONS,
    SIGNAL_FIELD_NAMES,
    TROPHIC_ROLES,
)


def build_species_population_series(
    *,
    ticks: list[int],
    frames: list[dict[str, object]],
    species_ids: list[str],
) -> dict[str, list[int]]:
    species_population: dict[str, list[int]] = {
        species_id: [0 for _ in ticks]
        for species_id in species_ids
    }
    for frame_index, frame in enumerate(frames):
        frame_counts = {str(species_id): count for species_id, count in frame["species_counts"]}
        for species_id in species_population:
            species_population[species_id][frame_index] = frame_counts.get(species_id, 0)
    return species_population


def build_collapse_events(
    ticks: list[int],
    species_population: dict[str, list[int]],
    *,
    speciation_events: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    split_source_species_by_tick: dict[int, set[int]] = defaultdict(set)
    for event in speciation_events or []:
        split_source_species_by_tick[int(event["tick"])].add(int(event["source_species_id"]))
    for species_id, counts in species_population.items():
        peak = 0
        collapse_recorded = False
        previous = 0
        for frame_index, count in enumerate(counts):
            peak = max(peak, count)
            tick = ticks[frame_index]
            if not collapse_recorded and peak >= 12 and count > 0 and count <= int(peak * 0.4):
                events.append(
                    {
                        "tick": tick,
                        "species_id": int(species_id),
                        "type": "collapse",
                        "peak": peak,
                        "current": count,
                    }
                )
                collapse_recorded = True
            if previous > 0 and count == 0:
                if int(species_id) in split_source_species_by_tick.get(tick, set()):
                    previous = count
                    continue
                events.append(
                    {
                        "tick": tick,
                        "species_id": int(species_id),
                        "type": "extinction",
                        "peak": peak,
                        "current": count,
                    }
                )
            previous = count
    events.sort(key=lambda event: (event["tick"], event["species_id"], event["type"]))
    return events


def build_replay_analytics(
    *,
    frames: list[dict[str, object]],
    species_ids: Iterable[int],
) -> dict[str, object]:
    ticks = [frame["tick"] for frame in frames]
    population = {
        "alive_agents": [frame["alive_agents"] for frame in frames],
        "species_count": [len(frame["species_counts"]) for frame in frames],
        "ecotype_count": [len(frame["ecotype_counts"]) for frame in frames],
        "births": [frame["births"] for frame in frames],
        "deaths": [frame["deaths"] for frame in frames],
    }
    traits = {
        "avg_max_energy": [frame["trait_means"]["avg_max_energy"] for frame in frames],
        "avg_max_health": [frame["trait_means"]["avg_max_health"] for frame in frames],
        "avg_move_cost": [frame["trait_means"]["avg_move_cost"] for frame in frames],
        "avg_heat_tolerance": [
            frame["trait_means"]["avg_heat_tolerance"] for frame in frames
        ],
        "avg_food_efficiency": [
            frame["trait_means"]["avg_food_efficiency"] for frame in frames
        ],
        "avg_water_efficiency": [
            frame["trait_means"]["avg_water_efficiency"] for frame in frames
        ],
        "avg_attack_power": [
            frame["trait_means"]["avg_attack_power"] for frame in frames
        ],
        "avg_meat_efficiency": [
            frame["trait_means"]["avg_meat_efficiency"] for frame in frames
        ],
        "avg_carrion_bias": [
            frame["trait_means"]["avg_carrion_bias"] for frame in frames
        ],
        "avg_live_prey_bias": [
            frame["trait_means"]["avg_live_prey_bias"] for frame in frames
        ],
        "avg_wetland_affinity": [
            frame["trait_means"]["avg_wetland_affinity"] for frame in frames
        ],
        "avg_rocky_affinity": [
            frame["trait_means"]["avg_rocky_affinity"] for frame in frames
        ],
    }
    trophic_roles = {
        role: [frame["trophic_role_counts"].get(role, 0) for frame in frames]
        for role in TROPHIC_ROLES
    }
    meat_modes = {
        mode: [frame["meat_mode_counts"].get(mode, 0) for frame in frames]
        for mode in MEAT_MODE_SERIES
    }
    diet_by_trophic_role = {
        role: {
            metric: [frame["diet_by_trophic_role"][role][metric] for frame in frames]
            for metric in DIET_SERIES_METRICS
        }
        for role in TROPHIC_ROLES
    }
    diet_by_meat_mode = {
        mode: {
            metric: [frame["diet_by_meat_mode"][mode][metric] for frame in frames]
            for metric in DIET_SERIES_METRICS
        }
        for mode in MEAT_MODE_SERIES
    }
    species_population = build_species_population_series(
        ticks=ticks,
        frames=frames,
        species_ids=[str(species_id) for species_id in species_ids],
    )

    return {
        "ticks": ticks,
        "population": population,
        "traits": traits,
        "trophic_roles": trophic_roles,
        "meat_modes": meat_modes,
        "species_population": species_population,
        "collapse_events": build_collapse_events(ticks, species_population),
        "habitat": {
            state: [frame["habitat_state_counts"].get(state, 0) for frame in frames]
            for state in HABITAT_STATES
        },
        "hydrology_primary": {
            **{
                reason: [
                    frame["hydrology_primary_counts"].get(reason, 0) for frame in frames
                ]
                for reason in HYDROLOGY_REASONS
            },
            "hard_access_tiles": [
                frame["hydrology_primary_stats"]["hard_access_tiles"] for frame in frames
            ],
        },
        "hydrology_support": {
            "shoreline_support": [
                frame["hydrology_support_counts"].get("shoreline_support", 0)
                for frame in frames
            ],
            "wetland_support": [
                frame["hydrology_support_counts"].get("wetland_support", 0)
                for frame in frames
            ],
            "flooded_support": [
                frame["hydrology_support_counts"].get("flooded_support", 0)
                for frame in frames
            ],
        },
        "refuge": {
            **{
                reason: [frame["refuge_counts"].get(reason, 0) for frame in frames]
                for reason in REFUGE_REASONS
            },
            "canopy_refuge_tiles": [
                frame["refuge_counts"].get("canopy_refuge", 0) for frame in frames
            ],
            "avg_refuge_score_forest_tiles": [
                frame["refuge_stats"]["avg_refuge_score_forest_tiles"]
                for frame in frames
            ],
        },
        "ecology": {
            **{
                state: [frame["ecology_state_counts"].get(state, 0) for frame in frames]
                for state in ECOLOGY_STATES
            },
            "avg_vegetation": [
                frame["ecology_stats"]["avg_vegetation"] for frame in frames
            ],
            "avg_recovery_debt": [
                frame["ecology_stats"]["avg_recovery_debt"] for frame in frames
            ],
        },
        "hazards": {
            **{
                hazard_type: [
                    frame["hazard_counts"].get(hazard_type, 0) for frame in frames
                ]
                for hazard_type in HAZARD_TYPES
            },
            "hazardous_tiles": [
                frame["hazard_stats"]["hazardous_tiles"] for frame in frames
            ],
            "avg_hazard_level": [
                frame["hazard_stats"]["avg_hazard_level"] for frame in frames
            ],
        },
        "biotic_fields": {
            field_name: {
                metric: [
                    frame["biotic_field_stats"][field_name][metric] for frame in frames
                ]
                for metric in ("avg", "max")
            }
            for field_name in BIOTIC_FIELD_NAMES
        },
        "signal_fields": {
            field_name: {
                metric: [
                    frame["signal_field_stats"][field_name][metric] for frame in frames
                ]
                for metric in ("avg", "max", "active_tiles")
            }
            for field_name in _reported_signal_field_names(frames)
        },
        "signal_flow": {
            "reproductive_emissions": [
                frame["signal_flow"]["reproductive_emissions"] for frame in frames
            ],
            "communication_emissions": [
                frame["signal_flow"]["communication_emissions"] for frame in frames
            ],
            "energy_spent": [frame["signal_flow"]["energy_spent"] for frame in frames],
        },
        "fresh_kill": {
            "fresh_kill_tiles": [
                frame["fresh_kill_stats"]["fresh_kill_tiles"] for frame in frames
            ],
            "total_fresh_kill_energy": [
                frame["fresh_kill_stats"]["total_fresh_kill_energy"]
                for frame in frames
            ],
            "deposit_count": [
                frame["fresh_kill_stats"]["deposit_count"] for frame in frames
            ],
            "mixed_source_tiles": [
                frame["fresh_kill_stats"]["mixed_source_tiles"] for frame in frames
            ],
            "deposition_events": [
                frame["fresh_kill_flow"]["deposition_events"] for frame in frames
            ],
            "fresh_kill_energy_deposited": [
                frame["fresh_kill_flow"]["fresh_kill_energy_deposited"]
                for frame in frames
            ],
            "fresh_kill_energy_converted_to_carcass": [
                frame["fresh_kill_flow"]["fresh_kill_energy_converted_to_carcass"]
                for frame in frames
            ],
            "consumption_events": [
                frame["fresh_kill_flow"]["consumption_events"] for frame in frames
            ],
            "fresh_kill_energy_consumed": [
                frame["fresh_kill_flow"]["fresh_kill_energy_consumed"]
                for frame in frames
            ],
            "fresh_kill_gained_energy": [
                frame["fresh_kill_flow"]["fresh_kill_gained_energy"] for frame in frames
            ],
        },
        "carcasses": {
            "carcass_tiles": [
                frame["carcass_stats"]["carcass_tiles"] for frame in frames
            ],
            "total_carcass_energy": [
                frame["carcass_stats"]["total_carcass_energy"] for frame in frames
            ],
            "avg_carcass_freshness": [
                frame["carcass_stats"]["avg_carcass_freshness"] for frame in frames
            ],
            "deposit_count": [
                frame["carcass_stats"]["deposit_count"] for frame in frames
            ],
            "mixed_source_tiles": [
                frame["carcass_stats"]["mixed_source_tiles"] for frame in frames
            ],
            "deposition_events": [
                frame["carcass_flow"]["deposition_events"] for frame in frames
            ],
            "carcass_energy_deposited": [
                frame["carcass_flow"]["carcass_energy_deposited"] for frame in frames
            ],
            "carcass_energy_decayed": [
                frame["carcass_flow"]["carcass_energy_decayed"] for frame in frames
            ],
            "consumption_events": [
                frame["carcass_flow"]["consumption_events"] for frame in frames
            ],
            "carcass_energy_consumed": [
                frame["carcass_flow"]["carcass_energy_consumed"] for frame in frames
            ],
            "carcass_gained_energy": [
                frame["carcass_flow"]["carcass_gained_energy"] for frame in frames
            ],
        },
        "diet": {
            "plant_events": [frame["diet_stats"]["plant_events"] for frame in frames],
            "plant_energy": [frame["diet_stats"]["plant_energy"] for frame in frames],
            "fresh_kill_events": [
                frame["diet_stats"]["fresh_kill_events"] for frame in frames
            ],
            "fresh_kill_energy": [
                frame["diet_stats"]["fresh_kill_energy"] for frame in frames
            ],
            "carcass_events": [
                frame["diet_stats"]["carcass_events"] for frame in frames
            ],
            "carcass_energy": [
                frame["diet_stats"]["carcass_energy"] for frame in frames
            ],
            "animal_events": [
                frame["diet_stats"]["animal_events"] for frame in frames
            ],
            "animal_energy": [
                frame["diet_stats"]["animal_energy"] for frame in frames
            ],
            "plant_energy_share": [
                frame["diet_stats"]["plant_energy_share"] for frame in frames
            ],
            "animal_energy_share": [
                frame["diet_stats"]["animal_energy_share"] for frame in frames
            ],
            "fresh_kill_energy_share": [
                frame["diet_stats"]["fresh_kill_energy_share"] for frame in frames
            ],
            "carcass_energy_share": [
                frame["diet_stats"]["carcass_energy_share"] for frame in frames
            ],
        },
        "diet_by_trophic_role": diet_by_trophic_role,
        "diet_by_meat_mode": diet_by_meat_mode,
        "combat": {
            "attack_attempts": [
                frame["combat_stats"]["attack_attempts"] for frame in frames
            ],
            "successful_attacks": [
                frame["combat_stats"]["successful_attacks"] for frame in frames
            ],
            "kills": [frame["combat_stats"]["kills"] for frame in frames],
            "damage_dealt": [
                frame["combat_stats"]["damage_dealt"] for frame in frames
            ],
            "attack_damage_taken": [
                frame["combat_stats"]["attack_damage_taken"] for frame in frames
            ],
            "hazard_damage_taken": [
                frame["combat_stats"]["hazard_damage_taken"] for frame in frames
            ],
        },
    }


def _reported_signal_field_names(
    frames: list[dict[str, object]],
) -> tuple[str, ...]:
    if not frames:
        return SIGNAL_FIELD_NAMES
    first_stats = frames[0].get("signal_field_stats")
    if not isinstance(first_stats, dict):
        return SIGNAL_FIELD_NAMES
    names = tuple(str(name) for name in first_stats)
    return names or SIGNAL_FIELD_NAMES


def rebuild_taxonomy_summary_and_analytics(
    *,
    summary: dict[str, object],
    viewer: dict[str, object],
    frames: list[dict[str, object]],
    frame_ticks: list[int],
    speciation_events: list[dict[str, object]],
    species_status_counts: dict[str, int],
    taxonomy_mode: str,
) -> None:
    species_catalog = viewer["species_catalog"]
    final_frame = frames[-1] if frames else None
    alive_species = [species_id for species_id, _ in (final_frame or {}).get("species_counts", [])]
    latest_species_metrics = (final_frame or {}).get("species_metrics", {})
    summary["taxonomy_mode"] = taxonomy_mode
    summary["species_created"] = len(species_catalog)
    summary["alive_species_count"] = len(alive_species)
    summary["alive_species"] = alive_species
    summary["top_species"] = build_taxonomy_top_species(species_catalog)
    summary["speciation_events"] = len(speciation_events)
    summary["species_status_counts"] = species_status_counts
    summary.update(build_species_metric_leaderboards(latest_species_metrics))

    analytics = viewer["analytics"]
    analytics["population"]["species_count"] = [
        len(frame["species_counts"]) for frame in frames
    ]
    species_population = build_species_population_series(
        ticks=frame_ticks,
        frames=frames,
        species_ids=sorted(species_catalog, key=lambda item: int(item)),
    )
    analytics["species_population"] = species_population
    analytics["collapse_events"] = build_collapse_events(
        frame_ticks,
        species_population,
        speciation_events=speciation_events,
    )
    analytics["speciation_events"] = speciation_events
