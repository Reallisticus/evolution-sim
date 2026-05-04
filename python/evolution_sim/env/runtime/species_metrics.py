from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from evolution_sim.env.runtime.reporting_contracts import (
    ECOLOGY_STATES,
    HABITAT_STATES,
    HAZARD_TYPES,
    MEAT_MODES,
    TROPHIC_ROLES,
)
from evolution_sim.env.runtime.summary_finalization import (
    empty_carcass_totals,
    empty_combat_totals,
    empty_diet_totals,
    empty_fresh_kill_totals,
    empty_hydrology_exposure_counts,
    empty_terrain_occupancy,
)


@dataclass(frozen=True, slots=True)
class SpeciesMetricSample:
    species_id: int
    terrain: str
    habitat: str
    ecology: str
    hazard: str
    trophic_role: str
    meat_mode: str
    water_reason: str
    has_water_access: bool
    shoreline_support: bool
    wetland_support: bool
    flooded_support: bool
    refuge_exposed: bool
    energy_ratio: float
    hydration_ratio: float
    health_ratio: float
    matched_diet_ratio: float
    age: float
    vegetation: float
    recovery_debt: float
    refuge_score: float
    injury_load: float
    reproduction_ready: bool


def empty_species_metric_record(
    *,
    births: int = 0,
    deaths: int = 0,
    reproduction_success: int = 0,
) -> dict[str, object]:
    return {
        "alive_count": 0,
        "terrain_occupancy": empty_terrain_occupancy(),
        "hydrology_exposure_counts": empty_hydrology_exposure_counts(),
        "habitat_occupancy": {state: 0 for state in HABITAT_STATES},
        "ecology_occupancy": {state: 0 for state in ECOLOGY_STATES},
        "hazard_occupancy": {hazard_type: 0 for hazard_type in HAZARD_TYPES},
        "trophic_role_occupancy": {role: 0 for role in TROPHIC_ROLES},
        "meat_mode_occupancy": {mode: 0 for mode in MEAT_MODES},
        "energy_ratio_total": 0.0,
        "hydration_ratio_total": 0.0,
        "health_ratio_total": 0.0,
        "age_total": 0.0,
        "vegetation_total": 0.0,
        "recovery_total": 0.0,
        "refuge_score_total": 0.0,
        "refuge_exposed_count": 0,
        "injury_count": 0,
        "hazard_exposed_count": 0,
        "energy_stressed_count": 0,
        "hydration_stressed_count": 0,
        "reproduction_ready_count": 0,
        "matched_diet_ratio_total": 0.0,
        "births": births,
        "deaths": deaths,
        "reproduction_success": reproduction_success,
    }


def resolve_species_id(
    species_map: dict[int, int],
    previous_species_map: dict[int, int],
    agent_id: int | None,
) -> int:
    if agent_id is None:
        return 0
    return species_map.get(agent_id, previous_species_map.get(agent_id, 0))


def build_species_metrics(
    *,
    occupancy_samples: Iterable[SpeciesMetricSample],
    species_map: dict[int, int],
    previous_species_map: dict[int, int],
    birth_pairs: Iterable[tuple[int, int]],
    dead_agent_ids: Iterable[int],
    attack_records: Iterable[tuple[int, bool, float, bool]],
    damage_records: Iterable[tuple[int, float, str]],
    carcass_deposit_records: Iterable[tuple[int | None, float]],
    fresh_kill_deposit_records: Iterable[tuple[int | None, float]],
    fresh_kill_consumption_records: Iterable[tuple[int, float, float]],
    carcass_consumption_records: Iterable[tuple[int, float, float]],
    feeding_records: Iterable[tuple[int, str, float]],
) -> dict[str, dict[str, object]]:
    metrics: dict[int, dict[str, object]] = {}
    births_by_child_species: dict[int, int] = defaultdict(int)
    births_by_parent_species: dict[int, int] = defaultdict(int)
    deaths_by_species: dict[int, int] = defaultdict(int)
    attack_stats_by_species: dict[int, dict[str, float]] = defaultdict(empty_combat_totals)
    fresh_kill_stats_by_species: dict[int, dict[str, float]] = defaultdict(empty_fresh_kill_totals)
    carcass_stats_by_species: dict[int, dict[str, float]] = defaultdict(empty_carcass_totals)
    diet_stats_by_species: dict[int, dict[str, float]] = defaultdict(empty_diet_totals)

    for parent_id, child_id in birth_pairs:
        child_species = species_map.get(child_id)
        if child_species is not None:
            births_by_child_species[child_species] += 1
        parent_species = species_map.get(parent_id, previous_species_map.get(parent_id))
        if parent_species is not None:
            births_by_parent_species[parent_species] += 1

    for dead_id in dead_agent_ids:
        dead_species = previous_species_map.get(dead_id)
        if dead_species is not None:
            deaths_by_species[dead_species] += 1

    for attacker_id, success, damage, kill in attack_records:
        species_id = resolve_species_id(species_map, previous_species_map, attacker_id)
        attack_stats_by_species[species_id]["attack_attempts"] += 1
        if success:
            attack_stats_by_species[species_id]["successful_attacks"] += 1
            attack_stats_by_species[species_id]["damage_dealt"] += damage
        if kill:
            attack_stats_by_species[species_id]["kills"] += 1

    for agent_id, amount, source in damage_records:
        species_id = resolve_species_id(species_map, previous_species_map, agent_id)
        attack_stats_by_species[species_id]["damage_taken"] += amount
        if source.startswith("hazard_"):
            attack_stats_by_species[species_id]["hazard_damage_taken"] += amount

    for species_id, deposited_energy in carcass_deposit_records:
        normalized_species_id = int(species_id or 0)
        carcass_stats_by_species[normalized_species_id]["deposition_events"] += 1
        carcass_stats_by_species[normalized_species_id]["energy_deposited"] += deposited_energy

    for species_id, deposited_energy in fresh_kill_deposit_records:
        normalized_species_id = int(species_id or 0)
        fresh_kill_stats_by_species[normalized_species_id]["deposition_events"] += 1
        fresh_kill_stats_by_species[normalized_species_id]["energy_deposited"] += deposited_energy

    for agent_id, consumed, gained_energy in fresh_kill_consumption_records:
        species_id = resolve_species_id(species_map, previous_species_map, agent_id)
        fresh_kill_stats_by_species[species_id]["consumption_events"] += 1
        fresh_kill_stats_by_species[species_id]["energy_consumed"] += consumed
        fresh_kill_stats_by_species[species_id]["gained_energy"] += gained_energy

    for agent_id, consumed, gained_energy in carcass_consumption_records:
        species_id = resolve_species_id(species_map, previous_species_map, agent_id)
        carcass_stats_by_species[species_id]["consumption_events"] += 1
        carcass_stats_by_species[species_id]["energy_consumed"] += consumed
        carcass_stats_by_species[species_id]["gained_energy"] += gained_energy

    for agent_id, food_source, gained_energy in feeding_records:
        species_id = resolve_species_id(species_map, previous_species_map, agent_id)
        prefix = food_source if food_source in {"plant", "fresh_kill", "carcass"} else "plant"
        diet_stats_by_species[species_id][f"{prefix}_events"] += 1
        diet_stats_by_species[species_id][f"{prefix}_energy"] += gained_energy

    for sample in occupancy_samples:
        species_id = sample.species_id
        record = metrics.setdefault(
            species_id,
            empty_species_metric_record(
                births=births_by_child_species.get(species_id, 0),
                deaths=deaths_by_species.get(species_id, 0),
                reproduction_success=births_by_parent_species.get(species_id, 0),
            ),
        )
        record["alive_count"] += 1
        record["energy_ratio_total"] += sample.energy_ratio
        record["hydration_ratio_total"] += sample.hydration_ratio
        record["health_ratio_total"] += sample.health_ratio
        record["matched_diet_ratio_total"] += sample.matched_diet_ratio
        record["age_total"] += sample.age
        record["vegetation_total"] += sample.vegetation
        record["recovery_total"] += sample.recovery_debt
        record["refuge_score_total"] += sample.refuge_score
        record["terrain_occupancy"][sample.terrain] += 1
        record["habitat_occupancy"][sample.habitat] += 1
        record["ecology_occupancy"][sample.ecology] += 1
        record["hazard_occupancy"][sample.hazard] += 1
        record["trophic_role_occupancy"][sample.trophic_role] += 1
        if sample.meat_mode != "none":
            record["meat_mode_occupancy"][sample.meat_mode] += 1
        record["hydrology_exposure_counts"][f"primary_{sample.water_reason}"] += 1
        if sample.has_water_access:
            record["terrain_occupancy"]["water_access"] += 1
        if sample.shoreline_support:
            record["hydrology_exposure_counts"]["shoreline_support"] += 1
        if sample.wetland_support:
            record["hydrology_exposure_counts"]["wetland_support"] += 1
        if sample.flooded_support:
            record["hydrology_exposure_counts"]["flooded_support"] += 1
        if sample.refuge_exposed:
            record["hydrology_exposure_counts"]["refuge_exposed"] += 1
            record["refuge_exposed_count"] += 1
        if sample.hazard != "none":
            record["hazard_exposed_count"] += 1
        if sample.injury_load >= 0.08:
            record["injury_count"] += 1
        if sample.energy_ratio < 0.35:
            record["energy_stressed_count"] += 1
        if sample.hydration_ratio < 0.35:
            record["hydration_stressed_count"] += 1
        if sample.reproduction_ready:
            record["reproduction_ready_count"] += 1

    referenced_species = (
        set(births_by_child_species)
        | set(births_by_parent_species)
        | set(deaths_by_species)
        | set(attack_stats_by_species)
        | set(fresh_kill_stats_by_species)
        | set(carcass_stats_by_species)
        | set(diet_stats_by_species)
    )
    for species_id in referenced_species:
        metrics.setdefault(
            species_id,
            empty_species_metric_record(
                births=births_by_child_species.get(species_id, 0),
                deaths=deaths_by_species.get(species_id, 0),
                reproduction_success=births_by_parent_species.get(species_id, 0),
            ),
        )

    return finalize_species_metrics(
        metrics=metrics,
        attack_stats_by_species=attack_stats_by_species,
        fresh_kill_stats_by_species=fresh_kill_stats_by_species,
        carcass_stats_by_species=carcass_stats_by_species,
        diet_stats_by_species=diet_stats_by_species,
    )


def finalize_species_metrics(
    *,
    metrics: dict[int, dict[str, object]],
    attack_stats_by_species: dict[int, dict[str, float]],
    fresh_kill_stats_by_species: dict[int, dict[str, float]],
    carcass_stats_by_species: dict[int, dict[str, float]],
    diet_stats_by_species: dict[int, dict[str, float]],
) -> dict[str, dict[str, object]]:
    finalized: dict[str, dict[str, object]] = {}
    for species_id, record in metrics.items():
        alive_count = max(int(record["alive_count"]), 1)
        plant_energy = diet_stats_by_species[species_id]["plant_energy"]
        fresh_kill_energy = diet_stats_by_species[species_id]["fresh_kill_energy"]
        carcass_energy = diet_stats_by_species[species_id]["carcass_energy"]
        animal_energy = fresh_kill_energy + carcass_energy
        total_diet_energy = plant_energy + animal_energy
        finalized[str(species_id)] = {
            "alive_count": int(record["alive_count"]),
            "births": int(record["births"]),
            "deaths": int(record["deaths"]),
            "reproduction_success": int(record["reproduction_success"]),
            "avg_energy_ratio": round(record["energy_ratio_total"] / alive_count, 4),
            "avg_hydration_ratio": round(record["hydration_ratio_total"] / alive_count, 4),
            "avg_health_ratio": round(record["health_ratio_total"] / alive_count, 4),
            "avg_age": round(record["age_total"] / alive_count, 2),
            "avg_tile_vegetation": round(record["vegetation_total"] / alive_count, 4),
            "avg_recovery_debt": round(record["recovery_total"] / alive_count, 4),
            "avg_refuge_score_occupied_tiles": round(record["refuge_score_total"] / alive_count, 4),
            "refuge_exposure_rate": round(record["refuge_exposed_count"] / alive_count, 4),
            "injury_rate": round(record["injury_count"] / alive_count, 4),
            "hazard_exposure_rate": round(record["hazard_exposed_count"] / alive_count, 4),
            "energy_stress_rate": round(record["energy_stressed_count"] / alive_count, 4),
            "hydration_stress_rate": round(record["hydration_stressed_count"] / alive_count, 4),
            "reproduction_ready_rate": round(record["reproduction_ready_count"] / alive_count, 4),
            "avg_matched_diet_ratio": round(record["matched_diet_ratio_total"] / alive_count, 4),
            "attack_attempts": int(attack_stats_by_species[species_id]["attack_attempts"]),
            "successful_attacks": int(attack_stats_by_species[species_id]["successful_attacks"]),
            "kills": int(attack_stats_by_species[species_id]["kills"]),
            "damage_dealt": round(attack_stats_by_species[species_id]["damage_dealt"], 4),
            "damage_taken": round(attack_stats_by_species[species_id]["damage_taken"], 4),
            "hazard_damage_taken": round(
                attack_stats_by_species[species_id]["hazard_damage_taken"],
                4,
            ),
            "plant_consumption": int(diet_stats_by_species[species_id]["plant_events"]),
            "plant_energy_consumed": round(plant_energy, 4),
            "fresh_kill_deposition": int(
                fresh_kill_stats_by_species[species_id]["deposition_events"]
            ),
            "fresh_kill_energy_deposited": round(
                fresh_kill_stats_by_species[species_id]["energy_deposited"],
                4,
            ),
            "fresh_kill_consumption": int(
                fresh_kill_stats_by_species[species_id]["consumption_events"]
            ),
            "fresh_kill_energy_consumed": round(
                fresh_kill_stats_by_species[species_id]["energy_consumed"],
                4,
            ),
            "fresh_kill_gained_energy": round(
                fresh_kill_stats_by_species[species_id]["gained_energy"],
                4,
            ),
            "carcass_deposition": int(carcass_stats_by_species[species_id]["deposition_events"]),
            "carcass_energy_deposited": round(
                carcass_stats_by_species[species_id]["energy_deposited"],
                4,
            ),
            "carcass_consumption": int(carcass_stats_by_species[species_id]["consumption_events"]),
            "carcass_energy_consumed": round(
                carcass_stats_by_species[species_id]["energy_consumed"],
                4,
            ),
            "carcass_gained_energy": round(
                carcass_stats_by_species[species_id]["gained_energy"],
                4,
            ),
            "realized_plant_share": round(plant_energy / max(total_diet_energy, 1e-9), 4)
            if total_diet_energy > 0
            else 0.0,
            "realized_animal_share": round(animal_energy / max(total_diet_energy, 1e-9), 4)
            if total_diet_energy > 0
            else 0.0,
            "realized_fresh_kill_share": round(
                fresh_kill_energy / max(total_diet_energy, 1e-9),
                4,
            )
            if total_diet_energy > 0
            else 0.0,
            "realized_carcass_share": round(carcass_energy / max(total_diet_energy, 1e-9), 4)
            if total_diet_energy > 0
            else 0.0,
            "terrain_occupancy": record["terrain_occupancy"],
            "hydrology_exposure_counts": record["hydrology_exposure_counts"],
            "habitat_occupancy": record["habitat_occupancy"],
            "ecology_occupancy": record["ecology_occupancy"],
            "hazard_occupancy": record["hazard_occupancy"],
            "trophic_role_occupancy": record["trophic_role_occupancy"],
            "meat_mode_occupancy": record["meat_mode_occupancy"],
        }
    return finalized
