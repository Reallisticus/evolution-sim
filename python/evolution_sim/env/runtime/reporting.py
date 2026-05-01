from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

LAND_TERRAINS = ("plain", "forest", "wetland", "rocky")
HYDROLOGY_REASONS = ("none", "adjacent_water", "wetland", "flooded")
HABITAT_STATES = ("stable", "bloom", "flooded", "parched")
ECOLOGY_STATES = ("stable", "lush", "recovering", "depleted")
HAZARD_TYPES = ("none", "exposure", "instability")
TROPHIC_ROLES = ("herbivore", "omnivore", "carnivore")
MEAT_MODES = ("scavenger", "hunter", "mixed")
MEAT_MODE_SERIES = ("none", "scavenger", "hunter", "mixed")
REFUGE_REASONS = ("none", "canopy_refuge")
DIET_SERIES_METRICS = (
    "plant_events",
    "plant_energy",
    "fresh_kill_events",
    "fresh_kill_energy",
    "carcass_events",
    "carcass_energy",
    "plant_energy_share",
    "animal_energy_share",
    "fresh_kill_energy_share",
    "carcass_energy_share",
)
BIOTIC_FIELD_NAMES = ("prey_biomass", "carrion", "predator_risk")
SIGNAL_FIELD_NAMES = ("reproductive_signal", "communication_signal")


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


def empty_combat_totals() -> dict[str, float]:
    return {
        "attack_attempts": 0.0,
        "successful_attacks": 0.0,
        "kills": 0.0,
        "damage_dealt": 0.0,
        "damage_taken": 0.0,
        "hazard_damage_taken": 0.0,
    }


def empty_carcass_totals() -> dict[str, float]:
    return {
        "deposition_events": 0.0,
        "energy_deposited": 0.0,
        "energy_decayed": 0.0,
        "consumption_events": 0.0,
        "energy_consumed": 0.0,
        "gained_energy": 0.0,
    }


def empty_fresh_kill_totals() -> dict[str, float]:
    return {
        "deposition_events": 0.0,
        "energy_deposited": 0.0,
        "energy_converted_to_carcass": 0.0,
        "consumption_events": 0.0,
        "energy_consumed": 0.0,
        "gained_energy": 0.0,
    }


def empty_diet_totals() -> dict[str, float]:
    return {
        "plant_events": 0.0,
        "plant_energy": 0.0,
        "fresh_kill_events": 0.0,
        "fresh_kill_energy": 0.0,
        "carcass_events": 0.0,
        "carcass_energy": 0.0,
        "animal_events": 0.0,
        "animal_energy": 0.0,
    }


def empty_grouped_diet_totals(groups: list[str] | dict[str, int]) -> dict[str, dict[str, float]]:
    return {str(group): empty_diet_totals() for group in groups}


def empty_terrain_occupancy() -> dict[str, int]:
    return {**{terrain: 0 for terrain in LAND_TERRAINS}, "water_access": 0}


def empty_hydrology_exposure_counts() -> dict[str, int]:
    return {
        **{f"primary_{reason}": 0 for reason in HYDROLOGY_REASONS},
        "shoreline_support": 0,
        "wetland_support": 0,
        "flooded_support": 0,
        "refuge_exposed": 0,
    }


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
            "fresh_kill_deposition": int(fresh_kill_stats_by_species[species_id]["deposition_events"]),
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


def build_species_metric_leaderboards(
    latest_species_metrics: dict[str, dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    return {
        "top_species_by_realized_animal_share": sorted(
            (
                {
                    "species_id": int(species_id),
                    "alive_count": int(metrics["alive_count"]),
                    "realized_animal_share": float(metrics["realized_animal_share"]),
                    "realized_fresh_kill_share": float(metrics["realized_fresh_kill_share"]),
                    "realized_carcass_share": float(metrics["realized_carcass_share"]),
                    "reproduction_success": int(metrics["reproduction_success"]),
                }
                for species_id, metrics in latest_species_metrics.items()
                if metrics["alive_count"] > 0 or metrics["reproduction_success"] > 0
            ),
            key=lambda item: (
                -item["realized_animal_share"],
                -item["reproduction_success"],
                item["species_id"],
            ),
        )[:10],
        "top_species_by_realized_fresh_kill_share": sorted(
            (
                {
                    "species_id": int(species_id),
                    "alive_count": int(metrics["alive_count"]),
                    "realized_fresh_kill_share": float(metrics["realized_fresh_kill_share"]),
                    "fresh_kill_gained_energy": float(metrics["fresh_kill_gained_energy"]),
                    "kills": int(metrics["kills"]),
                }
                for species_id, metrics in latest_species_metrics.items()
                if metrics["alive_count"] > 0
                or metrics["fresh_kill_gained_energy"] > 0
                or metrics["kills"] > 0
            ),
            key=lambda item: (
                -item["realized_fresh_kill_share"],
                -item["fresh_kill_gained_energy"],
                -item["kills"],
                item["species_id"],
            ),
        )[:10],
        "top_species_by_realized_carcass_share": sorted(
            (
                {
                    "species_id": int(species_id),
                    "alive_count": int(metrics["alive_count"]),
                    "realized_carcass_share": float(metrics["realized_carcass_share"]),
                    "carcass_gained_energy": float(metrics["carcass_gained_energy"]),
                    "carcass_energy_consumed": float(metrics["carcass_energy_consumed"]),
                    "attack_attempts": int(metrics["attack_attempts"]),
                    "kills": int(metrics["kills"]),
                }
                for species_id, metrics in latest_species_metrics.items()
                if metrics["alive_count"] > 0
                or metrics["carcass_gained_energy"] > 0
                or metrics["attack_attempts"] > 0
            ),
            key=lambda item: (
                -item["realized_carcass_share"],
                -item["carcass_gained_energy"],
                -item["attack_attempts"],
                item["species_id"],
            ),
        )[:10],
        "top_carnivore_species": sorted(
            (
                {
                    "species_id": int(species_id),
                    "alive_count": int(metrics["alive_count"]),
                    "realized_animal_share": float(metrics["realized_animal_share"]),
                    "realized_fresh_kill_share": float(metrics["realized_fresh_kill_share"]),
                    "realized_carcass_share": float(metrics["realized_carcass_share"]),
                }
                for species_id, metrics in latest_species_metrics.items()
                if int(metrics["trophic_role_occupancy"]["carnivore"]) > 0
            ),
            key=lambda item: (
                -item["alive_count"],
                -item["realized_animal_share"],
                item["species_id"],
            ),
        )[:10],
        "top_hunter_species": sorted(
            (
                {
                    "species_id": int(species_id),
                    "alive_count": int(metrics["alive_count"]),
                    "kills": int(metrics["kills"]),
                    "realized_fresh_kill_share": float(metrics["realized_fresh_kill_share"]),
                }
                for species_id, metrics in latest_species_metrics.items()
                if int(metrics["meat_mode_occupancy"]["hunter"]) > 0
            ),
            key=lambda item: (
                -item["alive_count"],
                -item["kills"],
                -item["realized_fresh_kill_share"],
                item["species_id"],
            ),
        )[:10],
        "top_scavenger_species": sorted(
            (
                {
                    "species_id": int(species_id),
                    "alive_count": int(metrics["alive_count"]),
                    "realized_carcass_share": float(metrics["realized_carcass_share"]),
                    "carcass_gained_energy": float(metrics["carcass_gained_energy"]),
                }
                for species_id, metrics in latest_species_metrics.items()
                if int(metrics["meat_mode_occupancy"]["scavenger"]) > 0
            ),
            key=lambda item: (
                -item["alive_count"],
                -item["realized_carcass_share"],
                -item["carcass_gained_energy"],
                item["species_id"],
            ),
        )[:10],
    }


def build_run_top_species(
    species_registry: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    return sorted(
        (
            {
                "species_id": species_id,
                "label": registry["label"],
                "peak_members": registry["peak_members"],
                "alive_members": registry.get("current_members", 0),
                "lineages": sorted(registry["lineages"]),
            }
            for species_id, registry in species_registry.items()
        ),
        key=lambda item: (-item["alive_members"], -item["peak_members"], item["species_id"]),
    )[:10]


def build_taxonomy_top_species(
    species_catalog: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    return sorted(
        (
            {
                "species_id": int(species_id),
                "label": payload["label"],
                "peak_members": payload["peak_members"],
                "alive_members": payload.get("current_members", 0),
                "lineages": payload["lineages"],
                "origin_kind": payload["taxonomy_origin"],
                "status": payload["status"],
                "parent_species_id": payload.get("parent_species_id"),
            }
            for species_id, payload in species_catalog.items()
        ),
        key=lambda item: (-item["alive_members"], -item["peak_members"], item["species_id"]),
    )[:10]


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
            for field_name in SIGNAL_FIELD_NAMES
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
