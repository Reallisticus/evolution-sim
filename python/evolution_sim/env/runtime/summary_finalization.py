from __future__ import annotations

from evolution_sim.env.runtime.reporting_contracts import (
    HYDROLOGY_REASONS,
    LAND_TERRAINS,
)


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


def finalize_fresh_kill_run_totals(
    run_totals: dict[str, float],
    fresh_kill_stats: dict[str, float],
) -> dict[str, float]:
    return {
        **run_totals,
        "fresh_kill_tiles": fresh_kill_stats["fresh_kill_tiles"],
        "total_fresh_kill_energy": fresh_kill_stats["total_fresh_kill_energy"],
    }


def finalize_carcass_run_totals(
    run_totals: dict[str, float],
    carcass_stats: dict[str, float],
) -> dict[str, float]:
    return {
        **run_totals,
        "carcass_tiles": carcass_stats["carcass_tiles"],
        "total_carcass_energy": carcass_stats["total_carcass_energy"],
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


def finalize_diet_totals(totals: dict[str, float]) -> dict[str, float]:
    animal_events = totals["fresh_kill_events"] + totals["carcass_events"]
    animal_energy = totals["fresh_kill_energy"] + totals["carcass_energy"]
    total_energy = totals["plant_energy"] + animal_energy
    return {
        **{
            key: round(value, 4) if isinstance(value, float) else value
            for key, value in totals.items()
        },
        "animal_events": animal_events,
        "animal_energy": round(animal_energy, 4),
        "plant_energy_share": round(
            totals["plant_energy"] / max(total_energy, 1e-9),
            4,
        )
        if total_energy > 0
        else 0.0,
        "animal_energy_share": round(
            animal_energy / max(total_energy, 1e-9),
            4,
        )
        if total_energy > 0
        else 0.0,
        "fresh_kill_energy_share": round(
            totals["fresh_kill_energy"] / max(total_energy, 1e-9),
            4,
        )
        if total_energy > 0
        else 0.0,
        "carcass_energy_share": round(
            totals["carcass_energy"] / max(total_energy, 1e-9),
            4,
        )
        if total_energy > 0
        else 0.0,
    }


def finalize_grouped_diet_totals(
    grouped_totals: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        group: finalize_diet_totals(totals)
        for group, totals in grouped_totals.items()
    }


def finalize_animal_resource_opportunity_counts(
    counts: dict[str, int | float],
) -> dict[str, int | float]:
    return {
        key: round(value, 4) if isinstance(value, float) else int(value)
        for key, value in counts.items()
    }


def finalize_grouped_animal_resource_opportunity_counts(
    grouped_counts: dict[str, dict[str, int | float]],
) -> dict[str, dict[str, int | float]]:
    return {
        group: finalize_animal_resource_opportunity_counts(counts)
        for group, counts in grouped_counts.items()
    }


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
