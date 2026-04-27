from __future__ import annotations

from typing import Any

from evolution_sim.genome import Genome


GENOME_PROFILE_FIELDS: tuple[str, ...] = (
    "max_energy",
    "max_hydration",
    "max_health",
    "move_cost",
    "food_efficiency",
    "water_efficiency",
    "attack_power",
    "attack_cost_multiplier",
    "defense_rating",
    "meat_efficiency",
    "healing_efficiency",
    "plant_bias",
    "carrion_bias",
    "live_prey_bias",
    "forest_affinity",
    "plain_affinity",
    "wetland_affinity",
    "rocky_affinity",
    "heat_tolerance",
    "reproduction_threshold",
    "mutation_scale",
)


def genome_profile_key(genome: Genome) -> tuple[float, ...]:
    return tuple(float(getattr(genome, field)) for field in GENOME_PROFILE_FIELDS)


def cached_trophic_profile(
    world: Any,
    genome: Genome,
    *,
    genome_vector: tuple[float, ...] | None = None,
):
    live_key = genome_profile_key(genome)
    key = genome_vector if genome_vector == live_key else live_key
    profile = world._trophic_profile_cache.get(key)
    if profile is None:
        profile = world._compute_trophic_profile_for_genome(genome)
        world._trophic_profile_cache[key] = profile
    return profile
