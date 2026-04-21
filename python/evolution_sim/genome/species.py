from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from collections.abc import Mapping

from evolution_sim.genome.schema import GENE_LIMITS, Genome

BASE_SPECIES_DIMENSIONS = 12

GENE_ORDER = [
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
]


@dataclass(slots=True)
class SpeciesMember:
    agent_id: int
    lineage_id: int
    genome: Genome


@dataclass(slots=True)
class SpeciesRecord:
    species_id: int
    label: str
    member_count: int
    lineages: list[int]
    centroid: dict[str, float]


def genome_vector(genome: Genome) -> tuple[float, ...]:
    values = genome.to_dict()
    return genome_vector_from_values(values)


def genome_vector_from_values(values: Mapping[str, float]) -> tuple[float, ...]:
    vector: list[float] = []
    for gene in GENE_ORDER:
        lower, upper = GENE_LIMITS[gene]
        raw = values[gene]
        vector.append((raw - lower) / (upper - lower))
    return tuple(vector)


def centroid_from_members(members: list[SpeciesMember]) -> dict[str, float]:
    if not members:
        raise ValueError("members must not be empty")

    totals = {gene: 0.0 for gene in GENE_ORDER}
    for member in members:
        values = member.genome.to_dict()
        for gene in GENE_ORDER:
            totals[gene] += values[gene]

    return {gene: totals[gene] / len(members) for gene in GENE_ORDER}


def vector_from_centroid(centroid: dict[str, float]) -> tuple[float, ...]:
    vector: list[float] = []
    for gene in GENE_ORDER:
        lower, upper = GENE_LIMITS[gene]
        raw = centroid[gene]
        vector.append((raw - lower) / (upper - lower))
    return tuple(vector)


def euclidean_distance(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    squared = [(lhs - rhs) ** 2 for lhs, rhs in zip(left, right, strict=True)]
    raw_distance = sqrt(sum(squared))
    dimension_scale = sqrt(max(len(squared), 1) / BASE_SPECIES_DIMENSIONS)
    return raw_distance / max(dimension_scale, 1e-9)
