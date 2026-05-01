from __future__ import annotations

from dataclasses import dataclass
from random import Random

from evolution_sim.genome.schema import (
    GENE_LIMITS,
    Genome,
    REPRODUCTIVE_GENE_LIMITS,
    ReproductiveGenome,
)

GENOME_RECOMBINATION_CONTRACT_VERSION = "genome_recombination_contract_v1"


@dataclass(frozen=True, slots=True)
class GenomeGroupSpec:
    """A future crossover unit kept separate from current gene arithmetic."""

    name: str
    genes: tuple[str, ...]
    reproductive_traits: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "genes": list(self.genes),
            "reproductive_traits": list(self.reproductive_traits),
        }


GENOME_GROUPS: tuple[GenomeGroupSpec, ...] = (
    GenomeGroupSpec(
        name="core_physiology",
        genes=("max_energy", "max_hydration", "max_health"),
    ),
    GenomeGroupSpec(
        name="metabolism",
        genes=("move_cost", "food_efficiency", "water_efficiency", "healing_efficiency"),
    ),
    GenomeGroupSpec(
        name="combat",
        genes=("attack_power", "attack_cost_multiplier", "defense_rating"),
    ),
    GenomeGroupSpec(
        name="trophic_strategy",
        genes=("meat_efficiency", "plant_bias", "carrion_bias", "live_prey_bias"),
    ),
    GenomeGroupSpec(
        name="habitat_affinity",
        genes=(
            "forest_affinity",
            "plain_affinity",
            "wetland_affinity",
            "rocky_affinity",
            "heat_tolerance",
        ),
    ),
    GenomeGroupSpec(
        name="reproductive_module",
        genes=("reproduction_threshold", "mutation_scale"),
        reproductive_traits=tuple(REPRODUCTIVE_GENE_LIMITS),
    ),
)


def genome_recombination_contract() -> dict[str, object]:
    return {
        "schema_version": GENOME_RECOMBINATION_CONTRACT_VERSION,
        "stage0_behavior": "single_parent_clone_then_mutate",
        "stage1_helper": "grouped_two_parent_crossover_without_mutation",
        "groups": [group.to_dict() for group in GENOME_GROUPS],
    }


def recombine_genomes(left: Genome, right: Genome, rng: Random) -> Genome:
    """Return a grouped crossover child without mutation or vitality bonuses."""
    values: dict[str, float] = {}
    reproductive_values: dict[str, float] = {}
    for group in GENOME_GROUPS:
        source = left if rng.random() < 0.5 else right
        for gene in group.genes:
            values[gene] = float(getattr(source, gene))
        source_reproductive = source.reproductive
        for trait in group.reproductive_traits:
            reproductive_values[trait] = float(getattr(source_reproductive, trait))
    return Genome(
        **values,
        reproductive=ReproductiveGenome(**reproductive_values),
    )


def apply_inbreeding_penalty(
    genome: Genome,
    *,
    penalty: float,
    scale: float,
) -> Genome:
    """Apply a bounded viability cost to close-kin sexual offspring genomes."""
    pressure = max(0.0, min(1.0, penalty)) * max(0.0, min(1.0, scale))
    if pressure <= 0.0:
        return genome

    def reduce_gene(name: str, value: float, multiplier: float) -> float:
        lower, upper = GENE_LIMITS[name]
        return max(lower, min(upper, value * multiplier))

    values = {gene: float(getattr(genome, gene)) for gene in GENE_LIMITS}
    health_multiplier = 1.0 - pressure
    efficiency_multiplier = 1.0 - pressure * 0.5
    values.update(
        {
            "max_health": reduce_gene(
                "max_health",
                genome.max_health,
                health_multiplier,
            ),
            "defense_rating": reduce_gene(
                "defense_rating",
                genome.defense_rating,
                efficiency_multiplier,
            ),
            "healing_efficiency": reduce_gene(
                "healing_efficiency",
                genome.healing_efficiency,
                efficiency_multiplier,
            ),
        }
    )
    return Genome(
        **values,
        reproductive=genome.reproductive,
    )
