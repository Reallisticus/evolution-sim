from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from evolution_sim.genome import Genome

MIND_INHERITANCE_PLACEHOLDER_VERSION = "mind_inheritance_placeholder_v1"


def empty_mind_inheritance_metadata() -> dict[str, object]:
    return {
        "schema_version": MIND_INHERITANCE_PLACEHOLDER_VERSION,
        "inherited_state": False,
        "state_size": 0,
    }


class RunMode(StrEnum):
    FULL_REPLAY = "full_replay"
    SUMMARY_ONLY = "summary_only"


@dataclass(slots=True)
class CarcassDeposit:
    energy_remaining: float
    freshness: float
    source_species: int | None
    source_agent_id: int | None
    death_tick: int
    cause: str
    killer_id: int | None = None


@dataclass(slots=True)
class FreshKillDeposit:
    energy_remaining: float
    source_species: int | None
    source_agent_id: int | None
    death_tick: int
    killer_id: int | None = None


@dataclass(slots=True)
class Tile:
    terrain: str
    food: float
    water: float
    fertility: float
    moisture: float
    heat: float
    vegetation: float
    shelter: float
    recovery_debt: float
    fresh_kill_deposits: list[FreshKillDeposit] = field(default_factory=list)
    carcass_deposits: list[CarcassDeposit] = field(default_factory=list)
    occupant_id: int | None = None

    @property
    def fresh_kill_energy(self) -> float:
        return sum(deposit.energy_remaining for deposit in self.fresh_kill_deposits)

    @property
    def carcass_energy(self) -> float:
        return sum(deposit.energy_remaining for deposit in self.carcass_deposits)

    @property
    def carcass_decay(self) -> float:
        total_energy = self.carcass_energy
        if total_energy <= 0:
            return 0.0
        return sum(
            deposit.energy_remaining * deposit.freshness for deposit in self.carcass_deposits
        ) / total_energy

    @property
    def carcass_source_species(self) -> int | None:
        active_deposits = [
            deposit for deposit in self.carcass_deposits if deposit.energy_remaining > 0
        ]
        if not active_deposits:
            return None
        source_species = {deposit.source_species for deposit in active_deposits}
        if None in source_species:
            return None
        if len(source_species) == 1:
            return next(iter(source_species))
        return None


@dataclass(slots=True)
class Agent:
    agent_id: int
    parent_id: int | None
    lineage_id: int
    birth_tick: int
    death_tick: int | None
    x: int
    y: int
    energy: float
    hydration: float
    health: float
    max_health: float
    injury_load: float
    age: int
    alive: bool
    last_reproduction_tick: int
    last_damage_source: str
    recent_plant_energy: float
    recent_fresh_kill_energy: float
    recent_carcass_energy: float
    genome_vector: tuple[float, ...]
    genome: Genome
    secondary_parent_id: int | None = None
    reproductive_group_id: int | None = None
    reproductive_stage: str = "stage0_asexual"
    reproductive_expression: str = "asexual"
    mind_inheritance_metadata: dict[str, object] = field(
        default_factory=empty_mind_inheritance_metadata
    )

    def reproduction_threshold(self) -> float:
        return self.genome.max_energy * self.genome.reproduction_threshold


@dataclass(frozen=True, slots=True)
class TrophicProfile:
    plant_share: float
    animal_share: float
    scavenger_share: float
    hunter_share: float
    breadth: float
    plant_drive: float
    animal_drive: float
    scavenger_drive: float
    hunter_drive: float
    role: str
    meat_mode: str


@dataclass(frozen=True, slots=True)
class BioticFieldState:
    prey_biomass: list[list[float]]
    carrion: list[list[float]]
    predator_risk: list[list[float]]

    def to_serializable(self) -> dict[str, list[list[float]]]:
        return {
            "prey_biomass": [
                [round(value, 4) for value in row] for row in self.prey_biomass
            ],
            "carrion": [[round(value, 4) for value in row] for row in self.carrion],
            "predator_risk": [
                [round(value, 4) for value in row] for row in self.predator_risk
            ],
        }


@dataclass(slots=True)
class SimulationWorldResult:
    run_id: str
    config: dict[str, object]
    summary: dict[str, object]
    events: list[dict[str, object]] | None
    viewer: dict[str, object] | None
    mode: RunMode = RunMode.FULL_REPLAY
