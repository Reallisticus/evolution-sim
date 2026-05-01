from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Iterable

import evolution_sim.env.runtime.mating as runtime_mating
from evolution_sim.env.events import EventType
from evolution_sim.env.runtime.state import (
    Agent,
    TrophicProfile,
    empty_mind_inheritance_metadata,
)
from evolution_sim.genome.schema import Genome
from evolution_sim.genome.recombination import apply_inbreeding_penalty, recombine_genomes
from evolution_sim.genome.schema import GENE_LIMITS
from evolution_sim.genome.species import genome_vector

REPRODUCTIVE_GROUP_CONTRACT_VERSION = "reproductive_group_contract_v1"
STAGE0_ASEXUAL = "stage0_asexual"
ASEXUAL_EXPRESSION = "asexual"


@dataclass(frozen=True, slots=True)
class ReproductiveState:
    """Agent-facing reproductive identity independent of replay taxonomy."""

    group_id: int
    stage: str
    expression: str


@dataclass(slots=True)
class ReproductiveGroupRecord:
    """Live compatibility group metadata, not a species or replay taxonomy record."""

    group_id: int
    founder_lineage_id: int
    founder_agent_id: int
    created_tick: int
    stage: str = STAGE0_ASEXUAL
    parent_group_ids: tuple[int, ...] = ()
    asexual_births: int = 0
    sexual_births: int = 0
    hybrid_births: int = 0
    last_seen_tick: int = 0

    def to_dict(
        self,
        *,
        member_count: int,
        alive_member_count: int,
    ) -> dict[str, object]:
        return {
            "group_id": self.group_id,
            "founder_lineage_id": self.founder_lineage_id,
            "founder_agent_id": self.founder_agent_id,
            "created_tick": self.created_tick,
            "last_seen_tick": self.last_seen_tick,
            "stage": self.stage,
            "parent_group_ids": list(self.parent_group_ids),
            "member_count": member_count,
            "alive_member_count": alive_member_count,
            "asexual_births": self.asexual_births,
            "sexual_births": self.sexual_births,
            "hybrid_births": self.hybrid_births,
        }


def reproductive_group_contract() -> dict[str, object]:
    return {
        "schema_version": REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        "identity": "live_reproductive_compatibility_group",
        "taxonomy_dependency": "independent_from_replay_species_and_ecotypes",
        "stage0": {
            "stage": STAGE0_ASEXUAL,
            "expression": ASEXUAL_EXPRESSION,
            "birth_mode": "single_parent_clone_then_mutate",
        },
        "stage1": {
            "stage": "stage1_facultative_sex",
            "expression": "same_group_compatible",
            "birth_mode": "rare_same_group_two_parent_recombination",
            "fallback": "single_parent_asexual_when_no_valid_partner",
        },
        "future_modes": [
            "role_expression_recombination",
            "rare_gated_hybridization",
        ],
    }


def founder_reproductive_state(group_id: int) -> ReproductiveState:
    return ReproductiveState(
        group_id=group_id,
        stage=STAGE0_ASEXUAL,
        expression=ASEXUAL_EXPRESSION,
    )


def asexual_child_reproductive_state(parent: Agent) -> ReproductiveState:
    return ReproductiveState(
        group_id=parent.reproductive_group_id or parent.lineage_id,
        stage=parent.reproductive_stage,
        expression=parent.reproductive_expression,
    )


def child_reproductive_state(
    *,
    group_id: int,
    stage: str,
    expression: str,
) -> ReproductiveState:
    return ReproductiveState(
        group_id=group_id,
        stage=stage,
        expression=expression,
    )


def build_child_agent(
    *,
    agent_id: int,
    primary_parent: Agent,
    secondary_parent: Agent | None,
    lineage_id: int,
    birth_tick: int,
    destination: tuple[int, int],
    genome: Genome,
    energy_fraction: float,
    hydration_fraction: float,
    health_fraction: float,
    reproductive_state: ReproductiveState,
    mind_inheritance_metadata: dict[str, object],
) -> Agent:
    return Agent(
        agent_id=agent_id,
        parent_id=primary_parent.agent_id,
        lineage_id=lineage_id,
        birth_tick=birth_tick,
        death_tick=None,
        x=destination[0],
        y=destination[1],
        energy=genome.max_energy * energy_fraction,
        hydration=genome.max_hydration * hydration_fraction,
        health=genome.max_health * health_fraction,
        max_health=genome.max_health,
        injury_load=0.0,
        age=0,
        alive=True,
        last_reproduction_tick=-10_000,
        last_damage_source="none",
        recent_plant_energy=0.0,
        recent_fresh_kill_energy=0.0,
        recent_carcass_energy=0.0,
        genome_vector=genome_vector(genome),
        genome=genome,
        secondary_parent_id=(
            secondary_parent.agent_id if secondary_parent is not None else None
        ),
        reproductive_group_id=reproductive_state.group_id,
        reproductive_stage=reproductive_state.stage,
        reproductive_expression=reproductive_state.expression,
        mind_inheritance_metadata=mind_inheritance_metadata,
    )


def register_founder_group(
    registry: dict[int, ReproductiveGroupRecord],
    agent: Agent,
    *,
    tick: int,
) -> None:
    group_id = agent.reproductive_group_id or agent.lineage_id
    record = registry.get(group_id)
    if record is None:
        registry[group_id] = ReproductiveGroupRecord(
            group_id=group_id,
            founder_lineage_id=agent.lineage_id,
            founder_agent_id=agent.agent_id,
            created_tick=tick,
            stage=agent.reproductive_stage,
            last_seen_tick=tick,
        )
        return
    record.last_seen_tick = max(record.last_seen_tick, tick)


def record_asexual_birth(
    registry: dict[int, ReproductiveGroupRecord],
    parent: Agent,
    child: Agent,
    *,
    tick: int,
) -> None:
    group_id = (
        child.reproductive_group_id
        or parent.reproductive_group_id
        or child.lineage_id
    )
    record = registry.get(group_id)
    if record is None:
        record = ReproductiveGroupRecord(
            group_id=group_id,
            founder_lineage_id=parent.lineage_id,
            founder_agent_id=parent.agent_id,
            created_tick=parent.birth_tick,
            stage=child.reproductive_stage,
            last_seen_tick=tick,
        )
        registry[group_id] = record
    record.asexual_births += 1
    record.last_seen_tick = tick


def record_sexual_birth(
    registry: dict[int, ReproductiveGroupRecord],
    primary_parent: Agent,
    secondary_parent: Agent,
    child: Agent,
    *,
    tick: int,
    hybrid: bool = False,
) -> None:
    group_id = (
        child.reproductive_group_id
        or primary_parent.reproductive_group_id
        or primary_parent.lineage_id
    )
    record = registry.get(group_id)
    if record is None:
        record = ReproductiveGroupRecord(
            group_id=group_id,
            founder_lineage_id=primary_parent.lineage_id,
            founder_agent_id=primary_parent.agent_id,
            created_tick=primary_parent.birth_tick,
            stage=child.reproductive_stage,
            parent_group_ids=tuple(
                sorted(
                    {
                        primary_parent.reproductive_group_id
                        or primary_parent.lineage_id,
                        secondary_parent.reproductive_group_id
                        or secondary_parent.lineage_id,
                    }
                )
            ),
            last_seen_tick=tick,
        )
        registry[group_id] = record
    record.sexual_births += 1
    if hybrid:
        record.hybrid_births += 1
    record.last_seen_tick = tick


def build_reproductive_group_catalog(
    registry: dict[int, ReproductiveGroupRecord],
    agents: Iterable[Agent],
) -> dict[str, object]:
    member_counts, alive_member_counts = _member_counts(agents)
    return {
        "schema_version": REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        "groups": {
            str(group_id): record.to_dict(
                member_count=member_counts.get(group_id, 0),
                alive_member_count=alive_member_counts.get(group_id, 0),
            )
            for group_id, record in sorted(registry.items())
        },
    }


def build_reproductive_group_summary(
    registry: dict[int, ReproductiveGroupRecord],
    agents: Iterable[Agent],
) -> dict[str, object]:
    agent_list = list(agents)
    member_counts, alive_member_counts = _member_counts(agent_list)
    return {
        "schema_version": REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        "group_count": len(registry),
        "alive_group_count": sum(
            1 for count in alive_member_counts.values() if count > 0
        ),
        "asexual_births": sum(record.asexual_births for record in registry.values()),
        "sexual_births": sum(record.sexual_births for record in registry.values()),
        "hybrid_births": sum(record.hybrid_births for record in registry.values()),
        "stage_counts": _stage_counts(registry),
        "alive_expression_counts": _alive_expression_counts(agent_list),
        "top_groups": [
            record.to_dict(
                member_count=member_counts.get(record.group_id, 0),
                alive_member_count=alive_member_counts.get(record.group_id, 0),
            )
            for record in sorted(
                registry.values(),
                key=lambda item: (
                    -alive_member_counts.get(item.group_id, 0),
                    -member_counts.get(item.group_id, 0),
                    item.group_id,
                ),
            )[:10]
        ],
    }


def _member_counts(agents: Iterable[Agent]) -> tuple[dict[int, int], dict[int, int]]:
    member_counts: Counter[int] = Counter()
    alive_member_counts: Counter[int] = Counter()
    for agent in agents:
        group_id = agent.reproductive_group_id or agent.lineage_id
        member_counts[group_id] += 1
        if agent.alive:
            alive_member_counts[group_id] += 1
    return dict(member_counts), dict(alive_member_counts)


def _stage_counts(registry: dict[int, ReproductiveGroupRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter(record.stage for record in registry.values())
    return {stage: counts[stage] for stage in sorted(counts)}


def _alive_expression_counts(agents: Iterable[Agent]) -> dict[str, int]:
    counts: Counter[str] = Counter(
        agent.reproductive_expression for agent in agents if agent.alive
    )
    return {expression: counts[expression] for expression in sorted(counts)}


def empty_reproduction_blocked_counts() -> dict[str, int]:
    return {
        "max_population": 0,
        "local_crowding": 0,
        "destination_unavailable": 0,
    }


def empty_reproduction_readiness_counts() -> dict[str, int]:
    return {
        "alive_agents": 0,
        "biologically_ready_agents": 0,
        "ready_agents": 0,
        "blocked_by_max_population_agents": 0,
        "blocked_by_local_crowding_agents": 0,
    }


def empty_reproduction_biological_blocker_counts() -> dict[str, int]:
    return {
        "age": 0,
        "cooldown": 0,
        "energy": 0,
        "hydration": 0,
        "health": 0,
        "matched_diet": 0,
    }


def empty_reproduction_energy_readiness_counts() -> dict[str, int | float]:
    return {
        "alive_agents": 0,
        "energy_shortfall_agents": 0,
        "energy_total": 0.0,
        "energy_required_total": 0.0,
        "energy_gap_total": 0.0,
    }


def finalize_reproduction_energy_readiness_counts(
    counts: dict[str, int | float],
) -> dict[str, int | float]:
    return {
        "alive_agents": int(counts["alive_agents"]),
        "energy_shortfall_agents": int(counts["energy_shortfall_agents"]),
        "energy_total": round(float(counts["energy_total"]), 4),
        "energy_required_total": round(float(counts["energy_required_total"]), 4),
        "energy_gap_total": round(float(counts["energy_gap_total"]), 4),
    }


def can_reproduce(world: Any, agent: Agent) -> bool:
    return reproduction_block_reason(world, agent) is None


def reproduction_energy_requirement(
    world: Any,
    agent: Agent,
    profile: TrophicProfile,
) -> float:
    requirement = agent.reproduction_threshold() * (
        1.0
        + profile.breadth * world.config.trophic.breadth_reproduction_penalty
    )
    if profile.meat_mode != "none":
        requirement *= world.config.reproduction.animal_mode_energy_requirement_multiplier
    return requirement


def biological_reproduction_block_reasons(
    world: Any,
    agent: Agent,
    profile: TrophicProfile | None = None,
) -> list[str]:
    profile = profile or world._trophic_profile(agent)
    matched_diet_ratio = world._matched_diet_ratio(agent, profile)
    reasons: list[str] = []
    if agent.age < world.config.reproduction.min_age:
        reasons.append("age")
    if world.tick - agent.last_reproduction_tick < world.config.reproduction.cooldown_ticks:
        reasons.append("cooldown")
    if agent.energy < reproduction_energy_requirement(world, agent, profile):
        reasons.append("energy")
    if (
        agent.hydration
        < agent.genome.max_hydration
        * world.config.reproduction.min_hydration_fraction
    ):
        reasons.append("hydration")
    health_requirement = world.config.combat.min_reproduction_health_ratio
    if profile.meat_mode == "scavenger":
        health_requirement = min(
            health_requirement,
            world.config.reproduction.scavenger_min_health_fraction,
        )
    if world._health_ratio(agent) < health_requirement:
        reasons.append("health")
    if matched_diet_ratio < world._matched_diet_threshold(profile):
        reasons.append("matched_diet")
    return reasons


def is_biologically_reproduction_ready(world: Any, agent: Agent) -> bool:
    return not biological_reproduction_block_reasons(world, agent)


def reproduction_block_reason(world: Any, agent: Agent) -> str | None:
    if not is_biologically_reproduction_ready(world, agent):
        return "biological"
    if len(world.alive_agents()) >= world.config.max_agents:
        return "max_population"
    if not world._has_empty_neighbor(agent.x, agent.y):
        return "local_crowding"
    return None


def is_reproduction_ready(world: Any, agent: Agent) -> bool:
    return reproduction_block_reason(world, agent) is None


def record_reproduction_blocked(world: Any, agent: Agent, reason: str) -> None:
    if reason not in world.run_reproduction_blocked_counts:
        raise ValueError(f"Unsupported reproduction block reason: {reason}")
    world.run_reproduction_blocked_counts[reason] += 1
    world.run_reproduction_blocked_counts_by_trophic_role[world._trophic_role(agent)][
        reason
    ] += 1
    world.run_reproduction_blocked_counts_by_meat_mode[world._meat_mode(agent)][
        reason
    ] += 1
    event = {
        "agent_id": agent.agent_id,
        "reason": reason,
        "x": agent.x,
        "y": agent.y,
        "alive_agents": len(world.alive_agents()),
        "max_agents": world.config.max_agents,
    }
    if world.record_tick_details:
        world.tick_reproduction_blocked_events.append(event)
    world._emit(
        EventType.AGENT_REPRODUCTION_BLOCKED,
        agent_id=agent.agent_id,
        data={
            "reason": reason,
            "x": agent.x,
            "y": agent.y,
            "alive_agents": event["alive_agents"],
            "max_agents": event["max_agents"],
        },
    )


def reproduction_readiness_counts(
    world: Any,
    alive: list[Agent],
    *,
    trophic_role_codes: dict[str, int],
    meat_mode_codes: dict[str, int],
) -> dict[str, object]:
    population_saturated = len(alive) >= world.config.max_agents
    readiness_by_role = {
        role: empty_reproduction_readiness_counts()
        for role in trophic_role_codes
        if role != "none"
    }
    readiness_by_mode = {
        mode: empty_reproduction_readiness_counts() for mode in meat_mode_codes
    }
    biological_blockers = empty_reproduction_biological_blocker_counts()
    biological_blockers_by_role = {
        role: empty_reproduction_biological_blocker_counts()
        for role in trophic_role_codes
        if role != "none"
    }
    biological_blockers_by_mode = {
        mode: empty_reproduction_biological_blocker_counts()
        for mode in meat_mode_codes
    }
    energy_readiness_by_role = {
        role: empty_reproduction_energy_readiness_counts()
        for role in trophic_role_codes
        if role != "none"
    }
    energy_readiness_by_mode = {
        mode: empty_reproduction_energy_readiness_counts()
        for mode in meat_mode_codes
    }
    biologically_ready = 0
    blocked_by_max_population = 0
    blocked_by_local_crowding = 0
    ready = 0

    def increment_readiness(agent: Agent, field: str) -> None:
        readiness_by_role[world._trophic_role(agent)][field] += 1
        readiness_by_mode[world._meat_mode(agent)][field] += 1

    def increment_blocker(agent: Agent, reason: str) -> None:
        biological_blockers[reason] += 1
        biological_blockers_by_role[world._trophic_role(agent)][reason] += 1
        biological_blockers_by_mode[world._meat_mode(agent)][reason] += 1

    def increment_energy_readiness(
        agent: Agent,
        profile: TrophicProfile,
    ) -> None:
        energy_required = reproduction_energy_requirement(world, agent, profile)
        energy_gap = max(0.0, energy_required - agent.energy)
        for counts in (
            energy_readiness_by_role[world._trophic_role(agent)],
            energy_readiness_by_mode[world._meat_mode(agent)],
        ):
            counts["alive_agents"] = int(counts["alive_agents"]) + 1
            if energy_gap > 0:
                counts["energy_shortfall_agents"] = (
                    int(counts["energy_shortfall_agents"]) + 1
                )
            counts["energy_total"] = float(counts["energy_total"]) + agent.energy
            counts["energy_required_total"] = (
                float(counts["energy_required_total"]) + energy_required
            )
            counts["energy_gap_total"] = float(counts["energy_gap_total"]) + energy_gap

    for agent in alive:
        increment_readiness(agent, "alive_agents")
        profile = world._trophic_profile(agent)
        increment_energy_readiness(agent, profile)
        block_reasons = biological_reproduction_block_reasons(world, agent, profile)
        if block_reasons:
            for reason in block_reasons:
                increment_blocker(agent, reason)
            continue
        biologically_ready += 1
        increment_readiness(agent, "biologically_ready_agents")
        if population_saturated:
            blocked_by_max_population += 1
            increment_readiness(agent, "blocked_by_max_population_agents")
        elif not world._has_empty_neighbor(agent.x, agent.y):
            blocked_by_local_crowding += 1
            increment_readiness(agent, "blocked_by_local_crowding_agents")
        else:
            ready += 1
            increment_readiness(agent, "ready_agents")

    return {
        "max_agents": world.config.max_agents,
        "saturation_at_end": round(len(alive) / max(world.config.max_agents, 1), 4),
        "peak_saturation": round(
            world.peak_alive_agents / max(world.config.max_agents, 1),
            4,
        ),
        "biologically_ready_agents": biologically_ready,
        "ready_agents": ready,
        "blocked_by_max_population_agents": blocked_by_max_population,
        "blocked_by_local_crowding_agents": blocked_by_local_crowding,
        "blocked_run_counts": dict(world.run_reproduction_blocked_counts),
        "blocked_run_counts_by_trophic_role": {
            role: dict(counts)
            for role, counts in world.run_reproduction_blocked_counts_by_trophic_role.items()
        },
        "blocked_run_counts_by_meat_mode": {
            mode: dict(counts)
            for mode, counts in world.run_reproduction_blocked_counts_by_meat_mode.items()
        },
        "biological_blocker_counts": biological_blockers,
        "biological_blocker_counts_by_trophic_role": biological_blockers_by_role,
        "biological_blocker_counts_by_meat_mode": biological_blockers_by_mode,
        "energy_readiness_by_trophic_role": {
            role: finalize_reproduction_energy_readiness_counts(counts)
            for role, counts in energy_readiness_by_role.items()
        },
        "energy_readiness_by_meat_mode": {
            mode: finalize_reproduction_energy_readiness_counts(counts)
            for mode, counts in energy_readiness_by_mode.items()
        },
        "by_trophic_role": readiness_by_role,
        "by_meat_mode": readiness_by_mode,
    }


def animal_mode_stabilized_child_genome(
    world: Any,
    parent_genome: Genome,
    child_genome: Genome,
    parent_profile: TrophicProfile,
) -> Genome:
    if parent_profile.meat_mode == "none":
        return child_genome

    def stabilize(stability: float) -> Genome:
        meat_efficiency = max(
            child_genome.meat_efficiency,
            _blend_gene(
                child_genome.meat_efficiency,
                parent_genome.meat_efficiency,
                stability,
            ),
        )
        plant_bias = min(
            child_genome.plant_bias,
            _blend_gene(child_genome.plant_bias, parent_genome.plant_bias, stability),
        )
        food_efficiency = min(
            child_genome.food_efficiency,
            _blend_gene(
                child_genome.food_efficiency,
                parent_genome.food_efficiency,
                stability,
            ),
        )
        carrion_bias = child_genome.carrion_bias
        live_prey_bias = child_genome.live_prey_bias
        attack_power = child_genome.attack_power
        attack_cost_multiplier = child_genome.attack_cost_multiplier
        defense_rating = child_genome.defense_rating

        if parent_profile.meat_mode in {"scavenger", "mixed"}:
            carrion_bias = max(
                child_genome.carrion_bias,
                _blend_gene(
                    child_genome.carrion_bias,
                    parent_genome.carrion_bias,
                    stability,
                ),
            )
        if parent_profile.meat_mode in {"hunter", "mixed"}:
            live_prey_bias = max(
                child_genome.live_prey_bias,
                _blend_gene(
                    child_genome.live_prey_bias,
                    parent_genome.live_prey_bias,
                    stability,
                ),
            )
            attack_power = max(
                child_genome.attack_power,
                _blend_gene(
                    child_genome.attack_power,
                    parent_genome.attack_power,
                    stability,
                ),
            )
            attack_cost_multiplier = min(
                child_genome.attack_cost_multiplier,
                _blend_gene(
                    child_genome.attack_cost_multiplier,
                    parent_genome.attack_cost_multiplier,
                    stability,
                ),
            )
            defense_rating = max(
                child_genome.defense_rating,
                _blend_gene(
                    child_genome.defense_rating,
                    parent_genome.defense_rating,
                    stability,
                ),
            )

        return replace(
            child_genome,
            food_efficiency=_clamp_gene_value("food_efficiency", food_efficiency),
            plant_bias=_clamp_gene_value("plant_bias", plant_bias),
            meat_efficiency=_clamp_gene_value("meat_efficiency", meat_efficiency),
            carrion_bias=_clamp_gene_value("carrion_bias", carrion_bias),
            live_prey_bias=_clamp_gene_value("live_prey_bias", live_prey_bias),
            attack_power=_clamp_gene_value("attack_power", attack_power),
            attack_cost_multiplier=_clamp_gene_value(
                "attack_cost_multiplier",
                attack_cost_multiplier,
            ),
            defense_rating=_clamp_gene_value("defense_rating", defense_rating),
        )

    configured_stability = (
        world.config.reproduction.animal_mode_offspring_trait_stability
    )
    stabilized = stabilize(configured_stability)
    stabilized_mode = world._trophic_profile_for_genome(stabilized).meat_mode
    if stabilized_mode == parent_profile.meat_mode or (
        parent_profile.meat_mode == "mixed" and stabilized_mode != "none"
    ):
        return stabilized
    return stabilize(1.0)


def child_starting_fraction(
    base_fraction: float,
    multiplier: float,
    parent_profile: TrophicProfile,
) -> float:
    if parent_profile.meat_mode == "none":
        return base_fraction
    return min(1.0, base_fraction * multiplier)


def reproduction_energy_cost(world: Any, parent_profile: TrophicProfile) -> float:
    if parent_profile.meat_mode == "none":
        return world.config.reproduction.energy_cost
    return (
        world.config.reproduction.energy_cost
        * world.config.reproduction.animal_mode_reproduction_cost_multiplier
    )


def sexual_reproduction_energy_cost(
    world: Any,
    parent_profile: TrophicProfile,
) -> float:
    return (
        reproduction_energy_cost(world, parent_profile)
        * world.config.reproduction.sexual_parent_cost_multiplier
    )


def reproductive_state_for_child(
    world: Any,
    parent: Agent,
    child_genome: Genome,
) -> ReproductiveState:
    return child_reproductive_state(
        group_id=parent.reproductive_group_id or parent.lineage_id,
        stage=runtime_mating.reproductive_stage_for_genome(
            child_genome,
            world.config.reproduction,
        ),
        expression=runtime_mating.reproductive_expression_for_genome(
            child_genome,
            world.config.reproduction,
        ),
    )


def sexual_partner_ready(world: Any, agent: Agent) -> bool:
    if not is_biologically_reproduction_ready(world, agent):
        return False
    return agent.energy >= sexual_reproduction_energy_cost(
        world,
        world._trophic_profile(agent),
    )


def reproduce(world: Any, parent: Agent) -> bool:
    destination = world._find_empty_neighbor(parent.x, parent.y)
    if destination is None:
        record_reproduction_blocked(world, parent, "destination_unavailable")
        return False

    parent_profile = world._trophic_profile(parent)
    if parent.energy >= sexual_reproduction_energy_cost(world, parent_profile):
        mate_candidate = runtime_mating.choose_same_group_mate(
            parent,
            world.agents.values(),
            config=world.config.reproduction,
            biologically_ready=lambda agent: sexual_partner_ready(world, agent),
        )
        if mate_candidate is not None:
            return reproduce_sexual(
                world,
                parent,
                mate_candidate,
                destination,
                parent_profile,
            )
    return reproduce_asexual(world, parent, destination, parent_profile)


def reproduce_asexual(
    world: Any,
    parent: Agent,
    destination: tuple[int, int],
    parent_profile: TrophicProfile,
) -> bool:
    child_genome = animal_mode_stabilized_child_genome(
        world,
        parent.genome,
        parent.genome.mutate(world.rng),
        parent_profile,
    )
    child_energy_fraction = child_starting_fraction(
        world.config.reproduction.child_energy_fraction,
        world.config.reproduction.animal_mode_child_energy_fraction_multiplier,
        parent_profile,
    )
    child_hydration_fraction = child_starting_fraction(
        world.config.reproduction.child_hydration_fraction,
        world.config.reproduction.animal_mode_child_hydration_fraction_multiplier,
        parent_profile,
    )
    reproductive_state = reproductive_state_for_child(world, parent, child_genome)
    child = build_child_agent(
        agent_id=world.next_agent_id,
        primary_parent=parent,
        secondary_parent=None,
        lineage_id=parent.lineage_id,
        birth_tick=world.tick,
        destination=destination,
        genome=child_genome,
        energy_fraction=child_energy_fraction,
        hydration_fraction=child_hydration_fraction,
        health_fraction=world.config.reproduction.child_health_fraction,
        reproductive_state=reproductive_state,
        mind_inheritance_metadata=empty_mind_inheritance_metadata(),
    )
    parent.energy -= reproduction_energy_cost(world, parent_profile)
    parent.last_reproduction_tick = world.tick
    world._place_agent(child)
    record_asexual_birth(
        world.reproductive_groups,
        parent,
        child,
        tick=world.tick,
    )
    world.next_agent_id += 1
    world.births += 1
    world.last_birth_tick = world.tick
    if world.record_tick_details:
        world.tick_birth_pairs.append((parent.agent_id, child.agent_id))
    world._invalidate_biotic_state()
    world._emit(
        EventType.AGENT_REPRODUCED,
        agent_id=parent.agent_id,
        data={
            "child_id": child.agent_id,
            "child_x": child.x,
            "child_y": child.y,
            "lineage_id": child.lineage_id,
            "birth_tick": child.birth_tick,
            "reproduction_mode": runtime_mating.ASEXUAL_REPRODUCTION_MODE,
            "parent_ids": [parent.agent_id],
        },
    )
    return True


def reproduce_sexual(
    world: Any,
    parent: Agent,
    mate_candidate: runtime_mating.MateCandidate,
    destination: tuple[int, int],
    parent_profile: TrophicProfile,
) -> bool:
    partner = mate_candidate.agent
    partner_profile = world._trophic_profile(partner)
    child_genome = recombine_genomes(parent.genome, partner.genome, world.rng).mutate(
        world.rng
    )
    child_genome = animal_mode_stabilized_child_genome(
        world,
        parent.genome,
        child_genome,
        parent_profile,
    )
    child_genome = apply_inbreeding_penalty(
        child_genome,
        penalty=mate_candidate.inbreeding_penalty,
        scale=world.config.reproduction.sexual_inbreeding_gene_penalty,
    )
    child_energy_fraction = child_starting_fraction(
        world.config.reproduction.child_energy_fraction,
        world.config.reproduction.animal_mode_child_energy_fraction_multiplier,
        parent_profile,
    )
    child_hydration_fraction = child_starting_fraction(
        world.config.reproduction.child_hydration_fraction,
        world.config.reproduction.animal_mode_child_hydration_fraction_multiplier,
        parent_profile,
    )
    reproductive_state = reproductive_state_for_child(world, parent, child_genome)
    child = build_child_agent(
        agent_id=world.next_agent_id,
        primary_parent=parent,
        secondary_parent=partner,
        lineage_id=parent.lineage_id,
        birth_tick=world.tick,
        destination=destination,
        genome=child_genome,
        energy_fraction=child_energy_fraction,
        hydration_fraction=child_hydration_fraction,
        health_fraction=world.config.reproduction.child_health_fraction,
        reproductive_state=reproductive_state,
        mind_inheritance_metadata=empty_mind_inheritance_metadata(),
    )
    parent.energy -= sexual_reproduction_energy_cost(world, parent_profile)
    partner.energy -= sexual_reproduction_energy_cost(world, partner_profile)
    parent.last_reproduction_tick = world.tick
    partner.last_reproduction_tick = world.tick
    world._place_agent(child)
    record_sexual_birth(
        world.reproductive_groups,
        parent,
        partner,
        child,
        tick=world.tick,
    )
    world.next_agent_id += 1
    world.births += 1
    world.last_birth_tick = world.tick
    if world.record_tick_details:
        world.tick_birth_pairs.append((parent.agent_id, child.agent_id))
    world._invalidate_biotic_state()
    world._emit(
        EventType.AGENT_REPRODUCED,
        agent_id=parent.agent_id,
        data={
            "child_id": child.agent_id,
            "child_x": child.x,
            "child_y": child.y,
            "lineage_id": child.lineage_id,
            "birth_tick": child.birth_tick,
            "reproduction_mode": runtime_mating.SEXUAL_REPRODUCTION_MODE,
            "partner_id": partner.agent_id,
            "parent_ids": [parent.agent_id, partner.agent_id],
            "compatibility_score": mate_candidate.compatibility_score,
            "inbreeding_penalty": mate_candidate.inbreeding_penalty,
        },
    )
    return True


def _clamp_gene_value(name: str, value: float) -> float:
    lower, upper = GENE_LIMITS[name]
    return max(lower, min(upper, value))


def _blend_gene(child_value: float, parent_value: float, stability: float) -> float:
    return child_value * (1.0 - stability) + parent_value * stability
