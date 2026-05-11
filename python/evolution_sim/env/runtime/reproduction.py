from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable

import evolution_sim.env.runtime.mating as runtime_mating
import evolution_sim.env.runtime.resources as runtime_resources
import evolution_sim.env.runtime.signals as runtime_signals
from evolution_sim.env.events import EventType
from evolution_sim.env.runtime.state import (
    Agent,
    TrophicProfile,
    empty_mind_inheritance_metadata,
)
from evolution_sim.genome.recombination import apply_inbreeding_penalty, recombine_genomes
from evolution_sim.genome.schema import (
    GENE_LIMITS,
    REPRODUCTIVE_GENE_LIMITS,
    Genome,
    ReproductiveGenome,
)
from evolution_sim.genome.species import genome_vector

REPRODUCTIVE_GROUP_CONTRACT_VERSION = "reproductive_group_contract_v1"
REPRODUCTION_EVENT_SCHEMA_VERSION = "reproduction_event_v1"
STAGE0_ASEXUAL = "stage0_asexual"
ASEXUAL_EXPRESSION = "asexual"
ANIMAL_REPRODUCTION_DIET_EPSILON = 1e-12
ANIMAL_REPRODUCTION_FULL_BONUS_SHARE = 0.2


@dataclass(frozen=True, slots=True)
class ReproductiveState:
    """Agent-facing reproductive identity independent of replay taxonomy."""

    group_id: int
    stage: str
    expression: str


@dataclass(frozen=True, slots=True)
class ReproductionAvailability:
    """Tick-local population availability for reproductive attempts."""

    alive_agents: int
    max_agents: int

    @property
    def population_saturated(self) -> bool:
        return self.alive_agents >= self.max_agents


@dataclass(frozen=True, slots=True)
class SexualOffspringPlan:
    """Resolved litter size before local destination clamping."""

    desired_count: int
    count: int
    limit_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChildBirthPlan:
    """Constructed children and per-child parent costs before world mutation."""

    children: tuple[Agent, ...]
    parent_energy_costs: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class BirthPlan:
    """Resolved birth output and metadata before applying world mutation."""

    children: tuple[Agent, ...]
    parents: tuple[Agent, ...]
    parent_energy_costs: tuple[float, ...]
    reproduction_mode: str
    mate_candidate: runtime_mating.MateCandidate | None = None
    multi_offspring_desired_count: int = 1
    multi_offspring_limit_reasons: tuple[str, ...] = ()

    @property
    def offspring_count(self) -> int:
        return len(self.children)

    @property
    def sibling_child_ids(self) -> list[int]:
        return [child.agent_id for child in self.children]

    @property
    def destinations(self) -> tuple[tuple[int, int], ...]:
        return tuple((child.x, child.y) for child in self.children)


@dataclass(frozen=True, slots=True)
class ReproductionPlacementContext:
    """World-owned placement and spatial mutation boundary for births."""

    max_agents: int
    current_alive_count: Callable[[], int]
    has_empty_neighbor: Callable[[int, int], bool]
    find_empty_neighbor: Callable[[int, int], tuple[int, int] | None]
    sibling_destination_candidates: Callable[[int, int], list[tuple[int, int]]]
    can_place_at: Callable[[int, int], bool]
    place_agent: Callable[[Agent], None]
    invalidate_spatial_state: Callable[[], None]


@dataclass(frozen=True, slots=True)
class ReproductionContext:
    """Explicit runtime authority used by reproduction helpers."""

    config: Any
    rng: Any
    tick: int
    next_agent_id: int | Callable[[], int]
    placement: ReproductionPlacementContext
    trophic_profile: Callable[[Agent], TrophicProfile]
    trophic_profile_for_genome: Callable[[Genome], TrophicProfile]
    trophic_role: Callable[[Agent], str]
    meat_mode: Callable[[Agent], str]
    matched_diet_ratio: Callable[[Agent, TrophicProfile], float]
    matched_diet_threshold: Callable[[TrophicProfile], float]
    health_ratio: Callable[[Agent], float]
    emit: Callable[..., None]
    signal_runtime_context: Callable[[], runtime_signals.SignalRuntimeContext]
    mind_inheritance_for_child: (
        Callable[[Agent, Agent | None, int], dict[str, object]] | None
    ) = None


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
        alive_stage_counts: dict[str, int] | None = None,
        alive_expression_counts: dict[str, int] | None = None,
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
            "alive_stage_counts": dict(alive_stage_counts or {}),
            "alive_expression_counts": dict(alive_expression_counts or {}),
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
        "stage2": {
            "stage": "stage2_proto_roles",
            "expression": [
                "proto_x_like",
                "proto_y_like",
                "proto_z_plastic",
            ],
            "birth_mode": "same_group_complementary_role_recombination",
            "fallback": "single_parent_asexual_when_no_valid_partner",
        },
        "stage3": {
            "stage": "stage3_x_y_z",
            "expression": ["x", "y", "z_plastic"],
            "birth_mode": "same_group_x_y_z_role_recombination",
            "fallback": "single_parent_asexual_when_no_valid_partner",
        },
        "future_modes": [
            "rare_gated_hybridization",
        ],
        "capability_flags": [
            "sexual_reproduction",
            "proto_role_differentiation",
            "xyz_expression",
            "hybridization",
            "multi_offspring",
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
    _promote_record_stage(record, agent.reproductive_stage)
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
    _promote_record_stage(record, parent.reproductive_stage, child.reproductive_stage)
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
    _promote_record_stage(
        record,
        primary_parent.reproductive_stage,
        secondary_parent.reproductive_stage,
        child.reproductive_stage,
    )
    record.sexual_births += 1
    if hybrid:
        record.hybrid_births += 1
    record.last_seen_tick = tick


def build_reproductive_group_catalog(
    registry: dict[int, ReproductiveGroupRecord],
    agents: Iterable[Agent],
) -> dict[str, object]:
    agent_list = list(agents)
    member_counts, alive_member_counts = _member_counts(agent_list)
    alive_stage_counts = _alive_grouped_attr_counts(
        agent_list,
        attr_name="reproductive_stage",
    )
    alive_expression_counts = _alive_grouped_attr_counts(
        agent_list,
        attr_name="reproductive_expression",
    )
    return {
        "schema_version": REPRODUCTIVE_GROUP_CONTRACT_VERSION,
        "groups": {
            str(group_id): record.to_dict(
                member_count=member_counts.get(group_id, 0),
                alive_member_count=alive_member_counts.get(group_id, 0),
                alive_stage_counts=alive_stage_counts.get(group_id, {}),
                alive_expression_counts=alive_expression_counts.get(group_id, {}),
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
    alive_stage_counts_by_group = _alive_grouped_attr_counts(
        agent_list,
        attr_name="reproductive_stage",
    )
    alive_expression_counts_by_group = _alive_grouped_attr_counts(
        agent_list,
        attr_name="reproductive_expression",
    )
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
        "alive_stage_counts": _alive_stage_counts(agent_list),
        "alive_expression_counts": _alive_expression_counts(agent_list),
        "alive_expression_counts_by_stage": _alive_expression_counts_by_stage(
            agent_list
        ),
        "top_groups": [
            record.to_dict(
                member_count=member_counts.get(record.group_id, 0),
                alive_member_count=alive_member_counts.get(record.group_id, 0),
                alive_stage_counts=alive_stage_counts_by_group.get(
                    record.group_id,
                    {},
                ),
                alive_expression_counts=alive_expression_counts_by_group.get(
                    record.group_id,
                    {},
                ),
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


def _alive_grouped_attr_counts(
    agents: Iterable[Agent],
    *,
    attr_name: str,
) -> dict[int, dict[str, int]]:
    counts_by_group: dict[int, Counter[str]] = {}
    for agent in agents:
        if not agent.alive:
            continue
        group_id = agent.reproductive_group_id or agent.lineage_id
        group_counts = counts_by_group.setdefault(group_id, Counter())
        group_counts[str(getattr(agent, attr_name))] += 1
    return {
        group_id: _sorted_count_dict(group_counts)
        for group_id, group_counts in sorted(counts_by_group.items())
    }


def _stage_counts(registry: dict[int, ReproductiveGroupRecord]) -> dict[str, int]:
    counts: Counter[str] = Counter(record.stage for record in registry.values())
    return _sorted_stage_count_dict(counts)


def _promote_record_stage(
    record: ReproductiveGroupRecord,
    *candidate_stages: str,
) -> None:
    record.stage = runtime_mating.max_reproductive_stage(
        (record.stage, *candidate_stages)
    )


def _alive_expression_counts(agents: Iterable[Agent]) -> dict[str, int]:
    counts: Counter[str] = Counter(
        agent.reproductive_expression for agent in agents if agent.alive
    )
    return _sorted_count_dict(counts)


def _alive_stage_counts(agents: Iterable[Agent]) -> dict[str, int]:
    counts: Counter[str] = Counter(
        agent.reproductive_stage for agent in agents if agent.alive
    )
    return _sorted_stage_count_dict(counts)


def _alive_expression_counts_by_stage(
    agents: Iterable[Agent],
) -> dict[str, dict[str, int]]:
    counts_by_stage: dict[str, Counter[str]] = {}
    for agent in agents:
        if not agent.alive:
            continue
        stage_counts = counts_by_stage.setdefault(agent.reproductive_stage, Counter())
        stage_counts[agent.reproductive_expression] += 1
    return {
        stage: _sorted_count_dict(expression_counts)
        for stage, expression_counts in sorted(
            counts_by_stage.items(),
            key=lambda item: (runtime_mating.stage_rank(item[0]), item[0]),
        )
    }


def _sorted_stage_count_dict(counts: Counter[str]) -> dict[str, int]:
    return {
        stage: counts[stage]
        for stage in sorted(
            counts,
            key=lambda item: (runtime_mating.stage_rank(item), item),
        )
    }


def _sorted_count_dict(counts: Counter[str]) -> dict[str, int]:
    return {key: counts[key] for key in sorted(counts)}


def empty_reproduction_blocked_counts() -> dict[str, int]:
    return {
        "max_population": 0,
        "local_crowding": 0,
        "destination_unavailable": 0,
    }


def empty_reproduction_mate_search_counts() -> dict[str, int]:
    counts = {
        "sexual_parent_candidates": 0,
        "sexual_searches": 0,
        "sexual_successes": 0,
        "scanned_agents": 0,
        "same_group_candidates": 0,
        "same_group_sexual_candidates": 0,
        "in_radius_candidates": 0,
        "biologically_ready_candidates": 0,
        "asexual_fallbacks_after_sexual_candidate": 0,
        "fallback_parent_energy_shortfall": 0,
        "fallback_no_same_group_partner": 0,
        "fallback_partner_sexual_locked": 0,
        "fallback_partner_out_of_radius": 0,
        "fallback_partner_not_ready": 0,
        "fallback_expression_incompatible": 0,
        "fallback_no_compatible_partner": 0,
        "expression_compatible_candidates": 0,
        "expression_incompatible_candidates": 0,
    }
    for reason in runtime_mating.MATE_SEARCH_BLOCK_REASON_KEYS:
        counts[f"candidate_{reason}"] = 0
        counts[f"constraint_{reason}"] = 0
    return counts


def finalize_reproduction_mate_search_counts(
    counts: dict[str, int],
) -> dict[str, int]:
    template = empty_reproduction_mate_search_counts()
    return {key: int(counts.get(key, 0)) for key in template}


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


def _resolve_reproduction_context(
    world: Any | None,
    context: ReproductionContext | None = None,
    *,
    placement_context: ReproductionPlacementContext | None = None,
) -> ReproductionContext:
    if context is not None:
        if (
            placement_context is not None
            and context.placement is not placement_context
        ):
            return replace(context, placement=placement_context)
        return context
    raise ValueError("reproduction context is required")


def _next_agent_id(context: ReproductionContext) -> int:
    source = context.next_agent_id
    if callable(source):
        return int(source())
    return int(source)


def can_reproduce(
    world: Any,
    agent: Agent,
    *,
    placement_context: ReproductionPlacementContext | None = None,
    context: ReproductionContext | None = None,
) -> bool:
    return (
        reproduction_block_reason(
            world,
            agent,
            placement_context=placement_context,
            context=context,
        )
        is None
    )


def reproduction_energy_requirement(
    world: Any | None,
    agent: Agent,
    profile: TrophicProfile,
    *,
    context: ReproductionContext | None = None,
) -> float:
    reproduction_context = _resolve_reproduction_context(world, context)
    requirement = agent.reproduction_threshold() * (
        1.0
        + profile.breadth
        * reproduction_context.config.trophic.breadth_reproduction_penalty
    )
    if profile.meat_mode != "none":
        requirement *= animal_mode_reproduction_resource_multiplier(
            agent,
            reproduction_context.config.reproduction.animal_mode_energy_requirement_multiplier,
        )
    return requirement


def recent_animal_energy_share(agent: Agent) -> float:
    animal_energy = max(0.0, agent.recent_fresh_kill_energy) + max(
        0.0,
        agent.recent_carcass_energy,
    )
    plant_energy = max(0.0, agent.recent_plant_energy)
    total_energy = animal_energy + plant_energy
    if total_energy <= ANIMAL_REPRODUCTION_DIET_EPSILON:
        return 0.0
    return max(0.0, min(1.0, animal_energy / total_energy))


def animal_mode_reproduction_resource_multiplier(
    agent: Agent,
    configured_multiplier: float,
) -> float:
    bonus_share = min(
        1.0,
        recent_animal_energy_share(agent) / ANIMAL_REPRODUCTION_FULL_BONUS_SHARE,
    )
    return 1.0 + (configured_multiplier - 1.0) * bonus_share


def biological_reproduction_block_reasons(
    world: Any | None,
    agent: Agent,
    profile: TrophicProfile | None = None,
    *,
    context: ReproductionContext | None = None,
) -> list[str]:
    reproduction_context = _resolve_reproduction_context(world, context)
    profile = profile or reproduction_context.trophic_profile(agent)
    matched_diet_ratio = reproduction_context.matched_diet_ratio(agent, profile)
    reasons: list[str] = []
    if agent.age < reproduction_context.config.reproduction.min_age:
        reasons.append("age")
    if (
        reproduction_context.tick - agent.last_reproduction_tick
        < reproduction_context.config.reproduction.cooldown_ticks
    ):
        reasons.append("cooldown")
    if agent.energy < reproduction_energy_requirement(
        world,
        agent,
        profile,
        context=reproduction_context,
    ):
        reasons.append("energy")
    if (
        agent.hydration
        < agent.genome.max_hydration
        * reproduction_context.config.reproduction.min_hydration_fraction
    ):
        reasons.append("hydration")
    health_requirement = reproduction_context.config.combat.min_reproduction_health_ratio
    if profile.meat_mode == "scavenger":
        health_requirement = min(
            health_requirement,
            reproduction_context.config.reproduction.scavenger_min_health_fraction,
        )
    if reproduction_context.health_ratio(agent) < health_requirement:
        reasons.append("health")
    if matched_diet_ratio < reproduction_context.matched_diet_threshold(profile):
        reasons.append("matched_diet")
    return reasons


def is_biologically_reproduction_ready(
    world: Any | None,
    agent: Agent,
    *,
    context: ReproductionContext | None = None,
) -> bool:
    return not biological_reproduction_block_reasons(
        world,
        agent,
        context=context,
    )


def reproduction_availability(
    world: Any | None,
    *,
    alive_agents: int | None = None,
    placement_context: ReproductionPlacementContext | None = None,
    context: ReproductionContext | None = None,
) -> ReproductionAvailability:
    reproduction_context = _resolve_reproduction_context(
        world,
        context,
        placement_context=placement_context,
    )
    placement = reproduction_context.placement
    current_alive_agents = (
        placement.current_alive_count() if alive_agents is None else alive_agents
    )
    return ReproductionAvailability(
        alive_agents=current_alive_agents,
        max_agents=placement.max_agents,
    )


def reproduction_block_reason(
    world: Any | None,
    agent: Agent,
    availability: ReproductionAvailability | None = None,
    placement_context: ReproductionPlacementContext | None = None,
    context: ReproductionContext | None = None,
) -> str | None:
    reproduction_context = _resolve_reproduction_context(
        world,
        context,
        placement_context=placement_context,
    )
    if not is_biologically_reproduction_ready(
        world,
        agent,
        context=reproduction_context,
    ):
        return "biological"
    availability = availability or reproduction_availability(
        world,
        context=reproduction_context,
    )
    if availability.population_saturated:
        return "max_population"
    if not reproduction_context.placement.has_empty_neighbor(agent.x, agent.y):
        return "local_crowding"
    return None


def is_reproduction_ready(
    world: Any | None,
    agent: Agent,
    *,
    placement_context: ReproductionPlacementContext | None = None,
    context: ReproductionContext | None = None,
) -> bool:
    return (
        reproduction_block_reason(
            world,
            agent,
            placement_context=placement_context,
            context=context,
        )
        is None
    )


def run_reproduction_phase(
    world: Any,
    *,
    context: ReproductionContext | None = None,
) -> int:
    """Run reproductive signaling and births for one tick."""

    reproduction_context = _resolve_reproduction_context(world, context)
    alive_agents = world.alive_agents()
    runtime_signals.emit_reproductive_readiness_signals(
        alive_agents,
        context=reproduction_context.signal_runtime_context(),
    )
    alive_count = len(alive_agents)
    births_this_tick = 0
    for agent_id in reproduction_phase_agent_order(world):
        agent = world.agents[agent_id]
        if not agent.alive:
            continue
        if agent.agent_id in world.tick_reproduction_parent_ids:
            continue
        availability = reproduction_availability(
            world,
            alive_agents=alive_count,
            context=reproduction_context,
        )
        block_reason = reproduction_block_reason(
            world,
            agent,
            availability,
            context=reproduction_context,
        )
        if block_reason is None:
            birth_count = reproduce_birth_count(
                world,
                agent,
                alive_count=alive_count,
                context=reproduction_context,
            )
            if birth_count > 0:
                births_this_tick += birth_count
                alive_count += birth_count
        elif block_reason != "biological":
            record_reproduction_blocked(
                world,
                agent,
                block_reason,
                alive_count=alive_count,
                context=reproduction_context,
            )
    return births_this_tick


def reproduction_phase_agent_order(world: Any) -> tuple[int, ...]:
    tick_action_order = getattr(world, "tick_action_order", None)
    if isinstance(tick_action_order, list) and tick_action_order:
        return tuple(
            int(agent_id)
            for agent_id in tick_action_order
            if int(agent_id) in world.agents
        )
    return tuple(sorted(world.agents))


def record_reproduction_blocked(
    world: Any,
    agent: Agent,
    reason: str,
    *,
    alive_count: int | None = None,
    context: ReproductionContext | None = None,
) -> None:
    reproduction_context = _resolve_reproduction_context(world, context)
    if reason not in world.run_reproduction_blocked_counts:
        raise ValueError(f"Unsupported reproduction block reason: {reason}")
    world.run_reproduction_blocked_counts[reason] += 1
    world.run_reproduction_blocked_counts_by_trophic_role[
        reproduction_context.trophic_role(agent)
    ][reason] += 1
    world.run_reproduction_blocked_counts_by_meat_mode[
        reproduction_context.meat_mode(agent)
    ][reason] += 1
    event = {
        "agent_id": agent.agent_id,
        "reason": reason,
        "x": agent.x,
        "y": agent.y,
        "alive_agents": len(world.alive_agents())
        if alive_count is None
        else alive_count,
        "max_agents": world.config.max_agents,
    }
    if world.record_tick_details:
        world.tick_reproduction_blocked_events.append(event)
    reproduction_context.emit(
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
    world: Any | None,
    alive: list[Agent],
    *,
    trophic_role_codes: dict[str, int],
    meat_mode_codes: dict[str, int],
    context: ReproductionContext | None = None,
) -> dict[str, object]:
    reproduction_context = _resolve_reproduction_context(world, context)
    population_saturated = len(alive) >= reproduction_context.config.max_agents
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
    stage_counts: Counter[str] = Counter()
    expression_counts: Counter[str] = Counter()
    capability_counts: Counter[str] = Counter(
        {
            "sexual_reproduction": 0,
            "proto_role_differentiation": 0,
            "xyz_expression": 0,
            "hybridization": 0,
            "multi_offspring": 0,
        }
    )
    biologically_ready_stage_counts: Counter[str] = Counter()
    ready_stage_counts: Counter[str] = Counter()
    biologically_ready_expression_counts: Counter[str] = Counter()
    ready_expression_counts: Counter[str] = Counter()
    biologically_ready_group_ids: set[int] = set()
    ready_group_ids: set[int] = set()
    biologically_ready = 0
    blocked_by_max_population = 0
    blocked_by_local_crowding = 0
    ready = 0

    def increment_readiness(agent: Agent, field: str) -> None:
        readiness_by_role[reproduction_context.trophic_role(agent)][field] += 1
        readiness_by_mode[reproduction_context.meat_mode(agent)][field] += 1

    def increment_blocker(agent: Agent, reason: str) -> None:
        biological_blockers[reason] += 1
        biological_blockers_by_role[reproduction_context.trophic_role(agent)][
            reason
        ] += 1
        biological_blockers_by_mode[reproduction_context.meat_mode(agent)][
            reason
        ] += 1

    def increment_energy_readiness(
        agent: Agent,
        profile: TrophicProfile,
    ) -> None:
        energy_required = reproduction_energy_requirement(
            world,
            agent,
            profile,
            context=reproduction_context,
        )
        energy_gap = max(0.0, energy_required - agent.energy)
        for counts in (
            energy_readiness_by_role[reproduction_context.trophic_role(agent)],
            energy_readiness_by_mode[reproduction_context.meat_mode(agent)],
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
        stage_counts[agent.reproductive_stage] += 1
        expression_counts[agent.reproductive_expression] += 1
        capabilities = runtime_mating.reproductive_capabilities_for_genome(
            agent.genome,
            reproduction_context.config.reproduction,
        )
        for capability, enabled in capabilities.items():
            if enabled:
                capability_counts[capability] += 1
        increment_readiness(agent, "alive_agents")
        profile = reproduction_context.trophic_profile(agent)
        increment_energy_readiness(agent, profile)
        block_reasons = biological_reproduction_block_reasons(
            world,
            agent,
            profile,
            context=reproduction_context,
        )
        if block_reasons:
            for reason in block_reasons:
                increment_blocker(agent, reason)
            continue
        biologically_ready += 1
        biologically_ready_group_ids.add(agent.reproductive_group_id or agent.lineage_id)
        biologically_ready_stage_counts[agent.reproductive_stage] += 1
        biologically_ready_expression_counts[agent.reproductive_expression] += 1
        increment_readiness(agent, "biologically_ready_agents")
        if population_saturated:
            blocked_by_max_population += 1
            increment_readiness(agent, "blocked_by_max_population_agents")
        elif not reproduction_context.placement.has_empty_neighbor(agent.x, agent.y):
            blocked_by_local_crowding += 1
            increment_readiness(agent, "blocked_by_local_crowding_agents")
        else:
            ready += 1
            ready_group_ids.add(agent.reproductive_group_id or agent.lineage_id)
            ready_stage_counts[agent.reproductive_stage] += 1
            ready_expression_counts[agent.reproductive_expression] += 1
            increment_readiness(agent, "ready_agents")

    return {
        "max_agents": reproduction_context.config.max_agents,
        "saturation_at_end": round(
            len(alive) / max(reproduction_context.config.max_agents, 1),
            4,
        ),
        "peak_saturation": round(
            world.peak_alive_agents / max(reproduction_context.config.max_agents, 1),
            4,
        ),
        "biologically_ready_agents": biologically_ready,
        "ready_agents": ready,
        "blocked_by_max_population_agents": blocked_by_max_population,
        "blocked_by_local_crowding_agents": blocked_by_local_crowding,
        "biologically_ready_group_count": len(biologically_ready_group_ids),
        "ready_group_count": len(ready_group_ids),
        "reproductive_stage_counts": _sorted_stage_count_dict(stage_counts),
        "reproductive_expression_counts": _sorted_count_dict(expression_counts),
        "reproductive_capability_counts": {
            key: int(capability_counts[key]) for key in sorted(capability_counts)
        },
        "biologically_ready_by_reproductive_stage": _sorted_stage_count_dict(
            biologically_ready_stage_counts
        ),
        "ready_by_reproductive_stage": _sorted_stage_count_dict(ready_stage_counts),
        "biologically_ready_by_reproductive_expression": _sorted_count_dict(
            biologically_ready_expression_counts
        ),
        "ready_by_reproductive_expression": _sorted_count_dict(
            ready_expression_counts
        ),
        "blocked_run_counts": dict(world.run_reproduction_blocked_counts),
        "mate_search_run_counts": finalize_reproduction_mate_search_counts(
            getattr(
                world,
                "run_reproduction_mate_search_counts",
                empty_reproduction_mate_search_counts(),
            )
        ),
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


def build_frame_reproduction_stats(
    world: Any,
    alive: list[Agent],
    *,
    trophic_role_codes: dict[str, int],
    meat_mode_codes: dict[str, int],
    context: ReproductionContext | None = None,
) -> dict[str, object]:
    reproduction_context = _resolve_reproduction_context(world, context)
    stats = reproduction_readiness_counts(
        world,
        alive,
        trophic_role_codes=trophic_role_codes,
        meat_mode_codes=meat_mode_codes,
        context=reproduction_context,
    )
    blocked_this_tick = empty_reproduction_blocked_counts()
    for event in world.tick_reproduction_blocked_events:
        blocked_this_tick[str(event["reason"])] += 1
    stats["blocked_this_tick"] = blocked_this_tick
    stats["mate_search_this_tick"] = finalize_reproduction_mate_search_counts(
        world.tick_reproduction_mate_search_counts
    )
    stats["mate_search_events"] = list(world.tick_reproduction_mate_search_events)
    return stats


def animal_mode_stabilized_child_genome(
    world: Any | None,
    parent_genome: Genome,
    child_genome: Genome,
    parent_profile: TrophicProfile,
    *,
    context: ReproductionContext | None = None,
) -> Genome:
    if parent_profile.meat_mode == "none":
        return child_genome
    reproduction_context = _resolve_reproduction_context(world, context)

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
        reproduction_context.config.reproduction.animal_mode_offspring_trait_stability
    )
    stabilized = stabilize(configured_stability)
    stabilized_mode = reproduction_context.trophic_profile_for_genome(
        stabilized
    ).meat_mode
    if stabilized_mode == parent_profile.meat_mode or (
        parent_profile.meat_mode == "mixed" and stabilized_mode != "none"
    ):
        return stabilized
    return stabilize(1.0)


def sexual_child_stabilized_genome(
    world: Any | None,
    primary_parent_genome: Genome,
    secondary_parent_genome: Genome,
    child_genome: Genome,
    primary_parent_profile: TrophicProfile,
    secondary_parent_profile: TrophicProfile,
    *,
    context: ReproductionContext | None = None,
) -> Genome:
    stabilization_profile = _sexual_child_stabilization_profile(
        primary_parent_profile,
        secondary_parent_profile,
    )
    if stabilization_profile is None:
        return child_genome
    anchor_genome = _blended_parent_genome(
        primary_parent_genome,
        secondary_parent_genome,
    )
    return animal_mode_stabilized_child_genome(
        world,
        anchor_genome,
        child_genome,
        stabilization_profile,
        context=context,
    )


def _sexual_child_stabilization_profile(
    primary_parent_profile: TrophicProfile,
    secondary_parent_profile: TrophicProfile,
) -> TrophicProfile | None:
    primary_mode = primary_parent_profile.meat_mode
    secondary_mode = secondary_parent_profile.meat_mode
    animal_modes = {primary_mode, secondary_mode} - {"none"}
    if not animal_modes:
        return None
    if len(animal_modes) == 1:
        mode = next(iter(animal_modes))
    else:
        mode = "mixed"
    if primary_mode == mode:
        return primary_parent_profile
    if secondary_mode == mode:
        return secondary_parent_profile
    return replace(primary_parent_profile, meat_mode=mode)


def child_starting_fraction(
    base_fraction: float,
    multiplier: float,
    parent: Agent | None,
    parent_profile: TrophicProfile,
) -> float:
    if parent_profile.meat_mode == "none" or parent is None:
        return base_fraction
    effective_multiplier = animal_mode_reproduction_resource_multiplier(
        parent,
        multiplier,
    )
    return min(1.0, base_fraction * effective_multiplier)


def sexual_child_starting_fraction(
    base_fraction: float,
    multiplier: float,
    primary_parent: Agent | None,
    primary_parent_profile: TrophicProfile,
    secondary_parent: Agent | None,
    secondary_parent_profile: TrophicProfile,
) -> float:
    primary_fraction = child_starting_fraction(
        base_fraction,
        multiplier,
        primary_parent,
        primary_parent_profile,
    )
    secondary_fraction = child_starting_fraction(
        base_fraction,
        multiplier,
        secondary_parent,
        secondary_parent_profile,
    )
    return min(1.0, (primary_fraction + secondary_fraction) / 2.0)


def reproduction_energy_cost(
    world: Any | None,
    parent: Agent | None,
    parent_profile: TrophicProfile,
    *,
    context: ReproductionContext | None = None,
) -> float:
    reproduction_context = _resolve_reproduction_context(world, context)
    if parent_profile.meat_mode == "none" or parent is None:
        return reproduction_context.config.reproduction.energy_cost
    return (
        reproduction_context.config.reproduction.energy_cost
        * animal_mode_reproduction_resource_multiplier(
            parent,
            reproduction_context.config.reproduction.animal_mode_reproduction_cost_multiplier,
        )
    )


def sexual_reproduction_energy_cost(
    world: Any | None,
    parent: Agent | None,
    parent_profile: TrophicProfile,
    *,
    context: ReproductionContext | None = None,
) -> float:
    reproduction_context = _resolve_reproduction_context(world, context)
    return (
        reproduction_energy_cost(
            world,
            parent,
            parent_profile,
            context=reproduction_context,
        )
        * reproduction_context.config.reproduction.sexual_parent_cost_multiplier
    )


def reproductive_state_for_child(
    world: Any | None,
    parent: Agent,
    child_genome: Genome,
    *,
    context: ReproductionContext | None = None,
) -> ReproductiveState:
    reproduction_context = _resolve_reproduction_context(world, context)
    return child_reproductive_state(
        group_id=parent.reproductive_group_id or parent.lineage_id,
        stage=runtime_mating.reproductive_stage_for_genome(
            child_genome,
            reproduction_context.config.reproduction,
        ),
        expression=runtime_mating.reproductive_expression_for_genome(
            child_genome,
            reproduction_context.config.reproduction,
        ),
    )


def sexual_reproductive_state_for_child(
    world: Any | None,
    primary_parent: Agent,
    secondary_parent: Agent,
    child_genome: Genome,
    *,
    context: ReproductionContext | None = None,
) -> ReproductiveState:
    reproduction_context = _resolve_reproduction_context(world, context)
    primary_group_id = primary_parent.reproductive_group_id or primary_parent.lineage_id
    secondary_group_id = (
        secondary_parent.reproductive_group_id or secondary_parent.lineage_id
    )
    if primary_group_id != secondary_group_id:
        raise ValueError(
            "sexual child reproductive state requires same reproductive group"
        )
    return child_reproductive_state(
        group_id=primary_group_id,
        stage=runtime_mating.reproductive_stage_for_genome(
            child_genome,
            reproduction_context.config.reproduction,
        ),
        expression=runtime_mating.reproductive_expression_for_genome(
            child_genome,
            reproduction_context.config.reproduction,
        ),
    )


def sexual_child_lineage_id(
    primary_parent: Agent,
    secondary_parent: Agent,
    *,
    child_agent_id: int,
) -> int:
    if primary_parent.lineage_id == secondary_parent.lineage_id:
        return primary_parent.lineage_id
    return child_agent_id


def sexual_partner_ready(
    world: Any | None,
    agent: Agent,
    *,
    context: ReproductionContext | None = None,
) -> bool:
    reproduction_context = _resolve_reproduction_context(world, context)
    if (
        world is not None
        and agent.agent_id in getattr(world, "tick_reproduction_parent_ids", set())
    ):
        return False
    if not is_biologically_reproduction_ready(
        world,
        agent,
        context=reproduction_context,
    ):
        return False
    return agent.energy >= sexual_reproduction_energy_cost(
        world,
        agent,
        reproduction_context.trophic_profile(agent),
        context=reproduction_context,
    )


def record_sexual_parent_energy_fallback(world: Any, parent: Agent) -> None:
    _increment_mate_search_counts(
        world,
        {
            "sexual_parent_candidates": 1,
            "asexual_fallbacks_after_sexual_candidate": 1,
            "fallback_parent_energy_shortfall": 1,
        },
    )
    _record_mate_search_event(
        world,
        parent,
        selected_partner_id=None,
        fallback_reason="parent_energy_shortfall",
        reason_counts={
            reason: 0 for reason in runtime_mating.MATE_SEARCH_BLOCK_REASON_KEYS
        },
        constraint_counts={
            reason: 0 for reason in runtime_mating.MATE_SEARCH_BLOCK_REASON_KEYS
        },
        scanned_agents=0,
        same_group_candidates=0,
        same_group_sexual_candidates=0,
        in_radius_candidates=0,
        biologically_ready_candidates=0,
        expression_compatible_candidates=0,
        expression_incompatible_candidates=0,
    )


def record_mate_search_report(
    world: Any,
    parent: Agent,
    report: runtime_mating.MateSearchReport,
) -> None:
    selected_partner_id = (
        report.selected.agent.agent_id if report.selected is not None else None
    )
    fallback_reason = (
        None
        if report.selected is not None
        else _mate_search_fallback_reason(report)
    )
    increments = {
        "sexual_parent_candidates": 1,
        "sexual_searches": 1,
        "scanned_agents": report.scanned_agents,
        "same_group_candidates": report.same_group_candidates,
        "same_group_sexual_candidates": report.same_group_sexual_candidates,
        "in_radius_candidates": report.in_radius_candidates,
        "biologically_ready_candidates": report.biologically_ready_candidates,
        "expression_compatible_candidates": report.expression_compatible_candidates,
        "expression_incompatible_candidates": report.expression_incompatible_candidates,
    }
    for reason, count in report.reason_counts.items():
        increments[f"candidate_{reason}"] = count
    for reason, count in report.constraint_counts.items():
        increments[f"constraint_{reason}"] = count
    if report.selected is not None:
        increments["sexual_successes"] = 1
    else:
        increments["asexual_fallbacks_after_sexual_candidate"] = 1
        increments[f"fallback_{fallback_reason}"] = 1
    _increment_mate_search_counts(world, increments)
    _record_mate_search_event(
        world,
        parent,
        selected_partner_id=selected_partner_id,
        fallback_reason=fallback_reason,
        reason_counts=report.reason_counts,
        constraint_counts=report.constraint_counts,
        scanned_agents=report.scanned_agents,
        same_group_candidates=report.same_group_candidates,
        same_group_sexual_candidates=report.same_group_sexual_candidates,
        in_radius_candidates=report.in_radius_candidates,
        biologically_ready_candidates=report.biologically_ready_candidates,
        expression_compatible_candidates=report.expression_compatible_candidates,
        expression_incompatible_candidates=report.expression_incompatible_candidates,
    )


def _increment_mate_search_counts(
    world: Any,
    increments: dict[str, int],
) -> None:
    for key, value in increments.items():
        if key not in world.tick_reproduction_mate_search_counts:
            raise ValueError(f"Unsupported mate search count: {key}")
        world.tick_reproduction_mate_search_counts[key] += int(value)
        world.run_reproduction_mate_search_counts[key] += int(value)


def _record_mate_search_event(
    world: Any,
    parent: Agent,
    *,
    selected_partner_id: int | None,
    fallback_reason: str | None,
    reason_counts: dict[str, int],
    constraint_counts: dict[str, int],
    scanned_agents: int,
    same_group_candidates: int,
    same_group_sexual_candidates: int,
    in_radius_candidates: int,
    biologically_ready_candidates: int,
    expression_compatible_candidates: int,
    expression_incompatible_candidates: int,
) -> None:
    if not world.record_tick_details:
        return
    world.tick_reproduction_mate_search_events.append(
        {
            "agent_id": parent.agent_id,
            "reproductive_group_id": parent.reproductive_group_id
            or parent.lineage_id,
            "reproductive_stage": parent.reproductive_stage,
            "reproductive_expression": parent.reproductive_expression,
            "selected_partner_id": selected_partner_id,
            "fallback_reason": fallback_reason,
            "scanned_agents": scanned_agents,
            "same_group_candidates": same_group_candidates,
            "same_group_sexual_candidates": same_group_sexual_candidates,
            "in_radius_candidates": in_radius_candidates,
            "biologically_ready_candidates": biologically_ready_candidates,
            "reason_counts": {
                reason: int(reason_counts.get(reason, 0))
                for reason in runtime_mating.MATE_SEARCH_BLOCK_REASON_KEYS
            },
            "constraint_counts": {
                reason: int(constraint_counts.get(reason, 0))
                for reason in runtime_mating.MATE_SEARCH_BLOCK_REASON_KEYS
            },
            "expression_compatible_candidates": expression_compatible_candidates,
            "expression_incompatible_candidates": expression_incompatible_candidates,
        }
    )


def _mate_search_fallback_reason(
    report: runtime_mating.MateSearchReport,
) -> str:
    if report.same_group_candidates == 0:
        return "no_same_group_partner"
    reason_counts = report.reason_counts
    if (
        reason_counts["expression_incompatible"] > 0
        and report.expression_compatible_candidates == 0
    ):
        return "expression_incompatible"
    for reason in (
        "partner_not_ready",
        "partner_out_of_radius",
        "partner_sexual_locked",
    ):
        if reason_counts[reason] > 0:
            return reason
    return "no_compatible_partner"


def reproduce(
    world: Any,
    parent: Agent,
    *,
    alive_count: int | None = None,
    placement_context: ReproductionPlacementContext | None = None,
    context: ReproductionContext | None = None,
) -> bool:
    return (
        reproduce_birth_count(
            world,
            parent,
            alive_count=alive_count,
            placement_context=placement_context,
            context=context,
        )
        > 0
    )


def reproduce_birth_count(
    world: Any,
    parent: Agent,
    *,
    alive_count: int | None = None,
    placement_context: ReproductionPlacementContext | None = None,
    context: ReproductionContext | None = None,
) -> int:
    reproduction_context = _resolve_reproduction_context(
        world,
        context,
        placement_context=placement_context,
    )
    if parent.agent_id in getattr(world, "tick_reproduction_parent_ids", set()):
        return 0
    placement = reproduction_context.placement
    destination = placement.find_empty_neighbor(parent.x, parent.y)
    if destination is None:
        record_reproduction_blocked(
            world,
            parent,
            "destination_unavailable",
            alive_count=alive_count,
            context=reproduction_context,
        )
        return 0

    parent_profile = reproduction_context.trophic_profile(parent)
    if runtime_mating.sexual_reproduction_unlocked(
        parent.genome,
        reproduction_context.config.reproduction,
    ):
        if parent.energy >= sexual_reproduction_energy_cost(
            world,
            parent,
            parent_profile,
            context=reproduction_context,
        ):
            mate_report = runtime_mating.same_group_mate_search_report(
                parent,
                world.agents.values(),
                config=reproduction_context.config.reproduction,
                biologically_ready=lambda agent: sexual_partner_ready(
                    world,
                    agent,
                    context=reproduction_context,
                ),
            )
            record_mate_search_report(world, parent, mate_report)
            if mate_report.selected is not None:
                mate_candidate = mate_report.selected
                return _reproduce_sexual_birth_count(
                    world,
                    parent,
                    mate_candidate,
                    destination,
                    parent_profile,
                    alive_count=alive_count,
                    context=reproduction_context,
                )
        else:
            record_sexual_parent_energy_fallback(world, parent)
    return (
        1
        if reproduce_asexual(
            world,
            parent,
            destination,
            parent_profile,
            context=reproduction_context,
        )
        else 0
    )


def reproduce_asexual(
    world: Any,
    parent: Agent,
    destination: tuple[int, int],
    parent_profile: TrophicProfile,
    *,
    placement_context: ReproductionPlacementContext | None = None,
    context: ReproductionContext | None = None,
) -> bool:
    reproduction_context = _resolve_reproduction_context(
        world,
        context,
        placement_context=placement_context,
    )
    birth_plan = build_asexual_birth_plan(
        world,
        parent,
        destination,
        parent_profile,
        context=reproduction_context,
    )
    return apply_birth_plan(world, birth_plan, context=reproduction_context) > 0


def build_asexual_birth_plan(
    world: Any | None,
    parent: Agent,
    destination: tuple[int, int],
    parent_profile: TrophicProfile,
    *,
    context: ReproductionContext | None = None,
) -> BirthPlan:
    child_plan = build_asexual_child_birth_plan(
        world,
        parent,
        destination,
        parent_profile,
        context=context,
    )
    return BirthPlan(
        children=child_plan.children,
        parents=(parent,),
        parent_energy_costs=child_plan.parent_energy_costs,
        reproduction_mode=runtime_mating.ASEXUAL_REPRODUCTION_MODE,
    )


def build_asexual_child_birth_plan(
    world: Any | None,
    parent: Agent,
    destination: tuple[int, int],
    parent_profile: TrophicProfile,
    *,
    context: ReproductionContext | None = None,
) -> ChildBirthPlan:
    reproduction_context = _resolve_reproduction_context(world, context)
    child_genome = animal_mode_stabilized_child_genome(
        world,
        parent.genome,
        parent.genome.mutate(reproduction_context.rng),
        parent_profile,
        context=reproduction_context,
    )
    child_energy_fraction = child_starting_fraction(
        reproduction_context.config.reproduction.child_energy_fraction,
        reproduction_context.config.reproduction.animal_mode_child_energy_fraction_multiplier,
        parent,
        parent_profile,
    )
    child_hydration_fraction = child_starting_fraction(
        reproduction_context.config.reproduction.child_hydration_fraction,
        reproduction_context.config.reproduction.animal_mode_child_hydration_fraction_multiplier,
        parent,
        parent_profile,
    )
    reproductive_state = reproductive_state_for_child(
        world,
        parent,
        child_genome,
        context=reproduction_context,
    )
    child_agent_id = _next_agent_id(reproduction_context)
    child = build_child_agent(
        agent_id=child_agent_id,
        primary_parent=parent,
        secondary_parent=None,
        lineage_id=parent.lineage_id,
        birth_tick=reproduction_context.tick,
        destination=destination,
        genome=child_genome,
        energy_fraction=child_energy_fraction,
        hydration_fraction=child_hydration_fraction,
        health_fraction=reproduction_context.config.reproduction.child_health_fraction,
        reproductive_state=reproductive_state,
        mind_inheritance_metadata=(
            reproduction_context.mind_inheritance_for_child(
                parent,
                None,
                child_agent_id,
            )
            if reproduction_context.mind_inheritance_for_child is not None
            else empty_mind_inheritance_metadata()
        ),
    )
    parent_cost = reproduction_energy_cost(
        world,
        parent,
        parent_profile,
        context=reproduction_context,
    )
    return ChildBirthPlan(
        children=(child,),
        parent_energy_costs=(parent_cost,),
    )


def reproduce_sexual(
    world: Any,
    parent: Agent,
    mate_candidate: runtime_mating.MateCandidate,
    destination: tuple[int, int],
    parent_profile: TrophicProfile,
    *,
    placement_context: ReproductionPlacementContext | None = None,
    context: ReproductionContext | None = None,
) -> bool:
    return (
        _reproduce_sexual_birth_count(
            world,
            parent,
            mate_candidate,
            destination,
            parent_profile,
            placement_context=placement_context,
            context=context,
        )
        > 0
    )


def _reproduce_sexual_birth_count(
    world: Any,
    parent: Agent,
    mate_candidate: runtime_mating.MateCandidate,
    destination: tuple[int, int],
    parent_profile: TrophicProfile,
    *,
    alive_count: int | None = None,
    placement_context: ReproductionPlacementContext | None = None,
    context: ReproductionContext | None = None,
) -> int:
    reproduction_context = _resolve_reproduction_context(
        world,
        context,
        placement_context=placement_context,
    )
    placement = reproduction_context.placement
    partner = mate_candidate.agent
    partner_profile = reproduction_context.trophic_profile(partner)
    parent_cost = sexual_reproduction_energy_cost(
        world,
        parent,
        parent_profile,
        context=reproduction_context,
    )
    partner_cost = sexual_reproduction_energy_cost(
        world,
        partner,
        partner_profile,
        context=reproduction_context,
    )
    offspring_plan = _sexual_offspring_plan(
        world,
        parent,
        partner,
        parent_cost=parent_cost,
        partner_cost=partner_cost,
        alive_count=alive_count,
        context=reproduction_context,
    )
    if offspring_plan.count <= 0:
        return 0
    destinations = _sexual_offspring_destinations(
        placement,
        parent,
        first_destination=destination,
        requested_count=offspring_plan.count,
    )
    offspring_count = len(destinations)
    if offspring_count <= 0:
        record_reproduction_blocked(
            world,
            parent,
            "destination_unavailable",
            alive_count=alive_count,
            context=reproduction_context,
        )
        return 0
    multi_offspring_limit_reasons = list(offspring_plan.limit_reasons)
    if offspring_count < offspring_plan.count:
        multi_offspring_limit_reasons.append("local_destination_capacity")
    birth_plan = build_sexual_birth_plan(
        world,
        parent,
        partner,
        destinations,
        parent_profile,
        partner_profile,
        mate_candidate,
        parent_cost=parent_cost,
        partner_cost=partner_cost,
        multi_offspring_desired_count=offspring_plan.desired_count,
        multi_offspring_limit_reasons=tuple(multi_offspring_limit_reasons),
        context=reproduction_context,
    )
    return apply_birth_plan(world, birth_plan, context=reproduction_context)


def build_sexual_birth_plan(
    world: Any | None,
    parent: Agent,
    partner: Agent,
    destinations: list[tuple[int, int]],
    parent_profile: TrophicProfile,
    partner_profile: TrophicProfile,
    mate_candidate: runtime_mating.MateCandidate,
    *,
    parent_cost: float,
    partner_cost: float,
    multi_offspring_desired_count: int,
    multi_offspring_limit_reasons: tuple[str, ...],
    context: ReproductionContext | None = None,
) -> BirthPlan:
    child_plan = build_sexual_child_birth_plan(
        world,
        parent,
        partner,
        destinations,
        parent_profile,
        partner_profile,
        mate_candidate,
        parent_cost=parent_cost,
        partner_cost=partner_cost,
        context=context,
    )
    return BirthPlan(
        children=child_plan.children,
        parents=(parent, partner),
        parent_energy_costs=child_plan.parent_energy_costs,
        reproduction_mode=runtime_mating.SEXUAL_REPRODUCTION_MODE,
        mate_candidate=mate_candidate,
        multi_offspring_desired_count=multi_offspring_desired_count,
        multi_offspring_limit_reasons=multi_offspring_limit_reasons,
    )


def apply_birth_plan(
    world: Any,
    birth_plan: BirthPlan,
    *,
    context: ReproductionContext | None = None,
) -> int:
    reproduction_context = _resolve_reproduction_context(world, context)
    offspring_count = birth_plan.offspring_count
    if offspring_count <= 0:
        return 0
    parent_ids = tuple(parent.agent_id for parent in birth_plan.parents)
    total_energy_cost = 0.0
    for index, parent in enumerate(birth_plan.parents):
        parent_cost = birth_plan.parent_energy_costs[index]
        parent.energy -= parent_cost * offspring_count
        total_energy_cost += parent_cost * offspring_count
        parent.last_reproduction_tick = reproduction_context.tick
    runtime_resources.record_energy_spent(
        world,
        "reproduction",
        total_energy_cost,
    )

    for index, child in enumerate(birth_plan.children):
        reproduction_context.placement.place_agent(child)
        if birth_plan.reproduction_mode == runtime_mating.ASEXUAL_REPRODUCTION_MODE:
            record_asexual_birth(
                world.reproductive_groups,
                birth_plan.parents[0],
                child,
                tick=reproduction_context.tick,
            )
        elif birth_plan.reproduction_mode == runtime_mating.SEXUAL_REPRODUCTION_MODE:
            record_sexual_birth(
                world.reproductive_groups,
                birth_plan.parents[0],
                birth_plan.parents[1],
                child,
                tick=reproduction_context.tick,
            )
        else:
            raise ValueError(
                f"Unsupported reproduction mode: {birth_plan.reproduction_mode}"
            )
        world.tick_reproduction_parent_child_groups.append(
            (parent_ids, child.agent_id)
        )
        if world.record_tick_details:
            world.tick_birth_pairs.append(
                (parent_ids[0], child.agent_id)
            )
        emit_reproduction_event(
            world,
            actor=birth_plan.parents[0],
            child=child,
            parents=birth_plan.parents,
            reproduction_mode=birth_plan.reproduction_mode,
            parent_energy_costs=birth_plan.parent_energy_costs,
            mate_candidate=birth_plan.mate_candidate,
            offspring_count=offspring_count,
            offspring_index=index + 1 if offspring_count > 1 else None,
            sibling_child_ids=(
                birth_plan.sibling_child_ids if offspring_count > 1 else None
            ),
            multi_offspring_desired_count=(
                birth_plan.multi_offspring_desired_count
            ),
            multi_offspring_limit_reasons=(
                birth_plan.multi_offspring_limit_reasons
            ),
            context=reproduction_context,
        )
    world.tick_reproduction_parent_ids.update(parent_ids)
    world.next_agent_id += offspring_count
    world.births += offspring_count
    world.last_birth_tick = reproduction_context.tick
    reproduction_context.placement.invalidate_spatial_state()
    return offspring_count


def build_sexual_child_birth_plan(
    world: Any | None,
    parent: Agent,
    partner: Agent,
    destinations: list[tuple[int, int]],
    parent_profile: TrophicProfile,
    partner_profile: TrophicProfile,
    mate_candidate: runtime_mating.MateCandidate,
    *,
    parent_cost: float,
    partner_cost: float,
    context: ReproductionContext | None = None,
) -> ChildBirthPlan:
    reproduction_context = _resolve_reproduction_context(world, context)
    child_energy_fraction = sexual_child_starting_fraction(
        reproduction_context.config.reproduction.child_energy_fraction,
        reproduction_context.config.reproduction.animal_mode_child_energy_fraction_multiplier,
        parent,
        parent_profile,
        partner,
        partner_profile,
    )
    child_hydration_fraction = sexual_child_starting_fraction(
        reproduction_context.config.reproduction.child_hydration_fraction,
        reproduction_context.config.reproduction.animal_mode_child_hydration_fraction_multiplier,
        parent,
        parent_profile,
        partner,
        partner_profile,
    )
    children: list[Agent] = []
    next_agent_id = _next_agent_id(reproduction_context)
    for index, child_destination in enumerate(destinations):
        child_genome = sexual_child_stabilized_genome(
            world,
            parent.genome,
            partner.genome,
            recombine_genomes(
                parent.genome,
                partner.genome,
                reproduction_context.rng,
            ).mutate(reproduction_context.rng),
            parent_profile,
            partner_profile,
            context=reproduction_context,
        )
        child_genome = apply_inbreeding_penalty(
            child_genome,
            penalty=mate_candidate.inbreeding_penalty,
            scale=reproduction_context.config.reproduction.sexual_inbreeding_gene_penalty,
        )
        child_agent_id = next_agent_id + index
        child_lineage_id = sexual_child_lineage_id(
            parent,
            partner,
            child_agent_id=child_agent_id,
        )
        reproductive_state = sexual_reproductive_state_for_child(
            world,
            parent,
            partner,
            child_genome,
            context=reproduction_context,
        )
        children.append(
            build_child_agent(
                agent_id=child_agent_id,
                primary_parent=parent,
                secondary_parent=partner,
                lineage_id=child_lineage_id,
                birth_tick=reproduction_context.tick,
                destination=child_destination,
                genome=child_genome,
                energy_fraction=child_energy_fraction,
                hydration_fraction=child_hydration_fraction,
                health_fraction=(
                    reproduction_context.config.reproduction.child_health_fraction
                ),
                reproductive_state=reproductive_state,
                mind_inheritance_metadata=(
                    reproduction_context.mind_inheritance_for_child(
                        parent,
                        partner,
                        child_agent_id,
                    )
                    if reproduction_context.mind_inheritance_for_child is not None
                    else empty_mind_inheritance_metadata()
                ),
            )
        )

    return ChildBirthPlan(
        children=tuple(children),
        parent_energy_costs=(parent_cost, partner_cost),
    )


def _sexual_offspring_plan(
    world: Any | None,
    parent: Agent,
    partner: Agent,
    *,
    parent_cost: float,
    partner_cost: float,
    alive_count: int | None,
    placement_context: ReproductionPlacementContext | None = None,
    context: ReproductionContext | None = None,
) -> SexualOffspringPlan:
    reproduction_context = _resolve_reproduction_context(
        world,
        context,
        placement_context=placement_context,
    )
    placement = reproduction_context.placement
    desired_count = runtime_mating.multi_offspring_count_for_pair(
        parent,
        partner,
        reproduction_context.config.reproduction,
    )
    current_alive = placement.current_alive_count() if alive_count is None else alive_count
    population_slots = max(0, int(placement.max_agents) - current_alive)
    if population_slots <= 0:
        return SexualOffspringPlan(
            desired_count=desired_count,
            count=0,
            limit_reasons=("population_capacity",),
        )
    parent_affordable = _affordable_offspring_count(
        parent.energy,
        parent_cost,
        desired_count,
    )
    partner_affordable = _affordable_offspring_count(
        partner.energy,
        partner_cost,
        desired_count,
    )
    count = max(
        0,
        min(
            desired_count,
            population_slots,
            parent_affordable,
            partner_affordable,
        ),
    )
    limit_reasons: list[str] = []
    if population_slots < desired_count:
        limit_reasons.append("population_capacity")
    if parent_affordable < desired_count:
        limit_reasons.append("parent_energy")
    if partner_affordable < desired_count:
        limit_reasons.append("partner_energy")
    return SexualOffspringPlan(
        desired_count=desired_count,
        count=count,
        limit_reasons=tuple(limit_reasons),
    )


def _affordable_offspring_count(
    energy: float,
    cost: float,
    desired_count: int,
) -> int:
    if cost <= 0:
        return desired_count
    return max(0, min(desired_count, math.floor((energy + 1e-12) / cost)))


def _sexual_offspring_destinations(
    placement_context: ReproductionPlacementContext,
    parent: Agent,
    *,
    first_destination: tuple[int, int],
    requested_count: int,
) -> list[tuple[int, int]]:
    if requested_count <= 0:
        return []
    destinations = [first_destination]
    if requested_count <= 1:
        return destinations
    for candidate in placement_context.sibling_destination_candidates(
        parent.x,
        parent.y,
    ):
        if candidate in destinations:
            continue
        if placement_context.can_place_at(*candidate):
            destinations.append(candidate)
            if len(destinations) >= requested_count:
                break
    return destinations


def emit_reproduction_event(
    world: Any | None,
    *,
    actor: Agent,
    child: Agent,
    parents: tuple[Agent, ...],
    reproduction_mode: str,
    parent_energy_costs: tuple[float, ...],
    mate_candidate: runtime_mating.MateCandidate | None = None,
    offspring_count: int = 1,
    offspring_index: int | None = None,
    sibling_child_ids: list[int] | None = None,
    multi_offspring_desired_count: int = 1,
    multi_offspring_limit_reasons: tuple[str, ...] = (),
    context: ReproductionContext | None = None,
) -> None:
    reproduction_context = _resolve_reproduction_context(world, context)
    reproduction_context.emit(
        EventType.AGENT_REPRODUCED,
        agent_id=actor.agent_id,
        data=build_reproduction_event_payload(
            child=child,
            parents=parents,
            reproduction_mode=reproduction_mode,
            parent_energy_costs=parent_energy_costs,
            mate_candidate=mate_candidate,
            offspring_count=offspring_count,
            offspring_index=offspring_index,
            sibling_child_ids=sibling_child_ids,
            multi_offspring_desired_count=multi_offspring_desired_count,
            multi_offspring_limit_reasons=multi_offspring_limit_reasons,
        ),
    )


def build_reproduction_event_payload(
    *,
    child: Agent,
    parents: tuple[Agent, ...],
    reproduction_mode: str,
    parent_energy_costs: tuple[float, ...],
    mate_candidate: runtime_mating.MateCandidate | None = None,
    offspring_count: int = 1,
    offspring_index: int | None = None,
    sibling_child_ids: list[int] | None = None,
    multi_offspring_desired_count: int = 1,
    multi_offspring_limit_reasons: tuple[str, ...] = (),
) -> dict[str, object]:
    parent_ids = [parent.agent_id for parent in parents]
    parent_group_ids = [
        parent.reproductive_group_id or parent.lineage_id for parent in parents
    ]
    payload: dict[str, object] = {
        "schema_version": REPRODUCTION_EVENT_SCHEMA_VERSION,
        "child_id": child.agent_id,
        "child_x": child.x,
        "child_y": child.y,
        "lineage_id": child.lineage_id,
        "child_lineage_id": child.lineage_id,
        "child_reproductive_group_id": child.reproductive_group_id,
        "child_reproductive_stage": child.reproductive_stage,
        "child_reproductive_expression": child.reproductive_expression,
        "birth_tick": child.birth_tick,
        "reproduction_mode": reproduction_mode,
        "parent_ids": parent_ids,
        "parent_lineage_ids": [parent.lineage_id for parent in parents],
        "parent_reproductive_group_ids": parent_group_ids,
        "parent_reproductive_stages": [
            parent.reproductive_stage for parent in parents
        ],
        "parent_reproductive_expressions": [
            parent.reproductive_expression for parent in parents
        ],
        "parent_energy_costs": [
            {
                "agent_id": parent.agent_id,
                "energy_cost": round(float(parent_energy_costs[index]), 4),
            }
            for index, parent in enumerate(parents)
        ],
        "offspring_count": offspring_count,
        "compatibility_score": None,
        "inbreeding_penalty": None,
        "mate_distance": None,
        "outbreeding_distance_score": None,
        "hybrid": False,
        "mind_inheritance": dict(child.mind_inheritance_metadata),
    }
    if len(parents) > 1:
        payload["partner_id"] = parents[1].agent_id
    if multi_offspring_desired_count > 1:
        payload["multi_offspring_desired_count"] = multi_offspring_desired_count
        payload["multi_offspring_actual_count"] = offspring_count
        payload["multi_offspring_limit_reasons"] = list(multi_offspring_limit_reasons)
    if offspring_count > 1:
        payload["offspring_index"] = offspring_index
        payload["sibling_child_ids"] = list(sibling_child_ids or [])
        payload["multi_offspring"] = True
        payload["parent_energy_costs_total"] = [
            {
                "agent_id": parent.agent_id,
                "energy_cost": round(
                    float(parent_energy_costs[index]) * offspring_count,
                    4,
                ),
            }
            for index, parent in enumerate(parents)
        ]
    if mate_candidate is not None:
        payload.update(
            {
                "compatibility_score": mate_candidate.compatibility_score,
                "inbreeding_penalty": mate_candidate.inbreeding_penalty,
                "mate_distance": mate_candidate.distance,
                "outbreeding_distance_score": None,
            }
        )
    return payload


def _clamp_gene_value(name: str, value: float) -> float:
    lower, upper = GENE_LIMITS[name]
    return max(lower, min(upper, value))


def _blended_parent_genome(left: Genome, right: Genome) -> Genome:
    reproductive = ReproductiveGenome(
        **{
            name: (
                getattr(left.reproductive, name)
                + getattr(right.reproductive, name)
            )
            / 2.0
            for name in REPRODUCTIVE_GENE_LIMITS
        }
    )
    return Genome(
        **{
            name: _clamp_gene_value(
                name,
                (getattr(left, name) + getattr(right, name)) / 2.0,
            )
            for name in GENE_LIMITS
        },
        reproductive=reproductive,
    )


def _blend_gene(child_value: float, parent_value: float, stability: float) -> float:
    return child_value * (1.0 - stability) + parent_value * stability
