from __future__ import annotations

from typing import Any

from evolution_sim.env.events import EventType
import evolution_sim.env.runtime.resources as runtime_resources
from evolution_sim.env.runtime.state import Agent
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
):
    key = genome_profile_key(genome)
    profile = world._trophic_profile_cache.get(key)
    if profile is None:
        profile = world._compute_trophic_profile_for_genome(genome)
        world._trophic_profile_cache[key] = profile
    return profile


def apply_metabolism(world: Any, agent: Agent, moved: bool) -> None:
    season = world._season_state()["name"]
    move_cost = (
        agent.genome.move_cost * (1.0 + agent.injury_load * 0.42)
        if moved
        else 0.0
    )
    profile = world._trophic_profile(agent)
    energy_modifier = world._agent_energy_drain_modifier(agent, season)
    hydration_modifier = world._agent_hydration_drain_modifier(agent, season)
    energy_modifier *= 1.0 + agent.injury_load * 0.14
    hydration_modifier *= 1.0 + agent.injury_load * 0.08
    energy_modifier *= (
        1.0 + profile.breadth * world.config.trophic.breadth_metabolism_penalty
    )
    hydration_modifier *= (
        1.0 + profile.breadth * world.config.trophic.breadth_hydration_penalty
    )
    if (
        not moved
        and profile.meat_mode in {"hunter", "scavenger", "mixed"}
        and world._energy_ratio(agent) < 0.58
    ):
        energy_modifier *= 0.35
        hydration_modifier *= 0.65

    base_energy_cost = world.config.base_energy_drain * energy_modifier
    movement_energy_cost = move_cost * energy_modifier
    agent.energy -= base_energy_cost + movement_energy_cost
    runtime_resources.record_energy_spent(world, "metabolism", base_energy_cost)
    runtime_resources.record_energy_spent(world, "movement", movement_energy_cost)
    agent.hydration -= (
        world.config.base_hydration_drain * hydration_modifier
        + (0.004 if moved else 0.0)
    )


def apply_health_and_hazards(world: Any, agent: Agent, moved: bool) -> None:
    if not agent.alive:
        return

    hazard_type, hazard_level = world._hazard_at(agent.x, agent.y)
    if hazard_type != "none" and hazard_level > 0:
        tile = world.grid[agent.y][agent.x]
        if hazard_type == "exposure":
            resistance = (
                agent.genome.heat_tolerance * 0.16
                + tile.shelter * 0.18
                + world._refuge_score(agent.x, agent.y) * 0.12
            )
            damage = (
                world.config.hazards.exposure_damage_rate
                * hazard_level
                * max(0.42, 1.0 - resistance)
            )
        else:
            resistance = agent.genome.defense_rating * 0.14 + tile.shelter * 0.06
            if tile.terrain == "rocky":
                resistance += agent.genome.rocky_affinity * 0.08
            damage = (
                world.config.hazards.instability_damage_rate
                * hazard_level
                * max(0.46, 1.0 - resistance)
            )
            if moved:
                damage *= 1.08
        if damage > 0:
            if world.record_tick_details:
                world.tick_hazard_exposure_agents.add(agent.agent_id)
            apply_damage(
                world,
                agent,
                damage,
                source=f"hazard_{hazard_type}",
                hazard_type=hazard_type,
            )
            if agent.health <= 0 and agent.alive:
                world._kill_agent(agent, cause=f"hazard_{hazard_type}")
                return

    if agent.health >= agent.max_health:
        agent.injury_load = world._clamp01(max(0.0, agent.injury_load - 0.004))
        return

    hazards = world.config.hazards
    if (
        world._energy_ratio(agent) >= hazards.min_energy_ratio_for_healing
        and world._hydration_ratio(agent) >= hazards.min_hydration_ratio_for_healing
        and hazard_level < hazards.healing_hazard_threshold
    ):
        heal_amount = (
            hazards.healing_base_rate
            * agent.genome.healing_efficiency
            * (
                0.44
                + world._energy_ratio(agent) * 0.28
                + world._hydration_ratio(agent) * 0.28
            )
            * (1.0 - hazard_level * 0.6)
        )
        previous = agent.health
        agent.health = min(agent.max_health, agent.health + heal_amount)
        if agent.health > previous:
            healed = agent.health - previous
            agent.injury_load = world._clamp01(
                max(0.0, agent.injury_load - healed / max(agent.max_health, 1e-9) * 0.84)
            )
            world._emit(
                EventType.AGENT_HEALED,
                agent_id=agent.agent_id,
                data={
                    "amount": round(healed, 4),
                    "health": round(agent.health, 4),
                },
            )


def apply_damage(
    world: Any,
    agent: Agent,
    amount: float,
    source: str,
    hazard_type: str | None = None,
    attacker_id: int | None = None,
) -> None:
    if amount <= 0 or not agent.alive:
        return
    agent.health -= amount
    agent.injury_load = world._clamp01(
        agent.injury_load + amount / max(agent.max_health, 1e-9)
    )
    agent.last_damage_source = source
    if world.record_tick_details:
        world.tick_damage_events.append(
            {
                "agent_id": agent.agent_id,
                "amount": round(amount, 4),
                "source": source,
                "hazard_type": hazard_type,
                "attacker_id": attacker_id,
            }
        )
    world.run_combat_totals["damage_taken"] += amount
    if source.startswith("hazard_"):
        world.run_combat_totals["hazard_damage_taken"] += amount
    world._emit(
        EventType.AGENT_DAMAGED,
        agent_id=agent.agent_id,
        data={
            "amount": round(amount, 4),
            "health": round(agent.health, 4),
            "source": source,
            "hazard_type": hazard_type,
            "attacker_id": attacker_id,
        },
    )


def record_death_cause(world: Any, agent: Agent, death_cause: str) -> None:
    role = world._trophic_role(agent)
    mode = world._meat_mode(agent)
    world.run_death_cause_counts[death_cause] = (
        world.run_death_cause_counts.get(death_cause, 0) + 1
    )
    role_counts = world.run_death_causes_by_trophic_role[role]
    role_counts[death_cause] = role_counts.get(death_cause, 0) + 1
    mode_counts = world.run_death_causes_by_meat_mode[mode]
    mode_counts[death_cause] = mode_counts.get(death_cause, 0) + 1


def should_die(world: Any, agent: Agent) -> bool:
    return (
        agent.energy <= 0
        or agent.hydration <= 0
        or agent.health <= 0
        or agent.age >= world.config.max_age
    )


def death_cause(world: Any, agent: Agent) -> str:
    if agent.health <= 0:
        return agent.last_damage_source or "health_depletion"
    if agent.energy <= 0:
        return "energy_depletion"
    if agent.hydration <= 0:
        return "hydration_depletion"
    if agent.age >= world.config.max_age:
        return "old_age"
    return "unknown"
