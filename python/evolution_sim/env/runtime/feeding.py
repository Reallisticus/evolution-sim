from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from evolution_sim.env.events import EventType
from evolution_sim.env.runtime import feeding_opportunity
from evolution_sim.env.runtime.state import Agent, TrophicProfile


FOOD_SOURCES = frozenset({"plant", "fresh_kill", "carcass"})
ANIMAL_FOOD_SOURCES = frozenset({"fresh_kill", "carcass"})
ANIMAL_RESOURCE_KINDS = feeding_opportunity.ANIMAL_RESOURCE_KINDS
ANIMAL_RESOURCE_POLICY_BLOCKERS = feeding_opportunity.ANIMAL_RESOURCE_POLICY_BLOCKERS


@dataclass(frozen=True, slots=True)
class FeedingContext:
    """Explicit runtime authority used by feeding helpers."""

    config: Any
    emit: Callable[..., None]
    species_id_for_agent: Callable[[int], int | None]
    trophic_profile: Callable[[Agent], TrophicProfile]
    matched_diet_ratio: Callable[[Agent, TrophicProfile], float]
    agent_reachable_animal_resources: Callable[..., dict[str, object]]
    scavenger_carcass_hydration_fraction: Callable[[Agent], float]
    hydration_ratio: Callable[[Agent], float]
    clamp01: Callable[[float], float]
    fresh_kill_tile_summary_for_position: Callable[[int, int], dict[str, object]]
    carcass_tile_summary_for_position: Callable[[int, int], dict[str, object]]


def build_feeding_context(world: Any) -> FeedingContext:
    return FeedingContext(
        config=world.config,
        emit=world._emit,
        species_id_for_agent=world._species_id_for_agent,
        trophic_profile=world._trophic_profile,
        matched_diet_ratio=world._matched_diet_ratio,
        agent_reachable_animal_resources=world._agent_reachable_animal_resources,
        scavenger_carcass_hydration_fraction=(
            world._scavenger_carcass_hydration_fraction
        ),
        hydration_ratio=world._hydration_ratio,
        clamp01=world._clamp01,
        fresh_kill_tile_summary_for_position=(
            world._fresh_kill_tile_summary_for_position
        ),
        carcass_tile_summary_for_position=world._carcass_tile_summary_for_position,
    )


def _resolve_feeding_context(
    world: Any,
    context: FeedingContext | None = None,
) -> FeedingContext:
    if context is not None:
        return context
    return build_feeding_context(world)


def empty_diet_totals() -> dict[str, float]:
    return {
        "plant_events": 0,
        "plant_energy": 0.0,
        "fresh_kill_events": 0,
        "fresh_kill_energy": 0.0,
        "carcass_events": 0,
        "carcass_energy": 0.0,
    }


def empty_grouped_diet_totals(
    groups: list[str] | dict[str, int],
) -> dict[str, dict[str, float]]:
    return {str(group): empty_diet_totals() for group in groups}


def empty_animal_resource_consumption_counts() -> dict[str, int | float]:
    return {
        "fresh_kill_consumption_events": 0,
        "fresh_kill_energy_consumed": 0.0,
        "fresh_kill_gained_energy": 0.0,
        "carcass_consumption_events": 0,
        "carcass_energy_consumed": 0.0,
        "carcass_gained_energy": 0.0,
        "animal_resource_consumption_events": 0,
        "animal_resource_energy_consumed": 0.0,
        "animal_resource_gained_energy": 0.0,
    }


def empty_grouped_animal_resource_consumption_counts(
    groups: list[str] | dict[str, int],
) -> dict[str, dict[str, int | float]]:
    return {
        str(group): empty_animal_resource_consumption_counts()
        for group in groups
    }


def empty_animal_resource_opportunity_counts() -> dict[str, int | float]:
    counts: dict[str, int | float] = {
        "alive_ticks": 0,
        "alive_agent_ticks": 0,
        "animal_resource_present_ticks": 0,
        "animal_resource_present_agent_ticks": 0,
        "animal_resource_absent_ticks": 0,
        "animal_resource_absent_agent_ticks": 0,
        "animal_resource_present_unconsumed_ticks": 0,
        "animal_resource_present_unconsumed_agent_ticks": 0,
        "animal_resource_reachable_ticks": 0,
        "animal_resource_reachable_agent_ticks": 0,
        "animal_resource_reachable_unconsumed_ticks": 0,
        "animal_resource_reachable_unconsumed_agent_ticks": 0,
        "animal_resource_present_unreachable_ticks": 0,
        "animal_resource_present_unreachable_agent_ticks": 0,
        "animal_resource_consumed_ticks": 0,
        "fresh_kill_present_ticks": 0,
        "fresh_kill_present_agent_ticks": 0,
        "fresh_kill_present_unconsumed_ticks": 0,
        "fresh_kill_present_unconsumed_agent_ticks": 0,
        "fresh_kill_reachable_ticks": 0,
        "fresh_kill_reachable_agent_ticks": 0,
        "fresh_kill_reachable_unconsumed_ticks": 0,
        "fresh_kill_reachable_unconsumed_agent_ticks": 0,
        "fresh_kill_present_unreachable_ticks": 0,
        "fresh_kill_present_unreachable_agent_ticks": 0,
        "fresh_kill_consumed_ticks": 0,
        "carcass_present_ticks": 0,
        "carcass_present_agent_ticks": 0,
        "carcass_present_unconsumed_ticks": 0,
        "carcass_present_unconsumed_agent_ticks": 0,
        "carcass_reachable_ticks": 0,
        "carcass_reachable_agent_ticks": 0,
        "carcass_reachable_unconsumed_ticks": 0,
        "carcass_reachable_unconsumed_agent_ticks": 0,
        "carcass_present_unreachable_ticks": 0,
        "carcass_present_unreachable_agent_ticks": 0,
        "carcass_consumed_ticks": 0,
        **empty_animal_resource_consumption_counts(),
    }
    for resource in ("animal_resource", *ANIMAL_RESOURCE_KINDS):
        counts[f"{resource}_policy_actionable_ticks"] = 0
        counts[f"{resource}_policy_actionable_agent_ticks"] = 0
        counts[f"{resource}_reachable_policy_blocked_ticks"] = 0
        counts[f"{resource}_reachable_policy_blocked_agent_ticks"] = 0
        for blocker in ANIMAL_RESOURCE_POLICY_BLOCKERS:
            counts[f"{resource}_policy_blocked_by_{blocker}_ticks"] = 0
            counts[f"{resource}_policy_blocked_by_{blocker}_agent_ticks"] = 0
    return counts


def empty_grouped_animal_resource_opportunity_counts(
    groups: list[str] | dict[str, int],
) -> dict[str, dict[str, int | float]]:
    return {
        str(group): empty_animal_resource_opportunity_counts()
        for group in groups
    }


def empty_animal_resource_reachability_tick_counts() -> dict[str, int]:
    counts = {
        "animal_resource_reachable_agents": 0,
        "fresh_kill_reachable_agents": 0,
        "carcass_reachable_agents": 0,
    }
    for resource in ("animal_resource", *ANIMAL_RESOURCE_KINDS):
        counts[f"{resource}_policy_actionable_agents"] = 0
        counts[f"{resource}_reachable_policy_blocked_agents"] = 0
        for blocker in ANIMAL_RESOURCE_POLICY_BLOCKERS:
            counts[f"{resource}_policy_blocked_by_{blocker}_agents"] = 0
    return counts


def accumulate_diet_totals(
    totals: dict[str, float],
    food_source: str,
    gained_energy: float,
) -> None:
    if food_source not in FOOD_SOURCES:
        raise ValueError(f"Unsupported food source: {food_source}")
    totals[f"{food_source}_events"] += 1
    totals[f"{food_source}_energy"] += gained_energy


def record_recent_diet(
    agent: Agent,
    food_source: str,
    gained_energy: float,
) -> None:
    if gained_energy <= 0:
        return
    if food_source == "plant":
        agent.recent_plant_energy += gained_energy
    elif food_source == "fresh_kill":
        agent.recent_fresh_kill_energy += gained_energy
    elif food_source == "carcass":
        agent.recent_carcass_energy += gained_energy


def record_feeding_event(
    world: Any,
    agent: Agent,
    food_source: str,
    consumed: float,
    gained_energy: float,
    profile: TrophicProfile,
    *,
    energy_before: float,
    energy_after: float,
    potential_energy: float | None = None,
    context: FeedingContext | None = None,
) -> None:
    feeding_context = _resolve_feeding_context(world, context)
    record_recent_diet(agent, food_source, gained_energy)
    if world.record_tick_details:
        world.tick_feeding_events.append(
            {
                "agent_id": agent.agent_id,
                "species_id": feeding_context.species_id_for_agent(agent.agent_id),
                "food_source": food_source,
                "consumed": round(consumed, 4),
                "gained_energy": round(gained_energy, 4),
                "energy_before": round(energy_before, 4),
                "energy_after": round(energy_after, 4),
                "potential_energy": round(
                    potential_energy if potential_energy is not None else gained_energy,
                    4,
                ),
                "trophic_role": profile.role,
                "meat_mode": profile.meat_mode,
                "matched_diet_ratio": round(
                    feeding_context.matched_diet_ratio(agent, profile),
                    4,
                ),
            }
        )
    accumulate_diet_totals(world.run_diet_totals, food_source, gained_energy)
    accumulate_diet_totals(
        world.run_diet_by_trophic_role[profile.role],
        food_source,
        gained_energy,
    )
    accumulate_diet_totals(
        world.run_diet_by_meat_mode[profile.meat_mode],
        food_source,
        gained_energy,
    )


def record_animal_resource_consumption(
    world: Any,
    meat_mode: str,
    food_source: str,
    consumed: float,
    gained_energy: float,
) -> None:
    if food_source not in ANIMAL_FOOD_SOURCES:
        return
    counts = world.tick_animal_resource_consumption_by_meat_mode[meat_mode]
    counts[f"{food_source}_consumption_events"] += 1
    counts[f"{food_source}_energy_consumed"] += consumed
    counts[f"{food_source}_gained_energy"] += gained_energy
    counts["animal_resource_consumption_events"] += 1
    counts["animal_resource_energy_consumed"] += consumed
    counts["animal_resource_gained_energy"] += gained_energy


def animal_resource_presence_this_tick(world: Any) -> dict[str, bool]:
    fresh_kill_consumed = any(
        counts["fresh_kill_consumption_events"] > 0
        for counts in world.tick_animal_resource_consumption_by_meat_mode.values()
    )
    carcass_consumed = any(
        counts["carcass_consumption_events"] > 0
        for counts in world.tick_animal_resource_consumption_by_meat_mode.values()
    )
    fresh_kill_present = (
        world.tick_fresh_kill_deposited_energy > 0
        or fresh_kill_consumed
        or any(
            tile.fresh_kill_energy > 1e-9
            for row in world.grid
            for tile in row
            if tile.terrain != "water"
        )
    )
    carcass_present = (
        world.tick_carcass_deposited_energy > 0
        or carcass_consumed
        or any(
            tile.carcass_energy > 1e-9
            for row in world.grid
            for tile in row
            if tile.terrain != "water"
        )
    )
    return {
        "fresh_kill": fresh_kill_present,
        "carcass": carcass_present,
        "animal_resource": fresh_kill_present or carcass_present,
    }


def animal_resource_reachability_by_meat_mode(
    world: Any,
    agents: list[Agent],
    *,
    meat_mode_codes: dict[str, int],
    radius: int,
    action_masks_by_agent: dict[int, dict[str, bool]] | None = None,
    resource_presence: dict[str, bool] | None = None,
    context: FeedingContext | None = None,
) -> dict[str, dict[str, int]]:
    feeding_context = _resolve_feeding_context(world, context)
    reachability = {
        mode: empty_animal_resource_reachability_tick_counts()
        for mode in meat_mode_codes
    }
    presence = resource_presence or animal_resource_presence_this_tick(world)
    if not presence["fresh_kill"] and not presence["carcass"]:
        return reachability
    for agent in agents:
        profile = feeding_context.trophic_profile(agent)
        mode_counts = reachability[profile.meat_mode]
        reachable = feeding_context.agent_reachable_animal_resources(
            agent,
            radius=radius,
            action_mask=(
                action_masks_by_agent.get(agent.agent_id)
                if action_masks_by_agent is not None
                else None
            ),
        )
        for resource in ANIMAL_RESOURCE_KINDS:
            if bool(reachable[resource]):
                mode_counts[f"{resource}_reachable_agents"] += 1
                if bool(reachable[f"{resource}_policy_actionable"]):
                    mode_counts[f"{resource}_policy_actionable_agents"] += 1
                else:
                    mode_counts[f"{resource}_reachable_policy_blocked_agents"] += 1
                    blockers = reachable[f"{resource}_policy_blockers"]
                    if isinstance(blockers, set):
                        for blocker in blockers:
                            mode_counts[
                                f"{resource}_policy_blocked_by_{blocker}_agents"
                            ] += 1
        if bool(reachable["animal_resource"]):
            mode_counts["animal_resource_reachable_agents"] += 1
            if bool(reachable["animal_resource_policy_actionable"]):
                mode_counts["animal_resource_policy_actionable_agents"] += 1
            else:
                mode_counts["animal_resource_reachable_policy_blocked_agents"] += 1
                blockers = reachable["animal_resource_policy_blockers"]
                if isinstance(blockers, set):
                    for blocker in blockers:
                        mode_counts[
                            f"animal_resource_policy_blocked_by_{blocker}_agents"
                        ] += 1
    return reachability


def record_animal_resource_opportunity_tick(
    world: Any,
    meat_mode_counts: dict[str, int],
    reachability_by_meat_mode: dict[str, dict[str, int]],
) -> None:
    feeding_opportunity.record_animal_resource_opportunity_tick_from_inputs(
        world.run_animal_resource_opportunity_by_meat_mode,
        meat_mode_counts=meat_mode_counts,
        tick_consumption_by_meat_mode=(
            world.tick_animal_resource_consumption_by_meat_mode
        ),
        reachability_by_meat_mode=reachability_by_meat_mode,
        resource_presence=animal_resource_presence_this_tick(world),
    )


def record_plant_intake(
    world: Any,
    agent: Agent,
    profile: TrophicProfile,
    *,
    consumed: float,
    energy_before: float,
    energy_after: float,
    potential_gain: float,
    tile: Any,
    context: FeedingContext | None = None,
) -> dict[str, object]:
    feeding_context = _resolve_feeding_context(world, context)
    gained = energy_after - energy_before
    record_feeding_event(
        world,
        agent,
        "plant",
        consumed,
        gained,
        profile,
        energy_before=energy_before,
        energy_after=energy_after,
        potential_energy=potential_gain,
        context=feeding_context,
    )
    feeding_context.emit(
        EventType.AGENT_ATE,
        agent_id=agent.agent_id,
        data={
            "food_source": "plant",
            "consumed": round(consumed, 4),
            "energy_before": round(energy_before, 4),
            "energy": round(energy_after, 4),
            "gained_energy": round(gained, 4),
            "potential_gained_energy": round(potential_gain, 4),
            "trophic_role": profile.role,
            "meat_mode": profile.meat_mode,
            "matched_diet_ratio": round(
                feeding_context.matched_diet_ratio(agent, profile),
                4,
            ),
            "vegetation": round(tile.vegetation, 4),
            "recovery_debt": round(tile.recovery_debt, 4),
            "shelter": round(tile.shelter, 4),
        },
    )
    return {
        "ate": True,
        "food_source": "plant",
        "x": agent.x,
        "y": agent.y,
        "consumed": round(consumed, 4),
        "gained_energy": round(gained, 4),
        "potential_gained_energy": round(potential_gain, 4),
        "immediate_kill_feed": False,
        "source_breakdown": [],
    }


def _apply_meat_hydration(
    world: Any,
    agent: Agent,
    food_source: str,
    consumed: float,
    profile: TrophicProfile,
    *,
    context: FeedingContext | None = None,
) -> None:
    feeding_context = _resolve_feeding_context(world, context)
    if food_source == "carcass" and profile.meat_mode == "scavenger":
        agent.hydration = min(
            agent.genome.max_hydration,
            agent.hydration
            + (
                consumed
                * feeding_context.scavenger_carcass_hydration_fraction(agent)
                * agent.genome.water_efficiency
            ),
        )
    elif (
        food_source == "carcass"
        and profile.meat_mode == "hunter"
        and feeding_context.hydration_ratio(agent)
        < feeding_context.config.carcasses.hunter_carcass_hydration_max_ratio
    ):
        agent.hydration = min(
            agent.genome.max_hydration,
            agent.hydration
            + (
                consumed
                * feeding_context.config.carcasses.hunter_carcass_hydration_fraction
                * agent.genome.water_efficiency
            ),
        )
    elif food_source == "carcass" and profile.meat_mode == "mixed":
        agent.hydration = min(
            agent.genome.max_hydration,
            agent.hydration
            + (
                consumed
                * feeding_context.config.carcasses.mixed_carcass_hydration_fraction
                * agent.genome.water_efficiency
            ),
        )
    if food_source == "fresh_kill" and profile.meat_mode in {"hunter", "mixed"}:
        agent.hydration = min(
            agent.genome.max_hydration,
            agent.hydration
            + (
                consumed
                * feeding_context.config.carcasses.fresh_kill_hydration_fraction
                * agent.genome.water_efficiency
            ),
        )


def _apply_meat_healing(
    world: Any,
    agent: Agent,
    food_source: str,
    consumed: float,
    profile: TrophicProfile,
    *,
    context: FeedingContext | None = None,
) -> None:
    feeding_context = _resolve_feeding_context(world, context)
    healing_multiplier = 1.0
    if food_source == "carcass" and profile.meat_mode == "scavenger":
        healing_multiplier = feeding_context.config.carcasses.scavenger_healing_multiplier
    elif food_source == "fresh_kill" and profile.meat_mode in {"hunter", "mixed"}:
        healing_multiplier = (
            feeding_context.config.carcasses.fresh_kill_hunter_healing_multiplier
        )
    healed = (
        consumed
        * feeding_context.config.carcasses.healing_fraction
        * healing_multiplier
        * agent.genome.healing_efficiency
    )
    if healed <= 0:
        return
    previous_health = agent.health
    agent.health = min(agent.max_health, agent.health + healed)
    if agent.health > previous_health:
        agent.injury_load = feeding_context.clamp01(
            max(
                0.0,
                agent.injury_load
                - (agent.health - previous_health) / max(agent.max_health, 1e-9),
            )
        )


def apply_meat_intake(
    world: Any,
    agent: Agent,
    food_source: str,
    consumed: float,
    potential_nutrition: float,
    profile: TrophicProfile,
    x: int,
    y: int,
    source_breakdown: list[dict[str, object]],
    deposit_breakdown: list[dict[str, object]],
    immediate_kill_feed: bool,
    freshness: float | None,
    context: FeedingContext | None = None,
) -> dict[str, object]:
    feeding_context = _resolve_feeding_context(world, context)
    energy_before = agent.energy
    agent.energy = min(agent.genome.max_energy, agent.energy + potential_nutrition)
    nutrition = agent.energy - energy_before
    _apply_meat_hydration(
        world,
        agent,
        food_source,
        consumed,
        profile,
        context=feeding_context,
    )
    _apply_meat_healing(
        world,
        agent,
        food_source,
        consumed,
        profile,
        context=feeding_context,
    )

    event_bucket = (
        world.tick_fresh_kill_events
        if food_source == "fresh_kill"
        else world.tick_carcass_events
    )
    if world.record_tick_details:
        event_bucket.append(
            {
                "agent_id": agent.agent_id,
                "species_id": feeding_context.species_id_for_agent(agent.agent_id),
                "consumed": round(consumed, 4),
                "energy": round(consumed, 4),
                "gained_energy": round(nutrition, 4),
                "potential_gained_energy": round(potential_nutrition, 4),
                "meat_mode": profile.meat_mode,
                "x": x,
                "y": y,
                "source_breakdown": source_breakdown,
                "immediate_kill_feed": immediate_kill_feed,
            }
        )
    if food_source == "fresh_kill":
        world.run_fresh_kill_totals["consumption_events"] += 1
        world.run_fresh_kill_totals["energy_consumed"] += consumed
        world.run_fresh_kill_totals["gained_energy"] += nutrition
    else:
        world.run_carcass_totals["consumption_events"] += 1
        world.run_carcass_totals["energy_consumed"] += consumed
        world.run_carcass_totals["gained_energy"] += nutrition
    record_animal_resource_consumption(
        world,
        profile.meat_mode,
        food_source,
        consumed,
        nutrition,
    )
    record_feeding_event(
        world,
        agent,
        food_source,
        consumed,
        nutrition,
        profile,
        energy_before=energy_before,
        energy_after=agent.energy,
        potential_energy=potential_nutrition,
        context=feeding_context,
    )
    if world.record_events:
        tile_state = (
            feeding_context.fresh_kill_tile_summary_for_position(x, y)
            if food_source == "fresh_kill"
            else feeding_context.carcass_tile_summary_for_position(x, y)
        )
        feeding_context.emit(
            EventType.AGENT_ATE,
            agent_id=agent.agent_id,
            data={
                "food_source": food_source,
                "consumed": round(consumed, 4),
                "energy_before": round(energy_before, 4),
                "energy": round(agent.energy, 4),
                "gained_energy": round(nutrition, 4),
                "potential_gained_energy": round(potential_nutrition, 4),
                "trophic_role": profile.role,
                "meat_mode": profile.meat_mode,
                "immediate_kill_feed": immediate_kill_feed,
                "health": round(agent.health, 4),
                "matched_diet_ratio": round(
                    feeding_context.matched_diet_ratio(agent, profile),
                    4,
                ),
                "x": x,
                "y": y,
                "source_breakdown": source_breakdown,
                "deposit_breakdown": deposit_breakdown,
                "tile_mixed_sources_after": tile_state["mixed_sources"],
                "tile_dominant_source_species_after": tile_state[
                    "dominant_source_species"
                ],
                "tile_source_breakdown_after": tile_state["source_breakdown"],
            },
        )
        if freshness is not None:
            world.events[-1].data["freshness"] = round(freshness, 4)
            world.events[-1].data["tile_carcass_energy_after"] = tile_state[
                "total_energy"
            ]
            world.events[-1].data["tile_avg_freshness_after"] = tile_state[
                "avg_freshness"
            ]
        else:
            world.events[-1].data["tile_fresh_kill_energy_after"] = tile_state[
                "total_energy"
            ]
    return {
        "ate": True,
        "food_source": food_source,
        "x": x,
        "y": y,
        "consumed": round(consumed, 4),
        "gained_energy": round(nutrition, 4),
        "potential_gained_energy": round(potential_nutrition, 4),
        "immediate_kill_feed": immediate_kill_feed,
        "source_breakdown": source_breakdown,
    }
