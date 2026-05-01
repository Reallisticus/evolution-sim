from __future__ import annotations

from typing import Any

import evolution_sim.env.runtime.reproduction as runtime_reproduction
import evolution_sim.env.runtime.signals as runtime_signals
from evolution_sim.env.events import EventType


def run_tick(
    world: Any,
    *,
    meat_mode_codes: dict[str, int],
) -> tuple[int, int]:
    births_this_tick = 0
    deaths_before_tick = world.deaths
    reset_tick_state(world, meat_mode_codes=meat_mode_codes)
    world._invalidate_biotic_state()
    world._decay_signal_emissions()
    climate_state = world._climate_state()
    world._emit(
        EventType.TICK_STARTED,
        data={
            "alive_agents": len(world.alive_agents()),
            "season": world._season_state()["name"],
            "disturbance_type": climate_state["disturbance_type"],
            "disturbance_strength": climate_state["disturbance_strength"],
        },
    )
    world._regrow_resources()
    tick_start_alive = world.alive_agents()
    _, opportunity_meat_mode_counts = world._population_trophic_counts(
        tick_start_alive
    )
    opportunity_reachability_by_meat_mode = (
        world._animal_resource_reachability_by_meat_mode(tick_start_alive)
    )
    observation_snapshots = {
        agent.agent_id: world._observe_agent(agent) for agent in tick_start_alive
    }
    trajectory_contexts = build_trajectory_contexts(
        world,
        tick_start_alive,
        observation_snapshots,
    )
    pending_trajectory_records: list[dict[str, object]] = []
    acted_trajectory_agent_ids: set[int] = set()

    for agent_id in sorted(world.agents):
        agent = world.agents[agent_id]
        if not agent.alive:
            continue
        world._decay_recent_diet(agent)
        trajectory_context = (
            trajectory_contexts[agent_id]
            if world.record_trajectory and agent_id in trajectory_contexts
            else None
        )
        action = world._choose_action(agent, observation_snapshots.get(agent_id))
        live_action_mask = world._action_mask(agent)
        if trajectory_context is not None:
            moved, action_outcome = world._resolve_action_with_outcome(
                agent,
                action,
                observation_action_mask=trajectory_context["action_mask"],
                resolution_action_mask=live_action_mask,
            )
            resolved_action = str(action_outcome["resolved_action"])
        else:
            moved = world._resolve_action(agent, action)
            resolved_action = action if live_action_mask.get(action, False) else "stay"
            action_outcome = None
        world._apply_metabolism(agent, moved=moved)
        world._apply_health_and_hazards(agent, moved=moved)
        agent.age += 1
        if trajectory_context is not None:
            trajectory_context.update(
                {
                    "requested_action": action,
                    "action_source": world._policy_action_source,
                    "policy_id": world._policy_id,
                    "policy_version": world._policy_version,
                    "resolution_action_mask": live_action_mask,
                    "resolved_action": resolved_action,
                    "moved": moved,
                    "action_outcome": action_outcome,
                }
            )
            pending_trajectory_records.append(trajectory_context)
            acted_trajectory_agent_ids.add(agent_id)
        world._invalidate_biotic_state()

    births_this_tick = runtime_reproduction.run_reproduction_phase(world)

    for agent_id in sorted(world.agents):
        agent = world.agents[agent_id]
        if agent.alive and world._should_die(agent):
            world._kill_agent(agent, cause=world._death_cause(agent))

    if world.record_trajectory:
        append_passive_trajectory_contexts(
            world,
            trajectory_contexts,
            acted_trajectory_agent_ids,
            pending_trajectory_records,
        )
        world._finalize_trajectory_decisions(pending_trajectory_records)

    deaths_this_tick = world.deaths - deaths_before_tick
    alive_count = len(world.alive_agents())
    world.peak_alive_agents = max(world.peak_alive_agents, alive_count)
    world._record_animal_resource_opportunity_tick(
        opportunity_meat_mode_counts,
        opportunity_reachability_by_meat_mode,
    )
    world._emit(
        EventType.TICK_COMPLETED,
        data={
            "alive_agents": alive_count,
            "births": births_this_tick,
            "deaths": deaths_this_tick,
            "season": world._season_state()["name"],
            "disturbance_type": climate_state["disturbance_type"],
            "disturbance_strength": climate_state["disturbance_strength"],
        },
    )
    return births_this_tick, deaths_this_tick


def reset_tick_state(
    world: Any,
    *,
    meat_mode_codes: dict[str, int],
) -> None:
    world.tick_birth_pairs = []
    world.tick_death_agent_ids = []
    world.tick_death_events = []
    world.tick_attack_events = []
    world.tick_damage_events = []
    world.tick_carcass_deposit_events = []
    world.tick_carcass_events = []
    world.tick_fresh_kill_events = []
    world.tick_fresh_kill_deposit_events = []
    world.tick_reproduction_blocked_events = []
    world.tick_reproduction_mate_search_events = []
    world.tick_fresh_kill_to_carcass_energy = 0.0
    world.tick_carcass_energy_decayed = 0.0
    world.tick_fresh_kill_deposited_energy = 0.0
    world.tick_carcass_deposited_energy = 0.0
    world.tick_feeding_events = []
    world.tick_trajectory_records = []
    world.tick_signal_emission_events = []
    world.tick_signal_totals = runtime_signals.empty_signal_totals()
    world.tick_reproduction_mate_search_counts = (
        runtime_reproduction.empty_reproduction_mate_search_counts()
    )
    world.tick_animal_resource_consumption_by_meat_mode = (
        world._empty_grouped_animal_resource_consumption_counts(meat_mode_codes)
    )
    world.tick_hazard_exposure_agents = set()


def build_trajectory_contexts(
    world: Any,
    tick_start_alive: list[Any],
    observation_snapshots: dict[int, object],
) -> dict[int, dict[str, object]]:
    if not world.record_trajectory:
        return {}
    return {
        agent.agent_id: world._begin_trajectory_decision(
            agent,
            observation_snapshots[agent.agent_id],
        )
        for agent in tick_start_alive
    }


def append_passive_trajectory_contexts(
    world: Any,
    trajectory_contexts: dict[int, dict[str, object]],
    acted_trajectory_agent_ids: set[int],
    pending_trajectory_records: list[dict[str, object]],
) -> None:
    for agent_id, trajectory_context in trajectory_contexts.items():
        if agent_id in acted_trajectory_agent_ids:
            continue
        agent = world.agents[agent_id]
        if agent.alive or agent.death_tick != world.tick:
            continue
        action_mask = dict(trajectory_context["action_mask"])
        action_mask["stay"] = True
        passive_outcome = world._base_action_outcome(
            requested_action="stay",
            resolved_action="stay",
            observation_action_mask=action_mask,
            resolution_action_mask=action_mask,
        )
        trajectory_context.update(
            {
                "requested_action": "stay",
                "action_source": "passive",
                "policy_id": None,
                "policy_version": None,
                "resolution_action_mask": action_mask,
                "resolved_action": "stay",
                "moved": False,
                "action_outcome": passive_outcome,
            }
        )
        pending_trajectory_records.append(trajectory_context)
