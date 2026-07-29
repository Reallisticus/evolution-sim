from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import evolution_sim.env.runtime.actions as runtime_actions
import evolution_sim.env.runtime.capacity as runtime_capacity
import evolution_sim.env.runtime.feeding as runtime_feeding
import evolution_sim.env.runtime.lifecycle as runtime_lifecycle
import evolution_sim.env.runtime.reproduction as runtime_reproduction
import evolution_sim.env.runtime.signals as runtime_signals
from evolution_sim.env.events import EventType

TURN_ORDER_POLICY = "seed_tick_agent_hash_permutation_v1"
_RUNTIME_OWNED_POLICY_TICK_START_AUTHORITY = object()
_RUNTIME_OWNED_POLICY_TICK_START_REGISTRY: dict[
    int,
    tuple[
        RuntimeOwnedPolicyTickStart,
        tuple[tuple[int, int], ...],
        tuple[tuple[int, str], ...],
    ],
] = {}


@dataclass(frozen=True, slots=True)
class RuntimeOwnedPolicyTickStart:
    """Internal witness for the snapshot/order minted by prepare_tick_start."""

    environment_seed: int
    tick: int
    ordered_agent_ids: tuple[int, ...]
    observation_snapshots: dict[int, dict[str, object]]
    _observation_integrity_sha256_by_agent: dict[int, str] = field(
        repr=False,
        compare=False,
    )
    _authority: object = field(repr=False, compare=False)


def _runtime_owned_observation_integrity_sha256(
    observation: dict[str, object],
) -> str:
    try:
        payload = pickle.dumps(observation, protocol=5)
    except (
        pickle.PickleError,
        AttributeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "runtime-owned observation integrity serialization failed"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _runtime_owned_policy_tick_start(
    *,
    environment_seed: int,
    tick: int,
    ordered_agent_ids: tuple[int, ...],
    observation_snapshots: dict[int, dict[str, object]],
) -> RuntimeOwnedPolicyTickStart:
    observation_integrity_sha256_by_agent = {
        agent_id: _runtime_owned_observation_integrity_sha256(
            observation_snapshots[agent_id]
        )
        for agent_id in ordered_agent_ids
    }
    witness = RuntimeOwnedPolicyTickStart(
        environment_seed=environment_seed,
        tick=tick,
        ordered_agent_ids=ordered_agent_ids,
        observation_snapshots=observation_snapshots,
        _observation_integrity_sha256_by_agent=(
            observation_integrity_sha256_by_agent
        ),
        _authority=_RUNTIME_OWNED_POLICY_TICK_START_AUTHORITY,
    )
    _RUNTIME_OWNED_POLICY_TICK_START_REGISTRY[id(witness)] = (
        witness,
        tuple(
            (agent_id, id(observation_snapshots[agent_id]))
            for agent_id in ordered_agent_ids
        ),
        tuple(
            (agent_id, observation_integrity_sha256_by_agent[agent_id])
            for agent_id in ordered_agent_ids
        ),
    )
    return witness


def validate_runtime_owned_policy_tick_start(
    value: object,
    *,
    environment_seed: int,
    tick: int,
) -> RuntimeOwnedPolicyTickStart:
    registered = _RUNTIME_OWNED_POLICY_TICK_START_REGISTRY.pop(id(value), None)
    if (
        type(value) is not RuntimeOwnedPolicyTickStart
        or registered is None
        or registered[0] is not value
        or value._authority is not _RUNTIME_OWNED_POLICY_TICK_START_AUTHORITY
        or value.environment_seed != environment_seed
        or value.tick != tick
        or set(value.observation_snapshots) != set(value.ordered_agent_ids)
        or set(value._observation_integrity_sha256_by_agent)
        != set(value.ordered_agent_ids)
        or registered[1]
        != tuple(
            (agent_id, id(value.observation_snapshots[agent_id]))
            for agent_id in value.ordered_agent_ids
        )
        or registered[2]
        != tuple(
            (agent_id, value._observation_integrity_sha256_by_agent[agent_id])
            for agent_id in value.ordered_agent_ids
        )
        or registered[2]
        != tuple(
            (
                agent_id,
                _runtime_owned_observation_integrity_sha256(
                    value.observation_snapshots[agent_id]
                ),
            )
            for agent_id in value.ordered_agent_ids
        )
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or digest != digest.lower()
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in value._observation_integrity_sha256_by_agent.values()
        )
        or any(
            type(observation) is not dict
            for observation in value.observation_snapshots.values()
        )
    ):
        raise ValueError("runtime-owned policy tick-start witness is invalid")
    return value


def discard_runtime_owned_policy_tick_start(
    value: RuntimeOwnedPolicyTickStart,
) -> None:
    _RUNTIME_OWNED_POLICY_TICK_START_REGISTRY.pop(id(value), None)


@dataclass(frozen=True, slots=True)
class TickPhaseContext:
    invalidate_biotic_state: Callable[[], None]
    decay_signal_emissions: Callable[[], None]
    climate_state: Callable[[], dict[str, object]]
    season_state: Callable[[], dict[str, object]]
    emit: Callable[[EventType, int | None, dict[str, object] | None], None]
    regrow_resources: Callable[[], None]
    population_trophic_counts: Callable[
        [list[Any]], tuple[dict[str, int], dict[str, int]]
    ]
    observe_agent: Callable[[Any], dict[str, object]]
    animal_resource_reachability_by_meat_mode: Callable[..., dict[str, dict[str, int]]]
    animal_resource_presence_this_tick: Callable[[], dict[str, bool]]
    decay_recent_diet: Callable[[Any], None]
    stage_policy_tick_start: Callable[[RuntimeOwnedPolicyTickStart], None]
    choose_action: Callable[[Any, dict[str, object] | None], str]
    action_mask: Callable[[Any], dict[str, bool]]
    action_resolution_context: Callable[[Any], runtime_actions.ActionResolutionContext]
    lifecycle_context: runtime_lifecycle.LifecycleContext
    reproduction_context: Callable[[], runtime_reproduction.ReproductionContext]
    kill_agent: Callable[..., None]
    finalize_trajectory_decisions: Callable[[list[dict[str, object]]], None]
    reconcile_policy_live_agent_ids: Callable[[tuple[int, ...]], None]
    record_animal_resource_opportunity_tick: Callable[
        [dict[str, int], dict[str, dict[str, int]], dict[str, bool]],
        None,
    ]
    begin_trajectory_decision: Callable[[Any, dict[str, object]], dict[str, object]]
    policy_metadata: Callable[[], dict[str, object]]


@dataclass(frozen=True, slots=True)
class PreparedTickStart:
    """Exact policy-visible state after tick-start ecology and before actions."""

    climate_state: dict[str, object]
    ordered_agent_ids: tuple[int, ...]
    observation_snapshots: dict[int, dict[str, object]]
    observation_action_masks: dict[int, dict[str, bool]]
    trajectory_contexts: dict[int, dict[str, object]]
    opportunity_meat_mode_counts: dict[str, int]
    opportunity_resource_presence: dict[str, bool]
    opportunity_reachability_by_meat_mode: dict[str, dict[str, int]]


def prepare_tick_start(
    world: Any,
    *,
    meat_mode_codes: dict[str, int],
    tick_context: TickPhaseContext,
    stage_policy: bool = True,
) -> PreparedTickStart:
    """Run the existing tick-start prefix exactly once, stopping before actions."""

    reset_tick_state(world, meat_mode_codes=meat_mode_codes)
    tick_context.invalidate_biotic_state()
    tick_context.decay_signal_emissions()
    climate_state = tick_context.climate_state()
    tick_context.emit(
        EventType.TICK_STARTED,
        agent_id=None,
        data={
            "alive_agents": len(world.alive_agents()),
            "season": tick_context.season_state()["name"],
            "disturbance_type": climate_state["disturbance_type"],
            "disturbance_strength": climate_state["disturbance_strength"],
        },
    )
    tick_context.regrow_resources()
    tick_start_alive = world.alive_agents()
    _, opportunity_meat_mode_counts = tick_context.population_trophic_counts(
        tick_start_alive
    )
    observation_snapshots = {
        agent.agent_id: tick_context.observe_agent(agent) for agent in tick_start_alive
    }
    observation_action_masks = {
        agent_id: dict(observation["action_mask"])
        for agent_id, observation in observation_snapshots.items()
    }
    opportunity_resource_presence = tick_context.animal_resource_presence_this_tick()
    opportunity_reachability_by_meat_mode = (
        tick_context.animal_resource_reachability_by_meat_mode(
            tick_start_alive,
            action_masks_by_agent=observation_action_masks,
            resource_presence=opportunity_resource_presence,
        )
    )
    trajectory_contexts = build_trajectory_contexts(
        world,
        tick_start_alive,
        observation_snapshots,
        tick_context=tick_context,
    )
    action_order = deterministic_agent_turn_order(
        (agent.agent_id for agent in tick_start_alive),
        seed=world.config.seed,
        tick=world.tick,
    )
    world.tick_action_order = list(action_order)
    if action_order and stage_policy:
        runtime_owned_tick_start = _runtime_owned_policy_tick_start(
            environment_seed=world.config.seed,
            tick=world.tick,
            ordered_agent_ids=action_order,
            observation_snapshots=observation_snapshots,
        )
        try:
            tick_context.stage_policy_tick_start(runtime_owned_tick_start)
        finally:
            discard_runtime_owned_policy_tick_start(runtime_owned_tick_start)
    return PreparedTickStart(
        climate_state=climate_state,
        ordered_agent_ids=action_order,
        observation_snapshots=observation_snapshots,
        observation_action_masks=observation_action_masks,
        trajectory_contexts=trajectory_contexts,
        opportunity_meat_mode_counts=opportunity_meat_mode_counts,
        opportunity_resource_presence=opportunity_resource_presence,
        opportunity_reachability_by_meat_mode=(opportunity_reachability_by_meat_mode),
    )


def run_tick(
    world: Any,
    *,
    meat_mode_codes: dict[str, int],
    tick_context: TickPhaseContext,
) -> tuple[int, int]:
    births_this_tick = 0
    deaths_before_tick = world.deaths
    prepared = prepare_tick_start(
        world,
        meat_mode_codes=meat_mode_codes,
        tick_context=tick_context,
    )
    climate_state = prepared.climate_state
    action_order = prepared.ordered_agent_ids
    observation_snapshots = prepared.observation_snapshots
    trajectory_contexts = prepared.trajectory_contexts
    opportunity_meat_mode_counts = prepared.opportunity_meat_mode_counts
    opportunity_resource_presence = prepared.opportunity_resource_presence
    opportunity_reachability_by_meat_mode = (
        prepared.opportunity_reachability_by_meat_mode
    )
    pending_trajectory_records: list[dict[str, object]] = []
    acted_trajectory_agent_ids: set[int] = set()
    lifecycle_context = tick_context.lifecycle_context

    for agent_id in action_order:
        agent = world.agents[agent_id]
        if not agent.alive:
            continue
        tick_context.decay_recent_diet(agent)
        trajectory_context = (
            trajectory_contexts[agent_id]
            if world.record_trajectory and agent_id in trajectory_contexts
            else None
        )
        action = tick_context.choose_action(agent, observation_snapshots.get(agent_id))
        live_action_mask = tick_context.action_mask(agent)
        if trajectory_context is not None:
            moved, action_outcome = runtime_actions.resolve_action_with_outcome(
                world,
                agent,
                action,
                observation_action_mask=trajectory_context["action_mask"],
                resolution_action_mask=live_action_mask,
                resolution_context=tick_context.action_resolution_context(agent),
            )
            resolved_action = str(action_outcome["resolved_action"])
        else:
            moved = runtime_actions.resolve_action(
                world,
                agent,
                action,
                resolution_action_mask=live_action_mask,
                resolution_context=tick_context.action_resolution_context(agent),
            )
            resolved_action = action if live_action_mask.get(action, False) else "stay"
            action_outcome = None
        runtime_lifecycle.apply_metabolism(
            world,
            agent,
            moved=moved,
            lifecycle_context=lifecycle_context,
        )
        runtime_lifecycle.apply_health_and_hazards(
            world,
            agent,
            moved=moved,
            lifecycle_context=lifecycle_context,
        )
        agent.age += 1
        if trajectory_context is not None:
            policy_metadata = tick_context.policy_metadata()
            trajectory_context.update(
                {
                    "requested_action": action,
                    "action_source": policy_metadata["action_source"],
                    "policy_id": policy_metadata["policy_id"],
                    "policy_version": policy_metadata["policy_version"],
                    "policy_decision_diagnostics": policy_metadata.get(
                        "decision_diagnostics"
                    ),
                    "resolution_action_mask": live_action_mask,
                    "resolved_action": resolved_action,
                    "moved": moved,
                    "action_outcome": action_outcome,
                }
            )
            pending_trajectory_records.append(trajectory_context)
            acted_trajectory_agent_ids.add(agent_id)
        tick_context.invalidate_biotic_state()

    births_this_tick = runtime_reproduction.run_reproduction_phase(
        world,
        context=tick_context.reproduction_context(),
    )
    post_reproduction_alive = len(world.alive_agents())

    for agent_id in sorted(world.agents):
        agent = world.agents[agent_id]
        if agent.alive and runtime_lifecycle.should_die(world, agent):
            tick_context.kill_agent(
                agent,
                cause=runtime_lifecycle.death_cause(world, agent),
            )

    if world.record_trajectory:
        append_passive_trajectory_contexts(
            world,
            trajectory_contexts,
            acted_trajectory_agent_ids,
            pending_trajectory_records,
        )
        tick_context.finalize_trajectory_decisions(pending_trajectory_records)
    tick_context.reconcile_policy_live_agent_ids(
        tuple(sorted(agent.agent_id for agent in world.alive_agents()))
    )

    deaths_this_tick = world.deaths - deaths_before_tick
    alive_count = len(world.alive_agents())
    world.peak_alive_agents = max(world.peak_alive_agents, alive_count)
    runtime_capacity.record_carrying_capacity_tick(
        world,
        post_reproduction_alive=post_reproduction_alive,
        final_alive=alive_count,
        births=births_this_tick,
        deaths=deaths_this_tick,
    )
    tick_context.record_animal_resource_opportunity_tick(
        opportunity_meat_mode_counts,
        opportunity_reachability_by_meat_mode,
        opportunity_resource_presence,
    )
    tick_context.emit(
        EventType.TICK_COMPLETED,
        agent_id=None,
        data={
            "alive_agents": alive_count,
            "births": births_this_tick,
            "deaths": deaths_this_tick,
            "season": tick_context.season_state()["name"],
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
    world.tick_action_order = []
    world.tick_birth_pairs = []
    world.tick_reproduction_parent_ids = set()
    world.tick_reproduction_parent_child_groups = []
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
        runtime_feeding.empty_grouped_animal_resource_consumption_counts(
            meat_mode_codes
        )
    )
    world.tick_hazard_exposure_agents = set()


def build_trajectory_contexts(
    world: Any,
    tick_start_alive: Sequence[Any],
    observation_snapshots: dict[int, object],
    *,
    tick_context: TickPhaseContext,
) -> dict[int, dict[str, object]]:
    if not world.record_trajectory:
        return {}
    return {
        agent.agent_id: tick_context.begin_trajectory_decision(
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
        passive_outcome = runtime_actions.base_action_outcome(
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
                "policy_decision_diagnostics": None,
                "resolution_action_mask": action_mask,
                "resolved_action": "stay",
                "moved": False,
                "action_outcome": passive_outcome,
            }
        )
        pending_trajectory_records.append(trajectory_context)


def deterministic_agent_turn_order(
    agent_ids: Sequence[int] | Any,
    *,
    seed: int,
    tick: int,
) -> tuple[int, ...]:
    ordered_ids = tuple(sorted(int(agent_id) for agent_id in agent_ids))
    return tuple(
        sorted(
            ordered_ids,
            key=lambda agent_id: (
                _turn_order_hash(seed=seed, tick=tick, agent_id=agent_id),
                agent_id,
            ),
        )
    )


def _turn_order_hash(*, seed: int, tick: int, agent_id: int) -> int:
    payload = f"{int(seed)}:{int(tick)}:{int(agent_id)}".encode("ascii")
    digest = hashlib.blake2b(
        payload,
        digest_size=8,
        person=b"turnorder",
    ).digest()
    return int.from_bytes(digest, byteorder="big")
