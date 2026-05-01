from __future__ import annotations

from typing import Any

from evolution_sim.env.runtime.action_contract import (
    ACTION_NAMES,
    ATTACK_ACTIONS,
    MOVEMENT_ACTIONS,
    action_names,
    communication_action_names,
)
import evolution_sim.env.runtime.signals as runtime_signals
from evolution_sim.env.runtime.state import Agent


def build_action_mask(world: Any, agent: Agent) -> dict[str, bool]:
    profile = world._trophic_profile(agent)
    tile = world.grid[agent.y][agent.x]
    mask = {action: False for action in action_names(world.config.signals)}
    mask["stay"] = True
    mask["eat"] = _can_eat(world, agent, tile, profile)
    mask["drink"] = world._has_water_access(agent)

    for action, dx, dy in world._movement_actions():
        x = agent.x + dx
        y = agent.y + dy
        mask[action] = world._can_move_to(x, y)
        mask[action.replace("move_", "attack_")] = _can_attack_tile(world, agent, x, y)
    for action in communication_action_names(world.config.signals):
        mask[action] = runtime_signals.communication_signal_action_available(
            world, agent, action
        )
    return mask


def _can_eat(world: Any, agent: Agent, tile: Any, profile: Any) -> bool:
    if world._plant_intake_useful(agent) and world._plant_food_value(agent, tile, profile) > 0:
        return True
    if (
        world._can_consume_fresh_kill(agent)
        and world._fresh_kill_intake_useful(agent)
        and world._fresh_kill_food_value(agent, tile, profile) > 0
    ):
        return True
    return (
        world._can_consume_carcass(agent)
        and world._carcass_intake_useful(agent)
        and (
            world._carcass_food_value(agent, tile, profile) > 0
            or world._adjacent_scavenger_carcass_target(agent, profile) is not None
        )
    )


def _can_attack_tile(world: Any, agent: Agent, x: int, y: int) -> bool:
    if not world._can_attack(agent) or not world._in_bounds(x, y):
        return False
    target_id = world.grid[y][x].occupant_id
    if target_id is None or target_id == agent.agent_id:
        return False
    target = world.agents.get(target_id)
    return target is not None and target.alive
