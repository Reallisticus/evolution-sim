from __future__ import annotations

from typing import Any

from evolution_sim.env.runtime.state import Agent


MOVEMENT_ACTIONS: tuple[str, ...] = (
    "move_north",
    "move_south",
    "move_east",
    "move_west",
)
ATTACK_ACTIONS: tuple[str, ...] = tuple(
    action.replace("move_", "attack_") for action in MOVEMENT_ACTIONS
)
ACTION_NAMES: tuple[str, ...] = ("stay", "eat", "drink", *MOVEMENT_ACTIONS, *ATTACK_ACTIONS)


def build_action_mask(world: Any, agent: Agent) -> dict[str, bool]:
    profile = world._trophic_profile(agent)
    tile = world.grid[agent.y][agent.x]
    mask = {action: False for action in ACTION_NAMES}
    mask["stay"] = True
    mask["eat"] = _can_eat(world, agent, tile, profile)
    mask["drink"] = world._has_water_access(agent)

    for action, dx, dy in world._movement_actions():
        x = agent.x + dx
        y = agent.y + dy
        mask[action] = world._can_move_to(x, y)
        mask[action.replace("move_", "attack_")] = _can_attack_tile(world, agent, x, y)
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
        and world._carcass_food_value(agent, tile, profile) > 0
    )


def _can_attack_tile(world: Any, agent: Agent, x: int, y: int) -> bool:
    if not world._can_attack(agent) or not world._in_bounds(x, y):
        return False
    target_id = world.grid[y][x].occupant_id
    if target_id is None or target_id == agent.agent_id:
        return False
    target = world.agents.get(target_id)
    return target is not None and target.alive
