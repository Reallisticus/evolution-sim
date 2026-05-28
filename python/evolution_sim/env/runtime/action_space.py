from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evolution_sim.env.runtime.action_contract import (
    ACTION_NAMES,
    action_names,
)


@dataclass(frozen=True, slots=True)
class MovementActionAvailability:
    action: str
    dx: int
    dy: int
    can_move: bool
    can_attack: bool


@dataclass(frozen=True, slots=True)
class ActionMaskContext:
    action_names: tuple[str, ...]
    can_eat: bool
    can_drink: bool
    movement: tuple[MovementActionAvailability, ...]
    communication_action_available: dict[str, bool]


def build_action_mask(context: ActionMaskContext) -> dict[str, bool]:
    return build_action_mask_from_context(context)


def build_action_mask_from_context(context: ActionMaskContext) -> dict[str, bool]:
    mask = {action: False for action in context.action_names}
    mask["stay"] = True
    mask["eat"] = context.can_eat
    mask["drink"] = context.can_drink

    for option in context.movement:
        mask[option.action] = option.can_move
        mask[option.action.replace("move_", "attack_")] = option.can_attack
    for action, available in context.communication_action_available.items():
        if action in mask:
            mask[action] = bool(available)
    return mask


def action_names_for_config(world: Any) -> tuple[str, ...]:
    return tuple(action_names(world.config.signals))


def can_eat_from_values(
    *,
    plant_intake_useful: bool,
    plant_food_value: float,
    can_consume_fresh_kill: bool,
    fresh_kill_intake_useful: bool,
    fresh_kill_food_value: float,
    can_consume_carcass: bool,
    carcass_intake_useful: bool,
    carcass_food_value: float,
    adjacent_carcass_available: bool,
) -> bool:
    """Return the utility-shaped eat affordance for the action mask.

    This intentionally answers "would the resolver accept a currently useful
    intake action?" rather than "is any edible matter physically present?".
    """
    if plant_intake_useful and plant_food_value > 0:
        return True
    if (
        can_consume_fresh_kill
        and fresh_kill_intake_useful
        and fresh_kill_food_value > 0
    ):
        return True
    return (
        can_consume_carcass
        and carcass_intake_useful
        and (
            carcass_food_value > 0
            or adjacent_carcass_available
        )
    )
