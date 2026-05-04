from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable

from evolution_sim.env.runtime.derived import DerivedTileMemo
from evolution_sim.env.runtime.state import Agent, Tile, TrophicProfile
import evolution_sim.env.runtime.trajectory as runtime_trajectory


@dataclass(frozen=True, slots=True)
class ActionResolutionContext:
    movement_actions: tuple[tuple[str, int, int], ...]
    eat_action_outcome: Callable[[], dict[str, object] | None]
    drink_action_outcome: Callable[[], dict[str, object] | None]
    signal_action_outcome: Callable[[str], dict[str, object]]
    attack_action_outcome: Callable[
        [str, int, int],
        tuple[dict[str, object], dict[str, object] | None],
    ]
    move_action: Callable[[str, int, int], tuple[bool, dict[str, object] | None]]


@dataclass(frozen=True, slots=True)
class ActionScoringContext:
    width: int
    height: int
    default_vision_radius: int
    animal_channel_threshold: float
    hunter_vulnerability_weight: float
    season: str
    grid: Sequence[Sequence[Tile]]
    agents: Mapping[int, Agent]
    movement_actions: tuple[tuple[str, int, int], ...]
    tile_memo: DerivedTileMemo
    random_choice: Callable[[list[str]], str]
    set_policy_action_source: Callable[[str], None]
    profile_for: Callable[[Agent], TrophicProfile]
    trophic_role: Callable[[Agent], str]
    energy_ratio: Callable[[Agent], float]
    hydration_ratio: Callable[[Agent], float]
    health_ratio: Callable[[Agent], float]
    plant_intake_useful: Callable[[Agent], bool]
    plant_food_value: Callable[..., float]
    can_consume_fresh_kill: Callable[[Agent], bool]
    fresh_kill_intake_useful: Callable[[Agent], bool]
    fresh_kill_food_value: Callable[..., float]
    can_consume_carcass: Callable[[Agent], bool]
    carcass_intake_useful: Callable[[Agent], bool]
    carcass_food_value: Callable[..., float]
    has_water_access: Callable[[Agent], bool]
    can_attack: Callable[[Agent], bool]
    attack_value: Callable[[Agent, Agent, TrophicProfile | None], float]
    prey_vulnerability: Callable[[Agent], float]
    biotic_field_score: Callable[..., float]
    can_move_to: Callable[[int, int], bool]
    in_bounds: Callable[[int, int], bool]
    water_access_reason: Callable[[int, int], str]
    refuge_score: Callable[[int, int], float]
    hazard_at: Callable[[int, int], tuple[str, float]]
    soft_refuge_reason: Callable[[int, int], str]
    terrain_preference_score: Callable[[Agent, str], float]
    field_preference_score: Callable[..., float]


@dataclass(slots=True)
class DecisionContext:
    season: str
    water_urgency: float
    food_urgency: float
    profile: TrophicProfile
    tile_memo: DerivedTileMemo
    scoring_context: ActionScoringContext


def _resolve_scoring_context(
    world: Any,
    agent: Agent,
    *,
    scoring_context: ActionScoringContext | None = None,
    context: DecisionContext | None = None,
) -> ActionScoringContext:
    if scoring_context is not None:
        return scoring_context
    if context is not None:
        return context.scoring_context
    raise ValueError("scoring_context or context is required")


def build_decision_context(
    world: Any,
    agent: Agent,
    *,
    profile: TrophicProfile | None = None,
    season: str | None = None,
    scoring_context: ActionScoringContext | None = None,
) -> DecisionContext:
    scoring = _resolve_scoring_context(world, agent, scoring_context=scoring_context)
    chosen_profile = profile or scoring.profile_for(agent)
    chosen_season = season or scoring.season
    return DecisionContext(
        season=chosen_season,
        water_urgency=max(0.0, 1.0 - scoring.hydration_ratio(agent)),
        food_urgency=max(0.0, 1.0 - scoring.energy_ratio(agent)),
        profile=chosen_profile,
        tile_memo=scoring.tile_memo,
        scoring_context=scoring,
    )


def choose_action(
    world: Any,
    agent: Agent,
    *,
    scoring_context: ActionScoringContext | None = None,
) -> str:
    scoring = _resolve_scoring_context(world, agent, scoring_context=scoring_context)
    tile = scoring.grid[agent.y][agent.x]
    energy_ratio = scoring.energy_ratio(agent)
    hydration_ratio = scoring.hydration_ratio(agent)
    profile = scoring.profile_for(agent)
    context = build_decision_context(world, agent, profile=profile, scoring_context=scoring)
    plant_value = scoring.plant_food_value(agent, tile, profile)
    fresh_kill_value = (
        scoring.fresh_kill_food_value(agent, tile, profile)
        if scoring.can_consume_fresh_kill(agent)
        else 0.0
    )
    carcass_value = (
        scoring.carcass_food_value(agent, tile, profile)
        if scoring.can_consume_carcass(agent)
        else 0.0
    )

    if fresh_kill_value > 0 and scoring.fresh_kill_intake_useful(agent) and (
        profile.role == "carnivore"
        or profile.meat_mode == "hunter"
        or fresh_kill_value >= max(plant_value, carcass_value) * (0.92 if energy_ratio < 0.72 else 1.04)
    ):
        return "eat"
    if carcass_value > 0 and scoring.carcass_intake_useful(agent) and (
        profile.role == "carnivore"
        or profile.meat_mode == "scavenger"
        or carcass_value >= plant_value * (0.92 if energy_ratio < 0.72 else 1.06)
    ):
        return "eat"
    if (
        scoring.has_water_access(agent)
        and hydration_ratio < 0.82
        and hydration_ratio <= energy_ratio + 0.08
    ):
        return "drink"
    prefer_biotic = (
        profile.role == "carnivore"
        or profile.meat_mode == "hunter"
        or profile.animal_drive > profile.plant_drive * 1.12
    )
    if prefer_biotic:
        biotic_target = best_visible_biotic_action(world,
            agent,
            profile=profile,
            context=context,
            scoring_context=scoring,
        )
        if biotic_target is not None:
            return biotic_target
    if plant_value > 0 and scoring.plant_intake_useful(agent) and energy_ratio < 0.84 and (
        carcass_value <= 0
        or profile.role == "herbivore"
        or plant_value >= carcass_value * (0.9 if profile.role == "herbivore" else 1.04)
    ):
        return "eat"

    if not prefer_biotic:
        biotic_target = best_visible_biotic_action(world,
            agent,
            profile=profile,
            context=context,
            scoring_context=scoring,
        )
        if biotic_target is not None:
            return biotic_target

    target = best_visible_action_toward_need(
        world,
        agent,
        profile=profile,
        context=context,
        scoring_context=scoring,
    )
    if target is not None:
        return target

    scoring.set_policy_action_source("heuristic_fallback")
    return scoring.random_choice(
        ["stay", "move_north", "move_south", "move_east", "move_west"]
    )


def invalid_action_reason(
    action: str,
    observation_action_mask: dict[str, bool],
    resolution_action_mask: dict[str, bool],
) -> str | None:
    observation_valid = bool(observation_action_mask.get(action, False))
    resolution_valid = bool(resolution_action_mask.get(action, False))
    if resolution_valid:
        return None
    if not observation_valid:
        return "not_in_observation_or_resolution_mask"
    return "not_in_resolution_action_mask"


def base_action_outcome(
    *,
    requested_action: str,
    resolved_action: str,
    observation_action_mask: dict[str, bool],
    resolution_action_mask: dict[str, bool],
) -> dict[str, object]:
    return runtime_trajectory.empty_action_outcome(
        requested_action=requested_action,
        resolved_action=resolved_action,
        observation_action_valid=bool(
            observation_action_mask.get(requested_action, False)
        ),
        resolution_action_valid=bool(
            resolution_action_mask.get(requested_action, False)
        ),
        invalid_reason=invalid_action_reason(
            requested_action,
            observation_action_mask,
            resolution_action_mask,
        ),
    )


def resolve_action(
    world: Any,
    agent: Agent,
    action: str,
    *,
    resolution_action_mask: dict[str, bool] | None = None,
    resolution_context: ActionResolutionContext | None = None,
) -> bool:
    moved, _ = resolve_action_with_outcome(
        world,
        agent,
        action,
        resolution_action_mask=resolution_action_mask,
        resolution_context=resolution_context,
    )
    return moved


def resolve_action_with_outcome(
    world: Any,
    agent: Agent,
    action: str,
    *,
    observation_action_mask: dict[str, bool] | None = None,
    resolution_action_mask: dict[str, bool] | None = None,
    resolution_context: ActionResolutionContext | None = None,
) -> tuple[bool, dict[str, object]]:
    if resolution_action_mask is None:
        raise ValueError("resolution_action_mask is required")
    if resolution_context is None:
        raise ValueError("resolution_context is required")
    resolution_mask = resolution_action_mask
    observation_mask = observation_action_mask or resolution_mask
    context = resolution_context
    resolved_action = action if resolution_mask.get(action, False) else "stay"
    outcome = base_action_outcome(
        requested_action=action,
        resolved_action=resolved_action,
        observation_action_mask=observation_mask,
        resolution_action_mask=resolution_mask,
    )
    if resolved_action == "stay":
        return False, outcome
    if action == "eat":
        feeding = context.eat_action_outcome()
        if feeding is not None:
            outcome["feeding"] = feeding
        return False, outcome
    if action == "drink":
        drinking = context.drink_action_outcome()
        if drinking is not None:
            outcome["drinking"] = drinking
        return False, outcome
    if action == "stay":
        return False, outcome
    if action.startswith("signal_"):
        outcome["signal"] = context.signal_action_outcome(action)
        return False, outcome
    if action.startswith("attack_"):
        return _resolve_attack_action(context, action, outcome)
    if action.startswith("move_"):
        return _resolve_movement_action(context, action, outcome)
    return False, outcome


def _resolve_attack_action(
    context: ActionResolutionContext,
    action: str,
    outcome: dict[str, object],
) -> tuple[bool, dict[str, object]]:
    for candidate, dx, dy in context.movement_actions:
        if action != candidate.replace("move_", "attack_"):
            continue
        attack, feeding = context.attack_action_outcome(action, dx, dy)
        outcome["attack"] = attack
        if feeding is not None:
            outcome["feeding"] = feeding
        return False, outcome
    return False, outcome


def _resolve_movement_action(
    context: ActionResolutionContext,
    action: str,
    outcome: dict[str, object],
) -> tuple[bool, dict[str, object]]:
    for candidate, dx, dy in context.movement_actions:
        if candidate != action:
            continue
        moved, movement = context.move_action(action, dx, dy)
        if movement is not None:
            outcome["movement"] = movement
        return moved, outcome
    return False, outcome


def best_visible_biotic_action(
    world: Any,
    agent: Agent,
    profile: TrophicProfile | None = None,
    context: DecisionContext | None = None,
    scoring_context: ActionScoringContext | None = None,
) -> str | None:
    scoring = _resolve_scoring_context(
        world,
        agent,
        scoring_context=scoring_context,
        context=context,
    )
    profile = profile or scoring.profile_for(agent)
    if profile.animal_share < scoring.animal_channel_threshold:
        return None

    context = context or build_decision_context(
        world,
        agent,
        profile=profile,
        scoring_context=scoring,
    )
    energy_ratio = scoring.energy_ratio(agent)
    tile = scoring.grid[agent.y][agent.x]
    plant_value = scoring.plant_food_value(agent, tile, profile)
    fresh_kill_action = best_visible_fresh_kill_action(world,
        agent,
        profile=profile,
        context=context,
        scoring_context=scoring,
    )
    carrion_action = best_visible_carrion_action(world,
        agent,
        profile=profile,
        context=context,
        scoring_context=scoring,
    )
    attack_action = best_adjacent_attack_action(
        world,
        agent,
        profile=profile,
        scoring_context=scoring,
    )
    prey_move_action = best_visible_prey_action(world,
        agent,
        profile=profile,
        context=context,
        scoring_context=scoring,
    )

    if profile.role == "carnivore":
        if profile.meat_mode == "hunter":
            return attack_action or prey_move_action or fresh_kill_action or carrion_action
        if profile.meat_mode == "scavenger":
            return carrion_action or fresh_kill_action or attack_action or prey_move_action
        if profile.hunter_drive >= profile.scavenger_drive * 0.92:
            return attack_action or prey_move_action or fresh_kill_action or carrion_action
        return carrion_action or fresh_kill_action or attack_action or prey_move_action

    if profile.meat_mode == "hunter":
        if attack_action is not None and (
            profile.hunter_drive >= max(0.22, profile.plant_drive * 0.72)
            or plant_value < 0.12
        ):
            return attack_action
        if prey_move_action is not None and (
            profile.hunter_drive >= profile.plant_drive * 0.7
            or energy_ratio < 0.72
        ):
            return prey_move_action
        if fresh_kill_action is not None:
            return fresh_kill_action
        if carrion_action is not None and energy_ratio < 0.56:
            return carrion_action

    if fresh_kill_action is not None and (
        profile.hunter_drive >= profile.plant_drive * 0.9
        or energy_ratio < 0.62
    ):
        return fresh_kill_action
    if carrion_action is not None and (
        energy_ratio < 0.7
        or plant_value < 0.08
        or profile.scavenger_drive >= profile.plant_drive
    ):
        return carrion_action
    if attack_action is not None and profile.hunter_drive >= profile.plant_drive * 0.88:
        return attack_action
    if (
        attack_action is not None
        and profile.hunter_drive >= 0.24
        and energy_ratio < 0.6
        and plant_value < 0.1
    ):
        return attack_action
    if (
        prey_move_action is not None
        and profile.hunter_drive >= 0.26
        and energy_ratio < 0.54
        and plant_value < 0.08
    ):
        return prey_move_action
    return None


def best_visible_fresh_kill_action(
    world: Any,
    agent: Agent,
    profile: TrophicProfile | None = None,
    context: DecisionContext | None = None,
    scoring_context: ActionScoringContext | None = None,
) -> str | None:
    scoring = _resolve_scoring_context(
        world,
        agent,
        scoring_context=scoring_context,
        context=context,
    )
    if not scoring.can_consume_fresh_kill(agent):
        return None
    if not scoring.fresh_kill_intake_useful(agent):
        return None
    profile = profile or scoring.profile_for(agent)
    context = context or build_decision_context(
        world,
        agent,
        profile=profile,
        scoring_context=scoring,
    )
    radius = max(
        1,
        scoring.default_vision_radius
        + (2 if profile.role == "carnivore" or profile.hunter_drive >= 0.18 else 0),
    )
    best_target: tuple[int, int] | None = None
    best_score = float("-inf")
    for y in range(max(0, agent.y - radius), min(scoring.height, agent.y + radius + 1)):
        for x in range(max(0, agent.x - radius), min(scoring.width, agent.x + radius + 1)):
            distance = abs(x - agent.x) + abs(y - agent.y)
            if distance > radius:
                continue
            tile = scoring.grid[y][x]
            if tile.terrain == "water" or tile.fresh_kill_energy <= 0:
                continue
            if distance == 0:
                return "eat"
            if tile.occupant_id is not None:
                continue
            score = scoring.fresh_kill_food_value(agent, tile, profile)
            score += candidate_tile_score(world,
                agent,
                x,
                y,
                context.season,
                context.water_urgency,
                context.food_urgency,
                profile=profile,
                include_plant_channel=False,
                include_vegetation_channel=False,
                include_fresh_kill_channel=False,
                include_carcass_channel=False,
                context=context,
                scoring_context=scoring,
            )
            score -= distance * 0.05
            if score > best_score:
                best_score = score
                best_target = (x, y)
    if best_target is None:
        return None
    return step_toward_target(world,
        agent,
        best_target[0],
        best_target[1],
        context.season,
        profile=profile,
        include_plant_channel=False,
        include_vegetation_channel=False,
        include_fresh_kill_channel=False,
        include_carcass_channel=False,
        context=context,
        scoring_context=scoring,
    )


def best_visible_carrion_action(
    world: Any,
    agent: Agent,
    profile: TrophicProfile | None = None,
    context: DecisionContext | None = None,
    scoring_context: ActionScoringContext | None = None,
) -> str | None:
    scoring = _resolve_scoring_context(
        world,
        agent,
        scoring_context=scoring_context,
        context=context,
    )
    if not scoring.can_consume_carcass(agent):
        return None
    if not scoring.carcass_intake_useful(agent):
        return None
    profile = profile or scoring.profile_for(agent)
    context = context or build_decision_context(
        world,
        agent,
        profile=profile,
        scoring_context=scoring,
    )
    radius = max(
        1,
        scoring.default_vision_radius
        + (
            2
            if profile.role == "carnivore" or profile.scavenger_drive >= 0.18
            else 0
        ),
    )
    best_target: tuple[int, int] | None = None
    best_score = float("-inf")
    for y in range(max(0, agent.y - radius), min(scoring.height, agent.y + radius + 1)):
        for x in range(max(0, agent.x - radius), min(scoring.width, agent.x + radius + 1)):
            distance = abs(x - agent.x) + abs(y - agent.y)
            if distance > radius:
                continue
            tile = scoring.grid[y][x]
            if tile.terrain == "water" or tile.carcass_energy <= 0:
                continue
            if distance == 0:
                return "eat" if scoring.carcass_intake_useful(agent) else None
            if tile.occupant_id is not None:
                continue
            score = scoring.carcass_food_value(agent, tile, profile)
            score += candidate_tile_score(world,
                agent,
                x,
                y,
                context.season,
                context.water_urgency,
                context.food_urgency,
                profile=profile,
                include_plant_channel=False,
                include_vegetation_channel=False,
                include_fresh_kill_channel=False,
                include_carcass_channel=False,
                context=context,
                scoring_context=scoring,
            )
            score -= distance * 0.05
            if score > best_score:
                best_score = score
                best_target = (x, y)
    if best_target is None:
        return None
    return step_toward_target(world,
        agent,
        best_target[0],
        best_target[1],
        context.season,
        profile=profile,
        include_plant_channel=False,
        include_vegetation_channel=False,
        include_fresh_kill_channel=False,
        include_carcass_channel=False,
        context=context,
        scoring_context=scoring,
    )


def best_adjacent_attack_action(
    world: Any,
    agent: Agent,
    profile: TrophicProfile | None = None,
    scoring_context: ActionScoringContext | None = None,
) -> str | None:
    scoring = _resolve_scoring_context(
        world,
        agent,
        scoring_context=scoring_context,
    )
    if not scoring.can_attack(agent):
        return None
    profile = profile or scoring.profile_for(agent)
    best_action: str | None = None
    best_score = float("-inf")
    for action, dx, dy in scoring.movement_actions:
        target = (
            scoring.agents.get(scoring.grid[agent.y + dy][agent.x + dx].occupant_id)
            if scoring.in_bounds(agent.x + dx, agent.y + dy)
            else None
        )
        if target is None or not target.alive or target.agent_id == agent.agent_id:
            continue
        score = scoring.attack_value(agent, target, profile)
        if score <= 0.005:
            continue
        target_role = scoring.trophic_role(target)
        if target_role == "herbivore":
            score += 0.06
        if scoring.health_ratio(target) > scoring.health_ratio(agent) + 0.16:
            score -= 0.12
        if score > best_score:
            best_score = score
            best_action = action.replace("move_", "attack_")
    return best_action


def best_visible_prey_action(
    world: Any,
    agent: Agent,
    profile: TrophicProfile | None = None,
    context: DecisionContext | None = None,
    scoring_context: ActionScoringContext | None = None,
) -> str | None:
    scoring = _resolve_scoring_context(
        world,
        agent,
        scoring_context=scoring_context,
        context=context,
    )
    if not scoring.can_attack(agent):
        return None
    profile = profile or scoring.profile_for(agent)
    context = context or build_decision_context(
        world,
        agent,
        profile=profile,
        scoring_context=scoring,
    )
    radius = max(
        1,
        scoring.default_vision_radius
        + (2 if profile.role == "carnivore" or profile.hunter_drive >= 0.18 else 0),
    )
    best_target: tuple[int, int] | None = None
    best_score = float("-inf")
    for y in range(max(0, agent.y - radius), min(scoring.height, agent.y + radius + 1)):
        for x in range(max(0, agent.x - radius), min(scoring.width, agent.x + radius + 1)):
            distance = abs(x - agent.x) + abs(y - agent.y)
            if distance <= 1 or distance > radius:
                continue
            tile = scoring.grid[y][x]
            target_id = tile.occupant_id
            if tile.terrain == "water" or target_id is None or target_id == agent.agent_id:
                continue
            target = scoring.agents.get(target_id)
            if target is None or not target.alive:
                continue
            target_role = scoring.trophic_role(target)
            score = scoring.attack_value(agent, target, profile)
            score += (
                scoring.prey_vulnerability(target)
                * scoring.hunter_vulnerability_weight
            )
            score += candidate_tile_score(world,
                agent,
                x,
                y,
                context.season,
                context.water_urgency,
                context.food_urgency,
                profile=profile,
                include_plant_channel=False,
                include_vegetation_channel=False,
                include_fresh_kill_channel=False,
                include_carcass_channel=False,
                context=context,
                scoring_context=scoring,
            )
            if target_role == "herbivore":
                score += 0.08
            score -= distance * 0.025
            if score > best_score:
                best_score = score
                best_target = (x, y)
    if best_target is None:
        return None
    return step_toward_target(world,
        agent,
        best_target[0],
        best_target[1],
        context.season,
        profile=profile,
        include_plant_channel=False,
        include_vegetation_channel=False,
        include_fresh_kill_channel=False,
        include_carcass_channel=False,
        context=context,
        scoring_context=scoring,
    )


def biotic_opportunity_score(
    world: Any,
    agent: Agent,
    x: int,
    y: int,
    profile: TrophicProfile,
    context: DecisionContext | None = None,
    scoring_context: ActionScoringContext | None = None,
) -> float:
    scoring = _resolve_scoring_context(
        world,
        agent,
        scoring_context=scoring_context,
        context=context,
    )
    biotic_state = (
        context.tile_memo.current_biotic_state() if context is not None else None
    )
    return scoring.biotic_field_score(agent, x, y, profile, biotic_state=biotic_state)


def step_toward_target(
    world: Any,
    agent: Agent,
    target_x: int,
    target_y: int,
    season: str,
    profile: TrophicProfile | None = None,
    include_plant_channel: bool = True,
    include_vegetation_channel: bool = True,
    include_fresh_kill_channel: bool = True,
    include_carcass_channel: bool = True,
    include_animal_signal: bool = True,
    context: DecisionContext | None = None,
    scoring_context: ActionScoringContext | None = None,
) -> str | None:
    scoring = _resolve_scoring_context(
        world,
        agent,
        scoring_context=scoring_context,
        context=context,
    )
    profile = profile or scoring.profile_for(agent)
    context = context or build_decision_context(
        world,
        agent,
        profile=profile,
        season=season,
        scoring_context=scoring,
    )
    water_urgency = context.water_urgency
    food_urgency = context.food_urgency
    current_distance = abs(target_x - agent.x) + abs(target_y - agent.y)
    best_action: str | None = None
    best_score = float("-inf")
    for action, dx, dy in scoring.movement_actions:
        x = agent.x + dx
        y = agent.y + dy
        if not scoring.can_move_to(x, y):
            continue
        next_distance = abs(target_x - x) + abs(target_y - y)
        if next_distance >= current_distance:
            continue
        score = candidate_tile_score(world,
            agent,
            x,
            y,
            season,
            water_urgency,
            food_urgency,
            profile=profile,
            include_plant_channel=include_plant_channel,
            include_vegetation_channel=include_vegetation_channel,
            include_fresh_kill_channel=include_fresh_kill_channel,
            include_carcass_channel=include_carcass_channel,
            include_animal_signal=include_animal_signal,
            context=context,
            scoring_context=scoring,
        )
        score -= next_distance * 0.04
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def candidate_tile_score(
    world: Any,
    agent: Agent,
    x: int,
    y: int,
    season: str,
    water_urgency: float,
    food_urgency: float,
    profile: TrophicProfile | None = None,
    include_plant_channel: bool = True,
    include_vegetation_channel: bool = True,
    include_fresh_kill_channel: bool = True,
    include_carcass_channel: bool = True,
    include_animal_signal: bool = True,
    context: DecisionContext | None = None,
    scoring_context: ActionScoringContext | None = None,
) -> float:
    scoring = _resolve_scoring_context(
        world,
        agent,
        scoring_context=scoring_context,
        context=context,
    )
    profile = profile or scoring.profile_for(agent)
    tile_memo = context.tile_memo if context is not None else None
    tile = scoring.grid[y][x]
    score = 0.0
    water_reason = (
        tile_memo.water_reason(x, y)
        if tile_memo is not None
        else scoring.water_access_reason(x, y)
    )
    refuge_score = (
        tile_memo.refuge_score(x, y)
        if tile_memo is not None
        else scoring.refuge_score(x, y)
    )
    hazard_type, hazard_level = (
        tile_memo.hazard(x, y) if tile_memo is not None else scoring.hazard_at(x, y)
    )
    if water_reason != "none":
        hard_water_bonus = {
            "adjacent_water": 1.52,
            "wetland": 1.44,
            "flooded": 1.14,
        }.get(water_reason, 1.4)
        score += hard_water_bonus * water_urgency * agent.genome.water_efficiency
    else:
        score += refuge_score * 0.16 * water_urgency
        if scoring.soft_refuge_reason(x, y) == "canopy_refuge":
            score += 0.06 * water_urgency
    if include_plant_channel and profile.plant_drive > 0:
        score += scoring.plant_food_value(agent, tile, profile, x=x, y=y) * (
            0.28 + food_urgency * 0.84
        )
    score += scoring.terrain_preference_score(agent, tile.terrain)
    score += scoring.field_preference_score(
        agent,
        x,
        y,
        season,
        water_urgency,
        food_urgency,
        profile=profile,
        tile_memo=tile_memo,
    )
    if include_vegetation_channel and profile.plant_drive > 0:
        score += tile.vegetation * profile.plant_drive * (0.05 + food_urgency * 0.14)
        score += (1.0 - tile.recovery_debt) * profile.plant_drive * 0.08
    score -= hazard_level * (0.14 + (1.0 - scoring.health_ratio(agent)) * 0.34)
    if hazard_type == "exposure" and (
        tile_memo.soft_refuge_reason(x, y) if tile_memo is not None else scoring.soft_refuge_reason(x, y)
    ) == "canopy_refuge":
        score += 0.05
    if (
        include_fresh_kill_channel
        and tile.fresh_kill_energy > 0
        and scoring.can_consume_fresh_kill(agent)
    ):
        score += scoring.fresh_kill_food_value(agent, tile, profile) * (0.52 + food_urgency * 1.12)
    if include_carcass_channel and tile.carcass_energy > 0 and scoring.can_consume_carcass(agent):
        score += scoring.carcass_food_value(agent, tile, profile) * (0.4 + food_urgency * 1.1)
    if include_animal_signal and (profile.role == "herbivore" or profile.animal_drive > 0):
        score += biotic_opportunity_score(
            world,
            agent,
            x,
            y,
            profile,
            context=context,
            scoring_context=scoring,
        ) * (
            0.22 + food_urgency * 0.78
        )
    return score


def best_visible_action_toward_need(
    world: Any,
    agent: Agent,
    profile: TrophicProfile | None = None,
    context: DecisionContext | None = None,
    scoring_context: ActionScoringContext | None = None,
) -> str | None:
    scoring = _resolve_scoring_context(
        world,
        agent,
        scoring_context=scoring_context,
        context=context,
    )
    profile = profile or scoring.profile_for(agent)
    context = context or build_decision_context(
        world,
        agent,
        profile=profile,
        scoring_context=scoring,
    )
    water_urgency = context.water_urgency
    food_urgency = context.food_urgency
    season = context.season
    radius = max(1, scoring.default_vision_radius)
    best_target: tuple[int, int] | None = None
    best_target_score = float("-inf")

    for y in range(max(0, agent.y - radius), min(scoring.height, agent.y + radius + 1)):
        for x in range(max(0, agent.x - radius), min(scoring.width, agent.x + radius + 1)):
            distance = abs(x - agent.x) + abs(y - agent.y)
            if distance == 0 or distance > radius:
                continue
            tile = scoring.grid[y][x]
            if tile.terrain == "water":
                continue
            if tile.occupant_id is not None and tile.occupant_id != agent.agent_id:
                continue
            score = candidate_tile_score(world,
                agent,
                x,
                y,
                season,
                water_urgency,
                food_urgency,
                profile=profile,
                context=context,
                scoring_context=scoring,
            )
            score -= distance * (0.05 + agent.genome.move_cost * 1.45)
            if score > best_target_score:
                best_target_score = score
                best_target = (x, y)

    if best_target is None:
        return None

    target_x, target_y = best_target
    current_distance = abs(target_x - agent.x) + abs(target_y - agent.y)
    best_action: str | None = None
    best_score = float("-inf")
    for action, dx, dy in scoring.movement_actions:
        x = agent.x + dx
        y = agent.y + dy
        if not scoring.can_move_to(x, y):
            continue
        next_distance = abs(target_x - x) + abs(target_y - y)
        if next_distance >= current_distance:
            continue
        score = candidate_tile_score(world,
            agent,
            x,
            y,
                season,
                water_urgency,
                food_urgency,
                profile=profile,
                context=context,
                scoring_context=scoring,
            )
        score -= next_distance * 0.04
        if score > best_score:
            best_score = score
            best_action = action

    return best_action
