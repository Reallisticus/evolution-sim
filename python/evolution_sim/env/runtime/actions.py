from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evolution_sim.env.runtime.derived import DerivedTileMemo
from evolution_sim.env.runtime.state import Agent, TrophicProfile


@dataclass(slots=True)
class DecisionContext:
    season: str
    water_urgency: float
    food_urgency: float
    profile: TrophicProfile
    tile_memo: DerivedTileMemo


def build_decision_context(
    world: Any,
    agent: Agent,
    *,
    profile: TrophicProfile | None = None,
    season: str | None = None,
) -> DecisionContext:
    chosen_profile = profile or world._trophic_profile(agent)
    chosen_season = season or world._season_state()["name"]
    return DecisionContext(
        season=chosen_season,
        water_urgency=max(0.0, 1.0 - world._hydration_ratio(agent)),
        food_urgency=max(0.0, 1.0 - world._energy_ratio(agent)),
        profile=chosen_profile,
        tile_memo=DerivedTileMemo(
            world=world,
            season=chosen_season,
            climate_state=world._climate_state(),
        ),
    )


def choose_action(world: Any, agent: Agent) -> str:
    tile = world.grid[agent.y][agent.x]
    energy_ratio = world._energy_ratio(agent)
    hydration_ratio = world._hydration_ratio(agent)
    profile = world._trophic_profile(agent)
    context = build_decision_context(world, agent, profile=profile)
    plant_value = world._plant_food_value(agent, tile, profile)
    fresh_kill_value = (
        world._fresh_kill_food_value(agent, tile, profile)
        if world._can_consume_fresh_kill(agent)
        else 0.0
    )
    carcass_value = (
        world._carcass_food_value(agent, tile, profile)
        if world._can_consume_carcass(agent)
        else 0.0
    )

    if fresh_kill_value > 0 and world._fresh_kill_intake_useful(agent) and (
        profile.role == "carnivore"
        or profile.meat_mode == "hunter"
        or fresh_kill_value >= max(plant_value, carcass_value) * (0.92 if energy_ratio < 0.72 else 1.04)
    ):
        return "eat"
    if carcass_value > 0 and world._carcass_intake_useful(agent) and (
        profile.role == "carnivore"
        or profile.meat_mode == "scavenger"
        or carcass_value >= plant_value * (0.92 if energy_ratio < 0.72 else 1.06)
    ):
        return "eat"
    if (
        world._has_water_access(agent)
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
        )
        if biotic_target is not None:
            return biotic_target
    if plant_value > 0 and world._plant_intake_useful(agent) and energy_ratio < 0.84 and (
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
        )
        if biotic_target is not None:
            return biotic_target

    target = best_visible_action_toward_need(world, agent, profile=profile, context=context)
    if target is not None:
        return target

    world._policy_action_source = "heuristic_fallback"
    return world.rng.choice(
        ["stay", "move_north", "move_south", "move_east", "move_west"]
    )


def best_visible_biotic_action(
    world: Any,
    agent: Agent,
    profile: TrophicProfile | None = None,
    context: DecisionContext | None = None,
) -> str | None:
    profile = profile or world._trophic_profile(agent)
    if profile.animal_share < world.config.trophic.animal_channel_threshold:
        return None

    context = context or build_decision_context(world, agent, profile=profile)
    energy_ratio = world._energy_ratio(agent)
    tile = world.grid[agent.y][agent.x]
    plant_value = world._plant_food_value(agent, tile, profile)
    fresh_kill_action = best_visible_fresh_kill_action(world,
        agent,
        profile=profile,
        context=context,
    )
    carrion_action = best_visible_carrion_action(world,
        agent,
        profile=profile,
        context=context,
    )
    attack_action = best_adjacent_attack_action(world, agent, profile=profile)
    prey_move_action = best_visible_prey_action(world,
        agent,
        profile=profile,
        context=context,
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
) -> str | None:
    if not world._can_consume_fresh_kill(agent):
        return None
    if not world._fresh_kill_intake_useful(agent):
        return None
    profile = profile or world._trophic_profile(agent)
    context = context or build_decision_context(world, agent, profile=profile)
    radius = max(
        1,
        world.config.default_vision_radius
        + (2 if profile.role == "carnivore" or profile.hunter_drive >= 0.18 else 0),
    )
    best_target: tuple[int, int] | None = None
    best_score = float("-inf")
    for y in range(max(0, agent.y - radius), min(world.config.height, agent.y + radius + 1)):
        for x in range(max(0, agent.x - radius), min(world.config.width, agent.x + radius + 1)):
            distance = abs(x - agent.x) + abs(y - agent.y)
            if distance > radius:
                continue
            tile = world.grid[y][x]
            if tile.terrain == "water" or tile.fresh_kill_energy <= 0:
                continue
            if distance == 0:
                return "eat"
            if tile.occupant_id is not None:
                continue
            score = world._fresh_kill_food_value(agent, tile, profile)
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
    )


def best_visible_carrion_action(
    world: Any,
    agent: Agent,
    profile: TrophicProfile | None = None,
    context: DecisionContext | None = None,
) -> str | None:
    if not world._can_consume_carcass(agent):
        return None
    if not world._carcass_intake_useful(agent):
        return None
    profile = profile or world._trophic_profile(agent)
    context = context or build_decision_context(world, agent, profile=profile)
    radius = max(
        1,
        world.config.default_vision_radius
        + (
            2
            if profile.role == "carnivore" or profile.scavenger_drive >= 0.18
            else 0
        ),
    )
    best_target: tuple[int, int] | None = None
    best_score = float("-inf")
    for y in range(max(0, agent.y - radius), min(world.config.height, agent.y + radius + 1)):
        for x in range(max(0, agent.x - radius), min(world.config.width, agent.x + radius + 1)):
            distance = abs(x - agent.x) + abs(y - agent.y)
            if distance > radius:
                continue
            tile = world.grid[y][x]
            if tile.terrain == "water" or tile.carcass_energy <= 0:
                continue
            if distance == 0:
                return "eat" if world._carcass_intake_useful(agent) else None
            if tile.occupant_id is not None:
                continue
            score = world._carcass_food_value(agent, tile, profile)
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
    )


def best_adjacent_attack_action(
    world: Any,
    agent: Agent,
    profile: TrophicProfile | None = None,
) -> str | None:
    if not world._can_attack(agent):
        return None
    profile = profile or world._trophic_profile(agent)
    best_action: str | None = None
    best_score = float("-inf")
    for action, dx, dy in world._movement_actions():
        target = (
            world.agents.get(world.grid[agent.y + dy][agent.x + dx].occupant_id)
            if world._in_bounds(agent.x + dx, agent.y + dy)
            else None
        )
        if target is None or not target.alive or target.agent_id == agent.agent_id:
            continue
        score = world._attack_value(agent, target, profile)
        if score <= 0.005:
            continue
        target_role = world._trophic_role(target)
        if target_role == "herbivore":
            score += 0.06
        if world._health_ratio(target) > world._health_ratio(agent) + 0.16:
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
) -> str | None:
    if not world._can_attack(agent):
        return None
    profile = profile or world._trophic_profile(agent)
    context = context or build_decision_context(world, agent, profile=profile)
    radius = max(
        1,
        world.config.default_vision_radius
        + (2 if profile.role == "carnivore" or profile.hunter_drive >= 0.18 else 0),
    )
    best_target: tuple[int, int] | None = None
    best_score = float("-inf")
    for y in range(max(0, agent.y - radius), min(world.config.height, agent.y + radius + 1)):
        for x in range(max(0, agent.x - radius), min(world.config.width, agent.x + radius + 1)):
            distance = abs(x - agent.x) + abs(y - agent.y)
            if distance <= 1 or distance > radius:
                continue
            tile = world.grid[y][x]
            target_id = tile.occupant_id
            if tile.terrain == "water" or target_id is None or target_id == agent.agent_id:
                continue
            target = world.agents.get(target_id)
            if target is None or not target.alive:
                continue
            target_role = world._trophic_role(target)
            score = world._attack_value(agent, target, profile)
            score += (
                world._prey_vulnerability(target)
                * world.config.biotic_fields.hunter_vulnerability_weight
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
    )


def biotic_opportunity_score(
    world: Any,
    agent: Agent,
    x: int,
    y: int,
    profile: TrophicProfile,
    context: DecisionContext | None = None,
) -> float:
    biotic_state = (
        context.tile_memo.current_biotic_state() if context is not None else None
    )
    return world._biotic_field_score(agent, x, y, profile, biotic_state=biotic_state)


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
) -> str | None:
    profile = profile or world._trophic_profile(agent)
    context = context or build_decision_context(world, agent, profile=profile, season=season)
    water_urgency = context.water_urgency
    food_urgency = context.food_urgency
    current_distance = abs(target_x - agent.x) + abs(target_y - agent.y)
    best_action: str | None = None
    best_score = float("-inf")
    for action, dx, dy in world._movement_actions():
        x = agent.x + dx
        y = agent.y + dy
        if not world._can_move_to(x, y):
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
) -> float:
    profile = profile or world._trophic_profile(agent)
    tile_memo = context.tile_memo if context is not None else None
    tile = world.grid[y][x]
    score = 0.0
    water_reason = (
        tile_memo.water_reason(x, y)
        if tile_memo is not None
        else world._water_access_reason(x, y)
    )
    refuge_score = (
        tile_memo.refuge_score(x, y)
        if tile_memo is not None
        else world._refuge_score(x, y)
    )
    hazard_type, hazard_level = (
        tile_memo.hazard(x, y) if tile_memo is not None else world._hazard_at(x, y)
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
        if world._soft_refuge_reason(x, y) == "canopy_refuge":
            score += 0.06 * water_urgency
    if include_plant_channel and profile.plant_drive > 0:
        score += world._plant_food_value(agent, tile, profile, x=x, y=y) * (
            0.28 + food_urgency * 0.84
        )
    score += world._terrain_preference_score(agent, tile.terrain)
    score += world._field_preference_score(
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
    score -= hazard_level * (0.14 + (1.0 - world._health_ratio(agent)) * 0.34)
    if hazard_type == "exposure" and (
        tile_memo.soft_refuge_reason(x, y) if tile_memo is not None else world._soft_refuge_reason(x, y)
    ) == "canopy_refuge":
        score += 0.05
    if (
        include_fresh_kill_channel
        and tile.fresh_kill_energy > 0
        and world._can_consume_fresh_kill(agent)
    ):
        score += world._fresh_kill_food_value(agent, tile, profile) * (0.52 + food_urgency * 1.12)
    if include_carcass_channel and tile.carcass_energy > 0 and world._can_consume_carcass(agent):
        score += world._carcass_food_value(agent, tile, profile) * (0.4 + food_urgency * 1.1)
    if include_animal_signal and (profile.role == "herbivore" or profile.animal_drive > 0):
        score += biotic_opportunity_score(world, agent, x, y, profile, context=context) * (
            0.22 + food_urgency * 0.78
        )
    return score


def best_visible_action_toward_need(
    world: Any,
    agent: Agent,
    profile: TrophicProfile | None = None,
    context: DecisionContext | None = None,
) -> str | None:
    profile = profile or world._trophic_profile(agent)
    context = context or build_decision_context(world, agent, profile=profile)
    water_urgency = context.water_urgency
    food_urgency = context.food_urgency
    season = context.season
    radius = max(1, world.config.default_vision_radius)
    best_target: tuple[int, int] | None = None
    best_target_score = float("-inf")

    for y in range(max(0, agent.y - radius), min(world.config.height, agent.y + radius + 1)):
        for x in range(max(0, agent.x - radius), min(world.config.width, agent.x + radius + 1)):
            distance = abs(x - agent.x) + abs(y - agent.y)
            if distance == 0 or distance > radius:
                continue
            tile = world.grid[y][x]
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
    for action, dx, dy in world._movement_actions():
        x = agent.x + dx
        y = agent.y + dy
        if not world._can_move_to(x, y):
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
            )
        score -= next_distance * 0.04
        if score > best_score:
            best_score = score
            best_action = action

    return best_action
