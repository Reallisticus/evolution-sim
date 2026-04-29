from __future__ import annotations

from typing import Any


TERRAIN_VEGETATION_BASE = {
    "plain": 0.5,
    "forest": 0.72,
    "wetland": 0.68,
    "rocky": 0.3,
}
TERRAIN_RESILIENCE_BASE = {
    "plain": 0.54,
    "forest": 0.7,
    "wetland": 0.78,
    "rocky": 0.6,
}
TERRAIN_SHELTER_BASE = {
    "plain": 0.08,
    "forest": 0.52,
    "wetland": 0.14,
    "rocky": 0.06,
}


def vegetation_target(world: Any, x: int, y: int, season: str) -> float:
    tile = world.grid[y][x]
    if tile.terrain == "water":
        return 1.0
    fertility, moisture, heat = world._effective_tile_fields(x, y)
    habitat_state = world._habitat_state_at(x, y)
    target = (
        TERRAIN_VEGETATION_BASE.get(tile.terrain, 0.5)
        + fertility * 0.18
        + moisture * 0.24
        - heat * 0.16
    )
    if habitat_state == "bloom":
        target += 0.08
    elif habitat_state == "flooded":
        target += 0.06 if tile.terrain == "wetland" else -0.04
    elif habitat_state == "parched":
        target -= 0.08 if tile.terrain == "rocky" else 0.14
    return world._clamp01(target)


def shelter_target(world: Any, x: int, y: int, season: str) -> float:
    tile = world.grid[y][x]
    if tile.terrain == "water":
        return 0.0
    fertility, moisture, heat = world._effective_tile_fields(x, y)
    forest_density = world._terrain_neighbor_ratio(x, y, terrain_filter={"forest"}, radius=1)
    habitat_state = world._habitat_state_at(x, y)
    target = (
        TERRAIN_SHELTER_BASE.get(tile.terrain, 0.08)
        + tile.vegetation * 0.28
        + forest_density * (0.34 if tile.terrain == "forest" else 0.08)
        + fertility * 0.06
        + moisture * 0.08
        - heat * 0.1
        - tile.recovery_debt * 0.18
    )
    if habitat_state == "bloom":
        target += 0.04
    elif habitat_state == "flooded":
        target -= 0.12
    elif habitat_state == "parched":
        target -= 0.22
    if tile.terrain != "forest":
        target *= 0.22
    return world._clamp01(target)


def food_capacity(world: Any, x: int, y: int, season: str) -> float:
    tile = world.grid[y][x]
    if tile.terrain == "water":
        return 0.0
    fertility, moisture, heat = world._effective_tile_fields(x, y)
    capacity = (
        0.06
        + tile.vegetation * 0.7
        + fertility * 0.18
        + moisture * 0.08
        - heat * 0.06
        - tile.recovery_debt * 0.18
    )
    if tile.terrain == "forest":
        capacity += 0.08
    elif tile.terrain == "wetland":
        capacity += 0.04
    elif tile.terrain == "rocky":
        capacity -= 0.04
    return world._clamp01(capacity)


def field_growth_multiplier(world: Any, x: int, y: int, season: str) -> float:
    tile = world.grid[y][x]
    fertility, moisture, heat = world._effective_tile_fields(x, y)
    growth = 0.42 + fertility * 0.72 + moisture * 0.44
    heat_penalty = max(0.0, heat - moisture) * 0.34
    vegetation_bonus = tile.vegetation * 0.34
    recovery_penalty = tile.recovery_debt * 0.42
    return max(0.22, growth + vegetation_bonus - heat_penalty - recovery_penalty)


def regrow_resources(world: Any) -> None:
    season = world._season_state()["name"]
    world._habitat_state_grid()
    resources = world.config.resources
    for y, row in enumerate(world.grid):
        for x, tile in enumerate(row):
            if tile.terrain == "water":
                tile.water = world.config.resources.water_refresh_amount
                continue

            if tile.fresh_kill_energy > 0:
                world._convert_fresh_kill_to_carcass(
                    tile,
                    x=x,
                    y=y,
                    conversion_rate=world.config.carcasses.fresh_kill_conversion_rate,
                )

            if tile.carcass_energy > 0:
                _, moisture, heat = world._effective_tile_fields(x, y)
                decay = (
                    world.config.carcasses.decay_base_rate
                    + heat * world.config.carcasses.decay_heat_factor
                    + moisture * world.config.carcasses.decay_moisture_factor
                )
                energy_decayed = world._decay_carcass_tile(tile, decay=decay)
                world.tick_carcass_energy_decayed += energy_decayed
                world.run_carcass_totals["energy_decayed"] += energy_decayed

            fertility, moisture, heat = world._effective_tile_fields(x, y)
            habitat_state = world._habitat_state_at(x, y)
            field_growth = field_growth_multiplier(world, x, y, season)
            vegetation_goal = vegetation_target(world, x, y, season)
            shelter_goal = shelter_target(world, x, y, season)
            forest_density = world._terrain_neighbor_ratio(x, y, terrain_filter={"forest"}, radius=1)
            recovery_support = max(0.28, 1.0 - tile.recovery_debt * 0.72)
            vegetation_growth = (
                resources.vegetation_regrowth_rate
                * terrain_growth_modifier(world, tile.terrain, season)
                * field_growth
                * recovery_support
            )
            vegetation_stress = resources.terrain_degradation_rate * (
                max(0.0, heat - moisture) * 0.64
                + max(0.0, 0.45 - tile.food) * 0.18
            )
            if habitat_state == "bloom":
                vegetation_growth *= 1.24
            elif habitat_state == "flooded":
                if tile.terrain == "wetland":
                    vegetation_growth *= 1.08
                else:
                    vegetation_stress += resources.terrain_degradation_rate * 0.16
            elif habitat_state == "parched":
                vegetation_stress += (
                    resources.terrain_degradation_rate
                    * (0.62 if tile.terrain == "rocky" else 0.92)
                )

            if tile.vegetation <= vegetation_goal:
                tile.vegetation = world._clamp01(
                    tile.vegetation
                    + (vegetation_goal - tile.vegetation) * vegetation_growth
                    - vegetation_stress * 0.18
                )
            else:
                tile.vegetation = world._clamp01(
                    tile.vegetation - (tile.vegetation - vegetation_goal) * (0.18 + vegetation_stress)
                )

            degradation = resources.terrain_degradation_rate * (
                max(0.0, 0.42 - tile.vegetation) * 0.94
                + max(0.0, heat - moisture) * 0.58
            )
            recovery = resources.terrain_recovery_rate * (
                0.44
                + tile.vegetation * 0.84
                + fertility * 0.3
                + moisture * 0.24
                + TERRAIN_RESILIENCE_BASE.get(tile.terrain, 0.56) * 0.32
            )
            if habitat_state == "bloom":
                recovery *= 1.18
            elif habitat_state == "flooded" and tile.terrain != "wetland":
                degradation *= 1.16
            elif habitat_state == "parched":
                degradation *= 1.34 if tile.terrain != "rocky" else 1.16
                recovery *= 0.72
            tile.recovery_debt = world._clamp01(tile.recovery_debt + degradation - recovery)

            shelter_growth = (
                resources.shelter_regrowth_rate
                * (0.44 + tile.vegetation * 0.42)
                * (0.36 + forest_density * 0.64)
                * max(0.28, 1.0 - tile.recovery_debt * 0.6)
            )
            shelter_stress = resources.shelter_degradation_rate * (
                max(0.0, heat - moisture) * 0.56
                + tile.recovery_debt * 0.34
                + max(0.0, 0.42 - tile.vegetation) * 0.38
            )
            if habitat_state == "bloom":
                shelter_growth *= 1.1
            elif habitat_state == "flooded":
                shelter_stress += resources.shelter_degradation_rate * 0.3
            elif habitat_state == "parched":
                shelter_stress += resources.shelter_degradation_rate * 0.42
                shelter_growth *= 0.72

            if tile.shelter <= shelter_goal:
                tile.shelter = world._clamp01(
                    tile.shelter
                    + (shelter_goal - tile.shelter) * shelter_growth
                    - shelter_stress * 0.1
                )
            else:
                tile.shelter = world._clamp01(
                    tile.shelter - (tile.shelter - shelter_goal) * (0.14 + shelter_stress)
                )

            if habitat_state == "parched":
                tile.food = max(
                    0.0,
                    tile.food - (0.008 if tile.terrain == "plain" else 0.0045),
                )
            elif habitat_state == "flooded" and tile.terrain == "plain":
                tile.food = max(0.0, tile.food - 0.003)

            food_regrowth = (
                terrain_regrowth_rate(world, tile.terrain)
                * terrain_growth_modifier(world, tile.terrain, season)
                * field_growth
                * (0.4 + tile.vegetation * 0.84)
                * max(0.24, 1.0 - tile.recovery_debt * 0.72)
                * habitat_regrowth_modifier(world, x, y)
            )
            tile.food = min(1.0, tile.food + food_regrowth)
            tile.food = min(
                tile.food,
                max(0.04, food_capacity(world, x, y, season)),
            )


def habitat_regrowth_modifier(world: Any, x: int, y: int) -> float:
    habitat_state = world._habitat_state_at(x, y)
    terrain = world.grid[y][x].terrain
    if habitat_state == "bloom":
        return 1.028
    if habitat_state == "flooded":
        return 1.012 if terrain == "wetland" else 0.986
    if habitat_state == "parched":
        return 0.972 if terrain == "rocky" else 0.94
    return 1.0


def terrain_regrowth_rate(world: Any, terrain: str) -> float:
    resources = world.config.resources
    if terrain == "forest":
        return resources.forest_food_rate
    if terrain == "wetland":
        return resources.wetland_food_rate
    if terrain == "rocky":
        return resources.rocky_food_rate
    return resources.plain_food_rate


def terrain_growth_modifier(world: Any, terrain: str, season: str) -> float:
    climate = world.config.climate
    if terrain == "forest":
        return (
            1.0 + climate.wet_forest_bonus
            if season == "wet"
            else 1.0 - climate.dry_forest_penalty
        )
    if terrain == "wetland":
        return (
            1.0 + climate.wet_forest_bonus * 0.85
            if season == "wet"
            else 1.0 - climate.dry_forest_penalty * 0.4
        )
    if terrain == "rocky":
        return (
            1.0 + climate.wet_plain_bonus * 0.35
            if season == "wet"
            else 1.0 - climate.dry_plain_penalty * 0.52
        )
    return (
        1.0 + climate.wet_plain_bonus
        if season == "wet"
        else 1.0 - climate.dry_plain_penalty
    )
