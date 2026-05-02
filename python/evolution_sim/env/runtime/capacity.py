from __future__ import annotations

from typing import Any


NEAR_CAPACITY_SATURATION_THRESHOLD = 0.9


def record_carrying_capacity_tick(
    world: Any,
    *,
    post_reproduction_alive: int,
    final_alive: int,
    births: int,
    deaths: int,
) -> None:
    max_agents = max(int(world.config.max_agents), 1)
    tick_peak_alive = max(int(post_reproduction_alive), int(final_alive))
    tick_peak_saturation = tick_peak_alive / max_agents

    if tick_peak_saturation >= NEAR_CAPACITY_SATURATION_THRESHOLD:
        world.carrying_capacity_near_cap_ticks += 1
    if tick_peak_alive >= max_agents:
        world.carrying_capacity_at_cap_ticks += 1
    if births > 0 and int(post_reproduction_alive) >= max_agents:
        world.carrying_capacity_saturation_births += int(births)
    if deaths > 0 and int(post_reproduction_alive) >= max_agents:
        world.carrying_capacity_saturation_deaths += int(deaths)


def build_carrying_capacity_summary(
    world: Any,
    *,
    ticks_executed: int,
) -> dict[str, int | float]:
    tick_count = max(int(ticks_executed), 1)
    near_cap_ticks = int(world.carrying_capacity_near_cap_ticks)
    at_cap_ticks = int(world.carrying_capacity_at_cap_ticks)
    return {
        "near_cap_saturation_threshold": NEAR_CAPACITY_SATURATION_THRESHOLD,
        "near_cap_ticks": near_cap_ticks,
        "at_cap_ticks": at_cap_ticks,
        "near_cap_tick_share": round(near_cap_ticks / tick_count, 4),
        "at_cap_tick_share": round(at_cap_ticks / tick_count, 4),
        "saturation_births": int(world.carrying_capacity_saturation_births),
        "saturation_deaths": int(world.carrying_capacity_saturation_deaths),
    }
