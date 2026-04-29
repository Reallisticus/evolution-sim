from __future__ import annotations

from functools import lru_cache
from typing import Any

from evolution_sim.env.runtime.state import BioticFieldState


@lru_cache(maxsize=None)
def _diffusion_offsets(radius: int) -> tuple[tuple[int, int, float], ...]:
    offsets: list[tuple[int, int, float]] = []
    for dy in range(-radius, radius + 1):
        span = radius - abs(dy)
        for dx in range(-span, span + 1):
            distance = abs(dx) + abs(dy)
            if distance > radius:
                continue
            offsets.append((dx, dy, distance + 1.0))
    return tuple(offsets)


def invalidate_biotic_state(world: Any) -> None:
    world.biotic_state_revision += 1
    world.cached_biotic_state_revision = None
    world.cached_biotic_state = None


def _record_runtime_cost(world: Any, name: str) -> None:
    recorder = getattr(world, "_record_runtime_cost", None)
    if callable(recorder):
        recorder(name)


def _diffusion_targets_for_world(
    world: Any,
    radius: int,
) -> tuple[tuple[tuple[int, float], ...], ...]:
    cache = world._biotic_diffusion_target_cache
    cached_targets = cache.get(radius)
    if cached_targets is not None:
        _record_runtime_cost(world, "biotic_diffusion_target_cache_hits")
        return cached_targets

    _record_runtime_cost(world, "biotic_diffusion_target_cache_misses")
    offsets = _diffusion_offsets(radius)
    width = world.config.width
    targets_by_source: list[tuple[tuple[int, float], ...]] = []
    for sy, row in enumerate(world.grid):
        for sx, tile in enumerate(row):
            targets: list[tuple[int, float]] = []
            if tile.terrain != "water":
                for dx, dy, divisor in offsets:
                    x = sx + dx
                    y = sy + dy
                    if not world._in_bounds(x, y) or world.grid[y][x].terrain == "water":
                        continue
                    targets.append((y * width + x, divisor))
            targets_by_source.append(tuple(targets))
    cached_targets = tuple(targets_by_source)
    cache[radius] = cached_targets
    return cached_targets


def diffuse_biotic_field(world: Any, sources: list[list[float]]) -> list[list[float]]:
    _record_runtime_cost(world, "biotic_diffusions")
    radius = max(1, world.config.biotic_fields.diffusion_radius)
    targets_by_source = _diffusion_targets_for_world(world, radius)
    width = world.config.width
    height = world.config.height
    field = [0.0 for _ in range(width * height)]
    source_index = 0
    for row in sources:
        for source in row:
            if source <= 1e-9:
                source_index += 1
                continue
            for target_index, divisor in targets_by_source[source_index]:
                field[target_index] += source / divisor
            source_index += 1
    return [
        field[row_start : row_start + width]
        for row_start in range(0, width * height, width)
    ]


def diffuse_sparse_biotic_field(world: Any, sources: dict[int, float]) -> list[list[float]]:
    _record_runtime_cost(world, "biotic_diffusions")
    radius = max(1, world.config.biotic_fields.diffusion_radius)
    targets_by_source = _diffusion_targets_for_world(world, radius)
    width = world.config.width
    height = world.config.height
    field = [0.0 for _ in range(width * height)]
    for source_index, source in sorted(sources.items()):
        if source <= 1e-9:
            continue
        for target_index, divisor in targets_by_source[source_index]:
            field[target_index] += source / divisor
    return [
        field[row_start : row_start + width]
        for row_start in range(0, width * height, width)
    ]


def build_biotic_state(world: Any) -> BioticFieldState:
    width = world.config.width
    prey_sources: dict[int, float] = {}
    carrion_sources: dict[int, float] = {}
    predator_sources: dict[int, float] = {}

    for agent in world.alive_agents():
        tile = world.grid[agent.y][agent.x]
        if tile.terrain == "water":
            continue
        source_index = agent.y * width + agent.x
        profile = world._trophic_profile(agent)
        biomass = world._agent_biomass(agent)
        vulnerability = world._prey_vulnerability(agent)
        if profile.role == "herbivore":
            prey_sources[source_index] = (
                prey_sources.get(source_index, 0.0) + biomass * vulnerability
            )
        elif profile.role == "omnivore":
            weakened = max(0.0, vulnerability - 1.12)
            if weakened > 0:
                prey_sources[source_index] = (
                    prey_sources.get(source_index, 0.0) + biomass * weakened * 0.56
                )
        if profile.role == "carnivore" or profile.hunter_drive >= 0.12:
            predator_sources[source_index] = (
                predator_sources.get(source_index, 0.0)
                + profile.hunter_drive * agent.genome.attack_power * world._health_ratio(agent)
            )

    for y, row in enumerate(world.grid):
        for x, tile in enumerate(row):
            if not tile.fresh_kill_deposits and not tile.carcass_deposits:
                continue
            if tile.terrain == "water":
                continue
            fresh_kill_energy = sum(
                deposit.energy_remaining for deposit in tile.fresh_kill_deposits
            )
            carcass_energy = sum(deposit.energy_remaining for deposit in tile.carcass_deposits)
            if fresh_kill_energy <= 1e-9 and carcass_energy <= 1e-9:
                continue
            carrion_source = fresh_kill_energy
            if carcass_energy > 0:
                carcass_decay = (
                    sum(
                        deposit.energy_remaining * deposit.freshness
                        for deposit in tile.carcass_deposits
                    )
                    / carcass_energy
                )
                carrion_source += carcass_energy * carcass_decay
            carrion_sources[y * width + x] = carrion_source

    return BioticFieldState(
        prey_biomass=diffuse_sparse_biotic_field(world, prey_sources),
        carrion=diffuse_sparse_biotic_field(world, carrion_sources),
        predator_risk=diffuse_sparse_biotic_field(world, predator_sources),
    )
