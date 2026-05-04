from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from evolution_sim.env.runtime.state import BioticFieldState

DiffusionTargetCache = MutableMapping[
    int,
    tuple[tuple[tuple[int, float], ...], ...],
]


@dataclass(frozen=True, slots=True)
class BioticDiffusionContext:
    width: int
    height: int
    grid: Sequence[Sequence[Any]]
    diffusion_radius: int
    target_cache: DiffusionTargetCache
    record_runtime_cost: Callable[[str, int], None]


@dataclass(frozen=True, slots=True)
class BioticStateContext:
    width: int
    grid: Sequence[Sequence[Any]]
    alive_agents: Sequence[Any]
    trophic_profile: Callable[[Any], Any]
    agent_biomass: Callable[[Any], float]
    prey_vulnerability: Callable[[Any], float]
    health_ratio: Callable[[Any], float]
    diffusion: BioticDiffusionContext


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


def _record_runtime_cost(
    context: BioticDiffusionContext,
    name: str,
    amount: int = 1,
) -> None:
    context.record_runtime_cost(name, amount)


def _diffusion_targets(
    context: BioticDiffusionContext,
    radius: int,
) -> tuple[tuple[tuple[int, float], ...], ...]:
    cache = context.target_cache
    cached_targets = cache.get(radius)
    if cached_targets is not None:
        _record_runtime_cost(context, "biotic_diffusion_target_cache_hits")
        return cached_targets

    _record_runtime_cost(context, "biotic_diffusion_target_cache_misses")
    offsets = _diffusion_offsets(radius)
    width = context.width
    targets_by_source: list[tuple[tuple[int, float], ...]] = []
    for sy, row in enumerate(context.grid):
        for sx, tile in enumerate(row):
            targets: list[tuple[int, float]] = []
            if tile.terrain != "water":
                for dx, dy, divisor in offsets:
                    x = sx + dx
                    y = sy + dy
                    if (
                        x < 0
                        or y < 0
                        or x >= context.width
                        or y >= context.height
                        or context.grid[y][x].terrain == "water"
                    ):
                        continue
                    targets.append((y * width + x, divisor))
            targets_by_source.append(tuple(targets))
    cached_targets = tuple(targets_by_source)
    cache[radius] = cached_targets
    return cached_targets


def diffuse_biotic_field(
    sources: list[list[float]],
    *,
    context: BioticDiffusionContext,
) -> list[list[float]]:
    _record_runtime_cost(context, "biotic_diffusions")
    radius = max(1, context.diffusion_radius)
    targets_by_source = _diffusion_targets(context, radius)
    width = context.width
    height = context.height
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


def diffuse_sparse_biotic_field(
    sources: dict[int, float],
    *,
    context: BioticDiffusionContext,
) -> list[list[float]]:
    _record_runtime_cost(context, "biotic_diffusions")
    radius = max(1, context.diffusion_radius)
    targets_by_source = _diffusion_targets(context, radius)
    width = context.width
    height = context.height
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


def build_biotic_state(context: BioticStateContext) -> BioticFieldState:
    width = context.width
    prey_sources: dict[int, float] = {}
    carrion_sources: dict[int, float] = {}
    predator_sources: dict[int, float] = {}

    for agent in context.alive_agents:
        tile = context.grid[agent.y][agent.x]
        if tile.terrain == "water":
            continue
        source_index = agent.y * width + agent.x
        profile = context.trophic_profile(agent)
        biomass = context.agent_biomass(agent)
        vulnerability = context.prey_vulnerability(agent)
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
                + profile.hunter_drive
                * agent.genome.attack_power
                * context.health_ratio(agent)
            )

    for y, row in enumerate(context.grid):
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
        prey_biomass=diffuse_sparse_biotic_field(
            prey_sources,
            context=context.diffusion,
        ),
        carrion=diffuse_sparse_biotic_field(
            carrion_sources,
            context=context.diffusion,
        ),
        predator_risk=diffuse_sparse_biotic_field(
            predator_sources,
            context=context.diffusion,
        ),
    )
