from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass, field
from typing import Any

from evolution_sim.env.runtime.bootstrap import build_static_topology
from evolution_sim.env.runtime.state import BioticFieldState


DiffusionTargetCache = MutableMapping[
    int,
    tuple[tuple[tuple[int, float], ...], ...],
]


def reset_derived_caches(
    world: Any,
    *,
    include_biotic: bool = True,
    biotic_diffusion_target_cache: DiffusionTargetCache,
    signal_diffusion_target_cache: DiffusionTargetCache,
    invalidate_biotic_state: Callable[[], None],
    invalidate_signal_state: Callable[[], None],
) -> None:
    world.terrain_map = [[tile.terrain for tile in row] for row in world.grid]
    world.static_topology = build_static_topology(world.terrain_map)
    world.cached_habitat_tick = None
    world.cached_habitat_grid = None
    world.cached_habitat_counts = None
    world.cached_climate_tick = None
    world.cached_climate_state = None
    world.cached_effective_fields_tick = None
    world.cached_effective_fields_grid = None
    biotic_diffusion_target_cache.clear()
    signal_diffusion_target_cache.clear()
    if include_biotic:
        invalidate_biotic_state()
        invalidate_signal_state()


@dataclass(slots=True)
class DerivedTileMemo:
    season: str
    climate_state: dict[str, object]
    effective_fields_for: Callable[[int, int], tuple[float, float, float]]
    water_reason_for: Callable[[int, int], str]
    soft_refuge_reason_for: Callable[[int, int], str]
    refuge_score_for: Callable[[int, int], float]
    hazard_for: Callable[[int, int], tuple[str, float]]
    current_biotic_state_for: Callable[[], BioticFieldState]
    effective_tile_fields: dict[tuple[int, int], tuple[float, float, float]] = field(
        default_factory=dict
    )
    water_reasons: dict[tuple[int, int], str] = field(default_factory=dict)
    soft_refuge_reasons: dict[tuple[int, int], str] = field(default_factory=dict)
    refuge_scores: dict[tuple[int, int], float] = field(default_factory=dict)
    hazards: dict[tuple[int, int], tuple[str, float]] = field(default_factory=dict)
    biotic_state: BioticFieldState | None = None

    def effective_fields(self, x: int, y: int) -> tuple[float, float, float]:
        key = (x, y)
        if key not in self.effective_tile_fields:
            self.effective_tile_fields[key] = self.effective_fields_for(x, y)
        return self.effective_tile_fields[key]

    def water_reason(self, x: int, y: int) -> str:
        key = (x, y)
        if key not in self.water_reasons:
            self.water_reasons[key] = self.water_reason_for(x, y)
        return self.water_reasons[key]

    def soft_refuge_reason(self, x: int, y: int) -> str:
        key = (x, y)
        if key not in self.soft_refuge_reasons:
            self.soft_refuge_reasons[key] = self.soft_refuge_reason_for(x, y)
        return self.soft_refuge_reasons[key]

    def refuge_score(self, x: int, y: int) -> float:
        key = (x, y)
        if key not in self.refuge_scores:
            self.refuge_scores[key] = self.refuge_score_for(x, y)
        return self.refuge_scores[key]

    def hazard(self, x: int, y: int) -> tuple[str, float]:
        key = (x, y)
        if key not in self.hazards:
            self.hazards[key] = self.hazard_for(x, y)
        return self.hazards[key]

    def current_biotic_state(self) -> BioticFieldState:
        if self.biotic_state is None:
            self.biotic_state = self.current_biotic_state_for()
        return self.biotic_state
