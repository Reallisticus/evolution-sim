from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evolution_sim.env.runtime.bootstrap import build_static_topology
from evolution_sim.env.runtime.state import BioticFieldState


def reset_derived_caches(world: Any, *, include_biotic: bool = True) -> None:
    world.terrain_map = [[tile.terrain for tile in row] for row in world.grid]
    world.static_topology = build_static_topology(world.terrain_map)
    world.cached_habitat_tick = None
    world.cached_habitat_grid = None
    world.cached_habitat_counts = None
    world.cached_climate_tick = None
    world.cached_climate_state = None
    world.cached_effective_fields_tick = None
    world.cached_effective_fields_grid = None
    world._biotic_diffusion_target_cache = {}
    world._signal_diffusion_target_cache = {}
    if include_biotic:
        world._invalidate_biotic_state()
        world._invalidate_signal_state()


@dataclass(slots=True)
class DerivedTileMemo:
    world: Any
    season: str
    climate_state: dict[str, object]
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
            self.effective_tile_fields[key] = self.world._effective_tile_fields(x, y)
        return self.effective_tile_fields[key]

    def water_reason(self, x: int, y: int) -> str:
        key = (x, y)
        if key not in self.water_reasons:
            self.water_reasons[key] = self.world._water_access_reason(x, y)
        return self.water_reasons[key]

    def soft_refuge_reason(self, x: int, y: int) -> str:
        key = (x, y)
        if key not in self.soft_refuge_reasons:
            self.soft_refuge_reasons[key] = self.world._soft_refuge_reason(x, y)
        return self.soft_refuge_reasons[key]

    def refuge_score(self, x: int, y: int) -> float:
        key = (x, y)
        if key not in self.refuge_scores:
            self.refuge_scores[key] = self.world._refuge_score(x, y)
        return self.refuge_scores[key]

    def hazard(self, x: int, y: int) -> tuple[str, float]:
        key = (x, y)
        if key not in self.hazards:
            self.hazards[key] = self.world._hazard_at(x, y)
        return self.hazards[key]

    def current_biotic_state(self) -> BioticFieldState:
        if self.biotic_state is None:
            self.biotic_state = self.world._current_biotic_state()
        return self.biotic_state
