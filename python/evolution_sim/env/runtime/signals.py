from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Iterable

SIGNAL_CONTRACT_VERSION = "foundation_signal_contract_v1"
REPRODUCTIVE_SIGNAL_FIELD = "reproductive_signal"
COMMUNICATION_SIGNAL_FIELD = "communication_signal"
SIGNAL_FIELD_NAMES = (REPRODUCTIVE_SIGNAL_FIELD, COMMUNICATION_SIGNAL_FIELD)
SIGNAL_EPSILON = 1e-9


@dataclass(frozen=True, slots=True)
class SignalProfile:
    profile_id: str
    intensity: float
    radius: int
    duration_ticks: int
    energy_cost: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SignalEmission:
    x: int
    y: int
    intensity: float
    remaining_ticks: int


@dataclass(frozen=True, slots=True)
class SignalFieldState:
    reproductive_signal: list[list[float]]
    communication_signal: list[list[float]]

    def to_serializable(self) -> dict[str, list[list[float]]]:
        return {
            REPRODUCTIVE_SIGNAL_FIELD: [
                [round(value, 4) for value in row]
                for row in self.reproductive_signal
            ],
            COMMUNICATION_SIGNAL_FIELD: [
                [round(value, 4) for value in row]
                for row in self.communication_signal
            ],
        }

    def field(self, name: str) -> list[list[float]]:
        if name == REPRODUCTIVE_SIGNAL_FIELD:
            return self.reproductive_signal
        if name == COMMUNICATION_SIGNAL_FIELD:
            return self.communication_signal
        raise KeyError(name)


def inert_signal_profile(profile_id: str) -> SignalProfile:
    return SignalProfile(
        profile_id=profile_id,
        intensity=0.0,
        radius=0,
        duration_ticks=0,
        energy_cost=0.0,
    )


def empty_signal_totals() -> dict[str, float]:
    return {
        "reproductive_emissions": 0.0,
        "communication_emissions": 0.0,
        "energy_spent": 0.0,
    }


def finalize_signal_totals(totals: dict[str, float]) -> dict[str, object]:
    return {
        "reproductive_emissions": int(totals.get("reproductive_emissions", 0.0)),
        "communication_emissions": int(totals.get("communication_emissions", 0.0)),
        "energy_spent": round(float(totals.get("energy_spent", 0.0)), 4),
    }


def signal_contract() -> dict[str, object]:
    return {
        "schema_version": SIGNAL_CONTRACT_VERSION,
        "substrate": "continuous_field_discrete_emission_profiles",
        "policy_semantics": "opaque",
        "fields": [REPRODUCTIVE_SIGNAL_FIELD, COMMUNICATION_SIGNAL_FIELD],
        "reproductive_readiness_profile": {
            "field": REPRODUCTIVE_SIGNAL_FIELD,
            "source": "biology_gated_reproduction_readiness",
            "policy_semantics": "opaque_numeric_field",
        },
        "reserved_profiles": [
            inert_signal_profile(COMMUNICATION_SIGNAL_FIELD).to_dict(),
        ],
        "enabled_in_scaffold": True,
        "reproductive_readiness_emission_enabled_by_default": True,
        "communication_emission_enabled_by_default": False,
    }


def invalidate_signal_state(world: Any) -> None:
    _record_runtime_cost(world, "signal_state_invalidations")
    world.signal_state_revision += 1
    world.cached_signal_state_revision = None
    world.cached_signal_state = None


def decay_signal_emissions(world: Any) -> None:
    config = world.config.signals
    changed = False
    if not config.enabled:
        if world.reproductive_signal_emissions:
            world.reproductive_signal_emissions = []
            changed = True
        if world.communication_signal_emissions:
            world.communication_signal_emissions = []
            changed = True
        if changed:
            invalidate_signal_state(world)
        return

    changed = (
        _decay_emission_list(
            world.reproductive_signal_emissions,
            decay_rate=config.reproductive_signal_decay_rate,
        )
        or changed
    )
    changed = (
        _decay_emission_list(
            world.communication_signal_emissions,
            decay_rate=config.reproductive_signal_decay_rate,
        )
        or changed
    )
    if changed:
        invalidate_signal_state(world)


def emit_reproductive_readiness_signals(
    world: Any,
    agents: Iterable[Any],
) -> dict[str, object]:
    config = world.config.signals
    if (
        not config.enabled
        or not config.reproductive_signal_emission_enabled
        or config.max_intensity <= 0
        or config.reproductive_signal_duration_ticks <= 0
    ):
        return finalize_signal_totals(empty_signal_totals())

    emitted = 0
    energy_spent = 0.0
    for agent in sorted(agents, key=lambda item: item.agent_id):
        if not agent.alive:
            continue
        if not world._in_bounds(agent.x, agent.y):
            continue
        if world.grid[agent.y][agent.x].terrain == "water":
            continue
        if not world._is_biologically_reproduction_ready(agent):
            continue
        intensity = _reproductive_signal_intensity(world, agent)
        if intensity <= SIGNAL_EPSILON:
            continue
        cost = _reproductive_signal_energy_cost(world, intensity)
        if cost > 0:
            agent.energy = max(0.0, agent.energy - cost)
            energy_spent += cost
        world.reproductive_signal_emissions.append(
            SignalEmission(
                x=agent.x,
                y=agent.y,
                intensity=intensity,
                remaining_ticks=config.reproductive_signal_duration_ticks,
            )
        )
        emitted += 1

    if emitted > 0:
        _record_runtime_cost(world, "signal_emissions", emitted)
        _accumulate_signal_totals(
            world.tick_signal_totals,
            reproductive_emissions=float(emitted),
            energy_spent=energy_spent,
        )
        _accumulate_signal_totals(
            world.run_signal_totals,
            reproductive_emissions=float(emitted),
            energy_spent=energy_spent,
        )
        invalidate_signal_state(world)

    return finalize_signal_totals(
        {
            "reproductive_emissions": float(emitted),
            "communication_emissions": 0.0,
            "energy_spent": energy_spent,
        }
    )


def build_signal_state(world: Any) -> SignalFieldState:
    config = world.config.signals
    if not config.enabled:
        return empty_signal_state(world.config.width, world.config.height)
    return SignalFieldState(
        reproductive_signal=_diffuse_signal_emissions(
            world,
            world.reproductive_signal_emissions,
            radius=config.reproductive_signal_radius,
            max_intensity=config.max_intensity,
        ),
        communication_signal=_diffuse_signal_emissions(
            world,
            world.communication_signal_emissions,
            radius=config.max_signal_radius,
            max_intensity=config.max_intensity,
        ),
    )


def empty_signal_state(width: int, height: int) -> SignalFieldState:
    return SignalFieldState(
        reproductive_signal=[[0.0 for _ in range(width)] for _ in range(height)],
        communication_signal=[[0.0 for _ in range(width)] for _ in range(height)],
    )


def signal_field_stats(
    state: SignalFieldState,
    grid: list[list[Any]],
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for name in SIGNAL_FIELD_NAMES:
        field = state.field(name)
        values = [
            field[y][x]
            for y, row in enumerate(grid)
            for x, tile in enumerate(row)
            if tile.terrain != "water"
        ]
        stats[name] = {
            "avg": round(sum(values) / max(len(values), 1), 4),
            "max": round(max(values, default=0.0), 4),
            "active_tiles": sum(1 for value in values if value > SIGNAL_EPSILON),
        }
    return stats


def _decay_emission_list(
    emissions: list[SignalEmission],
    *,
    decay_rate: float,
) -> bool:
    if not emissions:
        return False
    retained: list[SignalEmission] = []
    for emission in emissions:
        remaining_ticks = emission.remaining_ticks - 1
        intensity = emission.intensity * decay_rate
        if remaining_ticks <= 0 or intensity <= SIGNAL_EPSILON:
            continue
        emission.remaining_ticks = remaining_ticks
        emission.intensity = intensity
        retained.append(emission)
    emissions[:] = retained
    return True


def _reproductive_signal_intensity(world: Any, agent: Any) -> float:
    config = world.config.signals
    trait_bias = max(
        0.0,
        min(1.0, float(agent.genome.reproductive.signal_emission_bias)),
    )
    intensity = (
        config.reproductive_signal_base_intensity
        + config.reproductive_signal_trait_intensity_bonus * trait_bias
    )
    return max(0.0, min(config.max_intensity, intensity))


def _reproductive_signal_energy_cost(world: Any, intensity: float) -> float:
    max_intensity = max(float(world.config.signals.max_intensity), SIGNAL_EPSILON)
    scaled_cost = world.config.signals.base_emission_energy_cost * (
        intensity / max_intensity
    )
    return max(0.0, scaled_cost)


def _accumulate_signal_totals(
    totals: dict[str, float],
    *,
    reproductive_emissions: float = 0.0,
    communication_emissions: float = 0.0,
    energy_spent: float = 0.0,
) -> None:
    totals["reproductive_emissions"] = (
        totals.get("reproductive_emissions", 0.0) + reproductive_emissions
    )
    totals["communication_emissions"] = (
        totals.get("communication_emissions", 0.0) + communication_emissions
    )
    totals["energy_spent"] = totals.get("energy_spent", 0.0) + energy_spent


def _diffuse_signal_emissions(
    world: Any,
    emissions: list[SignalEmission],
    *,
    radius: int,
    max_intensity: float,
) -> list[list[float]]:
    width = world.config.width
    height = world.config.height
    if max_intensity <= 0:
        return [[0.0 for _ in range(width)] for _ in range(height)]

    sources: dict[int, float] = {}
    for emission in emissions:
        if emission.intensity <= SIGNAL_EPSILON or emission.remaining_ticks <= 0:
            continue
        if not world._in_bounds(emission.x, emission.y):
            continue
        if world.grid[emission.y][emission.x].terrain == "water":
            continue
        source_index = emission.y * width + emission.x
        sources[source_index] = min(
            max_intensity,
            sources.get(source_index, 0.0) + emission.intensity,
        )
    return _diffuse_sparse_signal_field(
        world,
        sources,
        radius=radius,
        max_intensity=max_intensity,
    )


def _diffuse_sparse_signal_field(
    world: Any,
    sources: dict[int, float],
    *,
    radius: int,
    max_intensity: float,
) -> list[list[float]]:
    _record_runtime_cost(world, "signal_diffusions")
    width = world.config.width
    height = world.config.height
    field = [0.0 for _ in range(width * height)]
    if not sources:
        return [
            field[row_start : row_start + width]
            for row_start in range(0, width * height, width)
        ]

    targets_by_source = _diffusion_targets_for_world(world, radius)
    for source_index, source in sorted(sources.items()):
        if source <= SIGNAL_EPSILON:
            continue
        for target_index, divisor in targets_by_source[source_index]:
            field[target_index] = min(
                max_intensity,
                field[target_index] + source / divisor,
            )
    return [
        field[row_start : row_start + width]
        for row_start in range(0, width * height, width)
    ]


@lru_cache(maxsize=None)
def _diffusion_offsets(radius: int) -> tuple[tuple[int, int, float], ...]:
    radius = max(0, radius)
    offsets: list[tuple[int, int, float]] = []
    for dy in range(-radius, radius + 1):
        span = radius - abs(dy)
        for dx in range(-span, span + 1):
            distance = abs(dx) + abs(dy)
            if distance > radius:
                continue
            offsets.append((dx, dy, distance + 1.0))
    return tuple(offsets)


def _diffusion_targets_for_world(
    world: Any,
    radius: int,
) -> tuple[tuple[tuple[int, float], ...], ...]:
    cache = world._signal_diffusion_target_cache
    cached_targets = cache.get(radius)
    if cached_targets is not None:
        _record_runtime_cost(world, "signal_diffusion_target_cache_hits")
        return cached_targets

    _record_runtime_cost(world, "signal_diffusion_target_cache_misses")
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


def _record_runtime_cost(world: Any, name: str, amount: int = 1) -> None:
    recorder = getattr(world, "_record_runtime_cost", None)
    if callable(recorder):
        recorder(name, amount)
