from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Iterable

from evolution_sim.config.schema import SignalConfig

SIGNAL_CONTRACT_VERSION = "foundation_signal_contract_v2"
REPRODUCTIVE_SIGNAL_FIELD = "reproductive_signal"
COMMUNICATION_SIGNAL_FIELD = "communication_signal"
SIGNAL_FIELD_NAMES = (REPRODUCTIVE_SIGNAL_FIELD, COMMUNICATION_SIGNAL_FIELD)
SIGNAL_EPSILON = 1e-9
REPRODUCTIVE_READINESS_PROFILE_ID = "reproductive_readiness"
COMMUNICATION_PROFILE_PREFIX = "communication_token"
_DEFAULT_SIGNAL_CONFIG = SignalConfig()
DEFAULT_COMMUNICATION_TOKEN_COUNT = int(
    _DEFAULT_SIGNAL_CONFIG.communication_token_count
)
DEFAULT_COMMUNICATION_PROFILES_PER_TOKEN = int(
    _DEFAULT_SIGNAL_CONFIG.communication_profiles_per_token
)


@dataclass(frozen=True, slots=True)
class SignalProfile:
    profile_id: str
    field_name: str
    source_kind: str
    token_id: int | None
    profile_index: int | None
    intensity: float
    radius: int
    duration_ticks: int
    decay_rate: float
    energy_cost: float
    policy_visible: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class SignalEmission:
    field_name: str
    profile_id: str
    source_kind: str
    source_agent_id: int | None
    token_id: int | None
    profile_index: int | None
    x: int
    y: int
    intensity: float
    radius: int
    duration_ticks: int
    remaining_ticks: int
    decay_rate: float
    energy_cost: float
    emitted_tick: int

    def to_debug_dict(self) -> dict[str, object]:
        return {
            "field_name": self.field_name,
            "profile_id": self.profile_id,
            "source_kind": self.source_kind,
            "source_agent_id": self.source_agent_id,
            "token_id": self.token_id,
            "profile_index": self.profile_index,
            "x": self.x,
            "y": self.y,
            "intensity": round(self.intensity, 4),
            "radius": self.radius,
            "duration_ticks": self.duration_ticks,
            "remaining_ticks": self.remaining_ticks,
            "decay_rate": round(self.decay_rate, 4),
            "energy_cost": round(self.energy_cost, 4),
            "emitted_tick": self.emitted_tick,
            "policy_visible": False,
        }


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


def inert_signal_profile(
    profile_id: str,
    *,
    field_name: str,
    source_kind: str,
    token_id: int | None = None,
    profile_index: int | None = None,
) -> SignalProfile:
    return SignalProfile(
        profile_id=profile_id,
        field_name=field_name,
        source_kind=source_kind,
        token_id=token_id,
        profile_index=profile_index,
        intensity=0.0,
        radius=0,
        duration_ticks=0,
        decay_rate=0.0,
        energy_cost=0.0,
        policy_visible=False,
    )


def reproductive_readiness_signal_profile(config: Any | None = None) -> SignalProfile:
    if config is None:
        config = SignalConfig()
    intensity = min(
        float(config.max_intensity),
        float(config.reproductive_signal_base_intensity)
        + float(config.reproductive_signal_trait_intensity_bonus),
    )
    radius = int(config.reproductive_signal_radius)
    duration_ticks = int(config.reproductive_signal_duration_ticks)
    decay_rate = float(config.reproductive_signal_decay_rate)
    energy_cost = float(config.base_emission_energy_cost)
    return SignalProfile(
        profile_id=REPRODUCTIVE_READINESS_PROFILE_ID,
        field_name=REPRODUCTIVE_SIGNAL_FIELD,
        source_kind="biology_gated_reproduction_readiness",
        token_id=None,
        profile_index=0,
        intensity=max(0.0, intensity),
        radius=max(0, radius),
        duration_ticks=max(0, duration_ticks),
        decay_rate=max(0.0, decay_rate),
        energy_cost=max(0.0, energy_cost),
        policy_visible=False,
    )


def communication_signal_profile(
    config: Any,
    *,
    token_id: int,
    profile_index: int,
) -> SignalProfile:
    profiles_per_token = max(1, int(config.communication_profiles_per_token))
    profile_scale = (profile_index + 1) / profiles_per_token
    intensity_ceiling = min(
        float(config.max_intensity),
        float(config.communication_signal_base_intensity)
        + float(config.communication_signal_trait_intensity_bonus),
    )
    radius = min(
        int(config.max_signal_radius),
        int(config.communication_signal_radius) + profile_index,
    )
    duration_ticks = min(
        int(config.max_duration_ticks),
        int(config.communication_signal_duration_ticks) + profile_index,
    )
    return SignalProfile(
        profile_id=f"{COMMUNICATION_PROFILE_PREFIX}_{token_id}_profile_{profile_index}",
        field_name=COMMUNICATION_SIGNAL_FIELD,
        source_kind="reserved_opaque_communication",
        token_id=token_id,
        profile_index=profile_index,
        intensity=max(0.0, intensity_ceiling * profile_scale),
        radius=max(0, radius),
        duration_ticks=max(0, duration_ticks),
        decay_rate=max(0.0, float(config.communication_signal_decay_rate)),
        energy_cost=max(0.0, float(config.base_emission_energy_cost) * profile_scale),
        policy_visible=False,
    )


def reserved_communication_signal_profiles(
    *,
    config: Any | None = None,
    token_count: int | None = None,
    profiles_per_token: int | None = None,
) -> list[SignalProfile]:
    if config is not None:
        token_count = int(config.communication_token_count)
        profiles_per_token = int(config.communication_profiles_per_token)
    token_count = DEFAULT_COMMUNICATION_TOKEN_COUNT if token_count is None else token_count
    profiles_per_token = (
        DEFAULT_COMMUNICATION_PROFILES_PER_TOKEN
        if profiles_per_token is None
        else profiles_per_token
    )
    profiles: list[SignalProfile] = []
    for token_id in range(token_count):
        for profile_index in range(profiles_per_token):
            if config is not None and communication_signal_emission_enabled(config):
                profiles.append(
                    communication_signal_profile(
                        config,
                        token_id=token_id,
                        profile_index=profile_index,
                    )
                )
                continue
            profiles.append(
                inert_signal_profile(
                    (
                        f"{COMMUNICATION_PROFILE_PREFIX}_{token_id}"
                        f"_profile_{profile_index}"
                    ),
                    field_name=COMMUNICATION_SIGNAL_FIELD,
                    source_kind="reserved_opaque_communication",
                    token_id=token_id,
                    profile_index=profile_index,
                )
            )
    return profiles


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


def signal_contract(config: Any | None = None) -> dict[str, object]:
    if config is None:
        config = SignalConfig()
    return {
        "schema_version": SIGNAL_CONTRACT_VERSION,
        "substrate": "continuous_field_discrete_emission_profiles",
        "policy_semantics": "opaque",
        "fields": [REPRODUCTIVE_SIGNAL_FIELD, COMMUNICATION_SIGNAL_FIELD],
        "profile_metadata_policy_visible": False,
        "emission_debug_metadata_fields": [
            "field_name",
            "profile_id",
            "source_kind",
            "source_agent_id",
            "token_id",
            "profile_index",
            "x",
            "y",
            "intensity",
            "radius",
            "duration_ticks",
            "remaining_ticks",
            "decay_rate",
            "energy_cost",
            "emitted_tick",
        ],
        "reproductive_readiness_profile": _contract_reproductive_profile(
            config
        ).to_dict(),
        "reserved_profiles": [
            profile.to_dict()
            for profile in reserved_communication_signal_profiles(
                config=config,
            )
        ],
        "communication_token_count": int(config.communication_token_count),
        "communication_profiles_per_token": int(config.communication_profiles_per_token),
        "signal_substrate_enabled": signal_substrate_enabled(config),
        "reproductive_signal_emission_enabled": (
            reproductive_signal_emission_enabled(config)
        ),
        "communication_signal_emission_enabled": (
            communication_signal_emission_enabled(config)
        ),
        "communication_signal_radius": int(config.communication_signal_radius),
        "communication_signal_duration_ticks": int(
            config.communication_signal_duration_ticks
        ),
        "communication_signal_decay_rate": float(config.communication_signal_decay_rate),
        "communication_signal_base_intensity": float(
            config.communication_signal_base_intensity
        ),
        "communication_signal_trait_intensity_bonus": float(
            config.communication_signal_trait_intensity_bonus
        ),
        "max_signal_radius": int(config.max_signal_radius),
        "max_duration_ticks": int(config.max_duration_ticks),
        "max_intensity": float(config.max_intensity),
        "communication_tokens_have_simulator_assigned_meaning": False,
        "enabled_in_scaffold": True,
        "reproductive_readiness_emission_enabled_by_default": True,
        "communication_emission_enabled_by_default": False,
    }


def _contract_reproductive_profile(config: Any) -> SignalProfile:
    if reproductive_signal_emission_enabled(config):
        return reproductive_readiness_signal_profile(config)
    return inert_signal_profile(
        REPRODUCTIVE_READINESS_PROFILE_ID,
        field_name=REPRODUCTIVE_SIGNAL_FIELD,
        source_kind="biology_gated_reproduction_readiness",
        profile_index=0,
    )


def signal_substrate_enabled(config: Any) -> bool:
    return bool(getattr(config, "enabled", True))


def reproductive_signal_emission_enabled(config: Any) -> bool:
    return (
        signal_substrate_enabled(config)
        and bool(getattr(config, "reproductive_signal_emission_enabled", False))
        and float(getattr(config, "max_intensity", 0.0)) > 0.0
        and int(getattr(config, "reproductive_signal_duration_ticks", 0)) > 0
        and _positive_signal_intensity_ceiling(
            config,
            base_attr="reproductive_signal_base_intensity",
            bonus_attr="reproductive_signal_trait_intensity_bonus",
        )
    )


def communication_signal_emission_enabled(config: Any) -> bool:
    return (
        signal_substrate_enabled(config)
        and bool(getattr(config, "communication_signal_emission_enabled", False))
        and float(getattr(config, "max_intensity", 0.0)) > 0.0
        and int(getattr(config, "communication_signal_duration_ticks", 0)) > 0
        and _positive_signal_intensity_ceiling(
            config,
            base_attr="communication_signal_base_intensity",
            bonus_attr="communication_signal_trait_intensity_bonus",
        )
    )


def _positive_signal_intensity_ceiling(
    config: Any,
    *,
    base_attr: str,
    bonus_attr: str,
) -> bool:
    configured_ceiling = (
        float(getattr(config, base_attr, 0.0))
        + float(getattr(config, bonus_attr, 0.0))
    )
    return min(float(getattr(config, "max_intensity", 0.0)), configured_ceiling) > 0.0


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

    changed = _decay_emission_list(world.reproductive_signal_emissions) or changed
    changed = _decay_emission_list(world.communication_signal_emissions) or changed
    if changed:
        invalidate_signal_state(world)


def emit_reproductive_readiness_signals(
    world: Any,
    agents: Iterable[Any],
) -> dict[str, object]:
    config = world.config.signals
    if not reproductive_signal_emission_enabled(config):
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
        emission = SignalEmission(
            field_name=REPRODUCTIVE_SIGNAL_FIELD,
            profile_id=REPRODUCTIVE_READINESS_PROFILE_ID,
            source_kind="biology_gated_reproduction_readiness",
            source_agent_id=agent.agent_id,
            token_id=None,
            profile_index=0,
            x=agent.x,
            y=agent.y,
            intensity=intensity,
            radius=config.reproductive_signal_radius,
            duration_ticks=config.reproductive_signal_duration_ticks,
            remaining_ticks=config.reproductive_signal_duration_ticks,
            decay_rate=config.reproductive_signal_decay_rate,
            energy_cost=cost,
            emitted_tick=int(getattr(world, "tick", 0)),
        )
        world.reproductive_signal_emissions.append(emission)
        _record_signal_emission_event(world, emission)
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


def communication_signal_action_available(
    world: Any,
    agent: Any,
    action: str,
) -> bool:
    config = world.config.signals
    if (
        not communication_signal_emission_enabled(config)
        or not agent.alive
        or not world._in_bounds(agent.x, agent.y)
        or world.grid[agent.y][agent.x].terrain == "water"
    ):
        return False
    parsed = parse_communication_signal_action(action, config)
    if parsed is None:
        return False
    return _communication_signal_trait_bias(agent) > SIGNAL_EPSILON


def emit_communication_signal_action(
    world: Any,
    agent: Any,
    action: str,
) -> dict[str, object]:
    config = world.config.signals
    parsed = parse_communication_signal_action(action, config)
    if parsed is None:
        return _communication_signal_outcome(
            emitted=False,
            invalid_reason="unknown_signal_profile",
        )
    token_id, profile_index = parsed
    if not communication_signal_action_available(world, agent, action):
        return _communication_signal_outcome(
            emitted=False,
            token_id=token_id,
            profile_index=profile_index,
            invalid_reason="communication_signal_not_available",
        )

    profile = communication_signal_profile(
        config,
        token_id=token_id,
        profile_index=profile_index,
    )
    trait_bias = _communication_signal_trait_bias(agent)
    intensity = min(config.max_intensity, profile.intensity * trait_bias)
    if intensity <= SIGNAL_EPSILON:
        return _communication_signal_outcome(
            emitted=False,
            token_id=token_id,
            profile_index=profile_index,
            invalid_reason="zero_intensity",
        )
    cost = profile.energy_cost * trait_bias
    if cost > 0:
        agent.energy = max(0.0, agent.energy - cost)

    emission = SignalEmission(
        field_name=COMMUNICATION_SIGNAL_FIELD,
        profile_id=profile.profile_id,
        source_kind="policy_requested_opaque_communication",
        source_agent_id=agent.agent_id,
        token_id=token_id,
        profile_index=profile_index,
        x=agent.x,
        y=agent.y,
        intensity=intensity,
        radius=profile.radius,
        duration_ticks=profile.duration_ticks,
        remaining_ticks=profile.duration_ticks,
        decay_rate=profile.decay_rate,
        energy_cost=cost,
        emitted_tick=int(getattr(world, "tick", 0)),
    )
    world.communication_signal_emissions.append(emission)
    _record_signal_emission_event(world, emission)
    _record_runtime_cost(world, "signal_emissions", 1)
    _accumulate_signal_totals(
        world.tick_signal_totals,
        communication_emissions=1.0,
        energy_spent=cost,
    )
    _accumulate_signal_totals(
        world.run_signal_totals,
        communication_emissions=1.0,
        energy_spent=cost,
    )
    invalidate_signal_state(world)
    return _communication_signal_outcome(
        emitted=True,
        token_id=token_id,
        profile_index=profile_index,
        intensity=intensity,
        radius=profile.radius,
        duration_ticks=profile.duration_ticks,
        decay_rate=profile.decay_rate,
        energy_cost=cost,
    )


def parse_communication_signal_action(
    action: str,
    config: Any,
) -> tuple[int, int] | None:
    if not action.startswith("signal_"):
        return None
    parts = action.split("_")
    if len(parts) != 4 or parts[2] != "profile":
        return None
    try:
        token_id = int(parts[1])
        profile_index = int(parts[3])
    except ValueError:
        return None
    if not (0 <= token_id < int(config.communication_token_count)):
        return None
    if not (0 <= profile_index < int(config.communication_profiles_per_token)):
        return None
    return token_id, profile_index


def build_signal_state(world: Any) -> SignalFieldState:
    config = world.config.signals
    if not config.enabled:
        return empty_signal_state(world.config.width, world.config.height)
    return SignalFieldState(
        reproductive_signal=_diffuse_signal_emissions(
            world,
            world.reproductive_signal_emissions,
            max_intensity=config.max_intensity,
        ),
        communication_signal=_diffuse_signal_emissions(
            world,
            world.communication_signal_emissions,
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


def signal_emission_debug_snapshot(world: Any) -> dict[str, object]:
    return {
        "schema_version": SIGNAL_CONTRACT_VERSION,
        "policy_visible": False,
        "events": list(getattr(world, "tick_signal_emission_events", [])),
        "active_counts": {
            REPRODUCTIVE_SIGNAL_FIELD: len(
                getattr(world, "reproductive_signal_emissions", [])
            ),
            COMMUNICATION_SIGNAL_FIELD: len(
                getattr(world, "communication_signal_emissions", [])
            ),
        },
    }


def _decay_emission_list(emissions: list[SignalEmission]) -> bool:
    if not emissions:
        return False
    retained: list[SignalEmission] = []
    for emission in emissions:
        remaining_ticks = emission.remaining_ticks - 1
        intensity = emission.intensity * emission.decay_rate
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


def _communication_signal_trait_bias(agent: Any) -> float:
    return max(
        0.0,
        min(1.0, float(agent.genome.reproductive.signal_emission_bias)),
    )


def _communication_signal_outcome(
    *,
    emitted: bool,
    token_id: int | None = None,
    profile_index: int | None = None,
    intensity: float = 0.0,
    radius: int = 0,
    duration_ticks: int = 0,
    decay_rate: float = 0.0,
    energy_cost: float = 0.0,
    invalid_reason: str | None = None,
) -> dict[str, object]:
    return {
        "emitted": emitted,
        "token_id": token_id,
        "profile_index": profile_index,
        "intensity": round(float(intensity), 4),
        "radius": int(radius),
        "duration_ticks": int(duration_ticks),
        "decay_rate": round(float(decay_rate), 4),
        "energy_cost": round(float(energy_cost), 4),
        "invalid_reason": invalid_reason,
    }


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
    max_intensity: float,
) -> list[list[float]]:
    width = world.config.width
    height = world.config.height
    if max_intensity <= 0:
        return [[0.0 for _ in range(width)] for _ in range(height)]

    sources_by_radius: dict[int, dict[int, float]] = {}
    for emission in emissions:
        if emission.intensity <= SIGNAL_EPSILON or emission.remaining_ticks <= 0:
            continue
        if not world._in_bounds(emission.x, emission.y):
            continue
        if world.grid[emission.y][emission.x].terrain == "water":
            continue
        source_index = emission.y * width + emission.x
        radius_sources = sources_by_radius.setdefault(max(0, emission.radius), {})
        radius_sources[source_index] = min(
            max_intensity,
            radius_sources.get(source_index, 0.0) + emission.intensity,
        )
    return _diffuse_sparse_signal_fields(
        world,
        sources_by_radius,
        max_intensity=max_intensity,
    )


def _record_signal_emission_event(world: Any, emission: SignalEmission) -> None:
    events = getattr(world, "tick_signal_emission_events", None)
    if events is not None:
        events.append(emission.to_debug_dict())


def _diffuse_sparse_signal_fields(
    world: Any,
    sources_by_radius: dict[int, dict[int, float]],
    *,
    max_intensity: float,
) -> list[list[float]]:
    _record_runtime_cost(world, "signal_diffusions")
    width = world.config.width
    height = world.config.height
    field = [0.0 for _ in range(width * height)]
    if not sources_by_radius:
        return [
            field[row_start : row_start + width]
            for row_start in range(0, width * height, width)
        ]

    for radius, sources in sorted(sources_by_radius.items()):
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
