from __future__ import annotations


RUNTIME_COST_COUNTER_NAMES: tuple[str, ...] = (
    "observation_builds",
    "action_mask_builds",
    "biotic_state_builds",
    "biotic_state_cache_hits",
    "biotic_state_invalidations",
    "biotic_diffusions",
    "biotic_diffusion_target_cache_hits",
    "biotic_diffusion_target_cache_misses",
    "signal_state_builds",
    "signal_state_cache_hits",
    "signal_state_invalidations",
    "signal_diffusions",
    "signal_diffusion_target_cache_hits",
    "signal_diffusion_target_cache_misses",
    "signal_emissions",
    "resource_pressure_accounting_updates",
)


def empty_runtime_cost_counters() -> dict[str, int]:
    return {name: 0 for name in RUNTIME_COST_COUNTER_NAMES}


def record_runtime_cost(
    counters: dict[str, int],
    name: str,
    amount: int = 1,
) -> None:
    counters[name] = counters.get(name, 0) + amount
