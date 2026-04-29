from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld


DEFAULT_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
ProgressCallback = Callable[[str], None]
DIET_TOTAL_FIELDS: frozenset[str] = frozenset(
    {
        "plant_events",
        "plant_energy",
        "fresh_kill_events",
        "fresh_kill_energy",
        "carcass_events",
        "carcass_energy",
        "animal_events",
        "animal_energy",
    }
)


def parse_seed_selection(seed_args: Sequence[int] | None, seeds_arg: str | None) -> list[int]:
    raw_seeds: list[int] = []
    if seeds_arg:
        for chunk in seeds_arg.split(","):
            value = chunk.strip()
            if value:
                raw_seeds.append(int(value))
    if seed_args:
        raw_seeds.extend(seed_args)
    if not raw_seeds:
        raw_seeds.extend(DEFAULT_SEEDS)

    seen: set[int] = set()
    seeds: list[int] = []
    for seed in raw_seeds:
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return seeds


def _round_float(value: float) -> float:
    return round(value, 4)


def _series_stats(values: Sequence[int | float]) -> dict[str, object]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "median": _round_float(float(statistics.median(values))),
        "mean": _round_float(statistics.fmean(values)),
        "max": max(values),
    }


def _count_totals(maps: Iterable[dict[str, int]]) -> dict[str, dict[str, int | float]]:
    maps = list(maps)
    totals: dict[str, int] = {}
    for count_map in maps:
        for key, value in count_map.items():
            totals[key] = totals.get(key, 0) + int(value)
    ordered_totals = {key: totals[key] for key in sorted(totals)}
    run_count = max(1, len(maps))
    return {
        "total": ordered_totals,
        "per_run_mean": {
            key: _round_float(value / run_count) for key, value in ordered_totals.items()
        },
    }


def _numeric_totals(
    maps: Iterable[dict[str, object]],
) -> dict[str, dict[str, int | float]]:
    maps = list(maps)
    totals: dict[str, float] = {}
    integer_only: dict[str, bool] = {}
    for numeric_map in maps:
        for key, value in numeric_map.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            totals[key] = totals.get(key, 0.0) + float(value)
            integer_only[key] = integer_only.get(key, True) and isinstance(value, int)
    ordered_totals = {
        key: int(value) if integer_only.get(key, False) else _round_float(value)
        for key, value in sorted(totals.items())
    }
    run_count = max(1, len(maps))
    return {
        "total": ordered_totals,
        "per_run_mean": {
            key: _round_float(float(value) / run_count)
            for key, value in ordered_totals.items()
        },
    }


def _diet_totals(
    maps: Iterable[dict[str, object]],
) -> dict[str, dict[str, int | float]]:
    return _numeric_totals(
        [
            {
                key: value
                for key, value in diet_map.items()
                if key in DIET_TOTAL_FIELDS
            }
            for diet_map in maps
        ]
    )


def _nested_count_totals(
    maps: Iterable[dict[str, dict[str, int]]],
) -> dict[str, dict[str, dict[str, int | float]]]:
    maps = list(maps)
    outer_keys = sorted({outer for nested in maps for outer in nested})
    return {
        outer_key: _count_totals(
            [
                nested.get(outer_key, {})
                for nested in maps
            ]
        )
        for outer_key in outer_keys
    }


def _three_level_count_totals(
    maps: Iterable[dict[str, dict[str, dict[str, int]]]],
) -> dict[str, dict[str, dict[str, dict[str, int | float]]]]:
    maps = list(maps)
    outer_keys = sorted({outer for nested in maps for outer in nested})
    return {
        outer_key: _nested_count_totals(
            [
                nested.get(outer_key, {})
                for nested in maps
            ]
        )
        for outer_key in outer_keys
    }


def _nested_numeric_totals(
    maps: Iterable[dict[str, dict[str, object]]],
) -> dict[str, dict[str, dict[str, int | float]]]:
    maps = list(maps)
    outer_keys = sorted({outer for nested in maps for outer in nested})
    return {
        outer_key: _numeric_totals(
            [
                nested.get(outer_key, {})
                for nested in maps
            ]
        )
        for outer_key in outer_keys
    }


def _nested_diet_totals(
    maps: Iterable[dict[str, dict[str, object]]],
) -> dict[str, dict[str, dict[str, int | float]]]:
    maps = list(maps)
    outer_keys = sorted({outer for nested in maps for outer in nested})
    return {
        outer_key: _diet_totals(
            [
                nested.get(outer_key, {})
                for nested in maps
            ]
        )
        for outer_key in outer_keys
    }


def _presence_run_counts(maps: Iterable[dict[str, int]]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for count_map in maps:
        for key, value in count_map.items():
            totals.setdefault(key, 0)
            if int(value) > 0:
                totals[key] += 1
    return {key: totals[key] for key in sorted(totals)}


def _series_stats_by_key(
    maps: Iterable[dict[str, object]],
) -> dict[str, dict[str, object]]:
    maps = list(maps)
    keys = sorted({key for count_map in maps for key in count_map})
    return {
        key: _series_stats(
            [
                value
                for count_map in maps
                if (value := count_map.get(key)) is not None
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
            ]
        )
        for key in keys
    }


def _animal_resource_opportunity_run_counts(
    runs: Sequence[dict[str, object]],
) -> dict[str, dict[str, int]]:
    counters: dict[str, dict[str, int]] = {}
    for run in runs:
        trophic = run.get("trophic")
        if not isinstance(trophic, dict):
            continue
        opportunities = trophic.get("animal_resource_opportunity_by_meat_mode")
        if not isinstance(opportunities, dict):
            continue
        for mode, raw_counts in opportunities.items():
            if mode == "none":
                continue
            if not isinstance(raw_counts, dict):
                continue
            mode_counts = counters.setdefault(
                str(mode),
                {
                    "alive_runs": 0,
                    "no_animal_consumption_runs": 0,
                    "animal_resource_absent_runs": 0,
                    "animal_resource_present_unconsumed_runs": 0,
                    "animal_resource_present_unreachable_runs": 0,
                    "animal_resource_reachable_unconsumed_runs": 0,
                    "animal_resource_policy_actionable_runs": 0,
                    "animal_resource_reachable_policy_blocked_runs": 0,
                },
            )
            alive_ticks = int(raw_counts.get("alive_ticks", 0))
            if alive_ticks <= 0:
                continue
            consumption_events = int(
                raw_counts.get("animal_resource_consumption_events", 0)
            )
            present_ticks = int(raw_counts.get("animal_resource_present_ticks", 0))
            reachable_ticks = int(raw_counts.get("animal_resource_reachable_ticks", 0))
            policy_actionable_ticks = int(
                raw_counts.get("animal_resource_policy_actionable_ticks", 0)
            )
            reachable_policy_blocked_ticks = int(
                raw_counts.get(
                    "animal_resource_reachable_policy_blocked_ticks",
                    0,
                )
            )
            mode_counts["alive_runs"] += 1
            if consumption_events <= 0:
                mode_counts["no_animal_consumption_runs"] += 1
            if present_ticks <= 0:
                mode_counts["animal_resource_absent_runs"] += 1
            if present_ticks > 0 and consumption_events <= 0:
                mode_counts["animal_resource_present_unconsumed_runs"] += 1
                if reachable_ticks <= 0:
                    mode_counts["animal_resource_present_unreachable_runs"] += 1
                else:
                    mode_counts["animal_resource_reachable_unconsumed_runs"] += 1
            if policy_actionable_ticks > 0:
                mode_counts["animal_resource_policy_actionable_runs"] += 1
            if reachable_policy_blocked_ticks > 0:
                mode_counts["animal_resource_reachable_policy_blocked_runs"] += 1
    return {mode: counters[mode] for mode in sorted(counters)}


def _aggregate_trophic_lifecycle(runs: Sequence[dict[str, object]]) -> dict[str, object]:
    lifecycle_runs = [
        run["trophic_lifecycle"]
        for run in runs
        if isinstance(run.get("trophic_lifecycle"), dict)
    ]
    if not lifecycle_runs:
        return {}
    late_windows = [
        lifecycle["late_window"]
        for lifecycle in lifecycle_runs
        if isinstance(lifecycle.get("late_window"), dict)
    ]
    meat_mode_persistence_runs = [
        lifecycle["meat_mode_persistence"]
        for lifecycle in lifecycle_runs
        if isinstance(lifecycle.get("meat_mode_persistence"), dict)
    ]
    meat_mode_persistence = {}
    if meat_mode_persistence_runs:
        meat_mode_persistence = {
            "last_alive_tick_by_meat_mode": _series_stats_by_key(
                [
                    persistence.get("last_alive_tick_by_meat_mode", {})
                    for persistence in meat_mode_persistence_runs
                ]
            ),
            "births_by_parent_meat_mode_by_tick_band": _nested_count_totals(
                [
                    persistence.get("births_by_parent_meat_mode_by_tick_band", {})
                    for persistence in meat_mode_persistence_runs
                ]
            ),
            "births_by_child_meat_mode_by_tick_band": _nested_count_totals(
                [
                    persistence.get("births_by_child_meat_mode_by_tick_band", {})
                    for persistence in meat_mode_persistence_runs
                ]
            ),
            "deaths_by_meat_mode_by_tick_band": _nested_count_totals(
                [
                    persistence.get("deaths_by_meat_mode_by_tick_band", {})
                    for persistence in meat_mode_persistence_runs
                ]
            ),
            "death_causes_by_meat_mode_by_tick_band": _three_level_count_totals(
                [
                    persistence.get("death_causes_by_meat_mode_by_tick_band", {})
                    for persistence in meat_mode_persistence_runs
                ]
            ),
        }
    return {
        "initial_trophic_role_counts": _count_totals(
            [lifecycle["initial_trophic_role_counts"] for lifecycle in lifecycle_runs]
        ),
        "initial_meat_mode_counts": _count_totals(
            [lifecycle["initial_meat_mode_counts"] for lifecycle in lifecycle_runs]
        ),
        "births_by_parent_trophic_role": _count_totals(
            [lifecycle["births_by_parent_trophic_role"] for lifecycle in lifecycle_runs]
        ),
        "births_by_child_trophic_role": _count_totals(
            [lifecycle["births_by_child_trophic_role"] for lifecycle in lifecycle_runs]
        ),
        "births_by_parent_meat_mode": _count_totals(
            [lifecycle["births_by_parent_meat_mode"] for lifecycle in lifecycle_runs]
        ),
        "births_by_child_meat_mode": _count_totals(
            [lifecycle["births_by_child_meat_mode"] for lifecycle in lifecycle_runs]
        ),
        "deaths_by_trophic_role": _count_totals(
            [lifecycle["deaths_by_trophic_role"] for lifecycle in lifecycle_runs]
        ),
        "deaths_by_meat_mode": _count_totals(
            [lifecycle["deaths_by_meat_mode"] for lifecycle in lifecycle_runs]
        ),
        "death_causes": _count_totals(
            [lifecycle["death_causes"] for lifecycle in lifecycle_runs]
        ),
        "death_causes_by_trophic_role": _nested_count_totals(
            [lifecycle["death_causes_by_trophic_role"] for lifecycle in lifecycle_runs]
        ),
        "death_causes_by_meat_mode": _nested_count_totals(
            [lifecycle["death_causes_by_meat_mode"] for lifecycle in lifecycle_runs]
        ),
        "late_window_trophic_role_presence_runs": _presence_run_counts(
            [
                late_window["presence_ticks_by_trophic_role"]
                for late_window in late_windows
            ]
        ),
        "late_window_meat_mode_presence_runs": _presence_run_counts(
            [
                late_window["presence_ticks_by_meat_mode"]
                for late_window in late_windows
            ]
        ),
        "meat_mode_persistence": meat_mode_persistence,
    }


def _dominant_lineage(summary: dict[str, object]) -> dict[str, object] | None:
    top_lineages = summary.get("top_lineages")
    if not isinstance(top_lineages, list) or not top_lineages:
        return None
    dominant = dict(top_lineages[0])
    alive_agents = int(summary.get("alive_agents", 0))
    alive_lineage_agents = int(dominant.get("alive_agents", 0))
    dominant["alive_share"] = (
        _round_float(alive_lineage_agents / alive_agents) if alive_agents else 0.0
    )
    return dominant


def _dominant_species(summary: dict[str, object]) -> dict[str, object] | None:
    top_species = summary.get("top_species")
    if not isinstance(top_species, list) or not top_species:
        return None
    dominant = dict(top_species[0])
    alive_agents = int(summary.get("alive_agents", 0))
    alive_species_agents = int(dominant.get("alive_members", 0))
    dominant["alive_share"] = (
        _round_float(alive_species_agents / alive_agents) if alive_agents else 0.0
    )
    return dominant


def run_evaluation(seed: int, ticks: int, mode: RunMode) -> dict[str, object]:
    result = SimulationWorld(WorldConfig(seed=seed, max_ticks=ticks)).run(mode=mode)
    summary = result.summary
    record: dict[str, object] = {
        "seed": seed,
        "run_id": summary["run_id"],
        "ticks_executed": summary["ticks_executed"],
        "extinct": summary["extinct"],
        "alive_agents": summary["alive_agents"],
        "births": summary["births"],
        "deaths": summary["deaths"],
        "peak_alive_agents": summary["peak_alive_agents"],
        "max_agents": summary["max_agents"],
        "max_agent_saturation_at_end": summary["max_agent_saturation_at_end"],
        "peak_max_agent_saturation": summary["peak_max_agent_saturation"],
        "total_agents_seen": summary["total_agents_seen"],
        "last_birth_tick": summary["last_birth_tick"],
        "dominant_lineage": _dominant_lineage(summary),
        "hazard": {
            "counts": summary["hazard_counts_at_end"],
            "hazardous_tiles": summary["hazard_stats_at_end"]["hazardous_tiles"],
            "avg_hazard_level": summary["hazard_stats_at_end"]["avg_hazard_level"],
        },
        "trophic": {
            "role_counts": summary["trophic_role_counts_at_end"],
            "meat_mode_counts": summary["meat_mode_counts_at_end"],
            "diet": summary["diet_end"],
            "diet_by_trophic_role": summary["diet_by_trophic_role_end"],
            "diet_by_meat_mode": summary["diet_by_meat_mode_end"],
            "animal_resource_opportunity_by_meat_mode": summary[
                "animal_resource_opportunity_by_meat_mode_end"
            ],
        },
        "trophic_lifecycle": summary["trophic_lifecycle"],
        "combat": summary["combat_end"],
        "reproduction": summary["reproduction_end"],
        "fresh_kill": summary["fresh_kill_end"],
        "carrion": summary["carcass_end"],
        "ecology": {
            "counts": summary["ecology_state_counts_at_end"],
            "avg_vegetation": summary["ecology_stats_at_end"]["avg_vegetation"],
            "avg_recovery_debt": summary["ecology_stats_at_end"]["avg_recovery_debt"],
        },
        "hydrology": {
            "primary_counts": summary["hydrology_primary_counts_at_end"],
            "hard_access_tiles": summary["hydrology_primary_stats_at_end"][
                "hard_access_tiles"
            ],
        },
    }
    if "top_species" in summary:
        record["species"] = {
            "species_created": summary["species_created"],
            "alive_species_count": summary["alive_species_count"],
            "status_counts": summary["species_status_counts"],
            "dominant_species": _dominant_species(summary),
        }
    return record


_run_evaluation = run_evaluation


def _aggregate_report(runs: Sequence[dict[str, object]]) -> dict[str, object]:
    aggregate: dict[str, object] = {
        "run_count": len(runs),
        "viable_runs": sum(1 for run in runs if not bool(run["extinct"])),
        "extinctions": sum(1 for run in runs if bool(run["extinct"])),
        "alive_agents": _series_stats([int(run["alive_agents"]) for run in runs]),
        "births": _series_stats([int(run["births"]) for run in runs]),
        "deaths": _series_stats([int(run["deaths"]) for run in runs]),
        "peak_alive_agents": _series_stats([int(run["peak_alive_agents"]) for run in runs]),
        "max_agent_saturation_at_end": _series_stats(
            [float(run["max_agent_saturation_at_end"]) for run in runs]
        ),
        "peak_max_agent_saturation": _series_stats(
            [float(run["peak_max_agent_saturation"]) for run in runs]
        ),
        "total_agents_seen": _series_stats([int(run["total_agents_seen"]) for run in runs]),
        "last_birth_tick": _series_stats(
            [
                int(run["last_birth_tick"])
                for run in runs
                if run["last_birth_tick"] is not None
            ]
        ),
        "dominant_lineage_alive_share": _series_stats(
            [
                float(run["dominant_lineage"]["alive_share"])
                for run in runs
                if run["dominant_lineage"] is not None
            ]
        ),
        "hazardous_tiles": _series_stats(
            [int(run["hazard"]["hazardous_tiles"]) for run in runs]
        ),
        "avg_hazard_level": _series_stats(
            [float(run["hazard"]["avg_hazard_level"]) for run in runs]
        ),
        "hard_access_tiles": _series_stats(
            [int(run["hydrology"]["hard_access_tiles"]) for run in runs]
        ),
        "combat_attack_attempts": _series_stats(
            [int(run["combat"]["attack_attempts"]) for run in runs]
        ),
        "combat_kills": _series_stats([int(run["combat"]["kills"]) for run in runs]),
        "reproduction_biologically_ready_at_end": _series_stats(
            [
                int(run["reproduction"]["biologically_ready_agents"])
                for run in runs
            ]
        ),
        "reproduction_ready_at_end": _series_stats(
            [int(run["reproduction"]["ready_agents"]) for run in runs]
        ),
        "reproduction_by_trophic_role_at_end": _nested_count_totals(
            [run["reproduction"]["by_trophic_role"] for run in runs]
        ),
        "reproduction_by_meat_mode_at_end": _nested_count_totals(
            [run["reproduction"]["by_meat_mode"] for run in runs]
        ),
        "reproduction_biological_blockers_at_end": _count_totals(
            [run["reproduction"]["biological_blocker_counts"] for run in runs]
        ),
        "reproduction_biological_blockers_by_trophic_role_at_end": (
            _nested_count_totals(
                [
                    run["reproduction"][
                        "biological_blocker_counts_by_trophic_role"
                    ]
                    for run in runs
                ]
            )
        ),
        "reproduction_biological_blockers_by_meat_mode_at_end": (
            _nested_count_totals(
                [
                    run["reproduction"][
                        "biological_blocker_counts_by_meat_mode"
                    ]
                    for run in runs
                ]
            )
        ),
        "reproduction_energy_readiness_by_trophic_role_at_end": (
            _nested_numeric_totals(
                [
                    run["reproduction"][
                        "energy_readiness_by_trophic_role"
                    ]
                    for run in runs
                ]
            )
        ),
        "reproduction_energy_readiness_by_meat_mode_at_end": (
            _nested_numeric_totals(
                [
                    run["reproduction"][
                        "energy_readiness_by_meat_mode"
                    ]
                    for run in runs
                ]
            )
        ),
        "reproduction_blocked_by_max_population": _series_stats(
            [
                int(run["reproduction"]["blocked_run_counts"]["max_population"])
                for run in runs
            ]
        ),
        "reproduction_blocked_by_local_crowding": _series_stats(
            [
                int(run["reproduction"]["blocked_run_counts"]["local_crowding"])
                for run in runs
            ]
        ),
        "reproduction_blocked_run_counts_by_trophic_role": _nested_count_totals(
            [
                run["reproduction"]["blocked_run_counts_by_trophic_role"]
                for run in runs
            ]
        ),
        "reproduction_blocked_run_counts_by_meat_mode": _nested_count_totals(
            [
                run["reproduction"]["blocked_run_counts_by_meat_mode"]
                for run in runs
            ]
        ),
        "hazard_damage_taken": _series_stats(
            [float(run["combat"]["hazard_damage_taken"]) for run in runs]
        ),
        "fresh_kill_energy_consumed": _series_stats(
            [float(run["fresh_kill"]["energy_consumed"]) for run in runs]
        ),
        "carrion_energy_deposited": _series_stats(
            [float(run["carrion"]["energy_deposited"]) for run in runs]
        ),
        "carrion_energy_consumed": _series_stats(
            [float(run["carrion"]["energy_consumed"]) for run in runs]
        ),
        "avg_vegetation": _series_stats(
            [float(run["ecology"]["avg_vegetation"]) for run in runs]
        ),
        "avg_recovery_debt": _series_stats(
            [float(run["ecology"]["avg_recovery_debt"]) for run in runs]
        ),
        "hazard_counts_at_end": _count_totals(
            [run["hazard"]["counts"] for run in runs]
        ),
        "trophic_role_counts_at_end": _count_totals(
            [run["trophic"]["role_counts"] for run in runs]
        ),
        "meat_mode_counts_at_end": _count_totals(
            [run["trophic"]["meat_mode_counts"] for run in runs]
        ),
        "diet_by_trophic_role_at_end": _nested_diet_totals(
            [run["trophic"]["diet_by_trophic_role"] for run in runs]
        ),
        "diet_by_meat_mode_at_end": _nested_diet_totals(
            [run["trophic"]["diet_by_meat_mode"] for run in runs]
        ),
        "animal_resource_opportunity_by_meat_mode_at_end": _nested_numeric_totals(
            [
                run["trophic"]["animal_resource_opportunity_by_meat_mode"]
                for run in runs
            ]
        ),
        "animal_resource_opportunity_run_counts_by_meat_mode": (
            _animal_resource_opportunity_run_counts(runs)
        ),
        "ecology_state_counts_at_end": _count_totals(
            [run["ecology"]["counts"] for run in runs]
        ),
        "hydrology_primary_counts_at_end": _count_totals(
            [run["hydrology"]["primary_counts"] for run in runs]
        ),
        "trophic_lifecycle": _aggregate_trophic_lifecycle(runs),
    }
    species_runs = [run for run in runs if "species" in run]
    if species_runs:
        aggregate["species_created"] = _series_stats(
            [int(run["species"]["species_created"]) for run in species_runs]
        )
        aggregate["alive_species_count"] = _series_stats(
            [int(run["species"]["alive_species_count"]) for run in species_runs]
        )
        aggregate["dominant_species_alive_share"] = _series_stats(
            [
                float(run["species"]["dominant_species"]["alive_share"])
                for run in species_runs
                if run["species"]["dominant_species"] is not None
            ]
        )
    return aggregate


def _build_flags(
    runs: Sequence[dict[str, object]],
    *,
    min_alive_agents: int,
    min_births: int,
    dominance_warning_share: float,
) -> list[dict[str, object]]:
    flags: list[dict[str, object]] = []
    for run in runs:
        seed = int(run["seed"])
        if bool(run["extinct"]):
            flags.append(
                {
                    "severity": "error",
                    "seed": seed,
                    "field": "extinct",
                    "message": "Run ended with no living agents.",
                }
            )
        if int(run["alive_agents"]) < min_alive_agents:
            flags.append(
                {
                    "severity": "error",
                    "seed": seed,
                    "field": "alive_agents",
                    "message": (
                        f"Run ended below the alive-agent floor of {min_alive_agents}."
                    ),
                }
            )
        if int(run["births"]) < min_births:
            flags.append(
                {
                    "severity": "warning",
                    "seed": seed,
                    "field": "births",
                    "message": f"Run produced fewer than {min_births} births.",
                }
            )
        dominant_lineage = run["dominant_lineage"]
        if (
            dominant_lineage is not None
            and float(dominant_lineage["alive_share"]) >= dominance_warning_share
        ):
            flags.append(
                {
                    "severity": "warning",
                    "seed": seed,
                    "field": "dominant_lineage.alive_share",
                    "message": (
                        "One lineage dominates the ending population at "
                        f"{dominant_lineage['alive_share']} share."
                    ),
                }
            )
    return flags


def build_evaluation_report(
    *,
    seeds: Sequence[int],
    ticks: int,
    mode: RunMode = RunMode.SUMMARY_ONLY,
    min_alive_agents: int = 1,
    min_births: int = 1,
    dominance_warning_share: float = 0.75,
    progress: ProgressCallback | None = None,
) -> dict[str, object]:
    if ticks <= 0:
        raise ValueError("ticks must be positive")
    if not seeds:
        raise ValueError("at least one seed is required")

    runs: list[dict[str, object]] = []
    for index, seed in enumerate(seeds, start=1):
        if progress is not None:
            progress(f"seed {seed} start ({index}/{len(seeds)})")
        started = time.perf_counter()
        run = run_evaluation(seed=seed, ticks=ticks, mode=mode)
        run["wall_seconds"] = _round_float(time.perf_counter() - started)
        runs.append(run)
        if progress is not None:
            progress(
                "seed "
                f"{seed} complete ({index}/{len(seeds)}) "
                f"wall_seconds={run['wall_seconds']}"
            )
    return build_evaluation_report_from_runs(
        seeds=seeds,
        ticks=ticks,
        mode=mode,
        min_alive_agents=min_alive_agents,
        min_births=min_births,
        dominance_warning_share=dominance_warning_share,
        runs=runs,
    )


def build_evaluation_report_from_runs(
    *,
    seeds: Sequence[int],
    ticks: int,
    mode: RunMode,
    min_alive_agents: int,
    min_births: int,
    dominance_warning_share: float,
    runs: Sequence[dict[str, object]],
    run_errors: Sequence[dict[str, object]] = (),
) -> dict[str, object]:
    return {
        "protocol": {
            "seeds": list(seeds),
            "ticks": ticks,
            "mode": mode.value,
            "requested_run_count": len(seeds),
            "run_count": len(runs),
            "min_alive_agents": min_alive_agents,
            "min_births": min_births,
            "dominance_warning_share": dominance_warning_share,
        },
        "runs": runs,
        "aggregate": _aggregate_report(runs),
        "flags": list(run_errors) + _build_flags(
            runs,
            min_alive_agents=min_alive_agents,
            min_births=min_births,
            dominance_warning_share=dominance_warning_share,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate simulator outcomes across seeds without writing replays."
    )
    parser.add_argument(
        "--seeds",
        help="Comma-separated seed list. Defaults to 1,2,3,4,5 when no seed is supplied.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Add one seed to the evaluation. May be supplied more than once.",
    )
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in RunMode],
        default=RunMode.SUMMARY_ONLY.value,
        help="Use summary_only for sweeps; full_replay includes compact species fields.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    parser.add_argument("--min-alive-agents", type=int, default=1)
    parser.add_argument("--min-births", type=int, default=1)
    parser.add_argument("--dominance-warning-share", type=float, default=0.75)
    parser.add_argument(
        "--fail-on-flags",
        action="store_true",
        help="Exit non-zero if any warning or error flag is emitted.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = parse_seed_selection(args.seed, args.seeds)
    report = build_evaluation_report(
        seeds=seeds,
        ticks=args.ticks,
        mode=RunMode(args.mode),
        min_alive_agents=args.min_alive_agents,
        min_births=args.min_births,
        dominance_warning_share=args.dominance_warning_share,
    )
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if args.fail_on_flags and report["flags"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
