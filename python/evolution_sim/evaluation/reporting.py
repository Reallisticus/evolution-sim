from __future__ import annotations

import statistics
from collections.abc import Iterable, Mapping, Sequence


def round_float(value: float) -> float:
    return round(value, 4)


def dominant_lineage(summary: Mapping[str, object]) -> dict[str, object] | None:
    top_lineages = summary.get("top_lineages")
    if not isinstance(top_lineages, list) or not top_lineages:
        return None
    dominant = dict(top_lineages[0])
    alive_agents = int(summary.get("alive_agents", 0))
    alive_lineage_agents = int(dominant.get("alive_agents", 0))
    dominant["alive_share"] = (
        round_float(alive_lineage_agents / alive_agents) if alive_agents else 0.0
    )
    return dominant


def aggregate_report(runs: Sequence[dict[str, object]]) -> dict[str, object]:
    return {
        "run_count": len(runs),
        "summary_schema_versions": sorted(
            {
                str(run["summary_schema_version"])
                for run in runs
                if "summary_schema_version" in run
            }
        ),
        "viable_runs": sum(1 for run in runs if not bool(run["extinct"])),
        "extinctions": sum(1 for run in runs if bool(run["extinct"])),
        "alive_agents": _series_stats([int(run["alive_agents"]) for run in runs]),
        "births": _series_stats([int(run["births"]) for run in runs]),
        "deaths": _series_stats([int(run["deaths"]) for run in runs]),
        "peak_alive_agents": _series_stats(
            [int(run["peak_alive_agents"]) for run in runs]
        ),
        "max_agent_saturation_at_end": _series_stats(
            [float(run["max_agent_saturation_at_end"]) for run in runs]
        ),
        "peak_max_agent_saturation": _series_stats(
            [float(run["peak_max_agent_saturation"]) for run in runs]
        ),
        "total_agents_seen": _series_stats(
            [int(run["total_agents_seen"]) for run in runs]
        ),
        "land_tile_count": _series_stats([int(run["land_tile_count"]) for run in runs]),
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
        "trophic_role_counts_at_end": _count_totals(
            [_nested_int_map(run, "trophic", "role_counts") for run in runs]
        ),
        "meat_mode_counts_at_end": _count_totals(
            [_nested_int_map(run, "trophic", "meat_mode_counts") for run in runs]
        ),
    }


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
        "median": round_float(float(statistics.median(values))),
        "mean": round_float(statistics.fmean(values)),
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
            key: round_float(value / run_count)
            for key, value in ordered_totals.items()
        },
    }


def _nested_int_map(
    run: Mapping[str, object],
    outer_field: str,
    inner_field: str,
) -> dict[str, int]:
    outer = run.get(outer_field)
    if not isinstance(outer, Mapping):
        return {}
    inner = outer.get(inner_field)
    if not isinstance(inner, Mapping):
        return {}
    parsed: dict[str, int] = {}
    for key, value in inner.items():
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        parsed[str(key)] = value
    return parsed
