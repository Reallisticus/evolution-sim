from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Iterable, Sequence

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld


DEFAULT_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)


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


def _run_evaluation(seed: int, ticks: int, mode: RunMode) -> dict[str, object]:
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
        },
        "combat": summary["combat_end"],
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


def _aggregate_report(runs: Sequence[dict[str, object]]) -> dict[str, object]:
    aggregate: dict[str, object] = {
        "run_count": len(runs),
        "viable_runs": sum(1 for run in runs if not bool(run["extinct"])),
        "extinctions": sum(1 for run in runs if bool(run["extinct"])),
        "alive_agents": _series_stats([int(run["alive_agents"]) for run in runs]),
        "births": _series_stats([int(run["births"]) for run in runs]),
        "deaths": _series_stats([int(run["deaths"]) for run in runs]),
        "peak_alive_agents": _series_stats([int(run["peak_alive_agents"]) for run in runs]),
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
        "ecology_state_counts_at_end": _count_totals(
            [run["ecology"]["counts"] for run in runs]
        ),
        "hydrology_primary_counts_at_end": _count_totals(
            [run["hydrology"]["primary_counts"] for run in runs]
        ),
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
) -> dict[str, object]:
    if ticks <= 0:
        raise ValueError("ticks must be positive")
    if not seeds:
        raise ValueError("at least one seed is required")

    runs = [_run_evaluation(seed=seed, ticks=ticks, mode=mode) for seed in seeds]
    return {
        "protocol": {
            "seeds": list(seeds),
            "ticks": ticks,
            "mode": mode.value,
            "run_count": len(runs),
            "min_alive_agents": min_alive_agents,
            "min_births": min_births,
            "dominance_warning_share": dominance_warning_share,
        },
        "runs": runs,
        "aggregate": _aggregate_report(runs),
        "flags": _build_flags(
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
