from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import resource
import statistics
import time
from dataclasses import asdict, dataclass

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld

from .golden_harness import GOLDEN_SPECIATION_SEED


@dataclass(frozen=True, slots=True)
class BenchScenario:
    name: str
    seed: int
    ticks: int
    mode: RunMode


SCENARIOS: tuple[BenchScenario, ...] = (
    BenchScenario("full_seed7_ticks20", 7, 20, RunMode.FULL_REPLAY),
    BenchScenario("full_seed7_ticks100", 7, 100, RunMode.FULL_REPLAY),
    BenchScenario(
        "full_speciation_seed_ticks320",
        GOLDEN_SPECIATION_SEED,
        320,
        RunMode.FULL_REPLAY,
    ),
    BenchScenario("summary_seed7_ticks20", 7, 20, RunMode.SUMMARY_ONLY),
    BenchScenario("summary_seed7_ticks100", 7, 100, RunMode.SUMMARY_ONLY),
    BenchScenario(
        "summary_speciation_seed_ticks320",
        GOLDEN_SPECIATION_SEED,
        320,
        RunMode.SUMMARY_ONLY,
    ),
)


def _run_once(scenario: BenchScenario) -> dict[str, float | int | None]:
    start = time.perf_counter()
    result = SimulationWorld(WorldConfig(seed=scenario.seed, max_ticks=scenario.ticks)).run(
        mode=scenario.mode
    )
    wall_seconds = time.perf_counter() - start
    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    replay_size = None
    if result.viewer is not None and result.events is not None:
        payload = {
            "run_id": result.run_id,
            "config": result.config,
            "summary": result.summary,
            "events": result.events,
            "viewer": result.viewer,
        }
        replay_size = len(json.dumps(payload, indent=2).encode("utf-8"))
    return {
        "wall_seconds": wall_seconds,
        "peak_rss_kib": int(peak_rss_kib),
        "replay_size_bytes": replay_size,
    }


def _multi_instance_worker(payload: tuple[int, int]) -> float:
    seed, ticks = payload
    start = time.perf_counter()
    SimulationWorld(WorldConfig(seed=seed, max_ticks=ticks)).run(mode=RunMode.SUMMARY_ONLY)
    return time.perf_counter() - start


def _percentile(values: list[float | int], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _scenario_stats(scenario: BenchScenario, runs: list[dict[str, float | int | None]]) -> dict[str, object]:
    wall = [float(run["wall_seconds"]) for run in runs]
    rss = [int(run["peak_rss_kib"]) for run in runs]
    replay_sizes = [
        int(run["replay_size_bytes"])
        for run in runs
        if run["replay_size_bytes"] is not None
    ]
    return {
        **asdict(scenario),
        "median_wall_seconds": round(statistics.median(wall), 4),
        "p95_wall_seconds": round(_percentile(wall, 0.95), 4),
        "median_peak_rss_kib": int(statistics.median(rss)),
        "p95_peak_rss_kib": int(_percentile(rss, 0.95)),
        "median_replay_size_bytes": int(statistics.median(replay_sizes)) if replay_sizes else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark simulator modes with a fixed protocol.")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=[scenario.name for scenario in SCENARIOS],
        help="Run only the named scenario. May be supplied more than once.",
    )
    parser.add_argument(
        "--skip-multiprocess",
        action="store_true",
        help="Skip the multi-process summary rollout batch.",
    )
    parser.add_argument(
        "--multiprocess-timeout-seconds",
        type=float,
        default=120.0,
        help="Timeout for each multi-process summary rollout worker.",
    )
    args = parser.parse_args()

    scenarios = (
        tuple(scenario for scenario in SCENARIOS if scenario.name in set(args.scenario))
        if args.scenario
        else SCENARIOS
    )
    results: list[dict[str, object]] = []
    for scenario in scenarios:
        for _ in range(args.warmup):
            _run_once(scenario)
        runs = [_run_once(scenario) for _ in range(args.runs)]
        results.append(_scenario_stats(scenario, runs))

    payload = {
        "protocol": {
            "warmup_runs": args.warmup,
            "measured_runs": args.runs,
            "machine_profile": {
                "cpu": platform.processor() or platform.machine(),
                "ram_gib": round(
                    (
                        os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
                    )
                    / (1024**3),
                    2,
                )
                if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names
                else None,
                "os": platform.platform(),
                "python": platform.python_version(),
            },
        },
        "scenarios": results,
        "multi_process_summary_rollout": None,
    }
    if not args.skip_multiprocess:
        worker_count = max(1, min(4, os.cpu_count() or 1))
        batch_payload = [(seed, 100) for seed in range(1, worker_count + 1)]
        with mp.Pool(processes=worker_count) as pool:
            jobs = [pool.apply_async(_multi_instance_worker, (payload,)) for payload in batch_payload]
            rollout_times = [
                job.get(timeout=args.multiprocess_timeout_seconds) for job in jobs
            ]
        payload["multi_process_summary_rollout"] = {
            "workers": worker_count,
            "median_wall_seconds": round(statistics.median(rollout_times), 4),
            "p95_wall_seconds": round(_percentile(rollout_times, 0.95), 4),
        }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
