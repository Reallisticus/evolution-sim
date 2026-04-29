from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import queue
import resource
import statistics
import time
import traceback
from dataclasses import asdict, dataclass
from tempfile import TemporaryDirectory

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.io import JsonlTrajectoryWriter

from .golden_harness import GOLDEN_SPECIATION_SEED


@dataclass(frozen=True, slots=True)
class BenchScenario:
    name: str
    seed: int
    ticks: int
    mode: RunMode
    record_trajectory: bool | None = None
    stream_trajectory: bool = False


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
        "trajectory_stream_seed7_ticks100",
        7,
        100,
        RunMode.SUMMARY_ONLY,
        stream_trajectory=True,
    ),
    BenchScenario(
        "summary_speciation_seed_ticks320",
        GOLDEN_SPECIATION_SEED,
        320,
        RunMode.SUMMARY_ONLY,
    ),
)


def _ru_maxrss_to_kib(raw_peak_rss: int, *, system: str | None = None) -> int:
    system_name = system or platform.system()
    if system_name == "Darwin":
        return int(math.ceil(raw_peak_rss / 1024))
    return int(raw_peak_rss)


def _run_once(scenario: BenchScenario) -> dict[str, object]:
    start = time.perf_counter()
    world = SimulationWorld(WorldConfig(seed=scenario.seed, max_ticks=scenario.ticks))
    trajectory_record_count = 0
    trajectory_output_bytes = None
    if scenario.stream_trajectory:
        with TemporaryDirectory() as tmpdir:
            output_path = f"{tmpdir}/{scenario.name}.jsonl.gz"
            writer = JsonlTrajectoryWriter(output_path)
            result = world.run(mode=scenario.mode, trajectory_sink=writer)
            trajectory_record_count = writer.record_count
            trajectory_output_bytes = os.path.getsize(output_path)
    else:
        result = world.run(
            mode=scenario.mode,
            record_trajectory=scenario.record_trajectory,
        )
        trajectory_record_count = len(world.trajectory_records)
    wall_seconds = time.perf_counter() - start
    peak_rss_kib = _ru_maxrss_to_kib(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
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
        "trajectory_record_count": trajectory_record_count,
        "trajectory_output_bytes": trajectory_output_bytes,
        "runtime_cost_counters": dict(world.runtime_cost_counters),
    }


def _run_once_worker(
    scenario: BenchScenario,
    result_queue: mp.Queue,
) -> None:
    try:
        result_queue.put({"result": _run_once(scenario)})
    except BaseException:
        result_queue.put({"error": traceback.format_exc()})


def _run_once_isolated(
    scenario: BenchScenario,
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    context = mp.get_context()
    result_queue = context.Queue()
    process = context.Process(target=_run_once_worker, args=(scenario, result_queue))
    process.start()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(
            f"Benchmark scenario {scenario.name} exceeded {timeout_seconds:g}s"
        )

    try:
        message = result_queue.get(timeout=1.0)
    except queue.Empty as exc:
        raise RuntimeError(
            f"Benchmark scenario {scenario.name} exited without a result "
            f"(exitcode={process.exitcode})"
        ) from exc
    finally:
        result_queue.close()

    if "error" in message:
        raise RuntimeError(
            f"Benchmark scenario {scenario.name} failed in worker:\n{message['error']}"
        )
    return message["result"]


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


def _scenario_stats(scenario: BenchScenario, runs: list[dict[str, object]]) -> dict[str, object]:
    wall = [float(run["wall_seconds"]) for run in runs]
    rss = [int(run["peak_rss_kib"]) for run in runs]
    replay_sizes = [
        int(run["replay_size_bytes"])
        for run in runs
        if run["replay_size_bytes"] is not None
    ]
    trajectory_record_counts = [int(run["trajectory_record_count"]) for run in runs]
    trajectory_output_sizes = [
        int(run["trajectory_output_bytes"])
        for run in runs
        if run["trajectory_output_bytes"] is not None
    ]
    counter_names = sorted(
        {
            name
            for run in runs
            for name in dict(run["runtime_cost_counters"]).keys()
        }
    )
    median_runtime_cost_counters = {
        name: int(
            statistics.median(
                int(dict(run["runtime_cost_counters"]).get(name, 0))
                for run in runs
            )
        )
        for name in counter_names
    }
    return {
        **asdict(scenario),
        "median_wall_seconds": round(statistics.median(wall), 4),
        "p95_wall_seconds": round(_percentile(wall, 0.95), 4),
        "median_peak_rss_kib": int(statistics.median(rss)),
        "p95_peak_rss_kib": int(_percentile(rss, 0.95)),
        "median_replay_size_bytes": int(statistics.median(replay_sizes)) if replay_sizes else None,
        "median_trajectory_record_count": int(statistics.median(trajectory_record_counts)),
        "median_trajectory_output_bytes": (
            int(statistics.median(trajectory_output_sizes))
            if trajectory_output_sizes
            else None
        ),
        "median_runtime_cost_counters": median_runtime_cost_counters,
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
    parser.add_argument(
        "--scenario-timeout-seconds",
        type=float,
        default=300.0,
        help="Timeout for each isolated benchmark scenario repetition.",
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
            _run_once_isolated(
                scenario,
                timeout_seconds=args.scenario_timeout_seconds,
            )
        runs = [
            _run_once_isolated(
                scenario,
                timeout_seconds=args.scenario_timeout_seconds,
            )
            for _ in range(args.runs)
        ]
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
            "rss_unit": "KiB",
            "scenario_repetition_isolation": "fresh_process",
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
