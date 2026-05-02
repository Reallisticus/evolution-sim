from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import resource
import statistics
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from tempfile import TemporaryDirectory
from typing import Any

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.io import (
    JsonlTrajectoryWriter,
    build_replay_payload,
    replay_payload_size_bytes,
)

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
        replay_size = replay_payload_size_bytes(build_replay_payload(result))
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
    result_connection: Any,
) -> None:
    try:
        result_connection.send({"result": _run_once(scenario)})
    except BaseException:
        result_connection.send({"error": traceback.format_exc()})
    finally:
        result_connection.close()


def _run_once_isolated(
    scenario: BenchScenario,
    *,
    timeout_seconds: float,
) -> dict[str, object]:
    context = mp.get_context()
    result_connection, worker_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_run_once_worker,
        args=(scenario, worker_connection),
    )
    process.start()
    worker_connection.close()

    try:
        if not result_connection.poll(timeout_seconds):
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=5.0)
                raise TimeoutError(
                    f"Benchmark scenario {scenario.name} exceeded {timeout_seconds:g}s"
                )
            raise RuntimeError(
                f"Benchmark scenario {scenario.name} exited without a result "
                f"(exitcode={process.exitcode})"
            )
        message = result_connection.recv()
    except EOFError as exc:
        raise RuntimeError(
            f"Benchmark scenario {scenario.name} exited without a result "
            f"(exitcode={process.exitcode})"
        ) from exc
    finally:
        result_connection.close()

    process.join(timeout=5.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)
        raise RuntimeError(
            f"Benchmark scenario {scenario.name} produced a result but did not exit"
        )

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


def _run_multi_process_summary_rollout(
    *,
    timeout_seconds: float,
    worker_count: int | None = None,
    ticks: int = 100,
) -> dict[str, object]:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if ticks <= 0:
        raise ValueError("ticks must be positive")
    resolved_worker_count = (
        max(1, min(4, os.cpu_count() or 1))
        if worker_count is None
        else worker_count
    )
    if resolved_worker_count <= 0:
        raise ValueError("worker_count must be positive")

    print(
        "[bench] multiprocess summary rollout start "
        f"workers={resolved_worker_count} ticks={ticks}",
        file=sys.stderr,
        flush=True,
    )
    pool = mp.Pool(processes=resolved_worker_count)
    try:
        pending: dict[int, Any] = {
            seed: pool.apply_async(_multi_instance_worker, ((seed, ticks),))
            for seed in range(1, resolved_worker_count + 1)
        }
        rollout_times: list[float] = []
        deadline = time.perf_counter() + timeout_seconds
        while pending:
            for seed, job in list(pending.items()):
                if not job.ready():
                    continue
                rollout_times.append(float(job.get(timeout=0.0)))
                del pending[seed]
                print(
                    "[bench] multiprocess summary worker "
                    f"seed={seed} complete "
                    f"({len(rollout_times)}/{resolved_worker_count})",
                    file=sys.stderr,
                    flush=True,
                )
            if not pending:
                break
            remaining_seconds = deadline - time.perf_counter()
            if remaining_seconds <= 0:
                pending_seeds = sorted(pending)
                completed = len(rollout_times)
                raise TimeoutError(
                    "Benchmark multi-process summary rollout exceeded "
                    f"{timeout_seconds:g}s; completed={completed}/"
                    f"{resolved_worker_count}; pending_seeds={pending_seeds}"
                )
            time.sleep(min(0.1, remaining_seconds))
    except BaseException:
        pool.terminate()
        pool.join()
        raise
    else:
        pool.close()
        pool.join()

    print("[bench] multiprocess summary rollout complete", file=sys.stderr, flush=True)
    return {
        "workers": resolved_worker_count,
        "median_wall_seconds": round(statistics.median(rollout_times), 4),
        "p95_wall_seconds": round(_percentile(rollout_times, 0.95), 4),
    }


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


def _benchmark_protocol(*, warmup_runs: int, measured_runs: int) -> dict[str, object]:
    return {
        "warmup_runs": warmup_runs,
        "measured_runs": measured_runs,
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
    }


def _validate_cli_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.warmup < 0:
        parser.error("--warmup must be greater than or equal to 0")
    if args.runs <= 0:
        parser.error("--runs must be greater than 0")
    for option_name, value in (
        ("--multiprocess-timeout-seconds", args.multiprocess_timeout_seconds),
        ("--scenario-timeout-seconds", args.scenario_timeout_seconds),
    ):
        if not math.isfinite(value) or value <= 0:
            parser.error(f"{option_name} must be a finite positive number")


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
    _validate_cli_args(args, parser)

    scenarios = (
        tuple(scenario for scenario in SCENARIOS if scenario.name in set(args.scenario))
        if args.scenario
        else SCENARIOS
    )
    results: list[dict[str, object]] = []
    payload = {
        "protocol": _benchmark_protocol(
            warmup_runs=args.warmup,
            measured_runs=args.runs,
        ),
        "complete": True,
        "scenarios": results,
        "multi_process_summary_rollout": None,
    }
    current_phase = "startup"
    try:
        for scenario in scenarios:
            current_phase = f"scenario {scenario.name}"
            print(f"[bench] scenario {scenario.name} start", file=sys.stderr, flush=True)
            for index in range(args.warmup):
                current_phase = (
                    f"scenario {scenario.name} warmup {index + 1}/{args.warmup}"
                )
                print(
                    f"[bench] scenario {scenario.name} warmup "
                    f"{index + 1}/{args.warmup} start",
                    file=sys.stderr,
                    flush=True,
                )
                _run_once_isolated(
                    scenario,
                    timeout_seconds=args.scenario_timeout_seconds,
                )
                print(
                    f"[bench] scenario {scenario.name} warmup "
                    f"{index + 1}/{args.warmup} complete",
                    file=sys.stderr,
                    flush=True,
                )
            runs: list[dict[str, object]] = []
            for index in range(args.runs):
                current_phase = (
                    f"scenario {scenario.name} run {index + 1}/{args.runs}"
                )
                print(
                    f"[bench] scenario {scenario.name} run {index + 1}/{args.runs} start",
                    file=sys.stderr,
                    flush=True,
                )
                runs.append(
                    _run_once_isolated(
                        scenario,
                        timeout_seconds=args.scenario_timeout_seconds,
                    )
                )
                print(
                    f"[bench] scenario {scenario.name} run {index + 1}/{args.runs} complete",
                    file=sys.stderr,
                    flush=True,
                )
            results.append(_scenario_stats(scenario, runs))
            print(f"[bench] scenario {scenario.name} complete", file=sys.stderr, flush=True)

        if not args.skip_multiprocess:
            current_phase = "multiprocess summary rollout"
            payload["multi_process_summary_rollout"] = _run_multi_process_summary_rollout(
                timeout_seconds=args.multiprocess_timeout_seconds,
            )
    except Exception as exc:
        payload["complete"] = False
        payload["error"] = {
            "phase": current_phase,
            "type": type(exc).__name__,
            "message": str(exc),
            "completed_scenarios": len(results),
            "requested_scenarios": len(scenarios),
        }
        print(
            f"[bench] failed during {current_phase}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        print(json.dumps(payload, indent=2))
        raise SystemExit(1)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
