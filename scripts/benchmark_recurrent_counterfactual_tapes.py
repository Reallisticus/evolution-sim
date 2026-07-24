#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
for import_root in (PYTHON_ROOT, REPO_ROOT):
    import_path = str(import_root)
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import torch

from evolution_sim.mind.provenance import stable_payload_digest
from evolution_sim.mind.recurrent_actor_critic import (
    PublicRecurrentActorCritic,
    RecurrentActorCriticConfig,
)
from evolution_sim.mind.recurrent_counterfactual_collection import (
    RecurrentCounterfactualCollectionConfig,
    RecurrentCounterfactualCollectionTask,
    collect_recurrent_counterfactual_bundles,
)
from evolution_sim.mind.recurrent_seed_registry import (
    SCALE_DEVELOPMENT_V2_SEED_REGISTRY,
)


BENCHMARK_SCHEMA_VERSION = "recurrent_counterfactual_tape_benchmark_v1"


def _integer_sequence(value: str, *, field: str, minimum: int = 1) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{field} must contain integers") from error
    if not parsed or any(item < minimum for item in parsed):
        raise argparse.ArgumentTypeError(
            f"{field} must contain integers greater than or equal to {minimum}"
        )
    return parsed


def _git_output(*args: str) -> str | None:
    result = subprocess.run(
        ("git", *args),
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _tasks(
    *,
    task_count: int,
    branch_ticks: tuple[int, ...],
) -> tuple[RecurrentCounterfactualCollectionTask, ...]:
    seeds = SCALE_DEVELOPMENT_V2_SEED_REGISTRY["scale_v2_curriculum"]
    if task_count > len(seeds):
        raise SystemExit(
            f"task count {task_count} exceeds the registered curriculum seed count"
        )
    return tuple(
        RecurrentCounterfactualCollectionTask(
            task_id=f"counterfactual-tape-benchmark-{index}",
            seed_role="scale_v2_curriculum",
            scenario="carrion_only",
            environment_seed=seeds[index],
            branch_tick_candidates=branch_ticks,
            source_policy_sampling_identity=(
                f"counterfactual-tape-benchmark-{index}:source"
            ),
            branch_selection_identity=(
                f"counterfactual-tape-benchmark-{index}:selection"
            ),
            branch_tick_stratum_index=index % len(branch_ticks),
        )
        for index in range(task_count)
    )


def _bundle_scientific_digest(result: object) -> str:
    bundles = getattr(result, "bundles")
    return stable_payload_digest(
        {
            "bundle_exact_digests": [bundle.exact_digest for bundle in bundles],
            "aggregate_compute": getattr(result, "aggregate_compute"),
        }
    )


def _run_once(
    model: PublicRecurrentActorCritic,
    tasks: Sequence[RecurrentCounterfactualCollectionTask],
    *,
    config: RecurrentCounterfactualCollectionConfig,
    workers: int,
) -> dict[str, object]:
    started = time.perf_counter()
    result = collect_recurrent_counterfactual_bundles(
        model,
        tasks,
        artifact_digest="b" * 64,
        config=config,
        workers=workers,
    )
    elapsed = time.perf_counter() - started
    return {
        "workers_requested": workers,
        "workers_resolved": result.workers_resolved,
        "elapsed_seconds": round(elapsed, 6),
        "execution_contract_version": result.execution_contract_version,
        "execution_compute": result.execution_compute,
        "aggregate_compute": result.aggregate_compute,
        "scientific_bundle_digest": _bundle_scientific_digest(result),
        "result_exact_digest": result.exact_digest,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark sequential versus tape-parallel exact counterfactual "
            "collection while proving identical scientific bundle evidence."
        )
    )
    parser.add_argument("--workers", default="1,8")
    parser.add_argument("--tasks", type=int, default=2)
    parser.add_argument("--tapes", type=int, default=8)
    parser.add_argument("--horizons", default="16,48")
    parser.add_argument("--branch-ticks", default="16,40,64,72")
    parser.add_argument("--terminal-target-tick", type=int, default=120)
    parser.add_argument("--encoder-size", type=int, default=192)
    parser.add_argument("--hidden-size", type=int, default=192)
    parser.add_argument("--model-seed", type=int, default=83)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    workers = _integer_sequence(args.workers, field="workers")
    horizons = _integer_sequence(args.horizons, field="horizons")
    branch_ticks = _integer_sequence(
        args.branch_ticks,
        field="branch-ticks",
        minimum=0,
    )
    for field, value in (
        ("tasks", args.tasks),
        ("tapes", args.tapes),
        ("terminal-target-tick", args.terminal_target_tick),
        ("encoder-size", args.encoder_size),
        ("hidden-size", args.hidden_size),
    ):
        if value <= 0:
            raise SystemExit(f"{field} must be positive")
    if args.model_seed < 0:
        raise SystemExit("model-seed must be nonnegative")
    if max(branch_ticks) + max(horizons) > args.terminal_target_tick:
        raise SystemExit(
            "largest branch tick plus largest relative horizon must not exceed "
            "terminal-target-tick"
        )

    torch.set_num_threads(1)
    model = PublicRecurrentActorCritic(
        RecurrentActorCriticConfig(
            encoder_size=args.encoder_size,
            hidden_size=args.hidden_size,
            recurrent_layers=1,
        ),
        initialization_seed=args.model_seed,
    )
    tasks = _tasks(task_count=args.tasks, branch_ticks=branch_ticks)
    config = RecurrentCounterfactualCollectionConfig(
        horizons=horizons,
        gamma=0.99,
        continuation_tape_count=args.tapes,
        terminal_target_world_tick=args.terminal_target_tick,
        uncertainty_penalty=0.25,
    )
    runs = [
        _run_once(model, tasks, config=config, workers=worker_count)
        for worker_count in workers
    ]
    scientific_digests = {
        str(run["scientific_bundle_digest"]) for run in runs
    }
    baseline_seconds = float(runs[0]["elapsed_seconds"])
    for run in runs:
        elapsed = float(run["elapsed_seconds"])
        run["speedup_vs_first_run"] = round(baseline_seconds / elapsed, 6)
    payload = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "source": {
            "commit": _git_output("rev-parse", "HEAD"),
            "branch": _git_output("branch", "--show-current"),
            "status_porcelain": _git_output("status", "--short") or "",
        },
        "runtime": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "logical_cpu_count": os.cpu_count(),
        },
        "configuration": {
            "workers": list(workers),
            "task_count": args.tasks,
            "continuation_tape_count": args.tapes,
            "horizons": list(horizons),
            "branch_ticks": list(branch_ticks),
            "terminal_target_tick": args.terminal_target_tick,
            "encoder_size": args.encoder_size,
            "hidden_size": args.hidden_size,
            "model_seed": args.model_seed,
        },
        "scientific_bundle_digests_match": len(scientific_digests) == 1,
        "runs": runs,
    }
    if not payload["scientific_bundle_digests_match"]:
        raise SystemExit("worker configurations produced different scientific evidence")
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
