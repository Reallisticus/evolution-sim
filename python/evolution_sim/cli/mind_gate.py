from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from collections.abc import Mapping, Sequence

from evolution_sim.config import WorldConfig
from evolution_sim.env import RunMode, SimulationWorld
from evolution_sim.io import JsonlTrajectoryWriter
from evolution_sim.mind.artifacts import write_model_artifact
from evolution_sim.mind.baseline import train_behavior_cloning_baseline
from evolution_sim.mind.dataset import combined_dataset_provenance, load_trajectory_jsonl
from evolution_sim.mind.evaluation import compare_heuristic_and_learned
from evolution_sim.mind.gates import normalize_mind_v1_gate_criteria
from evolution_sim.mind.learned_policy import load_learned_policy

DEFAULT_TRAIN_SEEDS: tuple[int, ...] = (3, 7, 11, 17)
DEFAULT_VALIDATION_SEEDS: tuple[int, ...] = (5, 13, 19, 29)
DEFAULT_TICKS = 120
DEFAULT_SPLIT_ID = "mind-v1-gate-train"
DEFAULT_TRAJECTORY_DIR = Path("output/trajectories/mind-v1-gate")
DEFAULT_ARTIFACT_OUTPUT = Path("output/mind/mind-v1-gate-artifact.json")
DEFAULT_REPORT_OUTPUT = Path("output/mind/mind-v1-gate-report.json")
DEFAULT_GATE_CRITERIA = normalize_mind_v1_gate_criteria(
    {"max_alive_agents_mean_regression": 0.5}
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a Mind v1 seed bank, train the guarded baseline, and "
            "evaluate it against the heuristic on held-out summary-only seeds."
        )
    )
    parser.add_argument(
        "--train-seeds",
        help="Comma-separated training seed list.",
    )
    parser.add_argument(
        "--train-seed",
        action="append",
        type=int,
        help="Add one training seed.",
    )
    parser.add_argument(
        "--validation-seeds",
        help="Comma-separated validation seed list.",
    )
    parser.add_argument(
        "--validation-seed",
        action="append",
        type=int,
        help="Add one validation seed.",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=DEFAULT_TICKS,
        help="Ticks for each training and validation run.",
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        default=DEFAULT_TRAJECTORY_DIR,
        help="Directory for generated training trajectories.",
    )
    parser.add_argument(
        "--artifact-output",
        type=Path,
        default=DEFAULT_ARTIFACT_OUTPUT,
        help="Path for the trained model artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPORT_OUTPUT,
        help="Path for the Mind gate JSON report.",
    )
    parser.add_argument(
        "--reuse-trajectories",
        action="store_true",
        help="Reuse existing trajectory files instead of regenerating them.",
    )
    parser.add_argument(
        "--max-alive-agents-mean-regression",
        type=float,
        default=DEFAULT_GATE_CRITERIA["max_alive_agents_mean_regression"],
        help="Allowed learned-policy terminal alive-agent mean regression.",
    )
    parser.add_argument(
        "--max-alive-agents-per-seed-regression",
        type=float,
        default=DEFAULT_GATE_CRITERIA["max_alive_agents_per_seed_regression"],
        help="Allowed learned-policy terminal alive-agent regression for one seed.",
    )
    parser.add_argument(
        "--max-births-mean-regression",
        type=float,
        default=DEFAULT_GATE_CRITERIA["max_births_mean_regression"],
        help="Allowed learned-policy births mean regression.",
    )
    parser.add_argument(
        "--max-births-per-seed-regression",
        type=float,
        default=DEFAULT_GATE_CRITERIA["max_births_per_seed_regression"],
        help="Allowed learned-policy births regression for one seed.",
    )
    parser.add_argument(
        "--max-invalid-action-rate",
        type=float,
        default=DEFAULT_GATE_CRITERIA["max_invalid_action_rate"],
        help="Maximum policy-visible invalid action rate.",
    )
    parser.add_argument(
        "--min-viable-run-share",
        type=float,
        default=DEFAULT_GATE_CRITERIA["min_viable_run_share"],
        help="Minimum share of validation runs ending with live agents.",
    )
    parser.add_argument(
        "--min-births-per-run-mean",
        type=float,
        default=DEFAULT_GATE_CRITERIA["min_births_per_run_mean"],
        help="Review floor for mean births per validation run.",
    )
    parser.add_argument(
        "--min-plant-energy-available-per-land-tile",
        type=float,
        default=DEFAULT_GATE_CRITERIA["min_plant_energy_available_per_land_tile"],
        help="Review floor for terminal plant energy available per land tile.",
    )
    parser.add_argument(
        "--fail-on-blockers",
        action="store_true",
        help="Exit non-zero when the Mind gate status is fail.",
    )
    parser.add_argument(
        "--fail-on-review",
        action="store_true",
        help="Exit non-zero when the Mind gate status is fail or review.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_seeds = _parse_seed_selection(
        args.train_seed,
        args.train_seeds,
        default=DEFAULT_TRAIN_SEEDS,
    )
    validation_seeds = _parse_seed_selection(
        args.validation_seed,
        args.validation_seeds,
        default=DEFAULT_VALIDATION_SEEDS,
    )
    if args.ticks <= 0:
        raise SystemExit("--ticks must be positive")
    gate_criteria = _gate_criteria_from_args(args)

    report = run_mind_gate(
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        ticks=args.ticks,
        trajectory_dir=args.trajectory_dir,
        artifact_output=args.artifact_output,
        report_output=args.output,
        reuse_trajectories=args.reuse_trajectories,
        gate_criteria=gate_criteria,
    )
    print(json.dumps(report, indent=2))
    status = report["readiness"]["status"]
    if args.fail_on_review and status in {"fail", "review"}:
        raise SystemExit(1)
    if args.fail_on_blockers and status == "fail":
        raise SystemExit(1)


def run_mind_gate(
    *,
    train_seeds: Sequence[int],
    validation_seeds: Sequence[int],
    ticks: int,
    trajectory_dir: Path,
    artifact_output: Path,
    report_output: Path | None,
    reuse_trajectories: bool = False,
    gate_criteria: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if not train_seeds:
        raise ValueError("at least one training seed is required")
    if not validation_seeds:
        raise ValueError("at least one validation seed is required")
    resolved_gate_criteria = normalize_mind_v1_gate_criteria(
        DEFAULT_GATE_CRITERIA if gate_criteria is None else gate_criteria
    )

    started = time.perf_counter()
    split_id = DEFAULT_SPLIT_ID
    trajectory_records: list[dict[str, object]] = []
    trajectory_paths: list[Path] = []
    for seed in train_seeds:
        trajectory_path = trajectory_dir / f"seed{seed}-ticks{ticks}.jsonl.gz"
        trajectory_paths.append(trajectory_path)
        if reuse_trajectories and trajectory_path.exists():
            _log(f"reuse trajectory seed={seed} path={trajectory_path}")
            dataset = load_trajectory_jsonl(trajectory_path)
            trajectory_records.append(
                _trajectory_record_from_dataset(seed=seed, dataset=dataset)
            )
            continue
        _log(f"collect trajectory seed={seed} ticks={ticks} path={trajectory_path}")
        trajectory_records.append(
            collect_training_trajectory(
                seed=seed,
                ticks=ticks,
                split_id=split_id,
                output_path=trajectory_path,
            )
        )

    _log(f"train artifact path={artifact_output}")
    datasets = [load_trajectory_jsonl(path) for path in trajectory_paths]
    records = (record for dataset in datasets for record in dataset.records)
    baseline = train_behavior_cloning_baseline(
        records,
        provenance=combined_dataset_provenance(datasets),
    )
    artifact = baseline.to_artifact()
    write_model_artifact(artifact_output, artifact)

    _log(f"evaluate validation_seeds={list(validation_seeds)} ticks={ticks}")
    policy = load_learned_policy(artifact_output, enable_mind=True)
    evaluation = compare_heuristic_and_learned(
        learned_policy=policy,
        seeds=validation_seeds,
        ticks=ticks,
        gate_criteria=resolved_gate_criteria,
    )
    readiness = evaluation["mind_v1_gates"]
    report = {
        "protocol": {
            "profile": "mind_v1_seedbank_gate",
            "train_seeds": list(train_seeds),
            "validation_seeds": list(validation_seeds),
            "ticks": ticks,
            "mode": RunMode.SUMMARY_ONLY.value,
            "trajectory_split_id": split_id,
            "trajectory_dir": str(trajectory_dir),
            "artifact_output": str(artifact_output),
            "report_output": str(report_output) if report_output else None,
            "reuse_trajectories": reuse_trajectories,
            "criteria": resolved_gate_criteria,
        },
        "complete": True,
        "trajectory_collection": trajectory_records,
        "artifact": {
            "path": str(artifact_output),
            "model_type": artifact["manifest"]["model_type"],
            "trained_record_count": artifact["manifest"]["trained_record_count"],
            "provenance": artifact["manifest"]["provenance"],
            "conditional_feature_count": len(
                artifact["model"].get("conditional_action_scores", {})
            ),
            "heuristic_guard_policy": artifact["model"].get("heuristic_guard_policy"),
        },
        "evaluation": evaluation,
        "readiness": readiness,
        "timings": {
            "total_wall_seconds": round(time.perf_counter() - started, 4),
        },
    }
    if report_output is not None:
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def collect_training_trajectory(
    *,
    seed: int,
    ticks: int,
    split_id: str,
    output_path: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    writer = JsonlTrajectoryWriter(
        output_path,
        source_seeds=[seed],
        split_id=split_id,
    )
    result = SimulationWorld(WorldConfig(seed=seed, max_ticks=ticks)).run(
        mode=RunMode.SUMMARY_ONLY,
        trajectory_sink=writer,
    )
    summary = result.summary
    return {
        "seed": seed,
        "path": str(output_path),
        "run_id": summary["run_id"],
        "ticks_executed": summary["ticks_executed"],
        "alive_agents": summary["alive_agents"],
        "births": summary["births"],
        "deaths": summary["deaths"],
        "trajectory_records": writer.record_count,
        "mean_reward": writer.trajectory_summary["mean_reward"],
        "wall_seconds": round(time.perf_counter() - started, 4),
    }


def _trajectory_record_from_dataset(
    *,
    seed: int,
    dataset: object,
) -> dict[str, object]:
    record_count = int(getattr(dataset, "record_count"))
    footer = getattr(dataset, "footer")
    if not isinstance(footer, dict):
        raise ValueError("trajectory dataset footer must be an object")
    summary = footer.get("summary")
    trajectory_summary = footer.get("trajectory_summary")
    if not isinstance(summary, dict) or not isinstance(trajectory_summary, dict):
        raise ValueError("trajectory dataset footer is missing summaries")
    return {
        "seed": seed,
        "path": str(getattr(dataset, "path")),
        "run_id": summary["run_id"],
        "ticks_executed": summary["ticks_executed"],
        "alive_agents": summary["alive_agents"],
        "births": summary["births"],
        "deaths": summary["deaths"],
        "trajectory_records": record_count,
        "mean_reward": trajectory_summary["mean_reward"],
        "wall_seconds": 0.0,
        "reused": True,
    }


def _parse_seed_selection(
    seed_args: Sequence[int] | None,
    seeds_arg: str | None,
    *,
    default: Sequence[int],
) -> list[int]:
    raw_seeds: list[int] = []
    if seeds_arg:
        for chunk in seeds_arg.split(","):
            value = chunk.strip()
            if value:
                raw_seeds.append(int(value))
    if seed_args:
        raw_seeds.extend(seed_args)
    if not raw_seeds:
        raw_seeds.extend(default)

    seen: set[int] = set()
    seeds: list[int] = []
    for seed in raw_seeds:
        if seed in seen:
            continue
        seen.add(seed)
        seeds.append(seed)
    return seeds


def _gate_criteria_from_args(args: argparse.Namespace) -> dict[str, float]:
    criteria = normalize_mind_v1_gate_criteria(
        {
            "max_alive_agents_mean_regression": args.max_alive_agents_mean_regression,
            "max_alive_agents_per_seed_regression": (
                args.max_alive_agents_per_seed_regression
            ),
            "max_births_mean_regression": args.max_births_mean_regression,
            "max_births_per_seed_regression": args.max_births_per_seed_regression,
            "max_invalid_action_rate": args.max_invalid_action_rate,
            "min_viable_run_share": args.min_viable_run_share,
            "min_births_per_run_mean": args.min_births_per_run_mean,
            "min_plant_energy_available_per_land_tile": (
                args.min_plant_energy_available_per_land_tile
            ),
        }
    )
    negative_criteria = [
        key for key, value in criteria.items() if value < 0.0
    ]
    if negative_criteria:
        raise SystemExit(
            "Mind v1 gate criteria must be non-negative: "
            + ", ".join(sorted(negative_criteria))
        )
    return criteria


def _log(message: str) -> None:
    print(f"[mind-gate] {message}", file=sys.stderr)


if __name__ == "__main__":
    main()
