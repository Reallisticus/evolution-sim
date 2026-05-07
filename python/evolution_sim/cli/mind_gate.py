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
from evolution_sim.mind.baseline import (
    CONTEXTUAL_PRIOR_TRAINER,
    TRAINER_CHOICES,
    train_baseline_with_trainer,
)
from evolution_sim.mind.dataset import (
    combined_dataset_provenance,
    load_trajectory_jsonl,
    records_with_trajectory_context,
)
from evolution_sim.mind.diagnostics import build_artifact_diagnostics
from evolution_sim.mind.evaluation import compare_heuristic_and_learned
from evolution_sim.mind.gates import normalize_mind_v1_gate_criteria
from evolution_sim.mind.learned_policy import load_learned_policy

DEFAULT_TRAIN_SEEDS: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    17,
    23,
    31,
)
DEFAULT_VALIDATION_SEEDS: tuple[int, ...] = (5, 13, 19, 29)
DEFAULT_TICKS = 120
DEFAULT_SPLIT_ID = "mind-v1-gate-train"
DEFAULT_ARTIFACT_DIAGNOSTIC_SPLIT_ID = "mind-v1-gate-artifact-diagnostic"
DEFAULT_TRAJECTORY_DIR = Path("output/trajectories/mind-v1-gate")
DEFAULT_ARTIFACT_OUTPUT = Path("output/mind/mind-v1-gate-artifact.json")
DEFAULT_REPORT_OUTPUT = Path("output/mind/mind-v1-gate-report.json")
DEFAULT_EXPERIMENT_LEDGER_OUTPUT = Path("output/mind/mind-experiment-ledger.jsonl")
STRICT_CONTROL_HARD_GUARD_RATE = 0.1190
STRICT_CONTROL_TOTAL_FALLBACK_RATE = 0.4780
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
        help="Ticks for each training run and the default validation horizon.",
    )
    parser.add_argument(
        "--validation-ticks",
        help="Comma-separated validation tick horizons.",
    )
    parser.add_argument(
        "--validation-tick",
        action="append",
        type=int,
        help="Add one validation tick horizon.",
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
        "--experiment-ledger-output",
        type=Path,
        default=DEFAULT_EXPERIMENT_LEDGER_OUTPUT,
        help="Append a compact Mind experiment ledger entry to this JSONL file.",
    )
    parser.add_argument(
        "--reuse-trajectories",
        action="store_true",
        help="Reuse existing trajectory files instead of regenerating them.",
    )
    parser.add_argument(
        "--trainer",
        choices=TRAINER_CHOICES,
        default=CONTEXTUAL_PRIOR_TRAINER,
        help="Offline baseline trainer to use.",
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
        "--max-guard-intervention-rate",
        type=float,
        default=DEFAULT_GATE_CRITERIA["max_guard_intervention_rate"],
        help="Maximum aggregate learned-policy guard intervention rate.",
    )
    parser.add_argument(
        "--max-total-heuristic-fallback-rate",
        type=float,
        default=DEFAULT_GATE_CRITERIA["max_total_heuristic_fallback_rate"],
        help=(
            "Maximum aggregate guard plus confidence-delegation fallback rate."
        ),
    )
    parser.add_argument(
        "--max-guard-intervention-rate-by-group",
        type=float,
        default=DEFAULT_GATE_CRITERIA["max_guard_intervention_rate_by_group"],
        help="Maximum guard intervention rate for each role and meat-mode group.",
    )
    parser.add_argument(
        "--min-guard-intervention-rate-reduction",
        type=float,
        default=DEFAULT_GATE_CRITERIA["min_guard_intervention_rate_reduction"],
        help="Required guard intervention rate reduction versus a prior artifact.",
    )
    parser.add_argument(
        "--reference-guard-intervention-rate",
        type=float,
        help="Prior artifact guard intervention rate used for reduction checks.",
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
    validation_ticks = _parse_tick_selection(
        args.validation_tick,
        args.validation_ticks,
        default=(args.ticks,),
    )
    if any(tick <= 0 for tick in validation_ticks):
        raise SystemExit("validation ticks must be positive")
    gate_criteria = _gate_criteria_from_args(args)

    report = run_mind_gate(
        train_seeds=train_seeds,
        validation_seeds=validation_seeds,
        ticks=args.ticks,
        validation_ticks=validation_ticks,
        trajectory_dir=args.trajectory_dir,
        artifact_output=args.artifact_output,
        report_output=args.output,
        experiment_ledger_output=args.experiment_ledger_output,
        reuse_trajectories=args.reuse_trajectories,
        gate_criteria=gate_criteria,
        reference_guard_intervention_rate=args.reference_guard_intervention_rate,
        trainer=args.trainer,
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
    experiment_ledger_output: Path | None = DEFAULT_EXPERIMENT_LEDGER_OUTPUT,
    validation_ticks: Sequence[int] | None = None,
    reuse_trajectories: bool = False,
    gate_criteria: Mapping[str, object] | None = None,
    reference_guard_intervention_rate: float | None = None,
    trainer: str = CONTEXTUAL_PRIOR_TRAINER,
) -> dict[str, object]:
    if not train_seeds:
        raise ValueError("at least one training seed is required")
    if not validation_seeds:
        raise ValueError("at least one validation seed is required")
    resolved_validation_ticks = list(validation_ticks or (ticks,))
    if not resolved_validation_ticks:
        raise ValueError("at least one validation tick horizon is required")
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
            dataset = _load_expected_trajectory_dataset(
                trajectory_path,
                seed=seed,
                ticks=ticks,
                split_id=split_id,
            )
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
    datasets = [
        _load_expected_trajectory_dataset(
            path,
            seed=seed,
            ticks=ticks,
            split_id=split_id,
        )
        for seed, path in zip(train_seeds, trajectory_paths, strict=True)
    ]
    records = records_with_trajectory_context(datasets)
    baseline = train_baseline_with_trainer(
        records,
        provenance=combined_dataset_provenance(datasets),
        trainer=trainer,
    )
    artifact = baseline.to_artifact()

    diagnostic_split_id = DEFAULT_ARTIFACT_DIAGNOSTIC_SPLIT_ID
    diagnostic_trajectory_dir = trajectory_dir / "artifact-diagnostics"
    diagnostic_records: list[dict[str, object]] = []
    diagnostic_paths: list[Path] = []
    for seed in validation_seeds:
        diagnostic_path = diagnostic_trajectory_dir / f"seed{seed}-ticks{ticks}.jsonl.gz"
        diagnostic_paths.append(diagnostic_path)
        if reuse_trajectories and diagnostic_path.exists():
            _log(f"reuse artifact diagnostic trajectory seed={seed} path={diagnostic_path}")
            dataset = _load_expected_trajectory_dataset(
                diagnostic_path,
                seed=seed,
                ticks=ticks,
                split_id=diagnostic_split_id,
            )
            diagnostic_records.append(
                _trajectory_record_from_dataset(seed=seed, dataset=dataset)
            )
            continue
        _log(
            "collect artifact diagnostic trajectory "
            f"seed={seed} ticks={ticks} path={diagnostic_path}"
        )
        diagnostic_records.append(
            collect_training_trajectory(
                seed=seed,
                ticks=ticks,
                split_id=diagnostic_split_id,
                output_path=diagnostic_path,
            )
        )
    diagnostic_datasets = [
        _load_expected_trajectory_dataset(
            path,
            seed=seed,
            ticks=ticks,
            split_id=diagnostic_split_id,
        )
        for seed, path in zip(validation_seeds, diagnostic_paths, strict=True)
    ]

    artifact_diagnostics = {
        "train": build_artifact_diagnostics(artifact, datasets),
        "held_out": build_artifact_diagnostics(artifact, diagnostic_datasets),
    }
    write_model_artifact(artifact_output, artifact)

    _log(
        "evaluate "
        f"validation_seeds={list(validation_seeds)} "
        f"validation_ticks={resolved_validation_ticks}"
    )
    policy = load_learned_policy(artifact_output, enable_mind=True)
    evaluation_matrix = [
        {
            "ticks": validation_tick,
            "evaluation": compare_heuristic_and_learned(
                learned_policy=policy,
                seeds=validation_seeds,
                ticks=validation_tick,
                gate_criteria=resolved_gate_criteria,
                reference_guard_intervention_rate=reference_guard_intervention_rate,
            ),
        }
        for validation_tick in resolved_validation_ticks
    ]
    evaluation = evaluation_matrix[0]["evaluation"]
    readiness = _combine_readiness(evaluation_matrix)
    report = {
        "protocol": {
            "profile": "mind_v1_seedbank_gate",
            "train_seeds": list(train_seeds),
            "validation_seeds": list(validation_seeds),
            "ticks": ticks,
            "validation_ticks": resolved_validation_ticks,
            "mode": RunMode.SUMMARY_ONLY.value,
            "trajectory_split_id": split_id,
            "artifact_diagnostic_split_id": diagnostic_split_id,
            "trajectory_dir": str(trajectory_dir),
            "artifact_output": str(artifact_output),
            "report_output": str(report_output) if report_output else None,
            "experiment_ledger_output": (
                str(experiment_ledger_output) if experiment_ledger_output else None
            ),
            "reuse_trajectories": reuse_trajectories,
            "trainer": trainer,
            "criteria": resolved_gate_criteria,
            "reference_guard_intervention_rate": reference_guard_intervention_rate,
        },
        "complete": True,
        "trajectory_collection": trajectory_records,
        "artifact_diagnostic_collection": diagnostic_records,
        "artifact": {
            "path": str(artifact_output),
            "model_type": artifact["manifest"]["model_type"],
            "trained_record_count": artifact["manifest"]["trained_record_count"],
            "provenance": artifact["manifest"]["provenance"],
            "conditional_feature_count": len(
                artifact["model"].get("conditional_action_scores", {})
            ),
            "conditional_min_records": artifact["model"].get(
                "conditional_min_records"
            ),
            "conditional_score_policy": artifact["model"].get(
                "conditional_score_policy"
            ),
            "conditional_prior_correction_exponent": artifact["model"].get(
                "conditional_prior_correction_exponent"
            ),
            "conditional_score_smoothing_alpha": artifact["model"].get(
                "conditional_score_smoothing_alpha"
            ),
            "trainer": artifact["model"].get("trainer"),
            "sample_weight_policy": artifact["model"].get("sample_weight_policy"),
            "sample_weight_base": artifact["model"].get("sample_weight_base"),
            "sample_weight_min": artifact["model"].get("sample_weight_min"),
            "sample_weight_max": artifact["model"].get("sample_weight_max"),
            "sample_weight_total": artifact["model"].get("sample_weight_total"),
            "reward_advantage_policy": artifact["model"].get(
                "reward_advantage_policy"
            ),
            "reward_advantage_scale": artifact["model"].get(
                "reward_advantage_scale"
            ),
            "reward_advantage_min_action_support": artifact["model"].get(
                "reward_advantage_min_action_support"
            ),
            "reward_advantage_blend_policy": artifact["model"].get(
                "reward_advantage_blend_policy"
            ),
            "reward_advantage_blend_weight": artifact["model"].get(
                "reward_advantage_blend_weight"
            ),
            "value_estimation_policy": artifact["model"].get(
                "value_estimation_policy"
            ),
            "value_score_blend_policy": artifact["model"].get(
                "value_score_blend_policy"
            ),
            "value_score_blend_weight": artifact["model"].get(
                "value_score_blend_weight"
            ),
            "value_min_action_support": artifact["model"].get(
                "value_min_action_support"
            ),
            "value_score_epsilon": artifact["model"].get(
                "value_score_epsilon"
            ),
            "value_supported_deviation_policy": artifact["model"].get(
                "value_supported_deviation_policy"
            ),
            "value_supported_deviation_min_support": artifact["model"].get(
                "value_supported_deviation_min_support"
            ),
            "value_supported_deviation_min_value_margin": artifact["model"].get(
                "value_supported_deviation_min_value_margin"
            ),
            "value_supported_deviation_min_learned_value": artifact["model"].get(
                "value_supported_deviation_min_learned_value"
            ),
            "value_supported_deviation_min_score_margin": artifact["model"].get(
                "value_supported_deviation_min_score_margin"
            ),
            "value_supported_deviation_min_predicted_advantage": artifact[
                "model"
            ].get("value_supported_deviation_min_predicted_advantage"),
            "neural_backend": artifact["model"].get("neural_backend"),
            "neural_architecture": artifact["model"].get("neural_architecture"),
            "neural_training_policy": artifact["model"].get(
                "neural_training_policy"
            ),
            "neural_input_size": artifact["model"].get("neural_input_size"),
            "neural_hidden_units": artifact["model"].get("neural_hidden_units"),
            "neural_hidden_activation": artifact["model"].get(
                "neural_hidden_activation"
            ),
            "neural_input_normalization": artifact["model"].get(
                "neural_input_normalization"
            ),
            "neural_state_value_policy": artifact["model"].get(
                "neural_state_value_policy"
            ),
            "neural_actor_prior_policy": artifact["model"].get(
                "neural_actor_prior_policy"
            ),
            "neural_actor_prior_blend_weight": artifact["model"].get(
                "neural_actor_prior_blend_weight"
            ),
            "heuristic_guard_policy": artifact["model"].get("heuristic_guard_policy"),
            "heuristic_confidence_threshold": artifact["model"].get(
                "heuristic_confidence_threshold"
            ),
            "heuristic_override_min_margin": artifact["model"].get(
                "heuristic_override_min_margin"
            ),
            "heuristic_delegate_policy": artifact["model"].get(
                "heuristic_delegate_policy"
            ),
            "heuristic_delegate_max_training_score_margin": artifact["model"].get(
                "heuristic_delegate_max_training_score_margin"
            ),
            "heuristic_safe_local_eat_min_score": artifact["model"].get(
                "heuristic_safe_local_eat_min_score"
            ),
            "heuristic_safe_local_eat_min_food": artifact["model"].get(
                "heuristic_safe_local_eat_min_food"
            ),
            "heuristic_safe_local_eat_min_plant_ratio": artifact["model"].get(
                "heuristic_safe_local_eat_min_plant_ratio"
            ),
            "heuristic_safe_plant_move_min_score": artifact["model"].get(
                "heuristic_safe_plant_move_min_score"
            ),
            "heuristic_safe_plant_move_min_strength": artifact["model"].get(
                "heuristic_safe_plant_move_min_strength"
            ),
            "heuristic_safe_plant_move_max_local_food_ratio": artifact[
                "model"
            ].get("heuristic_safe_plant_move_max_local_food_ratio"),
            "heuristic_safe_plant_move_max_distance": artifact["model"].get(
                "heuristic_safe_plant_move_max_distance"
            ),
        },
        "artifact_diagnostics": artifact_diagnostics,
        "evaluation": evaluation,
        "evaluation_matrix": evaluation_matrix,
        "readiness": readiness,
        "timings": {
            "total_wall_seconds": round(time.perf_counter() - started, 4),
        },
    }
    ledger_entry = _build_experiment_ledger_entry(report)
    report["experiment_ledger_entry"] = ledger_entry
    if experiment_ledger_output is not None:
        _append_experiment_ledger_entry(experiment_ledger_output, ledger_entry)
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


def _load_expected_trajectory_dataset(
    path: Path,
    *,
    seed: int,
    ticks: int,
    split_id: str,
):
    dataset = load_trajectory_jsonl(path)
    _validate_reused_trajectory_dataset(
        dataset,
        seed=seed,
        ticks=ticks,
        split_id=split_id,
    )
    return dataset


def _validate_reused_trajectory_dataset(
    dataset: object,
    *,
    seed: int,
    ticks: int,
    split_id: str,
) -> None:
    footer = getattr(dataset, "footer")
    if not isinstance(footer, dict):
        raise ValueError("trajectory dataset footer must be an object")
    summary = footer.get("summary")
    provenance = footer.get("provenance")
    if not isinstance(summary, dict) or not isinstance(provenance, dict):
        raise ValueError("trajectory dataset footer is missing summary or provenance")
    expected_run_id = f"seed-{seed}-ticks-{ticks}"
    if summary.get("run_id") != expected_run_id:
        raise ValueError(
            (
                f"reused trajectory for seed {seed} ticks {ticks} has run_id "
                f"{summary.get('run_id')!r}; expected {expected_run_id!r}"
            )
        )
    source_seeds = provenance.get("source_seeds")
    if source_seeds != [seed]:
        raise ValueError(
            (
                f"reused trajectory for seed {seed} has provenance source_seeds "
                f"{source_seeds!r}; expected [{seed}]"
            )
        )
    if provenance.get("split_id") != split_id:
        raise ValueError(
            (
                f"reused trajectory for seed {seed} has split_id "
                f"{provenance.get('split_id')!r}; expected {split_id!r}"
            )
        )


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


def _parse_tick_selection(
    tick_args: Sequence[int] | None,
    ticks_arg: str | None,
    *,
    default: Sequence[int],
) -> list[int]:
    raw_ticks: list[int] = []
    if ticks_arg:
        for chunk in ticks_arg.split(","):
            value = chunk.strip()
            if value:
                raw_ticks.append(int(value))
    if tick_args:
        raw_ticks.extend(tick_args)
    if not raw_ticks:
        raw_ticks.extend(default)

    seen: set[int] = set()
    ticks: list[int] = []
    for tick in raw_ticks:
        if tick in seen:
            continue
        seen.add(tick)
        ticks.append(tick)
    return ticks


def _combine_readiness(
    evaluation_matrix: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    blockers: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    for entry in evaluation_matrix:
        ticks = entry.get("ticks")
        evaluation = entry.get("evaluation")
        if not isinstance(evaluation, Mapping):
            continue
        readiness = evaluation.get("mind_v1_gates")
        if not isinstance(readiness, Mapping):
            continue
        blockers.extend(
            _flag_with_ticks(flag, ticks=ticks)
            for flag in _flag_list(readiness.get("blockers"))
        )
        warnings.extend(
            _flag_with_ticks(flag, ticks=ticks)
            for flag in _flag_list(readiness.get("warnings"))
        )
    return {
        "status": "fail" if blockers else ("review" if warnings else "pass"),
        "blockers": blockers,
        "warnings": warnings,
    }


def _flag_list(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, list):
        return []
    return [dict(flag) for flag in payload if isinstance(flag, Mapping)]


def _flag_with_ticks(
    flag: Mapping[str, object],
    *,
    ticks: object,
) -> dict[str, object]:
    payload = dict(flag)
    payload["validation_ticks"] = ticks
    return payload


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
            "max_guard_intervention_rate": args.max_guard_intervention_rate,
            "max_total_heuristic_fallback_rate": (
                args.max_total_heuristic_fallback_rate
            ),
            "max_guard_intervention_rate_by_group": (
                args.max_guard_intervention_rate_by_group
            ),
            "min_guard_intervention_rate_reduction": (
                args.min_guard_intervention_rate_reduction
            ),
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


def _build_experiment_ledger_entry(
    report: Mapping[str, object],
) -> dict[str, object]:
    protocol = _mapping(report.get("protocol"))
    artifact = _mapping(report.get("artifact"))
    readiness = _mapping(report.get("readiness"))
    evaluation = _mapping(report.get("evaluation"))
    learned = _mapping(evaluation.get("learned"))
    learned_aggregate = _mapping(learned.get("aggregate"))
    policy_diagnostics = _mapping(
        learned_aggregate.get("policy_diagnostics")
    )
    comparison = _mapping(evaluation.get("comparison"))
    guard_rate = _float_or_none(
        policy_diagnostics.get("guard_intervention_rate")
    )
    delegate_rate = _float_or_none(
        policy_diagnostics.get("heuristic_delegate_rate")
    )
    total_fallback = (
        round(guard_rate + delegate_rate, 4)
        if guard_rate is not None and delegate_rate is not None
        else None
    )
    min_alive_delta = _per_seed_min_delta(
        comparison,
        field="alive_agents_delta",
        fallback=comparison.get("alive_agents_mean_delta"),
    )
    min_births_delta = _per_seed_min_delta(
        comparison,
        field="births_delta",
        fallback=comparison.get("births_mean_delta"),
    )
    strict_control_target_passed = _strict_control_target_passed(
        status=readiness.get("status"),
        guard_rate=guard_rate,
        total_fallback=total_fallback,
        min_alive_delta=min_alive_delta,
        min_births_delta=min_births_delta,
    )
    return {
        "schema_version": "mind_experiment_ledger_v1",
        "trainer": artifact.get("trainer"),
        "model_type": artifact.get("model_type"),
        "artifact_path": artifact.get("path"),
        "report_path": protocol.get("report_output"),
        "train_seeds": protocol.get("train_seeds"),
        "validation_seeds": protocol.get("validation_seeds"),
        "ticks": protocol.get("ticks"),
        "validation_ticks": protocol.get("validation_ticks"),
        "status": readiness.get("status"),
        "decision": _ledger_decision(
            readiness.get("status"),
            strict_control_target_passed=strict_control_target_passed,
        ),
        "strict_control_target_passed": strict_control_target_passed,
        "hard_guard": guard_rate,
        "heuristic_delegate": delegate_rate,
        "total_fallback": total_fallback,
        "safe_deviation": _float_or_none(
            policy_diagnostics.get("safe_deviation_rate")
        ),
        "alive_delta": _float_or_none(
            comparison.get("alive_agents_mean_delta")
        ),
        "births_delta": _float_or_none(comparison.get("births_mean_delta")),
        "min_alive_delta": min_alive_delta,
        "min_births_delta": min_births_delta,
    }


def _append_experiment_ledger_entry(
    path: Path,
    entry: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(entry), sort_keys=True) + "\n")


def _ledger_decision(
    status: object,
    *,
    strict_control_target_passed: bool,
) -> str:
    if status == "pass":
        return (
            "strict_control_candidate"
            if strict_control_target_passed
            else "gate_pass_not_promoted"
        )
    if status == "review":
        return "review_probe"
    return "reject_probe"


def _strict_control_target_passed(
    *,
    status: object,
    guard_rate: float | None,
    total_fallback: float | None,
    min_alive_delta: float | None,
    min_births_delta: float | None,
) -> bool:
    return (
        status == "pass"
        and guard_rate is not None
        and total_fallback is not None
        and min_alive_delta is not None
        and min_births_delta is not None
        and guard_rate < STRICT_CONTROL_HARD_GUARD_RATE
        and total_fallback < STRICT_CONTROL_TOTAL_FALLBACK_RATE
        and min_alive_delta >= 0.0
        and min_births_delta >= 0.0
    )


def _per_seed_min_delta(
    comparison: Mapping[str, object],
    *,
    field: str,
    fallback: object,
) -> float | None:
    per_seed = comparison.get("per_seed")
    if isinstance(per_seed, list):
        values = [
            _float_or_none(row.get(field))
            for row in per_seed
            if isinstance(row, Mapping)
        ]
        parsed_values = [value for value in values if value is not None]
        if parsed_values:
            return round(min(parsed_values), 4)
    return _float_or_none(fallback)


def _mapping(payload: object) -> Mapping[str, object]:
    return payload if isinstance(payload, Mapping) else {}


def _float_or_none(payload: object) -> float | None:
    if isinstance(payload, bool) or not isinstance(payload, (int, float)):
        return None
    return round(float(payload), 4)


def _log(message: str) -> None:
    print(f"[mind-gate] {message}", file=sys.stderr)


if __name__ == "__main__":
    main()
