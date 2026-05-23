from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_recovery_distill import (
    DEFAULT_CARRION_RECOVERY_DISTILL_ARTIFACT_MODE,
    DEFAULT_CARRION_RECOVERY_DISTILL_EVAL_SEEDS,
    DEFAULT_CARRION_RECOVERY_DISTILL_EVAL_TICKS,
    DEFAULT_CARRION_RECOVERY_DISTILL_FIXTURES,
    DEFAULT_CARRION_RECOVERY_DISTILL_HIDDEN_UNITS,
    DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_CONTEXT_GATE,
    DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_MAX_LINEAR_OVERRIDE_MARGIN,
    DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_RECOVERY_PHASE_TICKS,
    DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_SCALE,
    MIND_V3_CARRION_RECOVERY_DISTILL_SCHEMA_VERSION,
    MIND_V3_CARRION_RECOVERY_DISTILL_UNIFORM_WEIGHT_POLICY,
    MIND_V3_CARRION_RECOVERY_DISTILL_WEIGHT_POLICY,
    CarrionRecoveryDistillError,
    build_carrion_recovery_distillation_report,
    write_carrion_recovery_distillation_report,
)
from evolution_sim.mind.v3_policy import (
    MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_NONE,
    MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_VISIBLE_CARRION_OR_RECOVERY_PHASE,
    MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_VISIBLE_CARRION_SCAVENGER,
)
from evolution_sim.mind.horizon_labels import (
    DEFAULT_HORIZON_TICKS,
    parse_horizon_ticks,
)
from evolution_sim.mind.v3_neural import (
    MIND_V3_NEURAL_ARTIFACT_MODE_ANCHORED,
    MIND_V3_NEURAL_ARTIFACT_MODE_HORIZON_FIXTURE,
    MIND_V3_NEURAL_DEFAULT_SEED,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Distill Mind v3 carrion recovery archive elites into a "
            "deterministic neural artifact and evaluate it beside the linear "
            "baseline."
        )
    )
    parser.add_argument(
        "--archive-report",
        type=Path,
        default=Path("output/mind/mind-v3-v39-outcome-recovery-archive.json"),
        help="Input mind_v3_carrion_recovery_archive_v1 report.",
    )
    parser.add_argument(
        "--archive-split-report",
        type=Path,
        default=None,
        help=(
            "Optional leakage-safe archive split manifest. When provided, "
            "distillation trains only on train records and reports held-out "
            "branch-state trajectory diagnostics."
        ),
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(horizon) for horizon in DEFAULT_HORIZON_TICKS),
        help="Comma-separated future tick horizons to label.",
    )
    parser.add_argument(
        "--artifact-mode",
        choices=[
            MIND_V3_NEURAL_ARTIFACT_MODE_ANCHORED,
            MIND_V3_NEURAL_ARTIFACT_MODE_HORIZON_FIXTURE,
        ],
        default=DEFAULT_CARRION_RECOVERY_DISTILL_ARTIFACT_MODE,
        help=(
            "Artifact runtime mode. The default keeps the linear anchor and "
            "uses the neural component as a recovery residual."
        ),
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=DEFAULT_CARRION_RECOVERY_DISTILL_HIDDEN_UNITS,
        help="Deterministic hidden projection width.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=MIND_V3_NEURAL_DEFAULT_SEED,
        help="Deterministic neural projection seed.",
    )
    parser.add_argument(
        "--neural-residual-scale",
        type=float,
        default=DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_SCALE,
        help=(
            "Artifact-scoped anchored-neural residual scale. This is lower "
            "than the generic anchored-neural default to reduce broad "
            "open-world regressions for recovery-distilled artifacts."
        ),
    )
    parser.add_argument(
        "--neural-residual-max-linear-override-margin",
        type=float,
        default=(
            DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_MAX_LINEAR_OVERRIDE_MARGIN
        ),
        help=(
            "Maximum linear-anchor score margin that the neural residual may "
            "override."
        ),
    )
    parser.add_argument(
        "--neural-residual-context-gate",
        choices=[
            MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_NONE,
            MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_VISIBLE_CARRION_SCAVENGER,
            (
                MIND_V3_NEURAL_RESIDUAL_CONTEXT_GATE_VISIBLE_CARRION_OR_RECOVERY_PHASE
            ),
        ],
        default=DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_CONTEXT_GATE,
        help=(
            "Artifact-scoped context gate for anchored-neural residuals. The "
            "default applies recovery residuals for visible carrion and a "
            "short post-contact recovery phase."
        ),
    )
    parser.add_argument(
        "--neural-residual-recovery-phase-ticks",
        type=int,
        default=DEFAULT_CARRION_RECOVERY_DISTILL_NEURAL_RESIDUAL_RECOVERY_PHASE_TICKS,
        help=(
            "Number of post-animal-resource decisions that keep the recovery "
            "residual context gate open."
        ),
    )
    parser.add_argument(
        "--weight-policy",
        choices=[
            MIND_V3_CARRION_RECOVERY_DISTILL_WEIGHT_POLICY,
            MIND_V3_CARRION_RECOVERY_DISTILL_UNIFORM_WEIGHT_POLICY,
        ],
        default=MIND_V3_CARRION_RECOVERY_DISTILL_WEIGHT_POLICY,
        help=(
            "Trajectory weighting policy. The default down-weights failure "
            "boundaries and gives more weight to survivor/reproduction/"
            "scavenging elites."
        ),
    )
    parser.add_argument(
        "--eval-seeds",
        default=",".join(
            str(seed) for seed in DEFAULT_CARRION_RECOVERY_DISTILL_EVAL_SEEDS
        ),
        help="Comma-separated broad open-world eval seeds.",
    )
    parser.add_argument(
        "--eval-ticks",
        type=int,
        default=DEFAULT_CARRION_RECOVERY_DISTILL_EVAL_TICKS,
        help="Broad open-world eval horizon.",
    )
    parser.add_argument(
        "--fixture-names",
        default=",".join(DEFAULT_CARRION_RECOVERY_DISTILL_FIXTURES),
        help="Comma-separated controlled fixture names.",
    )
    parser.add_argument(
        "--fixture-seeds",
        default=None,
        help="Optional comma-separated fixture seeds. Defaults to --eval-seeds.",
    )
    parser.add_argument(
        "--fixture-ticks",
        type=int,
        default=None,
        help="Optional controlled fixture horizon. Defaults to --eval-ticks.",
    )
    parser.add_argument(
        "--horizon-output",
        type=Path,
        default=Path(
            "output/mind/mind-v3-carrion-recovery-distill-horizon-labels.json"
        ),
        help="Output horizon label report path.",
    )
    parser.add_argument(
        "--artifact-output",
        type=Path,
        default=Path("output/mind/mind-v3-carrion-recovery-distill-artifact.json"),
        help="Output neural artifact path.",
    )
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        default=Path("output/mind/mind-v3-carrion-recovery-distill-evaluation.json"),
        help="Output broad/fixture evaluation report path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-carrion-recovery-distill-report.json"),
        help="Output top-level distillation report path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        eval_seeds = _parse_seeds(args.eval_seeds, field="--eval-seeds")
        fixture_seeds = (
            _parse_seeds(args.fixture_seeds, field="--fixture-seeds")
            if args.fixture_seeds
            else None
        )
        fixture_names = _parse_names(args.fixture_names, field="--fixture-names")
        report = build_carrion_recovery_distillation_report(
            archive_report_path=args.archive_report,
            archive_split_report_path=args.archive_split_report,
            horizons=parse_horizon_ticks(args.horizons),
            artifact_mode=str(args.artifact_mode),
            hidden_units=int(args.hidden_units),
            seed=int(args.seed),
            neural_residual_scale=float(args.neural_residual_scale),
            neural_residual_max_linear_override_margin=float(
                args.neural_residual_max_linear_override_margin
            ),
            neural_residual_context_gate=str(args.neural_residual_context_gate),
            neural_residual_recovery_phase_ticks=int(
                args.neural_residual_recovery_phase_ticks
            ),
            weight_policy=str(args.weight_policy),
            eval_seeds=eval_seeds,
            eval_ticks=int(args.eval_ticks),
            fixture_names=fixture_names,
            fixture_seeds=fixture_seeds,
            fixture_ticks=args.fixture_ticks,
            horizon_output_path=args.horizon_output,
            artifact_output_path=args.artifact_output,
            evaluation_output_path=args.evaluation_output,
        )
        write_carrion_recovery_distillation_report(report, args.output)
    except (OSError, ValueError, CarrionRecoveryDistillError) as exc:
        raise SystemExit(f"failed to distill carrion recovery archive: {exc}") from exc

    _print_report_summary(report, args)


def _print_report_summary(
    report: Mapping[str, object],
    args: argparse.Namespace,
) -> None:
    training = _mapping(report.get("training"))
    acceptance = _mapping(report.get("acceptance"))
    evaluation = _mapping(report.get("evaluation"))
    open_eval = _mapping(evaluation.get("open"))
    comparison = _mapping(open_eval.get("comparison"))
    candidate = _aggregate(comparison, "mind_v3_recovery_distilled")
    linear = _aggregate(comparison, "mind_v3_linear")
    heuristic = _aggregate(comparison, "heuristic")
    candidate_delta = _mapping(comparison.get("candidate_vs_linear_delta"))
    seed_regression_summary = _mapping(
        acceptance.get("open_per_seed_regression_summary")
    )

    print(f"carrion_recovery_distill={args.output}")
    print(f"horizon_labels={args.horizon_output}")
    print(f"mind_v3_neural_artifact={args.artifact_output}")
    print(f"evaluation={args.evaluation_output}")
    print(f"schema_version={MIND_V3_CARRION_RECOVERY_DISTILL_SCHEMA_VERSION}")
    print(f"artifact_mode={training.get('artifact_mode')}")
    print(f"neural_residual_scale={training.get('neural_residual_scale')}")
    print(
        "neural_residual_max_linear_override_margin="
        f"{training.get('neural_residual_max_linear_override_margin')}"
    )
    print(
        "neural_residual_context_gate="
        f"{training.get('neural_residual_context_gate')}"
    )
    print(
        "neural_residual_recovery_phase_ticks="
        f"{training.get('neural_residual_recovery_phase_ticks', 0)}"
    )
    recovery_bias = _mapping(training.get("recovery_phase_action_bias"))
    print(
        "recovery_phase_action_bias_policy="
        f"{recovery_bias.get('policy')}"
    )
    print(
        "recovery_phase_action_bias_record_count="
        f"{recovery_bias.get('record_count', 0)}"
    )
    print(f"weight_policy={_mapping(report.get('contract')).get('weight_policy')}")
    print(
        "archive_split_consumed="
        f"{training.get('archive_split_consumed', False)}"
    )
    print(f"selected_trajectory_count={training.get('selected_trajectory_count')}")
    print(f"trained_record_count={training.get('trained_record_count')}")
    heldout = _mapping(report.get("heldout_branch_state_evaluation"))
    print(f"heldout_branch_evaluation_enabled={heldout.get('enabled', False)}")
    print(f"heldout_branch_loaded_record_count={heldout.get('loaded_record_count', 0)}")
    print(f"heldout_branch_survivor_count={heldout.get('survivor_count', 0)}")
    print(f"heldout_branch_failure_count={heldout.get('failure_count', 0)}")
    print(
        "data_path_acceptance_passed="
        f"{acceptance.get('data_path_acceptance_passed')}"
    )
    print(
        "promotion_candidate_passed="
        f"{acceptance.get('promotion_candidate_passed')}"
    )
    _print_open_summary("open_heuristic", heuristic)
    _print_open_summary("open_linear", linear)
    _print_open_summary("open_candidate", candidate)
    print(
        "open_candidate_vs_linear_alive_agents_mean_delta="
        f"{candidate_delta.get('alive_agents_mean', 0)}"
    )
    print(
        "open_candidate_vs_linear_births_mean_delta="
        f"{candidate_delta.get('births_mean', 0)}"
    )
    print(
        "open_candidate_vs_linear_min_seed_alive_agents_delta="
        f"{seed_regression_summary.get('min_alive_agents_delta', 0)}"
    )
    print(
        "open_candidate_vs_linear_min_seed_births_delta="
        f"{seed_regression_summary.get('min_births_delta', 0)}"
    )
    fixture = _mapping(evaluation.get("fixture"))
    for label, suite_key in (
        ("fixture_candidate", "candidate_suite"),
        ("fixture_linear", "linear_baseline_suite"),
    ):
        _print_fixture_suite(label, _mapping(fixture.get(suite_key)))


def _print_open_summary(prefix: str, aggregate: Mapping[str, object]) -> None:
    print(f"{prefix}_alive_agents_mean={aggregate.get('alive_agents_mean', 0)}")
    print(f"{prefix}_births_mean={aggregate.get('births_mean', 0)}")
    print(f"{prefix}_deaths_mean={aggregate.get('deaths_mean', 0)}")
    print(
        f"{prefix}_heuristic_action_source_count="
        f"{aggregate.get('heuristic_action_source_count', 0)}"
    )
    _print_outcome_metrics(prefix, aggregate.get("outcome_metrics"))


def _print_fixture_suite(prefix: str, suite: Mapping[str, object]) -> None:
    policy_key = str(suite.get("evaluated_policy_key", "mind_v3"))
    fixtures = suite.get("fixtures")
    for fixture in fixtures if isinstance(fixtures, list) else []:
        if not isinstance(fixture, Mapping):
            continue
        fixture_name = str(fixture.get("fixture", "unknown"))
        comparison = _mapping(fixture.get("comparison"))
        policy = _mapping(comparison.get(policy_key))
        aggregate = _mapping(policy.get("aggregate"))
        metric_prefix = f"{prefix}_{fixture_name}"
        print(
            f"{metric_prefix}_alive_agents_mean="
            f"{aggregate.get('alive_agents_mean', 0)}"
        )
        print(f"{metric_prefix}_births_mean={aggregate.get('births_mean', 0)}")
        print(f"{metric_prefix}_deaths_mean={aggregate.get('deaths_mean', 0)}")
        print(
            f"{metric_prefix}_heuristic_action_source_count="
            f"{aggregate.get('heuristic_action_source_count', 0)}"
        )
        _print_outcome_metrics(metric_prefix, aggregate.get("outcome_metrics"))


def _print_outcome_metrics(prefix: str, metrics: object) -> None:
    payload = _mapping(metrics)
    keys = (
        "terminal_survivor_run_count",
        "extinct_run_count",
        "total_terminal_alive_agents",
        "total_births",
        "runs_with_births",
        "total_deaths",
        "total_scavenger_terminal_agents",
        "total_scavenger_parent_births",
        "total_scavenger_child_births",
        "total_animal_resource_consumption_events",
        "total_carcass_consumption_events",
        "total_fresh_kill_consumption_events",
        "total_scavenger_animal_resource_events",
        "total_scavenger_carcass_events",
        "total_scavenger_fresh_kill_events",
    )
    for key in keys:
        print(f"{prefix}_{key}={payload.get(key, 0)}")


def _aggregate(
    comparison: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    payload = _mapping(comparison.get(key))
    return _mapping(payload.get("aggregate"))


def _parse_seeds(raw: str, *, field: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in str(raw).split(",") if part.strip())
    if not values:
        raise CarrionRecoveryDistillError(f"{field} must include at least one seed")
    return values


def _parse_names(raw: str, *, field: str) -> tuple[str, ...]:
    values = tuple(dict.fromkeys(part.strip() for part in raw.split(",") if part.strip()))
    if not values:
        raise CarrionRecoveryDistillError(f"{field} must include at least one name")
    return values


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
