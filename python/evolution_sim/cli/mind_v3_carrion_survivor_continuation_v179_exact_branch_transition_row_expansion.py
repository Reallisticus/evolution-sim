from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v172_replay_target_dataset_expansion import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V172_REPORT_PATH,
    DEFAULT_TARGET_DATASET_OUTPUT_PATH as DEFAULT_V172_DATASET_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_v177_exact_branch_replay_expansion import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V177_REPORT_PATH,
    DEFAULT_TRANSITION_DATASET_OUTPUT_PATH as DEFAULT_V177_TRANSITION_DATASET_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_v179_exact_branch_transition_row_expansion import (
    DEFAULT_BRANCHES_PER_SEED,
    DEFAULT_MAX_FORCED_ACTIONS_PER_BRANCH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TICKS,
    DEFAULT_TRANSITION_DATASET_OUTPUT_PATH,
    EXPECTED_V172_CLASSIFICATION,
    EXPECTED_V172_DATASET_DIGEST,
    EXPECTED_V172_EXACT_DIGEST,
    EXPECTED_V172_ROW_COUNT,
    CarrionSurvivorContinuationV179ExactBranchTransitionRowExpansionError,
    run_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run diagnostics-only v179 exact-branch transition-row support "
            "expansion from v172 replay-target rows. This materializes current, "
            "forced, and next public transition rows through exact replay. It "
            "does not train, fit, evaluate, create runtime artifacts, change "
            "runtime action selection, relax gates, or authorize promotion."
        )
    )
    parser.add_argument("--v172-report", type=Path, default=DEFAULT_V172_REPORT_PATH)
    parser.add_argument("--v172-dataset", type=Path, default=DEFAULT_V172_DATASET_PATH)
    parser.add_argument(
        "--v177-report",
        "--source-report",
        dest="v177_report",
        type=Path,
        default=DEFAULT_V177_REPORT_PATH,
    )
    parser.add_argument(
        "--v177-transition-dataset",
        type=Path,
        default=DEFAULT_V177_TRANSITION_DATASET_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--transition-dataset-output",
        type=Path,
        default=DEFAULT_TRANSITION_DATASET_OUTPUT_PATH,
    )
    parser.add_argument(
        "--expected-v172-exact-digest",
        default=EXPECTED_V172_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v172-dataset-digest",
        default=EXPECTED_V172_DATASET_DIGEST,
    )
    parser.add_argument(
        "--expected-v172-classification",
        default=EXPECTED_V172_CLASSIFICATION,
    )
    parser.add_argument(
        "--expected-v172-row-count",
        type=int,
        default=EXPECTED_V172_ROW_COUNT,
    )
    parser.add_argument(
        "--expected-v177-report-exact-digest",
        "--expected-source-report-exact-digest",
        dest="expected_v177_report_exact_digest",
        default=None,
    )
    parser.add_argument("--expected-v177-dataset-digest", default=None)
    parser.add_argument(
        "--branches-per-seed",
        type=int,
        default=DEFAULT_BRANCHES_PER_SEED,
    )
    parser.add_argument(
        "--max-forced-actions-per-branch",
        type=int,
        default=DEFAULT_MAX_FORCED_ACTIONS_PER_BRANCH,
    )
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument(
        "--no-verify-replay",
        action="store_true",
        help="Skip deterministic replay verification. Reports remain diagnostics-only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v179_exact_branch_transition_row_expansion(
            v172_report_path=args.v172_report,
            v172_dataset_path=args.v172_dataset,
            v177_report_path=args.v177_report,
            v177_transition_dataset_path=args.v177_transition_dataset,
            output_path=args.output,
            transition_dataset_output_path=args.transition_dataset_output,
            expected_v172_exact_digest=args.expected_v172_exact_digest,
            expected_v172_dataset_digest=args.expected_v172_dataset_digest,
            expected_v172_classification=args.expected_v172_classification,
            expected_v172_row_count=args.expected_v172_row_count,
            expected_v177_report_exact_digest=args.expected_v177_report_exact_digest,
            expected_v177_dataset_digest=args.expected_v177_dataset_digest,
            branches_per_seed=args.branches_per_seed,
            max_forced_actions_per_branch=args.max_forced_actions_per_branch,
            ticks=args.ticks,
            verify_replay=not args.no_verify_replay,
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationV179ExactBranchTransitionRowExpansionError,
    ) as exc:
        raise SystemExit(
            "failed to run v179 carrion survivor-continuation exact-branch "
            f"transition-row expansion: {exc}"
        ) from exc
    _print_summary(report, args.output, args.transition_dataset_output)


def _print_summary(
    report: Mapping[str, object],
    output: Path,
    transition_dataset_output: Path,
) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_validation"))
    plan = _payload(report.get("plan_generation"))
    selection = _payload(report.get("selection"))
    materialization = _payload(report.get("branch_materialization"))
    metrics = _payload(report.get("metrics"))
    support = _payload(report.get("support_summary"))
    dataset = _payload(report.get("dataset"))
    route = _payload(report.get("route_recommendation"))
    support_observed = _payload(support.get("observed"))
    print(
        "carrion_survivor_continuation_v179_exact_branch_transition_row_"
        f"expansion_report={output}"
    )
    print(f"transition_dataset_output={transition_dataset_output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_validation_passed={source.get('passed')}")
    print(
        "source_validation.v177_source_digests_pinned="
        f"{source.get('v177_source_digests_pinned')}"
    )
    print(
        "source_validation.observed_v177_report_exact_digest="
        f"{source.get('observed_v177_report_exact_digest')}"
    )
    print(
        "source_validation.observed_v177_dataset_digest="
        f"{source.get('observed_v177_dataset_digest')}"
    )
    print(
        "source_validation.expected_v177_report_exact_digest_provided="
        f"{source.get('expected_v177_report_exact_digest_provided')}"
    )
    print(
        "source_validation.expected_v177_dataset_digest_provided="
        f"{source.get('expected_v177_dataset_digest_provided')}"
    )
    print(f"plan_generation_passed={plan.get('passed')}")
    print(f"selected_plan_row_count={plan.get('selected_plan_row_count')}")
    print(f"selected_branch_point_count={selection.get('selected_branch_point_count')}")
    print(
        "materialized_branch_point_count="
        f"{materialization.get('materialized_branch_point_count')}"
    )
    print(f"transition_row_count={dataset.get('row_count')}")
    print(f"support_row_count={support_observed.get('row_count')}")
    print(f"support_seed_count={support_observed.get('seed_count')}")
    print(f"support_branch_count={support_observed.get('branch_count')}")
    print(f"support_forced_action_count={support_observed.get('forced_action_count')}")
    print(f"v178_default_support_thresholds_met={support.get('passed')}")
    print(f"forced_action_used_count={metrics.get('forced_action_used_count')}")
    print(f"all_replays_verified={metrics.get('all_replays_verified')}")
    print(f"dataset_digest={dataset.get('dataset_digest')}")
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(
        "route_recommendation.v178_default_threshold_audit_recommended="
        f"{route.get('v178_default_threshold_audit_recommended')}"
    )
    print(
        "route_recommendation.transition_row_training_authorized="
        f"{route.get('transition_row_training_authorized')}"
    )
    print(f"training_ran={report.get('training_ran')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"exact_digest={report.get('exact_digest')}")


def _payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
