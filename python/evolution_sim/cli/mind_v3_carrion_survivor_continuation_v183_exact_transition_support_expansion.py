from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v180_transition_row_policy_training import (
    DEFAULT_ARTIFACT_OUTPUT_PATH as DEFAULT_V180_ARTIFACT_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V180_REPORT_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_v181_v180_failure_response import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V181_REPORT_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_v182_imputed_abstention_design import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V182_REPORT_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_v183_exact_transition_support_expansion import (
    DEFAULT_BROAD_BRANCHES_PER_SEED,
    DEFAULT_CARRION_BRANCHES_PER_SEED,
    DEFAULT_EXPANDED_TRANSITION_DATASET_OUTPUT_PATH,
    DEFAULT_MAX_FORCED_ACTIONS_PER_BRANCH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SOURCE_TRAJECTORY_DIR,
    DEFAULT_V179_REPORT_PATH,
    DEFAULT_V179_TRANSITION_DATASET_PATH,
    EXPECTED_V179_REPORT_EXACT_DIGEST,
    EXPECTED_V179_TRANSITION_DATASET_DIGEST,
    EXPECTED_V180_ARTIFACT_DIGEST,
    EXPECTED_V180_REPORT_EXACT_DIGEST,
    EXPECTED_V181_REPORT_EXACT_DIGEST,
    EXPECTED_V182_REPORT_EXACT_DIGEST,
    run_carrion_survivor_continuation_v183_exact_transition_support_expansion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run diagnostics-only v183 exact transition-support expansion from "
            "pinned v182/v181/v180/v179 evidence. It writes a report and compact "
            "transition-row dataset only; it does not train, create runtime "
            "artifacts, change runtime action selection, relax gates, or promote."
        )
    )
    parser.add_argument("--v182-report", type=Path, default=DEFAULT_V182_REPORT_PATH)
    parser.add_argument("--v181-report", type=Path, default=DEFAULT_V181_REPORT_PATH)
    parser.add_argument("--v180-report", type=Path, default=DEFAULT_V180_REPORT_PATH)
    parser.add_argument("--v180-artifact", type=Path, default=DEFAULT_V180_ARTIFACT_PATH)
    parser.add_argument("--v179-report", type=Path, default=DEFAULT_V179_REPORT_PATH)
    parser.add_argument(
        "--v179-transition-dataset",
        type=Path,
        default=DEFAULT_V179_TRANSITION_DATASET_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expanded-transition-dataset-output",
        type=Path,
        default=DEFAULT_EXPANDED_TRANSITION_DATASET_OUTPUT_PATH,
    )
    parser.add_argument(
        "--source-trajectory-dir",
        type=Path,
        default=DEFAULT_SOURCE_TRAJECTORY_DIR,
    )
    parser.add_argument(
        "--expected-v182-report-exact-digest",
        default=EXPECTED_V182_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v181-report-exact-digest",
        default=EXPECTED_V181_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v180-report-exact-digest",
        default=EXPECTED_V180_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v180-artifact-digest",
        default=EXPECTED_V180_ARTIFACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v179-report-exact-digest",
        default=EXPECTED_V179_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v179-dataset-digest",
        default=EXPECTED_V179_TRANSITION_DATASET_DIGEST,
    )
    parser.add_argument("--observed-support-floor", type=int, default=2)
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument(
        "--carrion-branches-per-seed",
        type=int,
        default=DEFAULT_CARRION_BRANCHES_PER_SEED,
    )
    parser.add_argument(
        "--broad-branches-per-seed",
        type=int,
        default=DEFAULT_BROAD_BRANCHES_PER_SEED,
    )
    parser.add_argument(
        "--max-forced-actions-per-branch",
        type=int,
        default=DEFAULT_MAX_FORCED_ACTIONS_PER_BRANCH,
    )
    parser.add_argument(
        "--skip-source-generation",
        action="store_true",
        help=(
            "Skip local v182 diagnostic trajectory generation. This is for "
            "source-validation and unit-test dry runs, not handoff evidence."
        ),
    )
    parser.add_argument(
        "--no-verify-replay",
        action="store_true",
        help="Skip deterministic replay verification. Reports remain diagnostics-only.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v183_exact_transition_support_expansion(
            v182_report_path=args.v182_report,
            v181_report_path=args.v181_report,
            v180_report_path=args.v180_report,
            v180_artifact_path=args.v180_artifact,
            v179_report_path=args.v179_report,
            v179_transition_dataset_path=args.v179_transition_dataset,
            output_path=args.output,
            expanded_transition_dataset_output_path=(
                args.expanded_transition_dataset_output
            ),
            source_trajectory_dir=args.source_trajectory_dir,
            expected_v182_report_exact_digest=args.expected_v182_report_exact_digest,
            expected_v181_report_exact_digest=args.expected_v181_report_exact_digest,
            expected_v180_report_exact_digest=args.expected_v180_report_exact_digest,
            expected_v180_artifact_digest=args.expected_v180_artifact_digest,
            expected_v179_report_exact_digest=args.expected_v179_report_exact_digest,
            expected_v179_dataset_digest=args.expected_v179_dataset_digest,
            observed_support_floor=args.observed_support_floor,
            ticks=args.ticks,
            carrion_branches_per_seed=args.carrion_branches_per_seed,
            broad_branches_per_seed=args.broad_branches_per_seed,
            max_forced_actions_per_branch=args.max_forced_actions_per_branch,
            generate_source_trajectories=not args.skip_source_generation,
            verify_replay=not args.no_verify_replay,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v183 carrion survivor-continuation exact "
            f"transition-support expansion: {exc}"
        ) from exc
    _print_summary(
        report,
        output=args.output,
        dataset_output=args.expanded_transition_dataset_output,
    )


def _print_summary(
    report: Mapping[str, object],
    *,
    output: Path,
    dataset_output: Path,
) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_validation"))
    plan = _payload(report.get("plan_generation"))
    materialization = _payload(report.get("branch_materialization"))
    support = _payload(report.get("support_summary"))
    support_observed = _payload(support.get("observed"))
    dataset = _payload(report.get("dataset"))
    target = _payload(report.get("target_support_delta"))
    carrion_delta = _payload(target.get("carrion_observed_support_coverage_delta"))
    broad_delta = _payload(target.get("broad_seed_19_support_hole_coverage_delta"))
    route = _payload(report.get("route_recommendation"))
    print(f"carrion_survivor_continuation_v183_exact_transition_support_expansion={output}")
    print(f"expanded_transition_dataset_output={dataset_output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_validation_passed={source.get('passed')}")
    print(f"selected_plan_row_count={plan.get('selected_plan_row_count')}")
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
    print(
        "carrion_observed_support_coverage_after="
        f"{carrion_delta.get('after_materialized_target_states')}"
    )
    print(
        "broad_seed_19_support_hole_coverage_after="
        f"{broad_delta.get('after_materialized_target_states')}"
    )
    print(f"dataset_digest={dataset.get('dataset_digest')}")
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(f"slice_2_training_authorized={route.get('slice_2_training_authorized')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"training_artifact_created={report.get('training_artifact_created')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"slice_2_training_consumed={report.get('slice_2_training_consumed')}")
    print(f"exact_digest={report.get('exact_digest')}")


def _payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
