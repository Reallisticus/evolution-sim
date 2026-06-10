from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v163_tied_set_branch_target_expansion import (
    DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TICKS,
    DEFAULT_TRAJECTORY_GLOB,
    DEFAULT_V162_REPORT_PATH,
    CarrionSurvivorContinuationV163TiedSetBranchTargetExpansionError,
    run_carrion_survivor_continuation_v163_tied_set_branch_target_expansion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v163 tied-set branch target expansion "
            "slice. The command validates v162, selects representative tied "
            "strict broad branch points, and only emits outcome evidence when "
            "exact deterministic branch materialization is proven."
        )
    )
    parser.add_argument("--v162-report", type=Path, default=DEFAULT_V162_REPORT_PATH)
    parser.add_argument(
        "--v160-artifact",
        type=Path,
        default=None,
        help="Optional override for the v160 artifact path recorded by v162.",
    )
    parser.add_argument("--trajectory-glob", default=DEFAULT_TRAJECTORY_GLOB)
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        default=None,
        help="Explicit strict broad trajectory JSONL/JSONL.GZ path. May be repeated.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument(
        "--max-branch-points-per-seed",
        type=int,
        default=DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    )
    parser.add_argument(
        "--skip-replay-verification",
        action="store_true",
        help="Skip second-run branch digest verification. Intended only for local debugging.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write the selected branch plan and fail closed without branch execution.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v163_tied_set_branch_target_expansion(
            v162_report_path=args.v162_report,
            v160_artifact_path=args.v160_artifact,
            trajectory_glob=args.trajectory_glob,
            trajectory_paths=args.trajectory,
            output_path=args.output,
            ticks=args.ticks,
            max_branch_points_per_seed=args.max_branch_points_per_seed,
            verify_replay=not args.skip_replay_verification,
            attempt_branch_replay=not args.plan_only,
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationV163TiedSetBranchTargetExpansionError,
    ) as exc:
        raise SystemExit(
            "failed to run v163 carrion survivor-continuation tied-set "
            f"branch target expansion: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: dict[str, object], output: Path) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    source = report.get("source_validation")
    source_payload = source if isinstance(source, dict) else {}
    selection = report.get("selection_plan")
    selection_payload = selection if isinstance(selection, dict) else {}
    materialization = report.get("branch_materialization")
    materialization_payload = materialization if isinstance(materialization, dict) else {}
    outcome = report.get("outcome_support")
    outcome_payload = outcome if isinstance(outcome, dict) else {}
    route = report.get("route_recommendation")
    route_payload = route if isinstance(route, dict) else {}
    print(f"carrion_survivor_continuation_v163_tied_set_branch_target_expansion_report={output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={source_payload.get('passed')}")
    print(
        "selected_branch_point_count="
        f"{selection_payload.get('selected_branch_point_count')}"
    )
    print(
        "materialized_branch_point_count="
        f"{materialization_payload.get('materialized_branch_point_count')}"
    )
    print(
        "materialization_passed="
        f"{materialization_payload.get('passed')}"
    )
    print(
        "candidate_run_count="
        f"{outcome_payload.get('candidate_run_count')}"
    )
    print(
        "outcome_support_noncollapsed="
        f"{outcome_payload.get('outcome_support_noncollapsed')}"
    )
    print(f"recommended_next_route={route_payload.get('recommended_next_route')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"live_ab_allowed={report.get('live_ab_allowed')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"exact_digest={report.get('exact_digest')}")


if __name__ == "__main__":
    main()
