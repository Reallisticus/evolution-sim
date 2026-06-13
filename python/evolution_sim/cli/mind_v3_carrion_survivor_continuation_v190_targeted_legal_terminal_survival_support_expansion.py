from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v190_targeted_legal_terminal_survival_support_expansion import (
    DEFAULT_CONTINUATION_SCRIPTS,
    DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    DEFAULT_MAX_DOMINANT_REQUESTED_ACTION_SHARE,
    DEFAULT_MIN_BRANCH_TICK,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SUPPORT_TRAJECTORY_DIR,
    DEFAULT_TICKS,
    DEFAULT_V189_REPORT_PATH,
    EXPECTED_V189_REPORT_EXACT_DIGEST,
    EXPECTED_V189_REQUIRED_ROUTE,
    run_carrion_survivor_continuation_v190_targeted_legal_terminal_survival_support_expansion,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run diagnostics-only v190 targeted legal terminal-survival support "
            "expansion before any slice-3 training."
        )
    )
    parser.add_argument("--v189-report", type=Path, default=DEFAULT_V189_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--support-trajectory-dir",
        type=Path,
        default=DEFAULT_SUPPORT_TRAJECTORY_DIR,
    )
    parser.add_argument(
        "--expected-v189-report-exact-digest",
        default=EXPECTED_V189_REPORT_EXACT_DIGEST,
    )
    parser.add_argument("--required-v189-route", default=EXPECTED_V189_REQUIRED_ROUTE)
    parser.add_argument("--seeds", default="13,19,29,37,41,43")
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument(
        "--continuation-scripts",
        default=",".join(DEFAULT_CONTINUATION_SCRIPTS),
    )
    parser.add_argument(
        "--max-branch-points-per-seed",
        type=int,
        default=DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    )
    parser.add_argument("--min-branch-tick", type=int, default=DEFAULT_MIN_BRANCH_TICK)
    parser.add_argument(
        "--max-dominant-requested-action-share",
        type=float,
        default=DEFAULT_MAX_DOMINANT_REQUESTED_ACTION_SHARE,
    )
    parser.add_argument(
        "--skip-replay-verification",
        action="store_true",
        help=(
            "Skip branch replay verification. This is diagnostic only; support "
            "counting still records the configured verification policy."
        ),
    )
    args = parser.parse_args()

    report = (
        run_carrion_survivor_continuation_v190_targeted_legal_terminal_survival_support_expansion(
            v189_report_path=args.v189_report,
            output_path=args.output,
            support_trajectory_dir=args.support_trajectory_dir,
            expected_v189_report_exact_digest=args.expected_v189_report_exact_digest,
            required_v189_route=args.required_v189_route,
            seeds=_parse_int_csv(args.seeds),
            ticks=args.ticks,
            continuation_scripts=_parse_str_csv(args.continuation_scripts),
            max_branch_points_per_seed=args.max_branch_points_per_seed,
            min_branch_tick=args.min_branch_tick,
            max_dominant_requested_action_share=(
                args.max_dominant_requested_action_share
            ),
            verify_replay=not args.skip_replay_verification,
        )
    )

    source = report["source_validation"]
    support = report["legal_support_audit"]
    route = report["route_decision"]
    search = report["expansion_search"]
    print(f"carrion_survivor_continuation_v190={args.output}")
    print(f"exact_digest={report['exact_digest']}")
    print(f"source_validation_passed={source.get('passed')}")
    print(f"expansion_ran={search.get('ran')}")
    print(f"branch_run_count={search.get('branch_run_count')}")
    print(
        "clean_legal_support_seed_count="
        f"{support.get('clean_legal_support_seed_count')}/"
        f"{support.get('target_seed_count')}"
    )
    print(
        "clean_terminal_survivors_by_seed="
        f"{support.get('clean_terminal_survivors_by_seed')}"
    )
    print(
        "aggregate_unsupported_resolved_action_count="
        f"{search.get('unsupported_resolved_action_count')}"
    )
    print(
        "dominant_requested_action_share="
        f"{support.get('dominant_requested_action_share')}"
    )
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(f"slice_3_training_authorized={route.get('slice_3_training_authorized')}")
    print(f"training_ran={report['training_ran']}")
    print(f"slice_3_training_consumed={report['slice_3_training_consumed']}")
    print(f"runtime_action_selection_changed={report['runtime_action_selection_changed']}")
    print(f"promotion_authorized={report['promotion_authorized']}")
    print(f"gate_relaxation_allowed={report['gate_relaxation_allowed']}")


def _parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_str_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


if __name__ == "__main__":
    main()
