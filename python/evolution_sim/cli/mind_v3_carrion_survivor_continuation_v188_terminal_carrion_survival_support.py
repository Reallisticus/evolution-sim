from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_counterfactual import DEFAULT_COUNTERFACTUAL_SCRIPTS
from evolution_sim.mind.carrion_survivor_continuation_v188_terminal_carrion_survival_support import (
    DEFAULT_BASE_SCRIPT,
    DEFAULT_CARRION_FIXTURE_SEEDS,
    DEFAULT_CONTINUATION_SCRIPTS,
    DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    DEFAULT_MIN_BRANCH_TICK,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SUPPORT_TRAJECTORY_DIR,
    DEFAULT_TICKS,
    DEFAULT_V187_REPORT_PATH,
    EXPECTED_V187_REPORT_EXACT_DIGEST,
    REQUIRED_V187_ROUTE,
    run_carrion_survivor_continuation_v188_terminal_carrion_survival_support,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v188 terminal carrion survival support "
            "search. The command validates the pinned v187 report digest and "
            "route, searches a bounded legal policy-visible branch/script "
            "space on carrion_only@120, and writes support evidence or an "
            "explicit bounded infeasibility scope. It does not train, consume "
            "slice 3, create a runtime artifact, change runtime action "
            "selection, relax gates, or authorize promotion."
        )
    )
    parser.add_argument("--v187-report", type=Path, default=DEFAULT_V187_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--support-trajectory-dir",
        type=Path,
        default=DEFAULT_SUPPORT_TRAJECTORY_DIR,
    )
    parser.add_argument(
        "--expected-v187-report-exact-digest",
        default=EXPECTED_V187_REPORT_EXACT_DIGEST,
    )
    parser.add_argument("--required-v187-route", default=REQUIRED_V187_ROUTE)
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_CARRION_FIXTURE_SEEDS),
        help="Comma-separated carrion fixture seeds.",
    )
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument(
        "--base-script",
        choices=DEFAULT_COUNTERFACTUAL_SCRIPTS,
        default=DEFAULT_BASE_SCRIPT,
    )
    parser.add_argument(
        "--continuation-script",
        action="append",
        choices=DEFAULT_COUNTERFACTUAL_SCRIPTS,
        help=(
            "Continuation script to fan out from each branch. Repeat to run "
            "multiple scripts. Defaults to all scripts."
        ),
    )
    parser.add_argument(
        "--max-branch-points-per-seed",
        type=int,
        default=DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    )
    parser.add_argument("--min-branch-tick", type=int, default=DEFAULT_MIN_BRANCH_TICK)
    parser.add_argument(
        "--no-verify-replay",
        action="store_true",
        help="Skip deterministic branch replay verification.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    continuation_scripts = (
        tuple(args.continuation_script)
        if args.continuation_script
        else DEFAULT_CONTINUATION_SCRIPTS
    )
    try:
        report = (
            run_carrion_survivor_continuation_v188_terminal_carrion_survival_support(
                v187_report_path=args.v187_report,
                output_path=args.output,
                support_trajectory_dir=args.support_trajectory_dir,
                expected_v187_report_exact_digest=(
                    args.expected_v187_report_exact_digest
                ),
                required_v187_route=args.required_v187_route,
                seeds=_parse_seeds(args.seeds),
                ticks=int(args.ticks),
                base_script=str(args.base_script),
                continuation_scripts=continuation_scripts,
                max_branch_points_per_seed=int(args.max_branch_points_per_seed),
                min_branch_tick=int(args.min_branch_tick),
                verify_replay=not bool(args.no_verify_replay),
            )
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v188 terminal carrion survival support search: "
            f"{exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: Mapping[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_validation"))
    support_search = _payload(report.get("support_search"))
    support_result = _payload(report.get("support_result"))
    legality = _payload(report.get("legality_action_mask_validation"))
    route = _payload(report.get("route_decision"))
    best = _payload(support_result.get("best_support"))
    print(f"carrion_survivor_continuation_v188_terminal_survival_support={output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_validation_passed={source.get('passed')}")
    print(
        "v187_report_exact_digest="
        f"{source.get('observed_v187_report_exact_digest')}"
    )
    print(f"v187_route={source.get('observed_v187_route')}")
    print(f"support_search_ran={support_search.get('ran')}")
    print(f"branch_point_count={support_search.get('branch_point_count')}")
    print(f"branch_run_count={support_search.get('branch_run_count')}")
    print(
        "successful_branch_run_count="
        f"{support_search.get('successful_branch_run_count')}"
    )
    print(f"positive_seed_count={support_search.get('positive_seed_count')}")
    print(f"positive_support_found={support_result.get('positive_support_found')}")
    print(
        "all_target_seeds_have_positive_support="
        f"{support_result.get('all_target_seeds_have_positive_support')}"
    )
    print(
        "terminal_survivors_by_seed="
        f"{support_result.get('terminal_survivors_by_seed')}"
    )
    print(f"legality_action_mask_passed={legality.get('passed')}")
    print(
        "unsupported_requested_action_count="
        f"{legality.get('unsupported_requested_action_count')}"
    )
    print(
        "unsupported_resolved_action_count="
        f"{legality.get('unsupported_resolved_action_count')}"
    )
    print(f"replay_verified={legality.get('replay_verified')}")
    if best:
        print(f"best_branch_id={best.get('branch_id')}")
        print(f"best_continuation_script={best.get('continuation_script')}")
        print(f"best_alive_agents={best.get('alive_agents')}")
        print(f"best_births={best.get('births')}")
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"training_artifact_created={report.get('training_artifact_created')}")
    print(f"slice_3_training_consumed={report.get('slice_3_training_consumed')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(
        "runtime_action_selection_changed="
        f"{report.get('runtime_action_selection_changed')}"
    )
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"gate_relaxation_allowed={report.get('gate_relaxation_allowed')}")
    print(f"exact_digest={report.get('exact_digest')}")


def _payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("--seeds must include at least one seed")
    return values


if __name__ == "__main__":
    main()
