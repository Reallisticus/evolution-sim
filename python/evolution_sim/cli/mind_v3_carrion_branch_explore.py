from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_branch_explore import (
    DEFAULT_CARRION_BRANCH_BASE_SCRIPT,
    DEFAULT_CARRION_BRANCH_POINTS_PER_SEED,
    MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION,
    CarrionBranchExploreError,
    build_carrion_branch_explore_report,
    write_carrion_branch_explore_report,
)
from evolution_sim.mind.carrion_counterfactual import (
    DEFAULT_CARRION_COUNTERFACTUAL_SEEDS,
    DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
    DEFAULT_COUNTERFACTUAL_SCRIPTS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic Mind v3 carrion fixture branch-and-explore "
            "slice from exact post-contact simulator states."
        )
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_CARRION_COUNTERFACTUAL_SEEDS),
        help="Comma-separated carrion fixture seeds.",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
        help="Fixture rollout horizon.",
    )
    parser.add_argument(
        "--base-script",
        choices=DEFAULT_COUNTERFACTUAL_SCRIPTS,
        default=DEFAULT_CARRION_BRANCH_BASE_SCRIPT,
        help="Policy-visible script used to replay the fixture to a branch point.",
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
        default=DEFAULT_CARRION_BRANCH_POINTS_PER_SEED,
        help="Maximum post-contact branch points retained for each seed.",
    )
    parser.add_argument(
        "--min-branch-tick",
        type=int,
        default=0,
        help="Ignore animal-resource contacts before this tick.",
    )
    parser.add_argument(
        "--no-verify-replay",
        action="store_true",
        help="Skip deterministic replay verification from each branch state.",
    )
    parser.add_argument(
        "--trajectory-output-dir",
        type=Path,
        default=None,
        help="Optional directory for per-branch trajectory JSONL.gz outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-carrion-branch-explore.json"),
        help="Output JSON report path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    continuation_scripts = (
        tuple(args.continuation_script)
        if args.continuation_script
        else DEFAULT_COUNTERFACTUAL_SCRIPTS
    )
    try:
        report = build_carrion_branch_explore_report(
            seeds=_parse_seeds(args.seeds),
            ticks=int(args.ticks),
            base_script=str(args.base_script),
            continuation_scripts=continuation_scripts,
            max_branch_points_per_seed=int(args.max_branch_points_per_seed),
            min_branch_tick=int(args.min_branch_tick),
            trajectory_output_dir=args.trajectory_output_dir,
            verify_replay=not bool(args.no_verify_replay),
        )
        write_carrion_branch_explore_report(report, args.output)
    except (OSError, ValueError, CarrionBranchExploreError) as exc:
        raise SystemExit(f"failed to run carrion branch explore: {exc}") from exc

    aggregate = report["aggregate"]  # type: ignore[index]
    acceptance = report["acceptance"]  # type: ignore[index]
    best = aggregate["best_branch_run"]  # type: ignore[index]
    print(f"carrion_branch_explore={args.output}")
    print(f"schema_version={MIND_V3_CARRION_BRANCH_EXPLORE_SCHEMA_VERSION}")
    print(f"branch_point_count={aggregate['branch_point_count']}")  # type: ignore[index]
    print(f"branch_run_count={aggregate['branch_run_count']}")  # type: ignore[index]
    print(
        "successful_branch_run_count="
        f"{aggregate['successful_branch_run_count']}"  # type: ignore[index]
    )
    print(f"positive_seed_count={aggregate['positive_seed_count']}")  # type: ignore[index]
    print(f"replay_verified={aggregate['replay_verified']}")  # type: ignore[index]
    print(
        "diagnostic_acceptance_passed="
        f"{acceptance['diagnostic_acceptance_passed']}"  # type: ignore[index]
    )
    if isinstance(best, dict):
        print(f"best_branch_id={best['branch_id']}")
        print(f"best_continuation_script={best['continuation_script']}")
        print(f"best_alive_agents={best['alive_agents']}")
        print(f"best_births={best['births']}")


def _parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise CarrionBranchExploreError("--seeds must include at least one seed")
    return values


if __name__ == "__main__":
    main()
