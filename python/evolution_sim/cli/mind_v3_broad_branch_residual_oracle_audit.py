from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.broad_branch_residual_oracle_audit import (
    V99_DEFAULT_BRANCH_POINTS_PER_SEED,
    V99_DEFAULT_SUPPORT_SEEDS,
    V99_DEFAULT_TICKS,
    V99_MIN_BRANCH_POINTS,
    build_broad_branch_residual_oracle_audit_report,
    write_broad_branch_residual_oracle_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the v99 broad non-strict branch residual oracle diagnostic. "
            "This replays broad linear Mind v3 states with forced first-action "
            "alternatives and does not train or promote a runtime policy."
        )
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in V99_DEFAULT_SUPPORT_SEEDS),
        help="Comma-separated non-strict support seeds.",
    )
    parser.add_argument("--ticks", type=int, default=V99_DEFAULT_TICKS)
    parser.add_argument(
        "--max-branch-points-per-seed",
        type=int,
        default=V99_DEFAULT_BRANCH_POINTS_PER_SEED,
    )
    parser.add_argument(
        "--min-branch-points",
        type=int,
        default=V99_MIN_BRANCH_POINTS,
    )
    parser.add_argument(
        "--skip-replay-verification",
        action="store_true",
        help="Disable branch replay verification. Intended only for fast debugging.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v99-broad-branch-residual-oracle-audit.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_broad_branch_residual_oracle_audit_report(
        seeds=_parse_seeds(args.seeds),
        ticks=args.ticks,
        max_branch_points_per_seed=args.max_branch_points_per_seed,
        min_branch_points=args.min_branch_points,
        verify_replay=not args.skip_replay_verification,
    )
    write_broad_branch_residual_oracle_audit_report(report, args.output)
    aggregate = report["aggregate"]
    acceptance = report["acceptance"]
    print(f"v99_broad_branch_residual_oracle_audit={args.output}")
    print(f"schema_version={report['schema_version']}")
    print(f"branch_point_count={aggregate['branch_point_count']}")
    print(f"safe_non_logged_override_count={aggregate['safe_non_logged_override_count']}")
    print(f"safe_non_logged_override_share={aggregate['safe_non_logged_override_share']}")
    print(
        "mean_target_local_score_delta="
        f"{aggregate['target_local_score_delta_summary']['mean']}"
    )
    print(
        "dominant_oracle_action_share="
        f"{aggregate['dominant_oracle_action_share']}"
    )
    print(
        "v99_broad_branch_residual_oracle_accepted="
        f"{acceptance['v99_broad_branch_residual_oracle_accepted']}"
    )
    print(f"blocker_count={acceptance['blocker_count']}")


def _parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise SystemExit("--seeds must include at least one seed")
    return values


if __name__ == "__main__":
    main()
