from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.branch_action_oracle_audit import (
    DEFAULT_BRANCH_ACTION_ORACLE_BASE_SCRIPT,
    DEFAULT_BRANCH_ACTION_ORACLE_CANDIDATE_ACTIONS,
    DEFAULT_BRANCH_ACTION_ORACLE_POINTS_PER_SEED,
    DEFAULT_BRANCH_ACTION_ORACLE_SEEDS,
    DEFAULT_BRANCH_ACTION_ORACLE_SELECTION_POLICY,
    DEFAULT_BRANCH_ACTION_ORACLE_TARGET_LABELS,
    MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION,
    MODE_BALANCED_BRANCH_ACTION_ORACLE_SELECTION_POLICY,
    BranchActionOracleAuditError,
    build_branch_action_oracle_audit_report,
    write_branch_action_oracle_audit_report,
)
from evolution_sim.mind.carrion_counterfactual import (
    DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
    DEFAULT_COUNTERFACTUAL_SCRIPTS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit exact branch-replay action labels for ambiguous post-carrion "
            "drink/eat states."
        )
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_BRANCH_ACTION_ORACLE_SEEDS),
        help="Comma-separated carrion fixture seeds.",
    )
    parser.add_argument("--ticks", type=int, default=DEFAULT_CARRION_COUNTERFACTUAL_TICKS)
    parser.add_argument(
        "--base-script",
        choices=DEFAULT_COUNTERFACTUAL_SCRIPTS,
        default=DEFAULT_BRANCH_ACTION_ORACLE_BASE_SCRIPT,
    )
    parser.add_argument(
        "--continuation-script",
        choices=DEFAULT_COUNTERFACTUAL_SCRIPTS,
        default=DEFAULT_BRANCH_ACTION_ORACLE_BASE_SCRIPT,
    )
    parser.add_argument(
        "--candidate-actions",
        default=",".join(DEFAULT_BRANCH_ACTION_ORACLE_CANDIDATE_ACTIONS),
        help="Comma-separated first actions to branch from each ambiguous state.",
    )
    parser.add_argument(
        "--target-labels",
        default=",".join(DEFAULT_BRANCH_ACTION_ORACLE_TARGET_LABELS),
        help="Comma-separated logged labels eligible for branch points.",
    )
    parser.add_argument(
        "--max-branch-points-per-seed",
        type=int,
        default=DEFAULT_BRANCH_ACTION_ORACLE_POINTS_PER_SEED,
    )
    parser.add_argument(
        "--branch-selection-policy",
        choices=(
            DEFAULT_BRANCH_ACTION_ORACLE_SELECTION_POLICY,
            MODE_BALANCED_BRANCH_ACTION_ORACLE_SELECTION_POLICY,
        ),
        default=DEFAULT_BRANCH_ACTION_ORACLE_SELECTION_POLICY,
    )
    parser.add_argument("--min-branch-tick", type=int, default=0)
    parser.add_argument("--min-oracle-changed-action-count", type=int, default=1)
    parser.add_argument("--min-terminal-alive-gain-total", type=int, default=1)
    parser.add_argument(
        "--no-verify-replay",
        action="store_true",
        help="Skip deterministic replay verification for each forced branch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-v66-branch-action-oracle-audit.json"),
        help="Output JSON report path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = build_branch_action_oracle_audit_report(
            seeds=_parse_ints(args.seeds),
            ticks=int(args.ticks),
            base_script=str(args.base_script),
            continuation_script=str(args.continuation_script),
            candidate_actions=_parse_strings(args.candidate_actions),
            target_labels=_parse_strings(args.target_labels),
            max_branch_points_per_seed=int(args.max_branch_points_per_seed),
            branch_selection_policy=str(args.branch_selection_policy),
            min_branch_tick=int(args.min_branch_tick),
            min_oracle_changed_action_count=int(args.min_oracle_changed_action_count),
            min_terminal_alive_gain_total=int(args.min_terminal_alive_gain_total),
            verify_replay=not bool(args.no_verify_replay),
        )
        write_branch_action_oracle_audit_report(report, args.output)
    except (OSError, ValueError, BranchActionOracleAuditError) as exc:
        raise SystemExit(f"failed to audit branch action oracle: {exc}") from exc

    aggregate = report["aggregate"]  # type: ignore[index]
    acceptance = report["acceptance"]  # type: ignore[index]
    print(f"branch_action_oracle_audit={args.output}")
    print(f"schema_version={MIND_V3_BRANCH_ACTION_ORACLE_AUDIT_SCHEMA_VERSION}")
    print(f"branch_point_count={aggregate['branch_point_count']}")  # type: ignore[index]
    print(f"action_run_count={aggregate['action_run_count']}")  # type: ignore[index]
    print(
        "oracle_changed_action_count="
        f"{aggregate['oracle_changed_action_count']}"  # type: ignore[index]
    )
    print(
        "terminal_alive_gain_total_vs_logged="
        f"{aggregate['terminal_alive_gain_total_vs_logged']}"  # type: ignore[index]
    )
    print(f"replay_verified={aggregate['replay_verified']}")  # type: ignore[index]
    print(
        "diagnostic_acceptance_passed="
        f"{acceptance['diagnostic_acceptance_passed']}"  # type: ignore[index]
    )
    print(f"blocker_count={len(acceptance['blockers'])}")  # type: ignore[index]


def _parse_ints(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise BranchActionOracleAuditError("integer list must not be empty")
    return values


def _parse_strings(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise BranchActionOracleAuditError("string list must not be empty")
    return values


if __name__ == "__main__":
    main()
