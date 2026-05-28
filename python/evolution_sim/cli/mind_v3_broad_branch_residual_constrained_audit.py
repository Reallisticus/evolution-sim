from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.broad_branch_residual_constrained_audit import (
    build_broad_branch_residual_constrained_audit_report,
    load_json_report,
    write_broad_branch_residual_constrained_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the v100 constrained broad-branch residual diagnostic from "
            "the replay-verified v99 broad branch oracle report."
        )
    )
    parser.add_argument(
        "--v99-report",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v99-broad-branch-residual-oracle-audit.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v100-broad-branch-residual-constrained-audit.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_broad_branch_residual_constrained_audit_report(
        v99_broad_branch_residual_oracle_report=load_json_report(args.v99_report)
    )
    write_broad_branch_residual_constrained_audit_report(report, args.output)
    best = report["acceptance"]["best_rule_for_diagnostics"]
    print(f"v100_broad_branch_residual_constrained_audit={args.output}")
    print(f"schema_version={report['schema_version']}")
    print(
        "v100_broad_branch_residual_constrained_diagnostic_accepted="
        f"{report['v100_broad_branch_residual_constrained_diagnostic_accepted']}"
    )
    print(f"blocker_count={report['blocker_count']}")
    if isinstance(best, dict):
        print(f"best_rule={best['rule']}")
        print(
            "dominant_predicted_action_share="
            f"{best['dominant_predicted_action_share']}"
        )
        print(
            "mean_target_local_score_delta="
            f"{best['mean_target_local_score_delta']}"
        )
        print(
            "safe_non_logged_override_count="
            f"{best['safe_non_logged_override_count']}"
        )


if __name__ == "__main__":
    main()
