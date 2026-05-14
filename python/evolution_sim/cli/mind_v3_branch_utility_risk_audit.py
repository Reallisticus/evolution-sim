from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.branch_utility_risk_audit import (
    MIND_V3_BRANCH_UTILITY_RISK_AUDIT_SCHEMA_VERSION,
    BranchUtilityRiskAuditError,
    build_branch_utility_risk_audit_report,
    load_json_report,
    write_branch_utility_risk_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Mind v3 v92 catastrophe-sensitive branch utility "
            "risk diagnostic."
        )
    )
    parser.add_argument(
        "--branch-action-oracle-labels",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v91-failure-frontier-branch-action-oracle-labels.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-v92-branch-utility-risk-audit.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        labels = load_json_report(args.branch_action_oracle_labels)
        report = build_branch_utility_risk_audit_report(labels)
        write_branch_utility_risk_audit_report(report, args.output)
    except (OSError, ValueError, BranchUtilityRiskAuditError) as exc:
        raise SystemExit(
            f"failed to build branch utility risk audit: {exc}"
        ) from exc

    acceptance = report["acceptance"]  # type: ignore[index]
    best = acceptance["best_rule_for_diagnostics"]  # type: ignore[index]
    print(f"branch_utility_risk_audit={args.output}")
    print(f"schema_version={MIND_V3_BRANCH_UTILITY_RISK_AUDIT_SCHEMA_VERSION}")
    print(f"best_rule={best.get('rule')}")  # type: ignore[union-attr]
    print(
        "best_rule_mean_target_local_score_delta="
        f"{best.get('mean_target_local_score_delta')}"  # type: ignore[union-attr]
    )
    print(
        "best_rule_target_alive_delta_negative_count="
        f"{best.get('target_alive_delta_negative_count')}"  # type: ignore[union-attr]
    )
    print(
        "v92_catastrophe_sensitive_branch_utility_accepted="
        f"{acceptance['v92_catastrophe_sensitive_branch_utility_accepted']}"  # type: ignore[index]
    )
    print(f"blocker_count={len(acceptance['strict_blockers'])}")  # type: ignore[index]


if __name__ == "__main__":
    main()
