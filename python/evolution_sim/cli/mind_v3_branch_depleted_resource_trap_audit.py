from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.branch_depleted_resource_trap_audit import (
    MIND_V3_DEPLETED_RESOURCE_TRAP_AUDIT_SCHEMA_VERSION,
    BranchDepletedResourceTrapAuditError,
    build_depleted_resource_trap_audit_report,
    load_json_report,
    write_depleted_resource_trap_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Mind v3 v93 depleted-resource trap support diagnostic."
        )
    )
    parser.add_argument(
        "--support-branch-action-oracle-labels",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v93-depleted-resource-trap-support-labels.json"
        ),
    )
    parser.add_argument(
        "--strict-branch-action-oracle-labels",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v91-failure-frontier-branch-action-oracle-labels.json"
        ),
    )
    parser.add_argument(
        "--support-branch-action-oracle-audit",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--v92-branch-utility-risk-audit",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v93-depleted-resource-trap-audit.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        support_labels = load_json_report(args.support_branch_action_oracle_labels)
        strict_labels = load_json_report(args.strict_branch_action_oracle_labels)
        support_audit = (
            load_json_report(args.support_branch_action_oracle_audit)
            if args.support_branch_action_oracle_audit is not None
            else None
        )
        v92_audit = (
            load_json_report(args.v92_branch_utility_risk_audit)
            if args.v92_branch_utility_risk_audit is not None
            else None
        )
        report = build_depleted_resource_trap_audit_report(
            support_branch_action_oracle_labels=support_labels,
            strict_branch_action_oracle_labels=strict_labels,
            support_branch_action_oracle_audit=support_audit,
            v92_branch_utility_risk_audit=v92_audit,
        )
        write_depleted_resource_trap_audit_report(report, args.output)
    except (OSError, ValueError, BranchDepletedResourceTrapAuditError) as exc:
        raise SystemExit(
            f"failed to build depleted resource trap audit: {exc}"
        ) from exc

    coverage = report["coverage"]  # type: ignore[index]
    acceptance = report["acceptance"]  # type: ignore[index]
    best = acceptance["best_rule_for_diagnostics"]  # type: ignore[index]
    print(f"branch_depleted_resource_trap_audit={args.output}")
    print(f"schema_version={MIND_V3_DEPLETED_RESOURCE_TRAP_AUDIT_SCHEMA_VERSION}")
    print(f"support_trap_row_count={coverage['support_trap_row_count']}")  # type: ignore[index]
    print(f"best_rule={best.get('rule')}")  # type: ignore[union-attr]
    print(
        "best_rule_mean_target_local_score_delta="
        f"{best.get('mean_target_local_score_delta')}"  # type: ignore[union-attr]
    )
    print(
        "seed_41_catastrophe_avoided="
        f"{best.get('seed_41_catastrophe_avoided')}"  # type: ignore[union-attr]
    )
    print(
        "v93_depleted_resource_trap_diagnostic_accepted="
        f"{acceptance['v93_depleted_resource_trap_diagnostic_accepted']}"  # type: ignore[index]
    )
    print(f"blocker_count={len(acceptance['strict_blockers'])}")  # type: ignore[index]


if __name__ == "__main__":
    main()
