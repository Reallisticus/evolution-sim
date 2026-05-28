from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.branch_constrained_planning_audit import (
    MIND_V3_BRANCH_CONSTRAINED_PLANNING_AUDIT_SCHEMA_VERSION,
    BranchConstrainedPlanningAuditError,
    build_branch_constrained_planning_audit_report,
    load_json_report,
    write_branch_constrained_planning_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Mind v3 v95 simulator-in-the-loop constrained "
            "planning diagnostic."
        )
    )
    parser.add_argument(
        "--strict-branch-action-oracle-labels",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v91-failure-frontier-branch-action-oracle-labels.json"
        ),
    )
    parser.add_argument(
        "--branch-sequence-continuation-scorer",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v94-branch-sequence-continuation-scorer.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v95-branch-constrained-planning-audit.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        labels = load_json_report(args.strict_branch_action_oracle_labels)
        sequence = load_json_report(args.branch_sequence_continuation_scorer)
        report = build_branch_constrained_planning_audit_report(
            strict_branch_action_oracle_labels=labels,
            branch_sequence_continuation_scorer_report=sequence,
        )
        write_branch_constrained_planning_audit_report(report, args.output)
    except (OSError, ValueError, BranchConstrainedPlanningAuditError) as exc:
        raise SystemExit(
            f"failed to build branch constrained planning audit: {exc}"
        ) from exc

    acceptance = report["acceptance"]  # type: ignore[index]
    best = acceptance["best_rule_for_diagnostics"]  # type: ignore[index]
    print(f"branch_constrained_planning_audit={args.output}")
    print(
        "schema_version="
        f"{MIND_V3_BRANCH_CONSTRAINED_PLANNING_AUDIT_SCHEMA_VERSION}"
    )
    print(f"best_rule={best.get('rule')}")  # type: ignore[union-attr]
    print(
        "best_rule_dominant_predicted_action_share="
        f"{best.get('dominant_predicted_action_share')}"  # type: ignore[union-attr]
    )
    print(
        "best_rule_mean_target_local_score_delta="
        f"{best.get('mean_target_local_score_delta')}"  # type: ignore[union-attr]
    )
    print(
        "target_local_utility_cost_vs_v94="
        f"{best.get('target_local_utility_cost_vs_v94')}"  # type: ignore[union-attr]
    )
    print(
        "v95_constrained_planning_diagnostic_accepted="
        f"{acceptance['v95_constrained_planning_diagnostic_accepted']}"  # type: ignore[index]
    )
    print(f"blocker_count={len(acceptance['strict_blockers'])}")  # type: ignore[index]


if __name__ == "__main__":
    main()
