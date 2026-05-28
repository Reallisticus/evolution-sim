from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.branch_planner_distillation_audit import (
    MIND_V3_PLANNER_DISTILLATION_AUDIT_SCHEMA_VERSION,
    BranchPlannerDistillationAuditError,
    build_branch_planner_distillation_audit_report,
    load_json_report,
    write_branch_planner_distillation_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Mind v3 v96 planner-distillation runtime-feasibility "
            "diagnostic."
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
        "--branch-sequence-continuation-scorer",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v94-branch-sequence-continuation-scorer.json"
        ),
    )
    parser.add_argument(
        "--branch-constrained-planning-audit",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v95-branch-constrained-planning-audit.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v96-planner-distillation-runtime-feasibility.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        support_labels = load_json_report(args.support_branch_action_oracle_labels)
        strict_labels = load_json_report(args.strict_branch_action_oracle_labels)
        sequence = load_json_report(args.branch_sequence_continuation_scorer)
        constrained = load_json_report(args.branch_constrained_planning_audit)
        report = build_branch_planner_distillation_audit_report(
            support_branch_action_oracle_labels=support_labels,
            strict_branch_action_oracle_labels=strict_labels,
            branch_sequence_continuation_scorer_report=sequence,
            branch_constrained_planning_audit_report=constrained,
        )
        write_branch_planner_distillation_audit_report(report, args.output)
    except (OSError, ValueError, BranchPlannerDistillationAuditError) as exc:
        raise SystemExit(
            f"failed to build branch planner distillation audit: {exc}"
        ) from exc

    acceptance = report["acceptance"]  # type: ignore[index]
    best = acceptance["best_rule_for_diagnostics"]  # type: ignore[index]
    print(f"branch_planner_distillation_audit={args.output}")
    print(
        "schema_version="
        f"{MIND_V3_PLANNER_DISTILLATION_AUDIT_SCHEMA_VERSION}"
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
        "v96_runtime_feasibility_accepted="
        f"{acceptance['v96_runtime_feasibility_accepted']}"  # type: ignore[index]
    )
    print(
        "v97_full_strict_promotion_run_allowed="
        f"{acceptance['v97_full_strict_promotion_run_allowed']}"  # type: ignore[index]
    )
    print(f"blocker_count={len(acceptance['strict_blockers'])}")  # type: ignore[index]


if __name__ == "__main__":
    main()
