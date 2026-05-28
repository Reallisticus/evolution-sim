from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.branch_mode_objective_audit import (
    MIND_V3_BRANCH_MODE_OBJECTIVE_AUDIT_SCHEMA_VERSION,
    BranchModeObjectiveAuditError,
    build_branch_mode_objective_audit_report,
    load_json_report,
    write_branch_mode_objective_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Mind v3 branch archive coverage and mode-balanced "
            "objective diagnostic from branch-action oracle labels."
        )
    )
    parser.add_argument(
        "--branch-action-oracle-labels",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v87-branch-action-oracle-labels-public-history-model.json"
        ),
    )
    parser.add_argument(
        "--source-branch-action-oracle-audit",
        type=Path,
        default=None,
        help="Optional source branch-action oracle audit JSON for selection coverage.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-v90-branch-mode-objective-audit.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        labels = load_json_report(args.branch_action_oracle_labels)
        audit = (
            load_json_report(args.source_branch_action_oracle_audit)
            if args.source_branch_action_oracle_audit is not None
            else None
        )
        report = build_branch_mode_objective_audit_report(
            labels,
            source_branch_action_oracle_audit=audit,
        )
        write_branch_mode_objective_audit_report(report, args.output)
    except (OSError, ValueError, BranchModeObjectiveAuditError) as exc:
        raise SystemExit(
            f"failed to build branch mode objective audit: {exc}"
        ) from exc

    coverage = report["coverage"]  # type: ignore[index]
    support = report["support"]  # type: ignore[index]
    option = support["option_mode"]  # type: ignore[index]
    acceptance = report["acceptance"]  # type: ignore[index]
    print(f"branch_mode_objective_audit={args.output}")
    print(f"schema_version={MIND_V3_BRANCH_MODE_OBJECTIVE_AUDIT_SCHEMA_VERSION}")
    print(f"label_count={coverage['label_count']}")  # type: ignore[index]
    print(
        "option_mode_best_accuracy="
        f"{option['best_mode_accuracy']}"  # type: ignore[index]
    )
    print(
        "option_mode_dominant_prediction_share="
        f"{option['dominant_prediction_mode_share']}"  # type: ignore[index]
    )
    print(
        "mode_balanced_objective_diagnostic_passed="
        f"{acceptance['mode_balanced_objective_diagnostic_passed']}"  # type: ignore[index]
    )
    print(f"blocker_count={len(acceptance['blockers'])}")  # type: ignore[index]


if __name__ == "__main__":
    main()
