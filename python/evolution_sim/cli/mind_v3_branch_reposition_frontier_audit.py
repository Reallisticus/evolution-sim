from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.branch_reposition_frontier_audit import (
    MIND_V3_REPOSITION_FRONTIER_AUDIT_SCHEMA_VERSION,
    BranchRepositionFrontierAuditError,
    build_reposition_frontier_audit_report,
    load_json_report,
    write_reposition_frontier_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the Mind v3 v91 failure-frontier reposition decomposition "
            "and branch-utility diagnostic."
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
        "--source-branch-action-oracle-audit",
        type=Path,
        default=None,
        help="Optional source branch-action oracle audit JSON for selection coverage.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-v91-reposition-frontier-audit.json"),
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
        report = build_reposition_frontier_audit_report(
            labels,
            source_branch_action_oracle_audit=audit,
        )
        write_reposition_frontier_audit_report(report, args.output)
    except (OSError, ValueError, BranchRepositionFrontierAuditError) as exc:
        raise SystemExit(
            f"failed to build branch reposition frontier audit: {exc}"
        ) from exc

    coverage = report["coverage"]  # type: ignore[index]
    decomposition = report["reposition_decomposition"]  # type: ignore[index]
    utility = report["branch_utility"]  # type: ignore[index]
    acceptance = report["acceptance"]  # type: ignore[index]
    best = decomposition["best_decoder"]  # type: ignore[index]
    utility_delta = utility["target_local_score_delta_summary"]  # type: ignore[index]
    print(f"branch_reposition_frontier_audit={args.output}")
    print(f"schema_version={MIND_V3_REPOSITION_FRONTIER_AUDIT_SCHEMA_VERSION}")
    print(f"label_count={coverage['label_count']}")  # type: ignore[index]
    print(
        "reposition_multi_move_label_count="
        f"{coverage['reposition_multi_move_label_count']}"  # type: ignore[index]
    )
    print(f"best_reposition_decoder={best['decoder']}")  # type: ignore[index]
    print(f"best_reposition_accuracy={best['accuracy']}")  # type: ignore[index]
    print(
        "predicted_utility_mean_target_local_score_delta="
        f"{utility_delta['mean']}"  # type: ignore[index]
    )
    print(
        "v91_reposition_frontier_diagnostic_accepted="
        f"{acceptance['v91_reposition_frontier_diagnostic_accepted']}"  # type: ignore[index]
    )
    print(f"blocker_count={len(acceptance['blockers'])}")  # type: ignore[index]


if __name__ == "__main__":
    main()
