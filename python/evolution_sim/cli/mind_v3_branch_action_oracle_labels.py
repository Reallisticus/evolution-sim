from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.branch_action_oracle_labels import (
    DEFAULT_BRANCH_ACTION_ORACLE_LABEL_MAX_DOMINANT_ACTION_SHARE,
    MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION,
    BranchActionOracleLabelError,
    build_branch_action_oracle_label_report,
    load_branch_action_oracle_audit_report,
    write_branch_action_oracle_label_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic policy-visible branch-action oracle labels "
            "from a replay-verified Mind v3 branch-action oracle audit."
        )
    )
    parser.add_argument(
        "--branch-action-oracle-audit",
        type=Path,
        default=Path("output/mind/mind-v3-v66-branch-action-oracle-audit.json"),
        help="Input branch-action oracle audit JSON path.",
    )
    parser.add_argument(
        "--allow-unaccepted-audit",
        action="store_true",
        help="Build labels even when the source audit failed diagnostic acceptance.",
    )
    parser.add_argument("--min-material-label-count", type=int, default=1)
    parser.add_argument(
        "--max-dominant-oracle-action-share",
        type=float,
        default=DEFAULT_BRANCH_ACTION_ORACLE_LABEL_MAX_DOMINANT_ACTION_SHARE,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-v67-branch-action-oracle-labels.json"),
        help="Output JSON report path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        source = load_branch_action_oracle_audit_report(
            args.branch_action_oracle_audit
        )
        report = build_branch_action_oracle_label_report(
            source,
            require_accepted_audit=not bool(args.allow_unaccepted_audit),
            min_material_label_count=int(args.min_material_label_count),
            max_dominant_oracle_action_share=float(
                args.max_dominant_oracle_action_share
            ),
        )
        write_branch_action_oracle_label_report(report, args.output)
    except (OSError, ValueError, BranchActionOracleLabelError) as exc:
        raise SystemExit(
            f"failed to build branch action oracle labels: {exc}"
        ) from exc

    aggregate = report["aggregate"]  # type: ignore[index]
    acceptance = report["acceptance"]  # type: ignore[index]
    print(f"branch_action_oracle_labels={args.output}")
    print(f"schema_version={MIND_V3_BRANCH_ACTION_ORACLE_LABEL_SCHEMA_VERSION}")
    print(f"label_count={aggregate['label_count']}")  # type: ignore[index]
    print(
        "material_oracle_gain_label_count="
        f"{aggregate['material_oracle_gain_label_count']}"  # type: ignore[index]
    )
    print(
        "dominant_oracle_action_share="
        f"{aggregate['dominant_oracle_action_share']}"  # type: ignore[index]
    )
    print(
        "terminal_alive_gain_total_vs_logged="
        f"{aggregate['terminal_alive_gain_total_vs_logged']}"  # type: ignore[index]
    )
    print(
        "label_archive_acceptance_passed="
        f"{acceptance['label_archive_acceptance_passed']}"  # type: ignore[index]
    )
    print(f"blocker_count={len(acceptance['blockers'])}")  # type: ignore[index]


if __name__ == "__main__":
    main()
