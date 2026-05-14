from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.branch_action_oracle_labels import (
    MIND_V3_BRANCH_CONTINUATION_ARCHIVE_SCORER_SCHEMA_VERSION,
    BranchActionOracleLabelError,
    build_branch_continuation_archive_scorer_report,
    load_branch_action_oracle_label_report,
    write_branch_continuation_archive_scorer_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Mind v3 branch-continuation archive scorer diagnostic "
            "from an existing branch-action oracle label report."
        )
    )
    parser.add_argument(
        "--branch-action-oracle-labels",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v87-branch-action-oracle-labels-public-history-model.json"
        ),
        help="Input branch-action oracle label report JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v89-branch-continuation-archive-scorer.json"
        ),
        help="Output JSON report path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        source = load_branch_action_oracle_label_report(
            args.branch_action_oracle_labels
        )
        report = build_branch_continuation_archive_scorer_report(source)
        write_branch_continuation_archive_scorer_report(report, args.output)
    except (OSError, ValueError, BranchActionOracleLabelError) as exc:
        raise SystemExit(
            f"failed to run branch continuation archive scorer: {exc}"
        ) from exc

    probe = report["support_probes"]["branch_continuation_archive_scorer"]  # type: ignore[index]
    acceptance = report["acceptance"]  # type: ignore[index]
    print(f"branch_continuation_archive_scorer={args.output}")
    print(
        "schema_version="
        f"{MIND_V3_BRANCH_CONTINUATION_ARCHIVE_SCORER_SCHEMA_VERSION}"
    )
    print(f"best_accuracy={probe['best_accuracy']}")  # type: ignore[index]
    print(f"best_correct_count={probe['best_correct_count']}")  # type: ignore[index]
    print(
        "material_support_accuracy_floor="
        f"{probe['material_support_accuracy_floor']}"  # type: ignore[index]
    )
    print(
        "branch_continuation_archive_scorer_gate_passed="
        f"{acceptance['branch_continuation_archive_scorer_gate_passed']}"  # type: ignore[index]
    )
    print(f"blocker_count={len(acceptance['blockers'])}")  # type: ignore[index]


if __name__ == "__main__":
    main()
