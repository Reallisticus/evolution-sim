from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.recovery_action_target_alignment import (
    RecoveryActionTargetAlignmentError,
    build_recovery_action_target_alignment_report,
    write_recovery_action_target_alignment_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a diagnostics-only audit of Mind v3 recovery residual "
            "action-target alignment across every held-out trajectory row."
        )
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="Input v65 Mind v3 neural residual artifact.",
    )
    parser.add_argument(
        "--distill-report",
        type=Path,
        required=True,
        help="Input v65 carrion recovery distillation report.",
    )
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        required=True,
        help="Input v65 carrion recovery distillation evaluation report.",
    )
    parser.add_argument(
        "--split-report",
        type=Path,
        required=True,
        help="Input v65 leakage-safe carrion recovery split report.",
    )
    parser.add_argument(
        "--residual-audit",
        type=Path,
        required=True,
        help="Input v68 residual counter calibration audit report.",
    )
    parser.add_argument(
        "--oracle-audit",
        type=Path,
        default=None,
        help=(
            "Optional branch-action oracle audit report. Missing or sparse "
            "oracle coverage is reported but does not fail this diagnostics audit."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v69-recovery-action-target-alignment-audit.json"
        ),
        help="Output action-target alignment audit report path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = build_recovery_action_target_alignment_report(
            artifact_path=args.artifact,
            distill_report_path=args.distill_report,
            evaluation_report_path=args.evaluation_report,
            split_report_path=args.split_report,
            residual_audit_path=args.residual_audit,
            oracle_audit_path=args.oracle_audit,
        )
        write_recovery_action_target_alignment_report(report, args.output)
    except (OSError, ValueError, RecoveryActionTargetAlignmentError) as exc:
        raise SystemExit(
            f"failed to audit recovery action target alignment: {exc}"
        ) from exc

    _print_report_summary(report, args.output)


def _print_report_summary(
    report: Mapping[str, object],
    output_path: Path,
) -> None:
    classification = _mapping(report.get("classification"))
    scoring = _mapping(report.get("heldout_scoring"))
    windows = _mapping(report.get("windows"))
    all_records = _mapping(
        _mapping(_mapping(windows.get("all_records")).get("by_outcome_class")).get(
            "all"
        )
    )
    recovery = _mapping(
        _mapping(_mapping(windows.get("recovery_phase")).get("by_outcome_class")).get(
            "all"
        )
    )
    oracle = _mapping(report.get("oracle_comparison"))
    print(f"recovery_action_target_alignment_audit={output_path}")
    print(f"classification={classification.get('primary')}")
    print(f"decision_row_count={scoring.get('decision_row_count', 0)}")
    print(
        "all_records_margin_ignored_would_change_count="
        f"{all_records.get('margin_ignored_would_change_count', 0)}"
    )
    print(
        "recovery_phase_survivor_minus_failure_rank_delta="
        f"{recovery.get('survivor_minus_failure_rank_delta')}"
    )
    print(
        "recovery_phase_extra_eat_pressure="
        f"{recovery.get('extra_eat_pressure_counts', {})}"
    )
    print(f"oracle_matched_row_count={oracle.get('matched_row_count', 0)}")
    print(f"oracle_unmatched_row_count={oracle.get('unmatched_row_count', 0)}")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
