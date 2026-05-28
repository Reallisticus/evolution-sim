from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_oracle_tie_break_audit import (
    DEFAULT_ARCHIVE_REPORT_PATH,
    DEFAULT_ARCHIVE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH,
    MIND_V3_FIRST_RECOVERY_ORACLE_TIE_BREAK_AUDIT_SCHEMA_VERSION,
    FirstRecoveryOracleTieBreakAuditError,
    build_first_recovery_oracle_tie_break_audit,
    write_first_recovery_oracle_tie_break_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 v117 first-recovery oracle "
            "tie-break audit over the existing v115 archive."
        )
    )
    parser.add_argument(
        "--archive-report",
        type=Path,
        default=DEFAULT_ARCHIVE_REPORT_PATH,
        help="Input v115 expanded first-recovery branch archive JSON report.",
    )
    parser.add_argument(
        "--archive-rows",
        type=Path,
        default=DEFAULT_ARCHIVE_ROWS_PATH,
        help="Input v115 expanded first-recovery branch archive gzip JSONL rows.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON report path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        build = build_first_recovery_oracle_tie_break_audit(
            archive_report_path=args.archive_report,
            archive_rows_path=args.archive_rows,
        )
        write_first_recovery_oracle_tie_break_audit_report(
            build,
            output_path=args.output,
        )
    except (OSError, ValueError, FirstRecoveryOracleTieBreakAuditError) as exc:
        raise SystemExit(
            "failed to build first-recovery oracle tie-break audit: " f"{exc}"
        ) from exc
    _print_summary(build.report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    classification = _mapping(report.get("classification"))
    analysis = _mapping(report.get("objective_tie_break_analysis"))
    alternatives = _mapping(analysis.get("tie_neutral_stay_counts"))
    recommendation = _mapping(report.get("recommendation"))
    print(f"first_recovery_oracle_tie_break_audit={output_path}")
    print(
        "schema_version="
        f"{MIND_V3_FIRST_RECOVERY_ORACLE_TIE_BREAK_AUDIT_SCHEMA_VERSION}"
    )
    print(f"classification={classification.get('primary')}")
    print(
        "current_serialized_oracle_stay_count="
        f"{analysis.get('current_serialized_oracle_stay_count')}"
    )
    print(
        "current_serialized_oracle_stay_share="
        f"{analysis.get('current_serialized_oracle_stay_share')}"
    )
    print(
        "unique_objective_best_branch_count="
        f"{analysis.get('unique_objective_best_branch_count')}"
    )
    print(
        "multiple_objective_best_branch_count="
        f"{analysis.get('multiple_objective_best_branch_count')}"
    )
    print(f"tie_neutral_stay_counts={dict(alternatives)}")
    print(f"recommendation={recommendation.get('next_step')}")
    print(
        "v113_readiness_rerun_allowed="
        f"{recommendation.get('v113_readiness_rerun_allowed')}"
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
