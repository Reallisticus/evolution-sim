from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_data_coverage_diagnostic import (
    DEFAULT_ARCHIVE_REPORT_PATH,
    DEFAULT_ARCHIVE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PUBLIC_SIGNAL_AUDIT_PATH,
    DEFAULT_STATE_ACTION_READINESS_AUDIT_PATH,
    MIND_V3_FIRST_RECOVERY_DATA_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION,
    FirstRecoveryDataCoverageDiagnosticError,
    build_first_recovery_data_coverage_diagnostic,
    write_first_recovery_data_coverage_diagnostic_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 first-recovery data coverage "
            "diagnostic for v109 selection bias before expanded archive replay."
        )
    )
    parser.add_argument(
        "--archive-report",
        type=Path,
        default=DEFAULT_ARCHIVE_REPORT_PATH,
        help="Input v109 first-recovery branch archive JSON report.",
    )
    parser.add_argument(
        "--archive-rows",
        type=Path,
        default=DEFAULT_ARCHIVE_ROWS_PATH,
        help="Input v109 first-recovery branch archive gzip JSONL rows.",
    )
    parser.add_argument(
        "--public-signal-audit",
        type=Path,
        default=DEFAULT_PUBLIC_SIGNAL_AUDIT_PATH,
        help="Input v112 first-recovery public signal audit JSON report.",
    )
    parser.add_argument(
        "--state-action-readiness-audit",
        type=Path,
        default=DEFAULT_STATE_ACTION_READINESS_AUDIT_PATH,
        help="Input v113 first-recovery state-action readiness audit JSON report.",
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
        build = build_first_recovery_data_coverage_diagnostic(
            archive_report_path=args.archive_report,
            archive_rows_path=args.archive_rows,
            public_signal_audit_path=args.public_signal_audit,
            state_action_readiness_audit_path=args.state_action_readiness_audit,
        )
        write_first_recovery_data_coverage_diagnostic_report(
            build,
            output_path=args.output,
        )
    except (OSError, ValueError, FirstRecoveryDataCoverageDiagnosticError) as exc:
        raise SystemExit(
            "failed to build first-recovery data coverage diagnostic: " f"{exc}"
        ) from exc
    _print_summary(build.report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    classification = _mapping(report.get("classification"))
    reconstruction = _mapping(report.get("reconstruction_coverage"))
    selected_vs = _mapping(report.get("selected_vs_reconstructed_coverage"))
    expansion = _mapping(report.get("expansion_budget"))
    leakage = _mapping(report.get("leakage_guard"))
    recommendation = _mapping(report.get("recommendation"))
    print(f"first_recovery_data_coverage_diagnostic={output_path}")
    print(
        "schema_version="
        f"{MIND_V3_FIRST_RECOVERY_DATA_COVERAGE_DIAGNOSTIC_SCHEMA_VERSION}"
    )
    print(f"classification={classification.get('primary')}")
    print(
        "reconstructed_target_count="
        f"{reconstruction.get('reconstructed_first_recovery_row_count')}"
    )
    print(f"selected_target_count={reconstruction.get('selected_target_count')}")
    print(f"selection_share={reconstruction.get('selection_share')}")
    print(f"selection_bias_detected={selected_vs.get('bias_detected')}")
    print(f"biased_dimensions={selected_vs.get('biased_dimensions')}")
    print(f"recommend_expanded_archive={expansion.get('recommend_expanded_archive')}")
    print(f"leakage_count={leakage.get('leakage_count')}")
    print(f"recommendation={recommendation.get('next_step')}")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
