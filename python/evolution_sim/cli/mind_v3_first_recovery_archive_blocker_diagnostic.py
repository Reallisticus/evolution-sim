from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_archive_blocker_diagnostic import (
    DEFAULT_ARCHIVE_REPORT_PATH,
    DEFAULT_ARCHIVE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH,
    MIND_V3_FIRST_RECOVERY_ARCHIVE_BLOCKER_DIAGNOSTIC_SCHEMA_VERSION,
    FirstRecoveryArchiveBlockerDiagnosticError,
    build_first_recovery_archive_blocker_diagnostic,
    write_first_recovery_archive_blocker_diagnostic_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 v116 first-recovery archive "
            "blocker diagnostic over the existing v115 exhaustive archive."
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
        build = build_first_recovery_archive_blocker_diagnostic(
            archive_report_path=args.archive_report,
            archive_rows_path=args.archive_rows,
        )
        write_first_recovery_archive_blocker_diagnostic_report(
            build,
            output_path=args.output,
        )
    except (OSError, ValueError, FirstRecoveryArchiveBlockerDiagnosticError) as exc:
        raise SystemExit(
            "failed to build first-recovery archive blocker diagnostic: " f"{exc}"
        ) from exc
    _print_summary(build.report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    classification = _mapping(report.get("classification"))
    verification = _mapping(report.get("source_archive_verification"))
    stay = _mapping(report.get("stay_dominance_analysis"))
    resolution = _mapping(report.get("resolution_invalid_analysis"))
    recommendation = _mapping(report.get("recommendation"))
    print(f"first_recovery_archive_blocker_diagnostic={output_path}")
    print(
        "schema_version="
        f"{MIND_V3_FIRST_RECOVERY_ARCHIVE_BLOCKER_DIAGNOSTIC_SCHEMA_VERSION}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"verification_passed={verification.get('verification_passed')}")
    print(f"selected_target_count={verification.get('selected_target_count')}")
    print(f"archive_row_count={verification.get('archive_row_count_jsonl')}")
    print(f"unique_branch_id_count={verification.get('unique_branch_id_count')}")
    print(f"replay_verified={verification.get('replay_verified')}")
    print(
        "heuristic_action_source_count="
        f"{verification.get('heuristic_action_source_count')}"
    )
    print(f"trainable_leak_count={verification.get('trainable_leak_count')}")
    print(f"stay_oracle_count={stay.get('stay_branch_count')}")
    print(f"stay_oracle_share={stay.get('stay_branch_share')}")
    print(f"resolution_invalid_count={resolution.get('invalid_count')}")
    print(f"unexplained_resolution_invalid_count={resolution.get('unexplained_count')}")
    print(f"recommendation={recommendation.get('next_step')}")
    print(
        "v113_readiness_rerun_allowed="
        f"{recommendation.get('v113_readiness_rerun_allowed')}"
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
