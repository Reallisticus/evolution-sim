from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_archive_blocker_diagnostic import (
    DEFAULT_ARCHIVE_REPORT_PATH,
    DEFAULT_ARCHIVE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V116_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_inspector_bundle import (
    DEFAULT_OUTPUT_PATH,
    MIND_V3_FIRST_RECOVERY_INSPECTOR_BUNDLE_SCHEMA_VERSION,
    build_first_recovery_inspector_bundle,
    write_first_recovery_inspector_bundle,
)
from evolution_sim.mind.first_recovery_oracle_tie_break_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V117_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_rare_action_coverage_targeting import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V121_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_repaired_label_contract_audit import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V119_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V119_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_repaired_label_split_support_feasibility import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V120_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_tie_aware_label_repair import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V118_REPORT_PATH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the read-only Mind v3 first-recovery inspector bundle from "
            "existing v115-v121 diagnostic artifacts."
        )
    )
    parser.add_argument(
        "--archive-report",
        type=Path,
        default=DEFAULT_ARCHIVE_REPORT_PATH,
        help="Input v115 first-recovery branch archive JSON report.",
    )
    parser.add_argument(
        "--archive-rows",
        type=Path,
        default=DEFAULT_ARCHIVE_ROWS_PATH,
        help="Input v115 first-recovery branch archive JSONL rows.",
    )
    parser.add_argument(
        "--v116-report",
        type=Path,
        default=DEFAULT_V116_REPORT_PATH,
        help="Input v116 archive blocker diagnostic report.",
    )
    parser.add_argument(
        "--v117-report",
        type=Path,
        default=DEFAULT_V117_REPORT_PATH,
        help="Input v117 oracle tie-break audit report.",
    )
    parser.add_argument(
        "--v118-report",
        type=Path,
        default=DEFAULT_V118_REPORT_PATH,
        help="Input v118 tie-aware label repair report.",
    )
    parser.add_argument(
        "--v119-report",
        type=Path,
        default=DEFAULT_V119_REPORT_PATH,
        help="Input v119 repaired-label contract audit report.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_V119_MANIFEST_PATH,
        help="Input v119 repaired-label JSONL manifest.",
    )
    parser.add_argument(
        "--v120-report",
        type=Path,
        default=DEFAULT_V120_REPORT_PATH,
        help="Input v120 split/support feasibility report.",
    )
    parser.add_argument(
        "--v121-report",
        type=Path,
        default=DEFAULT_V121_REPORT_PATH,
        help="Input v121 rare-action coverage targeting report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output inspector bundle JSON path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        build = build_first_recovery_inspector_bundle(
            archive_report_path=args.archive_report,
            archive_rows_path=args.archive_rows,
            v116_report_path=args.v116_report,
            v117_report_path=args.v117_report,
            v118_report_path=args.v118_report,
            v119_report_path=args.v119_report,
            manifest_path=args.manifest,
            v120_report_path=args.v120_report,
            v121_report_path=args.v121_report,
        )
        write_first_recovery_inspector_bundle(build, output_path=args.output)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"failed to build first-recovery inspector bundle: {exc}"
        ) from exc
    _print_summary(build.report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    classification = _mapping(report.get("classification"))
    integrity = _mapping(report.get("source_integrity"))
    blocker_summary = _mapping(report.get("blocker_summary"))
    rare = _mapping(
        blocker_summary.get("v121_no_recoverable_rare_attack_candidates")
    )
    print(f"first_recovery_inspector_bundle={output_path}")
    print(
        "schema_version="
        f"{MIND_V3_FIRST_RECOVERY_INSPECTOR_BUNDLE_SCHEMA_VERSION}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"source_integrity_passed={integrity.get('passed')}")
    print(f"branch_count={report.get('branch_count')}")
    print(f"rare_valid_candidate_counts={rare.get('valid_candidate_counts')}")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
