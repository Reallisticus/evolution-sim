from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_archive_blocker_diagnostic import (
    DEFAULT_ARCHIVE_REPORT_PATH,
    DEFAULT_ARCHIVE_ROWS_PATH,
)
from evolution_sim.mind.first_recovery_rare_action_coverage_targeting import (
    DEFAULT_OUTPUT_PATH,
    MIND_V3_FIRST_RECOVERY_RARE_ACTION_COVERAGE_TARGETING_SCHEMA_VERSION,
    build_first_recovery_rare_action_coverage_targeting,
    write_first_recovery_rare_action_coverage_targeting_report,
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
            "Build the diagnostics-only Mind v3 v121 first-recovery rare-action "
            "coverage targeting report."
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
        help="Input v120 repaired-label split/support feasibility report.",
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
        build = build_first_recovery_rare_action_coverage_targeting(
            archive_report_path=args.archive_report,
            archive_rows_path=args.archive_rows,
            v118_report_path=args.v118_report,
            v119_report_path=args.v119_report,
            manifest_path=args.manifest,
            v120_report_path=args.v120_report,
        )
        write_first_recovery_rare_action_coverage_targeting_report(
            build,
            output_path=args.output,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to build first-recovery rare-action coverage targeting "
            f"report: {exc}"
        ) from exc
    _print_summary(build.report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source_integrity"))
    support = _mapping(report.get("current_rare_action_support"))
    search = _mapping(report.get("candidate_search"))
    recommendation = _mapping(report.get("recommendation"))
    per_action = _mapping(search.get("per_action"))
    candidate_counts = {
        action: _mapping(payload).get("valid_candidate_count")
        for action, payload in per_action.items()
    }
    print(f"first_recovery_rare_action_coverage_targeting={output_path}")
    print(
        "schema_version="
        f"{MIND_V3_FIRST_RECOVERY_RARE_ACTION_COVERAGE_TARGETING_SCHEMA_VERSION}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"rare_action_counts={support.get('repaired_action_counts')}")
    print(f"valid_candidate_counts={candidate_counts}")
    print(f"next_step={recommendation.get('next_step')}")
    print(
        "v113_readiness_rerun_allowed="
        f"{recommendation.get('v113_readiness_rerun_allowed')}"
    )
    print(
        "downstream_shadow_scorer_allowed="
        f"{recommendation.get('downstream_shadow_scorer_allowed')}"
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
