from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_repaired_label_split_support_feasibility import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V119_MANIFEST_PATH,
    DEFAULT_V119_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_SCHEMA_VERSION,
    build_first_recovery_repaired_label_split_support_feasibility,
    write_first_recovery_repaired_label_split_support_feasibility_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 v120 first-recovery repaired-label "
            "split/support feasibility audit."
        )
    )
    parser.add_argument(
        "--v119-report",
        type=Path,
        default=DEFAULT_V119_REPORT_PATH,
        help="Input v119 repaired-label contract audit JSON report.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_V119_MANIFEST_PATH,
        help="Input v119 repaired-label JSONL manifest.",
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
        build = build_first_recovery_repaired_label_split_support_feasibility(
            v119_report_path=args.v119_report,
            manifest_path=args.manifest,
        )
        write_first_recovery_repaired_label_split_support_feasibility_report(
            build,
            output_path=args.output,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to build first-recovery repaired-label split support "
            f"feasibility audit: {exc}"
        ) from exc
    _print_summary(build.report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source_integrity"))
    support = _mapping(report.get("total_repaired_label_support"))
    scarcity = _mapping(report.get("scarcity_analysis"))
    recommendation = _mapping(report.get("recommendation"))
    print(f"first_recovery_repaired_label_split_support_feasibility={output_path}")
    print(
        "schema_version="
        f"{MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_SPLIT_SUPPORT_FEASIBILITY_SCHEMA_VERSION}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"manifest_row_count={source.get('manifest_row_count')}")
    print(f"repaired_action_counts={support.get('repaired_action_counts')}")
    print(
        "train2_validation1_test1_feasible="
        f"{scarcity.get('train2_validation1_test1_feasible_by_total_support')}"
    )
    print(
        "rare_action_additional_needed="
        f"{scarcity.get('rare_action_additional_needed_for_train2_validation1_test1')}"
    )
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
