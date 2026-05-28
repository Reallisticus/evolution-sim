from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V124_MANIFEST_PATH,
)
from evolution_sim.mind.first_recovery_active_coverage_archive import (
    DEFAULT_ARCHIVE_ROWS_OUTPUT_PATH as DEFAULT_V123_ARCHIVE_ROWS_PATH,
)
from evolution_sim.mind.first_recovery_candidate_set_shadow_execution import (
    DEFAULT_V115_ARCHIVE_ROWS_PATH,
)
from evolution_sim.mind.first_recovery_refreshed_candidate_public_feature_surface import (
    DEFAULT_FEATURE_ROWS_OUTPUT_PATH as DEFAULT_V131_FEATURE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V131_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_refreshed_surface_blocker_slice_audit import (
    DEFAULT_DETAIL_ROWS_OUTPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    build_first_recovery_refreshed_surface_blocker_slice_audit,
    write_first_recovery_refreshed_surface_blocker_slice_audit_report,
    write_first_recovery_refreshed_surface_blocker_slice_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only Mind v3 v132 first-recovery refreshed "
            "surface blocker slice audit."
        )
    )
    parser.add_argument("--v131-report", type=Path, default=DEFAULT_V131_REPORT_PATH)
    parser.add_argument(
        "--v131-feature-rows",
        type=Path,
        default=DEFAULT_V131_FEATURE_ROWS_PATH,
    )
    parser.add_argument("--v124-manifest", type=Path, default=DEFAULT_V124_MANIFEST_PATH)
    parser.add_argument(
        "--v115-archive-rows",
        type=Path,
        default=DEFAULT_V115_ARCHIVE_ROWS_PATH,
    )
    parser.add_argument(
        "--v123-archive-rows",
        type=Path,
        default=DEFAULT_V123_ARCHIVE_ROWS_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--detail-rows-output",
        type=str,
        default=DEFAULT_DETAIL_ROWS_OUTPUT_PATH,
        help=(
            "Optional diagnostics-only JSONL blocker detail rows output. "
            "Use an empty string to skip writing detail rows."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    detail_rows_output = (
        Path(args.detail_rows_output)
        if str(args.detail_rows_output).strip()
        else None
    )
    try:
        build = build_first_recovery_refreshed_surface_blocker_slice_audit(
            v131_report_path=args.v131_report,
            v131_feature_rows_path=args.v131_feature_rows,
            v124_manifest_path=args.v124_manifest,
            v115_archive_rows_path=args.v115_archive_rows,
            v123_archive_rows_path=args.v123_archive_rows,
        )
        write_first_recovery_refreshed_surface_blocker_slice_audit_report(
            build,
            output_path=args.output,
        )
        if detail_rows_output is not None:
            write_first_recovery_refreshed_surface_blocker_slice_rows(
                build,
                output_path=detail_rows_output,
            )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run first-recovery refreshed surface blocker slice "
            f"audit: {exc}"
        ) from exc
    _print_summary(build.report, args.output, detail_rows_output)


def _print_summary(
    report: Mapping[str, object],
    output_path: Path,
    detail_rows_output: Path | None,
) -> None:
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source_integrity"))
    extraction = _mapping(report.get("slice_extraction"))
    aggregate = _mapping(report.get("blocker_class_aggregate"))
    recommendation = _mapping(report.get("recommendation"))
    auth = _mapping(report.get("authorization_block"))
    print(f"first_recovery_refreshed_surface_blocker_slice_audit={output_path}")
    if detail_rows_output is not None:
        print(f"refreshed_surface_blocker_slices={detail_rows_output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"v131_metric_failures={source.get('v131_metric_failures')}")
    print(
        "heldout_validation_test_branch_count="
        f"{extraction.get('heldout_validation_test_branch_count')}"
    )
    print(
        "fixture_open_mind_v3_branch_count="
        f"{extraction.get('fixture_open_mind_v3_branch_count')}"
    )
    print(f"blocker_class_counts={aggregate.get('counts_by_blocker_class')}")
    print(f"recommendation={recommendation.get('next_step')}")
    print(
        "downstream_shadow_scorer_allowed="
        f"{auth.get('downstream_shadow_scorer_allowed')}"
    )
    print(
        "v113_readiness_rerun_allowed="
        f"{auth.get('v113_readiness_rerun_allowed')}"
    )
    print(f"training_executed={auth.get('training_executed')}")
    print(
        "runtime_policy_change_recommended="
        f"{auth.get('runtime_policy_change_recommended')}"
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
