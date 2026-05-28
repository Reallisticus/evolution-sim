from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V124_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V124_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_active_coverage_archive import (
    DEFAULT_ARCHIVE_ROWS_OUTPUT_PATH as DEFAULT_V123_ARCHIVE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V123_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_candidate_public_feature_surface import (
    DEFAULT_FEATURE_ROWS_OUTPUT_PATH as DEFAULT_V129_FEATURE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V129_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_candidate_ranker_capacity_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V128_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_candidate_set_shadow_execution import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V127_REPORT_PATH,
    DEFAULT_PREDICTIONS_OUTPUT_PATH as DEFAULT_V127_PREDICTIONS_PATH,
    DEFAULT_V115_ARCHIVE_ROWS_PATH,
    DEFAULT_V115_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_observation_candidate_context import (
    DEFAULT_CONTEXT_ROWS_OUTPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    build_first_recovery_observation_candidate_context,
    write_first_recovery_observation_candidate_context_report,
    write_first_recovery_observation_candidate_context_rows,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only Mind v3 v130 first-recovery "
            "observation-time candidate context extractor."
        )
    )
    parser.add_argument("--v129-report", type=Path, default=DEFAULT_V129_REPORT_PATH)
    parser.add_argument(
        "--v129-feature-rows",
        type=Path,
        default=DEFAULT_V129_FEATURE_ROWS_PATH,
    )
    parser.add_argument("--v128-report", type=Path, default=DEFAULT_V128_REPORT_PATH)
    parser.add_argument("--v127-report", type=Path, default=DEFAULT_V127_REPORT_PATH)
    parser.add_argument(
        "--v127-predictions",
        type=Path,
        default=DEFAULT_V127_PREDICTIONS_PATH,
    )
    parser.add_argument("--v124-report", type=Path, default=DEFAULT_V124_REPORT_PATH)
    parser.add_argument("--v124-manifest", type=Path, default=DEFAULT_V124_MANIFEST_PATH)
    parser.add_argument("--v115-report", type=Path, default=DEFAULT_V115_REPORT_PATH)
    parser.add_argument(
        "--v115-archive-rows",
        type=Path,
        default=DEFAULT_V115_ARCHIVE_ROWS_PATH,
    )
    parser.add_argument("--v123-report", type=Path, default=DEFAULT_V123_REPORT_PATH)
    parser.add_argument(
        "--v123-archive-rows",
        type=Path,
        default=DEFAULT_V123_ARCHIVE_ROWS_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--context-rows-output",
        type=Path,
        default=DEFAULT_CONTEXT_ROWS_OUTPUT_PATH,
        help=(
            "Optional diagnostics-only JSONL candidate context rows output. "
            "Use an empty string to skip writing context rows."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    context_rows_output = (
        args.context_rows_output
        if str(args.context_rows_output).strip()
        else None
    )
    try:
        build = build_first_recovery_observation_candidate_context(
            v129_report_path=args.v129_report,
            v129_feature_rows_path=args.v129_feature_rows,
            v128_report_path=args.v128_report,
            v127_report_path=args.v127_report,
            v127_predictions_path=args.v127_predictions,
            v124_report_path=args.v124_report,
            v124_manifest_path=args.v124_manifest,
            v115_report_path=args.v115_report,
            v115_archive_rows_path=args.v115_archive_rows,
            v123_report_path=args.v123_report,
            v123_archive_rows_path=args.v123_archive_rows,
        )
        write_first_recovery_observation_candidate_context_report(
            build,
            output_path=args.output,
        )
        if context_rows_output is not None:
            write_first_recovery_observation_candidate_context_rows(
                build,
                output_path=context_rows_output,
            )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run first-recovery observation candidate context extractor: "
            f"{exc}"
        ) from exc
    _print_summary(build.report, args.output, context_rows_output)


def _print_summary(
    report: Mapping[str, object],
    output_path: Path,
    context_rows_output: Path | None,
) -> None:
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source_integrity"))
    availability = _mapping(report.get("candidate_context_availability"))
    variance = _mapping(report.get("within_branch_variance"))
    target_variance = _mapping(variance.get("target_resource_neighborhood_variance"))
    forbidden = _mapping(report.get("forbidden_field_scan"))
    recommendation = _mapping(report.get("recommendation"))
    print(f"first_recovery_observation_candidate_context={output_path}")
    if context_rows_output is not None:
        print(f"observation_candidate_context_rows={context_rows_output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"branch_count={availability.get('branch_count')}")
    print(f"candidate_row_count={availability.get('candidate_row_count')}")
    print(
        "target_resource_neighborhood_varying_path_count="
        f"{target_variance.get('varying_path_count')}"
    )
    print(
        "missing_public_observation_fields="
        f"{availability.get('missing_public_observation_fields')}"
    )
    print(
        "forbidden_feature_path_count="
        f"{forbidden.get('forbidden_feature_path_count')}"
    )
    print(
        "downstream_shadow_scorer_allowed="
        f"{recommendation.get('downstream_shadow_scorer_allowed')}"
    )
    print(
        "v113_readiness_rerun_allowed="
        f"{recommendation.get('v113_readiness_rerun_allowed')}"
    )
    print(f"training_executed={recommendation.get('training_executed')}")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
