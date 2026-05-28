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
    DEFAULT_FEATURE_ROWS_OUTPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    build_first_recovery_candidate_public_feature_surface,
    write_first_recovery_candidate_public_feature_rows,
    write_first_recovery_candidate_public_feature_surface_report,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only Mind v3 v129 first-recovery "
            "candidate-conditioned public feature surface audit."
        )
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
        "--feature-rows-output",
        type=Path,
        default=DEFAULT_FEATURE_ROWS_OUTPUT_PATH,
        help=(
            "Optional diagnostics-only JSONL feature rows output. Use an empty "
            "string to skip writing feature rows."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    feature_rows_output = (
        args.feature_rows_output
        if str(args.feature_rows_output).strip()
        else None
    )
    try:
        build = build_first_recovery_candidate_public_feature_surface(
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
        write_first_recovery_candidate_public_feature_surface_report(
            build,
            output_path=args.output,
        )
        if feature_rows_output is not None:
            write_first_recovery_candidate_public_feature_rows(
                build,
                output_path=feature_rows_output,
            )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run first-recovery candidate public feature surface audit: "
            f"{exc}"
        ) from exc
    _print_summary(build.report, args.output, feature_rows_output)


def _print_summary(
    report: Mapping[str, object],
    output_path: Path,
    feature_rows_output: Path | None,
) -> None:
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source_integrity"))
    gate = _mapping(report.get("metric_gate"))
    variance = _mapping(_mapping(report.get("within_branch_feature_variance")).get(
        "candidate_public_features"
    ))
    inventory = _mapping(report.get("candidate_specific_public_signal_inventory"))
    comparisons = _mapping(report.get("probe_comparisons"))
    recommendation = _mapping(report.get("recommendation"))
    print(f"first_recovery_candidate_public_feature_surface={output_path}")
    if feature_rows_output is not None:
        print(f"candidate_public_feature_rows={feature_rows_output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"candidate_row_count={source.get('candidate_row_count')}")
    print(
        "target_resource_neighborhood_varying_path_count="
        f"{variance.get('target_resource_neighborhood_varying_path_count')}"
    )
    print(
        "candidate_specific_target_resource_neighborhood_fields_present="
        f"{inventory.get('candidate_specific_target_resource_neighborhood_fields_present')}"
    )
    print(
        "heldout_delta_vs_action_only="
        f"{comparisons.get('heldout_delta_vs_action_only')}"
    )
    print(f"metric_failures={gate.get('failures')}")
    print(
        "downstream_shadow_scorer_allowed="
        f"{recommendation.get('downstream_shadow_scorer_allowed')}"
    )
    print(
        "v113_readiness_rerun_allowed="
        f"{recommendation.get('v113_readiness_rerun_allowed')}"
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
