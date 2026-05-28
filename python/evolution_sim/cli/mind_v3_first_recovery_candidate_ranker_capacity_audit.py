from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V124_MANIFEST_PATH,
)
from evolution_sim.mind.first_recovery_active_coverage_archive import (
    DEFAULT_ARCHIVE_ROWS_OUTPUT_PATH as DEFAULT_V123_ARCHIVE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V123_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_candidate_ranker_capacity_audit import (
    DEFAULT_OUTPUT_PATH,
    build_first_recovery_candidate_ranker_capacity_audit,
    write_first_recovery_candidate_ranker_capacity_audit_report,
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
            "Run the diagnostics-only Mind v3 v128 first-recovery "
            "candidate-ranker capacity audit."
        )
    )
    parser.add_argument("--v127-report", type=Path, default=DEFAULT_V127_REPORT_PATH)
    parser.add_argument(
        "--v127-predictions",
        type=Path,
        default=DEFAULT_V127_PREDICTIONS_PATH,
    )
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        build = build_first_recovery_candidate_ranker_capacity_audit(
            v127_report_path=args.v127_report,
            v127_predictions_path=args.v127_predictions,
            v124_manifest_path=args.v124_manifest,
            v115_report_path=args.v115_report,
            v115_archive_rows_path=args.v115_archive_rows,
            v123_report_path=args.v123_report,
            v123_archive_rows_path=args.v123_archive_rows,
        )
        write_first_recovery_candidate_ranker_capacity_audit_report(
            build,
            output_path=args.output,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"failed to run first-recovery candidate-ranker capacity audit: {exc}"
        ) from exc
    _print_summary(build.report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source_integrity"))
    gate = _mapping(report.get("metric_gate"))
    feature_variance = _mapping(report.get("feature_variance"))
    comparisons = _mapping(report.get("probe_comparisons"))
    action_distribution = _mapping(report.get("action_distribution"))
    dominant = _mapping(action_distribution.get("dominant_predicted_action"))
    recommendation = _mapping(report.get("recommendation"))
    print(f"first_recovery_candidate_ranker_capacity_audit={output_path}")
    print(f"classification={classification.get('primary')}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"candidate_row_count={source.get('candidate_row_count')}")
    print(f"positive_row_count={source.get('positive_row_count')}")
    print(f"negative_row_count={source.get('negative_row_count')}")
    print(
        "non_action_public_fields_branch_constant="
        f"{feature_variance.get('non_action_public_fields_branch_constant')}"
    )
    print(
        "heldout_delta_vs_action_only="
        f"{comparisons.get('heldout_delta_vs_action_only')}"
    )
    print(f"dominant_predicted_action={dominant.get('action')}")
    print(f"dominant_predicted_action_share={dominant.get('share')}")
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
