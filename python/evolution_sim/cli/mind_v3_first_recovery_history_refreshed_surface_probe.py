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
from evolution_sim.mind.first_recovery_history_refreshed_surface_probe import (
    DEFAULT_HISTORY_REFRESHED_ROWS_OUTPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    build_first_recovery_history_refreshed_surface_probe,
    write_first_recovery_history_refreshed_surface_probe_report,
    write_first_recovery_history_refreshed_surface_rows,
)
from evolution_sim.mind.first_recovery_public_rollout_history_context_audit import (
    DEFAULT_HISTORY_ROWS_OUTPUT_PATH as DEFAULT_V133_HISTORY_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V133_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_refreshed_candidate_public_feature_surface import (
    DEFAULT_FEATURE_ROWS_OUTPUT_PATH as DEFAULT_V131_FEATURE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V131_REPORT_PATH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only Mind v3 v134 first-recovery history-"
            "refreshed candidate public feature surface probe."
        )
    )
    parser.add_argument("--v131-report", type=Path, default=DEFAULT_V131_REPORT_PATH)
    parser.add_argument(
        "--v131-feature-rows",
        type=Path,
        default=DEFAULT_V131_FEATURE_ROWS_PATH,
    )
    parser.add_argument("--v133-report", type=Path, default=DEFAULT_V133_REPORT_PATH)
    parser.add_argument(
        "--v133-history-rows",
        type=Path,
        default=DEFAULT_V133_HISTORY_ROWS_PATH,
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
        "--feature-rows-output",
        type=Path,
        default=DEFAULT_HISTORY_REFRESHED_ROWS_OUTPUT_PATH,
        help=(
            "Optional diagnostics-only JSONL history-refreshed feature rows "
            "output. Use an empty string to skip writing rows."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    feature_rows_output = (
        args.feature_rows_output if str(args.feature_rows_output).strip() else None
    )
    try:
        build = build_first_recovery_history_refreshed_surface_probe(
            v131_report_path=args.v131_report,
            v131_feature_rows_path=args.v131_feature_rows,
            v133_report_path=args.v133_report,
            v133_history_rows_path=args.v133_history_rows,
            v124_manifest_path=args.v124_manifest,
            v115_archive_rows_path=args.v115_archive_rows,
            v123_archive_rows_path=args.v123_archive_rows,
        )
        write_first_recovery_history_refreshed_surface_probe_report(
            build,
            output_path=args.output,
        )
        if feature_rows_output is not None:
            write_first_recovery_history_refreshed_surface_rows(
                build,
                output_path=feature_rows_output,
            )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run first-recovery history-refreshed surface probe: "
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
    summary = _mapping(report.get("feature_surface_summary"))
    comparison = _mapping(report.get("probe_comparison"))
    fixture_open = _mapping(report.get("fixture_open_evaluation"))
    open_group = _mapping(_mapping(fixture_open.get("groups")).get("open_mind_v3"))
    dominant = _mapping(open_group.get("dominant_predicted_action"))
    unsupported = _mapping(report.get("unsupported_action_audit"))
    material = _mapping(report.get("material_exact_match_recall"))
    metric_gate = _mapping(report.get("metric_gate"))
    auth = _mapping(report.get("authorization_block"))
    print(f"first_recovery_history_refreshed_surface_probe={output_path}")
    if feature_rows_output is not None:
        print(f"history_refreshed_surface_rows={feature_rows_output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"candidate_feature_row_count={summary.get('candidate_feature_row_count')}")
    print(f"heldout_accuracy={comparison.get('heldout_accuracy')}")
    print(
        "heldout_delta_vs_action_only="
        f"{comparison.get('heldout_delta_vs_action_only')}"
    )
    print(
        "heldout_delta_vs_action_order="
        f"{comparison.get('heldout_delta_vs_action_order')}"
    )
    print(f"fixture_open_dominant_predicted_action={dominant}")
    print(
        "unsupported_action_count="
        f"{unsupported.get('unsupported_action_count')}"
    )
    print(
        "material_exact_match_recall="
        f"{material.get('repaired_label_material_gain_exact_match_recall')}"
    )
    print(f"metric_failures={metric_gate.get('failures')}")
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
