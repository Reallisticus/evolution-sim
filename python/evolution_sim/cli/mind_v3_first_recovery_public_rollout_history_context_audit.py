from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V124_MANIFEST_PATH,
)
from evolution_sim.mind.first_recovery_public_rollout_history_context_audit import (
    DEFAULT_HISTORY_ROWS_OUTPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    build_first_recovery_public_rollout_history_context_audit,
    write_first_recovery_public_rollout_history_context_audit_report,
    write_first_recovery_public_rollout_history_context_rows,
)
from evolution_sim.mind.first_recovery_refreshed_surface_blocker_slice_audit import (
    DEFAULT_DETAIL_ROWS_OUTPUT_PATH as DEFAULT_V132_DETAIL_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V132_REPORT_PATH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only Mind v3 v133 first-recovery public "
            "rollout-history context feasibility audit."
        )
    )
    parser.add_argument("--v132-report", type=Path, default=DEFAULT_V132_REPORT_PATH)
    parser.add_argument(
        "--v132-detail-rows",
        type=Path,
        default=DEFAULT_V132_DETAIL_ROWS_PATH,
    )
    parser.add_argument("--v124-manifest", type=Path, default=DEFAULT_V124_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--history-rows-output",
        type=Path,
        default=DEFAULT_HISTORY_ROWS_OUTPUT_PATH,
        help=(
            "Optional diagnostics-only JSONL public history rows output. "
            "Use an empty string to skip writing history rows."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    history_rows_output = (
        args.history_rows_output
        if str(args.history_rows_output).strip()
        else None
    )
    try:
        build = build_first_recovery_public_rollout_history_context_audit(
            v132_report_path=args.v132_report,
            v132_detail_rows_path=args.v132_detail_rows,
            v124_manifest_path=args.v124_manifest,
        )
        write_first_recovery_public_rollout_history_context_audit_report(
            build,
            output_path=args.output,
        )
        if history_rows_output is not None:
            write_first_recovery_public_rollout_history_context_rows(
                build,
                output_path=history_rows_output,
            )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run first-recovery public rollout-history context "
            f"audit: {exc}"
        ) from exc
    _print_summary(build.report, args.output, history_rows_output)


def _print_summary(
    report: Mapping[str, object],
    output_path: Path,
    history_rows_output: Path | None,
) -> None:
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source_integrity"))
    availability = _mapping(report.get("public_history_availability"))
    variance = _mapping(report.get("within_branch_history_variance"))
    all_variance = _mapping(variance.get("all_blocker_rows"))
    contrast = _mapping(report.get("failed_vs_correct_history_contrast"))
    open_analysis = _mapping(report.get("open_fixture_collapse_analysis"))
    open_contrast = _mapping(open_analysis.get("collapse_vs_nonfailed_open_contrast"))
    recommendation = _mapping(report.get("recommendation"))
    auth = _mapping(report.get("authorization_block"))
    print(f"first_recovery_public_rollout_history_context_audit={output_path}")
    if history_rows_output is not None:
        print(f"public_rollout_history_context_rows={history_rows_output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"history_available_row_count={availability.get('history_available_row_count')}")
    print(f"history_missing_row_count={availability.get('history_missing_row_count')}")
    print(
        "all_blocker_varying_feature_path_count="
        f"{all_variance.get('varying_feature_path_count')}"
    )
    print(
        "failed_vs_correct_contrasting_feature_path_count="
        f"{contrast.get('contrasting_feature_path_count')}"
    )
    print(
        "open_collapse_contrasting_feature_path_count="
        f"{open_contrast.get('contrasting_feature_path_count')}"
    )
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
