from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_active_coverage_archive import (
    DEFAULT_ARCHIVE_ROWS_OUTPUT_PATH,
    DEFAULT_MAX_EVALUATED_BRANCHES,
    DEFAULT_MAX_SOURCE_RECORDS_PER_ACTION,
    DEFAULT_MAX_TRAJECTORIES,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_ROLLOUT_CONTEXT_REPORT_PATH,
    DEFAULT_TRAJECTORY_GLOB,
    DEFAULT_V122_RERUN_CANDIDATES_OUTPUT_PATH,
    DEFAULT_V122_RERUN_OUTPUT_PATH,
    build_first_recovery_active_coverage_archive,
    write_first_recovery_active_coverage_archive_outputs,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 v123 first-recovery active "
            "coverage archive for rare attack support."
        )
    )
    parser.add_argument("--v119-report", type=Path, default=DEFAULT_V119_REPORT_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_V119_MANIFEST_PATH)
    parser.add_argument("--v120-report", type=Path, default=DEFAULT_V120_REPORT_PATH)
    parser.add_argument("--v121-report", type=Path, default=DEFAULT_V121_REPORT_PATH)
    parser.add_argument(
        "--rollout-context-report",
        type=Path,
        default=DEFAULT_ROLLOUT_CONTEXT_REPORT_PATH,
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        default=[],
        help="Trajectory JSONL path. May be supplied more than once.",
    )
    parser.add_argument(
        "--trajectory-glob",
        action="append",
        default=[DEFAULT_TRAJECTORY_GLOB],
        help="Trajectory JSONL glob. May be supplied more than once.",
    )
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=DEFAULT_MAX_TRAJECTORIES,
    )
    parser.add_argument(
        "--max-source-records-per-action",
        type=int,
        default=DEFAULT_MAX_SOURCE_RECORDS_PER_ACTION,
    )
    parser.add_argument(
        "--max-evaluated-branches",
        type=int,
        default=DEFAULT_MAX_EVALUATED_BRANCHES,
    )
    parser.add_argument(
        "--skip-replay-verification",
        action="store_true",
        help="Skip replay verification. Intended only for local debugging.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--archive-rows-output",
        type=Path,
        default=DEFAULT_ARCHIVE_ROWS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--v122-rerun-output",
        type=Path,
        default=DEFAULT_V122_RERUN_OUTPUT_PATH,
    )
    parser.add_argument(
        "--v122-candidates-output",
        type=Path,
        default=DEFAULT_V122_RERUN_CANDIDATES_OUTPUT_PATH,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        build = build_first_recovery_active_coverage_archive(
            v119_report_path=args.v119_report,
            manifest_path=args.manifest,
            v120_report_path=args.v120_report,
            v121_report_path=args.v121_report,
            rollout_context_report_path=args.rollout_context_report,
            trajectory_paths=tuple(args.trajectory),
            trajectory_glob_patterns=tuple(args.trajectory_glob),
            max_trajectories=args.max_trajectories,
            max_source_records_per_action=args.max_source_records_per_action,
            max_evaluated_branches=args.max_evaluated_branches,
            verify_replay=not bool(args.skip_replay_verification),
        )
        write_first_recovery_active_coverage_archive_outputs(
            build,
            output_path=args.output,
            archive_rows_output_path=args.archive_rows_output,
            v122_rerun_output_path=args.v122_rerun_output,
            v122_candidates_output_path=args.v122_candidates_output,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"failed to build first-recovery active coverage archive: {exc}"
        ) from exc
    _print_summary(build.report, args.output, args.archive_rows_output)


def _print_summary(
    report: Mapping[str, object],
    output_path: Path,
    rows_path: Path,
) -> None:
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source_integrity"))
    v122 = _mapping(report.get("v122_rerun"))
    recommendation = _mapping(report.get("recommendation"))
    print(f"first_recovery_active_coverage_archive={output_path}")
    print(f"archive_rows={rows_path}")
    print(f"classification={classification.get('primary')}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"generated_branch_count={report.get('generated_branch_count')}")
    print(f"generated_archive_row_count={report.get('generated_archive_row_count')}")
    print(f"accepted_by_v122_candidate_counts={v122.get('found_candidate_counts')}")
    print(
        "would_clear_v120_rare_action_limitation_if_accepted="
        f"{recommendation.get('would_clear_v120_rare_action_limitation_if_accepted')}"
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
