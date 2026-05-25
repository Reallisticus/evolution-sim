from __future__ import annotations

import argparse
import glob
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_shadow_ranker import (
    DEFAULT_ARCHIVE_REPORT_PATH,
    DEFAULT_ARCHIVE_ROWS_PATH,
    DEFAULT_HISTORY_WINDOW,
    DEFAULT_OUTPUT_PATH,
    MIND_V3_FIRST_RECOVERY_SHADOW_RANKER_SCHEMA_VERSION,
    FirstRecoveryShadowRankerError,
    build_first_recovery_shadow_ranker,
    write_first_recovery_shadow_ranker_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 first-recovery shadow ranker report."
        )
    )
    parser.add_argument(
        "--archive-report",
        type=Path,
        default=DEFAULT_ARCHIVE_REPORT_PATH,
        help="Input v109 first-recovery branch archive JSON report.",
    )
    parser.add_argument(
        "--archive-rows",
        type=Path,
        default=DEFAULT_ARCHIVE_ROWS_PATH,
        help="Input v109 first-recovery branch archive gzip JSONL rows.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        default=[],
        help="Trajectory JSONL path for optional joined feature sets.",
    )
    parser.add_argument(
        "--trajectory-glob",
        action="append",
        default=[],
        help="Trajectory JSONL glob for optional joined feature sets.",
    )
    parser.add_argument(
        "--enable-observation-input",
        action="store_true",
        help="Enable optional joined public ecological observation-input ranker.",
    )
    parser.add_argument(
        "--enable-public-history",
        action="store_true",
        help="Enable optional same-agent public history-prefix ranker.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=DEFAULT_HISTORY_WINDOW,
        help="Number of same-agent prior public records for optional history features.",
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
    trajectory_paths = _trajectory_paths(args.trajectory, args.trajectory_glob)
    try:
        build = build_first_recovery_shadow_ranker(
            archive_report_path=args.archive_report,
            archive_rows_path=args.archive_rows,
            trajectory_paths=trajectory_paths,
            trajectory_glob_patterns=tuple(args.trajectory_glob),
            enable_observation_input=bool(args.enable_observation_input),
            enable_public_history=bool(args.enable_public_history),
            history_window=int(args.history_window),
        )
        write_first_recovery_shadow_ranker_report(build, output_path=args.output)
    except (OSError, ValueError, FirstRecoveryShadowRankerError) as exc:
        raise SystemExit(
            f"failed to build first-recovery shadow ranker report: {exc}"
        ) from exc
    _print_summary(build.report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    classification = _mapping(report.get("classification"))
    archive = _mapping(report.get("archive_summary"))
    current = _mapping(report.get("current_row_linear_ranker"))
    heldout = _mapping(current.get("heldout_summary"))
    action_distribution = _mapping(report.get("action_distribution"))
    unsupported = _mapping(report.get("unsupported_action_audit"))
    leakage = _mapping(report.get("leakage_audit"))
    recommendation = _mapping(report.get("research_recommendation"))
    print(f"first_recovery_shadow_ranker={output_path}")
    print(f"schema_version={MIND_V3_FIRST_RECOVERY_SHADOW_RANKER_SCHEMA_VERSION}")
    print(f"classification={classification.get('primary')}")
    print(f"archive_row_count={archive.get('loaded_archive_row_count', 0)}")
    print(f"branch_target_group_count={archive.get('branch_target_group_count', 0)}")
    print(f"current_row_answer={current.get('answer')}")
    print(f"leave_one_seed_answer={heldout.get('leave_one_seed_answer')}")
    print(f"fixture_open_answer={heldout.get('fixture_open_answer')}")
    print(f"seed29_answer={heldout.get('seed29_answer')}")
    print(
        "dominant_selected_action_share="
        f"{action_distribution.get('dominant_selected_action_share')}"
    )
    print(f"unsupported_action_rate={unsupported.get('unsupported_action_rate')}")
    print(f"leakage_count={leakage.get('leak_count')}")
    print(f"recommendation={recommendation.get('recommendation')}")
    print(f"missing_evidence_count={len(classification.get('missing_evidence', []))}")


def _trajectory_paths(explicit_paths: list[Path], patterns: list[str]) -> tuple[Path, ...]:
    paths = {Path(path) for path in explicit_paths}
    for pattern in patterns:
        paths.update(Path(path) for path in glob.glob(pattern))
    return tuple(sorted(paths, key=lambda path: str(path)))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
