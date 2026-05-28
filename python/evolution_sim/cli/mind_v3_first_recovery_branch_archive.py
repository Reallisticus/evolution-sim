from __future__ import annotations

import argparse
import glob
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_branch_archive import (
    DEFAULT_MAX_FIXTURE_TARGETS_PER_SEED,
    MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION,
    FirstRecoveryBranchArchiveError,
    build_first_recovery_branch_archive,
    default_archive_rows_path,
    write_first_recovery_branch_archive_outputs,
)
from evolution_sim.mind.first_recovery_branch_oracle_audit import (
    FirstRecoveryBranchOracleAuditError,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 first-recovery branch archive."
        )
    )
    parser.add_argument(
        "--v108-report",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v108-first-recovery-branch-oracle-audit.json"
        ),
        help="Input v108 first-recovery branch-oracle audit report.",
    )
    parser.add_argument(
        "--v107-report",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v107-transition-aligned-recovery-audit.json"
        ),
        help="Input v107 transition-aligned first-recovery audit report.",
    )
    parser.add_argument(
        "--rollout-context-report",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v5-rollout-context-search-80-120-diagnostic.json"
        ),
        help="Input v5 rollout-context controller search report.",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("output/mind/mind-v3-v4-baseline-search-80-120.json"),
        help="Input v4 baseline search report.",
    )
    parser.add_argument(
        "--previous-branch-oracle-audit",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v69-branch-action-oracle-audit-expanded-actions.json"
        ),
        help="Optional previous branch-oracle audit for provenance.",
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
        default=[],
        help="Glob of trajectory JSONL paths. May be supplied more than once.",
    )
    parser.add_argument(
        "--max-fixture-targets-per-seed",
        type=int,
        default=DEFAULT_MAX_FIXTURE_TARGETS_PER_SEED,
        help="Stratified fixture target cap per seed when not exhaustive.",
    )
    parser.add_argument(
        "--skip-open",
        action="store_true",
        help="Skip open-world rows. Defaults to including every open row.",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Branch every reconstructed v107 first-recovery row.",
    )
    parser.add_argument(
        "--skip-replay-verification",
        action="store_true",
        help="Skip deterministic replay verification. Intended for debugging.",
    )
    parser.add_argument(
        "--archive-rows-output",
        type=Path,
        default=None,
        help="Output gzip JSONL archive path. Defaults beside --output.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v109-first-recovery-branch-archive.json"
        ),
        help="Output JSON report path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trajectory_paths = _trajectory_paths(args.trajectory, args.trajectory_glob)
    archive_rows_path = (
        args.archive_rows_output
        if args.archive_rows_output is not None
        else default_archive_rows_path(args.output)
    )
    try:
        build = build_first_recovery_branch_archive(
            v108_report_path=args.v108_report,
            v107_report_path=args.v107_report,
            rollout_context_report_path=args.rollout_context_report,
            baseline_report_path=args.baseline_report,
            previous_branch_oracle_audit_path=args.previous_branch_oracle_audit,
            trajectory_paths=trajectory_paths,
            trajectory_glob_patterns=tuple(args.trajectory_glob),
            max_fixture_targets_per_seed=int(args.max_fixture_targets_per_seed),
            include_open=not bool(args.skip_open),
            exhaustive=bool(args.exhaustive),
            verify_replay=not bool(args.skip_replay_verification),
            archive_rows_path=archive_rows_path,
        )
        write_first_recovery_branch_archive_outputs(
            build,
            output_path=args.output,
            archive_rows_path=archive_rows_path,
        )
    except (
        OSError,
        ValueError,
        FirstRecoveryBranchArchiveError,
        FirstRecoveryBranchOracleAuditError,
    ) as exc:
        raise SystemExit(
            f"failed to build first-recovery branch archive: {exc}"
        ) from exc
    _print_summary(build.report, args.output, archive_rows_path)


def _print_summary(
    report: Mapping[str, object],
    output_path: Path,
    archive_rows_path: Path,
) -> None:
    classification = _mapping(report.get("classification"))
    reconstruction = _mapping(report.get("row_reconstruction"))
    selection = _mapping(report.get("target_selection"))
    summary = _mapping(report.get("branch_archive_summary"))
    oracle = _mapping(report.get("oracle_label_summary"))
    readiness = _mapping(report.get("learnability_readiness"))
    open_rows = _mapping(report.get("open_row_summary"))
    print(f"first_recovery_branch_archive={output_path}")
    print(f"archive_rows={archive_rows_path}")
    print(
        "schema_version="
        f"{MIND_V3_FIRST_RECOVERY_BRANCH_ARCHIVE_SCHEMA_VERSION}"
    )
    print(f"classification={classification.get('primary')}")
    print(
        "reconstructed_first_recovery_row_count="
        f"{reconstruction.get('reconstructed_first_recovery_row_count', 0)}"
    )
    print(f"row_count_matches_v107={reconstruction.get('row_count_matches_v107')}")
    print(f"row_count_matches_v108={reconstruction.get('row_count_matches_v108')}")
    print(f"selected_target_count={selection.get('selected_target_count', 0)}")
    print(f"open_selected_row_count={selection.get('open_selected_row_count', 0)}")
    print(f"archive_row_count={summary.get('archive_row_count', 0)}")
    print(f"branch_result_count={summary.get('branch_result_count', 0)}")
    print(f"action_run_count={summary.get('action_run_count', 0)}")
    print(f"replay_verified={summary.get('replay_verified')}")
    print(
        "heuristic_action_source_count="
        f"{summary.get('heuristic_action_source_count', 0)}"
    )
    print(f"dominant_oracle_action_share={oracle.get('dominant_oracle_action_share')}")
    print(f"open_rows={open_rows.get('answer')}")
    print(f"learnability_readiness={readiness.get('answer')}")
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
