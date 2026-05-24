from __future__ import annotations

import argparse
import glob
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_branch_oracle_audit import (
    MIND_V3_FIRST_RECOVERY_BRANCH_ORACLE_AUDIT_SCHEMA_VERSION,
    FirstRecoveryBranchOracleAuditError,
    build_first_recovery_branch_oracle_audit_report,
    write_first_recovery_branch_oracle_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only Mind v3 first-recovery branch-oracle "
            "support audit."
        )
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
        help="Optional previous branch-oracle audit for provenance/comparison.",
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
        "--max-targets-per-seed-source",
        type=int,
        default=1,
        help="Bounded target selection count per seed/source when not exhaustive.",
    )
    parser.add_argument(
        "--include-open",
        action="store_true",
        help="Also include bounded open-world rows. Fixture rows are selected first.",
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Branch all reconstructed v107 first-recovery rows.",
    )
    parser.add_argument(
        "--skip-replay-verification",
        action="store_true",
        help="Skip deterministic branch replay verification. Intended for debugging.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v108-first-recovery-branch-oracle-audit.json"
        ),
        help="Output JSON report path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trajectory_paths = _trajectory_paths(args.trajectory, args.trajectory_glob)
    try:
        report = build_first_recovery_branch_oracle_audit_report(
            v107_report_path=args.v107_report,
            rollout_context_report_path=args.rollout_context_report,
            baseline_report_path=args.baseline_report,
            previous_branch_oracle_audit_path=args.previous_branch_oracle_audit,
            trajectory_paths=trajectory_paths,
            trajectory_glob_patterns=tuple(args.trajectory_glob),
            max_targets_per_seed_source=int(args.max_targets_per_seed_source),
            include_open=bool(args.include_open),
            exhaustive=bool(args.exhaustive),
            verify_replay=not bool(args.skip_replay_verification),
        )
        write_first_recovery_branch_oracle_audit_report(report, args.output)
    except (OSError, ValueError, FirstRecoveryBranchOracleAuditError) as exc:
        raise SystemExit(
            f"failed to audit first-recovery branch oracle support: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    classification = _mapping(report.get("classification"))
    evidence = _mapping(report.get("evidence"))
    trajectories = _mapping(evidence.get("trajectories"))
    alignment = _mapping(report.get("v107_row_alignment"))
    selection = _mapping(report.get("branch_target_selection"))
    points = _mapping(report.get("branch_points"))
    oracle = _mapping(report.get("outcome_oracle"))
    support = _mapping(report.get("heldout_support"))
    print(f"first_recovery_branch_oracle_audit={output_path}")
    print(
        "schema_version="
        f"{MIND_V3_FIRST_RECOVERY_BRANCH_ORACLE_AUDIT_SCHEMA_VERSION}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"loaded_trajectory_count={trajectories.get('loaded_path_count', 0)}")
    print(f"trajectory_load_failure_count={trajectories.get('load_failure_count', 0)}")
    print(
        "trajectory_malformed_record_count="
        f"{trajectories.get('malformed_record_count', 0)}"
    )
    print(
        "reconstructed_first_recovery_row_count="
        f"{alignment.get('reconstructed_first_recovery_row_count', 0)}"
    )
    print(f"row_count_matches_v107={alignment.get('row_count_matches_v107')}")
    print(f"selected_target_count={selection.get('selected_target_count', 0)}")
    print(
        "materialized_branch_point_count="
        f"{points.get('materialized_branch_point_count', 0)}"
    )
    print(f"branch_result_count={oracle.get('branch_result_count', 0)}")
    print(f"action_run_count={oracle.get('action_run_count', 0)}")
    print(f"replay_verified={oracle.get('replay_verified')}")
    print(
        "heuristic_action_source_count="
        f"{oracle.get('heuristic_action_source_count', 0)}"
    )
    print(f"dominant_oracle_action_share={oracle.get('dominant_oracle_action_share')}")
    print(f"heldout_support={support.get('answer')}")
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
