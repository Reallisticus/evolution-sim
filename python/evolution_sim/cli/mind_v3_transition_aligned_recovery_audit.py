from __future__ import annotations

import argparse
import glob
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.transition_aligned_recovery_audit import (
    MIND_V3_TRANSITION_ALIGNED_RECOVERY_AUDIT_SCHEMA_VERSION,
    TransitionAlignedRecoveryAuditError,
    build_transition_aligned_recovery_audit_report,
    write_transition_aligned_recovery_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a diagnostics-only Mind v3 transition-aligned first-recovery "
            "support and mask-drift audit."
        )
    )
    parser.add_argument(
        "--v106-report",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v106-post-carrion-rollout-context-coverage-audit.json"
        ),
        help="Input v106 post-carrion rollout-context coverage audit report.",
    )
    parser.add_argument(
        "--recovery-action-target-audit",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v69-recovery-action-target-alignment-audit.json"
        ),
        help="Input v69 recovery action target alignment audit report.",
    )
    parser.add_argument(
        "--branch-oracle-audit",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v69-branch-action-oracle-audit-expanded-actions.json"
        ),
        help="Input v69 expanded-action branch oracle audit report.",
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
        help="Input v4 baseline same-shape search report.",
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
        "--min-overlap-present-share",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-v107-transition-aligned-recovery-audit.json"),
        help="Output JSON report path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trajectory_paths = _trajectory_paths(args.trajectory, args.trajectory_glob)
    try:
        report = build_transition_aligned_recovery_audit_report(
            v106_report_path=args.v106_report,
            recovery_action_target_audit_path=args.recovery_action_target_audit,
            branch_oracle_audit_path=args.branch_oracle_audit,
            rollout_context_report_path=args.rollout_context_report,
            baseline_report_path=args.baseline_report,
            trajectory_paths=trajectory_paths,
            trajectory_glob_patterns=tuple(args.trajectory_glob),
            min_overlap_present_share=float(args.min_overlap_present_share),
        )
        write_transition_aligned_recovery_audit_report(report, args.output)
    except (OSError, ValueError, TransitionAlignedRecoveryAuditError) as exc:
        raise SystemExit(
            f"failed to audit transition-aligned recovery support: {exc}"
        ) from exc

    _print_report_summary(report, args.output)


def _print_report_summary(
    report: Mapping[str, object],
    output_path: Path,
) -> None:
    classification = _mapping(report.get("classification"))
    evidence = _mapping(report.get("evidence"))
    trajectories = _mapping(evidence.get("trajectories"))
    recovery = _mapping(report.get("transition_aligned_first_recovery"))
    overlap = _mapping(report.get("branch_oracle_overlap"))
    legality = _mapping(report.get("action_support_legality"))
    drift = _mapping(report.get("mask_drift_root_cause"))
    heldout = _mapping(report.get("heldout_support"))
    print(f"transition_aligned_recovery_audit={output_path}")
    print(
        "schema_version="
        f"{MIND_V3_TRANSITION_ALIGNED_RECOVERY_AUDIT_SCHEMA_VERSION}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"loaded_trajectory_count={trajectories.get('loaded_path_count', 0)}")
    print(f"trajectory_load_failure_count={trajectories.get('load_failure_count', 0)}")
    print(
        "trajectory_malformed_record_count="
        f"{trajectories.get('malformed_record_count', 0)}"
    )
    print(
        "first_recovery_row_count="
        f"{recovery.get('constructible_first_recovery_row_count', 0)}"
    )
    print(
        "branch_oracle_matched_branch_result_count="
        f"{overlap.get('matched_branch_result_count', 0)}"
    )
    print(f"action_support_legality={legality.get('answer')}")
    print(f"mask_drift_root_cause={drift.get('answer')}")
    print(f"heldout_support={heldout.get('answer')}")
    print(f"missing_evidence_count={len(classification.get('missing_evidence', []))}")


def _trajectory_paths(
    explicit_paths: list[Path],
    patterns: list[str],
) -> tuple[Path, ...]:
    paths = {Path(path) for path in explicit_paths}
    for pattern in patterns:
        paths.update(Path(path) for path in glob.glob(pattern))
    return tuple(sorted(paths, key=lambda path: str(path)))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
