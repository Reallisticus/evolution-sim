from __future__ import annotations

import argparse
import glob
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.post_carrion_rollout_context_audit import (
    MIND_V3_POST_CARRION_ROLLOUT_CONTEXT_AUDIT_SCHEMA_VERSION,
    PostCarrionRolloutContextAuditError,
    build_post_carrion_rollout_context_audit_report,
    write_post_carrion_rollout_context_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a diagnostics-only post-carrion rollout-context coverage and "
            "fixture-failure audit for the Mind v3 v105 rollout-context checkpoint."
        )
    )
    parser.add_argument(
        "--rollout-context-report",
        "--v105-report",
        dest="rollout_context_report",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v5-rollout-context-search-80-120-diagnostic.json"
        ),
        help=(
            "Input v5 rollout-context controller search report. --v105-report "
            "is accepted as a compatibility alias."
        ),
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=Path("output/mind/mind-v3-v4-baseline-search-80-120.json"),
        help="Input v4 baseline same-shape search report.",
    )
    parser.add_argument(
        "--v64-rollout-context-audit",
        type=Path,
        default=Path("output/mind/mind-v3-v64-rollout-context-audit.json"),
        help="Input v64 rollout-context lookup audit report.",
    )
    parser.add_argument(
        "--v69-recovery-action-target-audit",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v69-recovery-action-target-alignment-audit.json"
        ),
        help="Input v69 recovery action-target alignment audit report.",
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
        "--min-broad-post-carrion-context-share",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--min-branch-oracle-exact-match-share",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v106-post-carrion-rollout-context-coverage-audit.json"
        ),
        help="Output JSON report path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trajectory_paths = _trajectory_paths(args.trajectory, args.trajectory_glob)
    try:
        report = build_post_carrion_rollout_context_audit_report(
            rollout_context_report_path=args.rollout_context_report,
            baseline_report_path=args.baseline_report,
            v64_rollout_context_audit_path=args.v64_rollout_context_audit,
            v69_recovery_action_target_audit_path=(
                args.v69_recovery_action_target_audit
            ),
            branch_oracle_audit_path=args.branch_oracle_audit,
            trajectory_paths=trajectory_paths,
            trajectory_glob_patterns=tuple(args.trajectory_glob),
            min_broad_post_carrion_context_share=float(
                args.min_broad_post_carrion_context_share
            ),
            min_branch_oracle_exact_match_share=float(
                args.min_branch_oracle_exact_match_share
            ),
        )
        write_post_carrion_rollout_context_audit_report(report, args.output)
    except (OSError, ValueError, PostCarrionRolloutContextAuditError) as exc:
        raise SystemExit(
            f"failed to audit post-carrion rollout context coverage: {exc}"
        ) from exc

    _print_report_summary(report, args.output)


def _print_report_summary(
    report: Mapping[str, object],
    output_path: Path,
) -> None:
    classification = _mapping(report.get("classification"))
    coverage = _mapping(report.get("trajectory_rollout_context_coverage"))
    aggregate = _mapping(coverage.get("aggregate"))
    fixture = _mapping(report.get("fixture_failure_audit"))
    gate = _mapping(fixture.get("fixture_gate_comparison"))
    carrion_120 = _mapping(gate.get("carrion_only_120"))
    deltas = _mapping(carrion_120.get("metric_deltas"))
    evidence = _mapping(report.get("evidence"))
    trajectories = _mapping(evidence.get("trajectories"))
    print(f"post_carrion_rollout_context_audit={output_path}")
    print(f"schema_version={MIND_V3_POST_CARRION_ROLLOUT_CONTEXT_AUDIT_SCHEMA_VERSION}")
    print(f"classification={classification.get('primary')}")
    print(f"loaded_trajectory_count={trajectories.get('loaded_path_count', 0)}")
    print(f"trajectory_load_failure_count={trajectories.get('load_failure_count', 0)}")
    print(
        "trajectory_malformed_record_count="
        f"{trajectories.get('malformed_record_count', 0)}"
    )
    print(
        "trajectory_post_carrion_context_count="
        f"{aggregate.get('rollout_context_post_carrion_context_count', 0)}"
    )
    print(
        "trajectory_post_carrion_context_share="
        f"{aggregate.get('rollout_context_post_carrion_context_share')}"
    )
    print(
        "carrion_only_120_alive_delta_vs_baseline="
        f"{deltas.get('alive_agents_mean')}"
    )
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
