from __future__ import annotations

import argparse
import glob
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_state_action_readiness_audit import (
    DEFAULT_ARCHIVE_REPORT_PATH,
    DEFAULT_ARCHIVE_ROWS_PATH,
    DEFAULT_HISTORY_WINDOW,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PUBLIC_SIGNAL_AUDIT_PATH,
    DEFAULT_SHADOW_RANKER_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_STATE_ACTION_READINESS_AUDIT_SCHEMA_VERSION,
    FirstRecoveryStateActionReadinessAuditError,
    build_first_recovery_state_action_readiness_audit,
    write_first_recovery_state_action_readiness_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 first-recovery state-action "
            "interaction and observation-field readiness audit."
        )
    )
    parser.add_argument(
        "--public-signal-audit",
        type=Path,
        default=DEFAULT_PUBLIC_SIGNAL_AUDIT_PATH,
        help="Input v112 first-recovery public signal audit JSON report.",
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
        "--shadow-ranker-report",
        type=Path,
        default=DEFAULT_SHADOW_RANKER_REPORT_PATH,
        help="Input v111 joined public shadow-ranker JSON report.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        default=[],
        help="Trajectory JSONL path for public observation/history joins.",
    )
    parser.add_argument(
        "--trajectory-glob",
        action="append",
        default=[],
        help="Trajectory JSONL glob for public observation/history joins.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=DEFAULT_HISTORY_WINDOW,
        help="Same-agent prior record window for public history signals.",
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
        build = build_first_recovery_state_action_readiness_audit(
            public_signal_audit_path=args.public_signal_audit,
            archive_report_path=args.archive_report,
            archive_rows_path=args.archive_rows,
            shadow_ranker_report_path=args.shadow_ranker_report,
            trajectory_paths=trajectory_paths,
            trajectory_glob_patterns=tuple(args.trajectory_glob),
            history_window=int(args.history_window),
        )
        write_first_recovery_state_action_readiness_audit_report(
            build,
            output_path=args.output,
        )
    except (OSError, ValueError, FirstRecoveryStateActionReadinessAuditError) as exc:
        raise SystemExit(
            "failed to build first-recovery state-action readiness audit: "
            f"{exc}"
        ) from exc
    _print_summary(build.report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source"))
    join = _mapping(report.get("join_evidence"))
    baselines = _mapping(report.get("baseline_evaluation"))
    heldout = _mapping(baselines.get("heldout_summary"))
    interaction = _mapping(baselines.get("state_action_interaction"))
    interaction_all = _mapping(interaction.get("all_data_descriptive"))
    interaction_dist = _mapping(interaction_all.get("selected_action_distribution"))
    aliasing = _mapping(report.get("alias_quantization_stability"))
    fields = _mapping(report.get("candidate_field_readiness"))
    leakage = _mapping(report.get("leakage_guard"))
    recommendation = _mapping(report.get("recommendation"))
    print(f"first_recovery_state_action_readiness_audit={output_path}")
    print(
        "schema_version="
        f"{MIND_V3_FIRST_RECOVERY_STATE_ACTION_READINESS_AUDIT_SCHEMA_VERSION}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"archive_row_count={source.get('archive_row_count')}")
    print(f"target_group_count={source.get('target_group_count')}")
    print(f"loaded_path_count={join.get('loaded_path_count')}")
    print(f"malformed_record_count={join.get('malformed_record_count')}")
    print(f"matched_archive_row_count={join.get('matched_archive_row_count')}")
    print(f"missing_archive_row_count={join.get('missing_archive_row_count')}")
    print(
        "state_action_beats_action_only="
        f"{heldout.get('go_criterion_state_action_beats_action_only')}"
    )
    print(
        "alias_resolution_stable="
        f"{aliasing.get('state_action_resolution_stable_exact_1_2_3_digit')}"
    )
    print(
        "contractable_missing_field_candidate_count="
        f"{fields.get('contractable_missing_field_candidate_count')}"
    )
    print(
        "dominant_selected_action_share="
        f"{interaction_dist.get('dominant_selected_action_share')}"
    )
    print(f"leakage_count={leakage.get('leakage_count')}")
    print(f"recommendation={recommendation.get('next_step')}")


def _trajectory_paths(explicit_paths: list[Path], patterns: list[str]) -> tuple[Path, ...]:
    paths = {Path(path) for path in explicit_paths}
    for pattern in patterns:
        paths.update(Path(path) for path in glob.glob(pattern))
    return tuple(sorted(paths, key=lambda path: str(path)))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
