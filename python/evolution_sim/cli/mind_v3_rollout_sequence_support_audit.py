from __future__ import annotations

import argparse
import glob
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.current_route_decision import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V136_REPORT_PATH,
)
from evolution_sim.mind.rollout_context import RolloutContextConfig
from evolution_sim.mind.rollout_sequence_support_audit import (
    DEFAULT_OUTPUT_PATH,
    MIND_V3_ROLLOUT_SEQUENCE_SUPPORT_AUDIT_SCHEMA_VERSION,
    build_rollout_sequence_support_audit_report,
    write_rollout_sequence_support_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only Mind v3 v137 rollout-level sequence/"
            "world-model support audit."
        )
    )
    parser.add_argument("--v136-report", type=Path, default=DEFAULT_V136_REPORT_PATH)
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        default=[],
        help="Trajectory JSONL/JSONL.gz path. May be supplied more than once.",
    )
    parser.add_argument(
        "--trajectory-glob",
        action="append",
        default=[],
        help="Glob of trajectory JSONL/JSONL.gz paths.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        action="append",
        default=[],
        help="JSON report path from which trajectory paths can be extracted.",
    )
    parser.add_argument(
        "--heldout-seeds",
        default="5,13,19,29,37,41",
        help="Comma-separated source seeds to hold out when seed is derivable.",
    )
    parser.add_argument(
        "--heldout-source-pattern",
        action="append",
        default=[],
        help="Substring matched against source paths for held-out rows.",
    )
    parser.add_argument("--heldout-fraction", type=float, default=0.25)
    parser.add_argument("--recent-window", type=int, default=3)
    parser.add_argument("--recovery-phase-ticks", type=int, default=12)
    parser.add_argument("--ticks-since-cap", type=int, default=16)
    parser.add_argument("--no-gain-eat-streak-cap", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        trajectory_paths = _trajectory_paths(args.trajectory, args.trajectory_glob)
        if not trajectory_paths and not args.report:
            raise ValueError(
                "at least one --trajectory, expanded --trajectory-glob, "
                "or --report input is required"
            )
        build = build_rollout_sequence_support_audit_report(
            v136_report_path=args.v136_report,
            trajectory_paths=trajectory_paths,
            report_paths=tuple(args.report),
            heldout_seed_values=_parse_seed_list(args.heldout_seeds),
            heldout_source_patterns=tuple(args.heldout_source_pattern),
            heldout_fraction=float(args.heldout_fraction),
            context_config=RolloutContextConfig(
                recent_window=int(args.recent_window),
                recovery_phase_ticks=int(args.recovery_phase_ticks),
                ticks_since_cap=int(args.ticks_since_cap),
                no_gain_eat_streak_cap=int(args.no_gain_eat_streak_cap),
            ),
        )
        write_rollout_sequence_support_audit_report(build, output_path=args.output)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"failed to run rollout sequence support audit: {exc}") from exc
    _print_summary(build.report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    source = _mapping(report.get("source_integrity"))
    floors = _mapping(report.get("support_floors"))
    classification = _mapping(report.get("classification"))
    summaries = _mapping(report.get("model_summaries"))
    sequence = _mapping(summaries.get("public_rollout_sequence_history_lookup"))
    auth = _mapping(report.get("authorization_block"))
    print(f"rollout_sequence_support_audit={output_path}")
    print(f"schema_version={MIND_V3_ROLLOUT_SEQUENCE_SUPPORT_AUDIT_SCHEMA_VERSION}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"classification={classification.get('primary')}")
    print(f"support_floors_passed={floors.get('passed')}")
    print(f"first_failed_floor={floors.get('first_failed_floor')}")
    print(f"heldout_accuracy={sequence.get('accuracy')}")
    print(
        "dominant_predicted_action_share="
        f"{sequence.get('dominant_predicted_action_share')}"
    )
    print(f"unsupported_action_count={sequence.get('unsupported_action_count')}")
    print(f"training_authorized={auth.get('training_authorized')}")
    print(
        "runtime_policy_change_authorized="
        f"{auth.get('runtime_policy_change_authorized')}"
    )


def _trajectory_paths(
    explicit_paths: list[Path],
    patterns: list[str],
) -> tuple[Path, ...]:
    paths = {Path(path) for path in explicit_paths}
    for pattern in patterns:
        paths.update(Path(path) for path in glob.glob(pattern))
    return tuple(sorted(paths, key=lambda path: str(path)))


def _parse_seed_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    seeds: list[int] = []
    for token in value.split(","):
        stripped = token.strip()
        if stripped:
            seeds.append(int(stripped))
    return tuple(seeds)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
