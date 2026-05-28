from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from evolution_sim.mind.rollout_context import RolloutContextConfig
from evolution_sim.mind.rollout_context_audit import (
    MIND_V3_ROLLOUT_CONTEXT_AUDIT_SCHEMA_VERSION,
    RolloutContextAuditError,
    build_rollout_context_audit_report_from_datasets,
    load_rollout_context_audit_datasets,
    write_rollout_context_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether deterministic per-agent rollout context derived only "
            "from trajectory rows separates Mind v3 IQL eat-vs-move/drink/stay aliases."
        )
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
        "--heldout-seeds",
        default="",
        help="Comma-separated source seeds to hold out when seed is derivable.",
    )
    parser.add_argument(
        "--heldout-source-pattern",
        action="append",
        default=[],
        help="Substring matched against source paths for held-out rows.",
    )
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--recent-window", type=int, default=3)
    parser.add_argument("--recovery-phase-ticks", type=int, default=12)
    parser.add_argument("--ticks-since-cap", type=int, default=16)
    parser.add_argument("--no-gain-eat-streak-cap", type=int, default=5)
    parser.add_argument(
        "--min-eat-overprediction-rate-reduction",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-v64-rollout-context-audit.json"),
        help="Output JSON report path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        paths = _trajectory_paths(args.trajectory, args.trajectory_glob)
        datasets = load_rollout_context_audit_datasets(paths)
        report = build_rollout_context_audit_report_from_datasets(
            datasets,
            heldout_seed_values=_parse_seed_list(args.heldout_seeds),
            heldout_source_patterns=tuple(args.heldout_source_pattern),
            heldout_fraction=float(args.heldout_fraction),
            min_eat_overprediction_rate_reduction=float(
                args.min_eat_overprediction_rate_reduction
            ),
            context_config=RolloutContextConfig(
                recent_window=int(args.recent_window),
                recovery_phase_ticks=int(args.recovery_phase_ticks),
                ticks_since_cap=int(args.ticks_since_cap),
                no_gain_eat_streak_cap=int(args.no_gain_eat_streak_cap),
            ),
        )
        write_rollout_context_audit_report(report, args.output)
    except (OSError, json.JSONDecodeError, ValueError, RolloutContextAuditError) as exc:
        raise SystemExit(f"failed to audit rollout context: {exc}") from exc

    assessment = report["failure_mode_assessment"]  # type: ignore[index]
    split = report["train_heldout_split"]  # type: ignore[index]
    print(f"rollout_context_audit={args.output}")
    print(f"schema_version={MIND_V3_ROLLOUT_CONTEXT_AUDIT_SCHEMA_VERSION}")
    print(f"train_record_count={split['train_record_count']}")  # type: ignore[index]
    print(f"heldout_record_count={split['heldout_record_count']}")  # type: ignore[index]
    print(
        "materially_improves_v62_failure_mode="
        f"{assessment['materially_improves_v62_failure_mode']}"  # type: ignore[index]
    )
    print(
        "movement_drink_stay_absolute_rate_reduction="
        f"{assessment['movement_drink_stay_absolute_rate_reduction']}"  # type: ignore[index]
    )
    print(f"blocker_count={len(assessment['blockers'])}")  # type: ignore[index]


def _trajectory_paths(
    explicit_paths: list[Path],
    patterns: list[str],
) -> tuple[Path, ...]:
    paths = {Path(path) for path in explicit_paths}
    for pattern in patterns:
        paths.update(Path(path) for path in glob.glob(pattern))
    resolved = tuple(sorted(paths, key=lambda path: str(path)))
    if not resolved:
        raise ValueError("at least one --trajectory or --trajectory-glob is required")
    return resolved


def _parse_seed_list(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    seeds: list[int] = []
    for token in value.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        seeds.append(int(stripped))
    return tuple(seeds)


if __name__ == "__main__":
    main()
