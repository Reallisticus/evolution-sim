from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from evolution_sim.mind.hydration_after_carrion_audit import (
    DEFAULT_HYDRATION_THRESHOLDS,
    HydrationAfterCarrionAuditError,
    build_hydration_after_carrion_audit_report_from_datasets,
    load_hydration_after_carrion_audit_datasets,
    write_hydration_after_carrion_audit_report,
)
from evolution_sim.mind.rollout_context import RolloutContextConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether post-carrion true drink predicted as eat is mostly "
            "explained by policy-visible hydration deficit or drink/eat cycle aliasing."
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
    parser.add_argument(
        "--hydration-thresholds",
        default=",".join(str(value) for value in DEFAULT_HYDRATION_THRESHOLDS),
        help="Comma-separated hydration thresholds for diagnostic drink intervention.",
    )
    parser.add_argument("--recent-window", type=int, default=3)
    parser.add_argument("--recovery-phase-ticks", type=int, default=12)
    parser.add_argument("--ticks-since-cap", type=int, default=16)
    parser.add_argument("--no-gain-eat-streak-cap", type=int, default=5)
    parser.add_argument(
        "--min-true-drink-eat-rate-reduction",
        type=float,
        default=0.05,
    )
    parser.add_argument("--max-eat-label-damage-rate", type=float, default=0.01)
    parser.add_argument("--min-cycle-alias-share", type=float, default=0.6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-v65-hydration-after-carrion-audit.json"),
        help="Output JSON report path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        paths = _trajectory_paths(args.trajectory, args.trajectory_glob)
        datasets = load_hydration_after_carrion_audit_datasets(paths)
        report = build_hydration_after_carrion_audit_report_from_datasets(
            datasets,
            heldout_seed_values=_parse_seed_list(args.heldout_seeds),
            heldout_source_patterns=tuple(args.heldout_source_pattern),
            heldout_fraction=float(args.heldout_fraction),
            hydration_thresholds=_parse_float_list(args.hydration_thresholds),
            min_true_drink_eat_rate_reduction=float(
                args.min_true_drink_eat_rate_reduction
            ),
            max_eat_label_damage_rate=float(args.max_eat_label_damage_rate),
            min_cycle_alias_share=float(args.min_cycle_alias_share),
            context_config=RolloutContextConfig(
                recent_window=int(args.recent_window),
                recovery_phase_ticks=int(args.recovery_phase_ticks),
                ticks_since_cap=int(args.ticks_since_cap),
                no_gain_eat_streak_cap=int(args.no_gain_eat_streak_cap),
            ),
        )
        write_hydration_after_carrion_audit_report(report, args.output)
    except (
        OSError,
        json.JSONDecodeError,
        ValueError,
        HydrationAfterCarrionAuditError,
    ) as exc:
        raise SystemExit(f"failed to audit hydration after carrion: {exc}") from exc

    assessment = report["failure_mode_assessment"]  # type: ignore[index]
    best = report["best_hydration_intervention"]  # type: ignore[index]
    cycle = report["cycle_alias_assessment"]  # type: ignore[index]
    print(f"hydration_after_carrion_audit={args.output}")
    print(f"status={assessment['status']}")  # type: ignore[index]
    print(
        "best_hydration_threshold="
        f"{best.get('hydration_threshold') if isinstance(best, dict) else None}"
    )
    print(
        "true_drink_eat_rate_reduction="
        f"{best.get('true_drink_predicted_eat_absolute_rate_reduction') if isinstance(best, dict) else None}"
    )
    print(
        "cycle_alias_within_2_share="
        f"{cycle['next_eat_with_resource_gain_within_2_share']}"  # type: ignore[index]
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
    return tuple(int(token.strip()) for token in value.split(",") if token.strip())


def _parse_float_list(value: str) -> tuple[float, ...]:
    if not value.strip():
        return ()
    return tuple(float(token.strip()) for token in value.split(",") if token.strip())


if __name__ == "__main__":
    main()
