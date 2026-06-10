from __future__ import annotations

import argparse
import glob
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.rollout_context import RolloutContextConfig
from evolution_sim.mind.rollout_sequence_strict_seed_support_recheck import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V138_REPORT_PATH,
)
from evolution_sim.mind.sequence_history_shadow_scorer import (
    DEFAULT_OUTPUT_PATH,
    MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_SCHEMA_VERSION,
    build_sequence_history_shadow_scorer_report,
    write_sequence_history_shadow_scorer_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 v139 deterministic "
            "sequence-history shadow scorer artifact."
        )
    )
    parser.add_argument("--v138-report", type=Path, default=DEFAULT_V138_REPORT_PATH)
    parser.add_argument(
        "--train-trajectory",
        type=Path,
        action="append",
        default=[],
        help="v98 numeric train-bank trajectory JSONL/JSONL.gz path.",
    )
    parser.add_argument(
        "--train-trajectory-glob",
        action="append",
        default=[],
        help="Glob of v98 numeric train-bank trajectory JSONL/JSONL.gz paths.",
    )
    parser.add_argument(
        "--strict-heldout-trajectory",
        type=Path,
        action="append",
        default=[],
        help="Strict heldout Mind v3 trajectory JSONL/JSONL.gz path.",
    )
    parser.add_argument(
        "--strict-heldout-trajectory-glob",
        action="append",
        default=[],
        help="Glob of strict heldout Mind v3 trajectory JSONL/JSONL.gz paths.",
    )
    parser.add_argument(
        "--strict-seeds",
        default="5,13,19,29,37,41",
        help="Comma-separated strict seed values used only for heldout evaluation.",
    )
    parser.add_argument("--recent-window", type=int, default=3)
    parser.add_argument("--recovery-phase-ticks", type=int, default=12)
    parser.add_argument("--ticks-since-cap", type=int, default=16)
    parser.add_argument("--no-gain-eat-streak-cap", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        train_paths = _trajectory_paths(
            args.train_trajectory,
            args.train_trajectory_glob,
        )
        strict_heldout_paths = _trajectory_paths(
            args.strict_heldout_trajectory,
            args.strict_heldout_trajectory_glob,
        )
        if not train_paths:
            raise ValueError(
                "at least one --train-trajectory or expanded "
                "--train-trajectory-glob input is required"
            )
        if not strict_heldout_paths:
            raise ValueError(
                "at least one --strict-heldout-trajectory or expanded "
                "--strict-heldout-trajectory-glob input is required"
            )
        build = build_sequence_history_shadow_scorer_report(
            v138_report_path=args.v138_report,
            train_trajectory_paths=train_paths,
            strict_heldout_trajectory_paths=strict_heldout_paths,
            strict_seed_values=_parse_seed_list(args.strict_seeds),
            context_config=RolloutContextConfig(
                recent_window=int(args.recent_window),
                recovery_phase_ticks=int(args.recovery_phase_ticks),
                ticks_since_cap=int(args.ticks_since_cap),
                no_gain_eat_streak_cap=int(args.no_gain_eat_streak_cap),
            ),
        )
        write_sequence_history_shadow_scorer_report(build, output_path=args.output)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"failed to build sequence-history shadow scorer: {exc}"
        ) from exc
    _print_summary(build.report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    source = _mapping(report.get("source_integrity"))
    floors = _mapping(report.get("support_floors"))
    classification = _mapping(report.get("classification"))
    metrics = _mapping(report.get("strict_heldout_metrics"))
    summaries = _mapping(metrics.get("model_summaries"))
    sequence = _mapping(summaries.get("public_rollout_sequence_history_lookup"))
    comparisons = _mapping(metrics.get("baseline_comparisons"))
    roundtrip = _mapping(report.get("artifact_roundtrip"))
    auth = _mapping(report.get("authorization_block"))
    print(f"sequence_history_shadow_scorer={output_path}")
    print(f"schema_version={MIND_V3_SEQUENCE_HISTORY_SHADOW_SCORER_SCHEMA_VERSION}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"classification={classification.get('primary')}")
    print(f"support_floors_passed={floors.get('passed')}")
    print(f"first_failed_floor={floors.get('first_failed_floor')}")
    print(f"strict_heldout_accuracy={sequence.get('accuracy')}")
    print(
        "strict_delta_vs_action_only="
        f"{comparisons.get('strict_heldout_accuracy_delta_vs_action_only')}"
    )
    print(
        "strict_delta_vs_action_order="
        f"{comparisons.get('strict_heldout_accuracy_delta_vs_action_order')}"
    )
    print(
        "dominant_predicted_action_share="
        f"{sequence.get('dominant_predicted_action_share')}"
    )
    print(f"unsupported_action_count={sequence.get('unsupported_action_count')}")
    print(
        "loaded_scores_match_pre_serialization="
        f"{roundtrip.get('loaded_artifact_scores_match_pre_serialization')}"
    )
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
