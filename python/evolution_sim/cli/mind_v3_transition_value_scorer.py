from __future__ import annotations

import argparse
import glob
from pathlib import Path

from evolution_sim.mind.rollout_sequence_support_audit import STRICT_HELDOUT_SEEDS
from evolution_sim.mind.transition_value_scorer import (
    DEFAULT_OUTPUT_PATH,
    MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION,
    build_transition_value_scorer_report,
    write_transition_value_scorer_report,
)

DEFAULT_TRAIN_TRAJECTORY_GLOB = (
    "output/mind/v98-broad-support-trajectories/open-mind-v3-[0-9]*-120.jsonl.gz"
)
DEFAULT_STRICT_HELDOUT_TRAJECTORY_GLOB = (
    "output/mind/v138-strict-heldout-trajectories/open-mind-v3-[0-9]*-120.jsonl.gz"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the v142 public transition-value Mind v3 scorer artifact."
        )
    )
    parser.add_argument(
        "--train-trajectory-glob",
        action="append",
        default=[],
        help="Glob for public non-strict-seed Mind v3 trajectory JSONL files.",
    )
    parser.add_argument(
        "--strict-heldout-trajectory-glob",
        action="append",
        default=[],
        help="Glob for strict heldout public Mind v3 trajectory JSONL files.",
    )
    parser.add_argument(
        "--strict-seeds",
        default=",".join(str(seed) for seed in STRICT_HELDOUT_SEEDS),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train_patterns = args.train_trajectory_glob or [DEFAULT_TRAIN_TRAJECTORY_GLOB]
    heldout_patterns = (
        args.strict_heldout_trajectory_glob
        or [DEFAULT_STRICT_HELDOUT_TRAJECTORY_GLOB]
    )
    train_paths = _expand_globs(train_patterns)
    heldout_paths = _expand_globs(heldout_patterns)
    build = build_transition_value_scorer_report(
        train_trajectory_paths=train_paths,
        strict_heldout_trajectory_paths=heldout_paths,
        strict_seed_values=_parse_seeds(args.strict_seeds),
    )
    write_transition_value_scorer_report(build, output_path=args.output)
    report = build.report
    print(f"transition_value_scorer={args.output}")
    print(f"schema_version={MIND_V3_TRANSITION_VALUE_SCORER_SCHEMA_VERSION}")
    print(f"source_integrity_passed={report['source_integrity']['passed']}")
    print(f"support_floors_passed={report['support_floors']['passed']}")
    print(f"classification={report['classification']['primary']}")


def _expand_globs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in sorted(glob.glob(pattern)))
    return list(dict.fromkeys(paths))


def _parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for token in value.split(","):
        stripped = token.strip()
        if stripped:
            seeds.append(int(stripped))
    return seeds


if __name__ == "__main__":
    main()
