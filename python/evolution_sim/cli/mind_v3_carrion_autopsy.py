from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_autopsy import (
    DEFAULT_CRITICAL_RATIO_FLOOR,
    DEFAULT_MAX_EXAMPLES,
    DEFAULT_POST_CONTACT_WINDOW_TICKS,
    DEFAULT_SEQUENCE_RECORD_LIMIT,
    CarrionAutopsyError,
    build_carrion_autopsy_report,
    write_carrion_autopsy_report,
)
from evolution_sim.mind.dataset import TrajectoryDatasetError, load_trajectory_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic Mind v3 carrion-failure autopsy from "
            "trajectory JSONL datasets."
        )
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        required=True,
        help="Trajectory JSONL or JSONL.gz input. Repeat to combine datasets.",
    )
    parser.add_argument(
        "--post-contact-window-ticks",
        type=int,
        default=DEFAULT_POST_CONTACT_WINDOW_TICKS,
        help=(
            "Number of ticks to analyze after the first carcass/fresh-kill "
            "consumption event for each agent."
        ),
    )
    parser.add_argument(
        "--sequence-record-limit",
        type=int,
        default=DEFAULT_SEQUENCE_RECORD_LIMIT,
        help="Maximum records kept in each example sequence excerpt.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=DEFAULT_MAX_EXAMPLES,
        help="Maximum concrete trajectory examples to keep in the report.",
    )
    parser.add_argument(
        "--critical-ratio-floor",
        type=float,
        default=DEFAULT_CRITICAL_RATIO_FLOOR,
        help="Terminal energy/hydration/health floor used for bottleneck labels.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-carrion-autopsy.json"),
        help="Output JSON report path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        datasets = [load_trajectory_jsonl(path) for path in args.trajectory]
        report = build_carrion_autopsy_report(
            datasets,
            post_contact_window_ticks=int(args.post_contact_window_ticks),
            sequence_record_limit=int(args.sequence_record_limit),
            max_examples=int(args.max_examples),
            critical_ratio_floor=float(args.critical_ratio_floor),
        )
        write_carrion_autopsy_report(report, args.output)
    except (OSError, ValueError, TrajectoryDatasetError, CarrionAutopsyError) as exc:
        raise SystemExit(f"failed to build carrion autopsy: {exc}") from exc

    aggregate = report["aggregate"]  # type: ignore[index]
    dominant = aggregate["dominant_death_path"]  # type: ignore[index]
    print(f"carrion_autopsy={args.output}")
    print(f"schema_version={report['schema_version']}")
    print(f"trajectory_count={report['source']['trajectory_count']}")  # type: ignore[index]
    print(f"source_record_count={report['source']['record_count']}")  # type: ignore[index]
    print(f"contact_episode_count={aggregate['contact_episode_count']}")  # type: ignore[index]
    print(f"death_after_contact_count={aggregate['death_after_contact_count']}")  # type: ignore[index]
    print(f"dominant_death_path={dominant['path']}")  # type: ignore[index]


if __name__ == "__main__":
    main()
