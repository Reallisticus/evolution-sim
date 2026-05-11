from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.dataset import load_trajectory_jsonl
from evolution_sim.mind.horizon_labels import (
    DEFAULT_HORIZON_TICKS,
    HorizonLabelError,
    build_horizon_label_report,
    parse_horizon_ticks,
    write_horizon_label_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate deterministic Mind horizon labels from trajectory JSONL "
            "datasets."
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
        "--horizons",
        default=",".join(str(horizon) for horizon in DEFAULT_HORIZON_TICKS),
        help="Comma-separated future tick horizons to label.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-horizon-labels.json"),
        help="Output JSON report path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        horizons = parse_horizon_ticks(args.horizons)
        datasets = [
            load_trajectory_jsonl(path)
            for path in args.trajectory
        ]
        report = build_horizon_label_report(datasets, horizons=horizons)
        write_horizon_label_report(report, args.output)
    except (OSError, ValueError, HorizonLabelError) as exc:
        raise SystemExit(f"failed to generate horizon labels: {exc}") from exc

    aggregate = report["aggregate"]
    horizon_summary = aggregate["horizons"]  # type: ignore[index]
    print(f"horizon_labels={args.output}")
    print(f"schema_version={report['schema_version']}")
    print(f"trajectory_count={report['source']['trajectory_count']}")  # type: ignore[index]
    print(f"source_record_count={report['source']['record_count']}")  # type: ignore[index]
    print(f"label_count={aggregate['label_count']}")  # type: ignore[index]
    print(f"horizons={','.join(str(horizon) for horizon in horizons)}")
    for horizon in horizons:
        payload = horizon_summary[str(horizon)]  # type: ignore[index]
        print(
            "horizon_"
            f"{horizon}_observed={payload['observed_count']}/"
            f"{payload['record_count']}"
        )


if __name__ == "__main__":
    main()
