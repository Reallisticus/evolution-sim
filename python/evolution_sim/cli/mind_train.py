from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.artifacts import write_model_artifact
from evolution_sim.mind.baseline import train_behavior_cloning_baseline
from evolution_sim.mind.dataset import (
    combined_dataset_provenance,
    load_trajectory_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a deterministic Mind v1 offline BC baseline artifact.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        required=True,
        action="append",
        help="Trajectory JSONL or JSONL.GZ input.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/bc-baseline-artifact.json"),
        help="Model artifact destination.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datasets = [load_trajectory_jsonl(path) for path in args.trajectory]
    records = (
        record
        for dataset in datasets
        for record in dataset.records
    )
    baseline = train_behavior_cloning_baseline(
        records,
        provenance=combined_dataset_provenance(datasets),
    )
    write_model_artifact(args.output, baseline.to_artifact())
    print(f"artifact={args.output}")
    print(f"model_type={baseline.to_artifact()['manifest']['model_type']}")
    print(f"trained_record_count={baseline.record_count}")
    print(f"source_records={sum(dataset.record_count for dataset in datasets)}")
    print(f"source_trajectories={len(datasets)}")


if __name__ == "__main__":
    main()
