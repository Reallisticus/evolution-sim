from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.dataset import load_trajectory_jsonl
from evolution_sim.mind.v3_neural import (
    MIND_V3_NEURAL_DEFAULT_HIDDEN_UNITS,
    MIND_V3_NEURAL_DEFAULT_SEED,
    MindV3NeuralArtifactError,
    load_json_report,
    train_mind_v3_neural_artifact,
    write_mind_v3_neural_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a deterministic frozen Mind v3 neural artifact from "
            "trajectory horizon labels and optional fixture blocker labels."
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
        "--trajectory-weight",
        type=float,
        action="append",
        help=(
            "Optional positive multiplier for each --trajectory in the same "
            "order. Repeat exactly once per trajectory."
        ),
    )
    parser.add_argument(
        "--horizon-labels",
        type=Path,
        required=True,
        help="mind_horizon_labels_v1 report generated from the same trajectories.",
    )
    parser.add_argument(
        "--fixture-labels",
        type=Path,
        help="Optional mind_fixture_blocker_labels_v1 report for global pressure.",
    )
    parser.add_argument(
        "--hidden-units",
        type=int,
        default=MIND_V3_NEURAL_DEFAULT_HIDDEN_UNITS,
        help="Deterministic hidden projection width.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=MIND_V3_NEURAL_DEFAULT_SEED,
        help="Deterministic projection seed.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-neural-artifact.json"),
        help="Output artifact path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        datasets = [load_trajectory_jsonl(path) for path in args.trajectory]
        horizon_label_report = load_json_report(args.horizon_labels)
        fixture_label_report = (
            load_json_report(args.fixture_labels)
            if args.fixture_labels is not None
            else None
        )
        artifact = train_mind_v3_neural_artifact(
            datasets,
            horizon_label_report=horizon_label_report,
            fixture_label_report=fixture_label_report,
            hidden_units=int(args.hidden_units),
            seed=int(args.seed),
            trajectory_weight_multipliers=args.trajectory_weight,
        )
        write_mind_v3_neural_artifact(artifact, args.output)
    except (OSError, ValueError, MindV3NeuralArtifactError) as exc:
        raise SystemExit(f"failed to train Mind v3 neural artifact: {exc}") from exc

    print(f"mind_v3_neural_artifact={args.output}")
    print(f"schema_version={artifact['schema_version']}")
    print(f"model_type={artifact['model_type']}")
    print(f"trained_record_count={artifact['trained_record_count']}")
    print(f"hidden_units={artifact['hidden_units']}")
    print(
        "fixture_pressure_total="
        f"{artifact['fixture_pressure_summary']['pressure_total']}"  # type: ignore[index]
    )


if __name__ == "__main__":
    main()
