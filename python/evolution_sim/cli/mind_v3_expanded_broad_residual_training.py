from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.expanded_broad_residual_training import (
    build_expanded_broad_residual_training_report,
    load_json_report,
    write_expanded_broad_residual_training_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the v102 expanded broad residual training-data diagnostic "
            "from replay-verified expanded v99 branch outcomes and the "
            "corresponding v100 constrained assignment."
        )
    )
    parser.add_argument(
        "--v99-report",
        type=Path,
        default=Path(
            "output/mind/"
            "mind-v3-v102-expanded-broad-residual-oracle-source.json"
        ),
    )
    parser.add_argument(
        "--v100-report",
        type=Path,
        default=Path(
            "output/mind/"
            "mind-v3-v102-expanded-broad-residual-constrained-source.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v102-expanded-broad-residual-training.json"
        ),
    )
    parser.add_argument(
        "--artifact-output",
        type=Path,
        default=Path(
            "output/mind/"
            "mind-v3-v102-expanded-broad-residual-training-artifact.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report, artifact = build_expanded_broad_residual_training_report(
        v99_expanded_oracle_report=load_json_report(args.v99_report),
        v100_constrained_report=load_json_report(args.v100_report),
    )
    write_expanded_broad_residual_training_report(
        report,
        output_path=args.output,
        artifact=artifact,
        artifact_output_path=args.artifact_output,
    )
    coverage = report["coverage"]
    loo = report["leave_one_source_seed_out_evaluation"]
    acceptance = report["acceptance"]
    print(f"v102_expanded_broad_residual_training={args.output}")
    print(f"v102_artifact={args.artifact_output}")
    print(f"schema_version={report['schema_version']}")
    print(
        "v102_expanded_broad_residual_training_accepted="
        f"{report['v102_expanded_broad_residual_training_accepted']}"
    )
    print(f"training_row_count={coverage['training_row_count']}")
    print(f"source_seed_count={coverage['source_seed_count']}")
    print(
        "dominant_teacher_action_share="
        f"{coverage['dominant_teacher_action_share']}"
    )
    print(
        "movement_reposition_label_share="
        f"{coverage['movement_reposition_label_share']}"
    )
    print(
        "loo_exact_action_accuracy="
        f"{loo['exact_action_accuracy']}"
    )
    print(
        "loo_dominant_predicted_action_share="
        f"{loo['dominant_predicted_action_share']}"
    )
    print(
        "loo_mean_target_local_score_delta="
        f"{loo['target_local_score_delta_mean']}"
    )
    print(f"blocker_count={acceptance['blocker_count']}")


if __name__ == "__main__":
    main()
