from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.broad_branch_residual_distillation_example import (
    build_broad_branch_residual_distillation_example_report,
    load_json_report,
    write_broad_branch_residual_distillation_example_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the v101 policy-visible broad residual distillation example "
            "from the accepted v100 constrained branch assignment. This emits "
            "a training-ready diagnostic artifact, not a runtime policy."
        )
    )
    parser.add_argument(
        "--v99-report",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v99-broad-branch-residual-oracle-audit.json"
        ),
    )
    parser.add_argument(
        "--v100-report",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v100-broad-branch-residual-constrained-audit.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/mind/"
            "mind-v3-v101-broad-branch-residual-distillation-example.json"
        ),
    )
    parser.add_argument(
        "--artifact-output",
        type=Path,
        default=Path(
            "output/mind/"
            "mind-v3-v101-broad-residual-distillation-example-artifact.json"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report, artifact = build_broad_branch_residual_distillation_example_report(
        v99_broad_branch_residual_oracle_report=load_json_report(args.v99_report),
        v100_broad_branch_residual_constrained_report=load_json_report(
            args.v100_report
        ),
    )
    write_broad_branch_residual_distillation_example_report(
        report,
        output_path=args.output,
        artifact=artifact,
        artifact_output_path=args.artifact_output,
    )
    coverage = report["coverage"]
    acceptance = report["acceptance"]
    print(f"v101_broad_branch_residual_distillation_example={args.output}")
    print(f"v101_artifact={args.artifact_output}")
    print(f"schema_version={report['schema_version']}")
    print(
        "v101_broad_residual_distillation_example_accepted="
        f"{report['v101_broad_residual_distillation_example_accepted']}"
    )
    print(f"training_row_count={coverage['training_row_count']}")
    print(f"source_seed_count={coverage['source_seed_count']}")
    print(
        "dominant_teacher_action_share="
        f"{coverage['dominant_teacher_action_share']}"
    )
    print(
        "training_accuracy="
        f"{report['training_evaluation']['training_accuracy']}"
    )
    print(f"blocker_count={acceptance['blocker_count']}")


if __name__ == "__main__":
    main()
