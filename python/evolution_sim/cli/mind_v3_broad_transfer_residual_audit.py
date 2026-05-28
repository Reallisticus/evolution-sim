from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.broad_transfer_residual_audit import (
    build_broad_transfer_residual_audit_report,
    load_json_mapping,
    write_broad_transfer_residual_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the v98 broad-transfer support-gated residual diagnostic "
            "from v97 planner-distilled rollout trajectories and non-strict "
            "linear Mind v3 support trajectories."
        )
    )
    parser.add_argument(
        "--v97-report",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v97-planner-distilled-runtime-promotion-report.json"
        ),
    )
    parser.add_argument(
        "--v97-trajectory-dir",
        type=Path,
        default=Path("output/mind/v97-planner-distilled-trajectories"),
    )
    parser.add_argument(
        "--planner-distilled-artifact",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v96-planner-distillation-runtime-feasibility.json"
        ),
    )
    parser.add_argument(
        "--support-trajectory-dir",
        type=Path,
        default=Path("output/mind/v98-broad-support-trajectories"),
    )
    parser.add_argument(
        "--support-archive-output",
        type=Path,
        default=Path("output/mind/mind-v3-v98-broad-transfer-support-archive.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-v98-broad-transfer-residual-audit.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report, support_archive = build_broad_transfer_residual_audit_report(
        v97_report=load_json_mapping(args.v97_report),
        v97_trajectory_dir=args.v97_trajectory_dir,
        planner_distilled_artifact=args.planner_distilled_artifact,
        support_trajectory_dir=args.support_trajectory_dir,
    )
    write_broad_transfer_residual_audit_report(
        report,
        output_path=args.output,
        support_archive=support_archive,
        support_archive_output_path=args.support_archive_output,
    )
    support = report["support_coverage"]
    residual = report["residual_gate_diagnostic"]
    print(f"v98_broad_transfer_residual_audit={args.output}")
    print(f"schema_version={report['schema_version']}")
    print(f"support_row_count={support['support_row_count']}")
    print(f"source_seed_count={support['source_seed_count']}")
    print(
        "dominant_support_teacher_action_share="
        f"{support['dominant_support_teacher_action_share']}"
    )
    print(
        "residual_abstention_rate="
        f"{residual['residual_abstention_rate']}"
    )
    print(
        "residual_dominant_override_action_share="
        f"{residual['dominant_override_action_share']}"
    )
    print(
        "v98_broad_transfer_residual_diagnostic_accepted="
        f"{report['v98_broad_transfer_residual_diagnostic_accepted']}"
    )
    print(f"blocker_count={report['blocker_count']}")


if __name__ == "__main__":
    main()
