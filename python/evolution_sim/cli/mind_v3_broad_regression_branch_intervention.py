from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.broad_regression_branch_intervention import (
    DEFAULT_DATASET_PATH,
    DEFAULT_LIVE_REPORT_PATH,
    DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    DEFAULT_MAX_CANDIDATE_ACTIONS,
    DEFAULT_REPORT_PATH,
    DEFAULT_SCORER_PATH,
    DEFAULT_V142_TRAJECTORY_DIR,
    BroadRegressionBranchInterventionError,
    build_broad_regression_branch_intervention_archive,
    write_broad_regression_branch_intervention_dataset,
    write_broad_regression_branch_intervention_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the v143 diagnostics-only broad regression branch intervention "
            "archive from the v142 transition-value live A/B failure."
        )
    )
    parser.add_argument("--v142-scorer", type=Path, default=DEFAULT_SCORER_PATH)
    parser.add_argument(
        "--v142-live-report",
        type=Path,
        default=DEFAULT_LIVE_REPORT_PATH,
    )
    parser.add_argument(
        "--v142-trajectory-output-dir",
        type=Path,
        default=DEFAULT_V142_TRAJECTORY_DIR,
    )
    parser.add_argument(
        "--max-branch-points-per-seed",
        type=int,
        default=DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    )
    parser.add_argument(
        "--max-candidate-actions",
        type=int,
        default=DEFAULT_MAX_CANDIDATE_ACTIONS,
        help="0 means evaluate all valid public action-mask actions.",
    )
    parser.add_argument(
        "--no-regenerate-v142-trajectories",
        action="store_true",
        help="Do not regenerate missing v142 baseline/override trajectory files.",
    )
    parser.add_argument(
        "--no-verify-replay",
        action="store_true",
        help="Skip second deterministic branch replay verification pass.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=DEFAULT_DATASET_PATH,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report, dataset_rows = build_broad_regression_branch_intervention_archive(
            scorer_report_path=args.v142_scorer,
            live_report_path=args.v142_live_report,
            v142_trajectory_output_dir=args.v142_trajectory_output_dir,
            max_branch_points_per_seed=args.max_branch_points_per_seed,
            max_candidate_actions=args.max_candidate_actions,
            regenerate_v142_trajectories=not args.no_regenerate_v142_trajectories,
            verify_replay=not args.no_verify_replay,
        )
    except (OSError, BroadRegressionBranchInterventionError, ValueError) as exc:
        raise SystemExit(
            f"failed to build v143 broad regression branch intervention archive: {exc}"
        ) from exc
    write_broad_regression_branch_intervention_report(report, args.output)
    write_broad_regression_branch_intervention_dataset(
        dataset_rows,
        args.dataset_output,
    )
    _print_summary(report, args.output, args.dataset_output)


def _print_summary(
    report: dict[str, object],
    output_path: Path,
    dataset_path: Path,
) -> None:
    classification = report.get("classification")
    classification_value = (
        classification.get("primary")
        if isinstance(classification, dict)
        else classification
    )
    acceptance = report.get("acceptance")
    acceptance_payload = acceptance if isinstance(acceptance, dict) else {}
    print(f"broad_regression_branch_intervention_report={output_path}")
    print(f"broad_regression_branch_intervention_dataset={dataset_path}")
    print(f"classification={classification_value}")
    print(f"acceptance_passed={acceptance_payload.get('passed')}")
    print(f"first_failed_floor={acceptance_payload.get('first_failed_floor')}")
    print(f"dataset_row_count={acceptance_payload.get('dataset_row_count')}")
    print(f"dominant_label_action_share={acceptance_payload.get('dominant_label_action_share')}")


if __name__ == "__main__":
    main()
