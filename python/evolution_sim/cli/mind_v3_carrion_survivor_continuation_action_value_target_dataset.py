from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    DEFAULT_MAX_DOMINANT_SAFE_ACTION_SHARE,
    DEFAULT_MIN_SAFE_RUNS_PER_ACTION,
    DEFAULT_MIN_SAFE_SHARE_PER_ACTION,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TARGET_DATASET_OUTPUT_PATH,
    DEFAULT_UNIQUE_WINNER_SCORE_MARGIN,
    DEFAULT_V154_DATASET_PATH,
    DEFAULT_V154_REPORT_PATH,
    DEFAULT_V155_REPORT_PATH,
    DEFAULT_V156_REPORT_PATH,
    DEFAULT_V157_REPORT_PATH,
    CarrionSurvivorContinuationActionValueTargetDatasetError,
    run_carrion_survivor_continuation_action_value_target_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the diagnostics-only v158 carrion survivor-continuation "
            "set-valued/action-value target JSONL dataset. The command creates "
            "no model artifact, runs no training, and performs no shadow/live A/B."
        )
    )
    parser.add_argument("--v154-report", type=Path, default=DEFAULT_V154_REPORT_PATH)
    parser.add_argument("--v154-dataset", type=Path, default=DEFAULT_V154_DATASET_PATH)
    parser.add_argument("--v155-report", type=Path, default=DEFAULT_V155_REPORT_PATH)
    parser.add_argument("--v156-report", type=Path, default=DEFAULT_V156_REPORT_PATH)
    parser.add_argument("--v157-report", type=Path, default=DEFAULT_V157_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--target-dataset-output",
        type=Path,
        default=DEFAULT_TARGET_DATASET_OUTPUT_PATH,
    )
    parser.add_argument(
        "--min-safe-runs-per-action",
        type=int,
        default=DEFAULT_MIN_SAFE_RUNS_PER_ACTION,
        help="Minimum safe continuation runs before an action can be robust.",
    )
    parser.add_argument(
        "--min-safe-share-per-action",
        type=float,
        default=DEFAULT_MIN_SAFE_SHARE_PER_ACTION,
        help="Minimum safe-run share before an action can be robust.",
    )
    parser.add_argument(
        "--unique-winner-score-margin",
        type=float,
        default=DEFAULT_UNIQUE_WINNER_SCORE_MARGIN,
        help="Mean action-value score margin required for a unique robust winner.",
    )
    parser.add_argument(
        "--max-dominant-safe-action-share",
        type=float,
        default=DEFAULT_MAX_DOMINANT_SAFE_ACTION_SHARE,
        help="Maximum dominant safe-action support share before support is collapsed.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_action_value_target_dataset(
            v154_report_path=args.v154_report,
            v154_dataset_path=args.v154_dataset,
            v155_report_path=args.v155_report,
            v156_report_path=args.v156_report,
            v157_report_path=args.v157_report,
            output_path=args.output,
            target_dataset_output_path=args.target_dataset_output,
            min_safe_runs_per_action=args.min_safe_runs_per_action,
            min_safe_share_per_action=args.min_safe_share_per_action,
            unique_winner_score_margin=args.unique_winner_score_margin,
            max_dominant_safe_action_share=args.max_dominant_safe_action_share,
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationActionValueTargetDatasetError,
    ) as exc:
        raise SystemExit(
            "failed to emit v158 carrion survivor-continuation "
            f"action-value target dataset: {exc}"
        ) from exc
    _print_summary(report, args.output, args.target_dataset_output)


def _print_summary(
    report: dict[str, object],
    output: Path,
    target_dataset_output: Path,
) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    validation = report.get("source_validation")
    validation_payload = validation if isinstance(validation, dict) else {}
    dataset = report.get("dataset")
    dataset_payload = dataset if isinstance(dataset, dict) else {}
    print(f"carrion_survivor_continuation_action_value_target_dataset_report={output}")
    print(f"target_dataset={target_dataset_output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={validation_payload.get('passed')}")
    print(f"group_count={dataset_payload.get('group_count')}")
    print(
        "action_value_target_row_count="
        f"{dataset_payload.get('action_value_target_row_count')}"
    )
    print(f"unique_winner_group_count={dataset_payload.get('unique_winner_group_count')}")
    print(
        "multi_action_safe_set_group_count="
        f"{dataset_payload.get('multi_action_safe_set_group_count')}"
    )
    print(f"unresolved_group_count={dataset_payload.get('unresolved_group_count')}")
    print(
        "dominant_safe_action_share="
        f"{dataset_payload.get('dominant_safe_action_share')}"
    )
    print(f"dataset_digest={dataset_payload.get('dataset_digest')}")
    print(f"artifact_created={report.get('artifact_created')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"shadow_live_ab_ran={report.get('shadow_live_ab_ran')}")
    print(f"training_authorized={report.get('training_authorized')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"runtime_promotion_allowed={report.get('runtime_promotion_allowed')}")
    print(f"exact_digest={report.get('exact_digest')}")


if __name__ == "__main__":
    main()
