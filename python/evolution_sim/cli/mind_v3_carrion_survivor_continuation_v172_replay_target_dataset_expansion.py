from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v172_replay_target_dataset_expansion import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TARGET_DATASET_OUTPUT_PATH,
    DEFAULT_V171_DATASET_PATH,
    DEFAULT_V171_REPORT_PATH,
    EXPECTED_V171_CLASSIFICATION,
    EXPECTED_V171_DATASET_DIGEST,
    EXPECTED_V171_EXACT_DIGEST,
    CarrionSurvivorContinuationV172ReplayTargetDatasetExpansionError,
    run_carrion_survivor_continuation_v172_replay_target_dataset_expansion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the diagnostics-only v172 carrion survivor-continuation "
            "replay target dataset from the complete v171 replay-verified "
            "merged dataset. This command does not train, retrain a scorer, "
            "create a runtime artifact, run shadow/live evaluation, tune k or "
            "thresholds, change runtime behavior, or change replay/viewer schema."
        )
    )
    parser.add_argument("--v171-report", type=Path, default=DEFAULT_V171_REPORT_PATH)
    parser.add_argument("--v171-dataset", type=Path, default=DEFAULT_V171_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--target-dataset-output",
        type=Path,
        default=DEFAULT_TARGET_DATASET_OUTPUT_PATH,
    )
    parser.add_argument(
        "--expected-v171-exact-digest",
        default=EXPECTED_V171_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v171-dataset-digest",
        default=EXPECTED_V171_DATASET_DIGEST,
    )
    parser.add_argument(
        "--expected-v171-classification",
        default=EXPECTED_V171_CLASSIFICATION,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v172_replay_target_dataset_expansion(
            v171_report_path=args.v171_report,
            v171_dataset_path=args.v171_dataset,
            output_path=args.output,
            target_dataset_output_path=args.target_dataset_output,
            expected_v171_exact_digest=args.expected_v171_exact_digest,
            expected_v171_dataset_digest=args.expected_v171_dataset_digest,
            expected_v171_classification=args.expected_v171_classification,
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationV172ReplayTargetDatasetExpansionError,
    ) as exc:
        raise SystemExit(
            "failed to emit v172 carrion survivor-continuation replay target "
            f"dataset expansion: {exc}"
        ) from exc
    _print_summary(report, args.output, args.target_dataset_output)


def _print_summary(
    report: dict[str, object],
    output: Path,
    target_dataset_output: Path,
) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    source = report.get("source_validation")
    source_payload = source if isinstance(source, dict) else {}
    dataset = report.get("dataset")
    dataset_payload = dataset if isinstance(dataset, dict) else {}
    action_support = report.get("action_support_distribution")
    action_support_payload = action_support if isinstance(action_support, dict) else {}
    width = report.get("safe_set_width_distribution")
    width_payload = width if isinstance(width, dict) else {}
    counts = report.get("unique_action_vs_tied_action_row_counts")
    counts_payload = counts if isinstance(counts, dict) else {}
    print(
        "carrion_survivor_continuation_v172_replay_target_dataset_expansion_report="
        f"{output}"
    )
    print(f"target_dataset={target_dataset_output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={source_payload.get('passed')}")
    print(f"row_count={dataset_payload.get('row_count')}")
    print(f"dataset_digest={dataset_payload.get('dataset_digest')}")
    print(
        "action_support_distribution="
        f"{action_support_payload.get('per_action_support_counts')}"
    )
    print(
        "safe_set_width_distribution="
        f"{width_payload.get('width_counts')}"
    )
    print(
        "average_safe_set_width="
        f"{width_payload.get('average_safe_set_width')}"
    )
    print(
        "unique_action_row_count="
        f"{counts_payload.get('unique_action_row_count')}"
    )
    print(
        "tied_action_row_count="
        f"{counts_payload.get('tied_action_row_count')}"
    )
    print(f"leakage_scan_passed={report.get('leakage_scan', {}).get('passed')}")
    print(
        "row_schema_validation_passed="
        f"{report.get('row_schema_validation', {}).get('passed')}"
    )
    print(f"training_ran={report.get('training_ran')}")
    print(f"scorer_retraining_ran={report.get('scorer_retraining_ran')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(
        "runtime_action_selection_changed="
        f"{report.get('runtime_action_selection_changed')}"
    )
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"exact_digest={report.get('exact_digest')}")


if __name__ == "__main__":
    main()
