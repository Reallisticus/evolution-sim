from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v165_preterminal_target_dataset_expansion import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TARGET_DATASET_OUTPUT_PATH,
    DEFAULT_V158_DATASET_PATH,
    DEFAULT_V164_REPORT_PATH,
    CarrionSurvivorContinuationV165PreterminalTargetDatasetExpansionError,
    run_carrion_survivor_continuation_v165_preterminal_target_dataset_expansion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the diagnostics-only v165 carrion survivor-continuation "
            "preterminal target-dataset expansion from replay-verified v164 "
            "tied-set branch evidence. The command creates no model artifact, "
            "runs no training, and performs no live A/B."
        )
    )
    parser.add_argument("--v164-report", type=Path, default=DEFAULT_V164_REPORT_PATH)
    parser.add_argument("--base-dataset", type=Path, default=DEFAULT_V158_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--target-dataset-output",
        type=Path,
        default=DEFAULT_TARGET_DATASET_OUTPUT_PATH,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v165_preterminal_target_dataset_expansion(
            v164_report_path=args.v164_report,
            base_dataset_path=args.base_dataset,
            output_path=args.output,
            target_dataset_output_path=args.target_dataset_output,
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationV165PreterminalTargetDatasetExpansionError,
    ) as exc:
        raise SystemExit(
            "failed to emit v165 carrion survivor-continuation preterminal "
            f"target dataset expansion: {exc}"
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
    preterminal = report.get("preterminal_rows")
    preterminal_payload = preterminal if isinstance(preterminal, dict) else {}
    route = report.get("route_recommendation")
    route_payload = route if isinstance(route, dict) else {}
    print(
        "carrion_survivor_continuation_v165_preterminal_target_dataset_expansion_report="
        f"{output}"
    )
    print(f"target_dataset={target_dataset_output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={source_payload.get('passed')}")
    print(f"combined_row_count={dataset_payload.get('combined_row_count')}")
    print(f"base_row_count={dataset_payload.get('base_row_count')}")
    print(f"preterminal_row_count={dataset_payload.get('preterminal_row_count')}")
    print(f"unique_winner_count={preterminal_payload.get('unique_winner_count')}")
    print(
        "multi_action_safe_set_count="
        f"{preterminal_payload.get('multi_action_safe_set_count')}"
    )
    print(f"unresolved_count={preterminal_payload.get('unresolved_count')}")
    print(
        "dominant_safe_action_share="
        f"{preterminal_payload.get('dominant_safe_action_share')}"
    )
    print(f"dataset_digest={dataset_payload.get('dataset_digest')}")
    print(f"recommended_next_route={route_payload.get('recommended_next_route')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"live_ab_allowed={report.get('live_ab_allowed')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(
        "runtime_action_selection_changed="
        f"{report.get('runtime_action_selection_changed')}"
    )
    print(f"exact_digest={report.get('exact_digest')}")


if __name__ == "__main__":
    main()
