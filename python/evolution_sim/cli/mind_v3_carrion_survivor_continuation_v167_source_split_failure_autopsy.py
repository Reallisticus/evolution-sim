from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v167_source_split_failure_autopsy import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V165_DATASET_PATH,
    DEFAULT_V166_ARTIFACT_PATH,
    DEFAULT_V166_REPORT_PATH,
    EXPECTED_V165_DATASET_DIGEST,
    EXPECTED_V166_ARTIFACT_DIGEST,
    EXPECTED_V166_EXACT_DIGEST,
    CarrionSurvivorContinuationV167SourceSplitFailureAutopsyError,
    run_carrion_survivor_continuation_v167_source_split_failure_autopsy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v167 source-split failure autopsy for "
            "the v166 carrion survivor-continuation scorer. The command does "
            "not retrain, tune k, tune thresholds, run shadow eval, create a "
            "runtime artifact, or change runtime action selection."
        )
    )
    parser.add_argument("--v166-report", type=Path, default=DEFAULT_V166_REPORT_PATH)
    parser.add_argument(
        "--v166-artifact",
        type=Path,
        default=DEFAULT_V166_ARTIFACT_PATH,
    )
    parser.add_argument("--v165-dataset", type=Path, default=DEFAULT_V165_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-v166-exact-digest",
        default=EXPECTED_V166_EXACT_DIGEST,
        help="Expected embedded stable digest for the v166 report.",
    )
    parser.add_argument(
        "--expected-v166-artifact-digest",
        default=EXPECTED_V166_ARTIFACT_DIGEST,
        help="Expected embedded stable digest for the v166 diagnostics artifact.",
    )
    parser.add_argument(
        "--expected-v165-dataset-digest",
        default=EXPECTED_V165_DATASET_DIGEST,
        help="Expected stable digest for the v165 expanded target dataset.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v167_source_split_failure_autopsy(
            v166_report_path=args.v166_report,
            v166_artifact_path=args.v166_artifact,
            v165_dataset_path=args.v165_dataset,
            output_path=args.output,
            expected_v166_exact_digest=args.expected_v166_exact_digest,
            expected_v166_artifact_digest=args.expected_v166_artifact_digest,
            expected_v165_dataset_digest=args.expected_v165_dataset_digest,
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationV167SourceSplitFailureAutopsyError,
    ) as exc:
        raise SystemExit(
            "failed to run v167 carrion survivor-continuation source-split "
            f"failure autopsy: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: dict[str, object], output: Path) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    source = report.get("source_validation")
    source_payload = source if isinstance(source, dict) else {}
    failure = report.get("failure_mode_summary")
    failure_payload = failure if isinstance(failure, dict) else {}
    route = report.get("route_recommendation")
    route_payload = route if isinstance(route, dict) else {}
    metrics = report.get("v166_source_split_metrics")
    metrics_payload = metrics if isinstance(metrics, dict) else {}
    print(
        "carrion_survivor_continuation_v167_source_split_failure_autopsy_report="
        f"{output}"
    )
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={source_payload.get('passed')}")
    print(
        "recomputed_v166_source_split_validation_passed="
        f"{report.get('recomputed_v166_source_split_validation', {}).get('passed')}"
    )
    print(f"primary_failure_mode={failure_payload.get('primary_failure_mode')}")
    print(
        "zero_safe_hit_source_seeds="
        f"{failure_payload.get('zero_safe_hit_source_seeds')}"
    )
    print(
        "safe_hit_margin_over_best_trivial="
        f"{metrics_payload.get('safe_hit_margin_over_best_trivial')}"
    )
    print(f"recommended_next_route={route_payload.get('recommended_next_route')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"k_tuning_ran={report.get('k_tuning_ran')}")
    print(f"threshold_tuning_ran={report.get('threshold_tuning_ran')}")
    print(f"shadow_eval_ran={report.get('shadow_eval_ran')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(
        "runtime_action_selection_changed="
        f"{report.get('runtime_action_selection_changed')}"
    )
    print(f"exact_digest={report.get('exact_digest')}")


if __name__ == "__main__":
    main()
