from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v173_source_split_scorer import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V172_DATASET_PATH,
    DEFAULT_V172_REPORT_PATH,
    EXPECTED_V172_DATASET_DIGEST,
    EXPECTED_V172_EXACT_DIGEST,
    CarrionSurvivorContinuationV173SourceSplitScorerError,
    run_carrion_survivor_continuation_v173_source_split_scorer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run diagnostics-only v173 source-split scorer evaluation over the "
            "v172 carrion survivor-continuation replay target dataset. This "
            "does not train, retrain, create a runtime artifact, run "
            "shadow/live evaluation, tune thresholds or k, integrate runtime "
            "actions, or authorize promotion."
        )
    )
    parser.add_argument("--v172-report", type=Path, default=DEFAULT_V172_REPORT_PATH)
    parser.add_argument("--v172-dataset", type=Path, default=DEFAULT_V172_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-v172-exact-digest",
        default=EXPECTED_V172_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v172-dataset-digest",
        default=EXPECTED_V172_DATASET_DIGEST,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v173_source_split_scorer(
            v172_report_path=args.v172_report,
            v172_dataset_path=args.v172_dataset,
            output_path=args.output,
            expected_v172_exact_digest=args.expected_v172_exact_digest,
            expected_v172_dataset_digest=args.expected_v172_dataset_digest,
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationV173SourceSplitScorerError,
    ) as exc:
        raise SystemExit(
            "failed to run v173 carrion survivor-continuation source-split "
            f"scorer diagnostics: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: dict[str, object], output: Path) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    source = report.get("source_validation")
    source_payload = source if isinstance(source, dict) else {}
    split = report.get("source_split_evaluation")
    split_payload = split if isinstance(split, dict) else {}
    singleton = split_payload.get("singleton_scorer")
    singleton_payload = singleton if isinstance(singleton, dict) else {}
    set_valued = split_payload.get("set_valued_ranker")
    set_payload = set_valued if isinstance(set_valued, dict) else {}
    baselines = split_payload.get("trivial_and_negative_controls")
    baseline_payload = baselines if isinstance(baselines, dict) else {}
    print(f"carrion_survivor_continuation_v173_source_split_scorer_report={output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={source_payload.get('passed')}")
    print(f"row_count={split_payload.get('row_count')}")
    print(f"unique_row_count={split_payload.get('unique_row_count')}")
    print(f"tied_row_count={split_payload.get('tied_row_count')}")
    print(f"singleton_safe_hit_rate={singleton_payload.get('safe_hit_rate')}")
    print(
        "safe_hit_margin_over_best_trivial="
        f"{singleton_payload.get('safe_hit_margin_over_best_trivial')}"
    )
    print(
        "robust_winner_hit_rate="
        f"{singleton_payload.get('robust_winner_hit_rate')}"
    )
    print(f"set_hit_rate={set_payload.get('set_hit_rate')}")
    print(f"tied_row_set_hit_rate={set_payload.get('tied_row_set_hit_rate')}")
    print(
        "unsupported_prediction_count="
        f"{singleton_payload.get('unsupported_prediction_count')}"
    )
    print(f"no_prediction_count={singleton_payload.get('no_prediction_count')}")
    print(
        "dominant_singleton_predicted_action_share="
        f"{singleton_payload.get('dominant_singleton_predicted_action_share')}"
    )
    print(
        "average_candidate_set_width="
        f"{set_payload.get('average_candidate_set_width')}"
    )
    print(f"full_set_share={set_payload.get('full_set_share')}")
    print(
        "best_trivial_singleton_hit_rate="
        f"{baseline_payload.get('best_trivial_singleton_hit_rate')}"
    )
    print(f"first_failing_seed={split_payload.get('first_failing_seed')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"scorer_retraining_ran={report.get('scorer_retraining_ran')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"shadow_eval_ran={report.get('shadow_eval_ran')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"exact_digest={report.get('exact_digest')}")


if __name__ == "__main__":
    main()
