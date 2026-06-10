from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v166_source_split_action_value_scorer import (
    DEFAULT_ARTIFACT_OUTPUT_PATH,
    DEFAULT_MAX_DOMINANT_PREDICTED_ACTION_SHARE,
    DEFAULT_MIN_SAFE_HIT_MARGIN_OVER_BEST_TRIVIAL,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V165_DATASET_PATH,
    DEFAULT_V165_REPORT_PATH,
    EXPECTED_V165_DATASET_DIGEST,
    EXPECTED_V165_EXACT_DIGEST,
    CarrionSurvivorContinuationV166SourceSplitActionValueScorerError,
    run_carrion_survivor_continuation_v166_source_split_action_value_scorer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v166 carrion survivor-continuation "
            "source-split action-value scorer retraining diagnostic. The "
            "command writes a diagnostics artifact only and never creates a "
            "runtime artifact or live override path."
        )
    )
    parser.add_argument("--v165-report", type=Path, default=DEFAULT_V165_REPORT_PATH)
    parser.add_argument("--v165-dataset", type=Path, default=DEFAULT_V165_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--artifact-output",
        type=Path,
        default=DEFAULT_ARTIFACT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--expected-v165-exact-digest",
        default=EXPECTED_V165_EXACT_DIGEST,
        help="Expected embedded stable digest for the v165 source report.",
    )
    parser.add_argument(
        "--expected-v165-dataset-digest",
        default=EXPECTED_V165_DATASET_DIGEST,
        help="Expected stable digest for the v165 expanded target dataset.",
    )
    parser.add_argument(
        "--max-dominant-predicted-action-share",
        type=float,
        default=DEFAULT_MAX_DOMINANT_PREDICTED_ACTION_SHARE,
        help="Maximum dominant source-split predicted-action share before closing.",
    )
    parser.add_argument(
        "--min-safe-hit-margin-over-best-trivial",
        type=float,
        default=DEFAULT_MIN_SAFE_HIT_MARGIN_OVER_BEST_TRIVIAL,
        help="Required source-split safe-hit margin over the best trivial baseline.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v166_source_split_action_value_scorer(
            v165_report_path=args.v165_report,
            v165_dataset_path=args.v165_dataset,
            output_path=args.output,
            artifact_output_path=args.artifact_output,
            expected_v165_exact_digest=args.expected_v165_exact_digest,
            expected_v165_dataset_digest=args.expected_v165_dataset_digest,
            max_dominant_predicted_action_share=(
                args.max_dominant_predicted_action_share
            ),
            min_safe_hit_margin_over_best_trivial=(
                args.min_safe_hit_margin_over_best_trivial
            ),
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationV166SourceSplitActionValueScorerError,
    ) as exc:
        raise SystemExit(
            "failed to run v166 carrion survivor-continuation source-split "
            f"action-value scorer diagnostic: {exc}"
        ) from exc
    _print_summary(report, args.output, args.artifact_output)


def _print_summary(
    report: dict[str, object],
    output: Path,
    artifact_output: Path,
) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    source = report.get("source_validation")
    source_payload = source if isinstance(source, dict) else {}
    split = report.get("source_split_diagnostics")
    split_payload = split if isinstance(split, dict) else {}
    route = report.get("route_recommendation")
    route_payload = route if isinstance(route, dict) else {}
    artifact = report.get("artifact")
    artifact_payload = artifact if isinstance(artifact, dict) else {}
    print(
        "carrion_survivor_continuation_v166_source_split_action_value_scorer_report="
        f"{output}"
    )
    print(f"diagnostic_scorer_artifact={artifact_output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={source_payload.get('passed')}")
    print(f"prediction_count={split_payload.get('prediction_count')}")
    print(f"no_prediction_count={split_payload.get('no_prediction_count')}")
    print(
        "unsupported_prediction_count="
        f"{split_payload.get('unsupported_prediction_count')}"
    )
    print(
        "dominant_predicted_action_share="
        f"{split_payload.get('dominant_predicted_action_share')}"
    )
    print(f"safe_hit_rate={split_payload.get('safe_hit_rate')}")
    print(
        "best_trivial_baseline_hit_rate="
        f"{split_payload.get('best_trivial_baseline_hit_rate')}"
    )
    print(
        "safe_hit_margin_over_best_trivial="
        f"{split_payload.get('safe_hit_margin_over_best_trivial')}"
    )
    print(
        "first_zero_safe_hit_preterminal_source_seed="
        f"{split_payload.get('first_zero_safe_hit_preterminal_source_seed')}"
    )
    print(
        "future_diagnostics_only_shadow_evaluation_recommended="
        f"{route_payload.get('future_diagnostics_only_shadow_evaluation_recommended')}"
    )
    print(f"artifact_digest={artifact_payload.get('artifact_digest')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"runtime_promotion_allowed={report.get('runtime_promotion_allowed')}")
    print(
        "runtime_action_selection_changed="
        f"{report.get('runtime_action_selection_changed')}"
    )
    print(f"dataset_digest={report.get('dataset_digest')}")
    print(f"exact_digest={report.get('exact_digest')}")


if __name__ == "__main__":
    main()
