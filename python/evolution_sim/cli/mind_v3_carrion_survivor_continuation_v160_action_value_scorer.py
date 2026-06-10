from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v160_action_value_scorer import (
    DEFAULT_ARTIFACT_OUTPUT_PATH,
    DEFAULT_MAX_DOMINANT_PREDICTED_ACTION_SHARE,
    DEFAULT_MIN_SAFE_HIT_MARGIN_OVER_BEST_TRIVIAL,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V158_DATASET_PATH,
    DEFAULT_V159_REPORT_PATH,
    CarrionSurvivorContinuationV160ActionValueScorerError,
    run_carrion_survivor_continuation_v160_action_value_scorer,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v160 carrion survivor-continuation "
            "action-value scorer training diagnostic. The command writes a "
            "serialized scorer artifact for inspection only and does not "
            "integrate with runtime action selection."
        )
    )
    parser.add_argument("--v158-dataset", type=Path, default=DEFAULT_V158_DATASET_PATH)
    parser.add_argument("--v159-report", type=Path, default=DEFAULT_V159_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--artifact-output",
        type=Path,
        default=DEFAULT_ARTIFACT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--max-dominant-predicted-action-share",
        type=float,
        default=DEFAULT_MAX_DOMINANT_PREDICTED_ACTION_SHARE,
        help="Maximum dominant LOO predicted-action share before closing the scorer.",
    )
    parser.add_argument(
        "--min-safe-hit-margin-over-best-trivial",
        type=float,
        default=DEFAULT_MIN_SAFE_HIT_MARGIN_OVER_BEST_TRIVIAL,
        help="Required safe-hit margin over the best trivial LOO baseline.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v160_action_value_scorer(
            v158_dataset_path=args.v158_dataset,
            v159_report_path=args.v159_report,
            output_path=args.output,
            artifact_output_path=args.artifact_output,
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
        CarrionSurvivorContinuationV160ActionValueScorerError,
    ) as exc:
        raise SystemExit(
            "failed to run v160 carrion survivor-continuation "
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
    loo = report.get("leave_one_row_out")
    loo_payload = loo if isinstance(loo, dict) else {}
    route = report.get("route_recommendation")
    route_payload = route if isinstance(route, dict) else {}
    artifact = report.get("artifact")
    artifact_payload = artifact if isinstance(artifact, dict) else {}
    print(f"carrion_survivor_continuation_v160_action_value_scorer_report={output}")
    print(f"diagnostic_scorer_artifact={artifact_output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={source_payload.get('passed')}")
    print(f"unsupported_prediction_count={loo_payload.get('unsupported_prediction_count')}")
    print(
        "dominant_predicted_action_share="
        f"{loo_payload.get('dominant_predicted_action_share')}"
    )
    print(f"safe_hit_rate={loo_payload.get('safe_hit_rate')}")
    print(
        "best_trivial_baseline_hit_rate="
        f"{loo_payload.get('best_trivial_baseline_hit_rate')}"
    )
    print(
        "safe_hit_margin_over_best_trivial="
        f"{loo_payload.get('safe_hit_margin_over_best_trivial')}"
    )
    print(
        "future_shadow_evaluation_recommended="
        f"{route_payload.get('future_shadow_evaluation_recommended')}"
    )
    print(f"artifact_digest={artifact_payload.get('artifact_digest')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"runtime_promotion_allowed={report.get('runtime_promotion_allowed')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"dataset_digest={report.get('dataset_digest')}")
    print(f"exact_digest={report.get('exact_digest')}")


if __name__ == "__main__":
    main()
