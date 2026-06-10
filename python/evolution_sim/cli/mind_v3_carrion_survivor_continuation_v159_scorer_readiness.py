from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v159_scorer_readiness import (
    DEFAULT_MAX_DOMINANT_PROPOSED_ACTION_SHARE,
    DEFAULT_MIN_EXACT_SHADOW_MINUS_BEST_TRIVIAL_COVERAGE_SHARE,
    DEFAULT_MIN_EXACT_SHADOW_SAFE_HIT_SHARE,
    DEFAULT_MIN_NEAR_EXACT_MASK_COMPARATOR_COVERAGE_SHARE,
    DEFAULT_MIN_NEAR_EXACT_MASK_SHADOW_SAFE_HIT_SHARE,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V154_DATASET_PATH,
    DEFAULT_V154_REPORT_PATH,
    DEFAULT_V155_REPORT_PATH,
    DEFAULT_V156_REPORT_PATH,
    DEFAULT_V157_REPORT_PATH,
    DEFAULT_V158_DATASET_PATH,
    DEFAULT_V158_REPORT_PATH,
    CarrionSurvivorContinuationV159ScorerReadinessError,
    run_carrion_survivor_continuation_v159_scorer_readiness,
)
from evolution_sim.mind.carrion_survivor_continuation_action_value_target_dataset import (
    DEFAULT_MAX_DOMINANT_SAFE_ACTION_SHARE,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Emit the diagnostics-only v159 carrion survivor-continuation "
            "action-value scorer-readiness report. The command creates no "
            "model artifact, runs no training, and performs no runtime action "
            "selection."
        )
    )
    parser.add_argument("--v154-report", type=Path, default=DEFAULT_V154_REPORT_PATH)
    parser.add_argument("--v154-dataset", type=Path, default=DEFAULT_V154_DATASET_PATH)
    parser.add_argument("--v155-report", type=Path, default=DEFAULT_V155_REPORT_PATH)
    parser.add_argument("--v156-report", type=Path, default=DEFAULT_V156_REPORT_PATH)
    parser.add_argument("--v157-report", type=Path, default=DEFAULT_V157_REPORT_PATH)
    parser.add_argument("--v158-report", type=Path, default=DEFAULT_V158_REPORT_PATH)
    parser.add_argument("--v158-dataset", type=Path, default=DEFAULT_V158_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--max-dominant-safe-action-share",
        type=float,
        default=DEFAULT_MAX_DOMINANT_SAFE_ACTION_SHARE,
        help="Maximum dominant safe-action support share before support is collapsed.",
    )
    parser.add_argument(
        "--max-dominant-proposed-action-share",
        type=float,
        default=DEFAULT_MAX_DOMINANT_PROPOSED_ACTION_SHARE,
        help="Maximum dominant exact-shadow proposed-action share.",
    )
    parser.add_argument(
        "--min-exact-shadow-safe-hit-share",
        type=float,
        default=DEFAULT_MIN_EXACT_SHADOW_SAFE_HIT_SHARE,
        help="Minimum exact-shadow top-value safe-hit share.",
    )
    parser.add_argument(
        "--min-exact-shadow-minus-best-trivial-coverage-share",
        type=float,
        default=DEFAULT_MIN_EXACT_SHADOW_MINUS_BEST_TRIVIAL_COVERAGE_SHARE,
        help="Minimum exact-shadow safe-hit margin over the best fixed-action baseline.",
    )
    parser.add_argument(
        "--min-near-exact-mask-comparator-coverage-share",
        type=float,
        default=DEFAULT_MIN_NEAR_EXACT_MASK_COMPARATOR_COVERAGE_SHARE,
        help="Minimum row share with a same-action-mask comparator.",
    )
    parser.add_argument(
        "--min-near-exact-mask-shadow-safe-hit-share",
        type=float,
        default=DEFAULT_MIN_NEAR_EXACT_MASK_SHADOW_SAFE_HIT_SHARE,
        help="Minimum all-row safe-hit share for same-mask leave-one-row shadow scoring.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v159_scorer_readiness(
            v154_report_path=args.v154_report,
            v154_dataset_path=args.v154_dataset,
            v155_report_path=args.v155_report,
            v156_report_path=args.v156_report,
            v157_report_path=args.v157_report,
            v158_report_path=args.v158_report,
            v158_dataset_path=args.v158_dataset,
            output_path=args.output,
            max_dominant_safe_action_share=args.max_dominant_safe_action_share,
            max_dominant_proposed_action_share=args.max_dominant_proposed_action_share,
            min_exact_shadow_safe_hit_share=args.min_exact_shadow_safe_hit_share,
            min_exact_shadow_minus_best_trivial_coverage_share=(
                args.min_exact_shadow_minus_best_trivial_coverage_share
            ),
            min_near_exact_mask_comparator_coverage_share=(
                args.min_near_exact_mask_comparator_coverage_share
            ),
            min_near_exact_mask_shadow_safe_hit_share=(
                args.min_near_exact_mask_shadow_safe_hit_share
            ),
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationV159ScorerReadinessError,
    ) as exc:
        raise SystemExit(
            "failed to emit v159 carrion survivor-continuation "
            f"scorer-readiness report: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: dict[str, object], output: Path) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    source = report.get("source_validation")
    source_payload = source if isinstance(source, dict) else {}
    support = report.get("action_support")
    support_payload = support if isinstance(support, dict) else {}
    shadow = report.get("shadow_coverage")
    shadow_payload = shadow if isinstance(shadow, dict) else {}
    route = report.get("route_recommendation")
    route_payload = route if isinstance(route, dict) else {}
    print(f"carrion_survivor_continuation_v159_scorer_readiness_report={output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={source_payload.get('passed')}")
    print(
        "action_support_non_collapsed="
        f"{support_payload.get('action_support_non_collapsed')}"
    )
    print(
        "proposed_action_non_collapsed="
        f"{support_payload.get('proposed_action_non_collapsed')}"
    )
    print(
        "dominant_safe_action_share="
        f"{support_payload.get('dominant_safe_action_share')}"
    )
    print(
        "dominant_proposed_action_share="
        f"{support_payload.get('dominant_proposed_action_share')}"
    )
    print(
        "exact_shadow_safe_hit_share="
        f"{shadow_payload.get('exact_shadow_safe_hit_share')}"
    )
    print(
        "near_exact_mask_shadow_safe_hit_share_all_rows="
        f"{shadow_payload.get('near_exact_mask_shadow_safe_hit_share_all_rows')}"
    )
    print(
        "future_opt_in_training_diagnostic_recommended="
        f"{route_payload.get('future_opt_in_training_diagnostic_recommended')}"
    )
    print(f"training_ran={report.get('training_ran')}")
    print(f"artifact_created={report.get('artifact_created')}")
    print(f"training_authorized={report.get('training_authorized')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"runtime_promotion_allowed={report.get('runtime_promotion_allowed')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"dataset_digest={report.get('dataset_digest')}")
    print(f"exact_digest={report.get('exact_digest')}")


if __name__ == "__main__":
    main()
