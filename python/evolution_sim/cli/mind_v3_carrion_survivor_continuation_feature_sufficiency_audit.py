from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_feature_sufficiency_audit import (
    DEFAULT_MATERIAL_CONFLICT_ROW_REDUCTION,
    DEFAULT_MIN_NN_OVER_TRIVIAL_MARGIN,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V154_DATASET_PATH,
    DEFAULT_V154_REPORT_PATH,
    DEFAULT_V155_REPORT_PATH,
    CarrionSurvivorContinuationFeatureSufficiencyError,
    run_carrion_survivor_continuation_feature_sufficiency_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v156 carrion survivor-continuation "
            "public feature sufficiency audit. The command creates no "
            "training artifact and only reports whether public pre-decision "
            "features can disambiguate v154 labels after the v155 closeout."
        )
    )
    parser.add_argument("--v154-report", type=Path, default=DEFAULT_V154_REPORT_PATH)
    parser.add_argument("--v154-dataset", type=Path, default=DEFAULT_V154_DATASET_PATH)
    parser.add_argument("--v155-report", type=Path, default=DEFAULT_V155_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--min-nn-over-trivial-margin",
        type=float,
        default=DEFAULT_MIN_NN_OVER_TRIVIAL_MARGIN,
        help=(
            "Minimum leave-one-seed-out nearest-neighbor accuracy margin over "
            "the best action-only or mask-only baseline."
        ),
    )
    parser.add_argument(
        "--material-conflict-row-reduction",
        type=float,
        default=DEFAULT_MATERIAL_CONFLICT_ROW_REDUCTION,
        help=(
            "Required conflicting-row reduction against the current "
            "observation/action-mask baseline before support can be ready."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_feature_sufficiency_audit(
            v154_report_path=args.v154_report,
            v154_dataset_path=args.v154_dataset,
            v155_report_path=args.v155_report,
            output_path=args.output,
            min_nn_over_trivial_margin=args.min_nn_over_trivial_margin,
            material_conflict_row_reduction=args.material_conflict_row_reduction,
        )
    except (OSError, ValueError, CarrionSurvivorContinuationFeatureSufficiencyError) as exc:
        raise SystemExit(
            "failed to run v156 carrion survivor-continuation public feature "
            f"sufficiency audit: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: dict[str, object], output: Path) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    validation = report.get("source_validation")
    validation_payload = validation if isinstance(validation, dict) else {}
    best = report.get("best_feature_policy")
    best_payload = best if isinstance(best, dict) else {}
    print(f"carrion_survivor_continuation_feature_sufficiency_report={output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={validation_payload.get('passed')}")
    print(f"feature_policy_count={report.get('feature_policy_count')}")
    print(f"best_feature_policy={best_payload.get('policy_id')}")
    print(f"best_policy_support_ready={best_payload.get('support_ready_no_training')}")
    print(f"artifact_created={report.get('artifact_created')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"shadow_live_ab_ran={report.get('shadow_live_ab_ran')}")
    print(f"training_authorized={report.get('training_authorized')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"runtime_promotion_allowed={report.get('runtime_promotion_allowed')}")
    print(f"exact_digest={report.get('exact_digest')}")


if __name__ == "__main__":
    main()
