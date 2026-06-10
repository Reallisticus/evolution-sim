from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_action_value_audit import (
    DEFAULT_MIN_SAFE_RUNS_PER_ACTION,
    DEFAULT_MIN_SAFE_SHARE_PER_ACTION,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_UNIQUE_WINNER_SCORE_MARGIN,
    DEFAULT_V154_DATASET_PATH,
    DEFAULT_V154_REPORT_PATH,
    DEFAULT_V155_REPORT_PATH,
    DEFAULT_V156_REPORT_PATH,
    CarrionSurvivorContinuationActionValueAuditError,
    run_carrion_survivor_continuation_action_value_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v157 carrion survivor-continuation "
            "public-state action-value audit. The command creates no training "
            "artifact and only reports whether v154 single-label conflicts are "
            "benign multi-action recovery sets or true public-state ambiguity."
        )
    )
    parser.add_argument("--v154-report", type=Path, default=DEFAULT_V154_REPORT_PATH)
    parser.add_argument("--v154-dataset", type=Path, default=DEFAULT_V154_DATASET_PATH)
    parser.add_argument("--v155-report", type=Path, default=DEFAULT_V155_REPORT_PATH)
    parser.add_argument("--v156-report", type=Path, default=DEFAULT_V156_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
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
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_action_value_audit(
            v154_report_path=args.v154_report,
            v154_dataset_path=args.v154_dataset,
            v155_report_path=args.v155_report,
            v156_report_path=args.v156_report,
            output_path=args.output,
            min_safe_runs_per_action=args.min_safe_runs_per_action,
            min_safe_share_per_action=args.min_safe_share_per_action,
            unique_winner_score_margin=args.unique_winner_score_margin,
        )
    except (OSError, ValueError, CarrionSurvivorContinuationActionValueAuditError) as exc:
        raise SystemExit(
            "failed to run v157 carrion survivor-continuation public-state "
            f"action-value audit: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: dict[str, object], output: Path) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    validation = report.get("source_validation")
    validation_payload = validation if isinstance(validation, dict) else {}
    best = report.get("best_feature_policy")
    best_payload = best if isinstance(best, dict) else {}
    reduction = report.get("conflict_reduction")
    reduction_payload = reduction if isinstance(reduction, dict) else {}
    print(f"carrion_survivor_continuation_action_value_audit_report={output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={validation_payload.get('passed')}")
    print(f"feature_policy_count={report.get('feature_policy_count')}")
    print(f"best_feature_policy={best_payload.get('policy_id')}")
    print(
        "best_policy_resolvable_conflicting_rows="
        f"{best_payload.get('action_value_resolvable_conflicting_row_count')}"
    )
    print(
        "would_reduce_119_conflicting_rows="
        f"{reduction_payload.get('would_reduce_119_conflicting_rows')}"
    )
    print(f"artifact_created={report.get('artifact_created')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"shadow_live_ab_ran={report.get('shadow_live_ab_ran')}")
    print(f"training_authorized={report.get('training_authorized')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"runtime_promotion_allowed={report.get('runtime_promotion_allowed')}")
    print(f"exact_digest={report.get('exact_digest')}")


if __name__ == "__main__":
    main()
