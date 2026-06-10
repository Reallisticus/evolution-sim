from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    DEFAULT_ARTIFACT_OUTPUT_PATH,
    DEFAULT_MIN_LABEL_COUNT,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V154_DATASET_PATH,
    DEFAULT_V154_REPORT_PATH,
    MAX_DOMINANT_LABEL_ACTION_SHARE,
    CarrionSurvivorContinuationTrainEvalError,
    run_carrion_survivor_continuation_train_eval,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the opt-in v155 carrion survivor-continuation archive "
            "train/eval slice. The command fails closed before artifact "
            "creation unless v154 source gates and pre-training diagnostics pass."
        )
    )
    parser.add_argument("--v154-report", type=Path, default=DEFAULT_V154_REPORT_PATH)
    parser.add_argument("--v154-dataset", type=Path, default=DEFAULT_V154_DATASET_PATH)
    parser.add_argument(
        "--artifact-output",
        type=Path,
        default=DEFAULT_ARTIFACT_OUTPUT_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--min-label-count",
        type=int,
        default=DEFAULT_MIN_LABEL_COUNT,
    )
    parser.add_argument(
        "--max-dominant-label-action-share",
        type=float,
        default=MAX_DOMINANT_LABEL_ACTION_SHARE,
    )
    parser.add_argument(
        "--expected-dataset-digest",
        default=None,
        help="Optional exact v154 dataset digest to require in addition to report self-consistency.",
    )
    parser.add_argument(
        "--min-nn-over-trivial-margin",
        type=float,
        default=0.05,
        help="Minimum leave-one-seed-out nearest-neighbor accuracy margin over the best action/mask baseline.",
    )
    parser.add_argument(
        "--max-trivial-baseline-accuracy",
        type=float,
        default=0.80,
        help="Fail closed when action-only or mask-only baseline reaches this accuracy.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="After successful pre-training review, create the diagnostic artifact but skip shadow/live evaluation.",
    )
    parser.add_argument(
        "--skip-live-ab",
        action="store_true",
        help="Run shadow evaluation only; live A/B remains skipped even if shadow passes.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_train_eval(
            v154_report_path=args.v154_report,
            v154_dataset_path=args.v154_dataset,
            artifact_output_path=args.artifact_output,
            output_path=args.output,
            min_label_count=args.min_label_count,
            max_dominant_label_action_share=args.max_dominant_label_action_share,
            min_nn_over_trivial_margin=args.min_nn_over_trivial_margin,
            max_trivial_baseline_accuracy=args.max_trivial_baseline_accuracy,
            expected_dataset_digest=args.expected_dataset_digest,
            run_evaluation=not bool(args.skip_evaluation),
            run_live_ab=not bool(args.skip_live_ab),
        )
    except (OSError, ValueError, CarrionSurvivorContinuationTrainEvalError) as exc:
        raise SystemExit(
            "failed to run v155 carrion survivor-continuation train/eval: "
            f"{exc}"
        ) from exc
    _print_summary(report, args.output, args.artifact_output)


def _print_summary(
    report: dict[str, object],
    output: Path,
    artifact_output: Path,
) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    validation = report.get("source_validation")
    validation_payload = validation if isinstance(validation, dict) else {}
    pretraining = report.get("pretraining_review")
    pretraining_payload = pretraining if isinstance(pretraining, dict) else {}
    artifact = report.get("artifact")
    artifact_payload = artifact if isinstance(artifact, dict) else {}
    training = report.get("training")
    training_payload = training if isinstance(training, dict) else {}
    acceptance = report.get("acceptance")
    acceptance_payload = acceptance if isinstance(acceptance, dict) else {}
    print(f"carrion_survivor_continuation_train_eval_report={output}")
    print(f"carrion_survivor_continuation_train_eval_artifact={artifact_output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={validation_payload.get('passed')}")
    print(f"pretraining_review_passed={pretraining_payload.get('passed')}")
    print(f"artifact_created={artifact_payload.get('created')}")
    print(f"diagnostic_training_ran={training_payload.get('diagnostic_training_ran')}")
    print(f"acceptance_passed={acceptance_payload.get('passed')}")
    print(f"training_authorized={report.get('training_authorized')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"runtime_promotion_allowed={report.get('runtime_promotion_allowed')}")
    print(f"exact_digest={report.get('exact_digest')}")


if __name__ == "__main__":
    main()
