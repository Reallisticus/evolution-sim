from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.candidate_campaign import (
    DEFAULT_MIN_SAFE_LABEL_COUNT,
    CandidateCampaignError,
    run_safe_archive_train_eval,
)

DEFAULT_CARRION_ARCHIVE_REPORT_PATH = Path(
    "output/mind/shards/v148-carrion/merged-report.json"
)
DEFAULT_CARRION_ARCHIVE_DATASET_PATH = Path(
    "output/mind/shards/v148-carrion/merged-dataset.jsonl"
)
DEFAULT_ARTIFACT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v149-carrion-specific-archive-support-gated-artifact.json"
)
DEFAULT_OUTPUT_PATH = Path(
    "output/mind/mind-v3-v149-carrion-specific-archive-train-eval.json"
)
EXPECTED_CARRION_ARCHIVE_DATASET_DIGEST = (
    "5ed2e76c5f98a6a55a6e4a8ea9b5fda91783faaa6ae4ba600137fd9fa80b5161"
)
EXPECTED_CARRION_ARCHIVE_BRANCH_EVIDENCE_DIGEST = (
    "e85af9912dada7bcff871236a7fed397593cbfd8dd9ada13b3f2afd374ca1c57"
)
EXPECTED_CARRION_ARCHIVE_SAFE_LABEL_COUNT = 88
CARRION_SPECIFIC_LEAKAGE_STRICT_HELDOUT_SEEDS = (13, 19, 29, 37, 41, 43)
CARRION_SPECIFIC_SUPPORT_READY_CLASSIFICATION = (
    "carrion_specific_archive_support_ready_no_training"
)
CARRION_SPECIFIC_TRAIN_EVAL_SCHEMA_VERSION = (
    "m3_carrion_specific_archive_train_eval_report_v1"
)
CARRION_SPECIFIC_TRAIN_EVAL_POLICY = (
    "diagnostics_only_m3_carrion_specific_archive_support_gated_train_eval_v1"
)
CARRION_SPECIFIC_CANDIDATE_ID = (
    "carrion_specific_archive_support_gated_residual"
)
CARRION_SPECIFIC_SUPPORT_MODE = "carrion_specific_archive_support"
CARRION_SPECIFIC_TRAINING_SOURCE_LABEL = "carrion_specific_archive"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one diagnostics-only Mind v3 carrion-specific archive "
            "support-gated train/eval slice. This writes a non-promoted "
            "artifact and report, and never authorizes promotion."
        )
    )
    parser.add_argument(
        "--carrion-archive-report",
        type=Path,
        default=DEFAULT_CARRION_ARCHIVE_REPORT_PATH,
    )
    parser.add_argument(
        "--carrion-archive-dataset",
        type=Path,
        default=DEFAULT_CARRION_ARCHIVE_DATASET_PATH,
    )
    parser.add_argument(
        "--branch-evidence-report",
        type=Path,
        default=None,
        help=(
            "Optional separate branch-evidence report. Defaults to "
            "--carrion-archive-report because the v148 merged report carries "
            "branch_results and branch_evidence_digest."
        ),
    )
    parser.add_argument(
        "--artifact-output",
        type=Path,
        default=DEFAULT_ARTIFACT_OUTPUT_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-dataset-digest",
        default=EXPECTED_CARRION_ARCHIVE_DATASET_DIGEST,
        help="Fail closed unless the dataset digest matches this v148 digest.",
    )
    parser.add_argument(
        "--expected-branch-evidence-digest",
        default=EXPECTED_CARRION_ARCHIVE_BRANCH_EVIDENCE_DIGEST,
        help=(
            "Fail closed unless the merged branch evidence digest matches "
            "this v148 digest."
        ),
    )
    parser.add_argument(
        "--expected-safe-label-count",
        type=int,
        default=EXPECTED_CARRION_ARCHIVE_SAFE_LABEL_COUNT,
    )
    parser.add_argument(
        "--expected-min-safe-label-count",
        type=int,
        default=DEFAULT_MIN_SAFE_LABEL_COUNT,
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Build the artifact/report but skip live broad+carrion A/B.",
    )
    parser.add_argument(
        "--allow-zero-live-carrion-overrides",
        action="store_true",
        help=(
            "Do not require the support-gated residual to apply at least "
            "once on the live carrion fixture. Defaults to fail closed."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    branch_evidence_report = args.branch_evidence_report or args.carrion_archive_report
    try:
        report = run_safe_archive_train_eval(
            safe_archive_report_path=args.carrion_archive_report,
            safe_archive_dataset_path=args.carrion_archive_dataset,
            branch_evidence_report_path=branch_evidence_report,
            artifact_output_path=args.artifact_output,
            output_path=args.output,
            run_evaluation=not bool(args.skip_evaluation),
            expected_report_classification=(
                CARRION_SPECIFIC_SUPPORT_READY_CLASSIFICATION
            ),
            leakage_strict_heldout_seeds=(
                CARRION_SPECIFIC_LEAKAGE_STRICT_HELDOUT_SEEDS
            ),
            expected_safe_label_count=args.expected_safe_label_count,
            expected_min_safe_label_count=args.expected_min_safe_label_count,
            expected_dataset_digest=args.expected_dataset_digest,
            expected_branch_evidence_digest=args.expected_branch_evidence_digest,
            candidate_id=CARRION_SPECIFIC_CANDIDATE_ID,
            support_mode=CARRION_SPECIFIC_SUPPORT_MODE,
            training_source_label=CARRION_SPECIFIC_TRAINING_SOURCE_LABEL,
            report_schema_version=CARRION_SPECIFIC_TRAIN_EVAL_SCHEMA_VERSION,
            report_policy=CARRION_SPECIFIC_TRAIN_EVAL_POLICY,
            require_live_carrion_override=not bool(
                args.allow_zero_live_carrion_overrides
            ),
        )
    except (OSError, ValueError, CandidateCampaignError) as exc:
        raise SystemExit(
            f"failed to run carrion-specific archive train/eval diagnostic: {exc}"
        ) from exc
    _print_summary(report, args.output, args.artifact_output)


def _print_summary(
    report: dict[str, object],
    output: Path,
    artifact_output: Path,
) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    acceptance = report.get("acceptance")
    acceptance_payload = acceptance if isinstance(acceptance, dict) else {}
    metrics = acceptance_payload.get("metrics")
    metrics_payload = metrics if isinstance(metrics, dict) else {}
    print(f"carrion_specific_archive_train_eval_report={output}")
    print(f"carrion_specific_archive_train_eval_artifact={artifact_output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"acceptance_passed={acceptance_payload.get('passed')}")
    print(f"blocker_count={len(acceptance_payload.get('blockers', []))}")
    print(
        "dominant_requested_action_share="
        f"{metrics_payload.get('dominant_requested_action_share')}"
    )
    print(
        "heuristic_action_source_count="
        f"{metrics_payload.get('heuristic_action_source_count')}"
    )
    print(
        "carrion_applied_override_count="
        f"{metrics_payload.get('carrion_applied_override_count')}"
    )
    print(f"training_authorized={report.get('training_authorized')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")


if __name__ == "__main__":
    main()
