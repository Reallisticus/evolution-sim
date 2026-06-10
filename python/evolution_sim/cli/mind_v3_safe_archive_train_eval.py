from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.candidate_campaign import (
    BP3_SAFE_ARCHIVE_BRANCH_EVIDENCE_DIGEST,
    BP3_SAFE_ARCHIVE_DATASET_DIGEST,
    CandidateCampaignError,
    run_safe_archive_train_eval,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one diagnostics-only Mind v3 bp3 safe-archive support-gated "
            "train/eval slice. This command writes a non-promoted artifact "
            "and report, and never authorizes promotion."
        )
    )
    parser.add_argument("--safe-archive-report", type=Path, required=True)
    parser.add_argument("--safe-archive-dataset", type=Path, required=True)
    parser.add_argument("--branch-evidence-report", type=Path, required=True)
    parser.add_argument("--artifact-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-dataset-digest",
        default=BP3_SAFE_ARCHIVE_DATASET_DIGEST,
        help="Fail closed unless the dataset digest matches this bp3 digest.",
    )
    parser.add_argument(
        "--expected-branch-evidence-digest",
        default=BP3_SAFE_ARCHIVE_BRANCH_EVIDENCE_DIGEST,
        help="Fail closed unless the branch evidence digest matches this bp3 digest.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_safe_archive_train_eval(
            safe_archive_report_path=args.safe_archive_report,
            safe_archive_dataset_path=args.safe_archive_dataset,
            branch_evidence_report_path=args.branch_evidence_report,
            artifact_output_path=args.artifact_output,
            output_path=args.output,
            expected_dataset_digest=args.expected_dataset_digest,
            expected_branch_evidence_digest=args.expected_branch_evidence_digest,
        )
    except (OSError, ValueError, CandidateCampaignError) as exc:
        raise SystemExit(f"failed to run safe archive train/eval diagnostic: {exc}") from exc
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
    print(f"safe_archive_train_eval_report={output}")
    print(f"safe_archive_train_eval_artifact={artifact_output}")
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
    print(f"training_authorized={report.get('training_authorized')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")


if __name__ == "__main__":
    main()
