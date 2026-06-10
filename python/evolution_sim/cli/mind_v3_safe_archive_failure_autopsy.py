from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.safe_archive_failure_autopsy import (
    DEFAULT_BP3_BRANCH_EVIDENCE_PATH,
    DEFAULT_BP3_DATASET_PATH,
    DEFAULT_BP3_DIAGNOSTIC_ARTIFACT_PATH,
    DEFAULT_BP3_FAILURE_AUTOPSY_OUTPUT_PATH,
    DEFAULT_BP3_TRAIN_EVAL_REPORT_PATH,
    build_safe_archive_failure_autopsy_report,
    load_json_report,
    load_jsonl_rows,
    write_safe_archive_failure_autopsy_report,
)
from evolution_sim.mind.support_gated_residual import (
    load_support_gated_residual_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnostics-only autopsy report for the bp3 safe-archive "
            "support-gated residual failure. This command reruns only strict "
            "failing cases and never trains or promotes."
        )
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_BP3_DIAGNOSTIC_ARTIFACT_PATH,
    )
    parser.add_argument(
        "--train-eval-report",
        type=Path,
        default=DEFAULT_BP3_TRAIN_EVAL_REPORT_PATH,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_BP3_DATASET_PATH,
    )
    parser.add_argument(
        "--branch-evidence",
        type=Path,
        default=DEFAULT_BP3_BRANCH_EVIDENCE_PATH,
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_BP3_FAILURE_AUTOPSY_OUTPUT_PATH,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact = load_support_gated_residual_artifact(args.artifact)
    train_eval_report = load_json_report(args.train_eval_report)
    dataset_rows = load_jsonl_rows(args.dataset)
    branch_evidence = load_json_report(args.branch_evidence)
    report = build_safe_archive_failure_autopsy_report(
        artifact=artifact,
        train_eval_report=train_eval_report,
        dataset_rows=dataset_rows,
        branch_evidence=branch_evidence,
        ticks=int(args.ticks),
    )
    write_safe_archive_failure_autopsy_report(report, args.output)
    aggregates = report.get("aggregates")
    aggregate_payload = aggregates if isinstance(aggregates, dict) else {}
    causes = report.get("failure_cause_assessment")
    cause_payload = causes if isinstance(causes, dict) else {}
    print(f"safe_archive_failure_autopsy_report={args.output}")
    print(f"classification={report['classification']['primary']}")
    print(f"override_count={aggregate_payload.get('override_count')}")
    print(f"recommended_next_route={report.get('recommended_next_route')}")
    print(f"failure_cause_assessment={cause_payload}")
    print(f"training_authorized={report.get('training_authorized')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")


if __name__ == "__main__":
    main()
