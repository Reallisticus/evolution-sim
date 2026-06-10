from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.safe_archive_failure_autopsy import (
    DEFAULT_BP3_BRANCH_EVIDENCE_PATH,
    DEFAULT_BP3_DATASET_PATH,
    DEFAULT_BP3_FAILURE_AUTOPSY_OUTPUT_PATH,
    DEFAULT_BP3_TRAIN_EVAL_REPORT_PATH,
    load_json_report,
    load_jsonl_rows,
)
from evolution_sim.mind.safe_archive_sequence_context_audit import (
    DEFAULT_BP3_SEQUENCE_CONTEXT_AUDIT_OUTPUT_PATH,
    build_safe_archive_sequence_context_audit_report,
    write_safe_archive_sequence_context_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnostics-only bp3 safe-archive sequence/rollout-context "
            "audit report. This command compares one-step support aliases with "
            "available public sequence context and never trains or promotes."
        )
    )
    parser.add_argument(
        "--autopsy-report",
        type=Path,
        default=DEFAULT_BP3_FAILURE_AUTOPSY_OUTPUT_PATH,
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
        "--output",
        type=Path,
        default=DEFAULT_BP3_SEQUENCE_CONTEXT_AUDIT_OUTPUT_PATH,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        autopsy_report = load_json_report(args.autopsy_report)
        train_eval_report = load_json_report(args.train_eval_report)
        dataset_rows = load_jsonl_rows(args.dataset)
        branch_evidence = load_json_report(args.branch_evidence)
        report = build_safe_archive_sequence_context_audit_report(
            autopsy_report=autopsy_report,
            train_eval_report=train_eval_report,
            dataset_rows=dataset_rows,
            branch_evidence=branch_evidence,
            input_paths={
                "autopsy_report": args.autopsy_report,
                "train_eval_report": args.train_eval_report,
                "dataset": args.dataset,
                "branch_evidence": args.branch_evidence,
            },
        )
        write_safe_archive_sequence_context_audit_report(report, args.output)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"failed to build safe archive sequence context audit: {exc}"
        ) from exc
    comparison = report.get("action_only_vs_sequence_support_comparison")
    comparison_payload = comparison if isinstance(comparison, dict) else {}
    sequence = comparison_payload.get("sequence_context")
    sequence_payload = sequence if isinstance(sequence, dict) else {}
    action_only = comparison_payload.get("action_only")
    action_payload = action_only if isinstance(action_only, dict) else {}
    print(f"safe_archive_sequence_context_audit_report={args.output}")
    print(f"classification={report['classification']['primary']}")
    print(f"audit_trace_count={report.get('audit_trace_count')}")
    print(
        "action_only_zero_distance_alias_count="
        f"{action_payload.get('zero_distance_alias_count')}"
    )
    print(
        "sequence_context_available_count="
        f"{sequence_payload.get('current_override_previous_context_available_count')}"
    )
    print(
        "sequence_context_separated_alias_count="
        f"{sequence_payload.get('separated_alias_count')}"
    )
    print(
        "sequence_primary_limitation="
        f"{sequence_payload.get('primary_limitation')}"
    )
    print(f"recommended_next_route={report.get('recommended_next_route')}")
    print(f"training_authorized={report.get('training_authorized')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")


if __name__ == "__main__":
    main()
