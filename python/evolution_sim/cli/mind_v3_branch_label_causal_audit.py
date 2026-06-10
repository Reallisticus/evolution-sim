from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.branch_label_causal_audit import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V143_DATASET_PATH,
    DEFAULT_V143_REPORT_PATH,
    DEFAULT_V144_REPORT_PATH,
    BranchLabelCausalAuditError,
    build_branch_label_causal_audit_report,
    load_inputs_for_audit,
    write_branch_label_causal_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit v144 branch-label applied overrides and downstream "
            "invalid-resolution attribution without training or changing runtime policy."
        )
    )
    parser.add_argument("--v143-report", type=Path, default=DEFAULT_V143_REPORT_PATH)
    parser.add_argument(
        "--v143-dataset",
        type=Path,
        default=DEFAULT_V143_DATASET_PATH,
    )
    parser.add_argument("--v144-report", type=Path, default=DEFAULT_V144_REPORT_PATH)
    parser.add_argument(
        "--artifact",
        type=Path,
        default=None,
        help="Optional v144 artifact path; defaults to artifact_output in the v144 report.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        (
            v143_report,
            dataset_rows,
            v144_report,
            artifact,
            artifact_path,
        ) = load_inputs_for_audit(
            v143_report_path=args.v143_report,
            v143_dataset_path=args.v143_dataset,
            v144_report_path=args.v144_report,
            artifact_path=args.artifact,
        )
        report = build_branch_label_causal_audit_report(
            v143_report=v143_report,
            dataset_rows=dataset_rows,
            v144_report=v144_report,
            artifact=artifact,
            v143_report_path=args.v143_report,
            v143_dataset_path=args.v143_dataset,
            v144_report_path=args.v144_report,
            artifact_path=artifact_path,
        )
        write_branch_label_causal_audit_report(report, args.output)
    except (OSError, ValueError, BranchLabelCausalAuditError) as exc:
        raise SystemExit(f"failed to run v145 branch-label causal audit: {exc}") from exc
    _print_summary(report, args.output)


def _print_summary(report: dict[str, object], output: Path) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    route = report.get("route_decision")
    route_payload = route if isinstance(route, dict) else {}
    broad = report.get("broad_resolved_invalid_increase_explanation")
    broad_payload = broad if isinstance(broad, dict) else {}
    print(f"v145_branch_label_causal_audit={output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"route={route_payload.get('route')}")
    print(f"applied_label_count={report.get('applied_label_count')}")
    print(f"label_blacklist_count={len(route_payload.get('label_blacklist') or [])}")
    print(f"broad_resolved_invalid_delta={broad_payload.get('delta')}")


if __name__ == "__main__":
    main()
