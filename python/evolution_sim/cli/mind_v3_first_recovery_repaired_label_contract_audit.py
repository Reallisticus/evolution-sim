from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_repaired_label_contract_audit import (
    DEFAULT_ARCHIVE_REPORT_PATH,
    DEFAULT_ARCHIVE_ROWS_PATH,
    DEFAULT_MANIFEST_OUTPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V118_REPORT_PATH,
    MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION,
    build_first_recovery_repaired_label_contract_audit,
    write_first_recovery_repaired_label_contract_audit_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 v119 first-recovery repaired "
            "label contract/export audit over v115 and v118."
        )
    )
    parser.add_argument(
        "--archive-report",
        type=Path,
        default=DEFAULT_ARCHIVE_REPORT_PATH,
        help="Input v115 expanded first-recovery branch archive JSON report.",
    )
    parser.add_argument(
        "--archive-rows",
        type=Path,
        default=DEFAULT_ARCHIVE_ROWS_PATH,
        help="Input v115 expanded first-recovery branch archive gzip JSONL rows.",
    )
    parser.add_argument(
        "--v118-report",
        type=Path,
        default=DEFAULT_V118_REPORT_PATH,
        help="Input v118 tie-aware label repair JSON report.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_MANIFEST_OUTPUT_PATH,
        help="Output JSONL repaired label manifest path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        build = build_first_recovery_repaired_label_contract_audit(
            archive_report_path=args.archive_report,
            archive_rows_path=args.archive_rows,
            v118_report_path=args.v118_report,
        )
        write_first_recovery_repaired_label_contract_audit_outputs(
            build,
            output_path=args.output,
            manifest_output_path=args.manifest_output,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to build first-recovery repaired label contract audit: "
            f"{exc}"
        ) from exc
    _print_summary(build.report, args.output, args.manifest_output)


def _print_summary(
    report: Mapping[str, object],
    output_path: Path,
    manifest_output_path: Path,
) -> None:
    classification = _mapping(report.get("classification"))
    contract = _mapping(report.get("contract_checks"))
    split = _mapping(report.get("split_support"))
    recommendation = _mapping(report.get("recommendation"))
    print(f"first_recovery_repaired_label_contract_audit={output_path}")
    print(f"first_recovery_repaired_label_manifest={manifest_output_path}")
    print(
        "schema_version="
        f"{MIND_V3_FIRST_RECOVERY_REPAIRED_LABEL_CONTRACT_AUDIT_SCHEMA_VERSION}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"manifest_row_count={contract.get('manifest_row_count')}")
    print(f"repaired_action_counts={contract.get('repaired_action_counts')}")
    print(f"contract_total_violation_count={contract.get('total_violation_count')}")
    print(f"split_support_adequate={split.get('support_adequate')}")
    print(f"split_warnings={split.get('warnings')}")
    print(
        "v113_readiness_rerun_allowed="
        f"{recommendation.get('v113_readiness_rerun_allowed')}"
    )
    print(
        "downstream_shadow_scorer_allowed="
        f"{recommendation.get('downstream_shadow_scorer_allowed')}"
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
