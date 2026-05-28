from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.v3_progress_ledger import (
    MIND_V3_STANDARD_PROGRESS_LEDGER_SCHEMA_VERSION,
    MindV3ProgressLedgerError,
    build_standard_progress_ledger_report,
    write_standard_progress_ledger_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic standardized Mind v3 progress ledger with "
            "one structured row per v-slice."
        )
    )
    parser.add_argument(
        "--docs",
        type=Path,
        default=Path("docs/mind-v3-autonomous-evolution.md"),
        help="Mind v3 evolution notes used as documentary evidence.",
    )
    parser.add_argument(
        "--legacy-ledger",
        type=Path,
        default=Path("output/mind/mind-v3-experiment-ledger.jsonl"),
        help="Existing legacy experiment ledger JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/mind"),
        help="Directory containing Mind v3 JSON artifacts.",
    )
    parser.add_argument("--start-version", type=int, default=1)
    parser.add_argument("--through-version", type=int, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-standard-progress-ledger.jsonl"),
        help="Output JSONL path with one row per v-slice.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("output/mind/mind-v3-standard-progress-ledger-report.json"),
        help="Output JSON report path containing summary and rows.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = build_standard_progress_ledger_report(
            docs_path=args.docs,
            legacy_ledger_path=args.legacy_ledger,
            output_dir=args.output_dir,
            start_version=int(args.start_version),
            through_version=args.through_version,
        )
        write_standard_progress_ledger_report(
            report,
            jsonl_output_path=args.output,
            summary_output_path=args.summary_output,
        )
    except (OSError, ValueError, MindV3ProgressLedgerError) as exc:
        raise SystemExit(f"failed to build Mind v3 progress ledger: {exc}") from exc

    summary = report["summary"]  # type: ignore[index]
    print(f"mind_v3_progress_ledger={args.output}")
    print(f"schema_version={MIND_V3_STANDARD_PROGRESS_LEDGER_SCHEMA_VERSION}")
    print(f"row_count={summary['row_count']}")  # type: ignore[index]
    print(
        "progress_passed_count="
        f"{summary['progress_passed_count']}"  # type: ignore[index]
    )
    print(
        "progress_passed_versions="
        f"{summary['progress_passed_versions']}"  # type: ignore[index]
    )


if __name__ == "__main__":
    main()
