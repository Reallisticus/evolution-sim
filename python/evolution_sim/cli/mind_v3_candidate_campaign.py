from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.candidate_campaign import (
    DEFAULT_LEDGER_PATH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_REPORT_PATH,
    CandidateCampaignError,
    run_candidate_campaign,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v146 Mind v3 candidate campaign across the strict "
            "broad+carrion matrix and rank candidate families."
        )
    )
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--keep-trajectories", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--ledger-output", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_candidate_campaign(
            mode=args.mode,
            workers=max(1, int(args.workers)),
            candidate_limit=args.candidate_limit,
            config_path=args.config,
            keep_trajectories=bool(args.keep_trajectories),
            output_dir=args.output_dir,
            report_path=args.output,
            ledger_path=args.ledger_output,
        )
    except (OSError, ValueError, CandidateCampaignError) as exc:
        raise SystemExit(f"failed to run v146 candidate campaign: {exc}") from exc
    _print_summary(report, args.output, args.ledger_output)


def _print_summary(
    report: dict[str, object],
    output: Path,
    ledger_output: Path,
) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    stop_rules = report.get("stop_rules")
    stop_payload = stop_rules if isinstance(stop_rules, dict) else {}
    print(f"v146_candidate_campaign_report={output}")
    print(f"v146_candidate_campaign_ledger={ledger_output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"candidate_count={report.get('candidate_count')}")
    print(f"safe_label_count={stop_payload.get('safe_label_count')}")
    print(
        "archive_support_insufficient="
        f"{stop_payload.get('archive_support_insufficient')}"
    )
    print(
        "next_route_carrion_specific_archive_expansion="
        f"{stop_payload.get('next_route_carrion_specific_archive_expansion')}"
    )


if __name__ == "__main__":
    main()
