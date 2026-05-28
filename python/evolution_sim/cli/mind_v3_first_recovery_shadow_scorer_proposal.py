from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V124_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V124_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_shadow_scorer_proposal import (
    DEFAULT_OUTPUT_PATH,
    build_first_recovery_shadow_scorer_proposal,
    write_first_recovery_shadow_scorer_proposal_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 v125 first-recovery "
            "shadow-scorer proposal."
        )
    )
    parser.add_argument("--v124-report", type=Path, default=DEFAULT_V124_REPORT_PATH)
    parser.add_argument("--v124-manifest", type=Path, default=DEFAULT_V124_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        build = build_first_recovery_shadow_scorer_proposal(
            v124_report_path=args.v124_report,
            v124_manifest_path=args.v124_manifest,
        )
        write_first_recovery_shadow_scorer_proposal_report(
            build,
            output_path=args.output,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"failed to build first-recovery shadow-scorer proposal: {exc}"
        ) from exc
    _print_summary(build.report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source_integrity"))
    recommendation = _mapping(report.get("recommendation"))
    dominant = _mapping(source.get("dominant_action"))
    print(f"first_recovery_shadow_scorer_proposal={output_path}")
    print(f"classification={classification.get('primary')}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"manifest_row_count={source.get('manifest_row_count')}")
    print(f"unique_branch_count={source.get('unique_branch_count')}")
    print(f"dominant_action={dominant.get('action')}")
    print(f"dominant_action_share={dominant.get('share')}")
    print(
        "shadow_scorer_execution_allowed="
        f"{recommendation.get('shadow_scorer_execution_allowed')}"
    )
    print(
        "v113_readiness_rerun_allowed="
        f"{recommendation.get('v113_readiness_rerun_allowed')}"
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
