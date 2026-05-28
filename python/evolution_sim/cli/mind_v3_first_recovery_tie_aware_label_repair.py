from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_tie_aware_label_repair import (
    DEFAULT_ARCHIVE_REPORT_PATH,
    DEFAULT_ARCHIVE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH,
    MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_SCHEMA_VERSION,
    build_first_recovery_tie_aware_label_repair,
    write_first_recovery_tie_aware_label_repair_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 v118 first-recovery tie-aware "
            "label-repair proposal over the existing v115 archive."
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON report path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        build = build_first_recovery_tie_aware_label_repair(
            archive_report_path=args.archive_report,
            archive_rows_path=args.archive_rows,
        )
        write_first_recovery_tie_aware_label_repair_report(
            build,
            output_path=args.output,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to build first-recovery tie-aware label repair: " f"{exc}"
        ) from exc
    _print_summary(build.report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    classification = _mapping(report.get("classification"))
    repair = _mapping(report.get("tie_aware_label_repair"))
    policies = _mapping(repair.get("policies"))
    best_name = repair.get("best_clearing_policy") or repair.get("best_valid_policy")
    best = _mapping(policies.get(str(best_name)))
    recommendation = _mapping(report.get("recommendation"))
    print(f"first_recovery_tie_aware_label_repair={output_path}")
    print(
        "schema_version="
        f"{MIND_V3_FIRST_RECOVERY_TIE_AWARE_LABEL_REPAIR_SCHEMA_VERSION}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"best_valid_policy={repair.get('best_valid_policy')}")
    print(f"best_clearing_policy={repair.get('best_clearing_policy')}")
    print(f"best_policy_dominant_action={best.get('dominant_action')}")
    print(f"best_policy_dominant_action_share={best.get('dominant_action_share')}")
    print(f"best_policy_changed_branch_count={best.get('changed_branch_count')}")
    print(f"recommendation={recommendation.get('next_step')}")
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
