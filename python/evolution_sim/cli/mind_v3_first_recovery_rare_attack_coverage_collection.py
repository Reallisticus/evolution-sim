from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_archive_blocker_diagnostic import (
    DEFAULT_ARCHIVE_REPORT_PATH,
    DEFAULT_ARCHIVE_ROWS_PATH,
)
from evolution_sim.mind.first_recovery_rare_action_coverage_targeting import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V121_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_rare_attack_coverage_collection import (
    DEFAULT_CANDIDATES_OUTPUT_PATH,
    DEFAULT_MAX_BRANCHES,
    DEFAULT_MAX_CANDIDATES_PER_ACTION,
    DEFAULT_MAX_SEEDS,
    DEFAULT_OUTPUT_PATH,
    MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_SCHEMA_VERSION,
    build_first_recovery_rare_attack_coverage_collection,
    write_first_recovery_rare_attack_coverage_collection_outputs,
)
from evolution_sim.mind.first_recovery_repaired_label_contract_audit import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V119_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V119_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_repaired_label_split_support_feasibility import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V120_REPORT_PATH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 v122 first-recovery rare-attack "
            "coverage collection report."
        )
    )
    parser.add_argument(
        "--v119-report",
        type=Path,
        default=DEFAULT_V119_REPORT_PATH,
        help="Input v119 repaired-label contract audit report.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_V119_MANIFEST_PATH,
        help="Input v119 repaired-label JSONL manifest.",
    )
    parser.add_argument(
        "--v120-report",
        type=Path,
        default=DEFAULT_V120_REPORT_PATH,
        help="Input v120 split/support feasibility report.",
    )
    parser.add_argument(
        "--v121-report",
        type=Path,
        default=DEFAULT_V121_REPORT_PATH,
        help="Input v121 rare-action coverage targeting report.",
    )
    parser.add_argument(
        "--candidate-archive-report",
        type=Path,
        default=DEFAULT_ARCHIVE_REPORT_PATH,
        help="Input candidate first-recovery archive JSON report; defaults to v115.",
    )
    parser.add_argument(
        "--candidate-archive-rows",
        type=Path,
        default=DEFAULT_ARCHIVE_ROWS_PATH,
        help="Input candidate first-recovery archive JSONL rows; defaults to v115.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        default=[],
        help="Optional seed allowlist for candidate branch evaluation.",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=DEFAULT_MAX_SEEDS,
        help="Maximum distinct candidate seeds to evaluate when --seed is absent.",
    )
    parser.add_argument(
        "--max-branches",
        type=int,
        default=DEFAULT_MAX_BRANCHES,
        help="Maximum candidate branches to evaluate.",
    )
    parser.add_argument(
        "--max-candidates-per-action",
        type=int,
        default=DEFAULT_MAX_CANDIDATES_PER_ACTION,
        help="Maximum accepted candidates to keep per target rare attack action.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--candidates-output",
        type=Path,
        default=DEFAULT_CANDIDATES_OUTPUT_PATH,
        help="Output JSONL accepted-candidate manifest path.",
    )
    parser.add_argument(
        "--no-candidates-output",
        action="store_true",
        help="Do not write the optional accepted-candidate JSONL manifest.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        build = build_first_recovery_rare_attack_coverage_collection(
            v119_report_path=args.v119_report,
            manifest_path=args.manifest,
            v120_report_path=args.v120_report,
            v121_report_path=args.v121_report,
            candidate_archive_report_path=args.candidate_archive_report,
            candidate_archive_rows_path=args.candidate_archive_rows,
            max_seeds=args.max_seeds,
            seed_allowlist=tuple(args.seed),
            max_branches=args.max_branches,
            max_candidates_per_action=args.max_candidates_per_action,
        )
        write_first_recovery_rare_attack_coverage_collection_outputs(
            build,
            output_path=args.output,
            candidates_output_path=(
                None if args.no_candidates_output else args.candidates_output
            ),
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to build first-recovery rare-attack coverage collection "
            f"report: {exc}"
        ) from exc
    _print_summary(build.report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source_integrity"))
    budget = _mapping(report.get("search_budget_used"))
    recommendation = _mapping(report.get("recommendation"))
    print(f"first_recovery_rare_attack_coverage_collection={output_path}")
    print(
        "schema_version="
        f"{MIND_V3_FIRST_RECOVERY_RARE_ATTACK_COVERAGE_COLLECTION_SCHEMA_VERSION}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"found_candidate_counts={report.get('found_candidate_counts')}")
    print(f"branches_evaluated={budget.get('branches_evaluated')}")
    print(
        "would_clear_v120_rare_action_limitation_if_accepted="
        f"{recommendation.get('would_clear_v120_rare_action_limitation_if_accepted')}"
    )
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
