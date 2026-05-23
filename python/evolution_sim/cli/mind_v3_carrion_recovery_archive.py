from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

from evolution_sim.mind.carrion_branch_explore import (
    DEFAULT_CARRION_BRANCH_BASE_SCRIPT,
    DEFAULT_CARRION_BRANCH_POINTS_PER_SEED,
)
from evolution_sim.mind.carrion_counterfactual import (
    DEFAULT_CARRION_COUNTERFACTUAL_SEEDS,
    DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
    DEFAULT_COUNTERFACTUAL_SCRIPTS,
)
from evolution_sim.mind.carrion_recovery_archive import (
    DEFAULT_RECOVERY_ARCHIVE_MAX_DATASET_RECORDS_PER_CLASS,
    DEFAULT_RECOVERY_ARCHIVE_MIN_FAILURE_CELLS,
    DEFAULT_RECOVERY_ARCHIVE_MIN_SURVIVOR_CELLS,
    MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION,
    MIND_V3_CARRION_RECOVERY_ARCHIVE_SOURCE_BRANCH,
    MIND_V3_CARRION_RECOVERY_ARCHIVE_SOURCE_FIXTURE_RERANK_PROBE,
    CarrionRecoveryArchiveError,
    build_fixture_rerank_recovery_probe_archive_report,
    build_carrion_recovery_archive_report,
    load_carrion_recovery_json_report,
    write_carrion_recovery_archive_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a quality-diverse Mind v3 carrion recovery archive from "
            "post-contact branch continuations."
        )
    )
    parser.add_argument(
        "--source",
        choices=[
            MIND_V3_CARRION_RECOVERY_ARCHIVE_SOURCE_BRANCH,
            MIND_V3_CARRION_RECOVERY_ARCHIVE_SOURCE_FIXTURE_RERANK_PROBE,
        ],
        default=MIND_V3_CARRION_RECOVERY_ARCHIVE_SOURCE_BRANCH,
        help=(
            "Archive source. The default keeps the existing branch-continuation "
            "archive path; fixture-rerank-recovery-probe builds a report-only "
            "audit from completed fixture_rerank candidate carrion probes."
        ),
    )
    parser.add_argument(
        "--search-report",
        action="append",
        default=[],
        help=(
            "Search report for --source fixture-rerank-recovery-probe. Use "
            "LABEL=PATH or PATH. Repeat to combine reports."
        ),
    )
    parser.add_argument(
        "--branch-report",
        type=Path,
        default=None,
        help=(
            "Optional existing carrion branch-explore report. When omitted, "
            "the branch report is generated first."
        ),
    )
    parser.add_argument(
        "--counterfactual-report",
        type=Path,
        default=None,
        help=(
            "Optional carrion counterfactual rollout report whose full "
            "fixture trajectories are appended to the recovery dataset."
        ),
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_CARRION_COUNTERFACTUAL_SEEDS),
        help="Comma-separated carrion fixture seeds used when generating branches.",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
        help="Fixture horizon used when generating branches.",
    )
    parser.add_argument(
        "--base-script",
        choices=DEFAULT_COUNTERFACTUAL_SCRIPTS,
        default=DEFAULT_CARRION_BRANCH_BASE_SCRIPT,
        help="Base script used to replay to branch points when generating branches.",
    )
    parser.add_argument(
        "--continuation-script",
        action="append",
        choices=DEFAULT_COUNTERFACTUAL_SCRIPTS,
        help=(
            "Continuation script to fan out from each generated branch. "
            "Repeat to run multiple scripts. Defaults to all scripts."
        ),
    )
    parser.add_argument(
        "--max-branch-points-per-seed",
        type=int,
        default=DEFAULT_CARRION_BRANCH_POINTS_PER_SEED,
        help="Maximum post-contact branch points retained for each generated seed.",
    )
    parser.add_argument(
        "--min-branch-tick",
        type=int,
        default=0,
        help="Ignore animal-resource contacts before this tick when generating.",
    )
    parser.add_argument(
        "--no-verify-replay",
        action="store_true",
        help="Skip deterministic branch replay verification when generating.",
    )
    parser.add_argument(
        "--trajectory-output-dir",
        type=Path,
        default=None,
        help="Optional per-branch trajectory output directory when generating.",
    )
    parser.add_argument(
        "--max-dataset-records-per-class",
        type=int,
        default=DEFAULT_RECOVERY_ARCHIVE_MAX_DATASET_RECORDS_PER_CLASS,
        help="Maximum survivor and failure records exported from archive elites.",
    )
    parser.add_argument(
        "--min-survivor-cells",
        type=int,
        default=DEFAULT_RECOVERY_ARCHIVE_MIN_SURVIVOR_CELLS,
        help="Minimum survivor descriptor cells required for archive acceptance.",
    )
    parser.add_argument(
        "--min-failure-cells",
        type=int,
        default=DEFAULT_RECOVERY_ARCHIVE_MIN_FAILURE_CELLS,
        help="Minimum failure descriptor cells required for archive acceptance.",
    )
    parser.add_argument(
        "--min-counterfactual-survivor-seeds",
        type=int,
        default=0,
        help=(
            "Minimum distinct seeds with terminal survivors in the optional "
            "--counterfactual-report."
        ),
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=Path("output/mind/mind-v3-carrion-recovery-dataset.jsonl"),
        help="Output JSONL path for balanced survivor/failure archive records.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-carrion-recovery-archive.json"),
        help="Output JSON archive report path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.source == MIND_V3_CARRION_RECOVERY_ARCHIVE_SOURCE_FIXTURE_RERANK_PROBE:
        _run_fixture_rerank_probe_archive(args)
        return
    continuation_scripts = (
        tuple(args.continuation_script)
        if args.continuation_script
        else DEFAULT_COUNTERFACTUAL_SCRIPTS
    )
    try:
        report = build_carrion_recovery_archive_report(
            branch_report_path=args.branch_report,
            counterfactual_report_path=args.counterfactual_report,
            seeds=_parse_seeds(args.seeds),
            ticks=int(args.ticks),
            base_script=str(args.base_script),
            continuation_scripts=continuation_scripts,
            max_branch_points_per_seed=int(args.max_branch_points_per_seed),
            min_branch_tick=int(args.min_branch_tick),
            trajectory_output_dir=args.trajectory_output_dir,
            verify_replay=not bool(args.no_verify_replay),
            dataset_output_path=args.dataset_output,
            max_dataset_records_per_class=int(args.max_dataset_records_per_class),
            min_survivor_cells=int(args.min_survivor_cells),
            min_failure_cells=int(args.min_failure_cells),
            min_counterfactual_survivor_seeds=int(
                args.min_counterfactual_survivor_seeds
            ),
        )
        write_carrion_recovery_archive_report(report, args.output)
    except (OSError, ValueError, CarrionRecoveryArchiveError) as exc:
        raise SystemExit(f"failed to build carrion recovery archive: {exc}") from exc

    aggregate = report["aggregate"]  # type: ignore[index]
    acceptance = report["acceptance"]  # type: ignore[index]
    print(f"carrion_recovery_archive={args.output}")
    print(f"schema_version={MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION}")
    print(f"cell_count={aggregate['cell_count']}")  # type: ignore[index]
    print(f"survivor_cell_count={aggregate['survivor_cell_count']}")  # type: ignore[index]
    print(f"failure_cell_count={aggregate['failure_cell_count']}")  # type: ignore[index]
    print(f"dataset_record_count={aggregate['dataset_record_count']}")  # type: ignore[index]
    print(
        "counterfactual_survivor_seed_count="
        f"{aggregate.get('counterfactual_survivor_seed_count', 0)}"
    )
    print(
        "archive_acceptance_passed="
        f"{acceptance['archive_acceptance_passed']}"  # type: ignore[index]
    )
    aggregate_payload = aggregate if isinstance(aggregate, Mapping) else {}
    _print_outcome_metrics(
        "archive",
        aggregate_payload.get("outcome_metrics", {}),
    )


def _run_fixture_rerank_probe_archive(args: argparse.Namespace) -> None:
    if not args.search_report:
        raise SystemExit(
            "failed to build carrion recovery archive: "
            "--source fixture-rerank-recovery-probe requires --search-report"
        )
    try:
        search_reports = []
        for raw in args.search_report:
            label, path = _parse_labeled_search_report(str(raw))
            search_reports.append(
                (
                    label,
                    load_carrion_recovery_json_report(path),
                    path,
                )
            )
        report = build_fixture_rerank_recovery_probe_archive_report(
            search_reports=search_reports,
        )
        write_carrion_recovery_archive_report(report, args.output)
    except (OSError, ValueError, CarrionRecoveryArchiveError) as exc:
        raise SystemExit(f"failed to build carrion recovery archive: {exc}") from exc

    input_summary = report["input_summary"]  # type: ignore[index]
    archive = report["archive"]  # type: ignore[index]
    selected = report["selected"]  # type: ignore[index]
    retention = report["retention"]  # type: ignore[index]
    print(f"carrion_recovery_archive={args.output}")
    print(f"schema_version={MIND_V3_CARRION_RECOVERY_ARCHIVE_SCHEMA_VERSION}")
    print(f"source={MIND_V3_CARRION_RECOVERY_ARCHIVE_SOURCE_FIXTURE_RERANK_PROBE}")
    print(f"candidate_count={input_summary['candidate_count']}")  # type: ignore[index]
    print(f"complete_probe_count={input_summary['complete_probe_count']}")  # type: ignore[index]
    print(f"missing_probe_count={input_summary['missing_probe_count']}")  # type: ignore[index]
    print(f"archive_cell_count={archive['cell_count']}")  # type: ignore[index]
    print(
        "selected_dominates_all_non_selected_recovery_cells="
        f"{selected['dominates_all_non_selected_recovery_cells']}"  # type: ignore[index]
    )
    print(
        "archive_retention_would_add_new_parent_candidates="
        f"{retention['would_add_new_parent_candidates']}"  # type: ignore[index]
    )


def _parse_labeled_search_report(raw: str) -> tuple[str, Path]:
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise CarrionRecoveryArchiveError("--search-report label is empty")
        return label, Path(path)
    path = Path(raw)
    return path.stem, path


def _print_outcome_metrics(prefix: str, metrics: object) -> None:
    payload = metrics if isinstance(metrics, Mapping) else {}
    keys = (
        "terminal_survivor_run_count",
        "extinct_run_count",
        "total_terminal_alive_agents",
        "total_births",
        "runs_with_births",
        "total_deaths",
        "total_scavenger_terminal_agents",
        "total_scavenger_parent_births",
        "total_scavenger_child_births",
        "total_animal_resource_consumption_events",
        "total_carcass_consumption_events",
        "total_fresh_kill_consumption_events",
        "total_scavenger_animal_resource_events",
        "total_scavenger_carcass_events",
        "total_scavenger_fresh_kill_events",
    )
    for key in keys:
        print(f"{prefix}_{key}={payload.get(key, 0)}")


def _parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise CarrionRecoveryArchiveError("--seeds must include at least one seed")
    return values


if __name__ == "__main__":
    main()
