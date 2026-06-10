from __future__ import annotations

import argparse
import json
from pathlib import Path

from evolution_sim.mind.candidate_campaign import (
    DEFAULT_MIN_SAFE_LABEL_COUNT,
    DEFAULT_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_PATH,
    DEFAULT_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_CHUNK_DIR,
    DEFAULT_SAFE_ARCHIVE_EXPANSION_DATASET_PATH,
    DEFAULT_SAFE_ARCHIVE_EXPANSION_MAX_BRANCH_POINTS_PER_SEED,
    DEFAULT_SAFE_ARCHIVE_EXPANSION_MAX_BROAD_REGRESSION_SEEDS,
    DEFAULT_SAFE_ARCHIVE_EXPANSION_MAX_CANDIDATE_ACTIONS,
    DEFAULT_SAFE_ARCHIVE_EXPANSION_MAX_CARRION_FIXTURE_SEEDS,
    DEFAULT_SAFE_ARCHIVE_EXPANSION_REPORT_PATH,
    DEFAULT_V145_REPORT_PATH,
    CandidateCampaignError,
    build_safe_archive_expansion_branch_evidence,
    build_safe_archive_expansion_report,
    merge_safe_archive_expansion_branch_evidence,
    write_safe_archive_expansion_branch_evidence,
    write_safe_archive_expansion_dataset,
    write_safe_archive_expansion_report,
)
from evolution_sim.mind.broad_regression_branch_intervention import (
    DEFAULT_LIVE_REPORT_PATH as DEFAULT_V142_LIVE_REPORT_PATH,
    DEFAULT_SCORER_PATH as DEFAULT_V142_SCORER_REPORT_PATH,
    DEFAULT_V142_TRAJECTORY_DIR,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 carrion/broad safe archive "
            "expansion report from expanded branch evidence. This command "
            "writes leakage-vetted rows but does not train."
        )
    )
    parser.add_argument(
        "--branch-evidence-report",
        type=Path,
        default=None,
        help=(
            "Existing JSON report containing expanded branch_results evidence. "
            "Requires --use-existing-branch-evidence."
        ),
    )
    parser.add_argument(
        "--use-existing-branch-evidence",
        action="store_true",
        help="Read --branch-evidence-report instead of generating fresh evidence.",
    )
    parser.add_argument(
        "--branch-evidence-output",
        type=Path,
        default=DEFAULT_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_PATH,
        help="Where to write generated diagnostics-only branch evidence.",
    )
    parser.add_argument(
        "--branch-evidence-chunk-dir",
        type=Path,
        default=DEFAULT_SAFE_ARCHIVE_EXPANSION_BRANCH_EVIDENCE_CHUNK_DIR,
        help="Directory for deterministic per-branch-result checkpoint chunks.",
    )
    parser.add_argument(
        "--resume-branch-evidence",
        action="store_true",
        help="Reuse branch-result checkpoint chunks from --branch-evidence-chunk-dir.",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help=(
            "Stop branch evidence generation between branch points after this "
            "many wall-clock seconds and write a partial no-training report."
        ),
    )
    parser.add_argument(
        "--fixture",
        choices=("both", "broad", "carrion_only"),
        default="both",
        help="Deterministic shard fixture selection.",
    )
    parser.add_argument(
        "--seed-include",
        type=str,
        default=None,
        help="Comma-separated explicit seed include list for shard generation.",
    )
    parser.add_argument(
        "--branch-index-start",
        type=int,
        default=None,
        help="First per-seed branch index to include in this shard.",
    )
    parser.add_argument(
        "--branch-index-count",
        type=int,
        default=None,
        help="Number of branch indexes to include starting at --branch-index-start.",
    )
    parser.add_argument(
        "--branch-index-include",
        type=str,
        default=None,
        help="Comma-separated explicit per-seed branch indexes to include.",
    )
    parser.add_argument(
        "--shard-id",
        type=str,
        default=None,
        help="Stable operator-provided shard id recorded in branch evidence inputs.",
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge shard branch evidence/chunk dirs before building the report.",
    )
    parser.add_argument(
        "--merge-shard-evidence",
        type=Path,
        action="append",
        default=[],
        help="Shard branch evidence JSON to merge. May be provided multiple times.",
    )
    parser.add_argument(
        "--merge-shard-chunk-dir",
        type=Path,
        action="append",
        default=[],
        help="Shard branch-result chunk directory to merge. May be provided multiple times.",
    )
    parser.add_argument(
        "--allow-partial-shard-evidence",
        action="store_true",
        help="Allow partial shard inputs and produce a partial no-training report.",
    )
    parser.add_argument(
        "--v142-scorer",
        type=Path,
        default=DEFAULT_V142_SCORER_REPORT_PATH,
    )
    parser.add_argument(
        "--v142-live-report",
        type=Path,
        default=DEFAULT_V142_LIVE_REPORT_PATH,
    )
    parser.add_argument(
        "--v142-trajectory-output-dir",
        type=Path,
        default=DEFAULT_V142_TRAJECTORY_DIR,
    )
    parser.add_argument(
        "--max-branch-points-per-seed",
        type=int,
        default=DEFAULT_SAFE_ARCHIVE_EXPANSION_MAX_BRANCH_POINTS_PER_SEED,
    )
    parser.add_argument(
        "--max-candidate-actions",
        type=int,
        default=DEFAULT_SAFE_ARCHIVE_EXPANSION_MAX_CANDIDATE_ACTIONS,
        help="0 means evaluate every valid public action-mask action.",
    )
    parser.add_argument(
        "--max-broad-regression-seeds",
        type=int,
        default=DEFAULT_SAFE_ARCHIVE_EXPANSION_MAX_BROAD_REGRESSION_SEEDS,
        help="0 means use every v142 broad regression seed.",
    )
    parser.add_argument(
        "--max-carrion-fixture-seeds",
        type=int,
        default=DEFAULT_SAFE_ARCHIVE_EXPANSION_MAX_CARRION_FIXTURE_SEEDS,
        help="0 means use every strict carrion fixture seed.",
    )
    parser.add_argument(
        "--no-regenerate-v142-trajectories",
        action="store_true",
        help="Do not regenerate missing v142 broad baseline/override trajectories.",
    )
    parser.add_argument(
        "--no-verify-replay",
        action="store_true",
        help="Skip second deterministic branch replay verification pass.",
    )
    parser.add_argument(
        "--v145-report",
        type=Path,
        default=DEFAULT_V145_REPORT_PATH,
        help="v145 causal blacklist report.",
    )
    parser.add_argument(
        "--min-safe-label-count",
        type=int,
        default=DEFAULT_MIN_SAFE_LABEL_COUNT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SAFE_ARCHIVE_EXPANSION_REPORT_PATH,
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=DEFAULT_SAFE_ARCHIVE_EXPANSION_DATASET_PATH,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.merge_shards:
            branch_evidence = merge_safe_archive_expansion_branch_evidence(
                shard_evidence_reports=[
                    _load_json(path) for path in args.merge_shard_evidence
                ],
                shard_chunk_dirs=args.merge_shard_chunk_dir,
                allow_partial_shard_evidence=bool(args.allow_partial_shard_evidence),
            )
            write_safe_archive_expansion_branch_evidence(
                branch_evidence,
                args.branch_evidence_output,
            )
            branch_evidence_output = args.branch_evidence_output
        elif args.use_existing_branch_evidence:
            if args.branch_evidence_report is None:
                raise CandidateCampaignError(
                    "--use-existing-branch-evidence requires --branch-evidence-report"
                )
            branch_evidence = _load_json(args.branch_evidence_report)
            branch_evidence_output = args.branch_evidence_report
        else:
            branch_evidence = build_safe_archive_expansion_branch_evidence(
                v142_scorer_report_path=args.v142_scorer,
                v142_live_report_path=args.v142_live_report,
                v142_trajectory_output_dir=args.v142_trajectory_output_dir,
                max_branch_points_per_seed=int(args.max_branch_points_per_seed),
                max_candidate_actions=int(args.max_candidate_actions),
                max_broad_regression_seeds=int(args.max_broad_regression_seeds),
                max_carrion_fixture_seeds=int(args.max_carrion_fixture_seeds),
                regenerate_v142_trajectories=not args.no_regenerate_v142_trajectories,
                verify_replay=not args.no_verify_replay,
                branch_evidence_chunk_dir=args.branch_evidence_chunk_dir,
                resume_branch_evidence=bool(args.resume_branch_evidence),
                max_wall_seconds=args.max_wall_seconds,
                progress_callback=_print_progress,
                fixtures=_fixture_selection(args.fixture),
                seed_include=_parse_int_list(args.seed_include),
                branch_index_start=args.branch_index_start,
                branch_index_count=args.branch_index_count,
                branch_index_include=_parse_int_list(args.branch_index_include),
                shard_id=args.shard_id,
            )
            write_safe_archive_expansion_branch_evidence(
                branch_evidence,
                args.branch_evidence_output,
            )
            branch_evidence_output = args.branch_evidence_output
        v145_report = _load_json(args.v145_report)
        report, rows = build_safe_archive_expansion_report(
            branch_results=_branch_results(branch_evidence),
            v145_report=v145_report,
            min_safe_label_count=int(args.min_safe_label_count),
            branch_evidence_status=_branch_evidence_status(branch_evidence),
        )
        write_safe_archive_expansion_report(report, args.output)
        write_safe_archive_expansion_dataset(rows, args.dataset_output)
    except (OSError, ValueError, CandidateCampaignError) as exc:
        raise SystemExit(f"failed to build safe archive expansion: {exc}") from exc
    _print_summary(report, rows, args.output, args.dataset_output, branch_evidence_output)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CandidateCampaignError(f"JSON payload must be an object: {path}")
    return payload


def _branch_results(payload: dict[str, object]) -> list[dict[str, object]]:
    value = payload.get("branch_results")
    if not isinstance(value, list):
        raise CandidateCampaignError("branch evidence report must contain branch_results")
    return [dict(item) for item in value if isinstance(item, dict)]


def _branch_evidence_status(payload: dict[str, object]) -> dict[str, object]:
    value = payload.get("generation_status")
    if isinstance(value, dict):
        return dict(value)
    return {}


def _fixture_selection(value: str) -> tuple[str, ...]:
    if value == "both":
        return ("broad", "carrion_only")
    return (value,)


def _parse_int_list(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    items = tuple(
        int(item.strip())
        for item in value.split(",")
        if item.strip()
    )
    return items


def _print_progress(payload: object) -> None:
    if not isinstance(payload, dict):
        return
    print(
        "safe_archive_expansion_progress="
        f"event={payload.get('event')} "
        f"source={payload.get('source')} "
        f"fixture={payload.get('fixture')} "
        f"seed={payload.get('seed')} "
        f"branch_point={payload.get('branch_point_index')} "
        f"branch_id={payload.get('branch_id')} "
        f"action_count={payload.get('action_count')} "
        f"elapsed_seconds={payload.get('elapsed_seconds')} "
        f"stop_reason={payload.get('stop_reason')}"
    )


def _print_summary(
    report: dict[str, object],
    rows: list[dict[str, object]],
    output: Path,
    dataset_output: Path,
    branch_evidence_output: Path,
) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    dataset = report.get("dataset")
    dataset_payload = dataset if isinstance(dataset, dict) else {}
    source_integrity = report.get("source_integrity")
    source_payload = source_integrity if isinstance(source_integrity, dict) else {}
    status = source_payload.get("branch_evidence_status")
    status_payload = status if isinstance(status, dict) else {}
    fixture_statuses = status_payload.get("fixture_statuses")
    generated_count = None
    resumed_count = None
    if isinstance(fixture_statuses, list):
        generated_count = sum(
            int(item.get("generated_branch_result_count", 0))
            for item in fixture_statuses
            if isinstance(item, dict)
        )
        resumed_count = sum(
            int(item.get("resumed_branch_result_count", 0))
            for item in fixture_statuses
            if isinstance(item, dict)
        )
    print(f"safe_archive_expansion_report={output}")
    print(f"safe_archive_expansion_dataset={dataset_output}")
    print(f"safe_archive_expansion_branch_evidence={branch_evidence_output}")
    print(f"branch_evidence_status={status_payload.get('state')}")
    print(f"branch_evidence_partial={status_payload.get('partial')}")
    print(f"branch_evidence_generated_count={generated_count}")
    print(f"branch_evidence_resumed_count={resumed_count}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"safe_label_count={dataset_payload.get('safe_label_count')}")
    print(f"dataset_row_count={len(rows)}")
    print(f"source_integrity_passed={source_payload.get('passed')}")
    print(f"source_integrity_failures={source_payload.get('failures')}")
    print(f"training_authorized={report.get('training_authorized')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")


if __name__ == "__main__":
    main()
