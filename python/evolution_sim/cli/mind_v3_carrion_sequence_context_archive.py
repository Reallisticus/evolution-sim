from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.mind.carrion_sequence_context_archive import (
    DEFAULT_BRANCH_RESULT_CHUNK_DIR,
    DEFAULT_DATASET_OUTPUT_PATH,
    DEFAULT_HISTORY_WINDOW,
    DEFAULT_MIN_PRIOR_PUBLIC_STEPS,
    DEFAULT_MIN_SAFE_COMPARATOR_COUNT,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V150_CARRION_OVERRIDE_AUTOPSY_PATH,
    DEFAULT_V148_CARRION_ARCHIVE_DATASET_PATH,
    DEFAULT_V148_CARRION_ARCHIVE_REPORT_PATH,
    DEFAULT_V149_CARRION_TRAIN_EVAL_REPORT_PATH,
    CarrionSequenceContextArchiveError,
    build_carrion_sequence_context_archive_report_from_paths,
    load_json_report,
    load_jsonl_rows,
    merge_carrion_sequence_context_archive_shards,
    write_carrion_sequence_context_archive_dataset,
    write_carrion_sequence_context_archive_report,
)
from evolution_sim.mind.carrion_specific_archive_expansion import (
    CARRION_BRANCH_REASONS,
    DEFAULT_TICKS,
    TARGET_CARRION_SEEDS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only v151 carrion hydration/reproduction "
            "sequence-context archive. This command never trains, promotes, "
            "or changes runtime defaults."
        )
    )
    parser.add_argument(
        "--v148-carrion-archive-report",
        type=Path,
        default=DEFAULT_V148_CARRION_ARCHIVE_REPORT_PATH,
    )
    parser.add_argument(
        "--v148-carrion-archive-dataset",
        type=Path,
        default=DEFAULT_V148_CARRION_ARCHIVE_DATASET_PATH,
    )
    parser.add_argument(
        "--v149-carrion-train-eval-report",
        type=Path,
        default=DEFAULT_V149_CARRION_TRAIN_EVAL_REPORT_PATH,
    )
    parser.add_argument(
        "--v150-carrion-override-autopsy",
        type=Path,
        default=DEFAULT_V150_CARRION_OVERRIDE_AUTOPSY_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=DEFAULT_DATASET_OUTPUT_PATH,
    )
    parser.add_argument(
        "--seed-include",
        type=str,
        default=None,
        help="Comma-separated carrion_only seed include list for shard generation.",
    )
    parser.add_argument(
        "--branch-index-include",
        action="append",
        default=None,
        help=(
            "Carrion branch indexes to include in this shard. May be repeated "
            "or comma-separated."
        ),
    )
    parser.add_argument(
        "--branch-index-start",
        type=int,
        default=None,
        help="First carrion branch index to include in this shard.",
    )
    parser.add_argument(
        "--branch-index-count",
        type=int,
        default=None,
        help="Number of carrion branch indexes to include from --branch-index-start.",
    )
    parser.add_argument("--shard-id", type=str, default=None)
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument("--history-window", type=int, default=DEFAULT_HISTORY_WINDOW)
    parser.add_argument(
        "--min-prior-public-steps",
        type=int,
        default=DEFAULT_MIN_PRIOR_PUBLIC_STEPS,
    )
    parser.add_argument(
        "--min-safe-comparator-count",
        type=int,
        default=DEFAULT_MIN_SAFE_COMPARATOR_COUNT,
    )
    parser.add_argument(
        "--branch-result-chunk-dir",
        type=Path,
        default=DEFAULT_BRANCH_RESULT_CHUNK_DIR,
    )
    parser.add_argument(
        "--resume-branch-results",
        action="store_true",
        help="Reuse matching v151 deterministic branch-result chunks.",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help=(
            "Stop between branch/context units after this many seconds and "
            "write a partial no-training report."
        ),
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge previously generated v151 shard reports.",
    )
    parser.add_argument(
        "--merge-shard-report",
        type=Path,
        action="append",
        default=None,
        help="Shard report JSON to merge. Repeat for every shard.",
    )
    parser.add_argument(
        "--merge-shard-chunk-dir",
        type=Path,
        action="append",
        default=None,
        help="Optional v151 branch-result chunk directory to verify during merge.",
    )
    parser.add_argument(
        "--allow-partial-shard-evidence",
        action="store_true",
        help="Allow partial shard reports during merge. Defaults to fail closed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        branch_index_include = _selected_branch_indexes(args)
        if args.merge_shards:
            shard_paths = list(args.merge_shard_report or [])
            if not shard_paths:
                raise CarrionSequenceContextArchiveError(
                    "--merge-shards requires at least one --merge-shard-report"
                )
            report, rows = merge_carrion_sequence_context_archive_shards(
                archive_report=load_json_report(args.v148_carrion_archive_report),
                dataset_rows=load_jsonl_rows(args.v148_carrion_archive_dataset),
                train_eval_report=load_json_report(
                    args.v149_carrion_train_eval_report
                ),
                autopsy_report=load_json_report(args.v150_carrion_override_autopsy),
                shard_reports=[load_json_report(path) for path in shard_paths],
                shard_report_paths=shard_paths,
                shard_chunk_dirs=list(args.merge_shard_chunk_dir or []),
                allow_partial_shard_evidence=bool(
                    args.allow_partial_shard_evidence
                ),
                input_paths={
                    "v148_carrion_archive_report": args.v148_carrion_archive_report,
                    "v148_carrion_archive_dataset": args.v148_carrion_archive_dataset,
                    "v149_carrion_train_eval_report": (
                        args.v149_carrion_train_eval_report
                    ),
                    "v150_carrion_override_autopsy": args.v150_carrion_override_autopsy,
                },
            )
        else:
            report, rows = build_carrion_sequence_context_archive_report_from_paths(
                archive_report_path=args.v148_carrion_archive_report,
                archive_dataset_path=args.v148_carrion_archive_dataset,
                train_eval_report_path=args.v149_carrion_train_eval_report,
                autopsy_report_path=args.v150_carrion_override_autopsy,
                seed_include=_parse_int_list(args.seed_include),
                branch_index_include=branch_index_include,
                ticks=int(args.ticks),
                history_window=int(args.history_window),
                min_prior_public_steps=int(args.min_prior_public_steps),
                min_safe_comparator_count=int(args.min_safe_comparator_count),
                branch_result_chunk_dir=args.branch_result_chunk_dir,
                resume_branch_results=bool(args.resume_branch_results),
                max_wall_seconds=args.max_wall_seconds,
                shard_id=args.shard_id,
                progress_callback=_print_progress,
            )
        write_carrion_sequence_context_archive_report(report, args.output)
        write_carrion_sequence_context_archive_dataset(rows, args.dataset_output)
    except (OSError, ValueError, CarrionSequenceContextArchiveError) as exc:
        raise SystemExit(
            f"failed to build v151 carrion sequence-context archive: {exc}"
        ) from exc
    _print_summary(report, rows, args.output, args.dataset_output)


def _selected_branch_indexes(args: argparse.Namespace) -> tuple[int, ...] | None:
    include = _parse_repeated_int_list(args.branch_index_include)
    if include is not None:
        return include
    if args.branch_index_start is None and args.branch_index_count is None:
        return None
    if args.branch_index_start is None or args.branch_index_count is None:
        raise CarrionSequenceContextArchiveError(
            "--branch-index-start and --branch-index-count must be provided together"
        )
    start = int(args.branch_index_start)
    count = int(args.branch_index_count)
    if start < 0 or count < 0:
        raise CarrionSequenceContextArchiveError(
            "branch index start/count must be non-negative"
        )
    return _validate_branch_indexes(tuple(range(start, start + count)))


def _parse_int_list(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_repeated_int_list(values: Sequence[str] | None) -> tuple[int, ...] | None:
    if not values:
        return None
    parsed: list[int] = []
    for value in values:
        parsed.extend(int(item.strip()) for item in value.split(",") if item.strip())
    return _validate_branch_indexes(tuple(sorted(set(parsed))))


def _validate_branch_indexes(values: tuple[int, ...]) -> tuple[int, ...]:
    invalid = [
        int(index)
        for index in values
        if int(index) < 0 or int(index) >= len(CARRION_BRANCH_REASONS)
    ]
    if invalid:
        raise CarrionSequenceContextArchiveError(
            f"invalid carrion branch indexes: {invalid}"
        )
    return values


def _print_progress(payload: Mapping[str, object]) -> None:
    print(
        "carrion_sequence_context_archive_progress="
        f"event={payload.get('event')} "
        f"source={payload.get('source')} "
        f"shard_id={payload.get('shard_id')} "
        f"fixture={payload.get('fixture')} "
        f"seed={payload.get('seed')} "
        f"branch_index={payload.get('branch_index')} "
        f"branch_reason={payload.get('branch_reason')} "
        f"branch_id={payload.get('branch_id')} "
        f"source_row_count={payload.get('source_row_count')} "
        f"elapsed_seconds={payload.get('elapsed_seconds')}"
    )


def _print_summary(
    report: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    output: Path,
    dataset_output: Path,
) -> None:
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source_integrity"))
    separation = _mapping(report.get("sequence_context_separation"))
    coverage = _mapping(report.get("coverage"))
    leakage = _mapping(report.get("leakage_scan"))
    replay = _mapping(report.get("replay_verification"))
    print(f"carrion_sequence_context_archive_report={output}")
    print(f"carrion_sequence_context_archive_dataset={dataset_output}")
    print(f"classification={classification.get('primary')}")
    print(f"dataset_row_count={len(rows)}")
    print(f"harmful_source_row_count={coverage.get('harmful_source_row_count')}")
    print(f"safe_comparator_row_count={coverage.get('safe_comparator_row_count')}")
    print(
        "sequence_separated_harmful_source_count="
        f"{separation.get('sequence_separated_harmful_source_count')}"
    )
    print(
        "all_harmful_sources_sequence_separated="
        f"{separation.get('all_harmful_sources_sequence_separated')}"
    )
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"source_integrity_failures={source.get('failures')}")
    print(f"replay_verification_complete={replay.get('complete')}")
    print(f"leakage_scan_passed={leakage.get('passed')}")
    print(f"exact_digest={report.get('exact_digest')}")
    print(f"training_authorized={report.get('training_authorized')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"runtime_promotion_allowed={report.get('runtime_promotion_allowed')}")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
