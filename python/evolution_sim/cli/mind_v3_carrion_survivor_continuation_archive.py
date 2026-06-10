from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.mind.candidate_campaign import (
    DEFAULT_MIN_SAFE_LABEL_COUNT,
    DEFAULT_V145_REPORT_PATH,
)
from evolution_sim.mind.carrion_archive_override_autopsy import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V150_AUTOPSY_PATH,
)
from evolution_sim.mind.carrion_sequence_context_comparator_support_closeout import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V153_CLOSEOUT_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_archive import (
    DEFAULT_BRANCH_RESULT_CHUNK_DIR,
    DEFAULT_DATASET_OUTPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TICKS,
    TARGET_CARRION_SEEDS,
    CarrionSurvivorContinuationArchiveError,
    build_carrion_survivor_continuation_archive_from_paths,
    load_json_report,
    merge_carrion_survivor_continuation_archive_shards,
    write_carrion_survivor_continuation_archive_dataset,
    write_carrion_survivor_continuation_archive_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only v154 carrion survivor-continuation "
            "archive. This command never trains, promotes, creates a runtime "
            "artifact, or changes runtime defaults."
        )
    )
    parser.add_argument("--v150-autopsy-report", type=Path, default=DEFAULT_V150_AUTOPSY_PATH)
    parser.add_argument("--v153-report", type=Path, default=DEFAULT_V153_CLOSEOUT_PATH)
    parser.add_argument("--v145-report", type=Path, default=DEFAULT_V145_REPORT_PATH)
    parser.add_argument(
        "--allow-missing-v145-report",
        action="store_true",
        help="Proceed with an empty invalid-resolution blacklist if v145 is absent.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_DATASET_OUTPUT_PATH)
    parser.add_argument(
        "--branch-result-chunk-dir",
        type=Path,
        default=DEFAULT_BRANCH_RESULT_CHUNK_DIR,
    )
    parser.add_argument(
        "--no-branch-result-chunks",
        action="store_true",
        help="Do not write v154 continuation branch-result chunks during generation.",
    )
    parser.add_argument("--shard-id", type=str, default=None)
    parser.add_argument(
        "--seed-include",
        type=str,
        default=None,
        help="Comma-separated carrion_only seeds for this shard.",
    )
    parser.add_argument(
        "--branch-index-include",
        action="append",
        default=None,
        help="Carrion branch indexes to include. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--continuation-index-include",
        action="append",
        default=None,
        help="Continuation indexes to include. May be repeated or comma-separated.",
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge previously generated v154 shard reports.",
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
        help="Optional v154 branch-result chunk directory to verify during merge.",
    )
    parser.add_argument(
        "--allow-partial-shard-evidence",
        action="store_true",
        help="Allow partial merged continuation evidence. Defaults to fail closed.",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=DEFAULT_TICKS,
    )
    parser.add_argument(
        "--min-label-count",
        type=int,
        default=DEFAULT_MIN_SAFE_LABEL_COUNT,
    )
    parser.add_argument(
        "--no-verify-replay",
        action="store_true",
        help="Skip deterministic replay verification. Reports fail closed.",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help="Stop between branch points and write a partial report after this budget.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        target_seeds = TARGET_CARRION_SEEDS
        seed_include = _parse_int_list(args.seed_include)
        branch_index_include = _parse_repeated_int_list(args.branch_index_include)
        continuation_index_include = _parse_repeated_int_list(
            args.continuation_index_include
        )
        if args.merge_shards:
            shard_paths = list(args.merge_shard_report or [])
            if not shard_paths:
                raise CarrionSurvivorContinuationArchiveError(
                    "--merge-shards requires at least one --merge-shard-report"
                )
            report, rows = merge_carrion_survivor_continuation_archive_shards(
                autopsy_report=load_json_report(args.v150_autopsy_report),
                v153_report=load_json_report(args.v153_report),
                v145_report=_load_v145_report(
                    args.v145_report,
                    allow_missing=bool(args.allow_missing_v145_report),
                ),
                shard_reports=[load_json_report(path) for path in shard_paths],
                shard_report_paths=shard_paths,
                shard_chunk_dirs=list(args.merge_shard_chunk_dir or []),
                target_seeds=target_seeds,
                allow_partial_shard_evidence=bool(args.allow_partial_shard_evidence),
                input_paths={
                    "v150_autopsy_report": args.v150_autopsy_report,
                    "v153_closeout_report": args.v153_report,
                    "v145_report": args.v145_report,
                    "merge_shard_reports": shard_paths,
                    "merge_shard_chunk_dirs": list(args.merge_shard_chunk_dir or []),
                },
            )
        else:
            report, rows = build_carrion_survivor_continuation_archive_from_paths(
                autopsy_report_path=args.v150_autopsy_report,
                v153_report_path=args.v153_report,
                v145_report_path=args.v145_report,
                allow_missing_v145_report=bool(args.allow_missing_v145_report),
                target_seeds=target_seeds,
                ticks=int(args.ticks),
                min_label_count=int(args.min_label_count),
                seed_include=seed_include,
                branch_index_include=branch_index_include,
                continuation_index_include=continuation_index_include,
                shard_id=args.shard_id,
                branch_result_chunk_dir=(
                    None
                    if args.no_branch_result_chunks
                    else args.branch_result_chunk_dir
                ),
                verify_replay=not bool(args.no_verify_replay),
                max_wall_seconds=args.max_wall_seconds,
            )
        write_carrion_survivor_continuation_archive_report(report, args.output)
        write_carrion_survivor_continuation_archive_dataset(rows, args.dataset_output)
    except (OSError, ValueError, CarrionSurvivorContinuationArchiveError) as exc:
        raise SystemExit(
            f"failed to build v154 carrion survivor-continuation archive: {exc}"
        ) from exc
    _print_summary(report, rows, args.output, args.dataset_output)


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
    return tuple(sorted(set(parsed)))


def _load_v145_report(path: Path, *, allow_missing: bool) -> dict[str, object]:
    if not path.exists():
        if allow_missing:
            return {"route_decision": {"label_blacklist": []}}
        raise CarrionSurvivorContinuationArchiveError(f"missing v145 report: {path}")
    return load_json_report(path)


def _print_summary(
    report: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    output: Path,
    dataset_output: Path,
) -> None:
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source_integrity"))
    action = _mapping(report.get("action_distribution"))
    replay = _mapping(report.get("replay_verification"))
    recommendation = _mapping(report.get("recommendation"))
    dataset = _mapping(report.get("dataset"))
    print(f"carrion_survivor_continuation_archive_report={output}")
    print(f"carrion_survivor_continuation_archive_dataset={dataset_output}")
    print(f"classification={classification.get('primary')}")
    print(f"label_count={len(rows)}")
    print(
        "dominant_label_action_share="
        f"{action.get('dominant_label_action_share')}"
    )
    print(f"replay_verification_complete={replay.get('complete')}")
    print(f"leakage_scan_passed={_mapping(report.get('leakage_scan')).get('passed')}")
    print(
        "invalid_resolution_blacklist_hit_count="
        f"{_mapping(report.get('blacklist')).get('blacklist_hit_count')}"
    )
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"source_integrity_failures={source.get('failures')}")
    print(f"dataset_digest={dataset.get('dataset_digest')}")
    print(f"exact_digest={report.get('exact_digest')}")
    print(f"recommended_next_route={recommendation.get('recommended_next_route')}")
    print(f"training_authorized={report.get('training_authorized')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"runtime_promotion_allowed={report.get('runtime_promotion_allowed')}")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
