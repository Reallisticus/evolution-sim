from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.mind.candidate_campaign import (
    DEFAULT_MIN_SAFE_LABEL_COUNT,
    DEFAULT_V145_REPORT_PATH,
    CandidateCampaignError,
)
from evolution_sim.mind.carrion_specific_archive_expansion import (
    DEFAULT_BRANCH_RESULT_CHUNK_DIR,
    DEFAULT_DATASET_OUTPUT_PATH,
    DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    DEFAULT_MAX_CANDIDATE_ACTIONS,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TICKS,
    TARGET_CARRION_SEEDS,
    CarrionSpecificArchiveExpansionError,
    build_carrion_specific_archive_expansion_report,
    generate_carrion_specific_archive_branch_results,
    load_carrion_specific_branch_results,
    merge_carrion_specific_archive_expansion_shards,
    write_carrion_specific_archive_expansion_dataset,
    write_carrion_specific_archive_expansion_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 carrion-specific archive "
            "expansion report. This command evaluates branch actions only; "
            "it never trains, promotes, or changes runtime defaults."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=DEFAULT_DATASET_OUTPUT_PATH,
    )
    parser.add_argument(
        "--branch-result-chunk-dir",
        type=Path,
        default=DEFAULT_BRANCH_RESULT_CHUNK_DIR,
    )
    parser.add_argument(
        "--resume-branch-results",
        action="store_true",
        help="Reuse matching deterministic branch-result chunks.",
    )
    parser.add_argument(
        "--use-existing-branch-results",
        action="store_true",
        help="Read --branch-results instead of generating fresh branch evidence.",
    )
    parser.add_argument(
        "--branch-results",
        type=Path,
        default=None,
        help="JSON list or report object containing branch_results.",
    )
    parser.add_argument(
        "--shard-id",
        type=str,
        default=None,
        help="Optional stable shard identifier recorded in generation evidence.",
    )
    parser.add_argument(
        "--branch-index-include",
        action="append",
        default=None,
        help=(
            "Carrion branch indexes to evaluate in this shard. May be repeated "
            "or comma-separated."
        ),
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge previously generated carrion-specific shard reports.",
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
        help="Optional branch-result chunk directory to verify during merge.",
    )
    parser.add_argument(
        "--allow-partial-shard-evidence",
        action="store_true",
        help="Allow partial shard reports during merge. Defaults to fail closed.",
    )
    parser.add_argument(
        "--v145-report",
        type=Path,
        default=DEFAULT_V145_REPORT_PATH,
        help="v145 invalid-resolution-risk label blacklist report.",
    )
    parser.add_argument(
        "--allow-missing-v145-report",
        action="store_true",
        help="Proceed with an empty blacklist if --v145-report is absent.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(seed) for seed in TARGET_CARRION_SEEDS),
        help="Comma-separated carrion_only seed list.",
    )
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument(
        "--max-branch-points-per-seed",
        type=int,
        default=DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    )
    parser.add_argument(
        "--max-candidate-actions",
        type=int,
        default=DEFAULT_MAX_CANDIDATE_ACTIONS,
        help="0 means evaluate every currently valid public-mask action.",
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
        help=(
            "Stop between branch points after this many seconds and write a "
            "partial no-training report."
        ),
    )
    parser.add_argument(
        "--min-safe-label-count",
        type=int,
        default=DEFAULT_MIN_SAFE_LABEL_COUNT,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        seeds = _parse_int_list(args.seeds)
        branch_index_include = _parse_repeated_int_list(args.branch_index_include)
        v145_report = _load_optional_json(
            args.v145_report,
            allow_missing=bool(args.allow_missing_v145_report),
        )
        if args.merge_shards:
            shard_report_paths = list(args.merge_shard_report or [])
            if not shard_report_paths:
                raise CarrionSpecificArchiveExpansionError(
                    "--merge-shards requires at least one --merge-shard-report"
                )
            shard_reports = [
                _load_json_object(path, label="merge shard report")
                for path in shard_report_paths
            ]
            report, rows = merge_carrion_specific_archive_expansion_shards(
                shard_reports=shard_reports,
                shard_report_paths=shard_report_paths,
                shard_chunk_dirs=list(args.merge_shard_chunk_dir or []),
                v145_report=v145_report,
                allow_partial_shard_evidence=bool(args.allow_partial_shard_evidence),
                target_seeds=seeds,
                input_paths={
                    "v145_report": args.v145_report,
                    "target_carrion_seeds": [int(seed) for seed in seeds],
                    "merge_shard_reports": shard_report_paths,
                    "merge_shard_chunk_dirs": list(args.merge_shard_chunk_dir or []),
                },
            )
        elif args.use_existing_branch_results:
            if args.branch_results is None:
                raise CarrionSpecificArchiveExpansionError(
                    "--use-existing-branch-results requires --branch-results"
                )
            branch_results = load_carrion_specific_branch_results(args.branch_results)
            generation_status = None
            generation_evidence: dict[str, object] = {
                "policy": "precomputed_carrion_specific_archive_branch_results_v1",
                "branch_results": str(args.branch_results),
                "shard_id": args.shard_id,
                "branch_index_include": (
                    None
                    if branch_index_include is None
                    else [int(index) for index in branch_index_include]
                ),
            }
            branch_results_path = args.branch_results
            report, rows = build_carrion_specific_archive_expansion_report(
                branch_results=branch_results,
                v145_report=v145_report,
                target_seeds=seeds,
                min_safe_label_count=int(args.min_safe_label_count),
                generation_status=generation_status,
                generation_evidence=generation_evidence,
                input_paths={
                    "v145_report": args.v145_report,
                    "branch_results": branch_results_path,
                },
            )
        else:
            generated = generate_carrion_specific_archive_branch_results(
                seeds=seeds,
                ticks=int(args.ticks),
                max_branch_points_per_seed=int(args.max_branch_points_per_seed),
                max_candidate_actions=int(args.max_candidate_actions),
                shard_id=args.shard_id,
                branch_index_include=branch_index_include,
                verify_replay=not bool(args.no_verify_replay),
                branch_result_chunk_dir=args.branch_result_chunk_dir,
                resume_branch_results=bool(args.resume_branch_results),
                max_wall_seconds=args.max_wall_seconds,
                progress_callback=_print_progress,
            )
            branch_results = _branch_results(generated)
            generation_status = _mapping(generated.get("generation_status"))
            generation_evidence = {
                key: value
                for key, value in generated.items()
                if key != "branch_results"
            }
            branch_results_path = None
            report, rows = build_carrion_specific_archive_expansion_report(
                branch_results=branch_results,
                v145_report=v145_report,
                target_seeds=seeds,
                min_safe_label_count=int(args.min_safe_label_count),
                generation_status=generation_status,
                generation_evidence=generation_evidence,
                input_paths={
                    "v145_report": args.v145_report,
                    "branch_results": branch_results_path,
                },
            )
        write_carrion_specific_archive_expansion_report(report, args.output)
        write_carrion_specific_archive_expansion_dataset(rows, args.dataset_output)
    except (
        OSError,
        ValueError,
        CandidateCampaignError,
        CarrionSpecificArchiveExpansionError,
    ) as exc:
        raise SystemExit(
            f"failed to build carrion-specific archive expansion: {exc}"
        ) from exc
    _print_summary(report, rows, args.output, args.dataset_output)


def _load_optional_json(path: Path, *, allow_missing: bool) -> dict[str, object]:
    if not path.exists():
        if allow_missing:
            return {"route_decision": {"label_blacklist": []}}
        raise CarrionSpecificArchiveExpansionError(f"missing v145 report: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CarrionSpecificArchiveExpansionError(
            f"v145 report must be a JSON object: {path}"
        )
    return payload


def _load_json_object(path: Path, *, label: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CarrionSpecificArchiveExpansionError(
            f"{label} must be a JSON object: {path}"
        )
    return payload


def _branch_results(payload: Mapping[str, object]) -> list[dict[str, object]]:
    value = payload.get("branch_results")
    if not isinstance(value, list):
        raise CarrionSpecificArchiveExpansionError(
            "generated payload missing branch_results"
        )
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _parse_int_list(value: str | None) -> tuple[int, ...]:
    if value is None:
        return tuple(TARGET_CARRION_SEEDS)
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parse_repeated_int_list(values: Sequence[str] | None) -> tuple[int, ...] | None:
    if not values:
        return None
    parsed: list[int] = []
    for value in values:
        parsed.extend(int(item.strip()) for item in value.split(",") if item.strip())
    return tuple(sorted(set(parsed)))


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _print_progress(payload: Mapping[str, object]) -> None:
    print(
        "carrion_specific_archive_expansion_progress="
        f"event={payload.get('event')} "
        f"source={payload.get('source')} "
        f"shard_id={payload.get('shard_id')} "
        f"fixture={payload.get('fixture')} "
        f"seed={payload.get('seed')} "
        f"branch_point={payload.get('branch_point_index')} "
        f"branch_reason={payload.get('branch_reason')} "
        f"branch_id={payload.get('branch_id')} "
        f"action_count={payload.get('action_count')} "
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
    action_dist = _mapping(report.get("action_distribution"))
    replay = _mapping(report.get("replay_verification"))
    leakage = _mapping(report.get("leakage_scan"))
    print(f"carrion_specific_archive_expansion_report={output}")
    print(f"carrion_specific_archive_expansion_dataset={dataset_output}")
    print(f"classification={classification.get('primary')}")
    print(f"safe_label_count={report.get('safe_label_count')}")
    print(f"dataset_row_count={len(rows)}")
    print(
        "dominant_safe_label_action_share="
        f"{action_dist.get('dominant_safe_label_action_share')}"
    )
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"source_integrity_failures={source.get('failures')}")
    print(f"replay_verification_complete={replay.get('complete')}")
    print(f"leakage_scan_passed={leakage.get('passed')}")
    print(f"exact_digest={report.get('exact_digest')}")
    print(f"training_authorized={report.get('training_authorized')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"runtime_promotion_allowed={report.get('runtime_promotion_allowed')}")


if __name__ == "__main__":
    main()
