from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.mind.carrion_sequence_context_ablation_shadow_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V152_REPORT_PATH,
    DEFAULT_SIMILAR_ACTION_MASK_THRESHOLD,
)
from evolution_sim.mind.carrion_sequence_context_archive import (
    DEFAULT_DATASET_OUTPUT_PATH as DEFAULT_V151_DATASET_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V151_REPORT_PATH,
    load_json_report,
    load_jsonl_rows,
)
from evolution_sim.mind.carrion_sequence_context_comparator_support_closeout import (
    DEFAULT_COMPARATOR_CHUNK_DIR,
    DEFAULT_OUTPUT_PATH,
    CarrionSequenceContextComparatorSupportCloseoutError,
    build_carrion_sequence_context_comparator_support_closeout_report_from_paths,
    merge_carrion_sequence_context_comparator_support_closeout_shards,
    write_carrion_sequence_context_comparator_support_closeout_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only v153 carrion sequence-context "
            "comparator-support expansion/closeout. This command never trains, "
            "promotes, creates a runtime artifact, or changes runtime defaults."
        )
    )
    parser.add_argument("--v151-report", type=Path, default=DEFAULT_V151_REPORT_PATH)
    parser.add_argument("--v151-dataset", type=Path, default=DEFAULT_V151_DATASET_PATH)
    parser.add_argument("--v152-report", type=Path, default=DEFAULT_V152_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--shard-id", type=str, default=None)
    parser.add_argument("--seed-include", type=str, default=None)
    parser.add_argument("--branch-index-include", action="append", default=None)
    parser.add_argument("--row-index-include", action="append", default=None)
    parser.add_argument("--row-index-start", type=int, default=None)
    parser.add_argument("--row-index-count", type=int, default=None)
    parser.add_argument(
        "--similar-action-mask-threshold",
        type=float,
        default=DEFAULT_SIMILAR_ACTION_MASK_THRESHOLD,
    )
    parser.add_argument(
        "--comparator-chunk-dir",
        type=Path,
        default=DEFAULT_COMPARATOR_CHUNK_DIR,
    )
    parser.add_argument(
        "--no-comparator-chunks",
        action="store_true",
        help="Do not write v153 comparator evidence chunks during generation.",
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge previously generated v153 shard reports.",
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
        help="Optional v153 comparator evidence chunk directory to verify.",
    )
    parser.add_argument(
        "--allow-partial-shard-evidence",
        action="store_true",
        help="Allow partial merged comparator evidence. Defaults to fail closed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.merge_shards:
            shard_paths = list(args.merge_shard_report or [])
            if not shard_paths:
                raise CarrionSequenceContextComparatorSupportCloseoutError(
                    "--merge-shards requires at least one --merge-shard-report"
                )
            report = merge_carrion_sequence_context_comparator_support_closeout_shards(
                v151_report=load_json_report(args.v151_report),
                v151_rows=load_jsonl_rows(args.v151_dataset),
                v152_report=load_json_report(args.v152_report),
                shard_reports=[load_json_report(path) for path in shard_paths],
                shard_report_paths=shard_paths,
                shard_chunk_dirs=list(args.merge_shard_chunk_dir or []),
                allow_partial_shard_evidence=bool(
                    args.allow_partial_shard_evidence
                ),
                input_paths={
                    "v151_report": args.v151_report,
                    "v151_dataset": args.v151_dataset,
                    "v152_report": args.v152_report,
                },
            )
        else:
            report = (
                build_carrion_sequence_context_comparator_support_closeout_report_from_paths(
                    v151_report_path=args.v151_report,
                    v151_dataset_path=args.v151_dataset,
                    v152_report_path=args.v152_report,
                    seed_include=_parse_int_list(args.seed_include),
                    branch_index_include=_parse_repeated_int_list(
                        args.branch_index_include
                    ),
                    row_index_include=_parse_repeated_int_list(
                        args.row_index_include
                    ),
                    row_index_start=args.row_index_start,
                    row_index_count=args.row_index_count,
                    shard_id=args.shard_id,
                    similar_action_mask_threshold=float(
                        args.similar_action_mask_threshold
                    ),
                    comparator_chunk_dir=(
                        None
                        if args.no_comparator_chunks
                        else args.comparator_chunk_dir
                    ),
                )
            )
        write_carrion_sequence_context_comparator_support_closeout_report(
            report,
            args.output,
        )
    except (OSError, ValueError, CarrionSequenceContextComparatorSupportCloseoutError) as exc:
        raise SystemExit(
            "failed to build v153 carrion sequence-context comparator-support "
            f"closeout: {exc}"
        ) from exc
    _print_summary(report, args.output)


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


def _print_summary(report: Mapping[str, object], output: Path) -> None:
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source_integrity"))
    support = _mapping(report.get("comparator_support"))
    recommendation = _mapping(report.get("recommendation"))
    print(f"carrion_sequence_context_comparator_support_closeout_report={output}")
    print(f"classification={classification.get('primary')}")
    print(
        "comparator_evidence_row_count="
        f"{report.get('comparator_evidence_row_count')}"
    )
    print(
        "exact_one_step_safe_comparator_total="
        f"{support.get('exact_one_step_safe_comparator_total')}"
    )
    print(
        "zero_prior_safe_comparator_total="
        f"{support.get('zero_prior_safe_comparator_total')}"
    )
    print(
        "nonzero_prior_harmful_comparator_total="
        f"{support.get('nonzero_prior_harmful_comparator_total')}"
    )
    print(
        "same_label_action_exact_mask_safe_comparator_total="
        f"{support.get('same_label_action_exact_mask_safe_comparator_total')}"
    )
    print(
        "same_label_action_similar_mask_safe_comparator_total="
        f"{support.get('same_label_action_similar_mask_safe_comparator_total')}"
    )
    print(f"route_closed={recommendation.get('route_closed')}")
    print(f"recommended_next_route={recommendation.get('recommended_next_route')}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"source_integrity_failures={source.get('failures')}")
    print(f"exact_digest={report.get('exact_digest')}")
    print(f"training_authorized={report.get('training_authorized')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"runtime_promotion_allowed={report.get('runtime_promotion_allowed')}")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
