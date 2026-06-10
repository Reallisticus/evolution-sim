from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path

from evolution_sim.mind.carrion_sequence_context_ablation_shadow_audit import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SIMILAR_ACTION_MASK_THRESHOLD,
    DEFAULT_V151_DATASET_PATH,
    DEFAULT_V151_REPORT_PATH,
    CarrionSequenceContextAblationShadowAuditError,
    build_carrion_sequence_context_ablation_shadow_audit_report_from_paths,
    load_json_report,
    merge_carrion_sequence_context_ablation_shadow_audit_shards,
    write_carrion_sequence_context_ablation_shadow_audit_report,
)
from evolution_sim.mind.carrion_archive_override_autopsy import (
    DEFAULT_CARRION_SEEDS,
    DEFAULT_TICKS,
    DEFAULT_V149_CARRION_ARTIFACT_PATH,
    DEFAULT_V149_CARRION_TRAIN_EVAL_REPORT_PATH,
)
from evolution_sim.mind.carrion_sequence_context_archive import load_jsonl_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only v152 carrion sequence-context "
            "ablation/shadow audit. This command never trains, promotes, "
            "creates a runtime artifact, or changes runtime defaults."
        )
    )
    parser.add_argument(
        "--v151-report",
        type=Path,
        default=DEFAULT_V151_REPORT_PATH,
    )
    parser.add_argument(
        "--v151-dataset",
        type=Path,
        default=DEFAULT_V151_DATASET_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--shard-id", type=str, default=None)
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
        "--row-index-include",
        action="append",
        default=None,
        help=(
            "v151 dataset row indexes to include in this shard. May be repeated "
            "or comma-separated."
        ),
    )
    parser.add_argument("--row-index-start", type=int, default=None)
    parser.add_argument("--row-index-count", type=int, default=None)
    parser.add_argument(
        "--similar-action-mask-threshold",
        type=float,
        default=DEFAULT_SIMILAR_ACTION_MASK_THRESHOLD,
    )
    parser.add_argument(
        "--shadow-live-probe",
        action="store_true",
        help=(
            "Run the optional report-only carrion live probe with in-memory "
            "suppression of the three v150 harmful support labels."
        ),
    )
    parser.add_argument(
        "--shadow-artifact",
        type=Path,
        default=DEFAULT_V149_CARRION_ARTIFACT_PATH,
    )
    parser.add_argument(
        "--shadow-train-eval-report",
        type=Path,
        default=DEFAULT_V149_CARRION_TRAIN_EVAL_REPORT_PATH,
    )
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument(
        "--target-seeds",
        type=str,
        default=",".join(str(seed) for seed in DEFAULT_CARRION_SEEDS),
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge previously generated v152 shard reports.",
    )
    parser.add_argument(
        "--merge-shard-report",
        type=Path,
        action="append",
        default=None,
        help="Shard report JSON to merge. Repeat for every shard.",
    )
    parser.add_argument(
        "--allow-partial-shard-evidence",
        action="store_true",
        help="Allow partial merged row coverage. Defaults to fail closed.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.merge_shards:
            shard_paths = list(args.merge_shard_report or [])
            if not shard_paths:
                raise CarrionSequenceContextAblationShadowAuditError(
                    "--merge-shards requires at least one --merge-shard-report"
                )
            report = merge_carrion_sequence_context_ablation_shadow_audit_shards(
                v151_report=load_json_report(args.v151_report),
                v151_rows=load_jsonl_rows(args.v151_dataset),
                shard_reports=[load_json_report(path) for path in shard_paths],
                shard_report_paths=shard_paths,
                allow_partial_shard_evidence=bool(
                    args.allow_partial_shard_evidence
                ),
                input_paths={
                    "v151_report": args.v151_report,
                    "v151_dataset": args.v151_dataset,
                },
            )
        else:
            report = (
                build_carrion_sequence_context_ablation_shadow_audit_report_from_paths(
                    v151_report_path=args.v151_report,
                    v151_dataset_path=args.v151_dataset,
                    shadow_artifact_path=args.shadow_artifact,
                    shadow_train_eval_report_path=args.shadow_train_eval_report,
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
                    shadow_live_probe=bool(args.shadow_live_probe),
                    target_seeds=_parse_int_list(args.target_seeds)
                    or DEFAULT_CARRION_SEEDS,
                    ticks=int(args.ticks),
                )
            )
        write_carrion_sequence_context_ablation_shadow_audit_report(
            report,
            args.output,
        )
    except (OSError, ValueError, CarrionSequenceContextAblationShadowAuditError) as exc:
        raise SystemExit(
            "failed to build v152 carrion sequence-context ablation/shadow "
            f"audit: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _parse_int_list(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return parsed


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
    coverage = _mapping(report.get("coverage"))
    comparator = _mapping(report.get("comparator_availability"))
    ablations = _mapping(report.get("ablations"))
    modes = _mapping(ablations.get("modes"))
    shadow = _mapping(report.get("shadow_live_probe"))
    print(f"carrion_sequence_context_ablation_shadow_audit_report={output}")
    print(f"classification={classification.get('primary')}")
    print(f"audit_row_count={report.get('audit_row_count')}")
    print(f"harmful_source_row_count={coverage.get('harmful_source_row_count')}")
    print(f"safe_comparator_row_count={coverage.get('safe_comparator_row_count')}")
    print(
        "same_label_action_available_count="
        f"{comparator.get('same_label_action_available_count')}"
    )
    print(
        "exact_one_step_available_count="
        f"{comparator.get('exact_one_step_available_count')}"
    )
    print(
        "zero_prior_safe_available_count="
        f"{comparator.get('zero_prior_safe_available_count')}"
    )
    print(
        "nonzero_prior_harmful_available_count="
        f"{comparator.get('nonzero_prior_harmful_available_count')}"
    )
    for key in (
        "one_step_only_features",
        "prior_length_only_features",
        "public_hydration_reproduction_marker_features",
        "full_public_prior_sequence_features",
    ):
        mode = _mapping(modes.get(key))
        print(
            f"{key}_all_harmful_sources_separated="
            f"{mode.get('all_harmful_sources_separated')}"
        )
    print(f"shadow_live_probe_status={shadow.get('status')}")
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
