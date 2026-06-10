from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.public_sequence_context_branch_evidence import (
    DEFAULT_BP3_SAFE_ARCHIVE_REPORT_PATH,
    DEFAULT_CHUNK_DIR,
    DEFAULT_HISTORY_WINDOW,
    DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    DEFAULT_MAX_CANDIDATE_ACTIONS,
    DEFAULT_MIN_PRIOR_PUBLIC_STEPS,
    DEFAULT_OUTPUT_PATH,
    PublicSequenceContextBranchEvidenceError,
    build_public_sequence_context_branch_evidence_report_from_paths,
    merge_public_sequence_context_branch_evidence_reports,
    write_public_sequence_context_branch_evidence_report,
)
from evolution_sim.mind.safe_archive_failure_autopsy import (
    DEFAULT_BP3_BRANCH_EVIDENCE_PATH,
    DEFAULT_BP3_DATASET_PATH,
    load_json_report,
)
from evolution_sim.mind.candidate_campaign import load_safe_archive_expansion_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect diagnostics-only public sequence-context branch evidence "
            "for bp3 one-step aliases. This command never trains or promotes."
        )
    )
    parser.add_argument(
        "--bp3-safe-archive-report",
        type=Path,
        default=DEFAULT_BP3_SAFE_ARCHIVE_REPORT_PATH,
    )
    parser.add_argument(
        "--bp3-dataset",
        type=Path,
        default=DEFAULT_BP3_DATASET_PATH,
    )
    parser.add_argument(
        "--bp3-branch-evidence",
        type=Path,
        default=DEFAULT_BP3_BRANCH_EVIDENCE_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--chunk-dir", type=Path, default=DEFAULT_CHUNK_DIR)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse matching deterministic branch-result chunks from --chunk-dir.",
    )
    parser.add_argument(
        "--max-wall-seconds",
        type=float,
        default=None,
        help=(
            "Stop between branch points after this many wall seconds and write "
            "a partial no-training report."
        ),
    )
    parser.add_argument(
        "--fixture",
        choices=("both", "broad", "carrion_only"),
        default="both",
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
        help="First per-seed later-context branch index to include in this shard.",
    )
    parser.add_argument(
        "--branch-index-count",
        type=int,
        default=None,
        help="Number of branch indexes to include from --branch-index-start.",
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
        help="Stable operator-provided shard id recorded in report inputs.",
    )
    parser.add_argument(
        "--max-branch-points-per-seed",
        type=int,
        default=DEFAULT_MAX_BRANCH_POINTS_PER_SEED,
    )
    parser.add_argument(
        "--max-candidate-actions",
        type=int,
        default=DEFAULT_MAX_CANDIDATE_ACTIONS,
        help="0 means evaluate every valid public action-mask action.",
    )
    parser.add_argument(
        "--history-window",
        type=int,
        default=DEFAULT_HISTORY_WINDOW,
    )
    parser.add_argument(
        "--min-prior-public-steps",
        type=int,
        default=DEFAULT_MIN_PRIOR_PUBLIC_STEPS,
    )
    parser.add_argument(
        "--merge-shards",
        action="store_true",
        help="Merge shard reports instead of generating fresh branch evidence.",
    )
    parser.add_argument(
        "--merge-shard-report",
        type=Path,
        action="append",
        default=[],
        help="Shard report JSON to merge. May be provided multiple times.",
    )
    parser.add_argument(
        "--merge-shard-chunk-dir",
        type=Path,
        action="append",
        default=[],
        help="Shard chunk directory to verify against reports during merge.",
    )
    parser.add_argument(
        "--allow-partial-shard-evidence",
        action="store_true",
        help="Allow partial shard inputs and produce a partial no-training report.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.merge_shards:
            safe_report = load_json_report(args.bp3_safe_archive_report)
            dataset_rows = load_safe_archive_expansion_dataset(args.bp3_dataset)
            branch_evidence = load_json_report(args.bp3_branch_evidence)
            report = merge_public_sequence_context_branch_evidence_reports(
                bp3_safe_archive_report=safe_report,
                bp3_dataset_rows=dataset_rows,
                bp3_branch_evidence=branch_evidence,
                shard_reports=[
                    load_json_report(path) for path in args.merge_shard_report
                ],
                shard_chunk_dirs=args.merge_shard_chunk_dir,
                allow_partial_shard_evidence=bool(args.allow_partial_shard_evidence),
                input_paths={
                    "bp3_safe_archive_report": args.bp3_safe_archive_report,
                    "bp3_dataset": args.bp3_dataset,
                    "bp3_branch_evidence": args.bp3_branch_evidence,
                },
            )
        else:
            report = build_public_sequence_context_branch_evidence_report_from_paths(
                bp3_safe_archive_report_path=args.bp3_safe_archive_report,
                bp3_dataset_path=args.bp3_dataset,
                bp3_branch_evidence_path=args.bp3_branch_evidence,
                max_branch_points_per_seed=int(args.max_branch_points_per_seed),
                max_candidate_actions=int(args.max_candidate_actions),
                history_window=int(args.history_window),
                min_prior_public_steps=int(args.min_prior_public_steps),
                fixture_selection=_fixture_selection(args.fixture),
                seed_include=_parse_int_list(args.seed_include),
                branch_index_include=_selected_branch_indexes(args),
                branch_evidence_chunk_dir=args.chunk_dir,
                resume_branch_evidence=bool(args.resume),
                max_wall_seconds=args.max_wall_seconds,
                shard_id=args.shard_id,
                progress_callback=_print_progress,
            )
        write_public_sequence_context_branch_evidence_report(report, args.output)
    except (
        OSError,
        ValueError,
        PublicSequenceContextBranchEvidenceError,
    ) as exc:
        raise SystemExit(
            f"failed to build public sequence-context branch evidence: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _selected_branch_indexes(args: argparse.Namespace) -> tuple[int, ...]:
    include = _parse_int_list(args.branch_index_include)
    return _safe_selected_branch_indexes(
        max_branch_points_per_seed=int(args.max_branch_points_per_seed),
        branch_index_start=args.branch_index_start,
        branch_index_count=args.branch_index_count,
        branch_index_include=include,
    )


def _safe_selected_branch_indexes(
    *,
    max_branch_points_per_seed: int,
    branch_index_start: int | None,
    branch_index_count: int | None,
    branch_index_include: tuple[int, ...] | None,
) -> tuple[int, ...]:
    from evolution_sim.mind.candidate_campaign import (
        _safe_archive_expansion_selected_branch_indexes,
    )

    return _safe_archive_expansion_selected_branch_indexes(
        max_branch_points_per_seed=max_branch_points_per_seed,
        branch_index_start=branch_index_start,
        branch_index_count=branch_index_count,
        branch_index_include=branch_index_include,
    )


def _fixture_selection(value: str) -> tuple[str, ...]:
    if value == "both":
        return ("broad", "carrion_only")
    return (value,)


def _parse_int_list(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _print_progress(payload: Mapping[str, object]) -> None:
    print(
        "public_sequence_context_branch_evidence_progress="
        f"event={payload.get('event')} "
        f"source={payload.get('source')} "
        f"fixture={payload.get('fixture')} "
        f"seed={payload.get('seed')} "
        f"branch_point={payload.get('branch_point_index')} "
        f"branch_id={payload.get('branch_id')} "
        f"action_count={payload.get('action_count')} "
        f"elapsed_seconds={payload.get('elapsed_seconds')}"
    )


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    source = report.get("source_integrity")
    source_payload = source if isinstance(source, dict) else {}
    aliasing = report.get("bp3_action_only_aliasing")
    alias_payload = aliasing if isinstance(aliasing, dict) else {}
    status = report.get("generation_status")
    status_payload = status if isinstance(status, dict) else {}
    auth = report.get("authorization_block")
    auth_payload = auth if isinstance(auth, dict) else {}
    print(f"public_sequence_context_branch_evidence_report={output_path}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"generation_state={status_payload.get('state')}")
    print(f"source_integrity_passed={source_payload.get('passed')}")
    print(f"branch_result_count={report.get('branch_result_count')}")
    print(
        "action_only_alias_count="
        f"{alias_payload.get('action_only_alias_count')}"
    )
    print(
        "sequence_separated_alias_count="
        f"{alias_payload.get('sequence_separated_alias_count')}"
    )
    print(
        "safe_action_only_alias_count="
        f"{alias_payload.get('safe_action_only_alias_count')}"
    )
    print(
        "safe_sequence_separated_alias_count="
        f"{alias_payload.get('safe_sequence_separated_alias_count')}"
    )
    print(f"training_authorized={auth_payload.get('training_authorized')}")
    print(f"promotion_authorized={auth_payload.get('promotion_authorized')}")
    print(
        "runtime_promotion_allowed="
        f"{auth_payload.get('runtime_promotion_allowed')}"
    )


if __name__ == "__main__":
    main()
