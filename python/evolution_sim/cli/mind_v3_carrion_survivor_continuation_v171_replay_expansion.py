from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v170_diagnostic_portfolio_matrix import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V170_REPORT_PATH,
    DEFAULT_SHARD_PLAN_OUTPUT_PATH as DEFAULT_V170_SHARD_PLAN_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_v171_replay_expansion import (
    DEFAULT_DATASET_OUTPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    EXPECTED_V170_CLASSIFICATION,
    EXPECTED_V170_EXACT_DIGEST,
    EXPECTED_V170_SHARD_PLAN_DIGEST,
    CarrionSurvivorContinuationV171ReplayExpansionError,
    run_carrion_survivor_continuation_v171_replay_expansion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run diagnostics-only v171 carrion survivor-continuation replay "
            "expansion from the v170 shard plan. This executes exact branch "
            "replay evidence or merges shard evidence; it does not train, "
            "retrain scorers, create runtime artifacts, tune thresholds/k, "
            "change runtime behavior, or change replay/viewer schema."
        )
    )
    parser.add_argument("--v170-report", type=Path, default=DEFAULT_V170_REPORT_PATH)
    parser.add_argument(
        "--shard-plan",
        type=Path,
        default=DEFAULT_V170_SHARD_PLAN_PATH,
        help="v170 replay-expansion shard-plan JSONL. Defaults to the v170 output path so v170 JSONL command shapes are valid.",
    )
    parser.add_argument("--v165-dataset", type=Path, default=None)
    parser.add_argument("--shard-id", default=None)
    parser.add_argument("--seed-include", type=int, default=None)
    parser.add_argument("--branch-window", action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=DEFAULT_DATASET_OUTPUT_PATH,
    )
    parser.add_argument("--fail-on-partial-shard", action="store_true")
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--merge-shard-report", type=Path, action="append", default=[])
    parser.add_argument("--merge-shard-dataset", type=Path, action="append", default=[])
    parser.add_argument(
        "--allow-partial-shard-evidence",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--expected-v170-exact-digest",
        default=EXPECTED_V170_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v170-classification",
        default=EXPECTED_V170_CLASSIFICATION,
    )
    parser.add_argument(
        "--expected-v170-shard-plan-digest",
        default=EXPECTED_V170_SHARD_PLAN_DIGEST,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v171_replay_expansion(
            v170_report_path=args.v170_report,
            shard_plan_path=args.shard_plan,
            v165_dataset_path=args.v165_dataset,
            shard_id=args.shard_id,
            seed_include=args.seed_include,
            branch_windows=args.branch_window,
            output_path=args.output,
            dataset_output_path=args.dataset_output,
            fail_on_partial_shard=args.fail_on_partial_shard,
            merge_shards=args.merge_shards,
            merge_shard_reports=args.merge_shard_report,
            merge_shard_datasets=args.merge_shard_dataset,
            allow_partial_shard_evidence=args.allow_partial_shard_evidence,
            expected_v170_exact_digest=args.expected_v170_exact_digest,
            expected_v170_classification=args.expected_v170_classification,
            expected_v170_shard_plan_digest=args.expected_v170_shard_plan_digest,
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationV171ReplayExpansionError,
    ) as exc:
        raise SystemExit(
            "failed to run v171 carrion survivor-continuation replay expansion: "
            f"{exc}"
        ) from exc
    _print_summary(report, args.output, args.dataset_output)
    classification = str(report.get("classification", {}).get("primary", ""))
    if args.fail_on_partial_shard and classification.endswith(
        "partial_shard_closed_no_training"
    ):
        raise SystemExit("v171 shard closed as partial")


def _print_summary(report: dict[str, object], output: Path, dataset_output: Path) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    source = report.get("source_validation")
    source_payload = source if isinstance(source, dict) else {}
    metrics = report.get("metrics")
    metrics_payload = metrics if isinstance(metrics, dict) else {}
    partial = report.get("partial_shard_status")
    partial_payload = partial if isinstance(partial, dict) else {}
    dataset = report.get("dataset")
    dataset_payload = dataset if isinstance(dataset, dict) else {}
    print(f"carrion_survivor_continuation_v171_replay_expansion_report={output}")
    print(f"dataset_output={dataset_output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={source_payload.get('passed')}")
    print(f"mode={report.get('mode')}")
    print(f"materialized_branch_point_count={metrics_payload.get('materialized_branch_point_count')}")
    print(f"replay_verified_run_count={metrics_payload.get('replay_verified_run_count')}")
    print(f"candidate_run_count={metrics_payload.get('candidate_run_count')}")
    print(
        "support_narrows_broad_set_valued_candidates="
        f"{metrics_payload.get('support_narrows_broad_set_valued_candidates')}"
    )
    print(
        "all_v170_zero_hit_seeds_have_replay_support="
        f"{metrics_payload.get('all_v170_zero_hit_seeds_have_replay_support')}"
    )
    print(f"partial_shard={partial_payload.get('partial')}")
    print(f"dataset_row_count={dataset_payload.get('row_count')}")
    print(f"dataset_digest={dataset_payload.get('dataset_digest')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"scorer_retraining_ran={report.get('scorer_retraining_ran')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"exact_digest={report.get('exact_digest')}")


if __name__ == "__main__":
    main()
