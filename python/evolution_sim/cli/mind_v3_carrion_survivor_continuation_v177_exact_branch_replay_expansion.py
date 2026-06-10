from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v176_transition_diagnostic_planner import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V176_REPORT_PATH,
    DEFAULT_V177_SHARD_PLAN_OUTPUT_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_v177_exact_branch_replay_expansion import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TRANSITION_DATASET_OUTPUT_PATH,
    EXPECTED_V176_CLASSIFICATION,
    EXPECTED_V176_EXACT_DIGEST,
    EXPECTED_V177_SHARD_PLAN_DIGEST,
    CarrionSurvivorContinuationV177ExactBranchReplayExpansionError,
    run_carrion_survivor_continuation_v177_exact_branch_replay_expansion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run diagnostics-only v177 exact branch replay expansion from the "
            "v176 shard plan. This materializes compact transition rows with "
            "current public inputs, forced action, previous same-agent public "
            "context, and next public inputs. It does not train, fit, create "
            "runtime artifacts, change runtime action selection, run "
            "shadow/live evaluation, relax gates, or authorize promotion."
        )
    )
    parser.add_argument("--v176-report", type=Path, default=DEFAULT_V176_REPORT_PATH)
    parser.add_argument(
        "--v177-shard-plan",
        type=Path,
        default=DEFAULT_V177_SHARD_PLAN_OUTPUT_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--transition-dataset-output",
        type=Path,
        default=DEFAULT_TRANSITION_DATASET_OUTPUT_PATH,
    )
    parser.add_argument("--seed-include", type=int, default=None)
    parser.add_argument("--priority-include", type=int, action="append", default=[])
    parser.add_argument("--max-plan-rows", type=int, default=None)
    parser.add_argument("--ticks", type=int, default=120)
    parser.add_argument(
        "--no-verify-replay",
        action="store_true",
        help="Skip the second deterministic replay pass for each forced action.",
    )
    parser.add_argument(
        "--expected-v176-exact-digest",
        default=EXPECTED_V176_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v176-classification",
        default=EXPECTED_V176_CLASSIFICATION,
    )
    parser.add_argument(
        "--expected-v177-shard-plan-digest",
        default=EXPECTED_V177_SHARD_PLAN_DIGEST,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v177_exact_branch_replay_expansion(
            v176_report_path=args.v176_report,
            v177_shard_plan_path=args.v177_shard_plan,
            output_path=args.output,
            transition_dataset_output_path=args.transition_dataset_output,
            expected_v176_exact_digest=args.expected_v176_exact_digest,
            expected_v176_classification=args.expected_v176_classification,
            expected_v177_shard_plan_digest=args.expected_v177_shard_plan_digest,
            seed_include=args.seed_include,
            priority_include=args.priority_include,
            max_plan_rows=args.max_plan_rows,
            ticks=args.ticks,
            verify_replay=not args.no_verify_replay,
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationV177ExactBranchReplayExpansionError,
    ) as exc:
        raise SystemExit(
            "failed to run v177 carrion survivor-continuation exact branch "
            f"replay expansion: {exc}"
        ) from exc
    _print_summary(report, args.output, args.transition_dataset_output)


def _print_summary(
    report: dict[str, object],
    output: Path,
    transition_dataset_output: Path,
) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_validation"))
    selection = _payload(report.get("selection"))
    materialization = _payload(report.get("branch_materialization"))
    metrics = _payload(report.get("metrics"))
    leakage = _payload(report.get("leakage_scan"))
    row_schema = _payload(report.get("row_schema_validation"))
    dataset = _payload(report.get("dataset"))
    route = _payload(report.get("route_recommendation"))
    print(
        "carrion_survivor_continuation_v177_exact_branch_replay_expansion_report="
        f"{output}"
    )
    print(f"transition_dataset_output={transition_dataset_output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_validation_passed={source.get('passed')}")
    print(f"selected_branch_point_count={selection.get('selected_branch_point_count')}")
    print(
        "materialized_branch_point_count="
        f"{materialization.get('materialized_branch_point_count')}"
    )
    print(f"transition_row_count={metrics.get('transition_row_count')}")
    print(f"forced_action_used_count={metrics.get('forced_action_used_count')}")
    print(f"replay_verified_row_count={metrics.get('replay_verified_row_count')}")
    print(f"all_replays_verified={metrics.get('all_replays_verified')}")
    print(
        "rows_with_next_public_observation="
        f"{metrics.get('rows_with_next_public_observation')}"
    )
    print(
        "rows_with_previous_same_agent_public_context="
        f"{metrics.get('rows_with_previous_same_agent_public_context')}"
    )
    print(f"leakage_scan_passed={leakage.get('passed')}")
    print(f"row_schema_validation_passed={row_schema.get('passed')}")
    print(f"dataset_digest={dataset.get('dataset_digest')}")
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"shadow_eval_ran={report.get('shadow_eval_ran')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"exact_digest={report.get('exact_digest')}")


def _payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
