from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training import (
    DEFAULT_ARTIFACT_OUTPUT_PATH,
    DEFAULT_AUTHORIZATION_REPORT_PATH,
    DEFAULT_BACKUP_DOC_PATHS,
    DEFAULT_BROAD_SEEDS,
    DEFAULT_CARRION_FIXTURE_SEEDS,
    DEFAULT_COMPACT_SUPPORT_DATASET_PATH,
    DEFAULT_FEATURE_KEY_LIMIT,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TICKS,
    EXPECTED_SELECTED_SAME_TICK_OCCUPANCY_DRIFT_COUNT,
    EXPECTED_UNEXPECTED_RESOLUTION_INVALID_COUNT,
    EXPECTED_UNSUPPORTED_REQUESTED_ACTION_COUNT,
    EXPECTED_V192_REPORT_EXACT_DIGEST,
    EXPECTED_V193_REPORT_EXACT_DIGEST,
    EXPECTED_V194_COMPACT_DATASET_DIGEST,
    EXPECTED_V194_COMPACT_DATASET_ROW_COUNT,
    EXPECTED_V194_REPORT_EXACT_DIGEST,
    EXPECTED_V194_ROUTE,
    FAILURE_ROUTE,
    SUCCESS_ROUTE,
    run_carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v195 explicit opt-in repaired-contract terminal-survival "
            "support training slice 3 from the v194 compact support dataset. "
            "The command fails closed unless the pinned v194 report and "
            "dataset validate exactly. It writes a training artifact and runs "
            "shadow evaluation only; it does not integrate runtime behavior, "
            "change runtime action selection, relax gates, or authorize promotion."
        )
    )
    parser.add_argument(
        "--authorization-report",
        type=Path,
        default=DEFAULT_AUTHORIZATION_REPORT_PATH,
    )
    parser.add_argument(
        "--compact-support-dataset",
        type=Path,
        default=DEFAULT_COMPACT_SUPPORT_DATASET_PATH,
    )
    parser.add_argument("--artifact-output", type=Path, default=DEFAULT_ARTIFACT_OUTPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-authorization-report-exact-digest",
        default=EXPECTED_V194_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-dataset-digest",
        default=EXPECTED_V194_COMPACT_DATASET_DIGEST,
    )
    parser.add_argument("--expected-training-route", default=EXPECTED_V194_ROUTE)
    parser.add_argument(
        "--expected-v193-report-exact-digest",
        default=EXPECTED_V193_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v192-report-exact-digest",
        default=EXPECTED_V192_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-dataset-row-count",
        type=int,
        default=EXPECTED_V194_COMPACT_DATASET_ROW_COUNT,
    )
    parser.add_argument(
        "--expected-unsupported-requested-action-count",
        type=int,
        default=EXPECTED_UNSUPPORTED_REQUESTED_ACTION_COUNT,
    )
    parser.add_argument(
        "--expected-selected-same-tick-occupancy-drift-count",
        type=int,
        default=EXPECTED_SELECTED_SAME_TICK_OCCUPANCY_DRIFT_COUNT,
    )
    parser.add_argument(
        "--expected-unexpected-resolution-invalid-count",
        type=int,
        default=EXPECTED_UNEXPECTED_RESOLUTION_INVALID_COUNT,
    )
    parser.add_argument("--feature-key-limit", type=int, default=DEFAULT_FEATURE_KEY_LIMIT)
    parser.add_argument("--broad-seeds", default=_csv(DEFAULT_BROAD_SEEDS))
    parser.add_argument(
        "--carrion-fixture-seeds",
        default=_csv(DEFAULT_CARRION_FIXTURE_SEEDS),
    )
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument(
        "--backup-doc",
        action="append",
        type=Path,
        dest="backup_docs",
        help=(
            "Doc path that records v194 backup metadata. May be repeated; "
            "defaults to durable repo docs."
        ),
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Train and write the artifact without running the shadow eval.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training(
            authorization_report_path=args.authorization_report,
            compact_support_dataset_path=args.compact_support_dataset,
            artifact_output_path=args.artifact_output,
            output_path=args.output,
            expected_authorization_report_exact_digest=(
                args.expected_authorization_report_exact_digest
            ),
            expected_dataset_digest=args.expected_dataset_digest,
            expected_training_route=args.expected_training_route,
            expected_v193_report_exact_digest=args.expected_v193_report_exact_digest,
            expected_v192_report_exact_digest=args.expected_v192_report_exact_digest,
            expected_dataset_row_count=args.expected_dataset_row_count,
            expected_unsupported_requested_action_count=(
                args.expected_unsupported_requested_action_count
            ),
            expected_selected_same_tick_occupancy_drift_count=(
                args.expected_selected_same_tick_occupancy_drift_count
            ),
            expected_unexpected_resolution_invalid_count=(
                args.expected_unexpected_resolution_invalid_count
            ),
            backup_doc_paths=tuple(args.backup_docs or DEFAULT_BACKUP_DOC_PATHS),
            feature_key_limit=args.feature_key_limit,
            run_evaluation=not args.skip_evaluation,
            broad_seeds=_parse_seed_csv(args.broad_seeds),
            carrion_fixture_seeds=_parse_seed_csv(args.carrion_fixture_seeds),
            ticks=args.ticks,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v195 repaired-contract terminal-survival support "
            f"training slice 3: {exc}"
        ) from exc
    _print_summary(report, args.output, args.artifact_output)


def _print_summary(
    report: Mapping[str, object],
    output: Path,
    artifact_output: Path,
) -> None:
    classification = _payload(report.get("classification"))
    auth = _payload(report.get("authorization_report_validation"))
    source = _payload(report.get("source_validation"))
    artifact = _payload(report.get("artifact"))
    training = _payload(report.get("training"))
    acceptance = _payload(report.get("acceptance"))
    controlled = _payload(acceptance.get("controlled_fixture"))
    route = _payload(report.get("route_decision"))
    budget = _payload(report.get("training_slice_budget"))
    dataset = _payload(report.get("dataset"))
    print(
        "carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training="
        f"{output}"
    )
    print(f"artifact_output={artifact_output}")
    print(f"classification={classification.get('primary')}")
    print(f"authorization_report_validation_passed={auth.get('passed')}")
    print(f"authorization_report_exact_digest={auth.get('observed_exact_digest')}")
    print(f"authorization_report_route={auth.get('observed_training_route')}")
    print(f"required_route={EXPECTED_V194_ROUTE}")
    print(f"source_validation_passed={source.get('passed')}")
    print(f"dataset_digest={dataset.get('dataset_digest')}")
    print(f"dataset_row_count={dataset.get('row_count')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"training_artifact_created={report.get('training_artifact_created')}")
    print(f"training_artifact_digest={artifact.get('digest')}")
    print(f"training_row_count={training.get('training_row_count')}")
    print(f"training_slice_index={training.get('training_slice_index')}")
    print(f"slice_3_training_consumed={report.get('slice_3_training_consumed')}")
    print(f"current_slices_consumed={budget.get('current_slices_consumed')}")
    print(f"shadow_eval_ran={report.get('shadow_eval_ran')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"gate_relaxation_allowed={report.get('gate_relaxation_allowed')}")
    print(f"acceptance_passed={acceptance.get('passed')}")
    print(
        "controlled_fixture_total_terminal_alive_agents="
        f"{controlled.get('total_terminal_alive_agents')}"
    )
    print(
        "dominant_requested_action_share="
        f"{acceptance.get('dominant_requested_action_share')}"
    )
    print(
        "heuristic_action_source_count="
        f"{acceptance.get('heuristic_action_source_count')}"
    )
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(f"success_route={SUCCESS_ROUTE}")
    print(f"failure_route={FAILURE_ROUTE}")
    print(f"acceptance_blockers={acceptance.get('blockers')}")
    print(f"exact_digest={report.get('exact_digest')}")


def _parse_seed_csv(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("seed list must not be empty")
    return seeds


def _csv(values: Sequence[int]) -> str:
    return ",".join(str(value) for value in values)


def _payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
