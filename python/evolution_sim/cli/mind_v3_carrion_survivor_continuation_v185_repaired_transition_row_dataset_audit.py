from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind import (
    carrion_survivor_continuation_v178_transition_row_dataset_audit as v178,
)
from evolution_sim.mind.carrion_survivor_continuation_v185_v183_target_resolution_repair import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V185_REPORT_PATH,
    DEFAULT_REPAIRED_DATASET_AUDIT_OUTPUT_PATH,
    DEFAULT_REPAIRED_TRANSITION_DATASET_OUTPUT_PATH,
    run_carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run diagnostics-only v185 repaired-dataset audit using the v178 "
            "transition-row contract. The command may authorize only a future "
            "explicit slice-2 opt-in training route; it does not train, create "
            "runtime artifacts, change runtime action selection, relax gates, or "
            "promote."
        )
    )
    parser.add_argument(
        "--v185-report",
        "--source-report",
        dest="v185_report",
        type=Path,
        default=DEFAULT_V185_REPORT_PATH,
    )
    parser.add_argument(
        "--transition-dataset",
        type=Path,
        default=DEFAULT_REPAIRED_TRANSITION_DATASET_OUTPUT_PATH,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_REPAIRED_DATASET_AUDIT_OUTPUT_PATH,
    )
    parser.add_argument(
        "--expected-v185-report-exact-digest",
        "--expected-source-report-exact-digest",
        dest="expected_v185_report_exact_digest",
        default=None,
        help="Required for future slice-2 route authorization.",
    )
    parser.add_argument(
        "--expected-dataset-digest",
        default=None,
        help="Required for future slice-2 route authorization.",
    )
    parser.add_argument("--min-row-count", type=int, default=v178.DEFAULT_MIN_ROW_COUNT)
    parser.add_argument("--min-seed-count", type=int, default=v178.DEFAULT_MIN_SEED_COUNT)
    parser.add_argument(
        "--min-branch-count",
        type=int,
        default=v178.DEFAULT_MIN_BRANCH_COUNT,
    )
    parser.add_argument(
        "--min-forced-action-count",
        type=int,
        default=v178.DEFAULT_MIN_FORCED_ACTION_COUNT,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = (
            run_carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit(
                v185_report_path=args.v185_report,
                transition_dataset_path=args.transition_dataset,
                output_path=args.output,
                expected_v185_report_exact_digest=(
                    args.expected_v185_report_exact_digest
                ),
                expected_dataset_digest=args.expected_dataset_digest,
                min_row_count=args.min_row_count,
                min_seed_count=args.min_seed_count,
                min_branch_count=args.min_branch_count,
                min_forced_action_count=args.min_forced_action_count,
            )
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v185 carrion survivor-continuation repaired "
            f"transition-row dataset audit: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: Mapping[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_validation"))
    schema = _payload(report.get("row_schema_validation"))
    leakage = _payload(report.get("leakage_scan"))
    value_leakage = _payload(report.get("value_leakage_scan"))
    identity = _payload(report.get("identity_audit"))
    action_mask = _payload(report.get("action_mask_audit"))
    observation = _payload(report.get("observation_audit"))
    target = _payload(report.get("target_audit"))
    support = _payload(report.get("support_readiness"))
    default_support = _payload(report.get("default_support_readiness"))
    authorization = _payload(report.get("training_authorization"))
    route = _payload(report.get("route_recommendation"))
    contract = _payload(report.get("contract"))
    dataset = _payload(report.get("dataset"))
    print(f"carrion_survivor_continuation_v185_repaired_transition_row_dataset_audit={output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_producer={source.get('source_producer')}")
    print(f"source_validation_passed={source.get('passed')}")
    print(f"row_schema_validation_passed={schema.get('passed')}")
    print(f"leakage_scan_passed={leakage.get('passed')}")
    print(f"value_leakage_scan_passed={value_leakage.get('passed')}")
    print(f"identity_audit_passed={identity.get('passed')}")
    print(f"action_mask_audit_passed={action_mask.get('passed')}")
    print(f"observation_audit_passed={observation.get('passed')}")
    print(f"target_audit_passed={target.get('passed')}")
    print(f"target_audit_failure_count={target.get('failure_count')}")
    print(f"support_minimums_met={support.get('passed')}")
    print(f"default_support_minimums_met={default_support.get('passed')}")
    print(f"dataset_digest={dataset.get('dataset_digest')}")
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(
        "route_recommendation.slice_2_opt_in_training_route_authorized="
        f"{route.get('slice_2_opt_in_training_route_authorized')}"
    )
    print(
        "route_recommendation.first_opt_in_training_slice_authorized="
        f"{route.get('first_opt_in_training_slice_authorized')}"
    )
    print(
        "training_authorization.next_same_lane_opt_in_training_slice_authorized="
        f"{authorization.get('next_same_lane_opt_in_training_slice_authorized')}"
    )
    print(
        "contract.slice_2_opt_in_training_route_authorized="
        f"{contract.get('slice_2_opt_in_training_route_authorized')}"
    )
    print(f"training_ran={report.get('training_ran')}")
    print(f"training_artifact_created={report.get('training_artifact_created')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"slice_2_training_consumed={report.get('slice_2_training_consumed')}")
    print(f"exact_digest={report.get('exact_digest')}")


def _payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
