from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind import (
    carrion_survivor_continuation_v178_transition_row_dataset_audit as v178,
)
from evolution_sim.mind.carrion_survivor_continuation_v185_v183_target_resolution_repair import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_REPAIRED_TRANSITION_DATASET_OUTPUT_PATH,
    DEFAULT_V183_TRANSITION_DATASET_PATH,
    run_carrion_survivor_continuation_v185_v183_target_resolution_repair,
)
from evolution_sim.mind.carrion_survivor_continuation_v184_v183_transition_row_dataset_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V184_REPORT_PATH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run diagnostics-only v185 repair for the v183 target-resolution "
            "blocker. It strict-filters invalid target-resolution rows and "
            "writes a repaired compact transition-row dataset; it does not train, "
            "create runtime artifacts, change runtime action selection, relax "
            "gates, or promote."
        )
    )
    parser.add_argument("--v184-report", type=Path, default=DEFAULT_V184_REPORT_PATH)
    parser.add_argument(
        "--v183-transition-dataset",
        "--transition-dataset",
        dest="v183_transition_dataset",
        type=Path,
        default=DEFAULT_V183_TRANSITION_DATASET_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--repaired-transition-dataset-output",
        type=Path,
        default=DEFAULT_REPAIRED_TRANSITION_DATASET_OUTPUT_PATH,
    )
    parser.add_argument(
        "--expected-v184-report-exact-digest",
        default=v178.EXPECTED_V184_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v183-report-exact-digest",
        default=v178.EXPECTED_V183_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v183-dataset-digest",
        default=v178.EXPECTED_V183_DATASET_DIGEST,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v185_v183_target_resolution_repair(
            v184_report_path=args.v184_report,
            v183_transition_dataset_path=args.v183_transition_dataset,
            output_path=args.output,
            repaired_transition_dataset_output_path=(
                args.repaired_transition_dataset_output
            ),
            expected_v184_report_exact_digest=args.expected_v184_report_exact_digest,
            expected_v183_report_exact_digest=args.expected_v183_report_exact_digest,
            expected_v183_dataset_digest=args.expected_v183_dataset_digest,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v185 carrion survivor-continuation v183 target-"
            f"resolution repair: {exc}"
        ) from exc
    _print_summary(
        report,
        output=args.output,
        dataset_output=args.repaired_transition_dataset_output,
    )


def _print_summary(
    report: Mapping[str, object],
    *,
    output: Path,
    dataset_output: Path,
) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_validation"))
    repair = _payload(report.get("repair_validation"))
    support = _payload(report.get("support_summary"))
    support_observed = _payload(support.get("observed"))
    dataset = _payload(report.get("dataset"))
    route = _payload(report.get("route_recommendation"))
    print(f"carrion_survivor_continuation_v185_v183_target_resolution_repair={output}")
    print(f"repaired_transition_dataset_output={dataset_output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_validation_passed={source.get('passed')}")
    print(f"repair_validation_passed={repair.get('passed')}")
    print(f"strict_filter_used={repair.get('strict_filter_used')}")
    print(f"backfill_used={repair.get('backfill_used')}")
    print(f"input_row_count={repair.get('input_row_count')}")
    print(f"invalid_input_row_count={repair.get('invalid_input_row_count')}")
    print(f"repaired_row_count={repair.get('repaired_row_count')}")
    print(
        "removed_or_replaced_invalid_row_count="
        f"{repair.get('removed_or_replaced_invalid_row_count')}"
    )
    print(f"support_row_count={support_observed.get('row_count')}")
    print(f"support_seed_count={support_observed.get('seed_count')}")
    print(f"support_branch_count={support_observed.get('branch_count')}")
    print(f"support_forced_action_count={support_observed.get('forced_action_count')}")
    print(f"v178_default_support_thresholds_met={support.get('passed')}")
    print(f"dataset_digest={dataset.get('dataset_digest')}")
    print(f"source_dataset_digest={dataset.get('source_dataset_digest')}")
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(f"slice_2_training_authorized={route.get('slice_2_training_authorized')}")
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
