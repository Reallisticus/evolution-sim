from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v192_action_resolution_contract_repair import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V190_REPORT_PATH,
    DEFAULT_V191_REPORT_PATH,
    EXPECTED_V191_REPORT_EXACT_DIGEST,
    EXPECTED_V190_REPORT_EXACT_DIGEST,
    REQUIRED_V191_ROUTE,
    run_carrion_survivor_continuation_v192_action_resolution_contract_repair,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run diagnostics-only v192 action-resolution contract repair for "
            "the v191 same-tick movement occupancy-race blocker."
        )
    )
    parser.add_argument("--v191-report", type=Path, default=DEFAULT_V191_REPORT_PATH)
    parser.add_argument("--v190-report", type=Path, default=DEFAULT_V190_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-v191-report-exact-digest",
        default=EXPECTED_V191_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v190-report-exact-digest",
        default=EXPECTED_V190_REPORT_EXACT_DIGEST,
    )
    parser.add_argument("--required-v191-route", default=REQUIRED_V191_ROUTE)
    args = parser.parse_args()

    report = run_carrion_survivor_continuation_v192_action_resolution_contract_repair(
        v191_report_path=args.v191_report,
        v190_report_path=args.v190_report,
        output_path=args.output,
        expected_v191_report_exact_digest=args.expected_v191_report_exact_digest,
        expected_v190_report_exact_digest=args.expected_v190_report_exact_digest,
        required_v191_route=args.required_v191_route,
    )

    source = report["source_validation"]
    movement = report["movement_target_blocker_audit"]
    repair = report["repair_scope_decision"]
    route = report["route_decision"]
    print(f"carrion_survivor_continuation_v192={args.output}")
    print(f"exact_digest={report['exact_digest']}")
    print(f"source_validation_passed={source.get('passed')}")
    print(
        "repair_contract_event_count="
        f"{movement.get('repair_contract_event_count')}"
    )
    print(
        "same_tick_occupancy_race_count="
        f"{movement.get('blocker_classification_counts', {}).get('resolution_invalid_same_tick_occupancy_race')}"
    )
    print(f"selected_repair_type={repair.get('selected_repair_type')}")
    print(f"contract_repair_sufficient={repair.get('contract_repair_sufficient')}")
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(f"slice_3_training_authorized={route.get('slice_3_training_authorized')}")
    print(f"training_ran={report['training_ran']}")
    print(f"support_generation_ran={report['support_generation_ran']}")
    print(f"runtime_action_selection_changed={report['runtime_action_selection_changed']}")
    print(f"promotion_authorized={report['promotion_authorized']}")


if __name__ == "__main__":
    main()
