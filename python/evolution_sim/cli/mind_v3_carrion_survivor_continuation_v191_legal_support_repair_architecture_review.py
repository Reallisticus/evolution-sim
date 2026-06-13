from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v191_legal_support_repair_architecture_review import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V190_REPORT_PATH,
    EXPECTED_V190_REPORT_EXACT_DIGEST,
    EXPECTED_V190_REQUIRED_ROUTE,
    run_carrion_survivor_continuation_v191_legal_support_repair_architecture_review,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run diagnostics-only v191 legal-support repair / architecture "
            "review for the v190 unsupported resolved-action blocker."
        )
    )
    parser.add_argument("--v190-report", type=Path, default=DEFAULT_V190_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-v190-report-exact-digest",
        default=EXPECTED_V190_REPORT_EXACT_DIGEST,
    )
    parser.add_argument("--required-v190-route", default=EXPECTED_V190_REQUIRED_ROUTE)
    args = parser.parse_args()

    report = (
        run_carrion_survivor_continuation_v191_legal_support_repair_architecture_review(
            v190_report_path=args.v190_report,
            output_path=args.output,
            expected_v190_report_exact_digest=args.expected_v190_report_exact_digest,
            required_v190_route=args.required_v190_route,
        )
    )

    source = report["source_validation"]
    classification = report["unsupported_resolved_action_classification"]
    root = classification["root_cause_assessment"]
    route = report["route_decision"]
    print(f"carrion_survivor_continuation_v191={args.output}")
    print(f"exact_digest={report['exact_digest']}")
    print(f"source_validation_passed={source.get('passed')}")
    print(
        "unsupported_resolved_action_count="
        f"{classification.get('observed_unsupported_resolved_action_count')}"
    )
    print(
        "unsupported_requested_action_count="
        f"{classification.get('v190_reported_unsupported_requested_action_count')}"
    )
    print(f"root_cause={root.get('primary')}")
    print(
        "event_count_matches_v190_report="
        f"{classification.get('event_count_matches_v190_report')}"
    )
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(f"slice_3_training_authorized={route.get('slice_3_training_authorized')}")
    print(f"training_ran={report['training_ran']}")
    print(f"support_generation_ran={report['support_generation_ran']}")
    print(f"runtime_action_selection_changed={report['runtime_action_selection_changed']}")
    print(f"promotion_authorized={report['promotion_authorized']}")


if __name__ == "__main__":
    main()
