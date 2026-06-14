from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V195_ARTIFACT_PATH,
    DEFAULT_V198_REPORT_PATH,
    DEFAULT_V199_REPORT_PATH,
    EXPECTED_V195_ARTIFACT_DIGEST,
    EXPECTED_V198_REPORT_EXACT_DIGEST,
    EXPECTED_V199_CLASSIFICATION,
    EXPECTED_V199_REPORT_EXACT_DIGEST,
    EXPECTED_V199_ROUTE,
    run_carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v200 no-training high-specificity action-complete repair "
            "design report. The command validates the pinned v199 audit, derives "
            "the missing-action matrix from that report, and writes a closed "
            "lifecycle design route."
        )
    )
    parser.add_argument("--v198-report", type=Path, default=DEFAULT_V198_REPORT_PATH)
    parser.add_argument("--v199-report", type=Path, default=DEFAULT_V199_REPORT_PATH)
    parser.add_argument("--v195-artifact", type=Path, default=DEFAULT_V195_ARTIFACT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-v198-report-exact-digest",
        default=EXPECTED_V198_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v199-report-exact-digest",
        default=EXPECTED_V199_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v199-classification",
        default=EXPECTED_V199_CLASSIFICATION,
    )
    parser.add_argument("--expected-v199-route", default=EXPECTED_V199_ROUTE)
    parser.add_argument(
        "--expected-v195-artifact-digest",
        default=EXPECTED_V195_ARTIFACT_DIGEST,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design(
            v198_report_path=args.v198_report,
            v199_report_path=args.v199_report,
            v195_artifact_path=args.v195_artifact,
            output_path=args.output,
            expected_v198_report_exact_digest=args.expected_v198_report_exact_digest,
            expected_v199_report_exact_digest=args.expected_v199_report_exact_digest,
            expected_v199_classification=args.expected_v199_classification,
            expected_v199_route=args.expected_v199_route,
            expected_v195_artifact_digest=args.expected_v195_artifact_digest,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v200 high-specificity action-complete repair design: "
            f"{exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: Mapping[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_pin_validation"))
    facts = _payload(report.get("v199_fact_assessment"))
    route = _payload(report.get("route_decision"))
    design = _payload(report.get("action_complete_repair_design"))
    counts = _payload(facts.get("counts"))
    print(
        "carrion_survivor_continuation_v200_high_specificity_action_complete_contract_repair_design="
        f"{output}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"source_pin_validation_passed={source.get('passed')}")
    print(f"v199_report_exact_digest={source.get('observed_v199_report_exact_digest')}")
    print(f"v199_route={source.get('observed_v199_route')}")
    print(f"facts_consistent={facts.get('facts_consistent')}")
    print(f"primary_blocker={facts.get('primary_blocker')}")
    print(f"selected_repair_class={design.get('selected_repair_class')}")
    print(
        "high_specificity_candidate_evaluated_count="
        f"{counts.get('high_specificity_candidate_evaluated_count')}"
    )
    print(f"key_absent_count={counts.get('key_absent_count')}")
    print(f"key_present_count={counts.get('key_present_count')}")
    print(
        "present_but_action_incomplete_count="
        f"{counts.get('present_but_action_incomplete_count')}"
    )
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"slice_4_training_consumed={report.get('slice_4_training_consumed')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(
        "runtime_action_selection_changed="
        f"{report.get('runtime_action_selection_changed')}"
    )
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"gate_relaxation_allowed={report.get('gate_relaxation_allowed')}")
    print(f"support_generation_ran={report.get('support_generation_ran')}")
    print(f"support_expansion_ran={report.get('support_expansion_ran')}")
    print(f"exact_digest={report.get('exact_digest')}")


def _payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
