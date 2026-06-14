from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V195_ARTIFACT_PATH,
    DEFAULT_V198_REPORT_PATH,
    DEFAULT_V199_REPORT_PATH,
    DEFAULT_V200_REPORT_PATH,
    EXPECTED_V195_ARTIFACT_DIGEST,
    EXPECTED_V198_REPORT_EXACT_DIGEST,
    EXPECTED_V199_REPORT_EXACT_DIGEST,
    EXPECTED_V200_REPORT_EXACT_DIGEST,
    run_carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v201 no-training high-specificity action-complete source "
            "contract audit. The command validates v200/v199/v198 lineage, "
            "audits existing v199 candidate-key rows against the frozen v195 "
            "artifact utility table, and writes a closed-lifecycle report."
        )
    )
    parser.add_argument("--v198-report", type=Path, default=DEFAULT_V198_REPORT_PATH)
    parser.add_argument("--v199-report", type=Path, default=DEFAULT_V199_REPORT_PATH)
    parser.add_argument("--v200-report", type=Path, default=DEFAULT_V200_REPORT_PATH)
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
        "--expected-v200-report-exact-digest",
        default=EXPECTED_V200_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v195-artifact-digest",
        default=EXPECTED_V195_ARTIFACT_DIGEST,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit(
            v198_report_path=args.v198_report,
            v199_report_path=args.v199_report,
            v200_report_path=args.v200_report,
            v195_artifact_path=args.v195_artifact,
            output_path=args.output,
            expected_v198_report_exact_digest=args.expected_v198_report_exact_digest,
            expected_v199_report_exact_digest=args.expected_v199_report_exact_digest,
            expected_v200_report_exact_digest=args.expected_v200_report_exact_digest,
            expected_v195_artifact_digest=args.expected_v195_artifact_digest,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v201 high-specificity action-complete source contract "
            f"audit: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: Mapping[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_pin_validation"))
    facts = _payload(report.get("lineage_fact_assessment"))
    audit = _payload(report.get("source_contract_audit"))
    counts = _payload(audit.get("counts"))
    gaps = _payload(audit.get("gap_classification"))
    route = _payload(report.get("route_decision"))
    print(
        "carrion_survivor_continuation_v201_high_specificity_action_complete_source_contract_audit="
        f"{output}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"source_pin_validation_passed={source.get('passed')}")
    print(f"lineage_facts_passed={facts.get('passed')}")
    print(f"primary_blocker={audit.get('primary_blocker')}")
    print(
        "total_missing_current_valid_action_demands="
        f"{counts.get('total_missing_current_valid_action_demands')}"
    )
    print(
        "same_high_specific_action_present_for_missing_demands_count="
        f"{counts.get('same_high_specific_action_present_for_missing_demands_count')}"
    )
    print(
        "lower_specificity_only_missing_action_count="
        f"{counts.get('lower_specificity_only_missing_action_count')}"
    )
    print(
        "lower_specificity_only_missing_action_share="
        f"{gaps.get('lower_specificity_only_missing_action_share')}"
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
