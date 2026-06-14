from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V198_REPORT_PATH,
    DEFAULT_V199_REPORT_PATH,
    DEFAULT_V200_REPORT_PATH,
    DEFAULT_V201_REPORT_PATH,
    EXPECTED_V198_REPORT_EXACT_DIGEST,
    EXPECTED_V199_REPORT_EXACT_DIGEST,
    EXPECTED_V200_REPORT_EXACT_DIGEST,
    EXPECTED_V201_REPORT_EXACT_DIGEST,
    run_carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v202 no-training public masked model-capacity harness "
            "contract report. The command validates v201/v200/v199/v198 "
            "lineage, validates v197 through v198 source pins, and writes a "
            "closed-lifecycle contract report for a future harness scaffold."
        )
    )
    parser.add_argument("--v198-report", type=Path, default=DEFAULT_V198_REPORT_PATH)
    parser.add_argument("--v199-report", type=Path, default=DEFAULT_V199_REPORT_PATH)
    parser.add_argument("--v200-report", type=Path, default=DEFAULT_V200_REPORT_PATH)
    parser.add_argument("--v201-report", type=Path, default=DEFAULT_V201_REPORT_PATH)
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
        "--expected-v201-report-exact-digest",
        default=EXPECTED_V201_REPORT_EXACT_DIGEST,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract(
            v198_report_path=args.v198_report,
            v199_report_path=args.v199_report,
            v200_report_path=args.v200_report,
            v201_report_path=args.v201_report,
            output_path=args.output,
            expected_v198_report_exact_digest=args.expected_v198_report_exact_digest,
            expected_v199_report_exact_digest=args.expected_v199_report_exact_digest,
            expected_v200_report_exact_digest=args.expected_v200_report_exact_digest,
            expected_v201_report_exact_digest=args.expected_v201_report_exact_digest,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v202 public masked model-capacity harness contract: "
            f"{exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: Mapping[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_pin_validation"))
    facts = _payload(report.get("v201_fact_assessment"))
    contract = _payload(report.get("public_masked_model_capacity_harness_contract"))
    route = _payload(report.get("route_decision"))
    budget = _payload(report.get("budget_state"))
    print(
        "carrion_survivor_continuation_v202_public_masked_model_capacity_harness_contract="
        f"{output}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"source_pin_validation_passed={source.get('passed')}")
    print(f"v201_facts_passed={facts.get('passed')}")
    print(
        "public_contract_complete="
        f"{contract.get('contract_complete_for_scaffold')}"
    )
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(f"training_slices_consumed={budget.get('training_slices_consumed')}")
    print(f"training_slice_budget={budget.get('training_slice_budget')}")
    print(f"training_started={report.get('training_started')}")
    print(
        "training_slice_4_consumed="
        f"{report.get('training_slice_4_consumed')}"
    )
    print(f"training_artifact_created={report.get('training_artifact_created')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(
        "runtime_action_selection_changed="
        f"{report.get('runtime_action_selection_changed')}"
    )
    print(f"default_policy_changed={report.get('default_policy_changed')}")
    print(f"support_generated={report.get('support_generated')}")
    print(f"support_expanded={report.get('support_expanded')}")
    print(f"dataset_mutated={report.get('dataset_mutated')}")
    print(f"gate_relaxed={report.get('gate_relaxed')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"exact_digest={report.get('exact_digest')}")


def _payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
