from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v203_public_masked_model_capacity_harness_scaffold import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V202_REPORT_PATH,
    EXPECTED_V202_REPORT_EXACT_DIGEST,
    run_carrion_survivor_continuation_v203_public_masked_model_capacity_harness_scaffold,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v203 no-training public masked recurrent-IPPO harness "
            "scaffold audit. The command validates the pinned v202 digest and "
            "route, executes in-memory contract probes, and leaves training, "
            "slice 4, runtime behavior, and promotion closed."
        )
    )
    parser.add_argument("--v202-report", type=Path, default=DEFAULT_V202_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-v202-report-exact-digest",
        default=EXPECTED_V202_REPORT_EXACT_DIGEST,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v203_public_masked_model_capacity_harness_scaffold(
            v202_report_path=args.v202_report,
            output_path=args.output,
            expected_v202_report_exact_digest=(args.expected_v202_report_exact_digest),
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"failed to run v203 public masked recurrent-IPPO harness scaffold: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: Mapping[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_pin_validation"))
    scaffold = _payload(report.get("public_recurrent_ippo_scaffold_surface_audit"))
    route = _payload(report.get("route_decision"))
    budget = _payload(report.get("budget_state"))
    print(
        "carrion_survivor_continuation_v203_public_masked_model_capacity_harness_scaffold="
        f"{output}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"v202_source_pin_validation_passed={source.get('passed')}")
    print(f"public_recurrent_ippo_scaffold_passed={scaffold.get('passed')}")
    print(f"scaffold_check_count={len(_payload(scaffold.get('checks')))}")
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(f"training_slices_consumed={budget.get('training_slices_consumed')}")
    print(f"training_slice_budget={budget.get('training_slice_budget')}")
    print(f"training_started={report.get('training_started')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"training_slice_4_consumed={report.get('training_slice_4_consumed')}")
    print(f"training_artifact_created={report.get('training_artifact_created')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(
        "runtime_action_selection_changed="
        f"{report.get('runtime_action_selection_changed')}"
    )
    print(f"default_policy_changed={report.get('default_policy_changed')}")
    print(f"dataset_created={report.get('dataset_created')}")
    print(f"dataset_mutated={report.get('dataset_mutated')}")
    print(f"gate_relaxed={report.get('gate_relaxed')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"exact_digest={report.get('exact_digest')}")


def _payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
