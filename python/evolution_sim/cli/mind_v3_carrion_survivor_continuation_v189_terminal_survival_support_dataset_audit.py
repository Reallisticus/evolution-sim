from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v189_terminal_survival_support_dataset_audit import (
    DEFAULT_MAX_DOMINANT_REQUESTED_ACTION_SHARE,
    DEFAULT_EXPECTED_SELECTED_SUPPORT,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_REQUIRED_TARGET_SEED_COVERAGE_FRACTION,
    DEFAULT_V188_REPORT_PATH,
    EXPECTED_V188_REPORT_EXACT_DIGEST,
    EXPECTED_V188_REQUIRED_ROUTE,
    run_carrion_survivor_continuation_v189_terminal_survival_support_dataset_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v189 terminal-survival support dataset "
            "audit. The command validates the pinned v188 exact digest and "
            "route, audits selected path-backed legal support trajectories, "
            "records historical dedupe constraints, and fail-closes slice-3 "
            "training when legal seed coverage or action diversity is "
            "insufficient. It does not train, consume slice 3, create a "
            "runtime artifact, change runtime action selection, relax gates, "
            "or authorize promotion."
        )
    )
    parser.add_argument("--v188-report", type=Path, default=DEFAULT_V188_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-v188-report-exact-digest",
        default=EXPECTED_V188_REPORT_EXACT_DIGEST,
    )
    parser.add_argument("--required-v188-route", default=EXPECTED_V188_REQUIRED_ROUTE)
    parser.add_argument(
        "--max-dominant-requested-action-share",
        type=float,
        default=DEFAULT_MAX_DOMINANT_REQUESTED_ACTION_SHARE,
    )
    parser.add_argument(
        "--required-target-seed-coverage-fraction",
        type=float,
        default=DEFAULT_REQUIRED_TARGET_SEED_COVERAGE_FRACTION,
    )
    parser.add_argument(
        "--skip-canonical-selected-support-check",
        action="store_true",
        help=(
            "Skip the canonical v188 selected-support identity check. This is "
            "for local synthetic diagnostics only."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = (
            run_carrion_survivor_continuation_v189_terminal_survival_support_dataset_audit(
                v188_report_path=args.v188_report,
                output_path=args.output,
                expected_v188_report_exact_digest=(
                    args.expected_v188_report_exact_digest
                ),
                required_v188_route=args.required_v188_route,
                max_dominant_requested_action_share=(
                    args.max_dominant_requested_action_share
                ),
                required_target_seed_coverage_fraction=(
                    args.required_target_seed_coverage_fraction
                ),
                expected_selected_support=(
                    None
                    if args.skip_canonical_selected_support_check
                    else DEFAULT_EXPECTED_SELECTED_SUPPORT
                ),
            )
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v189 terminal survival support dataset audit: "
            f"{exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: Mapping[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_validation"))
    coverage = _payload(report.get("support_coverage_audit"))
    aggregate = _payload(report.get("aggregate_attempted_continuation_audit"))
    selected = _payload(report.get("selected_support_trajectory_audit"))
    leakage = _payload(report.get("trainable_leakage_audit"))
    route = _payload(report.get("route_decision"))
    print(f"carrion_survivor_continuation_v189_dataset_audit={output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_validation_passed={source.get('passed')}")
    print(
        "v188_report_exact_digest="
        f"{source.get('observed_v188_report_exact_digest')}"
    )
    print(f"v188_route={source.get('observed_v188_route')}")
    print(
        "legal_positive_target_seed_count="
        f"{coverage.get('legal_positive_target_seed_count')}"
    )
    print(f"target_seed_count={coverage.get('target_seed_count')}")
    print(
        "legal_support_coverage_fraction="
        f"{coverage.get('legal_support_coverage_fraction')}"
    )
    print(
        "aggregate_unsupported_resolved_action_count="
        f"{aggregate.get('unsupported_resolved_action_count')}"
    )
    print(
        "aggregate_attempted_continuations_rejected_as_support="
        f"{aggregate.get('aggregate_attempted_continuations_rejected_as_support')}"
    )
    print(f"selected_support_passed={selected.get('passed')}")
    print(f"selected_support_run_count={selected.get('support_run_count')}")
    print(
        "dominant_requested_action="
        f"{selected.get('dominant_requested_action')}"
    )
    print(
        "dominant_requested_action_share="
        f"{selected.get('dominant_requested_action_share')}"
    )
    print(f"trainable_leakage_passed={leakage.get('passed')}")
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(f"slice_3_training_authorized={route.get('slice_3_training_authorized')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"training_artifact_created={report.get('training_artifact_created')}")
    print(f"slice_3_training_consumed={report.get('slice_3_training_consumed')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(
        "runtime_action_selection_changed="
        f"{report.get('runtime_action_selection_changed')}"
    )
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"gate_relaxation_allowed={report.get('gate_relaxation_allowed')}")
    print(f"exact_digest={report.get('exact_digest')}")


def _payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
