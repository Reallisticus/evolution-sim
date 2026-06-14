from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v180_transition_row_policy_training import (
    DEFAULT_BROAD_SEEDS,
    DEFAULT_CARRION_FIXTURE_SEEDS,
    DEFAULT_TICKS,
)
from evolution_sim.mind.carrion_survivor_continuation_v197_coverage_abstention_repair_design import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V195_ARTIFACT_PATH,
    DEFAULT_V195_REPORT_PATH,
    DEFAULT_V196_REPORT_PATH,
    EXPECTED_V195_ARTIFACT_DIGEST,
    EXPECTED_V195_REPORT_EXACT_DIGEST,
    EXPECTED_V196_MECHANISM,
    EXPECTED_V196_REPORT_EXACT_DIGEST,
    EXPECTED_V196_ROUTE,
    run_carrion_survivor_continuation_v197_coverage_abstention_repair_design,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v197 no-training coverage/abstention repair design for the "
            "v195/v196 low-specificity feature collapse. The command validates "
            "pinned v196/v195 source facts, runs a v197 source-key specificity "
            "gate diagnostic replay unless skipped, and recommends exactly one "
            "closed or future route. It does not train, consume slice 4, create "
            "a runtime artifact, change default runtime action selection, relax "
            "gates, or authorize promotion."
        )
    )
    parser.add_argument("--v196-report", type=Path, default=DEFAULT_V196_REPORT_PATH)
    parser.add_argument("--v195-report", type=Path, default=DEFAULT_V195_REPORT_PATH)
    parser.add_argument("--v195-artifact", type=Path, default=DEFAULT_V195_ARTIFACT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-v196-report-exact-digest",
        default=EXPECTED_V196_REPORT_EXACT_DIGEST,
    )
    parser.add_argument("--expected-v196-route", default=EXPECTED_V196_ROUTE)
    parser.add_argument("--expected-v196-mechanism", default=EXPECTED_V196_MECHANISM)
    parser.add_argument(
        "--expected-v195-report-exact-digest",
        default=EXPECTED_V195_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v195-artifact-digest",
        default=EXPECTED_V195_ARTIFACT_DIGEST,
    )
    parser.add_argument("--broad-seeds", default=_csv(DEFAULT_BROAD_SEEDS))
    parser.add_argument(
        "--carrion-fixture-seeds",
        default=_csv(DEFAULT_CARRION_FIXTURE_SEEDS),
    )
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument(
        "--skip-diagnostic-replay",
        action="store_true",
        help=(
            "Skip the v197 specificity-gate replay. This is for source-pin "
            "tests only; the default v197 run should not use it."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v197_coverage_abstention_repair_design(
            v196_report_path=args.v196_report,
            v195_report_path=args.v195_report,
            v195_artifact_path=args.v195_artifact,
            output_path=args.output,
            expected_v196_report_exact_digest=args.expected_v196_report_exact_digest,
            expected_v196_route=args.expected_v196_route,
            expected_v196_mechanism=args.expected_v196_mechanism,
            expected_v195_report_exact_digest=args.expected_v195_report_exact_digest,
            expected_v195_artifact_digest=args.expected_v195_artifact_digest,
            run_diagnostic_replay=not args.skip_diagnostic_replay,
            broad_seeds=_parse_seed_csv(args.broad_seeds),
            carrion_fixture_seeds=_parse_seed_csv(args.carrion_fixture_seeds),
            ticks=args.ticks,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v197 coverage/abstention repair design: " f"{exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: Mapping[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_pin_validation"))
    facts = _payload(report.get("v195_failure_facts"))
    after = _payload(report.get("after_specificity_gate"))
    combined = _payload(after.get("combined"))
    comparison = _payload(report.get("specificity_gate_comparison"))
    route = _payload(report.get("route_decision"))
    print(f"carrion_survivor_continuation_v197_coverage_abstention_repair={output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_pin_validation_passed={source.get('passed')}")
    print(f"v196_report_exact_digest={source.get('observed_v196_report_exact_digest')}")
    print(f"v196_route={source.get('observed_v196_route')}")
    print(f"v196_mechanism={source.get('observed_v196_mechanism')}")
    print(f"v195_report_exact_digest={source.get('observed_v195_report_exact_digest')}")
    print(f"v195_artifact_digest={source.get('observed_v195_artifact_digest')}")
    print(f"slice_3_consumed_by_v195={facts.get('v195_training_slice_consumed')}")
    print(f"carrion_terminal_survivors={facts.get('carrion_terminal_survivors')}")
    print(
        "v195_dominant_requested_action_share="
        f"{facts.get('dominant_requested_action_share')}"
    )
    print(f"specificity_gate_replay_ran={after.get('ran')}")
    print(f"after_override_applied_share={combined.get('override_applied_share')}")
    print(
        "low_specificity_rejected_decision_count="
        f"{comparison.get('low_specificity_rejected_decision_count')}"
    )
    print(
        "exact_or_high_specificity_hit_share="
        f"{comparison.get('exact_or_high_specificity_hit_share')}"
    )
    print(
        "after_dominant_requested_action="
        f"{comparison.get('after_dominant_requested_action')}"
    )
    print(
        "after_dominant_requested_action_share="
        f"{comparison.get('after_dominant_requested_action_share')}"
    )
    print(
        "after_heuristic_action_source_count="
        f"{comparison.get('after_heuristic_action_source_count')}"
    )
    print(
        "after_broad_alive_birth_regression_count="
        f"{comparison.get('after_broad_alive_birth_regression_count')}"
    )
    print(
        "low_specificity_collapse_blocked="
        f"{comparison.get('low_specificity_collapse_blocked')}"
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
    print(f"exact_digest={report.get('exact_digest')}")


def _payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _csv(values: object) -> str:
    return ",".join(str(int(value)) for value in values)


def _parse_seed_csv(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not seeds:
        raise SystemExit("seed list must not be empty")
    return seeds


if __name__ == "__main__":
    main()
