from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V195_ARTIFACT_PATH,
    DEFAULT_V197_REPORT_PATH,
    EXPECTED_V195_ARTIFACT_DIGEST,
    EXPECTED_V195_REPORT_EXACT_DIGEST,
    EXPECTED_V196_REPORT_EXACT_DIGEST,
    EXPECTED_V197_REPORT_EXACT_DIGEST,
    EXPECTED_V197_ROUTE,
    run_carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair,
)
from evolution_sim.mind.carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training import (
    DEFAULT_BROAD_SEEDS,
    DEFAULT_CARRION_FIXTURE_SEEDS,
    DEFAULT_TICKS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v198 no-training high-specificity coverage/model-capacity "
            "repair probe before any slice-4 training. The command validates the "
            "pinned v197 route, probes the frozen v195 artifact with transition "
            "value action override disabled, and writes a closed lifecycle report."
        )
    )
    parser.add_argument("--v197-report", type=Path, default=DEFAULT_V197_REPORT_PATH)
    parser.add_argument("--v195-artifact", type=Path, default=DEFAULT_V195_ARTIFACT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-v197-report-exact-digest",
        default=EXPECTED_V197_REPORT_EXACT_DIGEST,
    )
    parser.add_argument("--expected-v197-route", default=EXPECTED_V197_ROUTE)
    parser.add_argument(
        "--expected-v196-report-exact-digest",
        default=EXPECTED_V196_REPORT_EXACT_DIGEST,
    )
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
            "Skip the v198 real replay probe. This is for source-pin tests only; "
            "a durable v198 report should not use it."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair(
            v197_report_path=args.v197_report,
            v195_artifact_path=args.v195_artifact,
            output_path=args.output,
            expected_v197_report_exact_digest=args.expected_v197_report_exact_digest,
            expected_v197_route=args.expected_v197_route,
            expected_v196_report_exact_digest=args.expected_v196_report_exact_digest,
            expected_v195_report_exact_digest=args.expected_v195_report_exact_digest,
            expected_v195_artifact_digest=args.expected_v195_artifact_digest,
            run_diagnostic_replay=not args.skip_diagnostic_replay,
            broad_seeds=_parse_seed_csv(args.broad_seeds),
            carrion_fixture_seeds=_parse_seed_csv(args.carrion_fixture_seeds),
            ticks=args.ticks,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v198 high-specificity coverage/model-capacity repair: "
            f"{exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: Mapping[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_pin_validation"))
    probe = _payload(report.get("high_specificity_probe"))
    combined = _payload(probe.get("combined"))
    assessment = _payload(report.get("coverage_assessment"))
    route = _payload(report.get("route_decision"))
    print(
        "carrion_survivor_continuation_v198_high_specificity_coverage_or_model_capacity_repair="
        f"{output}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"source_pin_validation_passed={source.get('passed')}")
    print(f"v197_report_exact_digest={source.get('observed_v197_report_exact_digest')}")
    print(f"v197_route={source.get('observed_v197_route')}")
    print(f"v195_artifact_digest={source.get('observed_v195_artifact_digest')}")
    print(f"high_specificity_probe_ran={probe.get('ran')}")
    print(f"real_replay_provenance={assessment.get('probe_real_replay_provenance')}")
    print(
        "high_specificity_any_key_present_share="
        f"{combined.get('high_specificity_any_key_present_share')}"
    )
    print(
        "high_specificity_any_complete_share="
        f"{combined.get('high_specificity_any_complete_share')}"
    )
    print(
        "high_specificity_observed_support_floor_share="
        f"{combined.get('high_specificity_any_observed_support_floor_satisfied_share')}"
    )
    print(
        "high_specificity_selected_source_share="
        f"{combined.get('high_specificity_selected_source_share')}"
    )
    print(f"primary_blocker={assessment.get('primary_blocker')}")
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
    print(f"support_expansion_ran={report.get('support_expansion_ran')}")
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
