from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v195_repaired_contract_terminal_survival_support_training import (
    DEFAULT_BROAD_SEEDS,
    DEFAULT_CARRION_FIXTURE_SEEDS,
    DEFAULT_TICKS,
)
from evolution_sim.mind.carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V195_ARTIFACT_PATH,
    DEFAULT_V198_REPORT_PATH,
    EXPECTED_V195_ARTIFACT_DIGEST,
    EXPECTED_V198_ARCHIVE_PATH,
    EXPECTED_V198_ARCHIVE_SHA256,
    EXPECTED_V198_CLASSIFICATION,
    EXPECTED_V198_PRIMARY_BLOCKER,
    EXPECTED_V198_REPORT_EXACT_DIGEST,
    EXPECTED_V198_ROUTE,
    run_carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v199 no-training high-specificity action-complete contract "
            "audit. The command validates the pinned v198 route, reruns only the "
            "needed real replay candidate-key diagnostics with action override "
            "disabled, and writes a closed lifecycle report."
        )
    )
    parser.add_argument("--v198-report", type=Path, default=DEFAULT_V198_REPORT_PATH)
    parser.add_argument("--v195-artifact", type=Path, default=DEFAULT_V195_ARTIFACT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-v198-report-exact-digest",
        default=EXPECTED_V198_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v198-classification",
        default=EXPECTED_V198_CLASSIFICATION,
    )
    parser.add_argument("--expected-v198-route", default=EXPECTED_V198_ROUTE)
    parser.add_argument(
        "--expected-v198-primary-blocker",
        default=EXPECTED_V198_PRIMARY_BLOCKER,
    )
    parser.add_argument(
        "--expected-v198-archive-path",
        default=EXPECTED_V198_ARCHIVE_PATH,
    )
    parser.add_argument(
        "--expected-v198-archive-sha256",
        default=EXPECTED_V198_ARCHIVE_SHA256,
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
            "Skip the real replay audit. This is for source-pin tests only; a "
            "durable v199 report should not use it."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit(
            v198_report_path=args.v198_report,
            v195_artifact_path=args.v195_artifact,
            output_path=args.output,
            expected_v198_report_exact_digest=args.expected_v198_report_exact_digest,
            expected_v198_classification=args.expected_v198_classification,
            expected_v198_route=args.expected_v198_route,
            expected_v198_primary_blocker=args.expected_v198_primary_blocker,
            expected_v198_archive_path=args.expected_v198_archive_path,
            expected_v198_archive_sha256=args.expected_v198_archive_sha256,
            expected_v195_artifact_digest=args.expected_v195_artifact_digest,
            run_diagnostic_replay=not args.skip_diagnostic_replay,
            broad_seeds=_parse_seed_csv(args.broad_seeds),
            carrion_fixture_seeds=_parse_seed_csv(args.carrion_fixture_seeds),
            ticks=args.ticks,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v199 high-specificity action-complete contract audit: "
            f"{exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: Mapping[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_pin_validation"))
    audit = _payload(report.get("action_complete_audit"))
    combined = _payload(audit.get("combined"))
    assessment = _payload(report.get("contract_assessment"))
    route = _payload(report.get("route_decision"))
    print(
        "carrion_survivor_continuation_v199_high_specificity_action_complete_contract_audit="
        f"{output}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"source_pin_validation_passed={source.get('passed')}")
    print(f"v198_report_exact_digest={source.get('observed_v198_report_exact_digest')}")
    print(f"v198_route={source.get('observed_v198_route')}")
    print(f"high_specificity_action_audit_ran={audit.get('ran')}")
    print(f"real_replay_provenance={assessment.get('audit_real_replay_provenance')}")
    print(
        "high_specificity_candidate_evaluated_count="
        f"{combined.get('high_specificity_candidate_evaluated_count')}"
    )
    print(f"key_absent_count={combined.get('key_absent_count')}")
    print(f"key_present_count={combined.get('key_present_count')}")
    print(
        "present_but_action_incomplete_count="
        f"{combined.get('present_but_action_incomplete_count')}"
    )
    print(
        "present_complete_but_observed_support_floor_failed_count="
        f"{combined.get('present_complete_but_observed_support_floor_failed_count')}"
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
    print(f"support_generation_ran={report.get('support_generation_ran')}")
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
