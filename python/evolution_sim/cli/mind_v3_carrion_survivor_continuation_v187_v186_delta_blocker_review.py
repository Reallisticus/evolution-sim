from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v187_v186_delta_blocker_review import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V186_ARTIFACT_PATH,
    DEFAULT_V186_REPORT_PATH,
    EXPECTED_V186_ARTIFACT_DIGEST,
    EXPECTED_V186_CLASSIFICATION,
    EXPECTED_V186_REPORT_EXACT_DIGEST,
    run_carrion_survivor_continuation_v187_v186_delta_blocker_review,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v187 v186-delta blocker review. The "
            "command validates pinned v186 report/artifact digests, records "
            "old carrion conclusions as inherited, extracts only the new v186 "
            "delta, and recommends one no-training next route before slice 3. "
            "It does not train, rerun v186, run support expansion, integrate "
            "runtime behavior, relax gates, or authorize promotion."
        )
    )
    parser.add_argument("--v186-report", type=Path, default=DEFAULT_V186_REPORT_PATH)
    parser.add_argument("--v186-artifact", type=Path, default=DEFAULT_V186_ARTIFACT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-v186-report-exact-digest",
        default=EXPECTED_V186_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v186-artifact-digest",
        default=EXPECTED_V186_ARTIFACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v186-classification",
        default=EXPECTED_V186_CLASSIFICATION,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v187_v186_delta_blocker_review(
            v186_report_path=args.v186_report,
            v186_artifact_path=args.v186_artifact,
            output_path=args.output,
            expected_v186_report_exact_digest=args.expected_v186_report_exact_digest,
            expected_v186_artifact_digest=args.expected_v186_artifact_digest,
            expected_v186_classification=args.expected_v186_classification,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v187 carrion survivor-continuation v186 delta "
            f"blocker review: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: Mapping[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_validation"))
    delta = _payload(report.get("v186_delta"))
    delta_validation = _payload(report.get("delta_validation"))
    route = _payload(report.get("route_decision"))
    blocker_review = _payload(report.get("blocker_review"))
    print(f"carrion_survivor_continuation_v187_v186_delta_blocker_review={output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_validation_passed={source.get('passed')}")
    print(
        "v186_report_exact_digest="
        f"{source.get('observed_v186_report_exact_digest')}"
    )
    print(f"v186_artifact_digest={source.get('observed_v186_artifact_digest')}")
    print(f"delta_validation_passed={delta_validation.get('passed')}")
    print(
        "broad_alive_birth_regressions_empty="
        f"{delta.get('broad_alive_birth_regressions_empty')}"
    )
    print(
        "dominant_requested_action_share="
        f"{delta.get('dominant_requested_action_share')}"
    )
    print(
        "heuristic_action_source_count="
        f"{delta.get('heuristic_action_source_count')}"
    )
    print(
        "carrion_only_terminal_survivors="
        f"{delta.get('carrion_only_terminal_survivors')}"
    )
    print(f"carrion_fixture_births_mean={delta.get('carrion_fixture_births_mean')}")
    print(
        "inherited_prior_findings_status="
        "inherited_not_rediscovered"
    )
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(f"blocker_count={blocker_review.get('blocker_count')}")
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
