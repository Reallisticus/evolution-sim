from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v182_imputed_abstention_design import (
    DEFAULT_BROAD_SEEDS,
    DEFAULT_CARRION_FIXTURE_SEEDS,
    DEFAULT_OBSERVED_SUPPORT_FLOOR,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TICKS,
    DEFAULT_TRANSITION_DATASET_PATH,
    DEFAULT_V180_ARTIFACT_PATH,
    DEFAULT_V180_REPORT_PATH,
    DEFAULT_V181_REPORT_PATH,
    EXPECTED_DATASET_DIGEST,
    EXPECTED_V180_ARTIFACT_DIGEST,
    EXPECTED_V180_CLASSIFICATION,
    EXPECTED_V180_REPORT_EXACT_DIGEST,
    EXPECTED_V181_CLASSIFICATION,
    EXPECTED_V181_REPORT_EXACT_DIGEST,
    run_carrion_survivor_continuation_v182_imputed_abstention_design,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v182 failure-response design for the "
            "failed v180 transition-row policy slice. This validates pinned "
            "v181/v180 evidence, evaluates the imputed-action abstention rule "
            "offline, and writes a report only. It does not train, create a "
            "policy artifact, change default runtime behavior, relax gates, "
            "consume slice 2, or authorize promotion."
        )
    )
    parser.add_argument("--v181-report", type=Path, default=DEFAULT_V181_REPORT_PATH)
    parser.add_argument("--v180-report", type=Path, default=DEFAULT_V180_REPORT_PATH)
    parser.add_argument(
        "--v180-artifact",
        type=Path,
        default=DEFAULT_V180_ARTIFACT_PATH,
    )
    parser.add_argument(
        "--transition-dataset",
        type=Path,
        default=DEFAULT_TRANSITION_DATASET_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-v181-report-exact-digest",
        default=EXPECTED_V181_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v180-report-exact-digest",
        default=EXPECTED_V180_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v180-artifact-digest",
        default=EXPECTED_V180_ARTIFACT_DIGEST,
    )
    parser.add_argument(
        "--expected-dataset-digest",
        default=EXPECTED_DATASET_DIGEST,
    )
    parser.add_argument(
        "--expected-v181-classification",
        default=EXPECTED_V181_CLASSIFICATION,
    )
    parser.add_argument(
        "--expected-v180-classification",
        default=EXPECTED_V180_CLASSIFICATION,
    )
    parser.add_argument(
        "--observed-support-floor",
        type=int,
        default=DEFAULT_OBSERVED_SUPPORT_FLOOR,
    )
    parser.add_argument("--broad-seeds", default=_csv(DEFAULT_BROAD_SEEDS))
    parser.add_argument(
        "--carrion-fixture-seeds",
        default=_csv(DEFAULT_CARRION_FIXTURE_SEEDS),
    )
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument(
        "--skip-shadow-evaluation",
        action="store_true",
        help="Skip deterministic offline shadow/design evaluation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v182_imputed_abstention_design(
            v181_report_path=args.v181_report,
            v180_report_path=args.v180_report,
            v180_artifact_path=args.v180_artifact,
            transition_dataset_path=args.transition_dataset,
            output_path=args.output,
            expected_v181_report_exact_digest=(
                args.expected_v181_report_exact_digest
            ),
            expected_v180_report_exact_digest=(
                args.expected_v180_report_exact_digest
            ),
            expected_v180_artifact_digest=args.expected_v180_artifact_digest,
            expected_dataset_digest=args.expected_dataset_digest,
            expected_v181_classification=args.expected_v181_classification,
            expected_v180_classification=args.expected_v180_classification,
            observed_support_floor=args.observed_support_floor,
            run_shadow_evaluation=not args.skip_shadow_evaluation,
            broad_seeds=_parse_seed_csv(args.broad_seeds),
            carrion_fixture_seeds=_parse_seed_csv(args.carrion_fixture_seeds),
            ticks=args.ticks,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v182 carrion survivor-continuation imputed "
            f"abstention design: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: Mapping[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_validation"))
    route = _payload(report.get("route"))
    diagnostics = _payload(report.get("design_diagnostics"))
    broad = _payload(diagnostics.get("broad"))
    carrion = _payload(diagnostics.get("controlled_fixture"))
    broad_tv = _payload(broad.get("transition_value_scorer_diagnostics"))
    carrion_tv = _payload(carrion.get("transition_value_scorer_diagnostics"))
    print(f"carrion_survivor_continuation_v182_imputed_abstention_design={output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_validation_passed={source.get('passed')}")
    print(f"v181_report_exact_digest={source.get('observed_v181_report_exact_digest')}")
    print(f"v180_report_exact_digest={source.get('observed_v180_report_exact_digest')}")
    print(f"v180_artifact_digest={source.get('observed_v180_artifact_digest')}")
    print(f"dataset_digest={source.get('observed_dataset_digest')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"training_artifact_created={report.get('training_artifact_created')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"slice_2_training_consumed={report.get('slice_2_training_consumed')}")
    print(f"shadow_eval_ran={report.get('shadow_eval_ran')}")
    print(
        "broad_override_applied_count="
        f"{broad_tv.get('override_applied_count')}"
    )
    print(
        "broad_imputed_valid_action_decision_count="
        f"{broad_tv.get('imputed_valid_action_decision_count')}"
    )
    print(
        "carrion_observed_support_floor_satisfied_count="
        f"{carrion_tv.get('observed_support_floor_satisfied_count')}"
    )
    print(
        "strict_observed_support_leaves_carrion_coverage_zero="
        f"{diagnostics.get('strict_observed_support_leaves_carrion_coverage_zero')}"
    )
    print(f"next_route={route.get('next_route')}")
    print(f"route_reason={route.get('reason')}")
    print(f"exact_digest={report.get('exact_digest')}")


def _parse_seed_csv(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not seeds:
        raise ValueError("seed list must not be empty")
    return seeds


def _csv(values: tuple[int, ...]) -> str:
    return ",".join(str(value) for value in values)


def _payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
