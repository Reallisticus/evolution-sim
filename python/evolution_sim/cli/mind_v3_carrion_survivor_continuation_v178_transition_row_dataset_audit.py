from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v177_exact_branch_replay_expansion import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V177_REPORT_PATH,
    DEFAULT_TRANSITION_DATASET_OUTPUT_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_v178_transition_row_dataset_audit import (
    DEFAULT_MIN_BRANCH_COUNT,
    DEFAULT_MIN_FORCED_ACTION_COUNT,
    DEFAULT_MIN_ROW_COUNT,
    DEFAULT_MIN_SEED_COUNT,
    DEFAULT_OUTPUT_PATH,
    run_carrion_survivor_continuation_v178_transition_row_dataset_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the diagnostics-only v177 compact transition-row dataset. "
            "This validates source integrity, row contracts, leakage, action-mask "
            "support, observation decodability, branch/action coverage, and "
            "support readiness. It does not train, fit, create runtime artifacts, "
            "change runtime action selection, run shadow/live evaluation, or "
            "authorize promotion."
        )
    )
    parser.add_argument(
        "--transition-dataset",
        type=Path,
        default=DEFAULT_TRANSITION_DATASET_OUTPUT_PATH,
    )
    parser.add_argument("--v177-report", type=Path, default=DEFAULT_V177_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--expected-v177-report-exact-digest", default=None)
    parser.add_argument("--expected-dataset-digest", default=None)
    parser.add_argument("--min-row-count", type=int, default=DEFAULT_MIN_ROW_COUNT)
    parser.add_argument("--min-seed-count", type=int, default=DEFAULT_MIN_SEED_COUNT)
    parser.add_argument(
        "--min-branch-count",
        type=int,
        default=DEFAULT_MIN_BRANCH_COUNT,
    )
    parser.add_argument(
        "--min-forced-action-count",
        type=int,
        default=DEFAULT_MIN_FORCED_ACTION_COUNT,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v178_transition_row_dataset_audit(
            transition_dataset_path=args.transition_dataset,
            v177_report_path=args.v177_report,
            output_path=args.output,
            expected_v177_report_exact_digest=args.expected_v177_report_exact_digest,
            expected_dataset_digest=args.expected_dataset_digest,
            min_row_count=args.min_row_count,
            min_seed_count=args.min_seed_count,
            min_branch_count=args.min_branch_count,
            min_forced_action_count=args.min_forced_action_count,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v178 carrion survivor-continuation transition-row "
            f"dataset audit: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: Mapping[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_validation"))
    schema = _payload(report.get("row_schema_validation"))
    leakage = _payload(report.get("leakage_scan"))
    coverage = _payload(report.get("coverage_audit"))
    observation = _payload(report.get("observation_audit"))
    support = _payload(report.get("support_readiness"))
    route = _payload(report.get("route_recommendation"))
    dataset = _payload(report.get("dataset"))
    print(f"carrion_survivor_continuation_v178_transition_row_dataset_audit={output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_validation_passed={source.get('passed')}")
    print(f"row_schema_validation_passed={schema.get('passed')}")
    print(f"leakage_scan_passed={leakage.get('passed')}")
    print(f"row_count={dataset.get('row_count')}")
    print(f"seed_count={coverage.get('seed_count')}")
    print(f"branch_count={coverage.get('branch_count')}")
    print(f"forced_action_count={coverage.get('forced_action_count')}")
    print(
        "current_observation_decoded_count="
        f"{observation.get('current_observation_decoded_count')}"
    )
    print(
        "next_observation_decoded_count="
        f"{observation.get('next_observation_decoded_count')}"
    )
    print(f"support_minimums_met={support.get('passed')}")
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(f"dataset_digest={dataset.get('dataset_digest')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"exact_digest={report.get('exact_digest')}")


def _payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
