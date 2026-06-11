from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v181_v180_failure_response import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V180_ARTIFACT_PATH,
    DEFAULT_V180_REPORT_PATH,
    DEFAULT_TRANSITION_DATASET_PATH,
    DEFAULT_BROAD_SEEDS,
    DEFAULT_CARRION_FIXTURE_SEEDS,
    DEFAULT_TICKS,
    EXPECTED_V179_TRANSITION_DATASET_DIGEST,
    EXPECTED_V180_ARTIFACT_DIGEST,
    EXPECTED_V180_CLASSIFICATION,
    EXPECTED_V180_REPORT_EXACT_DIGEST,
    run_carrion_survivor_continuation_v181_v180_failure_response,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v181 failure response for the failed v180 "
            "transition-row policy slice. This command validates the pinned "
            "v180 report/artifact/dataset, diagnoses support and override "
            "failure mechanisms, and writes an autopsy report only. It does "
            "not train, create a policy artifact, integrate runtime behavior, "
            "relax gates, or authorize promotion."
        )
    )
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
        "--expected-v180-report-exact-digest",
        default=EXPECTED_V180_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v180-artifact-digest",
        default=EXPECTED_V180_ARTIFACT_DIGEST,
    )
    parser.add_argument(
        "--expected-dataset-digest",
        default=EXPECTED_V179_TRANSITION_DATASET_DIGEST,
    )
    parser.add_argument(
        "--expected-v180-classification",
        default=EXPECTED_V180_CLASSIFICATION,
    )
    parser.add_argument("--broad-seeds", default=_csv(DEFAULT_BROAD_SEEDS))
    parser.add_argument(
        "--carrion-fixture-seeds",
        default=_csv(DEFAULT_CARRION_FIXTURE_SEEDS),
    )
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument(
        "--skip-trace-replay",
        action="store_true",
        help="Skip deterministic diagnostic trace replay and use only v180 report aggregates.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v181_v180_failure_response(
            v180_report_path=args.v180_report,
            v180_artifact_path=args.v180_artifact,
            transition_dataset_path=args.transition_dataset,
            output_path=args.output,
            expected_v180_report_exact_digest=args.expected_v180_report_exact_digest,
            expected_v180_artifact_digest=args.expected_v180_artifact_digest,
            expected_dataset_digest=args.expected_dataset_digest,
            expected_v180_classification=args.expected_v180_classification,
            run_trace_replay=not args.skip_trace_replay,
            broad_seeds=_parse_seed_csv(args.broad_seeds),
            carrion_fixture_seeds=_parse_seed_csv(args.carrion_fixture_seeds),
            ticks=args.ticks,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v181 carrion survivor-continuation v180 failure "
            f"response: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: Mapping[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_validation"))
    mechanism = _payload(report.get("failure_mechanism"))
    artifact_support = _payload(report.get("artifact_support"))
    trace = _payload(report.get("trace_replay"))
    broad_trace = _payload(_payload(trace.get("broad")).get("aggregate"))
    carrion_trace = _payload(_payload(trace.get("controlled_fixture")).get("aggregate"))
    print(f"carrion_survivor_continuation_v181_v180_failure_response={output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_validation_passed={source.get('passed')}")
    print(f"v180_report_exact_digest={source.get('observed_v180_report_exact_digest')}")
    print(f"v180_artifact_digest={source.get('observed_v180_artifact_digest')}")
    print(f"dataset_digest={source.get('observed_dataset_digest')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"training_artifact_created={report.get('training_artifact_created')}")
    print(f"diagnostic_trace_replay_ran={report.get('diagnostic_trace_replay_ran')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"slice_2_training_consumed={mechanism.get('slice_2_training_consumed')}")
    print(f"mechanism_primary={mechanism.get('primary')}")
    print(f"mechanism_labels={mechanism.get('labels')}")
    print(
        "artifact_imputed_action_stat_share="
        f"{artifact_support.get('imputed_action_stat_share')}"
    )
    print(
        "broad_runtime_action_selection_changed_count="
        f"{broad_trace.get('runtime_action_selection_changed_count')}"
    )
    print(
        "broad_predicted_action_counts="
        f"{broad_trace.get('predicted_action_counts')}"
    )
    print(
        "carrion_runtime_action_selection_changed_count="
        f"{carrion_trace.get('runtime_action_selection_changed_count')}"
    )
    print(f"next_route={mechanism.get('next_route')}")
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
