from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v180_transition_row_policy_training import (
    DEFAULT_ARTIFACT_OUTPUT_PATH,
    DEFAULT_AUTHORIZATION_REPORT_PATH,
    DEFAULT_BROAD_SEEDS,
    DEFAULT_CARRION_FIXTURE_SEEDS,
    DEFAULT_FEATURE_KEY_LIMIT,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TICKS,
    DEFAULT_TRANSITION_DATASET_PATH,
    DEFAULT_UNOBSERVED_ACTION_UTILITY,
    EXPECTED_V178_AUTHORIZATION_CLASSIFICATION,
    EXPECTED_V178_AUTHORIZATION_REPORT_EXACT_DIGEST,
    EXPECTED_V179_TRANSITION_DATASET_DIGEST,
    run_carrion_survivor_continuation_v180_transition_row_policy_training,
)
from evolution_sim.mind.carrion_survivor_continuation_v178_transition_row_dataset_audit import (
    V179_SOURCE_PRODUCER,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v180 first explicit opt-in transition-row policy training "
            "slice. Training fails closed unless the durable pinned v178 "
            "authorization report authorizes the next same-lane training slice "
            "under default support thresholds. This command writes a training "
            "policy artifact and runs offline/shadow evaluation only; it does "
            "not integrate runtime behavior, relax gates, run live A/B, or "
            "authorize promotion."
        )
    )
    parser.add_argument(
        "--authorization-report",
        type=Path,
        default=DEFAULT_AUTHORIZATION_REPORT_PATH,
    )
    parser.add_argument(
        "--transition-dataset",
        type=Path,
        default=DEFAULT_TRANSITION_DATASET_PATH,
    )
    parser.add_argument("--artifact-output", type=Path, default=DEFAULT_ARTIFACT_OUTPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-authorization-report-exact-digest",
        default=EXPECTED_V178_AUTHORIZATION_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-dataset-digest",
        default=EXPECTED_V179_TRANSITION_DATASET_DIGEST,
    )
    parser.add_argument(
        "--expected-authorization-classification",
        default=EXPECTED_V178_AUTHORIZATION_CLASSIFICATION,
    )
    parser.add_argument("--expected-source-producer", default=V179_SOURCE_PRODUCER)
    parser.add_argument("--feature-key-limit", type=int, default=DEFAULT_FEATURE_KEY_LIMIT)
    parser.add_argument(
        "--unobserved-action-utility",
        type=float,
        default=DEFAULT_UNOBSERVED_ACTION_UTILITY,
    )
    parser.add_argument("--broad-seeds", default=_csv(DEFAULT_BROAD_SEEDS))
    parser.add_argument(
        "--carrion-fixture-seeds",
        default=_csv(DEFAULT_CARRION_FIXTURE_SEEDS),
    )
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Train and write the artifact without running the shadow eval.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v180_transition_row_policy_training(
            authorization_report_path=args.authorization_report,
            transition_dataset_path=args.transition_dataset,
            artifact_output_path=args.artifact_output,
            output_path=args.output,
            expected_authorization_report_exact_digest=(
                args.expected_authorization_report_exact_digest
            ),
            expected_dataset_digest=args.expected_dataset_digest,
            expected_authorization_classification=(
                args.expected_authorization_classification
            ),
            expected_source_producer=args.expected_source_producer,
            feature_key_limit=args.feature_key_limit,
            unobserved_action_utility=args.unobserved_action_utility,
            run_evaluation=not args.skip_evaluation,
            broad_seeds=_parse_seed_csv(args.broad_seeds),
            carrion_fixture_seeds=_parse_seed_csv(args.carrion_fixture_seeds),
            ticks=args.ticks,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v180 carrion survivor-continuation transition-row "
            f"policy training: {exc}"
        ) from exc
    _print_summary(report, args.output, args.artifact_output)


def _print_summary(
    report: Mapping[str, object],
    output: Path,
    artifact_output: Path,
) -> None:
    classification = _payload(report.get("classification"))
    auth = _payload(report.get("authorization_report_validation"))
    source = _payload(report.get("source_validation"))
    artifact = _payload(report.get("artifact"))
    training = _payload(report.get("training"))
    acceptance = _payload(report.get("acceptance"))
    controlled = _payload(acceptance.get("controlled_fixture"))
    print(
        "carrion_survivor_continuation_v180_transition_row_policy_training="
        f"{output}"
    )
    print(f"artifact_output={artifact_output}")
    print(f"classification={classification.get('primary')}")
    print(f"authorization_report={_payload(report.get('inputs')).get('authorization_report')}")
    print(f"authorization_report_validation_passed={auth.get('passed')}")
    print(f"authorization_report_exact_digest={auth.get('observed_exact_digest')}")
    print(f"source_validation_passed={source.get('passed')}")
    print(f"dataset_digest={_payload(report.get('dataset')).get('dataset_digest')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"training_artifact_created={report.get('training_artifact_created')}")
    print(f"training_artifact_digest={artifact.get('digest')}")
    print(f"training_row_count={training.get('training_row_count')}")
    print(f"shadow_eval_ran={report.get('shadow_eval_ran')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"acceptance_passed={acceptance.get('passed')}")
    print(
        "controlled_fixture_total_terminal_alive_agents="
        f"{controlled.get('total_terminal_alive_agents')}"
    )
    print(
        "dominant_requested_action_share="
        f"{acceptance.get('dominant_requested_action_share')}"
    )
    print(
        "heuristic_action_source_count="
        f"{acceptance.get('heuristic_action_source_count')}"
    )
    print(f"acceptance_blockers={acceptance.get('blockers')}")
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
