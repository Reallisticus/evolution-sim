from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v172_replay_target_dataset_expansion import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V172_REPORT_PATH,
    DEFAULT_TARGET_DATASET_OUTPUT_PATH as DEFAULT_V172_DATASET_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_v173_source_split_scorer import (
    EXPECTED_V172_DATASET_DIGEST,
)
from evolution_sim.mind.carrion_survivor_continuation_v174_mechanism_failure_battery import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V174_REPORT_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_v175_v174_route_correction_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V175_REPORT_PATH,
    DEFAULT_V176_PLAN_OUTPUT_PATH,
    EXPECTED_V174_EXACT_DIGEST,
)
from evolution_sim.mind.carrion_survivor_continuation_v176_transition_diagnostic_planner import (
    DEFAULT_GROUP_RELATIVE_OUTPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V177_SHARD_PLAN_OUTPUT_PATH,
    EXPECTED_V175_CLASSIFICATION,
    EXPECTED_V175_EXACT_DIGEST,
    EXPECTED_V176_PLAN_DIGEST,
    CarrionSurvivorContinuationV176TransitionDiagnosticPlannerError,
    run_carrion_survivor_continuation_v176_transition_diagnostic_planner,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run diagnostics-only v176 transition diagnostic planning from the "
            "v175 route correction. This command does not train, fit, deploy, "
            "create runtime artifacts, change runtime action selection, run "
            "shadow/live evaluation, relax gates, or authorize promotion."
        )
    )
    parser.add_argument("--v175-report", type=Path, default=DEFAULT_V175_REPORT_PATH)
    parser.add_argument("--v176-plan", type=Path, default=DEFAULT_V176_PLAN_OUTPUT_PATH)
    parser.add_argument("--v174-report", type=Path, default=DEFAULT_V174_REPORT_PATH)
    parser.add_argument("--v172-report", type=Path, default=DEFAULT_V172_REPORT_PATH)
    parser.add_argument("--v172-dataset", type=Path, default=DEFAULT_V172_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--group-relative-output",
        type=Path,
        default=DEFAULT_GROUP_RELATIVE_OUTPUT_PATH,
    )
    parser.add_argument(
        "--no-group-relative-output",
        action="store_true",
        help="Do not write group-relative transition-experience JSONL.",
    )
    parser.add_argument(
        "--v177-shard-plan-output",
        type=Path,
        default=DEFAULT_V177_SHARD_PLAN_OUTPUT_PATH,
    )
    parser.add_argument(
        "--no-v177-shard-plan-output",
        action="store_true",
        help="Do not write the v177 exact branch replay shard-plan JSONL.",
    )
    parser.add_argument(
        "--expected-v175-exact-digest",
        default=EXPECTED_V175_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v175-classification",
        default=EXPECTED_V175_CLASSIFICATION,
    )
    parser.add_argument(
        "--expected-v176-plan-digest",
        default=EXPECTED_V176_PLAN_DIGEST,
    )
    parser.add_argument(
        "--expected-v174-exact-digest",
        default=EXPECTED_V174_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v172-dataset-digest",
        default=EXPECTED_V172_DATASET_DIGEST,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v176_transition_diagnostic_planner(
            v175_report_path=args.v175_report,
            v176_plan_path=args.v176_plan,
            v174_report_path=args.v174_report,
            v172_report_path=args.v172_report,
            v172_dataset_path=args.v172_dataset,
            output_path=args.output,
            group_relative_output_path=(
                None if args.no_group_relative_output else args.group_relative_output
            ),
            v177_shard_plan_output_path=(
                None if args.no_v177_shard_plan_output else args.v177_shard_plan_output
            ),
            expected_v175_exact_digest=args.expected_v175_exact_digest,
            expected_v175_classification=args.expected_v175_classification,
            expected_v176_plan_digest=args.expected_v176_plan_digest,
            expected_v174_exact_digest=args.expected_v174_exact_digest,
            expected_v172_dataset_digest=args.expected_v172_dataset_digest,
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationV176TransitionDiagnosticPlannerError,
    ) as exc:
        raise SystemExit(
            "failed to run v176 carrion survivor-continuation transition "
            f"diagnostic planner: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: dict[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_validation"))
    lanes = _payload(report.get("lanes"))
    lane_a = _payload(lanes.get("lane_a_failed_seed_replay_target_map"))
    lane_b = _payload(lanes.get("lane_b_existing_transition_evidence_audit"))
    lane_d = _payload(lanes.get("lane_d_exact_branch_replay_shard_planner"))
    lane_e = _payload(
        lanes.get("lane_e_training_free_group_relative_transition_experience_probe")
    )
    lane_f = _payload(lanes.get("lane_f_route_decision"))
    group_output = _payload(report.get("group_relative_transition_experience_output"))
    shard_output = _payload(report.get("v177_shard_plan_output"))
    print(
        "carrion_survivor_continuation_v176_transition_diagnostic_planner_report="
        f"{output}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"source_validation_passed={source.get('passed')}")
    print(f"v175_exact_digest={source.get('observed_v175_exact_digest')}")
    print(f"v175_classification={source.get('observed_v175_classification')}")
    print(f"v176_plan_digest={source.get('observed_v176_plan_digest')}")
    print(f"v174_exact_digest={source.get('observed_v174_exact_digest')}")
    print(f"v172_dataset_digest={source.get('observed_v172_dataset_digest')}")
    print(
        "failed_seed_unique_row_failure_counts="
        f"{lane_a.get('unique_row_failure_counts_by_seed')}"
    )
    print(f"missing_transition_fields={lane_b.get('missing_transition_fields')}")
    print(
        "existing_transition_evidence_enough="
        f"{lane_b.get('existing_transition_evidence_enough_for_compact_diagnostic_dataset')}"
    )
    print(
        "group_relative_experience_rows="
        f"{lane_e.get('candidate_transition_experience_row_count')}"
    )
    print(
        "failed_seeds_have_enough_contrast="
        f"{lane_e.get('failed_seeds_have_enough_winner_loser_contrast')}"
    )
    print(f"v177_shard_rows={len(lane_d.get('v177_shard_plan_rows') or [])}")
    print(f"recommended_next_route={lane_f.get('recommended_next_route')}")
    print(f"group_relative_output_written={group_output.get('written')}")
    print(f"v177_shard_plan_written={shard_output.get('written')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"fit_ran={report.get('fit_ran')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"shadow_eval_ran={report.get('shadow_eval_ran')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"exact_digest={report.get('exact_digest')}")


def _payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
