from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v172_replay_target_dataset_expansion import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V172_REPORT_PATH,
    DEFAULT_TARGET_DATASET_OUTPUT_PATH as DEFAULT_V172_DATASET_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_v173_source_split_scorer import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V173_REPORT_PATH,
    EXPECTED_V172_DATASET_DIGEST,
)
from evolution_sim.mind.carrion_survivor_continuation_v174_mechanism_failure_battery import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V175_PLAN_OUTPUT_PATH,
    EXPECTED_V173_CLASSIFICATION,
    EXPECTED_V173_EXACT_DIGEST,
    CarrionSurvivorContinuationV174MechanismFailureBatteryError,
    run_carrion_survivor_continuation_v174_mechanism_failure_battery,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v174 mechanism-aware failure battery for "
            "the carrion survivor-continuation v173 scorer failure. This "
            "does not train, create a runtime artifact, change runtime action "
            "selection, run shadow/live evaluation, relax gates, or authorize "
            "promotion."
        )
    )
    parser.add_argument("--v173-report", type=Path, default=DEFAULT_V173_REPORT_PATH)
    parser.add_argument("--v172-report", type=Path, default=DEFAULT_V172_REPORT_PATH)
    parser.add_argument("--v172-dataset", type=Path, default=DEFAULT_V172_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--v175-plan-output",
        type=Path,
        default=DEFAULT_V175_PLAN_OUTPUT_PATH,
        help=(
            "Optional JSONL path used only when the planner recommends "
            "replay/world-model expansion."
        ),
    )
    parser.add_argument(
        "--no-v175-plan-output",
        action="store_true",
        help="Do not write optional v175 replay/world-model plan JSONL.",
    )
    parser.add_argument(
        "--expected-v173-exact-digest",
        default=EXPECTED_V173_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v173-classification",
        default=EXPECTED_V173_CLASSIFICATION,
    )
    parser.add_argument(
        "--expected-v172-dataset-digest",
        default=EXPECTED_V172_DATASET_DIGEST,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v174_mechanism_failure_battery(
            v173_report_path=args.v173_report,
            v172_report_path=args.v172_report,
            v172_dataset_path=args.v172_dataset,
            output_path=args.output,
            v175_plan_output_path=(
                None if args.no_v175_plan_output else args.v175_plan_output
            ),
            expected_v173_exact_digest=args.expected_v173_exact_digest,
            expected_v173_classification=args.expected_v173_classification,
            expected_v172_dataset_digest=args.expected_v172_dataset_digest,
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationV174MechanismFailureBatteryError,
    ) as exc:
        raise SystemExit(
            "failed to run v174 carrion survivor-continuation mechanism "
            f"failure battery: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: dict[str, object], output: Path) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    source = report.get("source_validation")
    source_payload = source if isinstance(source, dict) else {}
    lanes = report.get("lanes")
    lane_payload = lanes if isinstance(lanes, dict) else {}
    lane_a = _payload(lane_payload.get("lane_a_v173_score_mechanics_autopsy"))
    lane_b = _payload(lane_payload.get("lane_b_local_support_frontier"))
    lane_c = _payload(lane_payload.get("lane_c_action_ranking_capacity_probe"))
    lane_d = _payload(
        lane_payload.get("lane_d_conformal_action_conditional_set_calibration")
    )
    lane_e = _payload(lane_payload.get("lane_e_public_context_sufficiency_probe"))
    best_local = _payload(lane_b.get("best_frontier_entry"))
    best_ranking = _payload(lane_c.get("best_ranking_entry"))
    best_conformal = _payload(lane_d.get("best_entry"))
    plan = _payload(report.get("v175_plan_output"))
    print(
        "carrion_survivor_continuation_v174_mechanism_failure_battery_report="
        f"{output}"
    )
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={source_payload.get('passed')}")
    print(f"v173_exact_digest={source_payload.get('observed_v173_exact_digest')}")
    print(
        "v173_classification="
        f"{source_payload.get('observed_v173_classification')}"
    )
    print(
        "v172_dataset_digest="
        f"{source_payload.get('observed_v172_dataset_digest')}"
    )
    print(f"lane_a_full_set_share={lane_a.get('full_public_mask_set_share')}")
    print(
        "lane_a_positive_rule_explains_full_sets="
        f"{_payload(lane_a.get('full_set_positive_rule_proof')).get('full_set_behavior_explained_by_positive_score_rule')}"
    )
    print(f"lane_b_best_label={best_local.get('label')}")
    print(f"lane_b_best_singleton_hit_rate={best_local.get('singleton_hit_rate')}")
    print(f"lane_b_best_top_3_set_hit_rate={best_local.get('top_3_set_hit_rate')}")
    print(f"lane_c_best_label={best_ranking.get('label')}")
    print(f"lane_c_pairwise_accuracy={best_ranking.get('pairwise_accuracy')}")
    print(
        "lane_c_pairwise_margin_over_shuffled="
        f"{best_ranking.get('pairwise_accuracy_margin_over_shuffled_target')}"
    )
    print(f"lane_c_top_3_safe_hit_rate={best_ranking.get('top_3_safe_hit_rate')}")
    print(f"lane_d_best_coverage_rate={best_conformal.get('coverage_rate')}")
    print(f"lane_d_best_average_width={best_conformal.get('average_action_conditional_width')}")
    print(f"lane_d_best_full_set_share={best_conformal.get('full_set_share')}")
    print(f"lane_e_any_projection_improved={lane_e.get('any_projection_improved')}")
    print(f"v175_plan_written={plan.get('written')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"shadow_eval_ran={report.get('shadow_eval_ran')}")
    print(f"live_ab_ran={report.get('live_ab_ran')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"exact_digest={report.get('exact_digest')}")


def _payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    main()
