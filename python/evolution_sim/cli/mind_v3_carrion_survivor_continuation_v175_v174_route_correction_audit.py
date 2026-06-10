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
    DEFAULT_OUTPUT_PATH as DEFAULT_V174_REPORT_PATH,
    EXPECTED_V173_EXACT_DIGEST,
)
from evolution_sim.mind.carrion_survivor_continuation_v175_v174_route_correction_audit import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V176_PLAN_OUTPUT_PATH,
    EXPECTED_V174_CLASSIFICATION,
    EXPECTED_V174_EXACT_DIGEST,
    CarrionSurvivorContinuationV175V174RouteCorrectionAuditError,
    run_carrion_survivor_continuation_v175_v174_route_correction_audit,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v175 route-correction audit for the "
            "v174 action-ranking-fit recommendation. This command does not "
            "train, fit, deploy, create a runtime artifact, change runtime "
            "action selection, run shadow/live evaluation, relax gates, or "
            "authorize promotion."
        )
    )
    parser.add_argument("--v174-report", type=Path, default=DEFAULT_V174_REPORT_PATH)
    parser.add_argument("--v173-report", type=Path, default=DEFAULT_V173_REPORT_PATH)
    parser.add_argument("--v172-report", type=Path, default=DEFAULT_V172_REPORT_PATH)
    parser.add_argument("--v172-dataset", type=Path, default=DEFAULT_V172_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--v176-plan-output",
        type=Path,
        default=DEFAULT_V176_PLAN_OUTPUT_PATH,
        help=(
            "Optional JSONL path used only when replay/world-model transition "
            "diagnostics are recommended."
        ),
    )
    parser.add_argument(
        "--no-v176-plan-output",
        action="store_true",
        help="Do not write optional v176 replay/world-model plan JSONL.",
    )
    parser.add_argument(
        "--expected-v174-exact-digest",
        default=EXPECTED_V174_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v174-classification",
        default=EXPECTED_V174_CLASSIFICATION,
    )
    parser.add_argument(
        "--expected-v173-exact-digest",
        default=EXPECTED_V173_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v172-dataset-digest",
        default=EXPECTED_V172_DATASET_DIGEST,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v175_v174_route_correction_audit(
            v174_report_path=args.v174_report,
            v173_report_path=args.v173_report,
            v172_report_path=args.v172_report,
            v172_dataset_path=args.v172_dataset,
            output_path=args.output,
            v176_plan_output_path=(
                None if args.no_v176_plan_output else args.v176_plan_output
            ),
            expected_v174_exact_digest=args.expected_v174_exact_digest,
            expected_v174_classification=args.expected_v174_classification,
            expected_v173_exact_digest=args.expected_v173_exact_digest,
            expected_v172_dataset_digest=args.expected_v172_dataset_digest,
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationV175V174RouteCorrectionAuditError,
    ) as exc:
        raise SystemExit(
            "failed to run v175 carrion survivor-continuation v174 route "
            f"correction audit: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: dict[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_validation"))
    lanes = _payload(report.get("lanes"))
    lane_a = _payload(lanes.get("lane_a_per_seed_lane_c_recompute"))
    lane_b = _payload(lanes.get("lane_b_clean_negative_controls"))
    lane_c = _payload(lanes.get("lane_c_tied_row_inflation_audit"))
    lane_d = _payload(lanes.get("lane_d_route_correction"))
    audited = _payload(lane_a.get("audited_entry"))
    aggregate = _payload(audited.get("aggregate"))
    seed_41 = _payload(lane_a.get("seed_41"))
    plan = _payload(report.get("v176_plan_output"))
    print(
        "carrion_survivor_continuation_v175_v174_route_correction_audit_report="
        f"{output}"
    )
    print(f"classification={classification.get('primary')}")
    print(f"source_validation_passed={source.get('passed')}")
    print(f"v174_exact_digest={source.get('observed_v174_exact_digest')}")
    print(f"v174_classification={source.get('observed_v174_classification')}")
    print(f"v173_exact_digest={source.get('observed_v173_exact_digest')}")
    print(f"v172_dataset_digest={source.get('observed_v172_dataset_digest')}")
    print(f"audited_ranking_label={audited.get('label')}")
    print(f"aggregate_pairwise_accuracy={aggregate.get('pairwise_accuracy')}")
    print(f"aggregate_top_3_safe_hit_rate={aggregate.get('top_3_safe_hit_rate')}")
    print(f"failed_support_seeds={lane_a.get('failed_support_seeds')}")
    print(f"seed_41_pairwise_accuracy={seed_41.get('pairwise_accuracy')}")
    print(f"seed_41_top_3_safe_hit_rate={seed_41.get('top_3_safe_hit_rate')}")
    print(
        "clean_controls_erase_margin="
        f"{lane_b.get('clean_controls_erase_margin')}"
    )
    print(
        "tied_row_inflation_detected="
        f"{lane_c.get('broad_tied_sets_hide_unique_row_weakness')}"
    )
    print(f"recommended_next_route={lane_d.get('recommended_next_route')}")
    print(f"v176_plan_written={plan.get('written')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"fit_ran={report.get('fit_ran')}")
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
