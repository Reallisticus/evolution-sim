from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_sequence_context_comparator_support_closeout import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V153_REPORT_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_archive import (
    DEFAULT_DATASET_OUTPUT_PATH as DEFAULT_V154_DATASET_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V154_REPORT_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_train_eval import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V155_REPORT_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_v166_source_split_action_value_scorer import (
    DEFAULT_V165_DATASET_PATH,
    EXPECTED_V165_DATASET_DIGEST,
)
from evolution_sim.mind.carrion_survivor_continuation_v167_source_split_failure_autopsy import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V167_REPORT_PATH,
)
from evolution_sim.mind.carrion_survivor_continuation_v168_public_feature_contract_probe import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V168_REPORT_PATH,
    EXPECTED_V167_EXACT_DIGEST,
)
from evolution_sim.mind.carrion_survivor_continuation_v169_public_temporal_context_probe import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V169_REPORT_PATH,
    EXPECTED_V168_EXACT_DIGEST,
)
from evolution_sim.mind.carrion_survivor_continuation_v170_diagnostic_portfolio_matrix import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SHARD_PLAN_OUTPUT_PATH,
    EXPECTED_V153_EXACT_DIGEST,
    EXPECTED_V154_EXACT_DIGEST,
    EXPECTED_V155_EXACT_DIGEST,
    EXPECTED_V169_CLASSIFICATION,
    EXPECTED_V169_EXACT_DIGEST,
    CarrionSurvivorContinuationV170DiagnosticPortfolioMatrixError,
    run_carrion_survivor_continuation_v170_diagnostic_portfolio_matrix,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v170 carrion survivor-continuation "
            "portfolio matrix. This command evaluates cheap existing-evidence "
            "set-valued/ranking and sequence-alias lanes, emits a v171 shard "
            "plan, and does not train, tune, shadow/live eval, create runtime "
            "artifacts, or change runtime/replay/viewer behavior."
        )
    )
    parser.add_argument("--v169-report", type=Path, default=DEFAULT_V169_REPORT_PATH)
    parser.add_argument("--v168-report", type=Path, default=DEFAULT_V168_REPORT_PATH)
    parser.add_argument("--v167-report", type=Path, default=DEFAULT_V167_REPORT_PATH)
    parser.add_argument("--v165-dataset", type=Path, default=DEFAULT_V165_DATASET_PATH)
    parser.add_argument("--v155-report", type=Path, default=DEFAULT_V155_REPORT_PATH)
    parser.add_argument("--v154-report", type=Path, default=DEFAULT_V154_REPORT_PATH)
    parser.add_argument("--v154-dataset", type=Path, default=DEFAULT_V154_DATASET_PATH)
    parser.add_argument("--v153-report", type=Path, default=DEFAULT_V153_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--shard-plan-output",
        type=Path,
        default=DEFAULT_SHARD_PLAN_OUTPUT_PATH,
        help="Optional JSONL path for deterministic v171 replay-expansion shard plans.",
    )
    parser.add_argument(
        "--no-shard-plan-output",
        action="store_true",
        help="Do not write the optional shard-plan JSONL file.",
    )
    parser.add_argument(
        "--expected-v169-exact-digest",
        default=EXPECTED_V169_EXACT_DIGEST,
        help="Expected embedded exact digest for the v169 closeout report.",
    )
    parser.add_argument(
        "--expected-v169-classification",
        default=EXPECTED_V169_CLASSIFICATION,
        help="Expected v169 closed classification.",
    )
    parser.add_argument(
        "--expected-v168-exact-digest",
        default=EXPECTED_V168_EXACT_DIGEST,
        help="Expected embedded exact digest for the v168 probe report.",
    )
    parser.add_argument(
        "--expected-v167-exact-digest",
        default=EXPECTED_V167_EXACT_DIGEST,
        help="Expected embedded exact digest for the v167 autopsy report.",
    )
    parser.add_argument(
        "--expected-v165-dataset-digest",
        default=EXPECTED_V165_DATASET_DIGEST,
        help="Expected stable digest for the v165 expanded target dataset.",
    )
    parser.add_argument(
        "--expected-v155-exact-digest",
        default=EXPECTED_V155_EXACT_DIGEST,
        help="Expected embedded exact digest for the v155 train/eval closeout.",
    )
    parser.add_argument(
        "--expected-v154-exact-digest",
        default=EXPECTED_V154_EXACT_DIGEST,
        help="Expected embedded exact digest for the v154 archive report.",
    )
    parser.add_argument(
        "--expected-v153-exact-digest",
        default=EXPECTED_V153_EXACT_DIGEST,
        help="Expected embedded exact digest for the v153 comparator closeout.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    shard_output = None if args.no_shard_plan_output else args.shard_plan_output
    try:
        report = run_carrion_survivor_continuation_v170_diagnostic_portfolio_matrix(
            v169_report_path=args.v169_report,
            v168_report_path=args.v168_report,
            v167_report_path=args.v167_report,
            v165_dataset_path=args.v165_dataset,
            v155_report_path=args.v155_report,
            v154_report_path=args.v154_report,
            v154_dataset_path=args.v154_dataset,
            v153_report_path=args.v153_report,
            output_path=args.output,
            shard_plan_output_path=shard_output,
            expected_v169_exact_digest=args.expected_v169_exact_digest,
            expected_v169_classification=args.expected_v169_classification,
            expected_v168_exact_digest=args.expected_v168_exact_digest,
            expected_v167_exact_digest=args.expected_v167_exact_digest,
            expected_v165_dataset_digest=args.expected_v165_dataset_digest,
            expected_v155_exact_digest=args.expected_v155_exact_digest,
            expected_v154_exact_digest=args.expected_v154_exact_digest,
            expected_v153_exact_digest=args.expected_v153_exact_digest,
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationV170DiagnosticPortfolioMatrixError,
    ) as exc:
        raise SystemExit(
            "failed to run v170 carrion survivor-continuation diagnostic "
            f"portfolio matrix: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: dict[str, object], output: Path) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    source = report.get("source_validation")
    source_payload = source if isinstance(source, dict) else {}
    lanes = report.get("portfolio_lanes")
    lane_payload = lanes if isinstance(lanes, dict) else {}
    set_lane = lane_payload.get("set_valued_ranking")
    set_payload = set_lane if isinstance(set_lane, dict) else {}
    best_set = set_payload.get("best_matrix_entry")
    best_set_payload = best_set if isinstance(best_set, dict) else {}
    sequence_lane = lane_payload.get("sequence_memory_alias")
    sequence_payload = sequence_lane if isinstance(sequence_lane, dict) else {}
    best_sequence = sequence_payload.get("best_candidate")
    best_sequence_payload = best_sequence if isinstance(best_sequence, dict) else {}
    shard_output = report.get("shard_plan_output")
    shard_payload = shard_output if isinstance(shard_output, dict) else {}
    print(
        "carrion_survivor_continuation_v170_diagnostic_portfolio_matrix_report="
        f"{output}"
    )
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={source_payload.get('passed')}")
    print(
        "best_set_feature_family="
        f"{best_set_payload.get('feature_family')}"
    )
    print(f"best_set_top_n={best_set_payload.get('top_n')}")
    print(
        "best_set_zero_safe_hit_preterminal_source_seeds="
        f"{best_set_payload.get('zero_safe_hit_preterminal_source_seeds')}"
    )
    print(f"best_set_average_width={best_set_payload.get('average_set_width')}")
    print(f"best_set_sets_are_broad={best_set_payload.get('sets_are_broad')}")
    print(
        "best_sequence_conflicting_rows="
        f"{best_sequence_payload.get('conflicting_row_count')}"
    )
    print(
        "shard_plan_output="
        f"{shard_payload.get('path')}"
    )
    print(f"shard_plan_rows={shard_payload.get('jsonl_row_count')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"k_tuning_ran={report.get('k_tuning_ran')}")
    print(f"threshold_tuning_ran={report.get('threshold_tuning_ran')}")
    print(f"shadow_eval_ran={report.get('shadow_eval_ran')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(
        "runtime_action_selection_changed="
        f"{report.get('runtime_action_selection_changed')}"
    )
    print(f"exact_digest={report.get('exact_digest')}")


if __name__ == "__main__":
    main()
