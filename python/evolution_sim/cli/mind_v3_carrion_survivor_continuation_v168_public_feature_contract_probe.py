from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v166_source_split_action_value_scorer import (
    DEFAULT_V165_DATASET_PATH,
    EXPECTED_V165_DATASET_DIGEST,
)
from evolution_sim.mind.carrion_survivor_continuation_v167_source_split_failure_autopsy import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V167_REPORT_PATH,
    EXPECTED_V166_EXACT_DIGEST,
)
from evolution_sim.mind.carrion_survivor_continuation_v168_public_feature_contract_probe import (
    DEFAULT_OUTPUT_PATH,
    EXPECTED_V167_EXACT_DIGEST,
    CarrionSurvivorContinuationV168PublicFeatureContractProbeError,
    run_carrion_survivor_continuation_v168_public_feature_contract_probe,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v168 public feature-contract expansion "
            "probe for the v167 source-split failure. The command does not "
            "train a scorer for use, tune k or thresholds, run shadow eval, "
            "create a runtime artifact, or change runtime/replay/viewer schema."
        )
    )
    parser.add_argument("--v167-report", type=Path, default=DEFAULT_V167_REPORT_PATH)
    parser.add_argument("--v165-dataset", type=Path, default=DEFAULT_V165_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-v167-exact-digest",
        default=EXPECTED_V167_EXACT_DIGEST,
        help="Expected embedded stable digest for the v167 autopsy report.",
    )
    parser.add_argument(
        "--expected-v166-exact-digest",
        default=EXPECTED_V166_EXACT_DIGEST,
        help="Expected embedded stable digest for the source v166 report.",
    )
    parser.add_argument(
        "--expected-v165-dataset-digest",
        default=EXPECTED_V165_DATASET_DIGEST,
        help="Expected stable digest for the v165 expanded target dataset.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v168_public_feature_contract_probe(
            v167_report_path=args.v167_report,
            v165_dataset_path=args.v165_dataset,
            output_path=args.output,
            expected_v167_exact_digest=args.expected_v167_exact_digest,
            expected_v166_exact_digest=args.expected_v166_exact_digest,
            expected_v165_dataset_digest=args.expected_v165_dataset_digest,
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationV168PublicFeatureContractProbeError,
    ) as exc:
        raise SystemExit(
            "failed to run v168 carrion survivor-continuation public "
            f"feature-contract probe: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: dict[str, object], output: Path) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    source = report.get("source_validation")
    source_payload = source if isinstance(source, dict) else {}
    best = report.get("best_candidate")
    best_payload = best if isinstance(best, dict) else {}
    route = report.get("route_recommendation")
    route_payload = route if isinstance(route, dict) else {}
    baseline = report.get("v166_baseline_recomputed")
    baseline_payload = baseline if isinstance(baseline, dict) else {}
    print(
        "carrion_survivor_continuation_v168_public_feature_contract_probe_report="
        f"{output}"
    )
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={source_payload.get('passed')}")
    print(
        "baseline_zero_safe_hit_preterminal_source_seeds="
        f"{baseline_payload.get('zero_safe_hit_preterminal_source_seeds')}"
    )
    print(f"best_candidate_feature_family={best_payload.get('feature_family')}")
    print(
        "best_candidate_zero_safe_hit_preterminal_source_seeds="
        f"{best_payload.get('zero_safe_hit_preterminal_source_seeds')}"
    )
    print(
        "best_candidate_margin_over_best_trivial="
        f"{best_payload.get('safe_hit_margin_over_best_trivial')}"
    )
    print(f"recommended_next_route={route_payload.get('recommended_next_route')}")
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
