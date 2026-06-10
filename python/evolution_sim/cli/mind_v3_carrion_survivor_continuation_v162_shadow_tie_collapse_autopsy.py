from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy import (
    DEFAULT_MAX_DOMINANT_NEAREST_ROW_SHARE,
    DEFAULT_MAX_TOP2_NEAREST_ROW_SHARE,
    DEFAULT_MIN_BROAD_TOP_VALUE_SET_SHARE,
    DEFAULT_MIN_STAY_TIE_BREAK_ATTRIBUTED_SHARE,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TRAJECTORY_GLOB,
    DEFAULT_V160_ARTIFACT_PATH,
    DEFAULT_V160_REPORT_PATH,
    DEFAULT_V161_REPORT_PATH,
    CarrionSurvivorContinuationV162ShadowTieCollapseAutopsyError,
    run_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v162 shadow tie-collapse and "
            "nearest-neighbor concentration autopsy for the v160/v161 carrion "
            "survivor-continuation scorer path. The command does not change "
            "runtime action selection or create a runtime artifact."
        )
    )
    parser.add_argument("--v160-report", type=Path, default=DEFAULT_V160_REPORT_PATH)
    parser.add_argument(
        "--v160-artifact",
        type=Path,
        default=DEFAULT_V160_ARTIFACT_PATH,
    )
    parser.add_argument("--v161-report", type=Path, default=DEFAULT_V161_REPORT_PATH)
    parser.add_argument("--trajectory-glob", default=DEFAULT_TRAJECTORY_GLOB)
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        default=None,
        help="Explicit shadow evidence JSONL/JSONL.GZ path. May be repeated.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--max-dominant-nearest-row-share",
        type=float,
        default=DEFAULT_MAX_DOMINANT_NEAREST_ROW_SHARE,
        help="Maximum dominant nearest training-row share before flagging concentration.",
    )
    parser.add_argument(
        "--max-top2-nearest-row-share",
        type=float,
        default=DEFAULT_MAX_TOP2_NEAREST_ROW_SHARE,
        help="Maximum top-two nearest training-row share before flagging concentration.",
    )
    parser.add_argument(
        "--min-stay-tie-break-attributed-share",
        type=float,
        default=DEFAULT_MIN_STAY_TIE_BREAK_ATTRIBUTED_SHARE,
        help="Minimum tied-top-set share among stay predictions before blocking.",
    )
    parser.add_argument(
        "--min-broad-top-value-set-share",
        type=float,
        default=DEFAULT_MIN_BROAD_TOP_VALUE_SET_SHARE,
        help="Minimum multi-action top-value set share before recommending resolver work.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy(
            v160_report_path=args.v160_report,
            v160_artifact_path=args.v160_artifact,
            v161_report_path=args.v161_report,
            trajectory_glob=args.trajectory_glob,
            trajectory_paths=args.trajectory,
            output_path=args.output,
            max_dominant_nearest_row_share=args.max_dominant_nearest_row_share,
            max_top2_nearest_row_share=args.max_top2_nearest_row_share,
            min_stay_tie_break_attributed_share=(
                args.min_stay_tie_break_attributed_share
            ),
            min_broad_top_value_set_share=args.min_broad_top_value_set_share,
        )
    except (
        OSError,
        ValueError,
        CarrionSurvivorContinuationV162ShadowTieCollapseAutopsyError,
    ) as exc:
        raise SystemExit(
            "failed to run v162 carrion survivor-continuation shadow "
            f"tie-collapse autopsy: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: dict[str, object], output: Path) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    source = report.get("source_validation")
    source_payload = source if isinstance(source, dict) else {}
    autopsy = report.get("shadow_tie_collapse_autopsy")
    autopsy_payload = autopsy if isinstance(autopsy, dict) else {}
    nearest = autopsy_payload.get("nearest_neighbor_row_attribution")
    nearest_payload = nearest if isinstance(nearest, dict) else {}
    tie = autopsy_payload.get("deterministic_tie_break_attribution")
    tie_payload = tie if isinstance(tie, dict) else {}
    route = report.get("route_recommendation")
    route_payload = route if isinstance(route, dict) else {}
    print(f"carrion_survivor_continuation_v162_shadow_tie_collapse_autopsy_report={output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={source_payload.get('passed')}")
    print(f"prediction_count={autopsy_payload.get('prediction_count')}")
    print(f"predicted_action_counts={autopsy_payload.get('predicted_action_counts')}")
    print(
        "dominant_nearest_row_share="
        f"{nearest_payload.get('dominant_nearest_row_share')}"
    )
    print(f"top2_nearest_row_share={nearest_payload.get('top2_nearest_row_share')}")
    print(
        "stay_tie_break_attributed_share_of_stay_predictions="
        f"{tie_payload.get('stay_tie_break_attributed_share_of_stay_predictions')}"
    )
    print(
        "set_valued_candidates_noncollapsed_but_broad="
        f"{autopsy_payload.get('set_valued_candidates_noncollapsed_but_broad')}"
    )
    print(f"recommended_next_route={route_payload.get('recommended_next_route')}")
    print(f"live_ab_allowed={report.get('live_ab_allowed')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"exact_digest={report.get('exact_digest')}")


if __name__ == "__main__":
    main()
