from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v161_shadow_eval import (
    DEFAULT_MAX_DOMINANT_SHADOW_ACTION_SHARE,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TRAJECTORY_GLOB,
    DEFAULT_V160_ARTIFACT_PATH,
    DEFAULT_V160_REPORT_PATH,
    CarrionSurvivorContinuationV161ShadowEvalError,
    run_carrion_survivor_continuation_v161_shadow_eval,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only v161 shadow evaluation of the v160 "
            "carrion survivor-continuation action-value scorer. The command "
            "does not integrate the scorer with runtime action selection."
        )
    )
    parser.add_argument("--v160-report", type=Path, default=DEFAULT_V160_REPORT_PATH)
    parser.add_argument(
        "--v160-artifact",
        type=Path,
        default=DEFAULT_V160_ARTIFACT_PATH,
    )
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
        "--max-dominant-shadow-action-share",
        type=float,
        default=DEFAULT_MAX_DOMINANT_SHADOW_ACTION_SHARE,
        help="Maximum dominant shadow predicted-action share before blocking.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v161_shadow_eval(
            v160_report_path=args.v160_report,
            v160_artifact_path=args.v160_artifact,
            trajectory_glob=args.trajectory_glob,
            trajectory_paths=args.trajectory,
            output_path=args.output,
            max_dominant_shadow_action_share=(
                args.max_dominant_shadow_action_share
            ),
        )
    except (OSError, ValueError, CarrionSurvivorContinuationV161ShadowEvalError) as exc:
        raise SystemExit(
            "failed to run v161 carrion survivor-continuation shadow eval: "
            f"{exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: dict[str, object], output: Path) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    source = report.get("source_validation")
    source_payload = source if isinstance(source, dict) else {}
    shadow = report.get("shadow_evaluation")
    shadow_payload = shadow if isinstance(shadow, dict) else {}
    route = report.get("route_recommendation")
    route_payload = route if isinstance(route, dict) else {}
    print(f"carrion_survivor_continuation_v161_shadow_eval_report={output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"source_validation_passed={source_payload.get('passed')}")
    print(f"prediction_count={shadow_payload.get('prediction_count')}")
    print(
        "unsupported_shadow_prediction_count="
        f"{shadow_payload.get('unsupported_shadow_prediction_count')}"
    )
    print(
        "dominant_predicted_action_share="
        f"{shadow_payload.get('dominant_predicted_action_share')}"
    )
    print(f"would_change_count={shadow_payload.get('would_change_count')}")
    print(f"would_change_share={shadow_payload.get('would_change_share')}")
    print(
        "future_separate_opt_in_live_ab_diagnostic_recommended="
        f"{route_payload.get('future_separate_opt_in_live_ab_diagnostic_recommended')}"
    )
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"runtime_promotion_allowed={report.get('runtime_promotion_allowed')}")
    print(f"runtime_action_selection_changed={report.get('runtime_action_selection_changed')}")
    print(f"exact_digest={report.get('exact_digest')}")


if __name__ == "__main__":
    main()
