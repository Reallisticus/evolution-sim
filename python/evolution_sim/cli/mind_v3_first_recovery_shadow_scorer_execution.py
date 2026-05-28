from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V124_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V124_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_shadow_scorer_execution import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_PREDICTIONS_OUTPUT_PATH,
    build_first_recovery_shadow_scorer_execution,
    write_first_recovery_shadow_scorer_execution_outputs,
)
from evolution_sim.mind.first_recovery_shadow_scorer_proposal import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V125_REPORT_PATH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the diagnostics-only Mind v3 v126 first-recovery "
            "shadow-scorer execution."
        )
    )
    parser.add_argument("--v125-report", type=Path, default=DEFAULT_V125_REPORT_PATH)
    parser.add_argument("--v124-report", type=Path, default=DEFAULT_V124_REPORT_PATH)
    parser.add_argument("--v124-manifest", type=Path, default=DEFAULT_V124_MANIFEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=DEFAULT_PREDICTIONS_OUTPUT_PATH,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        build = build_first_recovery_shadow_scorer_execution(
            v125_report_path=args.v125_report,
            v124_report_path=args.v124_report,
            v124_manifest_path=args.v124_manifest,
        )
        write_first_recovery_shadow_scorer_execution_outputs(
            build,
            output_path=args.output,
            predictions_output_path=args.predictions_output,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"failed to run first-recovery shadow-scorer execution: {exc}"
        ) from exc
    _print_summary(build.report, args.output, args.predictions_output)


def _print_summary(
    report: Mapping[str, object],
    output_path: Path,
    predictions_output_path: Path,
) -> None:
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source_integrity"))
    summary = _mapping(report.get("prediction_summary"))
    action_distribution = _mapping(report.get("action_distribution"))
    dominant = _mapping(action_distribution.get("dominant_predicted_action"))
    material_gain = _mapping(report.get("material_gain_recall"))
    recommendation = _mapping(report.get("recommendation"))
    print(f"first_recovery_shadow_scorer_execution={output_path}")
    print(f"first_recovery_shadow_scorer_predictions={predictions_output_path}")
    print(f"classification={classification.get('primary')}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"prediction_row_count={summary.get('prediction_row_count')}")
    print(f"dominant_predicted_action={dominant.get('action')}")
    print(f"dominant_predicted_action_share={dominant.get('share')}")
    print(f"material_gain_recall={material_gain.get('recall')}")
    print(
        "downstream_shadow_scorer_allowed="
        f"{recommendation.get('downstream_shadow_scorer_allowed')}"
    )
    print(
        "v113_readiness_rerun_allowed="
        f"{recommendation.get('v113_readiness_rerun_allowed')}"
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
