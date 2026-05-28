from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_history_refreshed_surface_probe import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V134_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_public_context_ranker_closeout import (
    DEFAULT_OUTPUT_PATH,
    build_first_recovery_public_context_ranker_closeout,
    write_first_recovery_public_context_ranker_closeout_report,
)
from evolution_sim.mind.first_recovery_public_rollout_history_context_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V133_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_refreshed_candidate_public_feature_surface import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V131_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_refreshed_surface_blocker_slice_audit import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V132_REPORT_PATH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the diagnostics-only Mind v3 v135 first-recovery public-"
            "context ranker path closeout report."
        )
    )
    parser.add_argument("--v131-report", type=Path, default=DEFAULT_V131_REPORT_PATH)
    parser.add_argument("--v132-report", type=Path, default=DEFAULT_V132_REPORT_PATH)
    parser.add_argument("--v133-report", type=Path, default=DEFAULT_V133_REPORT_PATH)
    parser.add_argument("--v134-report", type=Path, default=DEFAULT_V134_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        build = build_first_recovery_public_context_ranker_closeout(
            v131_report_path=args.v131_report,
            v132_report_path=args.v132_report,
            v133_report_path=args.v133_report,
            v134_report_path=args.v134_report,
        )
        write_first_recovery_public_context_ranker_closeout_report(
            build,
            output_path=args.output,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to generate first-recovery public-context ranker closeout: "
            f"{exc}"
        ) from exc
    _print_summary(build.report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    classification = _mapping(report.get("classification"))
    recommendation = _mapping(report.get("recommendation"))
    source = _mapping(report.get("source_integrity"))
    terminal = _mapping(report.get("terminal_signal_summary"))
    fixture_open = _mapping(terminal.get("fixture_open_dominant_predicted_action"))
    overall = _mapping(terminal.get("overall_dominant_predicted_action"))
    auth = _mapping(report.get("authorization_block"))
    print(f"first_recovery_public_context_ranker_closeout={output_path}")
    print(f"classification={classification.get('primary')}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"final_recommendation={recommendation.get('next_step')}")
    print(f"heldout_accuracy={terminal.get('heldout_accuracy')}")
    print(
        "heldout_delta_vs_action_only="
        f"{terminal.get('heldout_delta_vs_action_only')}"
    )
    print(f"fixture_open_dominant_predicted_action={fixture_open}")
    print(f"seed29_passed={terminal.get('seed29_passed')}")
    print(f"seed29_accuracy={terminal.get('seed29_accuracy')}")
    print(f"overall_dominant_predicted_action={overall}")
    print(f"training_executed={auth.get('training_executed')}")
    print(
        "downstream_shadow_scorer_allowed="
        f"{auth.get('downstream_shadow_scorer_allowed')}"
    )
    print(
        "v113_readiness_rerun_allowed="
        f"{auth.get('v113_readiness_rerun_allowed')}"
    )
    print(
        "runtime_policy_change_recommended="
        f"{auth.get('runtime_policy_change_recommended')}"
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
