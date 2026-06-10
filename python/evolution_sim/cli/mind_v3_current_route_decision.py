from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.current_route_decision import (
    DEFAULT_OUTPUT_PATH,
    build_mind_v3_current_route_decision_report,
    write_mind_v3_current_route_decision_report,
)
from evolution_sim.mind.first_recovery_history_refreshed_surface_probe import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V134_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_public_context_ranker_closeout import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V135_REPORT_PATH,
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
            "Generate the diagnostics-only Mind v3 v136 current-route decision "
            "report from v131-v135 evidence."
        )
    )
    parser.add_argument("--v131-report", type=Path, default=DEFAULT_V131_REPORT_PATH)
    parser.add_argument("--v132-report", type=Path, default=DEFAULT_V132_REPORT_PATH)
    parser.add_argument("--v133-report", type=Path, default=DEFAULT_V133_REPORT_PATH)
    parser.add_argument("--v134-report", type=Path, default=DEFAULT_V134_REPORT_PATH)
    parser.add_argument("--v135-report", type=Path, default=DEFAULT_V135_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        build = build_mind_v3_current_route_decision_report(
            v131_report_path=args.v131_report,
            v132_report_path=args.v132_report,
            v133_report_path=args.v133_report,
            v134_report_path=args.v134_report,
            v135_report_path=args.v135_report,
        )
        write_mind_v3_current_route_decision_report(build, output_path=args.output)
    except (OSError, ValueError) as exc:
        raise SystemExit(
            f"failed to generate Mind v3 current-route decision report: {exc}"
        ) from exc
    _print_summary(build.report, args.output)


def _print_summary(report: Mapping[str, object], output_path: Path) -> None:
    source = _mapping(report.get("source_integrity"))
    metrics = _mapping(report.get("v135_source_integrity_and_terminal_metrics"))
    dominant = _mapping(metrics.get("dominant_predicted_action"))
    auth = _mapping(report.get("authorization_block"))
    print(f"mind_v3_current_route_decision={output_path}")
    print(f"latest_evidence_version={report.get('latest_evidence_version')}")
    print(f"current_closed_path={report.get('current_closed_path')}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"v135_terminal_metrics_match_expected={metrics.get('matches_expected')}")
    print(f"heldout_accuracy={metrics.get('heldout_accuracy')}")
    print(f"heldout_action_only_accuracy={metrics.get('heldout_action_only_accuracy')}")
    print(f"heldout_action_order_accuracy={metrics.get('heldout_action_order_accuracy')}")
    print(f"seed29_accuracy={metrics.get('seed29_accuracy')}")
    print(f"dominant_predicted_action={dominant}")
    print(f"training_authorized={auth.get('training_authorized')}")
    print(
        "runtime_policy_change_authorized="
        f"{auth.get('runtime_policy_change_authorized')}"
    )
    print(
        "shadow_scorer_execution_authorized="
        f"{auth.get('shadow_scorer_execution_authorized')}"
    )
    print(
        "next_allowed_research_direction="
        f"{report.get('next_allowed_research_direction')}"
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
