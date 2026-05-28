from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_recovery_residual_audit import (
    CarrionRecoveryResidualAuditError,
    build_carrion_recovery_residual_audit_report,
    write_carrion_recovery_residual_audit_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a diagnostics-only audit explaining why a Mind v3 carrion "
            "recovery residual artifact did or did not activate."
        )
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="Input Mind v3 neural residual artifact.",
    )
    parser.add_argument(
        "--distill-report",
        type=Path,
        required=True,
        help="Input carrion recovery distillation report.",
    )
    parser.add_argument(
        "--evaluation-report",
        type=Path,
        default=None,
        help=(
            "Optional carrion recovery distill evaluation report for counter "
            "reconciliation."
        ),
    )
    parser.add_argument(
        "--activation-audit",
        type=Path,
        default=None,
        help=(
            "Optional prior residual activation audit report for v66 fixture "
            "counter reconciliation."
        ),
    )
    parser.add_argument(
        "--archive-report",
        type=Path,
        required=True,
        help="Input carrion recovery archive report.",
    )
    parser.add_argument(
        "--split-report",
        type=Path,
        required=True,
        help="Input leakage-safe carrion recovery archive split report.",
    )
    parser.add_argument(
        "--fixture-seeds",
        default="13,19,29,37,41,43",
        help="Comma-separated carrion_only fixture seeds for the replay audit.",
    )
    parser.add_argument(
        "--fixture-ticks",
        type=int,
        default=120,
        help="Carrion-only fixture tick horizon for the replay audit.",
    )
    parser.add_argument(
        "--margin-sweep",
        default="",
        help=(
            "Optional comma-separated linear override margin thresholds for "
            "offline shadow calibration."
        ),
    )
    parser.add_argument(
        "--scale-sweep",
        default="",
        help=(
            "Optional comma-separated residual scales for offline shadow "
            "calibration."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "output/mind/mind-v3-v66-recovery-residual-activation-audit.json"
        ),
        help="Output residual activation audit report path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        fixture_seeds = _parse_seeds(args.fixture_seeds)
        report = build_carrion_recovery_residual_audit_report(
            artifact_path=args.artifact,
            distill_report_path=args.distill_report,
            evaluation_report_path=args.evaluation_report,
            activation_audit_path=args.activation_audit,
            archive_report_path=args.archive_report,
            split_report_path=args.split_report,
            fixture_seeds=fixture_seeds,
            fixture_ticks=int(args.fixture_ticks),
            margin_sweep=_parse_float_list(args.margin_sweep, field="--margin-sweep"),
            scale_sweep=_parse_float_list(args.scale_sweep, field="--scale-sweep"),
        )
        write_carrion_recovery_residual_audit_report(report, args.output)
    except (OSError, ValueError, CarrionRecoveryResidualAuditError) as exc:
        raise SystemExit(
            f"failed to audit carrion recovery residual activation: {exc}"
        ) from exc

    _print_report_summary(report, args.output)


def _print_report_summary(
    report: Mapping[str, object],
    output_path: Path,
) -> None:
    classification = _mapping(report.get("failure_classification"))
    feature_checks = _mapping(report.get("feature_contract_checks"))
    heldout = _mapping(
        _mapping(report.get("offline_score_summaries")).get("heldout")
    )
    fixture = _mapping(report.get("fixture_replay_diagnostics"))
    print(f"wrote {output_path}")
    print(f"classification={classification.get('primary')}")
    print(f"feature_contract_passed={feature_checks.get('passed')}")
    print(
        "heldout_configured_would_change_count="
        f"{heldout.get('configured_would_change_count', 0)}"
    )
    print(
        "heldout_shadow_forced_gate_would_change_count="
        f"{heldout.get('shadow_forced_gate_would_change_count', 0)}"
    )
    print(
        "heldout_shadow_margin_ignored_would_change_count="
        f"{heldout.get('shadow_margin_ignored_would_change_count', 0)}"
    )
    print(f"fixture_context_gate_pass_count={fixture.get('context_gate_pass_count', 0)}")
    print(f"fixture_residual_applied_count={fixture.get('residual_applied_count', 0)}")
    print(
        "fixture_actual_changed_linear_count="
        f"{fixture.get('actual_changed_linear_count', 0)}"
    )
    reconciliation = _mapping(report.get("residual_counter_reconciliation"))
    if reconciliation:
        mismatch = _mapping(reconciliation.get("classification"))
        print(f"mismatch_classification={mismatch.get('primary')}")
    calibration = _mapping(report.get("calibration_classification"))
    if calibration:
        print(f"calibration_classification={calibration.get('primary')}")
        strongest = _mapping(_mapping(report.get("shadow_calibration")).get("strongest_shadow_setting"))
        print(
            "strongest_shadow_setting="
            f"{strongest.get('surface')}:{strongest.get('label')}"
        )
        print(
            "strongest_shadow_would_change_count="
            f"{strongest.get('would_change_count', 0)}"
        )
        print(
            "strongest_shadow_action_collapse_risk="
            f"{strongest.get('action_collapse_risk', False)}"
        )


def _parse_seeds(raw: str) -> tuple[int, ...]:
    values: list[int] = []
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(int(stripped))
    if not values:
        raise ValueError("--fixture-seeds must include at least one seed")
    return tuple(values)


def _parse_float_list(raw: str, *, field: str) -> tuple[float, ...]:
    values: list[float] = []
    if not raw:
        return ()
    for token in raw.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    if any(value < 0.0 for value in values):
        raise ValueError(f"{field} values must be nonnegative")
    return tuple(values)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
