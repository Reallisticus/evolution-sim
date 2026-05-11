from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.artifacts import load_model_artifact
from evolution_sim.mind.dataset import load_trajectory_jsonl
from evolution_sim.mind.diagnostics import build_artifact_diagnostics


DIAGNOSTICS_REPORT_SCHEMA_VERSION = "mind_artifact_diagnostics_report_v1"
DIAGNOSTICS_LEDGER_SCHEMA_VERSION = "mind_artifact_diagnostics_ledger_v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Write held-out Mind artifact diagnostics for trajectory banks.",
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--trajectory",
        type=Path,
        required=True,
        action="append",
        help="Trajectory JSONL or JSONL.GZ input.",
    )
    parser.add_argument(
        "--calibration-trajectory",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional separate trajectory bank for calibration diagnostics; "
            "not used to score the primary held-out bank."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional diagnostics report destination.",
    )
    parser.add_argument(
        "--experiment-ledger-output",
        type=Path,
        help="Append a compact diagnostics ledger entry to this JSONL file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    artifact = load_model_artifact(args.artifact, enable_mind=True)
    datasets = [load_trajectory_jsonl(path) for path in args.trajectory]
    calibration_datasets = [
        load_trajectory_jsonl(path)
        for path in args.calibration_trajectory
    ]
    diagnostics = build_artifact_diagnostics(artifact, datasets)
    report = {
        "schema_version": DIAGNOSTICS_REPORT_SCHEMA_VERSION,
        "artifact_path": str(args.artifact),
        "trajectory_paths": [str(path) for path in args.trajectory],
        "calibration_trajectory_paths": [
            str(path)
            for path in args.calibration_trajectory
        ],
        "model_type": artifact["manifest"]["model_type"],
        "trainer": artifact["model"].get("trainer"),
        "trained_record_count": artifact["manifest"]["trained_record_count"],
        "held_out_record_count": sum(dataset.record_count for dataset in datasets),
        "artifact_diagnostics": diagnostics,
    }
    if calibration_datasets:
        report["calibration_record_count"] = sum(
            dataset.record_count
            for dataset in calibration_datasets
        )
        report["calibration_diagnostics"] = build_artifact_diagnostics(
            artifact,
            calibration_datasets,
        )
    ledger_entry = _build_diagnostics_ledger_entry(report)
    report["experiment_ledger_entry"] = ledger_entry
    if args.experiment_ledger_output is not None:
        _append_ledger_entry(args.experiment_ledger_output, ledger_entry)
    payload = json.dumps(report, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


def _build_diagnostics_ledger_entry(
    report: Mapping[str, object],
) -> dict[str, object]:
    diagnostics = _mapping(report.get("artifact_diagnostics"))
    neural = _mapping(diagnostics.get("neural_calibration"))
    viability = _mapping(diagnostics.get("viability_calibration"))
    calibration = _mapping(report.get("calibration_diagnostics"))
    calibration_viability = _mapping(calibration.get("viability_calibration"))
    return {
        "schema_version": DIAGNOSTICS_LEDGER_SCHEMA_VERSION,
        "artifact_path": report.get("artifact_path"),
        "trajectory_paths": list(_list(report.get("trajectory_paths"))),
        "calibration_trajectory_paths": list(
            _list(report.get("calibration_trajectory_paths"))
        ),
        "model_type": report.get("model_type"),
        "trainer": report.get("trainer"),
        "held_out_record_count": report.get("held_out_record_count"),
        "actor_top1_accuracy": _float_or_none(
            neural.get("actor_top1_accuracy")
        ),
        "action_value_mean_abs_error": _float_or_none(
            neural.get("action_value_mean_abs_error")
        ),
        "state_value_mean_abs_error": _float_or_none(
            neural.get("state_value_mean_abs_error")
        ),
        "action_value_return_mean_abs_error": _float_or_none(
            neural.get("action_value_return_mean_abs_error")
        ),
        "state_value_return_mean_abs_error": _float_or_none(
            neural.get("state_value_return_mean_abs_error")
        ),
        "viability_risk_score_policy": viability.get("risk_score_policy"),
        "viability_constraint_risk_rate": _float_or_none(
            viability.get("constraint_risk_rate")
        ),
        "viability_risk_brier_score": _float_or_none(
            viability.get("risk_brier_score")
        ),
        "viability_risk_auc": _float_or_none(viability.get("risk_auc")),
        "calibration_viability_constraint_risk_rate": _float_or_none(
            calibration_viability.get("constraint_risk_rate")
        ),
        "calibration_viability_risk_brier_score": _float_or_none(
            calibration_viability.get("risk_brier_score")
        ),
        "calibration_viability_risk_auc": _float_or_none(
            calibration_viability.get("risk_auc")
        ),
    }


def _append_ledger_entry(path: Path, entry: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(entry), sort_keys=True, allow_nan=False) + "\n")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _list(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _float_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


if __name__ == "__main__":
    main()
