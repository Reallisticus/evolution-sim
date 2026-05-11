from __future__ import annotations

import argparse
import json
from pathlib import Path

from evolution_sim.mind.carrion_counterfactual import DEFAULT_COUNTERFACTUAL_SCRIPTS
from evolution_sim.mind.carrion_counterfactual_labels import (
    DEFAULT_PRIMARY_COUNTERFACTUAL_LABEL_HORIZON,
    MIND_V3_CARRION_COUNTERFACTUAL_LABEL_SCHEMA_VERSION,
    CarrionCounterfactualLabelError,
    build_carrion_counterfactual_label_report,
    parse_counterfactual_label_horizons,
    write_carrion_counterfactual_label_report,
)
from evolution_sim.mind.dataset import TrajectoryDatasetError, load_trajectory_jsonl
from evolution_sim.mind.horizon_labels import DEFAULT_HORIZON_TICKS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build explicit Mind v3 carrion counterfactual action/value labels "
            "from scripted trajectory JSONL datasets."
        )
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        required=True,
        help="Counterfactual trajectory JSONL or JSONL.gz input. Repeat to combine.",
    )
    parser.add_argument(
        "--counterfactual-report",
        type=Path,
        default=None,
        help=(
            "Optional source counterfactual report. When --script is omitted, "
            "successful_scripts from this report are used as filters."
        ),
    )
    parser.add_argument(
        "--script",
        action="append",
        choices=DEFAULT_COUNTERFACTUAL_SCRIPTS,
        help=(
            "Counterfactual script to include. Repeat to include multiple "
            "scripts. Defaults to successful scripts from --counterfactual-report "
            "when supplied, otherwise all records."
        ),
    )
    parser.add_argument(
        "--horizons",
        default=",".join(str(horizon) for horizon in DEFAULT_HORIZON_TICKS),
        help="Comma-separated label horizons.",
    )
    parser.add_argument(
        "--primary-horizon",
        type=int,
        default=DEFAULT_PRIMARY_COUNTERFACTUAL_LABEL_HORIZON,
        help="Primary horizon used for aggregate action/value diagnostics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-carrion-counterfactual-labels.json"),
        help="Output JSON report path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        source_report = (
            _load_json_report(args.counterfactual_report)
            if args.counterfactual_report is not None
            else None
        )
        datasets = [load_trajectory_jsonl(path) for path in args.trajectory]
        report = build_carrion_counterfactual_label_report(
            datasets,
            horizons=parse_counterfactual_label_horizons(args.horizons),
            primary_horizon=int(args.primary_horizon),
            scripts=tuple(args.script) if args.script else None,
            source_counterfactual_report=source_report,
            source_counterfactual_report_path=args.counterfactual_report,
        )
        write_carrion_counterfactual_label_report(report, args.output)
    except (
        OSError,
        json.JSONDecodeError,
        ValueError,
        TrajectoryDatasetError,
        CarrionCounterfactualLabelError,
    ) as exc:
        raise SystemExit(
            f"failed to build carrion counterfactual labels: {exc}"
        ) from exc

    aggregate = report["aggregate"]  # type: ignore[index]
    print(f"carrion_counterfactual_labels={args.output}")
    print(f"schema_version={MIND_V3_CARRION_COUNTERFACTUAL_LABEL_SCHEMA_VERSION}")
    print(f"label_count={aggregate['label_count']}")  # type: ignore[index]
    print(f"primary_horizon={aggregate['primary_horizon']}")  # type: ignore[index]
    print(
        "primary_terminal_alive_rate="
        f"{aggregate['primary_terminal_alive_rate']}"  # type: ignore[index]
    )
    print(
        "primary_animal_resource_gain_total="
        f"{aggregate['primary_animal_resource_gain_total']}"  # type: ignore[index]
    )


def _load_json_report(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CarrionCounterfactualLabelError(
            f"counterfactual report must be a JSON object: {path}"
        )
    return payload


if __name__ == "__main__":
    main()
