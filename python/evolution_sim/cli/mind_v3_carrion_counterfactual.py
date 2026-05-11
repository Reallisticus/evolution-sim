from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_counterfactual import (
    DEFAULT_CARRION_COUNTERFACTUAL_SEEDS,
    DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
    DEFAULT_COUNTERFACTUAL_SCRIPTS,
    MIND_V3_CARRION_COUNTERFACTUAL_SCHEMA_VERSION,
    CarrionCounterfactualError,
    build_carrion_counterfactual_report,
    write_carrion_counterfactual_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic policy-visible carrion-water counterfactual "
            "scripts against the Mind v3 carrion-only fixture."
        )
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_CARRION_COUNTERFACTUAL_SEEDS),
        help="Comma-separated carrion fixture seeds.",
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=DEFAULT_CARRION_COUNTERFACTUAL_TICKS,
        help="Fixture rollout horizon.",
    )
    parser.add_argument(
        "--script",
        action="append",
        choices=DEFAULT_COUNTERFACTUAL_SCRIPTS,
        help=(
            "Counterfactual script to run. Repeat to run multiple scripts. "
            "Defaults to all scripts."
        ),
    )
    parser.add_argument(
        "--source-autopsy",
        type=Path,
        default=None,
        help="Optional source autopsy report path to record in provenance.",
    )
    parser.add_argument(
        "--trajectory-output-dir",
        type=Path,
        default=None,
        help="Optional directory for per-script trajectory JSONL.gz outputs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-carrion-counterfactual.json"),
        help="Output JSON report path. Use .gz for gzip compression.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    scripts = tuple(args.script) if args.script else DEFAULT_COUNTERFACTUAL_SCRIPTS
    try:
        report = build_carrion_counterfactual_report(
            seeds=_parse_seeds(args.seeds),
            ticks=int(args.ticks),
            scripts=scripts,
            trajectory_output_dir=args.trajectory_output_dir,
            source_autopsy_path=args.source_autopsy,
        )
        write_carrion_counterfactual_report(report, args.output)
    except (OSError, ValueError, CarrionCounterfactualError) as exc:
        raise SystemExit(f"failed to run carrion counterfactual: {exc}") from exc

    aggregate = report["aggregate"]  # type: ignore[index]
    best = aggregate["best_script_by_terminal_alive_then_births"]  # type: ignore[index]
    print(f"carrion_counterfactual={args.output}")
    print(f"schema_version={MIND_V3_CARRION_COUNTERFACTUAL_SCHEMA_VERSION}")
    print(f"script_count={aggregate['script_count']}")  # type: ignore[index]
    print(f"run_count={aggregate['run_count']}")  # type: ignore[index]
    print(
        "survivable_sequence_found="
        f"{aggregate['survivable_sequence_found']}"  # type: ignore[index]
    )
    if isinstance(best, dict):
        print(f"best_script={best['script_name']}")
        print(f"best_alive_agents_mean={best['alive_agents_mean']}")
        print(f"best_births_mean={best['births_mean']}")


def _parse_seeds(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values:
        raise CarrionCounterfactualError("--seeds must include at least one seed")
    return values


if __name__ == "__main__":
    main()
