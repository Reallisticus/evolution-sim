from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.evaluation_harness import (
    _aggregate_runs,
    _run_fixture_once,
)
from evolution_sim.mind.carrion_autopsy import (
    build_carrion_autopsy_report,
    load_carrion_autopsy_trajectory_jsonl,
    write_carrion_autopsy_report,
)
from evolution_sim.mind.carrion_objective_audit import (
    DEFAULT_CARRION_OBJECTIVE_SEARCH_REPORTS,
    DEFAULT_CARRION_OBJECTIVE_TRACE_REPORTS,
    DEFAULT_FOUNDATION_CARRION_BASELINE_SEEDS,
    DEFAULT_FOUNDATION_CARRION_BASELINE_TICKS,
    MIND_V3_GATE_ALIGNED_CARRION_SELECTOR_PROBE_POLICY,
    MIND_V3_CARRION_OBJECTIVE_AUDIT_SCHEMA_VERSION,
    CarrionObjectiveAuditError,
    build_carrion_objective_audit_report,
    foundation_heuristic_baseline_report,
    load_json_report,
    parse_labeled_path,
    write_json_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a diagnostics-only Mind v3 carrion objective-pressure audit "
            "from search reports and carrion-only trace reports."
        )
    )
    parser.add_argument(
        "--search-report",
        action="append",
        help=(
            "Search report JSON as LABEL=PATH. Repeat to override defaults. "
            "When omitted, v4/v5/v6 diagnostic report paths are used if present."
        ),
    )
    parser.add_argument(
        "--trace-report",
        action="append",
        help=(
            "Carrion trace JSON as LABEL=PATH. Repeat to override defaults. "
            "When omitted, v4/v5/v6 carrion-only trace paths are used if present."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/mind/mind-v3-carrion-objective-audit.json"),
    )
    parser.add_argument(
        "--selector-probe",
        choices=[MIND_V3_GATE_ALIGNED_CARRION_SELECTOR_PROBE_POLICY],
        help=(
            "Emit a report-only retrospective selector comparison. This does "
            "not affect runtime policy, scoring, gates, or candidate selection."
        ),
    )
    parser.add_argument(
        "--generate-heuristic-baseline",
        action="store_true",
        help=(
            "Generate the true Foundation/default-policy carrion_only baseline "
            "trajectories and autopsy trace. This uses policy=None and does not "
            "load a founder template."
        ),
    )
    parser.add_argument(
        "--baseline-seeds",
        default=",".join(
            str(seed) for seed in DEFAULT_FOUNDATION_CARRION_BASELINE_SEEDS
        ),
        help="Comma-separated seeds for --generate-heuristic-baseline.",
    )
    parser.add_argument(
        "--baseline-ticks",
        type=int,
        default=DEFAULT_FOUNDATION_CARRION_BASELINE_TICKS,
    )
    parser.add_argument(
        "--baseline-output",
        type=Path,
        default=Path("output/mind/mind-v3-carrion-only-120-heuristic-baseline.json"),
    )
    parser.add_argument(
        "--baseline-trajectory-output-dir",
        type=Path,
        default=Path("output/mind/mind-v3-carrion-only-120-heuristic-trajectories"),
    )
    parser.add_argument(
        "--baseline-trace-output",
        type=Path,
        default=Path("output/mind/mind-v3-carrion-only-120-heuristic-trace.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.generate_heuristic_baseline:
            baseline = _generate_foundation_heuristic_baseline(
                seeds=_parse_seeds(args.baseline_seeds),
                ticks=int(args.baseline_ticks),
                output_path=args.baseline_output,
                trajectory_output_dir=args.baseline_trajectory_output_dir,
                trace_output_path=args.baseline_trace_output,
            )
            print(f"foundation_carrion_baseline={args.baseline_output}")
            print(f"foundation_carrion_trace={args.baseline_trace_output}")
            print(
                "foundation_carrion_uses_founder_template="
                f"{baseline['policy']['uses_founder_template']}"  # type: ignore[index]
            )
            return

        search_reports, missing_search = _load_labeled_reports(
            args.search_report,
            defaults=DEFAULT_CARRION_OBJECTIVE_SEARCH_REPORTS,
            kind="search_report",
        )
        trace_reports, missing_trace = _load_labeled_reports(
            args.trace_report,
            defaults=DEFAULT_CARRION_OBJECTIVE_TRACE_REPORTS,
            kind="trace_report",
        )
        report = build_carrion_objective_audit_report(
            search_reports=search_reports,
            trace_reports=trace_reports,
            missing_inputs=missing_search + missing_trace,
            selector_probe=args.selector_probe,
        )
        write_json_report(report, args.output)
    except (OSError, ValueError, CarrionObjectiveAuditError) as exc:
        raise SystemExit(f"failed to build carrion objective audit: {exc}") from exc

    diagnosis = report["diagnosis"]  # type: ignore[index]
    missing = report["missing_data"]  # type: ignore[index]
    print(f"carrion_objective_audit={args.output}")
    print(f"schema_version={MIND_V3_CARRION_OBJECTIVE_AUDIT_SCHEMA_VERSION}")
    print(
        "objective_pressure_assessment="
        f"{diagnosis['objective_pressure_assessment']}"  # type: ignore[index]
    )
    print(
        "selected_candidate_count="
        f"{diagnosis['selected_candidate_count']}"  # type: ignore[index]
    )
    print(f"missing_data_count={missing['total_count']}")  # type: ignore[index]
    if args.selector_probe:
        probe = report["selector_probe"]  # type: ignore[index]
        print(
            "selector_probe_candidate_count="
            f"{probe['candidate_count_inspected']}"  # type: ignore[index]
        )
        print(
            "selector_probe_selection_change_count="
            f"{probe['selection_change_count']}"  # type: ignore[index]
        )


def _load_labeled_reports(
    raw_values: list[str] | None,
    *,
    defaults: tuple[tuple[str, str], ...],
    kind: str,
) -> tuple[list[tuple[str, dict[str, object], str]], list[dict[str, object]]]:
    labeled_paths = (
        [parse_labeled_path(value) for value in raw_values]
        if raw_values
        else [(label, Path(path)) for label, path in defaults]
    )
    reports: list[tuple[str, dict[str, object], str]] = []
    missing: list[dict[str, object]] = []
    for label, path in labeled_paths:
        if not path.exists():
            missing.append(
                {
                    "label": label,
                    "source": kind,
                    "path": str(path),
                    "field": kind,
                    "reason": "input_path_missing",
                }
            )
            continue
        reports.append((label, load_json_report(path), str(path)))
    return reports, missing


def _generate_foundation_heuristic_baseline(
    *,
    seeds: list[int],
    ticks: int,
    output_path: Path,
    trajectory_output_dir: Path,
    trace_output_path: Path,
) -> dict[str, object]:
    if not seeds:
        raise CarrionObjectiveAuditError("baseline seeds must not be empty")
    if ticks < 1:
        raise CarrionObjectiveAuditError("baseline ticks must be >= 1")
    trajectory_output_dir.mkdir(parents=True, exist_ok=True)
    runs = [
        _run_fixture_once(
            fixture_name="carrion_only",
            seed=seed,
            ticks=ticks,
            policy=None,
            trajectory_output_path=(
                trajectory_output_dir
                / f"foundation-carrion-only-heuristic-seed-{seed}-ticks-{ticks}.jsonl.gz"
            ),
            trajectory_split_id="foundation_default_carrion_only_heuristic",
        )
        for seed in seeds
    ]
    datasets = [
        load_carrion_autopsy_trajectory_jsonl(
            trajectory_output_dir
            / f"foundation-carrion-only-heuristic-seed-{seed}-ticks-{ticks}.jsonl.gz"
        )
        for seed in seeds
    ]
    trace_report = build_carrion_autopsy_report(datasets)
    write_carrion_autopsy_report(trace_report, trace_output_path)
    baseline = foundation_heuristic_baseline_report(
        runs=runs,
        aggregate=_aggregate_runs(runs),
        seeds=seeds,
        ticks=ticks,
        output_path=output_path,
        trajectory_output_dir=trajectory_output_dir,
        trace_output_path=trace_output_path,
    )
    write_json_report(baseline, output_path)
    return baseline


def _parse_seeds(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        values.append(int(text))
    if not values:
        raise CarrionObjectiveAuditError("seed list must not be empty")
    return list(dict.fromkeys(values))


if __name__ == "__main__":
    main()
