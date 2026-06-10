from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.branch_intervention_residual import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_REPORT_PATH,
    DEFAULT_V143_DATASET_PATH,
    DEFAULT_V143_REPORT_PATH,
    STRICT_BROAD_SEEDS,
    STRICT_CARRION_FIXTURE_SEEDS,
    STRICT_TICKS,
    BranchInterventionResidualError,
    build_branch_intervention_residual_artifact,
    build_branch_intervention_residual_live_ab_report,
    load_json_report,
    load_v143_branch_intervention_dataset,
    write_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train the v144 opt-in support-gated residual artifact from the "
            "v143 branch-intervention dataset and run the strict shadow/live A/B."
        )
    )
    parser.add_argument("--v143-report", type=Path, default=DEFAULT_V143_REPORT_PATH)
    parser.add_argument(
        "--v143-dataset",
        type=Path,
        default=DEFAULT_V143_DATASET_PATH,
    )
    parser.add_argument(
        "--artifact-output",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--broad-seeds",
        default=",".join(str(seed) for seed in STRICT_BROAD_SEEDS),
    )
    parser.add_argument("--ticks", type=int, default=STRICT_TICKS)
    parser.add_argument(
        "--fixture-seeds",
        default=",".join(str(seed) for seed in STRICT_CARRION_FIXTURE_SEEDS),
    )
    parser.add_argument("--fixture-ticks", type=int, default=STRICT_TICKS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        v143_report = load_json_report(args.v143_report)
        dataset_rows = load_v143_branch_intervention_dataset(args.v143_dataset)
        artifact, training_report = build_branch_intervention_residual_artifact(
            v143_report=v143_report,
            dataset_rows=dataset_rows,
            v143_report_path=args.v143_report,
            v143_dataset_path=args.v143_dataset,
        )
        write_json(args.artifact_output, artifact)
        report = build_branch_intervention_residual_live_ab_report(
            artifact=artifact,
            training_report=training_report,
            broad_seeds=_parse_seeds(args.broad_seeds),
            ticks=int(args.ticks),
            fixture_seeds=_parse_seeds(args.fixture_seeds),
            fixture_ticks=int(args.fixture_ticks),
            artifact_output=args.artifact_output,
        )
        write_json(args.output, report)
    except (OSError, ValueError, BranchInterventionResidualError) as exc:
        raise SystemExit(
            f"failed to run v144 branch-intervention residual live A/B: {exc}"
        ) from exc
    _print_summary(report, args.artifact_output, args.output)


def _parse_seeds(raw: str) -> list[int]:
    seeds = []
    for item in raw.split(","):
        text = item.strip()
        if text:
            seeds.append(int(text))
    if not seeds:
        raise SystemExit("seed list must not be empty")
    return seeds


def _print_summary(
    report: dict[str, object],
    artifact_output: Path,
    report_output: Path,
) -> None:
    acceptance = report.get("acceptance")
    acceptance_payload = acceptance if isinstance(acceptance, dict) else {}
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    print(f"v144_branch_intervention_residual_artifact={artifact_output}")
    print(f"v144_branch_intervention_residual_live_ab={report_output}")
    print(f"classification={classification_payload.get('primary')}")
    print(f"acceptance_passed={acceptance_payload.get('passed')}")
    print(f"first_failed_floor={acceptance_payload.get('first_failed_floor')}")
    print(f"first_failing_seed={acceptance_payload.get('first_failing_seed')}")
    print(f"first_failing_fixture={acceptance_payload.get('first_failing_fixture')}")
    print(f"applied_override_count={acceptance_payload.get('applied_override_count')}")


if __name__ == "__main__":
    main()
