from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_archive_override_autopsy import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TICKS,
    DEFAULT_V148_CARRION_ARCHIVE_DATASET_PATH,
    DEFAULT_V148_CARRION_ARCHIVE_REPORT_PATH,
    DEFAULT_V149_CARRION_ARTIFACT_PATH,
    DEFAULT_V149_CARRION_TRAIN_EVAL_REPORT_PATH,
    CarrionArchiveOverrideAutopsyError,
    build_carrion_archive_override_autopsy_report,
    load_json_report,
    load_jsonl_rows,
    write_carrion_archive_override_autopsy_report,
)
from evolution_sim.mind.support_gated_residual import (
    load_support_gated_residual_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 v150 carrion archive override "
            "autopsy. This reruns only the opt-in v149 carrion artifact to "
            "recover override traces; it never trains, promotes, or changes "
            "runtime defaults."
        )
    )
    parser.add_argument(
        "--archive-report",
        type=Path,
        default=DEFAULT_V148_CARRION_ARCHIVE_REPORT_PATH,
    )
    parser.add_argument(
        "--archive-dataset",
        type=Path,
        default=DEFAULT_V148_CARRION_ARCHIVE_DATASET_PATH,
    )
    parser.add_argument(
        "--train-eval-report",
        type=Path,
        default=DEFAULT_V149_CARRION_TRAIN_EVAL_REPORT_PATH,
    )
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_V149_CARRION_ARTIFACT_PATH,
    )
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        archive_report = load_json_report(args.archive_report)
        dataset_rows = load_jsonl_rows(args.archive_dataset)
        train_eval_report = load_json_report(args.train_eval_report)
        artifact = load_support_gated_residual_artifact(args.artifact)
        report = build_carrion_archive_override_autopsy_report(
            archive_report=archive_report,
            dataset_rows=dataset_rows,
            train_eval_report=train_eval_report,
            artifact=artifact,
            ticks=int(args.ticks),
            input_paths={
                "archive_report": args.archive_report,
                "archive_dataset": args.archive_dataset,
                "train_eval_report": args.train_eval_report,
                "artifact": args.artifact,
            },
        )
        write_carrion_archive_override_autopsy_report(report, args.output)
    except (OSError, ValueError, CarrionArchiveOverrideAutopsyError) as exc:
        raise SystemExit(
            f"failed to build carrion archive override autopsy: {exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: dict[str, object], output: Path) -> None:
    classification = report.get("classification")
    classification_payload = classification if isinstance(classification, dict) else {}
    aggregates = report.get("aggregates")
    aggregate_payload = aggregates if isinstance(aggregates, dict) else {}
    failure_modes = report.get("failure_mode_classification")
    failure_payload = failure_modes if isinstance(failure_modes, dict) else {}
    print(f"carrion_archive_override_autopsy_report={output}")
    print(f"classification={classification_payload.get('primary')}")
    print(
        "observed_live_carrion_override_count="
        f"{aggregate_payload.get('observed_live_carrion_override_count')}"
    )
    print(
        "expected_live_carrion_override_count="
        f"{aggregate_payload.get('expected_live_carrion_override_count')}"
    )
    print(f"failure_mode_primary={failure_payload.get('primary')}")
    print(f"recommended_next_route={report.get('recommended_next_route')}")
    print(f"exact_digest={report.get('exact_digest')}")
    print(f"training_authorized={report.get('training_authorized')}")
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"runtime_promotion_allowed={report.get('runtime_promotion_allowed')}")


if __name__ == "__main__":
    main()
