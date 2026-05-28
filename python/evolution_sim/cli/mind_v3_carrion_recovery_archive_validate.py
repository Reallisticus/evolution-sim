from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

from evolution_sim.mind.carrion_recovery_archive import (
    MIND_V3_CARRION_RECOVERY_ARCHIVE_VALIDATION_SCHEMA_VERSION,
    MIND_V3_CARRION_RECOVERY_SPLIT_POLICY_BRANCH_DIGEST_SEED_STRATIFIED,
    CarrionRecoveryArchiveError,
    build_carrion_recovery_archive_validation_report,
    write_carrion_recovery_archive_report,
    write_carrion_recovery_split_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a Mind v3 carrion recovery archive dataset and build a "
            "leakage-safe branch-state train/held-out split manifest."
        )
    )
    parser.add_argument(
        "--archive-report",
        type=Path,
        required=True,
        help="Input mind_v3_carrion_recovery_archive_v1 report.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Input archive dataset JSONL path.",
    )
    parser.add_argument(
        "--branch-report",
        type=Path,
        default=None,
        help=(
            "Optional source branch-explore report. Defaults to the path "
            "recorded in the archive report."
        ),
    )
    parser.add_argument(
        "--split-policy",
        choices=[MIND_V3_CARRION_RECOVERY_SPLIT_POLICY_BRANCH_DIGEST_SEED_STRATIFIED],
        default=MIND_V3_CARRION_RECOVERY_SPLIT_POLICY_BRANCH_DIGEST_SEED_STRATIFIED,
        help="Leakage-safe split policy.",
    )
    parser.add_argument(
        "--split-output",
        type=Path,
        required=True,
        help="Output split manifest JSON path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output validation report JSON path.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = build_carrion_recovery_archive_validation_report(
            archive_report_path=args.archive_report,
            dataset_path=args.dataset,
            branch_report_path=args.branch_report,
            split_policy=str(args.split_policy),
        )
        split = report["split"]  # type: ignore[index]
        assert isinstance(split, Mapping)
        write_carrion_recovery_split_report(split, args.split_output)
        write_carrion_recovery_archive_report(report, args.output)
    except (OSError, ValueError, CarrionRecoveryArchiveError) as exc:
        raise SystemExit(
            f"failed to validate carrion recovery archive: {exc}"
        ) from exc

    acceptance = _mapping(report.get("acceptance"))
    dataset_validation = _mapping(report.get("dataset_validation"))
    split_report = _mapping(report.get("split"))
    split_aggregate = _mapping(split_report.get("aggregate"))
    split_acceptance = _mapping(split_report.get("acceptance"))
    print(f"carrion_recovery_archive_validation={args.output}")
    print(f"carrion_recovery_split={args.split_output}")
    print(f"schema_version={MIND_V3_CARRION_RECOVERY_ARCHIVE_VALIDATION_SCHEMA_VERSION}")
    print(f"validation_passed={acceptance.get('validation_passed')}")
    print(f"training_blocked={acceptance.get('training_blocked')}")
    print(f"dataset_record_count={dataset_validation.get('record_count', 0)}")
    print(f"dataset_survivor_count={dataset_validation.get('survivor_count', 0)}")
    print(f"dataset_failure_count={dataset_validation.get('failure_count', 0)}")
    print(f"train_record_count={split_aggregate.get('train_record_count', 0)}")
    print(f"heldout_record_count={split_aggregate.get('heldout_record_count', 0)}")
    print(f"leakage_check_passed={_mapping(split_report.get('leakage_check')).get('passed')}")
    print(f"split_training_blocked={split_acceptance.get('training_blocked')}")


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
