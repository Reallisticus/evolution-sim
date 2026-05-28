from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.first_recovery_accepted_rare_attack_contract import (
    DEFAULT_MANIFEST_OUTPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    build_first_recovery_accepted_rare_attack_contract,
    write_first_recovery_accepted_rare_attack_contract_outputs,
)
from evolution_sim.mind.first_recovery_active_coverage_archive import (
    DEFAULT_ARCHIVE_ROWS_OUTPUT_PATH as DEFAULT_V123_ARCHIVE_ROWS_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V123_REPORT_PATH,
    DEFAULT_V122_RERUN_CANDIDATES_OUTPUT_PATH as DEFAULT_V122_OVER_V123_CANDIDATES_PATH,
    DEFAULT_V122_RERUN_OUTPUT_PATH as DEFAULT_V122_OVER_V123_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_rare_attack_coverage_collection import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V122_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_repaired_label_contract_audit import (
    DEFAULT_MANIFEST_OUTPUT_PATH as DEFAULT_V119_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH as DEFAULT_V119_REPORT_PATH,
)
from evolution_sim.mind.first_recovery_repaired_label_split_support_feasibility import (
    DEFAULT_OUTPUT_PATH as DEFAULT_V120_REPORT_PATH,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the diagnostics-only Mind v3 v124 accepted rare-attack "
            "manifest/split contract."
        )
    )
    parser.add_argument("--v119-report", type=Path, default=DEFAULT_V119_REPORT_PATH)
    parser.add_argument("--v119-manifest", type=Path, default=DEFAULT_V119_MANIFEST_PATH)
    parser.add_argument("--v120-report", type=Path, default=DEFAULT_V120_REPORT_PATH)
    parser.add_argument(
        "--v122-default-report",
        type=Path,
        default=DEFAULT_V122_REPORT_PATH,
    )
    parser.add_argument("--v123-report", type=Path, default=DEFAULT_V123_REPORT_PATH)
    parser.add_argument(
        "--v123-archive-rows",
        type=Path,
        default=DEFAULT_V123_ARCHIVE_ROWS_PATH,
    )
    parser.add_argument(
        "--v122-over-v123-report",
        type=Path,
        default=DEFAULT_V122_OVER_V123_REPORT_PATH,
    )
    parser.add_argument(
        "--v122-over-v123-candidates",
        type=Path,
        default=DEFAULT_V122_OVER_V123_CANDIDATES_PATH,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_MANIFEST_OUTPUT_PATH,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        build = build_first_recovery_accepted_rare_attack_contract(
            v119_report_path=args.v119_report,
            v119_manifest_path=args.v119_manifest,
            v120_report_path=args.v120_report,
            v122_default_report_path=args.v122_default_report,
            v123_report_path=args.v123_report,
            v123_archive_rows_path=args.v123_archive_rows,
            v122_over_v123_report_path=args.v122_over_v123_report,
            v122_over_v123_candidates_path=args.v122_over_v123_candidates,
        )
        write_first_recovery_accepted_rare_attack_contract_outputs(
            build,
            output_path=args.output,
            manifest_output_path=args.manifest_output,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to build first-recovery accepted rare-attack contract: "
            f"{exc}"
        ) from exc
    _print_summary(build.report, args.output, args.manifest_output)


def _print_summary(
    report: Mapping[str, object],
    output_path: Path,
    manifest_output_path: Path,
) -> None:
    classification = _mapping(report.get("classification"))
    source = _mapping(report.get("source_integrity"))
    checks = _mapping(report.get("contract_checks"))
    split = _mapping(report.get("split_support"))
    recommendation = _mapping(report.get("recommendation"))
    print(f"first_recovery_accepted_rare_attack_contract={output_path}")
    print(f"accepted_rare_attack_manifest={manifest_output_path}")
    print(f"classification={classification.get('primary')}")
    print(f"source_integrity_passed={source.get('passed')}")
    print(f"manifest_row_count={checks.get('manifest_row_count')}")
    print(f"repaired_action_counts={checks.get('repaired_action_counts')}")
    print(
        "strict_train_validation_test_support_met="
        f"{split.get('strict_train_validation_test_support_met')}"
    )
    print(
        "v113_readiness_rerun_allowed="
        f"{recommendation.get('v113_readiness_rerun_allowed')}"
    )
    print(
        "downstream_shadow_scorer_allowed="
        f"{recommendation.get('downstream_shadow_scorer_allowed')}"
    )


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


if __name__ == "__main__":
    main()
