from __future__ import annotations

import argparse
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v194_repaired_contract_terminal_survival_support_dataset_audit import (
    DEFAULT_BACKUP_DOC_PATHS,
    DEFAULT_COMPACT_SUPPORT_DATASET_OUTPUT_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_V193_REPORT_PATH,
    EXPECTED_V190_REPORT_EXACT_DIGEST,
    EXPECTED_V191_REPORT_EXACT_DIGEST,
    EXPECTED_V192_REPORT_EXACT_DIGEST,
    EXPECTED_V193_REPORT_EXACT_DIGEST,
    REQUIRED_V193_ROUTE,
    run_carrion_survivor_continuation_v194_repaired_contract_terminal_survival_support_dataset_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run diagnostics-only v194 repaired-contract terminal-survival "
            "support dataset audit before any slice-3 training."
        )
    )
    parser.add_argument("--v193-report", type=Path, default=DEFAULT_V193_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--compact-support-dataset-output",
        type=Path,
        default=DEFAULT_COMPACT_SUPPORT_DATASET_OUTPUT_PATH,
    )
    parser.add_argument(
        "--expected-v193-report-exact-digest",
        default=EXPECTED_V193_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v192-report-exact-digest",
        default=EXPECTED_V192_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v191-report-exact-digest",
        default=EXPECTED_V191_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v190-report-exact-digest",
        default=EXPECTED_V190_REPORT_EXACT_DIGEST,
    )
    parser.add_argument("--required-v193-route", default=REQUIRED_V193_ROUTE)
    parser.add_argument(
        "--backup-doc",
        action="append",
        type=Path,
        dest="backup_docs",
        help=(
            "Doc path that records v193 backup metadata. May be repeated; "
            "defaults to durable repo docs."
        ),
    )
    parser.add_argument(
        "--skip-compact-support-dataset",
        action="store_true",
        help=(
            "Do not write the compact support dataset. The audit will fail "
            "closed because v194 normally pins the dataset digest."
        ),
    )
    args = parser.parse_args()

    report = run_carrion_survivor_continuation_v194_repaired_contract_terminal_survival_support_dataset_audit(
        v193_report_path=args.v193_report,
        output_path=args.output,
        compact_support_dataset_output_path=args.compact_support_dataset_output,
        expected_v193_report_exact_digest=args.expected_v193_report_exact_digest,
        expected_v192_report_exact_digest=args.expected_v192_report_exact_digest,
        expected_v191_report_exact_digest=args.expected_v191_report_exact_digest,
        expected_v190_report_exact_digest=args.expected_v190_report_exact_digest,
        required_v193_route=args.required_v193_route,
        backup_doc_paths=tuple(args.backup_docs or DEFAULT_BACKUP_DOC_PATHS),
        create_compact_support_dataset=not args.skip_compact_support_dataset,
    )

    source = report["source_validation"]
    selected = report["selected_support_facts_audit"]
    dataset = report["dataset_audit"]
    route = report["route_decision"]
    print(f"carrion_survivor_continuation_v194={args.output}")
    print(f"exact_digest={report['exact_digest']}")
    print(f"source_validation_passed={source.get('passed')}")
    print(
        "selected_support_seed_count="
        f"{selected.get('selected_seed_count')}/"
        f"{selected.get('expected_selected_seed_count')}"
    )
    print(f"dataset_created={report['compact_support_dataset_created']}")
    print(f"dataset_path={dataset.get('dataset_path')}")
    print(f"dataset_digest={dataset.get('dataset_digest')}")
    print(f"dataset_ready={dataset.get('dataset_ready_for_future_slice_3_training')}")
    print(
        "unsupported_requested_action_count="
        f"{selected.get('unsupported_requested_action_count')}"
    )
    print(
        "expected_same_tick_occupancy_drift_count="
        f"{selected.get('expected_same_tick_occupancy_drift_count')}"
    )
    print(
        "selected_expected_same_tick_occupancy_drift_count="
        f"{selected.get('selected_expected_same_tick_occupancy_drift_count')}"
    )
    print(
        "unexpected_resolution_invalid_count="
        f"{selected.get('unexpected_resolution_invalid_count')}"
    )
    print(
        "dominant_requested_action_share="
        f"{selected.get('dominant_requested_action_share')}"
    )
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(
        "future_explicit_slice_3_training_route_authorized="
        f"{route.get('future_explicit_slice_3_training_route_authorized')}"
    )
    print(f"training_ran={report['training_ran']}")
    print(f"slice_3_training_consumed={report['slice_3_training_consumed']}")
    print(f"runtime_artifact_created={report['runtime_artifact_created']}")
    print(f"runtime_action_selection_changed={report['runtime_action_selection_changed']}")
    print(f"promotion_authorized={report['promotion_authorized']}")
    print(f"gate_relaxation_allowed={report['gate_relaxation_allowed']}")


if __name__ == "__main__":
    main()
