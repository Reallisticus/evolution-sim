from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

from evolution_sim.mind.carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response import (
    DEFAULT_OUTPUT_PATH,
    DEFAULT_BACKUP_DOC_PATHS,
    DEFAULT_V186_REPORT_PATH,
    DEFAULT_V195_ARTIFACT_PATH,
    DEFAULT_V195_REPORT_PATH,
    EXPECTED_V194_COMPACT_DATASET_DIGEST,
    EXPECTED_V194_REPORT_EXACT_DIGEST,
    EXPECTED_V195_ARTIFACT_DIGEST,
    EXPECTED_V195_REPORT_EXACT_DIGEST,
    EXPECTED_V195_ROUTE,
    run_carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response,
)
from evolution_sim.mind.carrion_survivor_continuation_v180_transition_row_policy_training import (
    DEFAULT_BROAD_SEEDS,
    DEFAULT_CARRION_FIXTURE_SEEDS,
    DEFAULT_TICKS,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v196 diagnostics-only failure response for the v195 "
            "repaired-contract slice-3 training failure. The command validates "
            "pinned v195/v194 source digests, inspects the frozen v195 artifact, "
            "runs lookup/coverage diagnostics unless explicitly skipped, and "
            "recommends one no-training route. It does not train, consume "
            "slice 4, integrate runtime behavior, relax gates, or authorize "
            "promotion."
        )
    )
    parser.add_argument("--v195-report", type=Path, default=DEFAULT_V195_REPORT_PATH)
    parser.add_argument("--v195-artifact", type=Path, default=DEFAULT_V195_ARTIFACT_PATH)
    parser.add_argument("--v186-report", type=Path, default=DEFAULT_V186_REPORT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--expected-v195-report-exact-digest",
        default=EXPECTED_V195_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v195-artifact-digest",
        default=EXPECTED_V195_ARTIFACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v194-report-exact-digest",
        default=EXPECTED_V194_REPORT_EXACT_DIGEST,
    )
    parser.add_argument(
        "--expected-v194-dataset-digest",
        default=EXPECTED_V194_COMPACT_DATASET_DIGEST,
    )
    parser.add_argument("--expected-v195-route", default=EXPECTED_V195_ROUTE)
    parser.add_argument("--broad-seeds", default=_csv(DEFAULT_BROAD_SEEDS))
    parser.add_argument(
        "--carrion-fixture-seeds",
        default=_csv(DEFAULT_CARRION_FIXTURE_SEEDS),
    )
    parser.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    parser.add_argument(
        "--backup-doc",
        action="append",
        type=Path,
        dest="backup_docs",
        help=(
            "Doc path that records v195 backup metadata. May be repeated; "
            "defaults to durable repo docs."
        ),
    )
    parser.add_argument(
        "--skip-diagnostic-replay",
        action="store_true",
        help=(
            "Skip the per-decision lookup/coverage replay. This is for narrow "
            "source-pin tests only; the default v196 run should not use it."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_carrion_survivor_continuation_v196_repaired_contract_slice_3_failure_response(
            v195_report_path=args.v195_report,
            v195_artifact_path=args.v195_artifact,
            v186_report_path=args.v186_report,
            output_path=args.output,
            expected_v195_report_exact_digest=args.expected_v195_report_exact_digest,
            expected_v195_artifact_digest=args.expected_v195_artifact_digest,
            expected_v194_report_exact_digest=args.expected_v194_report_exact_digest,
            expected_v194_dataset_digest=args.expected_v194_dataset_digest,
            expected_v195_route=args.expected_v195_route,
            backup_doc_paths=tuple(args.backup_docs or DEFAULT_BACKUP_DOC_PATHS),
            run_diagnostic_replay=not args.skip_diagnostic_replay,
            broad_seeds=_parse_seed_csv(args.broad_seeds),
            carrion_fixture_seeds=_parse_seed_csv(args.carrion_fixture_seeds),
            ticks=args.ticks,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(
            "failed to run v196 repaired-contract slice-3 failure response: "
            f"{exc}"
        ) from exc
    _print_summary(report, args.output)


def _print_summary(report: Mapping[str, object], output: Path) -> None:
    classification = _payload(report.get("classification"))
    source = _payload(report.get("source_pin_validation"))
    artifact = _payload(report.get("artifact_digest_validation"))
    facts = _payload(report.get("v195_failure_facts"))
    lookup = _payload(report.get("lookup_coverage_diagnostics"))
    combined = _payload(lookup.get("combined"))
    mechanism = _payload(report.get("mechanism_analysis"))
    route = _payload(report.get("route_decision"))
    print(f"carrion_survivor_continuation_v196_failure_response={output}")
    print(f"classification={classification.get('primary')}")
    print(f"source_pin_validation_passed={source.get('passed')}")
    print(f"v195_report_exact_digest={source.get('observed_v195_report_exact_digest')}")
    print(f"v195_artifact_digest={artifact.get('observed_artifact_digest')}")
    print(f"v194_report_exact_digest={source.get('observed_v194_report_exact_digest')}")
    print(f"v194_dataset_digest={source.get('observed_v194_dataset_digest')}")
    print(f"slice_3_consumed_by_v195={facts.get('v195_training_slice_consumed')}")
    print(f"carrion_terminal_survivors={facts.get('carrion_terminal_survivors')}")
    print(
        "dominant_requested_action_share="
        f"{facts.get('dominant_requested_action_share')}"
    )
    print(
        "training_dominant_action_share="
        f"{facts.get('training_dominant_action_share')}"
    )
    print(f"lookup_diagnostic_replay_ran={lookup.get('ran')}")
    print(f"combined_override_applied_share={combined.get('override_applied_share')}")
    print(
        "combined_dominant_applied_override_action="
        f"{combined.get('dominant_applied_override_action')}"
    )
    print(
        "combined_dominant_applied_override_action_share="
        f"{combined.get('dominant_applied_override_action_share')}"
    )
    print(
        "missing_states_defaulted_to_stay="
        f"{combined.get('missing_states_defaulted_to_stay')}"
    )
    print(f"primary_mechanism={mechanism.get('primary_mechanism')}")
    print(f"recommended_next_route={route.get('recommended_next_route')}")
    print(f"training_ran={report.get('training_ran')}")
    print(f"slice_4_training_consumed={report.get('slice_4_training_consumed')}")
    print(f"runtime_artifact_created={report.get('runtime_artifact_created')}")
    print(
        "runtime_action_selection_changed="
        f"{report.get('runtime_action_selection_changed')}"
    )
    print(f"promotion_authorized={report.get('promotion_authorized')}")
    print(f"gate_relaxation_allowed={report.get('gate_relaxation_allowed')}")
    print(f"exact_digest={report.get('exact_digest')}")


def _payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _csv(values: object) -> str:
    return ",".join(str(int(value)) for value in values)


def _parse_seed_csv(raw: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not seeds:
        raise SystemExit("seed list must not be empty")
    return seeds


if __name__ == "__main__":
    main()
